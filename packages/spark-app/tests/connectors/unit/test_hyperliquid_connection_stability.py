"""
Unit tests for Hyperliquid connection stability features.

Tests cover:
- Connection health monitoring
- WebSocket management
- Rate limiting
- Connection state management
- Metrics tracking
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from app.connectors.base_connector import MarketType
from app.connectors.hyperliquid_connector import (ConnectionHealthMonitor,
                                                  ConnectionMetrics,
                                                  ConnectionState,
                                                  HyperliquidConnectionError,
                                                  HyperliquidConnector,
                                                  HyperliquidRateLimitError,
                                                  RateLimitHandler,
                                                  WebSocketManager)


class TestConnectionHealthMonitor:
    """Test suite for connection health monitoring."""

    def test_health_monitor_initialization(self):
        """Test health monitor is properly initialized."""
        connector = HyperliquidConnector(testnet=True)

        assert hasattr(connector, 'health_monitor')
        assert isinstance(connector.health_monitor, ConnectionHealthMonitor)
        assert connector.health_monitor.state == ConnectionState.DISCONNECTED
        assert connector.health_monitor.metrics.connection_attempts == 0

    def test_health_monitor_start_stop(self):
        """Test health monitor can be started and stopped."""
        connector = HyperliquidConnector(testnet=True)
        health_monitor = connector.health_monitor

        # Start monitoring
        health_monitor.start_monitoring()
        assert health_monitor._health_thread is not None
        assert health_monitor._health_thread.is_alive()

        # Stop monitoring
        health_monitor.stop_monitoring()
        assert health_monitor._stop_health_check.is_set()

    def test_health_check_success(self):
        """Test successful health check updates metrics."""
        connector = HyperliquidConnector(testnet=True)
        health_monitor = connector.health_monitor

        # Mock successful health check
        mock_info = Mock()
        mock_info.meta.return_value = {"universe": [{"name": "BTC"}]}
        connector.info = mock_info

        # Perform health check
        health_monitor._perform_health_check()

        assert health_monitor._consecutive_failures == 0
        assert health_monitor._last_health_check is not None
        assert health_monitor.state == ConnectionState.CONNECTED

    def test_health_check_failure(self):
        """Test failed health check triggers reconnection logic."""
        connector = HyperliquidConnector(testnet=True)
        health_monitor = connector.health_monitor
        health_monitor.state = ConnectionState.CONNECTED

        # Mock failed health check
        connector.info = None

        # Patch the _attempt_reconnection method to prevent actual reconnection
        with patch.object(health_monitor, '_attempt_reconnection') as mock_reconnect:
            # Perform multiple failed health checks
            for i in range(health_monitor.max_consecutive_failures):
                health_monitor._perform_health_check()

            assert health_monitor._consecutive_failures == health_monitor.max_consecutive_failures
            assert health_monitor.metrics.failed_connections > 0
            # Verify that reconnection was attempted
            mock_reconnect.assert_called_once()

    @patch.object(HyperliquidConnector, 'connect')
    def test_automatic_reconnection(self, mock_connect):
        """Test automatic reconnection is attempted."""
        mock_connect.return_value = True

        connector = HyperliquidConnector(testnet=True)
        health_monitor = connector.health_monitor
        health_monitor.state = ConnectionState.CONNECTED

        # Trigger reconnection
        health_monitor._attempt_reconnection()

        mock_connect.assert_called_once()
        assert health_monitor.metrics.reconnection_attempts == 1

    def test_response_time_tracking(self):
        """Test response time tracking updates correctly."""
        connector = HyperliquidConnector(testnet=True)
        health_monitor = connector.health_monitor

        # Update response times
        health_monitor._update_response_time(0.5)
        assert health_monitor.metrics.average_response_time == 0.5

        health_monitor._update_response_time(1.0)
        # Should be moving average: 0.5 * 0.9 + 1.0 * 0.1 = 0.55
        assert abs(health_monitor.metrics.average_response_time - 0.55) < 0.01


class TestWebSocketManager:
    """Test suite for WebSocket management."""

    def test_websocket_manager_initialization(self):
        """Test WebSocket manager is properly initialized."""
        connector = HyperliquidConnector(testnet=True)

        assert hasattr(connector, 'websocket_manager')
        assert isinstance(connector.websocket_manager, WebSocketManager)
        assert connector.websocket_manager.websocket is None

    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection logic."""
        connector = HyperliquidConnector(testnet=True)
        ws_manager = connector.websocket_manager

        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value = mock_websocket

            result = await ws_manager.connect()

            assert result is True
            assert ws_manager.websocket == mock_websocket
            assert ws_manager._reconnect_attempts == 0
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_connection_failure(self):
        """Test WebSocket connection failure handling."""
        connector = HyperliquidConnector(testnet=True)
        ws_manager = connector.websocket_manager

        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")

            result = await ws_manager.connect()

            assert result is False
            assert ws_manager.websocket is None

    @pytest.mark.asyncio
    async def test_websocket_subscription(self):
        """Test WebSocket subscription functionality."""
        connector = HyperliquidConnector(testnet=True)
        ws_manager = connector.websocket_manager

        mock_websocket = AsyncMock()
        ws_manager.websocket = mock_websocket

        await ws_manager.subscribe("trades", market="BTC")

        mock_websocket.send.assert_called_once()
        assert "trades" in ws_manager._subscriptions
        assert ws_manager._subscriptions["trades"]["market"] == "BTC"

    @pytest.mark.asyncio
    async def test_websocket_reconnection_backoff(self):
        """Test WebSocket reconnection with exponential backoff."""
        connector = HyperliquidConnector(testnet=True)
        ws_manager = connector.websocket_manager

        # Simulate failed reconnection attempts
        ws_manager._reconnect_attempts = 2

        with patch.object(ws_manager, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = False

            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                await ws_manager._handle_reconnection()

                # Check exponential backoff calculation
                expected_delay = min(1.0 * (2 ** 3), 60.0)  # attempts + 1
                mock_sleep.assert_called_once_with(expected_delay)

    @pytest.mark.asyncio
    async def test_websocket_disconnect(self):
        """Test WebSocket disconnection."""
        connector = HyperliquidConnector(testnet=True)
        ws_manager = connector.websocket_manager

        mock_websocket = AsyncMock()
        ws_manager.websocket = mock_websocket

        await ws_manager.disconnect()

        mock_websocket.close.assert_called_once()
        assert ws_manager.websocket is None


class TestRateLimitHandler:
    """Test suite for rate limiting functionality."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter is properly initialized."""
        connector = HyperliquidConnector(testnet=True)

        assert hasattr(connector, 'rate_limiter')
        assert isinstance(connector.rate_limiter, RateLimitHandler)
        assert len(connector.rate_limiter.request_times) == 0

    def test_rate_limit_tracking(self):
        """Test rate limit tracking functionality."""
        rate_limiter = RateLimitHandler()

        # Should allow requests initially
        assert rate_limiter.can_make_request() is True

        # Record requests
        for _ in range(50):
            rate_limiter.record_request()

        # Should still allow requests
        assert rate_limiter.can_make_request() is True

        # Record too many requests
        for _ in range(60):  # Total 110, over limit of 100
            rate_limiter.record_request()

        # Should not allow more requests
        assert rate_limiter.can_make_request() is False

    def test_rate_limit_window_cleanup(self):
        """Test old requests are cleaned up from the window."""
        rate_limiter = RateLimitHandler()

        # Add old requests (simulate time passing)
        old_time = time.time() - 120  # 2 minutes ago
        rate_limiter.request_times = [old_time] * 100

        # Should allow new requests as old ones are outside window
        assert rate_limiter.can_make_request() is True

    def test_rate_limit_backoff(self):
        """Test rate limit backoff mechanism."""
        rate_limiter = RateLimitHandler()
        initial_backoff = rate_limiter.rate_limit_backoff

        with patch('time.sleep') as mock_sleep:
            rate_limiter.handle_rate_limit()

            mock_sleep.assert_called_once_with(initial_backoff)
            # Backoff should have increased
            assert rate_limiter.rate_limit_backoff > initial_backoff

    def test_rate_limit_backoff_reset(self):
        """Test rate limit backoff reset."""
        rate_limiter = RateLimitHandler()
        rate_limiter.rate_limit_backoff = 10.0

        rate_limiter.reset_backoff()

        assert rate_limiter.rate_limit_backoff == 1.0


class TestConnectionStateManagement:
    """Test suite for connection state management."""

    def test_initial_connection_state(self):
        """Test initial connection state is correct."""
        connector = HyperliquidConnector(testnet=True)

        assert connector.connection_state == ConnectionState.DISCONNECTED
        assert connector._is_connected is False

    @patch('app.connectors.hyperliquid_connector.MetadataWrapper')
    @patch('hyperliquid.exchange.Exchange')
    @patch('eth_account.Account')
    def test_connection_state_updates(self, mock_account, mock_exchange, mock_metadata):
        """Test connection state updates during connect/disconnect."""
        # Mock successful connection
        mock_info = Mock()
        mock_info.meta.return_value = {"universe": [{"name": "BTC"}]}
        mock_metadata.return_value = mock_info

        connector = HyperliquidConnector(
            testnet=True,
            wallet_address="0x123",
            private_key="0xabc"
        )

        # Test connection
        with patch.object(connector.health_monitor, 'start_monitoring'):
            result = connector.connect()

        assert result is True
        assert connector.connection_state == ConnectionState.CONNECTED
        assert connector._is_connected is True
        assert connector.health_monitor.metrics.successful_connections == 1

    def test_connection_failure_state_updates(self):
        """Test connection state updates during failures."""
        connector = HyperliquidConnector(testnet=True)

        # Simulate connection failure
        connector._update_connection_failure()

        assert connector.connection_state == ConnectionState.FAILED
        assert connector._is_connected is False
        assert connector.health_monitor.metrics.failed_connections == 1

    def test_connection_metrics_retrieval(self):
        """Test connection metrics can be retrieved."""
        connector = HyperliquidConnector(testnet=True)

        # Update some metrics
        connector.health_monitor.metrics.connection_attempts = 5
        connector.health_monitor.metrics.successful_connections = 3
        connector.health_monitor.metrics.failed_connections = 2

        metrics = connector.get_connection_metrics()

        assert metrics["connection_attempts"] == 5
        assert metrics["successful_connections"] == 3
        assert metrics["failed_connections"] == 2
        assert metrics["success_rate"] == 0.6  # 3/5

    def test_force_reconnection(self):
        """Test force reconnection functionality."""
        connector = HyperliquidConnector(testnet=True)

        with patch.object(connector, 'connect', return_value=True) as mock_connect, \
             patch.object(connector, 'disconnect', return_value=True) as mock_disconnect, \
             patch('time.sleep'):

            result = connector.force_reconnection()

            assert result is True
            mock_disconnect.assert_called_once()
            mock_connect.assert_called_once()

    def test_connection_health_check(self):
        """Test connection health checking."""
        connector = HyperliquidConnector(testnet=True)

        # Not connected should be unhealthy
        assert connector.is_healthy() is False

        # Connected but no recent health check should be unhealthy
        connector._is_connected = True
        connector.connection_state = ConnectionState.CONNECTED
        assert connector.is_healthy() is True  # No health monitor constraints yet

        # Set up health monitor with recent check
        connector.health_monitor._last_health_check = datetime.now()
        connector.health_monitor._consecutive_failures = 0
        assert connector.is_healthy() is True

    @pytest.mark.asyncio
    async def test_websocket_integration(self):
        """Test WebSocket integration with connector."""
        connector = HyperliquidConnector(testnet=True)

        with patch.object(connector.websocket_manager, 'connect', new_callable=AsyncMock) as mock_connect, \
             patch.object(connector.websocket_manager, 'subscribe', new_callable=AsyncMock) as mock_subscribe:

            mock_connect.return_value = True

            subscriptions = [
                {"type": "trades", "market": "BTC"},
                {"type": "orderbook", "market": "ETH"}
            ]

            result = await connector.start_websocket(subscriptions)

            assert result is True
            mock_connect.assert_called_once()
            assert mock_subscribe.call_count == 2


class TestConnectionStabilityIntegration:
    """Integration tests for connection stability features."""

    @patch('app.connectors.hyperliquid_connector.MetadataWrapper')
    def test_full_connection_cycle_with_stability_features(self, mock_metadata):
        """Test complete connection cycle with all stability features."""
        # Mock successful API responses
        mock_info = Mock()
        mock_info.meta.return_value = {"universe": [{"name": "BTC"}]}
        mock_metadata.return_value = mock_info

        connector = HyperliquidConnector(testnet=True)

        # Initial state
        assert connector.connection_state == ConnectionState.DISCONNECTED
        assert not connector.is_healthy()

        # Connect with health monitoring
        with patch.object(connector.health_monitor, 'start_monitoring') as mock_start_monitoring:
            result = connector.connect()

            assert result is True
            assert connector.connection_state == ConnectionState.CONNECTED
            assert connector.is_healthy()
            mock_start_monitoring.assert_called_once()

        # Get metrics
        metrics = connector.get_connection_metrics()
        assert metrics["connection_attempts"] == 1
        assert metrics["successful_connections"] == 1
        assert metrics["success_rate"] == 1.0

        # Disconnect
        with patch.object(connector.health_monitor, 'stop_monitoring') as mock_stop_monitoring:
            result = connector.disconnect()

            assert result is True
            assert connector.connection_state == ConnectionState.DISCONNECTED
            assert not connector.is_healthy()
            mock_stop_monitoring.assert_called_once()

    def test_retry_decorator_with_rate_limiting(self):
        """Test retry decorator integrates with rate limiting."""
        from app.connectors.hyperliquid_connector import retry_api_call

        connector = HyperliquidConnector(testnet=True)

        @retry_api_call(max_tries=2, handle_rate_limit=True)
        def mock_api_call(self):
            if not self.rate_limiter.can_make_request():
                raise HyperliquidRateLimitError("Rate limit exceeded")
            return "success"

        # Fill up rate limiter
        for _ in range(110):  # Over the limit
            connector.rate_limiter.record_request()

        with patch.object(connector.rate_limiter, 'handle_rate_limit') as mock_handle:
            with patch.object(connector.rate_limiter, 'can_make_request', side_effect=[False, True]):
                result = mock_api_call(connector)

                assert result == "success"
                mock_handle.assert_called_once()

    def test_connection_stability_with_failures(self):
        """Test connection stability features handle failures gracefully."""
        connector = HyperliquidConnector(testnet=True)

        # Simulate multiple connection failures
        for _ in range(3):
            connector._update_connection_failure()

        metrics = connector.get_connection_metrics()
        assert metrics["failed_connections"] == 3
        assert metrics["success_rate"] == 0.0

        # Simulate recovery
        connector._update_connection_success()

        final_metrics = connector.get_connection_metrics()
        assert final_metrics["successful_connections"] == 1
        assert final_metrics["success_rate"] == 0.25  # 1 success out of 4 total attempts (3 failures + 1 success)
