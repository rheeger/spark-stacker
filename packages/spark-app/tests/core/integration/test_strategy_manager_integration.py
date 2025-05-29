"""
Unit tests for StrategyManager integration with strategies and indicators.

This module tests the strategy-driven execution flow, signal generation with
strategy context, symbol conversion, and error handling scenarios.
"""

import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pandas as pd
import pytest
from app.core.strategy_manager import StrategyManager
from app.core.trading_engine import TradingEngine
from app.indicators.base_indicator import (BaseIndicator, Signal,
                                           SignalDirection)
from app.indicators.macd_indicator import MACDIndicator
from app.indicators.rsi_indicator import RSIIndicator


class MockIndicator(BaseIndicator):
    """Mock indicator for testing purposes."""

    def __init__(self, name: str = "test_indicator", timeframe: str = "1h", required_periods: int = 20):
        super().__init__(name, params={'timeframe': timeframe})
        self.required_periods = required_periods
        self.process_called = False
        self.process_data = None
        self.process_timeframe = None
        self.calculate_called = False
        self.generate_signal_called = False

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mock calculate method that adds test data."""
        self.calculate_called = True
        processed_data = data.copy() if not data.empty else pd.DataFrame()
        if not processed_data.empty:
            processed_data['test_value'] = [1.0] * len(processed_data)
            processed_data['test_signal_value'] = [0.8] * len(processed_data)
        return processed_data

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Mock generate_signal method that returns a test signal."""
        self.generate_signal_called = True

        if data.empty:
            return None

        return Signal(
            direction=SignalDirection.BUY,
            symbol="ETH-USD",  # Use full symbol for consistency
            indicator=self.name,
            confidence=0.8,
            params={'test': True}
        )

    def process(self, data: pd.DataFrame, strategy_timeframe: str = None) -> tuple[pd.DataFrame, Signal]:
        """Mock process method that records calls and returns test signal."""
        self.process_called = True
        self.process_data = data.copy() if not data.empty else pd.DataFrame()
        self.process_timeframe = strategy_timeframe

        # Call the parent implementation which calls calculate and generate_signal
        return super().process(data, strategy_timeframe)


class TestStrategyManagerIntegration:
    """Test cases for StrategyManager strategy integration."""

    @pytest.fixture
    def mock_trading_engine(self):
        """Create a mock trading engine."""
        engine = Mock(spec=TradingEngine)
        engine.process_signal = AsyncMock(return_value=True)

        # Mock main connector
        engine.main_connector = Mock()
        engine.main_connector.get_ticker.return_value = {"last_price": 2000.0}
        engine.main_connector.get_historical_candles.return_value = [
            {
                'timestamp': 1640995200000,  # 2022-01-01 00:00:00
                'open': 1950.0,
                'high': 2050.0,
                'low': 1900.0,
                'close': 2000.0,
                'volume': 1000.0
            },
            {
                'timestamp': 1640998800000,  # 2022-01-01 01:00:00
                'open': 2000.0,
                'high': 2100.0,
                'low': 1950.0,
                'close': 2050.0,
                'volume': 1200.0
            }
        ]

        return engine

    @pytest.fixture
    def test_indicators(self):
        """Create test indicators."""
        return {
            "rsi_4h": MockIndicator("rsi_4h", timeframe="4h", required_periods=14),
            "macd_1h": MockIndicator("macd_1h", timeframe="1h", required_periods=26),
            "sma_daily": MockIndicator("sma_daily", timeframe="1d", required_periods=20)
        }

    @pytest.fixture
    def test_strategies(self):
        """Create test strategy configurations."""
        return [
            {
                "name": "eth_multi_timeframe",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "timeframe": "4h",
                "indicators": ["rsi_4h", "macd_1h"],
                "enabled": True
            },
            {
                "name": "btc_daily_strategy",
                "market": "BTC-USD",
                "exchange": "coinbase",
                "timeframe": "1d",
                "indicators": ["sma_daily"],
                "enabled": True
            },
            {
                "name": "disabled_strategy",
                "market": "LTC-USD",
                "exchange": "hyperliquid",
                "timeframe": "1h",
                "indicators": ["rsi_4h"],
                "enabled": False
            }
        ]

    @pytest.fixture
    def strategy_manager(self, mock_trading_engine, test_indicators, test_strategies):
        """Create a StrategyManager instance with test data."""
        with patch('app.core.strategy_manager.update_candle_data'), \
             patch('app.core.strategy_manager.update_macd_indicator'), \
             patch('app.core.strategy_manager.update_mvp_signal_state'), \
             patch('app.core.strategy_manager.publish_historical_time_series'), \
             patch('app.core.strategy_manager.verify_historical_data'), \
             patch('app.core.strategy_manager.record_mvp_signal_latency'):

            return StrategyManager(
                trading_engine=mock_trading_engine,
                indicators=test_indicators,
                strategies=test_strategies,
                data_window_size=100,
                config={"metrics_publish_historical": False}
            )

    def test_strategy_manager_initialization_with_strategies(self, strategy_manager, test_strategies):
        """Test StrategyManager initialization with strategy configurations."""
        # Verify strategies are stored
        assert strategy_manager.strategies == test_strategies
        assert len(strategy_manager.strategies) == 3

        # Verify strategy mappings are built
        assert len(strategy_manager.strategy_indicators) == 3
        assert strategy_manager.strategy_indicators["rsi_4h"] == ["rsi_4h"]
        assert strategy_manager.strategy_indicators["macd_1h"] == ["macd_1h"]
        assert strategy_manager.strategy_indicators["sma_daily"] == ["sma_daily"]
        assert strategy_manager.strategy_indicators["disabled_strategy"] == ["rsi_4h"]

    def test_build_strategy_mappings(self, mock_trading_engine, test_indicators):
        """Test _build_strategy_mappings() method."""
        strategies = [
            {
                "name": "test_strategy_1",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "indicators": ["rsi_4h", "macd_1h"]
            },
            {
                "name": "test_strategy_2",
                "market": "BTC-USD",
                "exchange": "coinbase",
                "indicators": ["sma_daily"]
            },
            {
                # Strategy missing name (should be skipped)
                "market": "LTC-USD",
                "exchange": "hyperliquid",
                "indicators": ["rsi_4h"]
            },
            {
                "name": "empty_indicators",
                "market": "ADA-USD",
                "exchange": "hyperliquid",
                "indicators": []  # Empty indicators (should warn)
            }
        ]

        with patch('app.core.strategy_manager.update_candle_data'), \
             patch('app.core.strategy_manager.update_macd_indicator'), \
             patch('app.core.strategy_manager.update_mvp_signal_state'), \
             patch('app.core.strategy_manager.publish_historical_time_series'), \
             patch('app.core.strategy_manager.verify_historical_data'), \
             patch('app.core.strategy_manager.record_mvp_signal_latency'):

            manager = StrategyManager(
                trading_engine=mock_trading_engine,
                indicators=test_indicators,
                strategies=strategies
            )

        # Should only map valid strategies
        assert len(manager.strategy_indicators) == 2
        assert manager.strategy_indicators["test_strategy_1"] == ["rsi_4h", "macd_1h"]
        assert manager.strategy_indicators["test_strategy_2"] == ["sma_daily"]
        assert "empty_indicators" not in manager.strategy_indicators

    def test_run_strategy_indicators_success(self, strategy_manager, test_indicators):
        """Test run_strategy_indicators() method with successful execution."""
        strategy_config = {
            "name": "test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "timeframe": "4h",
            "indicators": ["rsi_4h", "macd_1h"]
        }

        with patch.object(strategy_manager, '_prepare_indicator_data') as mock_prepare:
            # Mock successful data preparation with sufficient data
            mock_data = pd.DataFrame({
                'timestamp': list(range(1640995200000, 1640995200000 + 30 * 3600000, 3600000)),
                'open': [1950.0 + i for i in range(30)],
                'high': [2050.0 + i for i in range(30)],
                'low': [1900.0 + i for i in range(30)],
                'close': [2000.0 + i for i in range(30)],
                'volume': [1000.0 + i for i in range(30)]
            })
            mock_prepare.return_value = mock_data

            signals = strategy_manager.run_strategy_indicators(
                strategy_config, "ETH-USD", ["rsi_4h", "macd_1h"]
            )

            # Verify signals were generated
            assert len(signals) == 2  # One per indicator

            # Verify signals have strategy context
            for signal in signals:
                assert signal.strategy_name == "test_strategy"
                assert signal.market == "ETH-USD"
                assert signal.exchange == "hyperliquid"
                assert signal.timeframe == "4h"

            # Verify indicators were called with strategy timeframe
            assert test_indicators["rsi_4h"].process_called
            assert test_indicators["macd_1h"].process_called
            assert test_indicators["rsi_4h"].process_timeframe == "4h"
            assert test_indicators["macd_1h"].process_timeframe == "4h"

    def test_run_strategy_indicators_missing_indicators(self, strategy_manager):
        """Test run_strategy_indicators() with missing indicators."""
        strategy_config = {
            "name": "test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "timeframe": "4h",
            "indicators": ["missing_indicator_1", "missing_indicator_2"]
        }

        signals = strategy_manager.run_strategy_indicators(
            strategy_config, "ETH-USD", ["missing_indicator_1", "missing_indicator_2"]
        )

        # Should return empty signals list
        assert signals == []

    def test_run_strategy_indicators_insufficient_data(self, strategy_manager):
        """Test run_strategy_indicators() with insufficient data."""
        strategy_config = {
            "name": "test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "timeframe": "4h",
            "indicators": ["rsi_4h"]
        }

        with patch.object(strategy_manager, '_prepare_indicator_data') as mock_prepare:
            # Mock insufficient data (less than required_periods)
            mock_data = pd.DataFrame({
                'timestamp': [1640995200000],
                'close': [2000.0]
            })
            mock_prepare.return_value = mock_data

            signals = strategy_manager.run_strategy_indicators(
                strategy_config, "ETH-USD", ["rsi_4h"]
            )

            # Should return empty signals due to insufficient data
            assert signals == []

    def test_run_strategy_indicators_no_data(self, strategy_manager):
        """Test run_strategy_indicators() with no data available."""
        strategy_config = {
            "name": "test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "timeframe": "4h",
            "indicators": ["rsi_4h"]
        }

        with patch.object(strategy_manager, '_prepare_indicator_data') as mock_prepare:
            # Mock no data available
            mock_prepare.return_value = pd.DataFrame()

            signals = strategy_manager.run_strategy_indicators(
                strategy_config, "ETH-USD", ["rsi_4h"]
            )

            # Should return empty signals
            assert signals == []

    @patch('app.core.strategy_manager.convert_symbol_for_exchange')
    def test_prepare_indicator_data_symbol_conversion(self, mock_convert, strategy_manager):
        """Test _prepare_indicator_data() symbol conversion."""
        # Setup mock symbol conversion
        mock_convert.return_value = "ETH"  # Hyperliquid format

        with patch.object(strategy_manager, '_fetch_historical_data') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({
                'timestamp': [1640995200000],
                'open': [1950.0],
                'high': [2050.0],
                'low': [1900.0],
                'close': [2000.0],
                'volume': [1000.0]
            })

            result = strategy_manager._prepare_indicator_data(
                market_symbol="ETH-USD",
                timeframe="4h",
                exchange="hyperliquid",
                required_periods=20
            )

            # Verify symbol conversion was called
            mock_convert.assert_called_once_with("ETH-USD", "hyperliquid")

            # Verify fetch was called with converted symbol
            mock_fetch.assert_called_once_with(
                symbol="ETH",  # Converted symbol
                interval="4h",
                limit=strategy_manager.data_window_size,
                periods=20
            )

            # Verify data was returned
            assert not result.empty
            assert len(result) == 1

    @patch('app.core.strategy_manager.convert_symbol_for_exchange')
    def test_prepare_indicator_data_caching(self, mock_convert, strategy_manager):
        """Test _prepare_indicator_data() data caching mechanism."""
        mock_convert.return_value = "ETH"

        # Pre-populate cache with sufficient data
        cache_key = "ETH-USD_4h"
        cached_data = pd.DataFrame({
            'timestamp': list(range(1640995200000, 1640995200000 + 50 * 3600000, 3600000)),
            'close': [2000.0 + i for i in range(50)]
        })
        strategy_manager.price_data[cache_key] = cached_data
        strategy_manager.historical_data_fetched[("ETH-USD", "4h")] = True

        with patch.object(strategy_manager, '_fetch_historical_data') as mock_fetch:
            result = strategy_manager._prepare_indicator_data(
                market_symbol="ETH-USD",
                timeframe="4h",
                exchange="hyperliquid",
                required_periods=20
            )

            # Should not fetch new data when cache has sufficient data
            mock_fetch.assert_not_called()

            # Should return cached data
            assert len(result) == 50
            pd.testing.assert_frame_equal(result, cached_data)

    @pytest.mark.asyncio
    async def test_run_cycle_strategy_driven_execution(self, strategy_manager, test_indicators):
        """Test run_cycle() with strategy-driven execution."""
        with patch.object(strategy_manager, 'run_strategy_indicators') as mock_run_indicators:
            # Mock strategy indicators returning signals
            mock_signals = [
                Mock(strategy_name="eth_multi_timeframe", market="ETH-USD"),
                Mock(strategy_name="btc_daily_strategy", market="BTC-USD")
            ]
            mock_run_indicators.side_effect = [
                [mock_signals[0]],  # eth_multi_timeframe generates 1 signal
                [mock_signals[1]]   # btc_daily_strategy generates 1 signal
            ]

            signal_count = await strategy_manager.run_cycle()

            # Should process signals from enabled strategies only
            assert signal_count == 2

            # Verify strategy indicators were called for enabled strategies
            assert mock_run_indicators.call_count == 2

            # Verify trading engine process_signal was called
            assert strategy_manager.trading_engine.process_signal.call_count == 2

    @pytest.mark.asyncio
    async def test_run_cycle_disabled_strategies_skipped(self, strategy_manager):
        """Test run_cycle() skips disabled strategies."""
        with patch.object(strategy_manager, 'run_strategy_indicators') as mock_run_indicators:
            mock_run_indicators.return_value = []

            signal_count = await strategy_manager.run_cycle()

            # Should only call enabled strategies (2 out of 3)
            assert mock_run_indicators.call_count == 2

            # Should not process disabled_strategy
            call_args = [call[0][0]['name'] for call in mock_run_indicators.call_args_list]
            assert "rsi_4h" in call_args
            assert "macd_1h" in call_args
            assert "disabled_strategy" not in call_args

    @pytest.mark.asyncio
    async def test_run_cycle_invalid_strategy_configurations(self, mock_trading_engine, test_indicators):
        """Test run_cycle() handles invalid strategy configurations."""
        invalid_strategies = [
            {
                "name": "missing_market",
                "exchange": "hyperliquid",
                "indicators": ["rsi_4h"],
                "enabled": True
                # Missing 'market' field
            },
            {
                "name": "missing_exchange",
                "market": "ETH-USD",
                "indicators": ["rsi_4h"],
                "enabled": True
                # Missing 'exchange' field
            },
            {
                "name": "invalid_market_format",
                "market": "ETHUSD",  # Missing separator
                "exchange": "hyperliquid",
                "indicators": ["rsi_4h"],
                "enabled": True
            },
            {
                "name": "no_indicators",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "indicators": [],  # No indicators
                "enabled": True
            }
        ]

        with patch('app.core.strategy_manager.update_candle_data'), \
             patch('app.core.strategy_manager.update_macd_indicator'), \
             patch('app.core.strategy_manager.update_mvp_signal_state'), \
             patch('app.core.strategy_manager.publish_historical_time_series'), \
             patch('app.core.strategy_manager.verify_historical_data'), \
             patch('app.core.strategy_manager.record_mvp_signal_latency'):

            manager = StrategyManager(
                trading_engine=mock_trading_engine,
                indicators=test_indicators,
                strategies=invalid_strategies
            )

        with patch.object(manager, 'run_strategy_indicators') as mock_run_indicators:
            signal_count = await manager.run_cycle()

            # Should not call run_strategy_indicators for any invalid strategies
            mock_run_indicators.assert_not_called()
            assert signal_count == 0

    @pytest.mark.asyncio
    async def test_run_cycle_symbol_filtering(self, strategy_manager):
        """Test run_cycle() with symbol filtering."""
        with patch.object(strategy_manager, 'run_strategy_indicators') as mock_run_indicators:
            mock_run_indicators.return_value = []

            # Filter to only ETH-USD
            signal_count = await strategy_manager.run_cycle(symbols=["ETH-USD"])

            # Should only process strategies for ETH-USD
            assert mock_run_indicators.call_count == 1
            call_args = mock_run_indicators.call_args_list[0][0]
            assert call_args[0]['name'] == "rsi_4h"
            assert call_args[1] == "ETH-USD"

    @pytest.mark.asyncio
    async def test_run_cycle_signal_processing_error_handling(self, strategy_manager):
        """Test run_cycle() handles signal processing errors gracefully."""
        with patch.object(strategy_manager, 'run_strategy_indicators') as mock_run_indicators:
            # Mock signal generation
            mock_signal = Mock(strategy_name="test_strategy", market="ETH-USD")
            mock_run_indicators.return_value = [mock_signal]

            # Mock trading engine to raise exception on first signal, succeed on second
            strategy_manager.trading_engine.process_signal.side_effect = [
                Exception("Processing error"),
                True
            ]

            signal_count = await strategy_manager.run_cycle()

            # Should handle error gracefully and continue processing
            # Only btc_daily_strategy should succeed (eth_multi_timeframe fails)
            assert signal_count == 1

    @pytest.mark.asyncio
    async def test_run_cycle_no_strategies_configured(self, mock_trading_engine, test_indicators):
        """Test run_cycle() with no strategies configured."""
        with patch('app.core.strategy_manager.update_candle_data'), \
             patch('app.core.strategy_manager.update_macd_indicator'), \
             patch('app.core.strategy_manager.update_mvp_signal_state'), \
             patch('app.core.strategy_manager.publish_historical_time_series'), \
             patch('app.core.strategy_manager.verify_historical_data'), \
             patch('app.core.strategy_manager.record_mvp_signal_latency'):

            manager = StrategyManager(
                trading_engine=mock_trading_engine,
                indicators=test_indicators,
                strategies=[]  # No strategies
            )

        signal_count = await manager.run_cycle()

        # Should return 0 signals
        assert signal_count == 0

    def test_signal_generation_with_strategy_context(self, strategy_manager):
        """Test that signals are properly enhanced with strategy context."""
        strategy_config = {
            "name": "context_test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "timeframe": "4h"
        }

        # Create a custom indicator that returns a signal without context
        class ContextTestIndicator(BaseIndicator):
            def __init__(self, name: str, params: dict = None):
                super().__init__(name, params)
                self.required_periods = 10  # Set required periods

            def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
                """Mock calculate method."""
                return data

            def generate_signal(self, data: pd.DataFrame) -> Signal:
                """Mock generate_signal method."""
                return Signal(
                    direction=SignalDirection.BUY,
                    symbol="ETH-USD",
                    indicator="context_test",
                    confidence=0.5
                )

            def process(self, data: pd.DataFrame, strategy_timeframe: str = None) -> tuple[pd.DataFrame, Signal]:
                signal = Signal(
                    direction=SignalDirection.BUY,
                    symbol="ETH-USD",
                    indicator="context_test",
                    confidence=0.5
                )
                return data, signal

        test_indicator = ContextTestIndicator("context_test", params={'timeframe': "4h"})
        strategy_manager.indicators["context_test"] = test_indicator

        with patch.object(strategy_manager, '_prepare_indicator_data') as mock_prepare:
            mock_prepare.return_value = pd.DataFrame({
                'timestamp': [1640995200000] * 20,
                'close': [2000.0] * 20
            })

            signals = strategy_manager.run_strategy_indicators(
                strategy_config, "ETH-USD", ["context_test"]
            )

            assert len(signals) == 1
            signal = signals[0]

            # Verify strategy context was added
            assert signal.strategy_name == "context_test_strategy"
            assert signal.market == "ETH-USD"
            assert signal.exchange == "hyperliquid"
            assert signal.timeframe == "4h"

    def test_error_handling_indicator_process_exception(self, strategy_manager):
        """Test error handling when indicator process() raises exception."""
        strategy_config = {
            "name": "error_test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "timeframe": "4h"
        }

        # Create an indicator that raises an exception
        class ErrorIndicator(BaseIndicator):
            def __init__(self, name: str, params: dict = None):
                super().__init__(name, params)
                self.required_periods = 10  # Set required periods

            def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
                """Mock calculate method."""
                return data

            def generate_signal(self, data: pd.DataFrame) -> Signal:
                """Mock generate_signal method."""
                return None

            def process(self, data: pd.DataFrame, strategy_timeframe: str = None) -> tuple[pd.DataFrame, Signal]:
                raise ValueError("Test indicator error")

        error_indicator = ErrorIndicator("error_test", params={'timeframe': "4h"})
        error_indicator.required_periods = 10
        strategy_manager.indicators["error_test"] = error_indicator

        with patch.object(strategy_manager, '_prepare_indicator_data') as mock_prepare:
            mock_prepare.return_value = pd.DataFrame({
                'timestamp': [1640995200000] * 20,
                'close': [2000.0] * 20
            })

            signals = strategy_manager.run_strategy_indicators(
                strategy_config, "ETH-USD", ["error_test"]
            )

            # Should handle error gracefully and return empty signals
            assert signals == []

    @patch('app.core.strategy_manager.convert_symbol_for_exchange')
    def test_symbol_conversion_error_handling(self, mock_convert, strategy_manager):
        """Test error handling when symbol conversion fails."""
        # Mock symbol conversion to raise exception
        mock_convert.side_effect = ValueError("Unsupported exchange")

        result = strategy_manager._prepare_indicator_data(
            market_symbol="ETH-USD",
            timeframe="4h",
            exchange="unknown_exchange",
            required_periods=20
        )

        # Should handle error gracefully and return empty DataFrame
        assert result.empty

    def test_strategy_timeframe_override(self, strategy_manager, test_indicators):
        """Test that strategy timeframe overrides indicator default timeframe."""
        strategy_config = {
            "name": "timeframe_test",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "timeframe": "1h"  # Different from indicator's default "4h"
        }

        with patch.object(strategy_manager, '_prepare_indicator_data') as mock_prepare:
            mock_prepare.return_value = pd.DataFrame({
                'timestamp': [1640995200000] * 20,
                'close': [2000.0] * 20
            })

            signals = strategy_manager.run_strategy_indicators(
                strategy_config, "ETH-USD", ["rsi_4h"]
            )

            # Verify _prepare_indicator_data was called with strategy timeframe
            # The actual required_periods calculation takes the maximum from all indicators
            mock_prepare.assert_called_once_with(
                market_symbol="ETH-USD",
                timeframe="1h",  # Strategy timeframe, not indicator's "4h"
                exchange="hyperliquid",
                required_periods=50  # Default if no specific periods set or max from multiple indicators
            )

            # Verify indicator was called with strategy timeframe
            assert test_indicators["rsi_4h"].process_timeframe == "1h"
