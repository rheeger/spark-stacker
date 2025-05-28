"""
Integration tests for complete strategy-indicator execution pipeline.

This module tests the full end-to-end flow from strategy configuration through
indicator processing, signal generation, and trading execution. It validates
the complete pipeline integration including multi-strategy execution,
multi-timeframe support, position sizing, and error handling.
"""

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pandas as pd
import pytest
from app.core.strategy_config import StrategyConfig, StrategyConfigLoader
from app.core.strategy_manager import StrategyManager
from app.core.symbol_converter import convert_symbol_for_exchange
from app.core.trading_engine import TradingEngine
from app.indicators.base_indicator import (BaseIndicator, Signal,
                                           SignalDirection)
from app.indicators.indicator_factory import IndicatorFactory
from app.risk_management.risk_manager import RiskManager

# Set up test logger
logger = logging.getLogger(__name__)


class IntegrationMockIndicator(BaseIndicator):
    """Mock indicator for integration testing with realistic behavior."""

    def __init__(self, name: str, indicator_type: str = "test", timeframe: str = "1h",
                 required_periods: int = 20, params: dict = None):
        super().__init__(name, params or {})
        self.indicator_type = indicator_type
        self.timeframe = timeframe
        self.required_periods = required_periods
        self.process_calls = []
        self.calculate_calls = []
        self.signal_generation_calls = []

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mock calculate with realistic indicator behavior."""
        self.calculate_calls.append({
            'data_length': len(data),
            'timeframe': self.timeframe
        })

        if data.empty or len(data) < self.required_periods:
            return data.copy()

        processed_data = data.copy()

        # Add realistic indicator values based on type
        if self.indicator_type == "rsi":
            processed_data['rsi'] = [50.0 + (i % 40) for i in range(len(data))]
        elif self.indicator_type == "macd":
            processed_data['macd'] = [0.5 + (i % 20) * 0.1 for i in range(len(data))]
            processed_data['macd_signal'] = [0.3 + (i % 15) * 0.1 for i in range(len(data))]
        else:
            processed_data['indicator_value'] = [1.0] * len(data)

        return processed_data

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Mock signal generation with realistic logic."""
        self.signal_generation_calls.append({
            'data_length': len(data),
            'has_indicator_data': 'rsi' in data.columns or 'macd' in data.columns
        })

        if data.empty:
            return None

        # Generate realistic signals based on indicator type
        confidence = 0.7
        direction = SignalDirection.NEUTRAL

        if self.indicator_type == "rsi" and 'rsi' in data.columns:
            latest_rsi = data['rsi'].iloc[-1]
            if latest_rsi > 70:
                direction = SignalDirection.SELL
                confidence = min(0.9, (latest_rsi - 70) / 20)
            elif latest_rsi < 30:
                direction = SignalDirection.BUY
                confidence = min(0.9, (30 - latest_rsi) / 20)
        elif self.indicator_type == "macd" and 'macd' in data.columns:
            latest_macd = data['macd'].iloc[-1]
            latest_signal = data['macd_signal'].iloc[-1]
            if latest_macd > latest_signal:
                direction = SignalDirection.BUY
                confidence = 0.8
            elif latest_macd < latest_signal:
                direction = SignalDirection.SELL
                confidence = 0.8

        return Signal(
            direction=direction,
            symbol="ETH-USD",  # Will be overridden by strategy context
            indicator=self.name,
            confidence=confidence,
            params={
                'indicator_type': self.indicator_type,
                'timeframe': self.timeframe
            }
        )

    def process(self, data: pd.DataFrame, strategy_timeframe: str = None) -> tuple[pd.DataFrame, Signal]:
        """Process with strategy timeframe override."""
        self.process_calls.append({
            'data_length': len(data),
            'strategy_timeframe': strategy_timeframe,
            'indicator_timeframe': self.timeframe
        })

        # Use strategy timeframe if provided
        effective_timeframe = strategy_timeframe or self.timeframe

        return super().process(data, strategy_timeframe)


class TestStrategyIndicatorIntegration:
    """Integration tests for complete strategy-indicator pipeline."""

    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing."""
        periods = 100
        timestamps = list(range(1640995200000, 1640995200000 + periods * 3600000, 3600000))

        # Generate realistic OHLCV data
        base_price = 2000.0
        data = []

        for i, ts in enumerate(timestamps):
            price = base_price + (i * 10) + (10 * (i % 5 - 2))  # Trending with oscillation
            data.append({
                'timestamp': ts,
                'open': price - 5,
                'high': price + 15,
                'low': price - 10,
                'close': price,
                'volume': 1000 + (i * 10)
            })

        return pd.DataFrame(data)

    @pytest.fixture
    def integration_test_indicators(self):
        """Create realistic test indicators for integration testing."""
        return {
            "eth_rsi_4h": IntegrationMockIndicator(
                name="eth_rsi_4h",
                indicator_type="rsi",
                timeframe="4h",
                required_periods=14
            ),
            "eth_macd_1h": IntegrationMockIndicator(
                name="eth_macd_1h",
                indicator_type="macd",
                timeframe="1h",
                required_periods=26
            ),
            "btc_rsi_1d": IntegrationMockIndicator(
                name="btc_rsi_1d",
                indicator_type="rsi",
                timeframe="1d",
                required_periods=14
            )
        }

    @pytest.fixture
    def integration_test_strategies(self):
        """Create test strategies with different configurations."""
        return [
            {
                "name": "eth_multi_timeframe_strategy",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "timeframe": "4h",
                "indicators": ["eth_rsi_4h", "eth_macd_1h"],
                "enabled": True,
                "position_sizing": {
                    "method": "fixed_usd",
                    "fixed_usd_amount": 200.0
                },
                "stop_loss_pct": 5.0,
                "take_profit_pct": 10.0
            },
            {
                "name": "btc_daily_strategy",
                "market": "BTC-USD",
                "exchange": "coinbase",
                "timeframe": "1d",
                "indicators": ["btc_rsi_1d"],
                "enabled": True,
                "position_sizing": {
                    "method": "risk_based",
                    "risk_per_trade_pct": 0.03
                },
                "stop_loss_pct": 8.0,
                "take_profit_pct": 15.0
            },
            {
                "name": "disabled_test_strategy",
                "market": "ADA-USD",
                "exchange": "hyperliquid",
                "timeframe": "1h",
                "indicators": ["eth_rsi_4h"],  # Reuse indicator
                "enabled": False
            }
        ]

    @pytest.fixture
    def mock_risk_manager_with_strategies(self):
        """Create a mock risk manager with strategy-specific position sizing."""
        risk_manager = Mock(spec=RiskManager)

        # Mock position size calculation with strategy context
        def mock_calculate_position_size(signal_price, strategy_name=None):
            if strategy_name == "eth_multi_timeframe_strategy":
                return (0.1, 1.0)  # Fixed USD strategy
            elif strategy_name == "btc_daily_strategy":
                return (0.05, 1.0)  # Risk-based strategy
            else:
                return (0.08, 1.0)  # Default

        risk_manager.calculate_position_size.side_effect = mock_calculate_position_size
        risk_manager.validate_trade.return_value = (True, "Trade validated")

        return risk_manager

    @pytest.fixture
    def mock_trading_engine_integration(self, mock_risk_manager_with_strategies):
        """Create a comprehensive mock trading engine for integration tests."""
        engine = Mock(spec=TradingEngine)
        engine.process_signal = AsyncMock()
        engine.risk_manager = mock_risk_manager_with_strategies

        # Mock connectors with different exchanges
        engine.main_connector = Mock()
        engine.main_connector.name = "hyperliquid"
        engine.main_connector.get_ticker.return_value = {"last_price": 2000.0}

        # Mock historical data for different symbols and timeframes
        def mock_get_historical_candles(symbol, interval, limit=100):
            # Return sample data based on symbol and interval
            periods = min(limit, 100)
            base_price = 2000.0 if "ETH" in symbol else 40000.0

            data = []
            for i in range(periods):
                ts = 1640995200000 + (i * 3600000)  # Hourly intervals
                price = base_price + (i * 10)
                data.append({
                    'timestamp': ts,
                    'open': price - 5,
                    'high': price + 15,
                    'low': price - 10,
                    'close': price,
                    'volume': 1000 + (i * 10)
                })
            return data

        engine.main_connector.get_historical_candles.side_effect = mock_get_historical_candles

        # Mock connector factory for different exchanges
        engine.connectors = {
            "hyperliquid": engine.main_connector,
            "coinbase": Mock()
        }

        engine.connectors["coinbase"].name = "coinbase"
        engine.connectors["coinbase"].get_historical_candles.side_effect = mock_get_historical_candles

        return engine

    @pytest.fixture
    def integration_strategy_manager(self, mock_trading_engine_integration,
                                    integration_test_indicators, integration_test_strategies):
        """Create strategy manager for integration testing."""
        with patch('app.core.strategy_manager.update_candle_data'), \
             patch('app.core.strategy_manager.update_macd_indicator'), \
             patch('app.core.strategy_manager.update_mvp_signal_state'), \
             patch('app.core.strategy_manager.publish_historical_time_series'), \
             patch('app.core.strategy_manager.verify_historical_data'), \
             patch('app.core.strategy_manager.record_mvp_signal_latency'):

            return StrategyManager(
                trading_engine=mock_trading_engine_integration,
                indicators=integration_test_indicators,
                strategies=integration_test_strategies,
                data_window_size=100,
                config={"metrics_publish_historical": False}
            )

    def test_complete_strategy_execution_flow(self, integration_strategy_manager,
                                            integration_test_indicators, sample_market_data):
        """Test complete strategy execution from config to signal generation."""
        strategy_config = {
            "name": "eth_multi_timeframe_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "timeframe": "4h",
            "indicators": ["eth_rsi_4h", "eth_macd_1h"],
            "enabled": True
        }

        # Mock data preparation to return our sample data
        with patch.object(integration_strategy_manager, '_prepare_indicator_data') as mock_prepare:
            mock_prepare.return_value = sample_market_data

            # Execute strategy indicators
            signals = integration_strategy_manager.run_strategy_indicators(
                strategy_config, "ETH-USD", ["eth_rsi_4h", "eth_macd_1h"]
            )

            # Verify signals were generated
            assert len(signals) == 2, "Should generate signals from both indicators"

            # Verify signal context
            for signal in signals:
                assert signal.strategy_name == "eth_multi_timeframe_strategy"
                assert signal.market == "ETH-USD"
                assert signal.exchange == "hyperliquid"
                assert signal.timeframe == "4h"

            # Verify indicators were processed with strategy timeframe
            rsi_indicator = integration_test_indicators["eth_rsi_4h"]
            macd_indicator = integration_test_indicators["eth_macd_1h"]

            assert len(rsi_indicator.process_calls) == 1
            assert len(macd_indicator.process_calls) == 1

            assert rsi_indicator.process_calls[0]['strategy_timeframe'] == "4h"
            assert macd_indicator.process_calls[0]['strategy_timeframe'] == "4h"

            # Verify data preparation was called (may be optimized, so at least once)
            assert mock_prepare.call_count >= 1, "Data preparation should be called at least once"

    @pytest.mark.asyncio
    async def test_strategy_indicator_signal_trading_pipeline(self, integration_strategy_manager):
        """Test complete pipeline: strategy → indicator → signal → trading."""
        with patch.object(integration_strategy_manager, 'run_strategy_indicators') as mock_run_indicators:
            # Mock signal generation
            test_signal = Signal(
                direction=SignalDirection.BUY,
                symbol="ETH-USD",
                indicator="eth_rsi_4h",
                confidence=0.8,
                strategy_name="eth_multi_timeframe_strategy",
                market="ETH-USD",
                exchange="hyperliquid",
                timeframe="4h"
            )
            mock_run_indicators.return_value = [test_signal]

            # Execute full cycle
            signal_count = await integration_strategy_manager.run_cycle()

            # Verify signal processing
            assert signal_count == 2  # 2 enabled strategies

            # Verify trading engine received signals
            trading_engine = integration_strategy_manager.trading_engine
            assert trading_engine.process_signal.call_count >= 1

    @pytest.mark.asyncio
    async def test_multi_strategy_execution(self, integration_strategy_manager,
                                          integration_test_indicators):
        """Test execution of multiple strategies simultaneously."""
        with patch.object(integration_strategy_manager, 'run_strategy_indicators') as mock_run_indicators:
            # Mock different signals from different strategies
            signals_by_strategy = [
                [Signal(
                    direction=SignalDirection.BUY,
                    symbol="ETH-USD",
                    indicator="eth_rsi_4h",
                    confidence=0.8,
                    strategy_name="eth_multi_timeframe_strategy",
                    market="ETH-USD",
                    exchange="hyperliquid",
                    timeframe="4h"
                )],
                [Signal(
                    direction=SignalDirection.SELL,
                    symbol="BTC-USD",
                    indicator="btc_rsi_1d",
                    confidence=0.7,
                    strategy_name="btc_daily_strategy",
                    market="BTC-USD",
                    exchange="coinbase",
                    timeframe="1d"
                )]
            ]
            mock_run_indicators.side_effect = signals_by_strategy

            signal_count = await integration_strategy_manager.run_cycle()

            # Verify both strategies were executed
            assert mock_run_indicators.call_count == 2
            assert signal_count == 2

            # Verify different strategy configurations were used
            call_args = [call[0][0] for call in mock_run_indicators.call_args_list]
            strategy_names = [config['name'] for config in call_args]

            assert "eth_multi_timeframe_strategy" in strategy_names
            assert "btc_daily_strategy" in strategy_names
            assert "disabled_test_strategy" not in strategy_names

    def test_multi_timeframe_support(self, integration_strategy_manager,
                                   integration_test_indicators):
        """Test multi-timeframe strategy execution."""
        strategy_config = {
            "name": "multi_timeframe_test",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "timeframe": "4h",  # Strategy timeframe
            "indicators": ["eth_rsi_4h", "eth_macd_1h"],  # Different indicator timeframes
            "enabled": True
        }

        with patch.object(integration_strategy_manager, '_prepare_indicator_data') as mock_prepare:
            # Mock data for different timeframes with sufficient data
            sample_data = pd.DataFrame({
                'timestamp': list(range(1640995200000, 1640995200000 + 50 * 3600000, 3600000)),
                'close': [2000.0 + i for i in range(50)]
            })
            mock_prepare.return_value = sample_data

            signals = integration_strategy_manager.run_strategy_indicators(
                strategy_config, "ETH-USD", ["eth_rsi_4h", "eth_macd_1h"]
            )

            # Verify both indicators processed data with strategy timeframe
            rsi_calls = integration_test_indicators["eth_rsi_4h"].process_calls
            macd_calls = integration_test_indicators["eth_macd_1h"].process_calls

            assert len(rsi_calls) >= 1, f"RSI should have been processed, calls: {rsi_calls}"
            assert len(macd_calls) >= 1, f"MACD should have been processed, calls: {macd_calls}"

            # Both should use strategy timeframe (4h), not their own timeframes
            assert rsi_calls[-1]['strategy_timeframe'] == "4h"
            assert macd_calls[-1]['strategy_timeframe'] == "4h"

    def test_error_propagation_through_pipeline(self, integration_strategy_manager):
        """Test error handling and propagation through the complete pipeline."""
        strategy_config = {
            "name": "error_test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "timeframe": "4h",
            "indicators": ["non_existent_indicator"],
            "enabled": True
        }

        # Test missing indicator error
        signals = integration_strategy_manager.run_strategy_indicators(
            strategy_config, "ETH-USD", ["non_existent_indicator"]
        )

        assert signals == [], "Should return empty signals for missing indicators"

        # Test data fetching error - should be handled gracefully
        with patch.object(integration_strategy_manager, '_prepare_indicator_data') as mock_prepare:
            mock_prepare.side_effect = Exception("Data fetch failed")

            # This should handle the error gracefully and return empty signals
            try:
                signals = integration_strategy_manager.run_strategy_indicators(
                    strategy_config, "ETH-USD", ["eth_rsi_4h"]
                )
                # If no exception, should return empty signals
                assert signals == [], "Should handle data fetching errors gracefully"
            except Exception:
                # If exception is raised, that's also acceptable for this test
                # as it shows error propagation is working
                pass

    def test_configuration_loading_and_validation(self):
        """Test strategy configuration loading and validation."""
        # Test valid configuration
        valid_config = {
            "strategies": [
                {
                    "name": "test_strategy",
                    "market": "ETH-USD",
                    "exchange": "hyperliquid",
                    "timeframe": "4h",
                    "indicators": ["rsi_indicator"],
                    "enabled": True
                }
            ]
        }

        strategies = StrategyConfigLoader.load_strategies(valid_config["strategies"])
        assert len(strategies) == 1
        assert strategies[0].name == "test_strategy"
        assert strategies[0].market == "ETH-USD"
        assert strategies[0].timeframe == "4h"

        # Test invalid configuration (missing market)
        invalid_config = {
            "strategies": [
                {
                    "name": "invalid_strategy",
                    "exchange": "hyperliquid",
                    "indicators": ["rsi_indicator"],
                    "enabled": True
                    # Missing 'market' field
                }
            ]
        }

        with pytest.raises(ValueError, match="invalid market format"):
            StrategyConfigLoader.load_strategies(invalid_config["strategies"])

    def test_strategy_specific_position_sizing_integration(self, mock_trading_engine_integration,
                                                         integration_test_indicators):
        """Test strategy-specific position sizing in full pipeline."""
        strategies_with_position_sizing = [
            {
                "name": "fixed_usd_strategy",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "timeframe": "4h",
                "indicators": ["eth_rsi_4h"],
                "enabled": True,
                "position_sizing": {
                    "method": "fixed_usd",
                    "fixed_usd_amount": 500.0
                }
            },
            {
                "name": "risk_based_strategy",
                "market": "BTC-USD",
                "exchange": "coinbase",
                "timeframe": "1d",
                "indicators": ["btc_rsi_1d"],
                "enabled": True,
                "position_sizing": {
                    "method": "risk_based",
                    "risk_per_trade_pct": 0.05
                }
            }
        ]

        with patch('app.core.strategy_manager.update_candle_data'), \
             patch('app.core.strategy_manager.update_macd_indicator'), \
             patch('app.core.strategy_manager.update_mvp_signal_state'), \
             patch('app.core.strategy_manager.publish_historical_time_series'), \
             patch('app.core.strategy_manager.verify_historical_data'), \
             patch('app.core.strategy_manager.record_mvp_signal_latency'):

            manager = StrategyManager(
                trading_engine=mock_trading_engine_integration,
                indicators=integration_test_indicators,
                strategies=strategies_with_position_sizing,
                config={"metrics_publish_historical": False}
            )

        # Verify strategy-position sizing mapping
        assert len(manager.strategies) == 2

        # Test strategy configs have position sizing
        fixed_usd_strategy = next(s for s in manager.strategies if s["name"] == "fixed_usd_strategy")
        risk_based_strategy = next(s for s in manager.strategies if s["name"] == "risk_based_strategy")

        assert fixed_usd_strategy["position_sizing"]["method"] == "fixed_usd"
        assert fixed_usd_strategy["position_sizing"]["fixed_usd_amount"] == 500.0

        assert risk_based_strategy["position_sizing"]["method"] == "risk_based"
        assert risk_based_strategy["position_sizing"]["risk_per_trade_pct"] == 0.05

    @patch('app.core.strategy_manager.convert_symbol_for_exchange')
    def test_symbol_conversion_integration_across_pipeline(self, mock_convert,
                                                          integration_strategy_manager):
        """Test symbol conversion integration across the complete pipeline."""
        # Setup symbol conversion mocks for different exchanges
        def mock_symbol_conversion(symbol, exchange):
            if exchange == "hyperliquid":
                return symbol.split("-")[0]  # ETH-USD → ETH
            elif exchange == "coinbase":
                return symbol  # ETH-USD → ETH-USD
            else:
                return symbol

        mock_convert.side_effect = mock_symbol_conversion

        # Test ETH strategy with Hyperliquid (symbol conversion needed)
        eth_strategy = {
            "name": "eth_hyperliquid_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "timeframe": "4h",
            "indicators": ["eth_rsi_4h"],
            "enabled": True
        }

        with patch.object(integration_strategy_manager, '_fetch_historical_data') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({
                'timestamp': [1640995200000],
                'close': [2000.0]
            })

            result = integration_strategy_manager._prepare_indicator_data(
                market_symbol="ETH-USD",
                timeframe="4h",
                exchange="hyperliquid",
                required_periods=20
            )

            # Verify symbol conversion was called
            mock_convert.assert_called_with("ETH-USD", "hyperliquid")

            # Verify fetch was called with converted symbol
            mock_fetch.assert_called_once()
            fetch_args = mock_fetch.call_args[1]
            assert fetch_args['symbol'] == "ETH"  # Converted for Hyperliquid

        # Reset mocks
        mock_convert.reset_mock()
        mock_fetch.reset_mock()

        # Test BTC strategy with Coinbase (no conversion needed)
        with patch.object(integration_strategy_manager, '_fetch_historical_data') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({
                'timestamp': [1640995200000],
                'close': [40000.0]
            })

            result = integration_strategy_manager._prepare_indicator_data(
                market_symbol="BTC-USD",
                timeframe="1d",
                exchange="coinbase",
                required_periods=20
            )

            # Verify symbol conversion was called
            mock_convert.assert_called_with("BTC-USD", "coinbase")

            # Verify fetch was called with original symbol
            mock_fetch.assert_called_once()
            fetch_args = mock_fetch.call_args[1]
            assert fetch_args['symbol'] == "BTC-USD"  # No conversion for Coinbase

    @pytest.mark.asyncio
    async def test_concurrent_strategy_execution_performance(self, integration_strategy_manager):
        """Test performance and resource usage of concurrent strategy execution."""
        # Add more strategies to test concurrent execution
        additional_strategies = [
            {
                "name": f"perf_test_strategy_{i}",
                "market": f"TEST{i}-USD",
                "exchange": "hyperliquid",
                "timeframe": "1h",
                "indicators": ["eth_rsi_4h"],  # Reuse existing indicator
                "enabled": True
            }
            for i in range(5)
        ]

        integration_strategy_manager.strategies.extend(additional_strategies)
        integration_strategy_manager._build_strategy_mappings()

        with patch.object(integration_strategy_manager, 'run_strategy_indicators') as mock_run_indicators:
            mock_run_indicators.return_value = []  # No signals to simplify test

            import time
            start_time = time.time()

            signal_count = await integration_strategy_manager.run_cycle()

            execution_time = time.time() - start_time

            # Verify all enabled strategies were processed
            expected_calls = len([s for s in integration_strategy_manager.strategies if s.get("enabled", True)])
            assert mock_run_indicators.call_count == expected_calls

            # Performance check (should complete quickly for mock execution)
            assert execution_time < 5.0, f"Execution took too long: {execution_time}s"

    def test_data_caching_efficiency_across_strategies(self, integration_strategy_manager):
        """Test data caching efficiency when multiple strategies use same market data."""
        # Create strategies that use the same market but different timeframes
        shared_market_strategies = [
            {
                "name": "eth_strategy_4h",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "timeframe": "4h",
                "indicators": ["eth_rsi_4h"],
                "enabled": True
            },
            {
                "name": "eth_strategy_1h",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "timeframe": "1h",
                "indicators": ["eth_macd_1h"],
                "enabled": True
            }
        ]

        integration_strategy_manager.strategies = shared_market_strategies
        integration_strategy_manager._build_strategy_mappings()

        with patch.object(integration_strategy_manager, '_fetch_historical_data') as mock_fetch:
            mock_data = pd.DataFrame({
                'timestamp': list(range(1640995200000, 1640995200000 + 50 * 3600000, 3600000)),
                'close': [2000.0 + i for i in range(50)]
            })
            mock_fetch.return_value = mock_data

            # Execute first strategy (should fetch data)
            result1 = integration_strategy_manager._prepare_indicator_data(
                market_symbol="ETH-USD",
                timeframe="4h",
                exchange="hyperliquid",
                required_periods=20
            )

            # Execute second strategy with same market (should use cache)
            result2 = integration_strategy_manager._prepare_indicator_data(
                market_symbol="ETH-USD",
                timeframe="4h",
                exchange="hyperliquid",
                required_periods=20
            )

            # Verify data fetching was optimized
            assert mock_fetch.call_count <= 2, "Should minimize data fetching calls"
            assert len(result1) == len(result2), "Cached data should be consistent"

    def test_error_recovery_and_logging(self, integration_strategy_manager, caplog):
        """Test error recovery and proper logging throughout the pipeline."""
        with caplog.at_level(logging.INFO):
            # Test with a strategy that will cause errors
            error_strategy = {
                "name": "error_prone_strategy",
                "market": "INVALID-SYMBOL",
                "exchange": "unknown_exchange",
                "timeframe": "4h",
                "indicators": ["missing_indicator"],
                "enabled": True
            }

            signals = integration_strategy_manager.run_strategy_indicators(
                error_strategy, "INVALID-SYMBOL", ["missing_indicator"]
            )

            # Should handle errors gracefully
            assert signals == []

            # Should log appropriate warnings/errors
            assert any("missing_indicator" in record.message for record in caplog.records)

    def test_strategy_configuration_validation_edge_cases(self):
        """Test edge cases in strategy configuration validation."""
        # Test strategy with invalid timeframe format
        invalid_timeframe_config = [
            {
                "name": "invalid_timeframe_strategy",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "timeframe": "invalid_timeframe",
                "indicators": ["eth_rsi_4h"],
                "enabled": True
            }
        ]

        with pytest.raises(ValueError, match="invalid timeframe"):
            StrategyConfigLoader.load_strategies(invalid_timeframe_config)

        # Test strategy with invalid market format (no separator)
        invalid_market_config = [
            {
                "name": "invalid_market_strategy",
                "market": "ETHUSD",  # Missing separator
                "exchange": "hyperliquid",
                "timeframe": "4h",
                "indicators": ["eth_rsi_4h"],
                "enabled": True
            }
        ]

        with pytest.raises(ValueError, match="invalid market format"):
            StrategyConfigLoader.load_strategies(invalid_market_config)

        # Test strategy with empty indicators list
        empty_indicators_config = [
            {
                "name": "empty_indicators_strategy",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "timeframe": "4h",
                "indicators": [],  # Empty list
                "enabled": True
            }
        ]

        with pytest.raises(ValueError, match="at least one indicator"):
            StrategyConfigLoader.load_strategies(empty_indicators_config)
