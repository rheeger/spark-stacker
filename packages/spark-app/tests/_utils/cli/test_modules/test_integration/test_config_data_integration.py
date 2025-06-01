"""
Integration tests for ConfigManager and DataManager interaction.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from core.config_manager import ConfigManager
from core.data_manager import DataFetchError, DataManager


class TestConfigDataIntegration:
    """Test suite for ConfigManager and DataManager integration."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        config_data = {
            "exchange_configs": {
                "hyperliquid": {
                    "name": "Hyperliquid",
                    "markets": ["ETH-USD", "BTC-USD"],
                    "timeframes": ["1m", "5m", "1h", "4h", "1d"]
                },
                "coinbase": {
                    "name": "Coinbase",
                    "markets": ["ETH-USD", "BTC-USD"],
                    "timeframes": ["1m", "5m", "1h", "6h", "1d"]
                }
            },
            "strategy_configs": {
                "eth_momentum_strategy": {
                    "name": "ETH Momentum Strategy",
                    "market": "ETH-USD",
                    "exchange": "hyperliquid",
                    "timeframe": "1h",
                    "indicators": {
                        "rsi": {
                            "class": "RSIIndicator",
                            "timeframe": "1h",
                            "window": 14
                        },
                        "macd": {
                            "class": "MACDIndicator",
                            "timeframe": "4h",
                            "fast_period": 12,
                            "slow_period": 26
                        }
                    },
                    "position_sizing": {
                        "method": "fixed_usd",
                        "amount": 100
                    },
                    "enabled": True
                },
                "btc_scalping_strategy": {
                    "name": "BTC Scalping Strategy",
                    "market": "BTC-USD",
                    "exchange": "coinbase",
                    "timeframe": "5m",
                    "indicators": {
                        "ema": {
                            "class": "EMAIndicator",
                            "timeframe": "5m",
                            "window": 20
                        }
                    },
                    "position_sizing": {
                        "method": "percentage",
                        "percentage": 2.0
                    },
                    "enabled": True
                }
            },
            "global_settings": {
                "default_position_sizing": {
                    "method": "fixed_usd",
                    "amount": 50
                },
                "cache_settings": {
                    "enabled": True,
                    "timeout_hours": 1
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name

        yield temp_file

        # Cleanup
        os.unlink(temp_file)

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def config_manager(self, temp_config_file):
        """Create a ConfigManager instance with test config."""
        return ConfigManager(config_path=temp_config_file)

    @pytest.fixture
    def data_manager(self, temp_cache_dir):
        """Create a DataManager instance with test cache directory."""
        return DataManager(cache_dir=temp_cache_dir)

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=168, freq='1H')  # 7 days of hourly data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [100 + i * 0.1 for i in range(168)],
            'high': [101 + i * 0.1 for i in range(168)],
            'low': [99 + i * 0.1 for i in range(168)],
            'close': [100.5 + i * 0.1 for i in range(168)],
            'volume': [1000 + i * 10 for i in range(168)]
        })
        return data

    def test_strategy_data_requirements_extraction(self, config_manager, data_manager):
        """Test extracting data requirements from strategy configuration."""
        strategy_config = config_manager.get_strategy_config("eth_momentum_strategy")

        # Extract data requirements
        market = strategy_config["market"]
        exchange = strategy_config["exchange"]
        timeframes = set()

        # Strategy timeframe
        timeframes.add(strategy_config["timeframe"])

        # Indicator timeframes
        for indicator_config in strategy_config["indicators"].values():
            timeframes.add(indicator_config["timeframe"])

        # Verify extracted requirements
        assert market == "ETH-USD"
        assert exchange == "hyperliquid"
        assert "1h" in timeframes
        assert "4h" in timeframes
        assert len(timeframes) == 2

    @patch('core.data_manager.HyperliquidConnector')
    def test_multi_timeframe_data_fetching(self, mock_connector, config_manager, data_manager, sample_market_data):
        """Test fetching data for multiple timeframes required by a strategy."""
        # Mock the HyperliquidConnector
        mock_instance = MagicMock()
        mock_instance.get_historical_candles.return_value = sample_market_data.to_dict('records')
        mock_connector.return_value = mock_instance

        strategy_config = config_manager.get_strategy_config("eth_momentum_strategy")

        # Get required timeframes
        timeframes = set([strategy_config["timeframe"]])
        for indicator_config in strategy_config["indicators"].values():
            timeframes.add(indicator_config["timeframe"])

        # Set the mock connector on the data manager for test context
        data_manager._mock_connector = mock_instance

        # Fetch data for each timeframe
        data_dict = {}
        for timeframe in timeframes:
            data = data_manager.fetch_real_data(
                market=strategy_config["market"],
                exchange=strategy_config["exchange"],
                timeframe=timeframe,
                days=7
            )
            data_dict[timeframe] = data

        # Verify all timeframes were fetched
        assert len(data_dict) == 2
        assert "1h" in data_dict
        assert "4h" in data_dict
        assert all(data is not None for data in data_dict.values())

        # Verify connector was called for each timeframe
        assert mock_instance.get_historical_candles.call_count == 2

    def test_exchange_validation_integration(self, config_manager, data_manager):
        """Test that data manager validates exchanges from config manager."""
        strategy_config = config_manager.get_strategy_config("eth_momentum_strategy")
        exchange_config = config_manager.get_exchange_config(strategy_config["exchange"])

        # Verify exchange exists in config
        assert exchange_config is not None
        assert exchange_config["name"] == "Hyperliquid"

        # Verify market is supported by exchange
        assert strategy_config["market"] in exchange_config["markets"]

        # Verify timeframe is supported by exchange
        assert strategy_config["timeframe"] in exchange_config["timeframes"]

    def test_cache_key_generation_from_config(self, config_manager, data_manager):
        """Test generating cache keys from strategy configuration."""
        strategy_config = config_manager.get_strategy_config("eth_momentum_strategy")

        # Generate cache keys for strategy data requirements
        cache_keys = []
        timeframes = set([strategy_config["timeframe"]])
        for indicator_config in strategy_config["indicators"].values():
            timeframes.add(indicator_config["timeframe"])

        for timeframe in timeframes:
            cache_key = f"{strategy_config['market']}_{strategy_config['exchange']}_{timeframe}_7days"
            cache_keys.append(cache_key)

        # Verify cache keys are unique and properly formatted
        assert len(cache_keys) == len(set(cache_keys))  # All unique
        assert "ETH-USD_hyperliquid_1h_7days" in cache_keys
        assert "ETH-USD_hyperliquid_4h_7days" in cache_keys

    def test_global_settings_integration(self, config_manager, data_manager, temp_cache_dir):
        """Test integration of global settings between config and data managers."""
        # Get global cache settings from config
        global_settings = config_manager.config.get("global_settings", {})
        cache_settings = global_settings.get("cache_settings", {})

        # Create data manager with settings from config
        cache_enabled = cache_settings.get("enabled", True)
        cache_timeout = cache_settings.get("timeout_hours", 1) * 3600  # Convert to seconds

        integrated_data_manager = DataManager(
            cache_dir=temp_cache_dir,
            cache_enabled=cache_enabled,
            cache_timeout=cache_timeout
        )

        # Verify settings were applied
        assert integrated_data_manager.cache_enabled == cache_enabled
        assert integrated_data_manager.cache_timeout == cache_timeout

    def test_strategy_comparison_data_coordination(self, config_manager, data_manager, sample_market_data):
        """Test coordinating data fetching for multiple strategies comparison."""
        # Get multiple strategies
        strategies = ["eth_momentum_strategy", "btc_scalping_strategy"]
        strategy_configs = {}
        data_requirements = {}

        for strategy_name in strategies:
            config = config_manager.get_strategy_config(strategy_name)
            strategy_configs[strategy_name] = config

            # Extract data requirements
            data_requirements[strategy_name] = {
                "market": config["market"],
                "exchange": config["exchange"],
                "timeframes": set([config["timeframe"]])
            }

            # Add indicator timeframes
            for indicator_config in config["indicators"].values():
                data_requirements[strategy_name]["timeframes"].add(indicator_config["timeframe"])

        # Verify each strategy has unique data requirements
        assert data_requirements["eth_momentum_strategy"]["market"] == "ETH-USD"
        assert data_requirements["btc_scalping_strategy"]["market"] == "BTC-USD"
        assert data_requirements["eth_momentum_strategy"]["exchange"] == "hyperliquid"
        assert data_requirements["btc_scalping_strategy"]["exchange"] == "coinbase"

    def test_error_handling_invalid_strategy_config(self, config_manager, data_manager):
        """Test error handling when strategy config has invalid data requirements."""
        # Try to get data for non-existent strategy
        strategy_config = config_manager.get_strategy_config("nonexistent_strategy")
        assert strategy_config is None

        # Try to get data with invalid exchange
        with pytest.raises(DataFetchError):
            data_manager.fetch_real_data(
                market="ETH-USD",
                exchange="invalid_exchange",
                timeframe="1h",
                days=7
            )

    def test_configuration_validation_with_data_constraints(self, config_manager, data_manager):
        """Test validating strategy configuration against data manager constraints."""
        strategy_config = config_manager.get_strategy_config("eth_momentum_strategy")

        # Validate exchange support
        exchange_config = config_manager.get_exchange_config(strategy_config["exchange"])
        assert strategy_config["market"] in exchange_config["markets"]
        assert strategy_config["timeframe"] in exchange_config["timeframes"]

        # Validate indicator timeframes
        for indicator_name, indicator_config in strategy_config["indicators"].items():
            assert indicator_config["timeframe"] in exchange_config["timeframes"]

    def test_caching_coordination_across_strategies(self, config_manager, data_manager, sample_market_data):
        """Test that caching works efficiently across multiple strategies."""
        # Cache some data
        cache_key = "ETH-USD_hyperliquid_1h_7days"
        data_manager.cache_data(cache_key, sample_market_data)

        # Verify cached data can be retrieved
        cached_data = data_manager.get_cached_data(cache_key)
        assert cached_data is not None

        # Verify cache key format matches strategy requirements
        strategy_config = config_manager.get_strategy_config("eth_momentum_strategy")
        expected_cache_key = f"{strategy_config['market']}_{strategy_config['exchange']}_{strategy_config['timeframe']}_7days"
        assert cache_key == expected_cache_key

    def test_position_sizing_data_requirements(self, config_manager, data_manager):
        """Test that position sizing requirements are properly handled."""
        strategy_config = config_manager.get_strategy_config("eth_momentum_strategy")

        # Extract position sizing requirements
        position_sizing = strategy_config["position_sizing"]

        # For fixed USD position sizing, we need current price data
        if position_sizing["method"] == "fixed_usd":
            # This would require current market price for calculation
            assert "amount" in position_sizing
            assert isinstance(position_sizing["amount"], (int, float))

        # Verify percentage-based strategy
        btc_strategy = config_manager.get_strategy_config("btc_scalping_strategy")
        btc_position_sizing = btc_strategy["position_sizing"]
        if btc_position_sizing["method"] == "percentage":
            assert "percentage" in btc_position_sizing
            assert isinstance(btc_position_sizing["percentage"], (int, float))

    def test_data_quality_validation_integration(self, config_manager, data_manager):
        """Test data quality validation based on strategy requirements."""
        strategy_config = config_manager.get_strategy_config("eth_momentum_strategy")

        # Create minimal data that might not meet strategy requirements
        minimal_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1H'),
            'open': range(10),
            'high': range(1, 11),
            'low': range(10),
            'close': range(10),
            'volume': range(10)
        })

        # Validate data quality (should fail for insufficient data)
        is_valid, errors = data_manager.validate_data_quality(minimal_data, min_rows=168)  # Need 7 days
        assert not is_valid
        assert len(errors) > 0

    def test_multi_exchange_strategy_support(self, config_manager, data_manager):
        """Test support for strategies across different exchanges."""
        # Get strategies from different exchanges
        eth_strategy = config_manager.get_strategy_config("eth_momentum_strategy")
        btc_strategy = config_manager.get_strategy_config("btc_scalping_strategy")

        # Verify different exchanges
        assert eth_strategy["exchange"] != btc_strategy["exchange"]
        assert eth_strategy["exchange"] == "hyperliquid"
        assert btc_strategy["exchange"] == "coinbase"

        # Verify exchange configs exist for both
        hyperliquid_config = config_manager.get_exchange_config("hyperliquid")
        coinbase_config = config_manager.get_exchange_config("coinbase")

        assert hyperliquid_config is not None
        assert coinbase_config is not None

    def test_timeframe_hierarchy_validation(self, config_manager, data_manager):
        """Test validation of timeframe hierarchy in multi-timeframe strategies."""
        strategy_config = config_manager.get_strategy_config("eth_momentum_strategy")

        # Extract all timeframes
        strategy_timeframe = strategy_config["timeframe"]
        indicator_timeframes = [
            indicator["timeframe"]
            for indicator in strategy_config["indicators"].values()
        ]

        # Convert to minutes for comparison
        strategy_minutes = data_manager.get_timeframe_minutes(strategy_timeframe)
        indicator_minutes = [
            data_manager.get_timeframe_minutes(tf)
            for tf in indicator_timeframes
        ]

        # Verify timeframes are valid
        assert strategy_minutes > 0
        assert all(minutes > 0 for minutes in indicator_minutes)

        # Verify we can handle the timeframe hierarchy
        all_timeframes = [strategy_timeframe] + indicator_timeframes
        all_minutes = [strategy_minutes] + indicator_minutes

        # Should be able to sort by timeframe size
        sorted_timeframes = sorted(zip(all_minutes, all_timeframes))
        assert len(sorted_timeframes) > 0
