"""
Unit tests for CLI ConfigManager module.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from core.config_manager import ConfigManager


class TestConfigManager:
    """Test suite for ConfigManager functionality."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        config_data = {
            "exchange_configs": {
                "hyperliquid": {
                    "name": "Hyperliquid",
                    "markets": ["ETH-USD", "BTC-USD"]
                }
            },
            "strategy_configs": {
                "test_strategy": {
                    "name": "Test Strategy",
                    "market": "ETH-USD",
                    "exchange": "hyperliquid",
                    "timeframe": "1h",
                    "indicators": {
                        "rsi": {
                            "class": "RSIIndicator",
                            "timeframe": "1h",
                            "window": 14
                        }
                    },
                    "position_sizing": {
                        "method": "fixed_usd",
                        "amount": 100
                    },
                    "enabled": True
                }
            },
            "global_settings": {
                "default_position_sizing": {
                    "method": "fixed_usd",
                    "amount": 100
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
    def config_manager(self, temp_config_file):
        """Create a ConfigManager instance with test config."""
        return ConfigManager(config_path=temp_config_file)

    def test_initialization_with_valid_config(self, temp_config_file):
        """Test ConfigManager initialization with valid config file."""
        manager = ConfigManager(config_path=temp_config_file)
        assert manager.config_path == temp_config_file
        assert manager.config is not None
        assert "strategy_configs" in manager.config

    def test_initialization_with_invalid_config(self):
        """Test ConfigManager initialization with invalid config file."""
        with pytest.raises(FileNotFoundError):
            ConfigManager(config_path="/nonexistent/config.json")

    def test_load_config_success(self, config_manager):
        """Test successful config loading."""
        config = config_manager.load_config()
        assert "strategy_configs" in config
        assert "exchange_configs" in config
        assert "test_strategy" in config["strategy_configs"]

    def test_load_config_with_environment_variables(self, temp_config_file):
        """Test config loading with environment variable expansion."""
        # Create config with environment variable
        config_with_env = {
            "strategy_configs": {
                "test_strategy": {
                    "position_sizing": {
                        "amount": "${TEST_AMOUNT:100}"
                    }
                }
            }
        }

        with open(temp_config_file, 'w') as f:
            json.dump(config_with_env, f)

        # Test without environment variable (should use default)
        manager = ConfigManager(config_path=temp_config_file)
        config = manager.load_config()
        assert config["strategy_configs"]["test_strategy"]["position_sizing"]["amount"] == "100"

        # Test with environment variable
        with patch.dict(os.environ, {'TEST_AMOUNT': '200'}):
            manager = ConfigManager(config_path=temp_config_file)
            config = manager.load_config()
            assert config["strategy_configs"]["test_strategy"]["position_sizing"]["amount"] == "200"

    def test_validate_config_success(self, config_manager):
        """Test successful config validation."""
        config = config_manager.load_config()
        is_valid, errors = config_manager.validate_config(config)
        assert is_valid
        assert len(errors) == 0

    def test_validate_config_missing_required_fields(self, temp_config_file):
        """Test config validation with missing required fields."""
        invalid_config = {
            "strategy_configs": {
                "invalid_strategy": {
                    "name": "Invalid Strategy"
                    # Missing required fields: market, exchange, timeframe, indicators
                }
            }
        }

        with open(temp_config_file, 'w') as f:
            json.dump(invalid_config, f)

        manager = ConfigManager(config_path=temp_config_file)
        config = manager.load_config()
        is_valid, errors = manager.validate_config(config)
        assert not is_valid
        assert len(errors) > 0

    def test_get_strategy_config_existing(self, config_manager):
        """Test getting existing strategy configuration."""
        strategy_config = config_manager.get_strategy_config("test_strategy")
        assert strategy_config is not None
        assert strategy_config["name"] == "Test Strategy"
        assert strategy_config["market"] == "ETH-USD"

    def test_get_strategy_config_nonexistent(self, config_manager):
        """Test getting non-existent strategy configuration."""
        strategy_config = config_manager.get_strategy_config("nonexistent_strategy")
        assert strategy_config is None

    def test_list_strategies_all(self, config_manager):
        """Test listing all strategies."""
        strategies = config_manager.list_strategies()
        assert len(strategies) == 1
        assert any(s.get("name") == "Test Strategy" for s in strategies)

    def test_list_strategies_enabled_only(self, config_manager):
        """Test listing enabled strategies only."""
        strategies = config_manager.list_strategies(filter_enabled=True)
        assert len(strategies) == 1
        assert any(s.get("name") == "Test Strategy" for s in strategies)

    def test_list_strategies_with_filters(self, config_manager):
        """Test listing strategies with filters."""
        # Filter by market
        strategies = config_manager.list_strategies(filter_market="ETH-USD")
        assert len(strategies) == 1
        assert any(s.get("name") == "Test Strategy" for s in strategies)

        # Filter by exchange
        strategies = config_manager.list_strategies(filter_exchange="hyperliquid")
        assert len(strategies) == 1
        assert any(s.get("name") == "Test Strategy" for s in strategies)

        # Filter by non-matching criteria
        strategies = config_manager.list_strategies(filter_market="BTC-USD")
        assert len(strategies) == 0

    def test_cache_functionality(self, config_manager):
        """Test configuration caching functionality."""
        # First load should read from file
        config1 = config_manager.load_config()

        # Second load should use cache
        config2 = config_manager.load_config()

        # Configs should be equal (cached)
        assert config1 == config2

    def test_reload_config(self, config_manager, temp_config_file):
        """Test configuration reloading."""
        # Load initial config
        config1 = config_manager.load_config()

        # Modify the config file
        new_config = {
            "strategy_configs": {
                "new_strategy": {
                    "name": "New Strategy",
                    "market": "BTC-USD",
                    "exchange": "hyperliquid",
                    "timeframe": "4h",
                    "indicators": {},
                    "enabled": True
                }
            }
        }

        with open(temp_config_file, 'w') as f:
            json.dump(new_config, f)

        # Reload config
        config2 = config_manager.reload_config()

        # Configs should be different
        assert config1 is not config2
        assert "new_strategy" in config2["strategy_configs"]
        assert "test_strategy" not in config2["strategy_configs"]

    def test_error_handling_malformed_json(self, temp_config_file):
        """Test handling of malformed JSON config file."""
        # Write invalid JSON
        with open(temp_config_file, 'w') as f:
            f.write("{ invalid json }")

        from core.config_manager import ConfigurationError
        with pytest.raises(ConfigurationError):
            ConfigManager(config_path=temp_config_file)

    def test_error_handling_missing_file(self):
        """Test handling of missing config file."""
        with pytest.raises(FileNotFoundError):
            ConfigManager(config_path="/nonexistent/path/config.json")

    def test_get_exchange_config(self, config_manager):
        """Test getting exchange configuration."""
        exchange_config = config_manager.get_exchange_config("hyperliquid")
        assert exchange_config is not None
        assert exchange_config["name"] == "Hyperliquid"
        assert "ETH-USD" in exchange_config["markets"]

    def test_get_exchange_config_nonexistent(self, config_manager):
        """Test getting non-existent exchange configuration."""
        exchange_config = config_manager.get_exchange_config("nonexistent")
        assert exchange_config is None

    def test_performance_with_large_config(self):
        """Test performance with large configuration files."""
        # Create a large config with many strategies
        large_config = {
            "strategy_configs": {},
            "exchange_configs": {
                "test_exchange": {
                    "name": "Test Exchange",
                    "markets": ["ETH-USD"]
                }
            }
        }

        # Add 100 strategies
        for i in range(100):
            large_config["strategy_configs"][f"strategy_{i}"] = {
                "name": f"Strategy {i}",
                "market": "ETH-USD",
                "exchange": "test_exchange",
                "timeframe": "1h",
                "indicators": {},
                "enabled": True
            }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(large_config, f)
            temp_file = f.name

        try:
            # Test that loading is still fast
            import time
            start_time = time.time()
            manager = ConfigManager(config_path=temp_file)
            strategies = manager.list_strategies()
            end_time = time.time()

            assert len(strategies) == 100
            assert (end_time - start_time) < 1.0  # Should complete in under 1 second
        finally:
            os.unlink(temp_file)
