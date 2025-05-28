"""
Unit tests for strategy configuration classes.

This module tests:
- StrategyConfig validation and creation
- StrategyConfigLoader functionality
- Strategy-indicator relationship validation
- Position sizing configuration validation
- Error handling for invalid configurations
"""

from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from app.core.strategy_config import StrategyConfig, StrategyConfigLoader


class TestStrategyConfig:
    """Test cases for StrategyConfig class."""

    def test_valid_strategy_config_creation(self):
        """Test creation of valid strategy configuration."""
        # ARRANGE
        valid_data = {
            "name": "test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "indicators": ["rsi", "macd"],
            "enabled": True,
            "timeframe": "1h"
        }

        # ACT
        strategy = StrategyConfig.from_dict(valid_data)

        # ASSERT
        assert strategy.name == "test_strategy"
        assert strategy.market == "ETH-USD"
        assert strategy.exchange == "hyperliquid"
        assert strategy.indicators == ["rsi", "macd"]
        assert strategy.enabled is True
        assert strategy.timeframe == "1h"

    def test_strategy_config_with_all_parameters(self):
        """Test strategy config creation with all optional parameters."""
        # ARRANGE
        complete_data = {
            "name": "complete_strategy",
            "market": "BTC-USD",
            "exchange": "coinbase",
            "indicators": ["bollinger_bands"],
            "enabled": False,
            "timeframe": "4h",
            "main_leverage": 2.0,
            "hedge_leverage": 1.5,
            "hedge_ratio": 0.5,
            "stop_loss_pct": 3.0,
            "take_profit_pct": 15.0,
            "max_position_size": 0.2,
            "max_position_size_usd": 2000.0,
            "risk_per_trade_pct": 0.03,
            "position_sizing": {
                "method": "equity_percentage",
                "equity_percentage": 0.1
            }
        }

        # ACT
        strategy = StrategyConfig.from_dict(complete_data)

        # ASSERT
        assert strategy.name == "complete_strategy"
        assert strategy.market == "BTC-USD"
        assert strategy.exchange == "coinbase"
        assert strategy.indicators == ["bollinger_bands"]
        assert strategy.enabled is False
        assert strategy.timeframe == "4h"
        assert strategy.main_leverage == 2.0
        assert strategy.hedge_leverage == 1.5
        assert strategy.hedge_ratio == 0.5
        assert strategy.stop_loss_pct == 3.0
        assert strategy.take_profit_pct == 15.0
        assert strategy.max_position_size == 0.2
        assert strategy.max_position_size_usd == 2000.0
        assert strategy.risk_per_trade_pct == 0.03
        assert strategy.position_sizing["method"] == "equity_percentage"

    def test_strategy_config_defaults(self):
        """Test that default values are applied correctly."""
        # ARRANGE
        minimal_data = {
            "name": "minimal_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "indicators": ["rsi"]
        }

        # ACT
        strategy = StrategyConfig.from_dict(minimal_data)

        # ASSERT
        assert strategy.enabled is True  # default
        assert strategy.timeframe == "1h"  # default
        assert strategy.main_leverage == 1.0  # default
        assert strategy.hedge_leverage == 1.0  # default
        assert strategy.hedge_ratio == 0.0  # default
        assert strategy.stop_loss_pct == 5.0  # default
        assert strategy.take_profit_pct == 10.0  # default
        assert strategy.max_position_size == 0.1  # default
        assert strategy.max_position_size_usd == 1000.0  # default
        assert strategy.risk_per_trade_pct == 0.02  # default
        assert strategy.position_sizing is None  # default

    @pytest.mark.parametrize("invalid_market", [
        "",  # empty string
        "ETH",  # no separator
        "ETHUSD",  # no separator
        "ETH_USD",  # wrong separator
        "ETH:USD",  # wrong separator
        None,  # None value
    ])
    def test_invalid_market_format_raises_error(self, invalid_market):
        """Test that invalid market formats raise ValueError."""
        # ARRANGE
        data = {
            "name": "test_strategy",
            "market": invalid_market,
            "exchange": "hyperliquid",
            "indicators": ["rsi"]
        }

        # ACT & ASSERT
        with pytest.raises(ValueError, match="invalid market format"):
            StrategyConfig.from_dict(data)

    @pytest.mark.parametrize("invalid_exchange", [
        "",  # empty string
        None,  # None value
    ])
    def test_missing_exchange_raises_error(self, invalid_exchange):
        """Test that missing exchange raises ValueError."""
        # ARRANGE
        data = {
            "name": "test_strategy",
            "market": "ETH-USD",
            "exchange": invalid_exchange,
            "indicators": ["rsi"]
        }

        # ACT & ASSERT
        with pytest.raises(ValueError, match="missing required 'exchange' field"):
            StrategyConfig.from_dict(data)

    @pytest.mark.parametrize("invalid_indicators", [
        [],  # empty list
        None,  # None value
    ])
    def test_empty_indicators_list_raises_error(self, invalid_indicators):
        """Test that empty indicators list raises ValueError."""
        # ARRANGE
        data = {
            "name": "test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "indicators": invalid_indicators
        }

        # ACT & ASSERT
        with pytest.raises(ValueError, match="must specify at least one indicator"):
            StrategyConfig.from_dict(data)

    @pytest.mark.parametrize("valid_timeframe", [
        "1m", "5m", "15m", "30m",  # valid minutes
        "1h", "4h", "12h",  # valid hours
        "1d",  # valid days
        "1w",  # valid weeks
    ])
    def test_valid_timeframe_formats(self, valid_timeframe):
        """Test that valid timeframe formats are accepted."""
        # ARRANGE
        data = {
            "name": "test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "indicators": ["rsi"],
            "timeframe": valid_timeframe
        }

        # ACT
        strategy = StrategyConfig.from_dict(data)

        # ASSERT
        assert strategy.timeframe == valid_timeframe

    @pytest.mark.parametrize("invalid_timeframe", [
        "2m",  # invalid minute
        "1h30m",  # complex format
        "2h",  # invalid hour
        "2d",  # invalid day
        "1y",  # invalid unit
        "hour",  # non-numeric
        "",  # empty
        "1",  # no unit
    ])
    def test_invalid_timeframe_formats_raise_error(self, invalid_timeframe):
        """Test that invalid timeframe formats raise ValueError."""
        # ARRANGE
        data = {
            "name": "test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "indicators": ["rsi"],
            "timeframe": invalid_timeframe
        }

        # ACT & ASSERT
        with pytest.raises(ValueError, match="invalid timeframe"):
            StrategyConfig.from_dict(data)

    @pytest.mark.parametrize("invalid_leverage", [0, -1, -0.5])
    def test_invalid_leverage_values_raise_error(self, invalid_leverage):
        """Test that invalid leverage values raise ValueError."""
        # ARRANGE
        data = {
            "name": "test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "indicators": ["rsi"],
            "main_leverage": invalid_leverage
        }

        # ACT & ASSERT
        with pytest.raises(ValueError, match="main_leverage must be positive"):
            StrategyConfig.from_dict(data)

    @pytest.mark.parametrize("invalid_ratio", [-0.1, 1.1, 2.0])
    def test_invalid_hedge_ratio_raises_error(self, invalid_ratio):
        """Test that hedge ratio outside 0-1 range raises ValueError."""
        # ARRANGE
        data = {
            "name": "test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "indicators": ["rsi"],
            "hedge_ratio": invalid_ratio
        }

        # ACT & ASSERT
        with pytest.raises(ValueError, match="hedge_ratio must be between 0 and 1"):
            StrategyConfig.from_dict(data)

    @pytest.mark.parametrize("invalid_risk_pct", [0, -0.1, 1.1, 2.0])
    def test_invalid_risk_per_trade_pct_raises_error(self, invalid_risk_pct):
        """Test that invalid risk per trade percentage raises ValueError."""
        # ARRANGE
        data = {
            "name": "test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "indicators": ["rsi"],
            "risk_per_trade_pct": invalid_risk_pct
        }

        # ACT & ASSERT
        with pytest.raises(ValueError, match="risk_per_trade_pct must be between 0 and 1"):
            StrategyConfig.from_dict(data)

    def test_to_dict_conversion(self):
        """Test that strategy config can be converted back to dictionary."""
        # ARRANGE
        original_data = {
            "name": "test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "indicators": ["rsi", "macd"],
            "enabled": True,
            "timeframe": "4h",
            "main_leverage": 2.0,
            "position_sizing": {"method": "fixed_usd", "fixed_usd_amount": 500.0}
        }

        # ACT
        strategy = StrategyConfig.from_dict(original_data)
        result_dict = strategy.to_dict()

        # ASSERT
        assert result_dict["name"] == original_data["name"]
        assert result_dict["market"] == original_data["market"]
        assert result_dict["exchange"] == original_data["exchange"]
        assert result_dict["indicators"] == original_data["indicators"]
        assert result_dict["enabled"] == original_data["enabled"]
        assert result_dict["timeframe"] == original_data["timeframe"]
        assert result_dict["main_leverage"] == original_data["main_leverage"]
        assert result_dict["position_sizing"] == original_data["position_sizing"]


class TestStrategyConfigLoader:
    """Test cases for StrategyConfigLoader class."""

    def test_load_single_strategy_success(self):
        """Test loading a single valid strategy configuration."""
        # ARRANGE
        strategies_data = [
            {
                "name": "test_strategy",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "indicators": ["rsi"]
            }
        ]

        # ACT
        with patch('app.core.strategy_config.logger'):
            strategies = StrategyConfigLoader.load_strategies(strategies_data)

        # ASSERT
        assert len(strategies) == 1
        assert strategies[0].name == "test_strategy"
        assert strategies[0].market == "ETH-USD"
        assert strategies[0].exchange == "hyperliquid"
        assert strategies[0].indicators == ["rsi"]

    def test_load_multiple_strategies_success(self):
        """Test loading multiple valid strategy configurations."""
        # ARRANGE
        strategies_data = [
            {
                "name": "eth_strategy",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "indicators": ["rsi"]
            },
            {
                "name": "btc_strategy",
                "market": "BTC-USD",
                "exchange": "coinbase",
                "indicators": ["macd", "bollinger_bands"]
            }
        ]

        # ACT
        with patch('app.core.strategy_config.logger'):
            strategies = StrategyConfigLoader.load_strategies(strategies_data)

        # ASSERT
        assert len(strategies) == 2
        assert strategies[0].name == "eth_strategy"
        assert strategies[1].name == "btc_strategy"
        assert strategies[0].exchange == "hyperliquid"
        assert strategies[1].exchange == "coinbase"

    def test_duplicate_strategy_names_raise_error(self):
        """Test that duplicate strategy names raise ValueError."""
        # ARRANGE
        strategies_data = [
            {
                "name": "duplicate_name",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "indicators": ["rsi"]
            },
            {
                "name": "duplicate_name",
                "market": "BTC-USD",
                "exchange": "coinbase",
                "indicators": ["macd"]
            }
        ]

        # ACT & ASSERT
        with pytest.raises(ValueError, match="Duplicate strategy name"):
            StrategyConfigLoader.load_strategies(strategies_data)

    def test_invalid_strategy_in_list_raises_error(self):
        """Test that invalid strategy in list raises ValueError with index."""
        # ARRANGE
        strategies_data = [
            {
                "name": "valid_strategy",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "indicators": ["rsi"]
            },
            {
                "name": "invalid_strategy",
                "market": "INVALID",  # invalid market format
                "exchange": "hyperliquid",
                "indicators": ["rsi"]
            }
        ]

        # ACT & ASSERT
        with pytest.raises(ValueError, match="Strategy configuration error at index 1"):
            StrategyConfigLoader.load_strategies(strategies_data)

    def test_empty_strategies_list(self):
        """Test loading empty strategies list."""
        # ARRANGE
        strategies_data = []

        # ACT
        with patch('app.core.strategy_config.logger'):
            strategies = StrategyConfigLoader.load_strategies(strategies_data)

        # ASSERT
        assert len(strategies) == 0

    def test_validate_indicators_success(self):
        """Test successful validation of strategy-indicator relationships."""
        # ARRANGE
        strategies = [
            StrategyConfig(
                name="test_strategy",
                market="ETH-USD",
                exchange="hyperliquid",
                indicators=["rsi", "macd"]
            )
        ]
        indicators = {
            "rsi": {"type": "rsi", "period": 14},
            "macd": {"type": "macd", "fast_period": 12}
        }

        # ACT & ASSERT (should not raise)
        with patch('app.core.strategy_config.logger'):
            StrategyConfigLoader.validate_indicators(strategies, indicators)

    def test_validate_indicators_missing_indicator_raises_error(self):
        """Test that missing indicator raises ValueError."""
        # ARRANGE
        strategies = [
            StrategyConfig(
                name="test_strategy",
                market="ETH-USD",
                exchange="hyperliquid",
                indicators=["rsi", "missing_indicator"]
            )
        ]
        indicators = {
            "rsi": {"type": "rsi", "period": 14}
        }

        # ACT & ASSERT
        with pytest.raises(ValueError, match="references unknown indicator: 'missing_indicator'"):
            StrategyConfigLoader.validate_indicators(strategies, indicators)

    def test_validate_indicators_reports_unused_indicators(self):
        """Test that unused indicators are reported in logs."""
        # ARRANGE
        strategies = [
            StrategyConfig(
                name="test_strategy",
                market="ETH-USD",
                exchange="hyperliquid",
                indicators=["rsi"]
            )
        ]
        indicators = {
            "rsi": {"type": "rsi", "period": 14},
            "unused_macd": {"type": "macd", "fast_period": 12},
            "unused_bollinger": {"type": "bollinger_bands", "period": 20}
        }

        # ACT & ASSERT (should not raise but log warnings)
        with patch('app.core.strategy_config.logger') as mock_logger:
            StrategyConfigLoader.validate_indicators(strategies, indicators)
            # Check that unused indicators are logged
            mock_logger.warning.assert_called()

    def test_validate_position_sizing_configs_success(self):
        """Test successful validation of position sizing configurations."""
        # ARRANGE
        strategies = [
            StrategyConfig(
                name="test_strategy",
                market="ETH-USD",
                exchange="hyperliquid",
                indicators=["rsi"],
                position_sizing={
                    "method": "fixed_usd",
                    "fixed_usd_amount": 500.0
                }
            )
        ]
        global_position_sizing = {
            "method": "equity_percentage",
            "equity_percentage": 0.05
        }

        # ACT & ASSERT (should not raise)
        with patch('app.core.strategy_config.logger'):
            StrategyConfigLoader.validate_position_sizing_configs(strategies, global_position_sizing)

    def test_validate_position_sizing_missing_method_raises_error(self):
        """Test that missing position sizing method raises ValueError."""
        # ARRANGE
        strategies = [
            StrategyConfig(
                name="test_strategy",
                market="ETH-USD",
                exchange="hyperliquid",
                indicators=["rsi"],
                position_sizing={
                    "fixed_usd_amount": 500.0  # missing method
                }
            )
        ]
        global_position_sizing = {}

        # ACT & ASSERT
        with pytest.raises(ValueError, match="position sizing config missing 'method' field"):
            StrategyConfigLoader.validate_position_sizing_configs(strategies, global_position_sizing)

    def test_validate_position_sizing_invalid_method_raises_error(self):
        """Test that invalid position sizing method raises ValueError."""
        # ARRANGE
        strategies = [
            StrategyConfig(
                name="test_strategy",
                market="ETH-USD",
                exchange="hyperliquid",
                indicators=["rsi"],
                position_sizing={
                    "method": "invalid_method"
                }
            )
        ]
        global_position_sizing = {}

        # ACT & ASSERT
        with pytest.raises(ValueError, match="invalid position sizing method 'invalid_method'"):
            StrategyConfigLoader.validate_position_sizing_configs(strategies, global_position_sizing)

    @pytest.mark.parametrize("method,config,should_raise,error_match", [
        # Valid configurations
        ("fixed_usd", {"fixed_usd_amount": 500.0}, False, None),
        ("equity_percentage", {"equity_percentage": 0.1}, False, None),
        ("risk_based", {"risk_per_trade_pct": 0.02}, False, None),
        ("fixed_units", {"fixed_units": 0.5}, False, None),
        ("kelly", {"kelly_win_rate": 0.6, "kelly_avg_win": 0.03, "kelly_avg_loss": 0.02}, False, None),

        # Invalid configurations
        ("fixed_usd", {"fixed_usd_amount": 0}, True, "requires positive 'fixed_usd_amount'"),
        ("fixed_usd", {"fixed_usd_amount": -100}, True, "requires positive 'fixed_usd_amount'"),
        ("equity_percentage", {"equity_percentage": 0}, True, "'equity_percentage' between 0 and 1"),
        ("equity_percentage", {"equity_percentage": 1.5}, True, "'equity_percentage' between 0 and 1"),
        ("risk_based", {"risk_per_trade_pct": 0}, True, "'risk_per_trade_pct' between 0 and 1"),
        ("risk_based", {"risk_per_trade_pct": 1.5}, True, "'risk_per_trade_pct' between 0 and 1"),
        ("fixed_units", {"fixed_units": 0}, True, "requires positive 'fixed_units'"),
        ("fixed_units", {"fixed_units": -1}, True, "requires positive 'fixed_units'"),
        ("kelly", {"kelly_win_rate": 0, "kelly_avg_win": 0.03, "kelly_avg_loss": 0.02}, True, "requires positive 'kelly_win_rate'"),
        ("kelly", {"kelly_win_rate": 0.6, "kelly_avg_win": 0, "kelly_avg_loss": 0.02}, True, "requires positive 'kelly_avg_win'"),
        ("kelly", {"kelly_win_rate": 0.6, "kelly_avg_win": 0.03, "kelly_avg_loss": 0}, True, "requires positive 'kelly_avg_loss'"),
    ])
    def test_validate_position_sizing_method_specific_parameters(
        self, method, config, should_raise, error_match
    ):
        """Test validation of method-specific position sizing parameters."""
        # ARRANGE
        position_sizing_config = {"method": method, **config}
        strategies = [
            StrategyConfig(
                name="test_strategy",
                market="ETH-USD",
                exchange="hyperliquid",
                indicators=["rsi"],
                position_sizing=position_sizing_config
            )
        ]
        global_position_sizing = {}

        # ACT & ASSERT
        if should_raise:
            with pytest.raises(ValueError, match=error_match):
                StrategyConfigLoader.validate_position_sizing_configs(strategies, global_position_sizing)
        else:
            with patch('app.core.strategy_config.logger'):
                StrategyConfigLoader.validate_position_sizing_configs(strategies, global_position_sizing)

    def test_validate_position_sizing_strategy_without_custom_config(self):
        """Test validation passes for strategies without custom position sizing."""
        # ARRANGE
        strategies = [
            StrategyConfig(
                name="test_strategy",
                market="ETH-USD",
                exchange="hyperliquid",
                indicators=["rsi"]
                # No position_sizing config - should use global
            )
        ]
        global_position_sizing = {
            "method": "equity_percentage",
            "equity_percentage": 0.05
        }

        # ACT & ASSERT (should not raise)
        with patch('app.core.strategy_config.logger'):
            StrategyConfigLoader.validate_position_sizing_configs(strategies, global_position_sizing)

    def test_validate_multiple_strategies_mixed_position_sizing(self):
        """Test validation of multiple strategies with mixed position sizing configs."""
        # ARRANGE
        strategies = [
            StrategyConfig(
                name="custom_strategy",
                market="ETH-USD",
                exchange="hyperliquid",
                indicators=["rsi"],
                position_sizing={
                    "method": "fixed_usd",
                    "fixed_usd_amount": 500.0
                }
            ),
            StrategyConfig(
                name="global_strategy",
                market="BTC-USD",
                exchange="coinbase",
                indicators=["macd"]
                # Uses global position sizing
            )
        ]
        global_position_sizing = {
            "method": "equity_percentage",
            "equity_percentage": 0.05
        }

        # ACT & ASSERT (should not raise)
        with patch('app.core.strategy_config.logger'):
            StrategyConfigLoader.validate_position_sizing_configs(strategies, global_position_sizing)
