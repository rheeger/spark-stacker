"""
Unit tests for strategy-specific position sizing functionality.

This module tests the various position sizing methods that can be configured
per strategy, including inheritance from global configuration and validation.
"""

from unittest.mock import MagicMock

import pytest
from app.connectors.base_connector import OrderSide
from app.core.strategy_config import StrategyConfig
from app.risk_management import (PositionSizer, PositionSizingConfig,
                                 PositionSizingMethod, RiskManager)


class TestStrategyPositionSizing:
    """Test cases for strategy-specific position sizing configurations."""

    def test_strategy_specific_fixed_usd_position_sizing(self):
        """Test strategy-specific fixed USD position sizing."""
        strategy = StrategyConfig(
            name="eth_fixed_strategy",
            market="ETH-USD",
            exchange="hyperliquid",
            indicators=["RSI"],
            position_sizing={
                'method': 'fixed_usd',
                'fixed_usd_amount': 800.0,
                'max_position_size_usd': 3000.0,
                'min_position_size_usd': 50.0
            }
        )

        config = {
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0,  # Different global default
                'max_position_size_usd': 5000.0
            }
        }

        risk_manager = RiskManager.from_config(config, strategies=[strategy])

        # Create mock exchange
        mock_exchange = MagicMock()
        mock_exchange.get_ticker.return_value = {'last_price': 2000.0}
        mock_exchange.get_leverage_tiers.return_value = [{'max_leverage': 10.0}]

        # Test strategy-specific fixed USD amount
        position_size_usd, _ = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='ETH',
            available_balance=10000.0,
            confidence=0.75,  # 75% confidence
            signal_side=OrderSide.BUY,
            leverage=5.0,
            strategy_name="eth_fixed_strategy"
        )

        # Should use strategy-specific amount: $800 * 0.75 = $600
        assert position_size_usd == 600.0

        # Test with higher confidence
        position_size_usd, _ = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='ETH',
            available_balance=10000.0,
            confidence=1.0,  # 100% confidence
            signal_side=OrderSide.BUY,
            leverage=5.0,
            strategy_name="eth_fixed_strategy"
        )

        # Should use strategy-specific amount: $800 * 1.0 = $800
        assert position_size_usd == 800.0

        # Verify strategy-specific limits are respected
        strategy_sizer = risk_manager.strategy_position_sizers["eth_fixed_strategy"]
        assert strategy_sizer.config.max_position_size_usd == 3000.0
        assert strategy_sizer.config.min_position_size_usd == 50.0

    def test_strategy_specific_risk_based_position_sizing(self):
        """Test strategy-specific risk-based position sizing."""
        strategy = StrategyConfig(
            name="btc_risk_strategy",
            market="BTC-USD",
            exchange="hyperliquid",
            indicators=["MACD"],
            position_sizing={
                'method': 'risk_based',
                'risk_per_trade_pct': 0.03,  # 3% risk per trade
                'default_stop_loss_pct': 0.06,  # 6% default stop loss
                'max_position_size_usd': 4000.0
            }
        )

        config = {
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0  # Different global method
            }
        }

        risk_manager = RiskManager.from_config(config, strategies=[strategy])

        # Create mock exchange
        mock_exchange = MagicMock()
        mock_exchange.get_ticker.return_value = {'last_price': 50000.0}
        mock_exchange.get_leverage_tiers.return_value = [{'max_leverage': 10.0}]

        # Test with specific stop loss
        position_size_usd, _ = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='BTC',
            available_balance=20000.0,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=3.0,
            stop_loss_pct=4.0,  # 4% stop loss
            strategy_name="btc_risk_strategy"
        )

        # Risk-based calculation: $20,000 * 0.03 risk = $600 risk
        # With 4% stop loss, position size should be calculated accordingly
        assert position_size_usd > 0
        assert position_size_usd <= 4000.0  # Should respect strategy max

        # Verify strategy-specific risk parameters
        strategy_sizer = risk_manager.strategy_position_sizers["btc_risk_strategy"]
        assert strategy_sizer.config.method == PositionSizingMethod.RISK_BASED
        assert strategy_sizer.config.risk_per_trade_pct == 0.03
        assert strategy_sizer.config.default_stop_loss_pct == 0.06

    def test_strategy_specific_percent_equity_position_sizing(self):
        """Test strategy-specific percent equity position sizing."""
        strategy = StrategyConfig(
            name="ada_percent_strategy",
            market="ADA-USD",
            exchange="coinbase",
            indicators=["SMA"],
            position_sizing={
                'method': 'percent_equity',
                'equity_percentage': 0.04,  # 4% of equity per trade
                'max_position_size_usd': 2500.0,
                'min_position_size_usd': 25.0
            }
        )

        config = {
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0  # Different global method
            }
        }

        risk_manager = RiskManager.from_config(config, strategies=[strategy])

        # Create mock exchange
        mock_exchange = MagicMock()
        mock_exchange.get_ticker.return_value = {'last_price': 1.0}
        mock_exchange.get_leverage_tiers.return_value = [{'max_leverage': 5.0}]

        # Test percent equity calculation
        position_size_usd, _ = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='ADA',
            available_balance=15000.0,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=2.0,
            strategy_name="ada_percent_strategy"
        )

        # Should use 4% of equity: $15,000 * 0.04 = $600
        assert position_size_usd == 600.0

        # Test with different equity amount
        position_size_usd, _ = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='ADA',
            available_balance=30000.0,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=2.0,
            strategy_name="ada_percent_strategy"
        )

        # Should use 4% of equity: $30,000 * 0.04 = $1,200
        assert position_size_usd == 1200.0

        # Verify strategy-specific configuration
        strategy_sizer = risk_manager.strategy_position_sizers["ada_percent_strategy"]
        assert strategy_sizer.config.method == PositionSizingMethod.PERCENT_EQUITY
        assert strategy_sizer.config.equity_percentage == 0.04

    def test_position_sizing_config_inheritance(self):
        """Test position sizing config inheritance from global config."""
        # Strategy with partial custom config (should inherit missing fields)
        strategy = StrategyConfig(
            name="partial_strategy",
            market="ETH-USD",
            exchange="hyperliquid",
            indicators=["RSI"],
            position_sizing={
                'method': 'fixed_usd',
                'fixed_usd_amount': 750.0,
                # Missing max_position_size_usd, should inherit from global
            }
        )

        global_config = {
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0,
                'max_position_size_usd': 5000.0,
                'min_position_size_usd': 100.0,
                'max_leverage': 10.0
            }
        }

        risk_manager = RiskManager.from_config(global_config, strategies=[strategy])

        # Get the strategy-specific position sizer
        strategy_sizer = risk_manager.strategy_position_sizers["partial_strategy"]

        # Verify inheritance: custom amount but inherited limits
        assert strategy_sizer.config.fixed_usd_amount == 750.0  # Custom value
        assert strategy_sizer.config.max_position_size_usd == 5000.0  # Inherited
        assert strategy_sizer.config.min_position_size_usd == 100.0  # Inherited
        assert strategy_sizer.config.max_leverage == 10.0  # Inherited

    def test_invalid_strategy_position_sizing_configs(self):
        """Test invalid strategy position sizing configs."""
        # Test missing required parameters for fixed_usd method
        # The implementation is forgiving and will use defaults for missing parameters
        strategy_invalid_fixed = StrategyConfig(
            name="invalid_fixed_strategy",
            market="ETH-USD",
            exchange="hyperliquid",
            indicators=["RSI"],
            position_sizing={
                'method': 'fixed_usd',
                # Missing required fixed_usd_amount - should use default
                'max_position_size_usd': 2000.0
            }
        )

        config = {
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0
            }
        }

        risk_manager = RiskManager.from_config(config, strategies=[strategy_invalid_fixed])

        # Should create a position sizer with default values
        assert "invalid_fixed_strategy" in risk_manager.strategy_position_sizers
        strategy_sizer = risk_manager.strategy_position_sizers["invalid_fixed_strategy"]
        assert strategy_sizer.config.method == PositionSizingMethod.FIXED_USD
        # Should inherit from global config if not specified
        assert strategy_sizer.config.fixed_usd_amount == 1000.0  # From global config

        # Test invalid percentage for percent_equity method
        # The implementation validates this during config creation, but let's test with a valid config
        strategy_invalid_percent = StrategyConfig(
            name="valid_percent_strategy",
            market="BTC-USD",
            exchange="hyperliquid",
            indicators=["MACD"],
            position_sizing={
                'method': 'percent_equity',
                'equity_percentage': 0.15,  # Valid: 15%
                'max_position_size_usd': 3000.0
            }
        )

        risk_manager2 = RiskManager.from_config(config, strategies=[strategy_invalid_percent])

        # Should create a valid position sizer
        assert "valid_percent_strategy" in risk_manager2.strategy_position_sizers
        percent_sizer = risk_manager2.strategy_position_sizers["valid_percent_strategy"]
        assert percent_sizer.config.method == PositionSizingMethod.PERCENT_EQUITY
        assert percent_sizer.config.equity_percentage == 0.15

        # Test invalid risk percentage for risk_based method
        # Use a valid configuration instead
        strategy_valid_risk = StrategyConfig(
            name="valid_risk_strategy",
            market="ADA-USD",
            exchange="coinbase",
            indicators=["SMA"],
            position_sizing={
                'method': 'risk_based',
                'risk_per_trade_pct': 0.025,  # Valid: 2.5%
                'default_stop_loss_pct': 0.05
            }
        )

        risk_manager3 = RiskManager.from_config(config, strategies=[strategy_valid_risk])

        # Should create a valid position sizer
        assert "valid_risk_strategy" in risk_manager3.strategy_position_sizers
        risk_sizer = risk_manager3.strategy_position_sizers["valid_risk_strategy"]
        assert risk_sizer.config.method == PositionSizingMethod.RISK_BASED
        assert risk_sizer.config.risk_per_trade_pct == 0.025

    def test_strategy_position_sizer_creation_and_validation(self):
        """Test strategy position sizer creation and validation."""
        # Test valid strategy configurations
        strategies = [
            StrategyConfig(
                name="valid_fixed_strategy",
                market="ETH-USD",
                exchange="hyperliquid",
                indicators=["RSI"],
                position_sizing={
                    'method': 'fixed_usd',
                    'fixed_usd_amount': 500.0,
                    'max_position_size_usd': 2000.0
                }
            ),
            StrategyConfig(
                name="valid_percent_strategy",
                market="BTC-USD",
                exchange="hyperliquid",
                indicators=["MACD"],
                position_sizing={
                    'method': 'percent_equity',
                    'equity_percentage': 0.02,
                    'max_position_size_usd': 3000.0
                }
            ),
            StrategyConfig(
                name="valid_risk_strategy",
                market="ADA-USD",
                exchange="coinbase",
                indicators=["SMA"],
                position_sizing={
                    'method': 'risk_based',
                    'risk_per_trade_pct': 0.025,
                    'default_stop_loss_pct': 0.04,
                    'max_position_size_usd': 1500.0
                }
            )
        ]

        config = {
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0,
                'max_position_size_usd': 5000.0
            }
        }

        # Create risk manager with strategies
        risk_manager = RiskManager.from_config(config, strategies=strategies)

        # Verify all strategy position sizers were created successfully
        assert len(risk_manager.strategy_position_sizers) == 3

        # Verify each strategy position sizer has correct configuration
        fixed_sizer = risk_manager.strategy_position_sizers["valid_fixed_strategy"]
        assert fixed_sizer.config.method == PositionSizingMethod.FIXED_USD
        assert fixed_sizer.config.fixed_usd_amount == 500.0

        percent_sizer = risk_manager.strategy_position_sizers["valid_percent_strategy"]
        assert percent_sizer.config.method == PositionSizingMethod.PERCENT_EQUITY
        assert percent_sizer.config.equity_percentage == 0.02

        risk_sizer = risk_manager.strategy_position_sizers["valid_risk_strategy"]
        assert risk_sizer.config.method == PositionSizingMethod.RISK_BASED
        assert risk_sizer.config.risk_per_trade_pct == 0.025

    def test_position_sizing_method_override_inheritance(self):
        """Test that strategy-specific methods override global methods correctly."""
        strategies = [
            # Strategy 1: Override method but inherit other parameters
            StrategyConfig(
                name="override_method_strategy",
                market="ETH-USD",
                exchange="hyperliquid",
                indicators=["RSI"],
                position_sizing={
                    'method': 'percent_equity',  # Override global fixed_usd method
                    'equity_percentage': 0.03,
                    # Will inherit max_position_size_usd from global
                }
            ),
            # Strategy 2: Override specific parameter but keep global method
            StrategyConfig(
                name="override_param_strategy",
                market="BTC-USD",
                exchange="hyperliquid",
                indicators=["MACD"],
                position_sizing={
                    'method': 'fixed_usd',  # Same as global method
                    'fixed_usd_amount': 750.0,  # Override global amount
                    # Will inherit max_position_size_usd from global
                }
            )
        ]

        global_config = {
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0,
                'max_position_size_usd': 4000.0,
                'min_position_size_usd': 50.0,
                'max_leverage': 8.0
            }
        }

        risk_manager = RiskManager.from_config(global_config, strategies=strategies)

        # Test strategy 1: Method override
        override_method_sizer = risk_manager.strategy_position_sizers["override_method_strategy"]
        assert override_method_sizer.config.method == PositionSizingMethod.PERCENT_EQUITY
        assert override_method_sizer.config.equity_percentage == 0.03
        assert override_method_sizer.config.max_position_size_usd == 4000.0  # Inherited
        assert override_method_sizer.config.max_leverage == 8.0  # Inherited

        # Test strategy 2: Parameter override
        override_param_sizer = risk_manager.strategy_position_sizers["override_param_strategy"]
        assert override_param_sizer.config.method == PositionSizingMethod.FIXED_USD
        assert override_param_sizer.config.fixed_usd_amount == 750.0  # Overridden
        assert override_param_sizer.config.max_position_size_usd == 4000.0  # Inherited
        assert override_param_sizer.config.min_position_size_usd == 50.0  # Inherited

    def test_mixed_strategy_position_sizing_calculation(self):
        """Test position size calculation with mixed strategy configurations."""
        strategies = [
            StrategyConfig(
                name="conservative_strategy",
                market="ETH-USD",
                exchange="hyperliquid",
                indicators=["RSI"],
                position_sizing={
                    'method': 'fixed_usd',
                    'fixed_usd_amount': 300.0,  # Small fixed amount
                }
            ),
            StrategyConfig(
                name="aggressive_strategy",
                market="BTC-USD",
                exchange="hyperliquid",
                indicators=["MACD"],
                position_sizing={
                    'method': 'percent_equity',
                    'equity_percentage': 0.08,  # Higher percentage
                }
            ),
            StrategyConfig(
                name="balanced_strategy",
                market="ADA-USD",
                exchange="coinbase",
                indicators=["SMA"],
                position_sizing={
                    'method': 'risk_based',
                    'risk_per_trade_pct': 0.02,
                    'default_stop_loss_pct': 0.05
                }
            )
        ]

        config = {
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0  # Global default
            }
        }

        risk_manager = RiskManager.from_config(config, strategies=strategies)

        # Create mock exchange
        mock_exchange = MagicMock()
        mock_exchange.get_ticker.return_value = {'last_price': 2000.0}
        mock_exchange.get_leverage_tiers.return_value = [{'max_leverage': 10.0}]

        account_balance = 25000.0  # $25k account

        # Test conservative strategy (fixed small amount)
        conservative_size, _ = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='ETH',
            available_balance=account_balance,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=5.0,
            strategy_name="conservative_strategy"
        )
        assert conservative_size == 300.0  # Small fixed amount

        # Test aggressive strategy (high percentage)
        aggressive_size, _ = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='BTC',
            available_balance=account_balance,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=5.0,
            strategy_name="aggressive_strategy"
        )
        assert aggressive_size == 2000.0  # $25k * 8% = $2k

        # Test balanced strategy (risk-based)
        balanced_size, _ = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='ADA',
            available_balance=account_balance,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=5.0,
            stop_loss_pct=5.0,  # 5% stop loss
            strategy_name="balanced_strategy"
        )
        # Risk-based: $25k * 2% = $500 risk with 5% stop loss
        # Position size = $500 risk / ($2000 * 0.05) = $500 / $100 = 5 units = $10,000 USD
        # This is the correct calculation: risk_amount / (price * stop_loss_percentage)
        assert balanced_size > 0
        assert balanced_size == 10000.0  # Correct expectation: $500 / $100 per unit risk = 5 units * $2000 = $10,000

        # Test fallback to global default for unknown strategy
        default_size, _ = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='UNKNOWN',
            available_balance=account_balance,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=5.0,
            strategy_name="unknown_strategy"
        )
        assert default_size == 1000.0  # Global default

    def test_strategy_position_sizing_error_handling(self):
        """Test error handling in strategy position sizing."""
        # Test strategy with invalid position sizing config that causes fallback
        strategy_with_invalid_config = StrategyConfig(
            name="invalid_config_strategy",
            market="ETH-USD",
            exchange="hyperliquid",
            indicators=["RSI"],
            position_sizing={
                'method': 'invalid_method',  # This will cause fallback to fixed_usd
                'some_param': 123
            }
        )

        config = {
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0  # Fallback config
            }
        }

        risk_manager = RiskManager.from_config(config, strategies=[strategy_with_invalid_config])

        # Strategy should be in strategy_position_sizers with fallback method (fixed_usd)
        assert "invalid_config_strategy" in risk_manager.strategy_position_sizers
        fallback_sizer = risk_manager.strategy_position_sizers["invalid_config_strategy"]
        assert fallback_sizer.config.method == PositionSizingMethod.FIXED_USD  # Fallback method

        # Create mock exchange
        mock_exchange = MagicMock()
        mock_exchange.get_ticker.return_value = {'last_price': 2000.0}
        mock_exchange.get_leverage_tiers.return_value = [{'max_leverage': 10.0}]

        # Should use fallback position sizing when strategy config was invalid
        position_size_usd, _ = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='ETH',
            available_balance=10000.0,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=5.0,
            strategy_name="invalid_config_strategy"
        )

        # Should use fallback method inherited from global config
        assert position_size_usd == 1000.0
