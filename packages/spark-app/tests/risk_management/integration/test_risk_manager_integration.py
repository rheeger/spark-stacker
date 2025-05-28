from unittest.mock import MagicMock

import pytest
from app.connectors.base_connector import OrderSide
from app.core.strategy_config import StrategyConfig
from app.risk_management import (PositionSizingConfig, PositionSizingMethod,
                                 RiskManager)


class TestRiskManagerIntegration:
    """Test cases for risk manager integration with position sizing."""

    def test_risk_manager_from_config_with_position_sizing(self):
        """Test creating risk manager from config with position sizing configuration."""
        config = {
            'max_account_risk_pct': 2.0,
            'max_leverage': 10.0,
            'max_position_size_usd': 5000.0,
            'max_positions': 3,
            'min_margin_buffer_pct': 25.0,
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0,
                'max_position_size_usd': 5000.0,
                'min_position_size_usd': 100.0,
                'max_leverage': 10.0
            }
        }

        risk_manager = RiskManager.from_config(config)

        # Verify risk manager parameters
        assert risk_manager.max_account_risk_pct == 2.0
        assert risk_manager.max_leverage == 10.0
        assert risk_manager.max_position_size_usd == 5000.0
        assert risk_manager.max_positions == 3
        assert risk_manager.min_margin_buffer_pct == 25.0

        # Verify position sizer is created and configured correctly
        assert risk_manager.position_sizer is not None
        assert risk_manager.position_sizer.config.method == PositionSizingMethod.FIXED_USD
        assert risk_manager.position_sizer.config.fixed_usd_amount == 1000.0
        assert risk_manager.position_sizer.config.max_position_size_usd == 5000.0

    def test_risk_manager_from_config_without_position_sizing(self):
        """Test creating risk manager from config without position sizing configuration."""
        config = {
            'max_account_risk_pct': 1.5,
            'max_leverage': 5.0,
            'max_position_size_usd': 2000.0,
            'max_positions': 2,
            'min_margin_buffer_pct': 30.0
        }

        risk_manager = RiskManager.from_config(config)

        # Verify risk manager parameters
        assert risk_manager.max_account_risk_pct == 1.5
        assert risk_manager.max_leverage == 5.0
        assert risk_manager.max_position_size_usd == 2000.0

        # Verify default position sizer is created
        assert risk_manager.position_sizer is not None
        assert risk_manager.position_sizer.config.method == PositionSizingMethod.FIXED_USD
        assert risk_manager.position_sizer.config.fixed_usd_amount == 2000.0  # Should use max_position_size_usd

    def test_calculate_position_size_integration(self):
        """Test position size calculation using integrated position sizer."""
        config = {
            'max_account_risk_pct': 2.0,
            'max_leverage': 10.0,
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0,
                'max_position_size_usd': 5000.0,
                'min_position_size_usd': 100.0
            }
        }

        risk_manager = RiskManager.from_config(config)

        # Create mock exchange
        mock_exchange = MagicMock()
        mock_exchange.get_ticker.return_value = {'last_price': 50000.0}
        mock_exchange.get_leverage_tiers.return_value = [{'max_leverage': 20.0}]

        # Test position size calculation
        position_size_usd, leverage = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='BTC-USD',
            available_balance=10000.0,
            confidence=0.8,
            signal_side=OrderSide.BUY,
            leverage=5.0
        )

                # Should return $800 position (fixed USD amount * confidence)
        # $1000 * 0.8 confidence = $800
        assert position_size_usd == 800.0
        assert leverage == 5.0  # Should be capped appropriately

    def test_calculate_position_size_with_risk_based_method(self):
        """Test position size calculation using risk-based method."""
        config = {
            'max_account_risk_pct': 2.0,
            'max_leverage': 10.0,
            'position_sizing': {
                'method': 'risk_based',
                'risk_per_trade_pct': 0.02,  # 2% risk per trade
                'default_stop_loss_pct': 0.05,  # 5% stop loss
                'max_position_size_usd': 5000.0,
                'min_position_size_usd': 100.0
            }
        }

        risk_manager = RiskManager.from_config(config)

        # Create mock exchange
        mock_exchange = MagicMock()
        mock_exchange.get_ticker.return_value = {'last_price': 50000.0}
        mock_exchange.get_leverage_tiers.return_value = [{'max_leverage': 20.0}]

        # Test position size calculation with stop loss
        position_size_usd, leverage = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='BTC-USD',
            available_balance=10000.0,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=5.0,
            stop_loss_pct=5.0  # 5% stop loss
        )

        # Should calculate based on risk (2% of $10,000 = $200 risk with 5% stop loss)
        # Position size should be calculated accordingly
        assert position_size_usd > 0
        assert position_size_usd <= 5000.0  # Should not exceed max

    def test_position_sizing_method_from_config_key(self):
        """Test that the position sizing method key is correctly parsed."""
        config_dict = {
            'position_sizing': {
                'method': 'percent_equity',  # Using 'method' key
                'equity_percentage': 0.03,
                'max_position_size_usd': 2000.0,
                'min_position_size_usd': 50.0
            }
        }

        risk_manager = RiskManager.from_config(config_dict)

        # Verify the method was parsed correctly
        assert risk_manager.position_sizer.config.method == PositionSizingMethod.PERCENT_EQUITY
        assert risk_manager.position_sizer.config.equity_percentage == 0.03

    def test_position_sizing_method_legacy_key(self):
        """Test that the legacy position sizing method key is correctly parsed."""
        config_dict = {
            'position_sizing': {
                'position_sizing_method': 'kelly_criterion',  # Using legacy key
                'kelly_win_rate': 0.65,
                'kelly_avg_win': 0.04,
                'kelly_avg_loss': 0.025,
                'max_position_size_usd': 3000.0
            }
        }

        risk_manager = RiskManager.from_config(config_dict)

        # Verify the method was parsed correctly
        assert risk_manager.position_sizer.config.method == PositionSizingMethod.KELLY_CRITERION
        assert risk_manager.position_sizer.config.kelly_win_rate == 0.65

    def test_leverage_limits_integration(self):
        """Test that leverage limits are properly integrated between risk manager and position sizer."""
        config = {
            'max_leverage': 5.0,  # Risk manager limit
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0,
                'max_leverage': 3.0  # Position sizer limit (more restrictive)
            }
        }

        risk_manager = RiskManager.from_config(config)

        # Create mock exchange with high leverage available
        mock_exchange = MagicMock()
        mock_exchange.get_ticker.return_value = {'last_price': 50000.0}
        mock_exchange.get_leverage_tiers.return_value = [{'max_leverage': 20.0}]

        # Test that the most restrictive leverage limit is applied
        position_size_usd, leverage = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='BTC-USD',
            available_balance=10000.0,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=10.0  # Request high leverage
        )

        # Should be capped at the position sizer's max leverage (3.0)
        assert leverage == 3.0

    # --- NEW TESTS FOR STRATEGY-SPECIFIC POSITION SIZING ---

    def test_risk_manager_creation_with_strategy_specific_position_sizing(self):
        """Test RiskManager creation with strategy-specific position sizing."""
        # Create sample strategies with different position sizing configs
        strategy1 = StrategyConfig(
            name="eth_fixed_strategy",
            market="ETH-USD",
            exchange="hyperliquid",
            indicators=["RSI"],
            position_sizing={
                'method': 'fixed_usd',
                'fixed_usd_amount': 500.0,
                'max_position_size_usd': 2000.0
            }
        )

        strategy2 = StrategyConfig(
            name="btc_risk_strategy",
            market="BTC-USD",
            exchange="hyperliquid",
            indicators=["MACD"],
            position_sizing={
                'method': 'risk_based',
                'risk_per_trade_pct': 0.03,
                'default_stop_loss_pct': 0.08
            }
        )

        strategy3 = StrategyConfig(
            name="no_custom_strategy",
            market="ADA-USD",
            exchange="coinbase",
            indicators=["SMA"]
            # No position_sizing config - should use global default
        )

        config = {
            'max_account_risk_pct': 2.0,
            'max_leverage': 10.0,
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0,  # Global default
                'max_position_size_usd': 5000.0
            }
        }

        strategies = [strategy1, strategy2, strategy3]
        risk_manager = RiskManager.from_config(config, strategies=strategies)

        # Verify strategy-specific position sizers were created
        assert len(risk_manager.strategy_position_sizers) == 2  # Only strategies with custom configs

        # Verify strategy1 position sizer
        eth_sizer = risk_manager.strategy_position_sizers["eth_fixed_strategy"]
        assert eth_sizer.config.method == PositionSizingMethod.FIXED_USD
        assert eth_sizer.config.fixed_usd_amount == 500.0
        assert eth_sizer.config.max_position_size_usd == 2000.0

        # Verify strategy2 position sizer
        btc_sizer = risk_manager.strategy_position_sizers["btc_risk_strategy"]
        assert btc_sizer.config.method == PositionSizingMethod.RISK_BASED
        assert btc_sizer.config.risk_per_trade_pct == 0.03
        assert btc_sizer.config.default_stop_loss_pct == 0.08

        # Verify strategy3 is not in strategy_position_sizers (uses default)
        assert "no_custom_strategy" not in risk_manager.strategy_position_sizers

        # Verify default position sizer
        assert risk_manager.position_sizer.config.method == PositionSizingMethod.FIXED_USD
        assert risk_manager.position_sizer.config.fixed_usd_amount == 1000.0

    def test_calculate_position_size_with_strategy_context(self):
        """Test calculate_position_size() with strategy context."""
        # Create strategies with different position sizing methods
        strategy1 = StrategyConfig(
            name="fixed_strategy",
            market="ETH-USD",
            exchange="hyperliquid",
            indicators=["RSI"],
            position_sizing={
                'method': 'fixed_usd',
                'fixed_usd_amount': 750.0
            }
        )

        strategy2 = StrategyConfig(
            name="percent_strategy",
            market="BTC-USD",
            exchange="hyperliquid",
            indicators=["MACD"],
            position_sizing={
                'method': 'percent_equity',
                'equity_percentage': 0.05  # 5% of equity
            }
        )

        config = {
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0  # Global default
            }
        }

        strategies = [strategy1, strategy2]
        risk_manager = RiskManager.from_config(config, strategies=strategies)

        # Create mock exchange
        mock_exchange = MagicMock()
        mock_exchange.get_ticker.return_value = {'last_price': 2000.0}
        mock_exchange.get_leverage_tiers.return_value = [{'max_leverage': 10.0}]

        # Test with strategy1 (fixed USD)
        position_size_usd, leverage = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='ETH',
            available_balance=10000.0,
            confidence=0.8,
            signal_side=OrderSide.BUY,
            leverage=5.0,
            strategy_name="fixed_strategy"
        )

        # Should use strategy1's fixed amount: $750 * 0.8 = $600
        assert position_size_usd == 600.0

        # Test with strategy2 (percent equity)
        position_size_usd, leverage = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='BTC',
            available_balance=10000.0,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=5.0,
            strategy_name="percent_strategy"
        )

        # Should use strategy2's percent: $10,000 * 0.05 = $500
        assert position_size_usd == 500.0

        # Test with unknown strategy (should use default)
        position_size_usd, leverage = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='ADA',
            available_balance=10000.0,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=5.0,
            strategy_name="unknown_strategy"
        )

        # Should use global default: $1000 * 1.0 = $1000
        assert position_size_usd == 1000.0

    def test_strategy_position_sizer_routing(self):
        """Test strategy position sizer routing."""
        strategy = StrategyConfig(
            name="test_strategy",
            market="ETH-USD",
            exchange="hyperliquid",
            indicators=["RSI"],
            position_sizing={
                'method': 'fixed_usd',
                'fixed_usd_amount': 500.0
            }
        )

        config = {
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0  # Different global default
            }
        }

        risk_manager = RiskManager.from_config(config, strategies=[strategy])

        # Create mock exchange
        mock_exchange = MagicMock()
        mock_exchange.get_ticker.return_value = {'last_price': 2000.0}
        mock_exchange.get_leverage_tiers.return_value = [{'max_leverage': 10.0}]

        # Test routing to strategy-specific sizer
        position_size_usd, _ = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='ETH',
            available_balance=10000.0,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=5.0,
            strategy_name="test_strategy"
        )

        # Should use strategy-specific amount
        assert position_size_usd == 500.0

        # Test routing to default sizer when no strategy specified
        position_size_usd, _ = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='ETH',
            available_balance=10000.0,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=5.0,
            strategy_name=None
        )

        # Should use global default amount
        assert position_size_usd == 1000.0

    def test_fallback_to_default_position_sizer(self):
        """Test fallback to default position sizer."""
        strategy = StrategyConfig(
            name="has_custom_sizer",
            market="ETH-USD",
            exchange="hyperliquid",
            indicators=["RSI"],
            position_sizing={
                'method': 'fixed_usd',
                'fixed_usd_amount': 500.0
            }
        )

        config = {
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0  # Global fallback
            }
        }

        risk_manager = RiskManager.from_config(config, strategies=[strategy])

        # Create mock exchange
        mock_exchange = MagicMock()
        mock_exchange.get_ticker.return_value = {'last_price': 2000.0}
        mock_exchange.get_leverage_tiers.return_value = [{'max_leverage': 10.0}]

        # Test with strategy that has custom sizer
        position_size_usd, _ = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='ETH',
            available_balance=10000.0,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=5.0,
            strategy_name="has_custom_sizer"
        )
        assert position_size_usd == 500.0

        # Test with strategy that doesn't exist (fallback to default)
        position_size_usd, _ = risk_manager.calculate_position_size(
            exchange=mock_exchange,
            symbol='ETH',
            available_balance=10000.0,
            confidence=1.0,
            signal_side=OrderSide.BUY,
            leverage=5.0,
            strategy_name="nonexistent_strategy"
        )
        assert position_size_usd == 1000.0

    def test_multiple_strategies_with_different_position_sizing_methods(self):
        """Test multiple strategies with different position sizing methods."""
        strategies = [
            StrategyConfig(
                name="fixed_strategy",
                market="ETH-USD",
                exchange="hyperliquid",
                indicators=["RSI"],
                position_sizing={
                    'method': 'fixed_usd',
                    'fixed_usd_amount': 400.0
                }
            ),
            StrategyConfig(
                name="percent_strategy",
                market="BTC-USD",
                exchange="hyperliquid",
                indicators=["MACD"],
                position_sizing={
                    'method': 'percent_equity',
                    'equity_percentage': 0.03
                }
            ),
            StrategyConfig(
                name="risk_strategy",
                market="ADA-USD",
                exchange="coinbase",
                indicators=["SMA"],
                position_sizing={
                    'method': 'risk_based',
                    'risk_per_trade_pct': 0.025,
                    'default_stop_loss_pct': 0.05
                }
            )
        ]

        config = {
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0
            }
        }

        risk_manager = RiskManager.from_config(config, strategies=strategies)

        # Verify all strategy-specific position sizers were created
        assert len(risk_manager.strategy_position_sizers) == 3

        # Verify each strategy has the correct position sizer type
        fixed_sizer = risk_manager.strategy_position_sizers["fixed_strategy"]
        assert fixed_sizer.config.method == PositionSizingMethod.FIXED_USD
        assert fixed_sizer.config.fixed_usd_amount == 400.0

        percent_sizer = risk_manager.strategy_position_sizers["percent_strategy"]
        assert percent_sizer.config.method == PositionSizingMethod.PERCENT_EQUITY
        assert percent_sizer.config.equity_percentage == 0.03

        risk_sizer = risk_manager.strategy_position_sizers["risk_strategy"]
        assert risk_sizer.config.method == PositionSizingMethod.RISK_BASED
        assert risk_sizer.config.risk_per_trade_pct == 0.025

    def test_position_sizer_factory_method(self):
        """Test position sizer factory method."""
        # Test strategy with custom position sizing
        strategy_with_custom = StrategyConfig(
            name="custom_strategy",
            market="ETH-USD",
            exchange="hyperliquid",
            indicators=["RSI"],
            position_sizing={
                'method': 'fixed_usd',
                'fixed_usd_amount': 750.0,
                'max_position_size_usd': 3000.0
            }
        )

        # Test strategy without custom position sizing
        strategy_without_custom = StrategyConfig(
            name="default_strategy",
            market="BTC-USD",
            exchange="hyperliquid",
            indicators=["MACD"]
            # No position_sizing config
        )

        config = {
            'position_sizing': {
                'method': 'fixed_usd',
                'fixed_usd_amount': 1000.0,
                'max_position_size_usd': 5000.0
            }
        }

        risk_manager = RiskManager.from_config(config)

        # Test factory method with custom strategy
        custom_sizer = risk_manager._create_position_sizer_for_strategy(strategy_with_custom)
        assert custom_sizer.config.method == PositionSizingMethod.FIXED_USD
        assert custom_sizer.config.fixed_usd_amount == 750.0
        assert custom_sizer.config.max_position_size_usd == 3000.0

        # Test factory method with default strategy (should inherit global config)
        default_sizer = risk_manager._create_position_sizer_for_strategy(strategy_without_custom)
        assert default_sizer.config.method == PositionSizingMethod.FIXED_USD
        assert default_sizer.config.fixed_usd_amount == 1000.0  # From global config
        assert default_sizer.config.max_position_size_usd == 5000.0  # From global config
