from unittest.mock import MagicMock

import pytest
from app.connectors.base_connector import OrderSide
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
