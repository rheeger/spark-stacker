import pytest
from app.risk_management.position_sizing import (PositionSizer,
                                                 PositionSizingConfig,
                                                 PositionSizingMethod)


class TestPositionSizing:
    """Test cases for position sizing functionality."""

    def test_fixed_usd_position_sizing(self):
        """Test fixed USD position sizing method."""
        config = PositionSizingConfig(
            method=PositionSizingMethod.FIXED_USD,
            fixed_usd_amount=1000.0
        )
        sizer = PositionSizer(config)

        # Test position size calculation
        position_size = sizer.calculate_position_size(
            current_equity=10000.0,
            current_price=50000.0
        )

        expected_size = 1000.0 / 50000.0  # $1000 / $50000 per BTC = 0.02 BTC
        assert abs(position_size - expected_size) < 1e-6

    def test_percent_equity_position_sizing(self):
        """Test percentage of equity position sizing method."""
        config = PositionSizingConfig(
            method=PositionSizingMethod.PERCENT_EQUITY,
            equity_percentage=0.05  # 5% of equity
        )
        sizer = PositionSizer(config)

        position_size = sizer.calculate_position_size(
            current_equity=10000.0,
            current_price=50000.0
        )

        # 5% of $10,000 = $500, at $50,000 per BTC = 0.01 BTC
        expected_size = (10000.0 * 0.05) / 50000.0
        assert abs(position_size - expected_size) < 1e-6

    def test_risk_based_position_sizing(self):
        """Test risk-based position sizing method."""
        config = PositionSizingConfig(
            method=PositionSizingMethod.RISK_BASED,
            risk_per_trade_pct=0.02  # 2% risk per trade
        )
        sizer = PositionSizer(config)

        current_price = 50000.0
        stop_loss_price = 48000.0  # 4% stop loss

        position_size = sizer.calculate_position_size(
            current_equity=10000.0,
            current_price=current_price,
            stop_loss_price=stop_loss_price
        )

        # Risk = 2% of $10,000 = $200
        # Risk per unit = $50,000 - $48,000 = $2,000
        # Position size = $200 / $2,000 = 0.1 BTC
        expected_size = (10000.0 * 0.02) / (50000.0 - 48000.0)
        assert abs(position_size - expected_size) < 1e-6

    def test_position_size_limits(self):
        """Test that position size limits are applied correctly."""
        config = PositionSizingConfig(
            method=PositionSizingMethod.FIXED_USD,
            fixed_usd_amount=5000.0,  # Large amount
            max_position_size_usd=1000.0,  # But limited to $1000
            min_position_size_usd=100.0
        )
        sizer = PositionSizer(config)

        # Test max limit
        position_size = sizer.calculate_position_size(
            current_equity=10000.0,
            current_price=50000.0
        )

        # Should be capped at $1000 max
        expected_size = 1000.0 / 50000.0
        assert abs(position_size - expected_size) < 1e-6

        # Test min limit with very small amount
        config.fixed_usd_amount = 50.0  # Below minimum
        sizer = PositionSizer(config)

        position_size = sizer.calculate_position_size(
            current_equity=10000.0,
            current_price=50000.0
        )

        # Should be increased to $100 minimum
        expected_size = 100.0 / 50000.0
        assert abs(position_size - expected_size) < 1e-6

    def test_config_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            'position_sizing_method': 'fixed_usd',
            'fixed_usd_amount': 500.0,
            'max_position_size_usd': 2000.0,
            'min_position_size_usd': 50.0
        }

        config = PositionSizingConfig.from_config_dict(config_dict)

        assert config.method == PositionSizingMethod.FIXED_USD
        assert config.fixed_usd_amount == 500.0
        assert config.max_position_size_usd == 2000.0
        assert config.min_position_size_usd == 50.0

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        config = PositionSizingConfig(method=PositionSizingMethod.FIXED_USD)
        sizer = PositionSizer(config)

        # Test with zero equity
        position_size = sizer.calculate_position_size(
            current_equity=0.0,
            current_price=50000.0
        )
        assert position_size == 0.0

        # Test with zero price
        position_size = sizer.calculate_position_size(
            current_equity=10000.0,
            current_price=0.0
        )
        assert position_size == 0.0

    def test_signal_strength_modifier(self):
        """Test that signal strength modifies position size."""
        config = PositionSizingConfig(
            method=PositionSizingMethod.FIXED_USD,
            fixed_usd_amount=1000.0
        )
        sizer = PositionSizer(config)

        # Full strength signal
        full_size = sizer.calculate_position_size(
            current_equity=10000.0,
            current_price=50000.0,
            signal_strength=1.0
        )

        # Half strength signal
        half_size = sizer.calculate_position_size(
            current_equity=10000.0,
            current_price=50000.0,
            signal_strength=0.5
        )

        assert abs(half_size - (full_size * 0.5)) < 1e-6

    def test_validation(self):
        """Test position size validation."""
        config = PositionSizingConfig(
            method=PositionSizingMethod.FIXED_USD,
            max_position_size_usd=1000.0,
            min_position_size_usd=100.0
        )
        sizer = PositionSizer(config)

        current_price = 50000.0
        current_equity = 10000.0

        # Valid position size
        valid_size = 500.0 / current_price  # $500 position
        assert sizer.validate_position_size(valid_size, current_price, current_equity)

        # Too large position size
        large_size = 2000.0 / current_price  # $2000 position (> $1000 max)
        assert not sizer.validate_position_size(large_size, current_price, current_equity)

        # Too small position size
        small_size = 50.0 / current_price  # $50 position (< $100 min)
        assert not sizer.validate_position_size(small_size, current_price, current_equity)

        # Zero position size
        assert not sizer.validate_position_size(0.0, current_price, current_equity)
