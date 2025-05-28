import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from app.risk_management.risk_manager import RiskManager
from app.connectors.base_connector import OrderSide


def test_risk_manager_initialization():
    """Test risk manager initialization with default and custom parameters."""
    # Test default initialization
    risk_manager = RiskManager()
    assert risk_manager.max_account_risk_pct == 2.0
    assert risk_manager.max_leverage == 25.0
    assert risk_manager.max_position_size_usd is None
    assert risk_manager.max_positions == 3
    assert risk_manager.min_margin_buffer_pct == 20.0

    # Test custom initialization
    custom_risk_manager = RiskManager(
        max_account_risk_pct=1.0,
        max_leverage=10.0,
        max_position_size_usd=1000.0,
        max_positions=2,
        min_margin_buffer_pct=30.0,
    )
    assert custom_risk_manager.max_account_risk_pct == 1.0
    assert custom_risk_manager.max_leverage == 10.0
    assert custom_risk_manager.max_position_size_usd == 1000.0
    assert custom_risk_manager.max_positions == 2
    assert custom_risk_manager.min_margin_buffer_pct == 30.0


def test_calculate_position_size(mock_connector):
    """Test position size calculation based on risk parameters."""
    risk_manager = RiskManager(max_account_risk_pct=2.0, max_leverage=10.0)

    # Set up the mock connector
    mock_connector.get_ticker.return_value = {"symbol": "ETH", "last_price": 1500.0}
    mock_connector.get_leverage_tiers.return_value = [{"max_leverage": 10.0}]

    # Test position size calculation for a buy order
    position_size, adjusted_leverage = risk_manager.calculate_position_size(
        exchange=mock_connector,
        symbol="ETH",
        available_balance=10000.0,
        confidence=0.8,
        signal_side=OrderSide.BUY,
        leverage=5.0,
        stop_loss_pct=10.0,
    )

    # With 2% risk of $10,000, confidence of 0.8, and leverage of 5x
    # Risk amount: $10,000 * 0.02 = $200
    # Confidence adjusted: $200 * (0.5 + 0.8*0.5) = $200 * 0.9 = $180
    # Position size with 10% stop loss at 5x leverage: approximately $900
    assert position_size > 0
    assert adjusted_leverage == 5.0  # Leverage should remain as requested

    # Test with maximum leverage exceeding limits
    position_size_max, adjusted_leverage_max = risk_manager.calculate_position_size(
        exchange=mock_connector,
        symbol="ETH",
        available_balance=10000.0,
        confidence=0.8,
        signal_side=OrderSide.BUY,
        leverage=20.0,  # Exceeds our max of 10.0
        stop_loss_pct=10.0,
    )

    assert adjusted_leverage_max == 10.0  # Should be capped at max_leverage

    # Test with provided price instead of fetching
    position_size_with_price, _ = risk_manager.calculate_position_size(
        exchange=mock_connector,
        symbol="ETH",
        available_balance=10000.0,
        confidence=0.8,
        signal_side=OrderSide.SELL,
        leverage=5.0,
        price=2000.0,  # Higher price than the one from get_ticker
        stop_loss_pct=10.0,
    )

    # Same calculation but with price=2000 instead of 1500
    # This would normally use the provided price directly
    assert position_size_with_price > 0

    # Test with max_position_size_usd constraint
    capped_risk_manager = RiskManager(
        max_account_risk_pct=2.0,
        max_leverage=10.0,
        max_position_size_usd=500.0,  # Cap at $500
    )

    position_size_capped, _ = capped_risk_manager.calculate_position_size(
        exchange=mock_connector,
        symbol="ETH",
        available_balance=10000.0,
        confidence=0.8,
        signal_side=OrderSide.BUY,
        leverage=5.0,
        stop_loss_pct=10.0,
    )

    assert position_size_capped <= 500.0  # Should be capped at max_position_size_usd


def test_calculate_hedge_parameters():
    """Test calculation of hedge position parameters."""
    risk_manager = RiskManager()

    # Test with standard parameters
    main_size = 1000.0
    main_leverage = 5.0
    hedge_ratio = 0.2

    hedge_size, hedge_leverage = risk_manager.calculate_hedge_parameters(
        main_position_size=main_size,
        main_leverage=main_leverage,
        hedge_ratio=hedge_ratio,
    )

    # Main notional: $1000 * 5 = $5000
    # Hedge notional: $5000 * 0.2 = $1000
    # Hedge size with same leverage: $1000 / 5 = $200
    assert hedge_size == 200.0
    assert hedge_leverage == 5.0

    # Test with custom hedge leverage
    hedge_size_custom, hedge_leverage_custom = risk_manager.calculate_hedge_parameters(
        main_position_size=main_size,
        main_leverage=main_leverage,
        hedge_ratio=hedge_ratio,
        max_hedge_leverage=3.0,  # Lower than main leverage
    )

    # Main notional: $1000 * 5 = $5000
    # Hedge notional: $5000 * 0.2 = $1000
    # Hedge size with 3x leverage: $1000 / 3 = $333.33
    assert abs(hedge_size_custom - 333.33) < 0.01
    assert hedge_leverage_custom == 3.0

    # Test with invalid hedge ratio
    (
        hedge_size_invalid,
        hedge_leverage_invalid,
    ) = risk_manager.calculate_hedge_parameters(
        main_position_size=main_size,
        main_leverage=main_leverage,
        hedge_ratio=-0.1,  # Invalid, should use default 0.2
    )

    # Should use default ratio of 0.2
    assert abs(hedge_size_invalid - 200.0) < 0.01
    assert hedge_leverage_invalid == 5.0


def test_validate_trade(mock_connector):
    """Test trade validation against risk parameters."""
    risk_manager = RiskManager(
        max_account_risk_pct=2.0, max_leverage=10.0, max_positions=2
    )

    # Initialize the risk manager with some existing positions
    risk_manager.positions = {
        "BTC": {"symbol": "BTC", "size": 1.0, "mark_price": 25000.0, "leverage": 5.0}
    }
    risk_manager.total_exposure = 25000.0  # BTC position notional value

    # Set up the mock connector
    mock_connector.get_account_balance.return_value = {"USD": 10000.0}
    mock_connector.get_ticker.return_value = {"symbol": "ETH", "last_price": 1500.0}
    mock_connector.get_markets.return_value = [
        {"symbol": "ETH", "min_size": 0.01, "base_asset": "ETH", "quote_asset": "USD"}
    ]

    # Test a valid trade
    is_valid, reason = risk_manager.validate_trade(
        exchange=mock_connector,
        symbol="ETH",
        position_size=1000.0,
        leverage=5.0,
        side=OrderSide.BUY,
    )

    assert is_valid is True
    assert reason == "Trade validated"

    # Test with position size exceeding available balance
    is_valid_exceed, reason_exceed = risk_manager.validate_trade(
        exchange=mock_connector,
        symbol="ETH",
        position_size=15000.0,  # Exceeds balance of 10000
        leverage=5.0,
        side=OrderSide.BUY,
    )

    assert is_valid_exceed is False
    assert "exceeds available balance" in reason_exceed

    # Test with leverage exceeding maximum
    is_valid_leverage, reason_leverage = risk_manager.validate_trade(
        exchange=mock_connector,
        symbol="ETH",
        position_size=1000.0,
        leverage=15.0,  # Exceeds max of 10.0
        side=OrderSide.BUY,
    )

    assert is_valid_leverage is False
    assert "exceeds maximum" in reason_leverage

    # Test with maximum positions reached
    risk_manager.positions["ETH"] = {
        "symbol": "ETH",
        "size": 1.0,
        "mark_price": 1500.0,
        "leverage": 5.0,
    }
    risk_manager.total_exposure += 1500.0

    is_valid_max, reason_max = risk_manager.validate_trade(
        exchange=mock_connector,
        symbol="SOL",  # New position would exceed max
        position_size=500.0,
        leverage=5.0,
        side=OrderSide.BUY,
    )

    assert is_valid_max is False
    assert "Max positions" in reason_max


def test_update_positions():
    """Test updating the internal positions tracking."""
    risk_manager = RiskManager()

    # Create sample positions
    positions = [
        {
            "symbol": "ETH",
            "size": 1.0,
            "mark_price": 1500.0,
            "leverage": 5.0,
            "margin": 300.0,
        },
        {
            "symbol": "BTC",
            "size": 0.1,
            "mark_price": 25000.0,
            "leverage": 10.0,
            "margin": 250.0,
        },
        {
            "symbol": "XRP",
            "size": 0.0,  # Zero size should be ignored
            "mark_price": 0.5,
            "leverage": 5.0,
            "margin": 0.0,
        },
    ]

    risk_manager.update_positions(positions)

    # Should have two positions (XRP ignored due to zero size)
    assert len(risk_manager.positions) == 2
    assert "ETH" in risk_manager.positions
    assert "BTC" in risk_manager.positions
    assert "XRP" not in risk_manager.positions

    # Total exposure should be sum of notional values
    # ETH: 1.0 * 1500.0 = 1500.0
    # BTC: 0.1 * 25000.0 = 2500.0
    # Total: 4000.0
    assert abs(risk_manager.total_exposure - 4000.0) < 0.01


def test_should_close_position():
    """Test criteria for closing positions."""
    risk_manager = RiskManager(min_margin_buffer_pct=20.0)

    # Test stop loss trigger
    long_position_loss = {
        "symbol": "ETH",
        "entry_price": 1500.0,
        "mark_price": 1350.0,  # 10% down
        "side": "LONG",
        "unrealized_pnl": -150.0,
        "margin": 300.0,  # 50% loss on margin
        "liquidation_price": 1200.0,
    }

    should_close, reason = risk_manager.should_close_position(
        long_position_loss, stop_loss_pct=10.0, take_profit_pct=20.0
    )

    assert should_close is True
    assert "Stop loss triggered" in reason

    # Test take profit trigger
    short_position_profit = {
        "symbol": "BTC",
        "entry_price": 25000.0,
        "mark_price": 20000.0,  # 20% down
        "side": "SHORT",
        "unrealized_pnl": 500.0,
        "margin": 250.0,  # 200% gain on margin
        "liquidation_price": 30000.0,
    }

    should_close, reason = risk_manager.should_close_position(
        short_position_profit, stop_loss_pct=10.0, take_profit_pct=20.0
    )

    assert should_close is True
    assert "Take profit triggered" in reason

    # Test close to liquidation (long position)
    long_position_liquidation = {
        "symbol": "ETH",
        "entry_price": 1500.0,
        "mark_price": 1300.0,
        "side": "LONG",
        "unrealized_pnl": -10.0,  # Low unrealized PnL to avoid triggering stop loss
        "margin": 300.0,
        "liquidation_price": 1250.0,  # Only 3.8% away from mark price
    }

    should_close, reason = risk_manager.should_close_position(
        long_position_liquidation, stop_loss_pct=20.0, take_profit_pct=20.0
    )

    assert should_close is True
    assert "liquidation" in reason.lower()

    # Test position that shouldn't be closed
    safe_position = {
        "symbol": "ETH",
        "entry_price": 1500.0,
        "mark_price": 1550.0,  # 3.3% up
        "side": "LONG",
        "unrealized_pnl": 50.0,
        "margin": 300.0,  # 16.7% gain on margin
        "liquidation_price": 1000.0,  # Far from liquidation
    }

    should_close, reason = risk_manager.should_close_position(
        safe_position, stop_loss_pct=10.0, take_profit_pct=20.0
    )

    assert should_close is False
    assert "within risk parameters" in reason


def test_manage_hedge_position():
    """Test hedge position management logic."""
    risk_manager = RiskManager()

    # Scenario 1: Main position profitable, hedge is losing
    main_position_profit = {
        "symbol": "ETH",
        "side": "LONG",
        "entry_price": 1400.0,
        "mark_price": 1600.0,
        "unrealized_pnl": 200.0,
        "margin": 700.0,  # About 28.6% gain
    }

    hedge_position_loss = {
        "symbol": "ETH",
        "side": "SHORT",
        "entry_price": 1400.0,
        "mark_price": 1600.0,
        "unrealized_pnl": -40.0,
        "margin": 200.0,  # About 20% loss
    }

    # Should reduce hedge since main is very profitable
    should_adjust, reason, adjustment = risk_manager.manage_hedge_position(
        main_position_profit, hedge_position_loss
    )

    assert should_adjust is True
    assert "reduce hedge" in reason.lower()
    assert adjustment["action"] == "reduce"
    assert adjustment["position"] == "hedge"

    # Scenario 2: Main position losing badly, hedge is profitable
    main_position_big_loss = {
        "symbol": "BTC",
        "side": "LONG",
        "entry_price": 25000.0,
        "mark_price": 20000.0,
        "unrealized_pnl": -500.0,
        "margin": 3000.0,  # About 16.7% loss
    }

    hedge_position_profit = {
        "symbol": "BTC",
        "side": "SHORT",
        "entry_price": 25000.0,
        "mark_price": 20000.0,
        "unrealized_pnl": 100.0,
        "margin": 500.0,  # 20% gain
    }

    # Should close main position since it's losing badly
    should_adjust, reason, adjustment = risk_manager.manage_hedge_position(
        main_position_big_loss, hedge_position_profit
    )

    assert should_adjust is True
    assert "close" in reason.lower()
    assert adjustment["action"] == "close"
    assert adjustment["position"] == "main"

    # Scenario 3: Both positions losing (rare)
    main_position_loss = {
        "symbol": "SOL",
        "side": "LONG",
        "entry_price": 100.0,
        "mark_price": 90.0,
        "unrealized_pnl": -60.0,  # Make the loss larger than 10%
        "margin": 500.0,  # 12% loss
    }

    hedge_position_also_loss = {
        "symbol": "SOL",
        "side": "SHORT",
        "entry_price": 80.0,
        "mark_price": 90.0,
        "unrealized_pnl": -20.0,
        "margin": 200.0,  # 10% loss
    }

    # Both losing significantly, should close both
    should_adjust, reason, adjustment = risk_manager.manage_hedge_position(
        main_position_loss, hedge_position_also_loss
    )

    assert should_adjust is True
    assert "close all" in reason.lower()
    assert adjustment["action"] == "close"
    assert adjustment["position"] == "both"

    # Scenario 4: Both positions doing fine
    main_position_ok = {
        "symbol": "ETH",
        "side": "LONG",
        "entry_price": 1500.0,
        "mark_price": 1520.0,
        "unrealized_pnl": 20.0,
        "margin": 500.0,  # 4% gain
    }

    hedge_position_small_loss = {
        "symbol": "ETH",
        "side": "SHORT",
        "entry_price": 1500.0,
        "mark_price": 1520.0,
        "unrealized_pnl": -4.0,
        "margin": 200.0,  # 2% loss
    }

    # No significant changes needed
    should_adjust, reason, adjustment = risk_manager.manage_hedge_position(
        main_position_ok, hedge_position_small_loss
    )

    assert should_adjust is False
    assert "No adjustment needed" in reason
