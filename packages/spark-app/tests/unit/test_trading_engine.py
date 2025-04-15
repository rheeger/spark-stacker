import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.connectors.base_connector import OrderSide, OrderType
from app.core.trading_engine import TradingEngine, TradingState
from app.indicators.base_indicator import Signal, SignalDirection


def test_trading_engine_initialization(mock_connector, mock_risk_manager):
    """Test trading engine initialization."""
    engine = TradingEngine(
        main_connector=mock_connector,
        hedge_connector=mock_connector,
        risk_manager=mock_risk_manager,
        dry_run=True,
        polling_interval=60,
        max_parallel_trades=1,
    )

    assert engine.main_connector == mock_connector
    assert engine.hedge_connector == mock_connector
    assert engine.risk_manager == mock_risk_manager
    assert engine.dry_run is True
    assert engine.polling_interval == 60
    assert engine.max_parallel_trades == 1
    assert engine.state == TradingState.IDLE
    assert engine.active_trades == {}
    assert engine.pending_signals == []


def test_start_stop_engine(trading_engine, mock_connector):
    """Test starting and stopping the trading engine."""
    # Configure mock
    mock_connector.is_connected = False

    # Test starting the engine
    result = trading_engine.start()
    assert result is True
    assert trading_engine.state == TradingState.RUNNING

    # Mock connector should have been connected
    mock_connector.connect.assert_called_once()

    # Monitor thread should be running
    assert trading_engine.monitor_thread is not None

    # Test stopping the engine
    result = trading_engine.stop()
    assert result is True
    assert trading_engine.state == TradingState.IDLE

    # Allow some time for the monitor thread to exit
    time.sleep(0.1)

    # Test starting when already running
    trading_engine.state = TradingState.RUNNING
    result = trading_engine.start()
    assert result is True  # Should return success without doing anything


def test_pause_resume_engine(trading_engine):
    """Test pausing and resuming the trading engine."""
    # Start the engine
    trading_engine.start()

    # Test pausing
    result = trading_engine.pause()
    assert result is True
    assert trading_engine.state == TradingState.PAUSED

    # Test resuming
    result = trading_engine.resume()
    assert result is True
    assert trading_engine.state == TradingState.RUNNING

    # Test invalid state transitions
    trading_engine.state = TradingState.IDLE
    result = trading_engine.pause()
    assert result is False  # Cannot pause when idle

    trading_engine.state = TradingState.RUNNING
    result = trading_engine.resume()
    assert result is False  # Cannot resume when already running

    # Clean up
    trading_engine.stop()


@pytest.mark.asyncio
async def test_process_signal(
    trading_engine, sample_signal, mock_risk_manager, mock_connector
):
    """Test processing a trading signal."""
    # Set up the mocks
    mock_risk_manager.calculate_position_size.return_value = (100.0, 5.0)
    mock_risk_manager.calculate_hedge_parameters.return_value = (20.0, 2.0)
    mock_risk_manager.validate_trade.return_value = (True, "Trade validated")

    mock_connector.get_ticker.return_value = {"symbol": "ETH", "last_price": 1500.0}
    mock_connector.place_order = AsyncMock(return_value={
        "status": "FILLED",
        "order_id": "test_order_123",
        "symbol": "ETH",
        "side": "BUY",
        "size": 100.0,
        "price": 1500.0,
        "timestamp": int(time.time() * 1000)
    })

    # Start the engine with dry_run=False to test actual order placement
    trading_engine.dry_run = False
    trading_engine.start()

    # Process a buy signal
    result = await trading_engine.process_signal(sample_signal)
    assert result is True, "Signal processing should succeed"

    # Check that orders were placed
    assert mock_connector.place_order.call_count == 2  # Main and hedge orders

    # Verify the order parameters
    main_call = mock_connector.place_order.call_args_list[0]
    hedge_call = mock_connector.place_order.call_args_list[1]

    assert main_call.kwargs["side"] == OrderSide.BUY
    assert main_call.kwargs["amount"] == 100.0
    assert hedge_call.kwargs["side"] == OrderSide.SELL
    assert hedge_call.kwargs["amount"] == 20.0

    # Check that an active trade was created
    assert "ETH" in trading_engine.active_trades
    trade = trading_engine.active_trades["ETH"]
    assert trade["symbol"] == "ETH"
    assert trade["main_position"]["side"] == "BUY"
    assert trade["hedge_position"]["side"] == "SELL"
    assert trade["status"] == "open"

    # Test with a neutral signal (should be ignored)
    neutral_signal = Signal(
        direction=SignalDirection.NEUTRAL,
        symbol="ETH",
        indicator="test_indicator",
        confidence=0.5,
    )

    result = await trading_engine.process_signal(neutral_signal)
    assert result is False  # Neutral signals should be ignored

    # Test with engine in incorrect state
    trading_engine.state = TradingState.PAUSED
    result = await trading_engine.process_signal(sample_signal)
    assert result is False  # Should queue the signal instead of processing
    assert len(trading_engine.pending_signals) == 1

    # Test with maximum trades reached
    trading_engine.state = TradingState.RUNNING
    trading_engine.active_trades = {"ETH": {}, "BTC": {}}
    trading_engine.max_parallel_trades = 2

    new_signal = Signal(
        direction=SignalDirection.BUY,
        symbol="SOL",
        indicator="test_indicator",
        confidence=0.8,
    )

    result = await trading_engine.process_signal(new_signal)
    assert result is False  # Should queue the signal
    assert len(trading_engine.pending_signals) == 2

    # Clean up
    trading_engine.active_trades = {}
    trading_engine.pending_signals = []
    trading_engine.stop()


def test_execute_hedged_trade(trading_engine, mock_risk_manager, mock_connector):
    """Test the execution of a hedged trade."""
    # Set up the mocks
    mock_risk_manager.calculate_position_size.return_value = (100.0, 5.0)
    mock_risk_manager.calculate_hedge_parameters.return_value = (20.0, 2.0)
    mock_risk_manager.validate_trade.return_value = (True, "Trade validated")

    mock_connector.get_account_balance.return_value = {"USD": 10000.0}
    mock_connector.get_ticker.return_value = {"symbol": "ETH", "last_price": 1500.0}

    # Start the engine
    trading_engine.start()

    # Execute a hedged trade in dry run mode
    result = trading_engine._execute_hedged_trade(
        symbol="ETH",
        main_side=OrderSide.BUY,
        hedge_side=OrderSide.SELL,
        confidence=0.8,
        main_leverage=5.0,
        hedge_leverage=2.0,
        hedge_ratio=0.2,
        stop_loss_pct=10.0,
        take_profit_pct=20.0,
    )

    assert result is True

    # Verify interactions with risk manager
    mock_risk_manager.calculate_position_size.assert_called_once()
    mock_risk_manager.calculate_hedge_parameters.assert_called_once()
    mock_risk_manager.validate_trade.assert_called_once()

    # Verify active trade creation
    assert "ETH" in trading_engine.active_trades
    trade = trading_engine.active_trades["ETH"]
    assert trade["main_position"]["side"] == "BUY"
    assert trade["main_position"]["size"] == 100.0
    assert trade["main_position"]["leverage"] == 5.0
    assert trade["hedge_position"]["side"] == "SELL"
    assert trade["hedge_position"]["size"] == 20.0
    assert trade["hedge_position"]["leverage"] == 2.0

    # Test with trade validation failure
    mock_risk_manager.validate_trade.return_value = (False, "Invalid trade")

    result = trading_engine._execute_hedged_trade(
        symbol="BTC", main_side=OrderSide.SELL, hedge_side=OrderSide.BUY, confidence=0.7
    )

    assert result is False

    # Test with insufficient balance
    mock_connector.get_account_balance.return_value = {"USD": 0.0}
    mock_risk_manager.validate_trade.return_value = (True, "Trade validated")

    result = trading_engine._execute_hedged_trade(
        symbol="BTC", main_side=OrderSide.SELL, hedge_side=OrderSide.BUY, confidence=0.7
    )

    assert result is False

    # Clean up
    trading_engine.active_trades = {}
    trading_engine.stop()


def test_check_active_trades(trading_engine, mock_connector):
    """Test checking and managing active trades."""
    # Create a mock position
    eth_position = {
        "symbol": "ETH",
        "side": "LONG",
        "entry_price": 1500.0,
        "mark_price": 1600.0,  # In profit
        "size": 1.0,
        "leverage": 5.0,
        "unrealized_pnl": 100.0,
        "margin": 300.0,
        "liquidation_price": 1200.0,
    }

    # Create a mock trade record
    trading_engine.active_trades = {
        "ETH": {
            "symbol": "ETH",
            "main_position": {
                "side": "BUY",
                "size": 1.0,
                "leverage": 5.0,
                "entry_price": 1500.0,
            },
            "hedge_position": {
                "side": "SELL",
                "size": 0.2,
                "leverage": 2.0,
                "entry_price": 1500.0,
            },
            "stop_loss_pct": 10.0,
            "take_profit_pct": 20.0,
            "status": "open",
        }
    }

    # Call the check active trades method
    main_positions = [eth_position]
    hedge_positions = []  # No hedge position found (could be closed)

    trading_engine._check_active_trades(main_positions, hedge_positions)

    # The trade should be updated with the current position details
    updated_trade = trading_engine.active_trades["ETH"]
    assert updated_trade["main_position"]["mark_price"] == 1600.0
    assert updated_trade["main_position"]["unrealized_pnl"] == 100.0

    # Test removing closed positions
    trading_engine.active_trades = {
        "ETH": {
            "symbol": "ETH",
            "main_position": {},
            "hedge_position": {},
            "status": "open",
        }
    }

    # Both positions are closed (not in the positions lists)
    trading_engine._check_active_trades([], [])

    # The trade should be removed
    assert "ETH" not in trading_engine.active_trades

    # Clean up
    trading_engine.stop()


def test_get_active_trades_and_history(trading_engine):
    """Test getting active trades and trade history."""
    # Add some active trades
    trading_engine.active_trades = {
        "ETH": {"symbol": "ETH", "status": "open"},
        "BTC": {"symbol": "BTC", "status": "closing"},
    }

    # Get active trades
    trades = trading_engine.get_active_trades()
    assert len(trades) == 2
    assert "ETH" in trades
    assert "BTC" in trades

    # Get trade history (currently just returns an empty list)
    history = trading_engine.get_trade_history()
    assert isinstance(history, list)
    assert len(history) == 0  # No history implementation yet

    # Clean up
    trading_engine.active_trades = {}
    trading_engine.stop()


def test_close_all_positions(trading_engine, mock_connector):
    """Test closing all positions."""
    # Add some active trades
    trading_engine.active_trades = {
        "ETH": {"symbol": "ETH", "status": "open"},
        "BTC": {"symbol": "BTC", "status": "open"},
    }

    # Mock get_positions to return some positions
    mock_connector.get_positions.return_value = [
        {"symbol": "ETH", "size": 1.0},
        {"symbol": "BTC", "size": 0.1},
    ]

    # Test in dry run mode
    result = trading_engine.close_all_positions()
    assert result is True
    assert len(trading_engine.active_trades) == 0  # Should clear trades
    assert mock_connector.close_position.call_count == 0  # No actual close in dry run

    # Test in live mode
    trading_engine.dry_run = False
    trading_engine.active_trades = {
        "ETH": {"symbol": "ETH", "status": "open"},
        "BTC": {"symbol": "BTC", "status": "open"},
    }

    result = trading_engine.close_all_positions()
    assert result is True
    assert mock_connector.close_position.call_count == 2  # Should close both positions

    # Test with error in closing
    mock_connector.close_position.reset_mock()
    mock_connector.close_position.side_effect = Exception("Test error")

    # Reset the active trades
    trading_engine.active_trades = {
        "ETH": {"symbol": "ETH", "status": "open"},
        "BTC": {"symbol": "BTC", "status": "open"},
    }

    result = trading_engine.close_all_positions()
    assert result is False  # Should return failure due to error

    # Clean up
    mock_connector.close_position.side_effect = None
    trading_engine.dry_run = True
    trading_engine.active_trades = {}
    trading_engine.stop()
