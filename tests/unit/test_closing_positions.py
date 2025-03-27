from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.connectors.base_connector import MarketType, OrderSide, OrderType
from app.core.trading_engine import TradingEngine
from app.risk_management.risk_manager import RiskManager


class TestClosingPositions:
    """Test cases specifically for closing positions functionality."""

    def setup_method(self):
        """Setup method that runs before each test."""
        # Create mock connectors
        self.mock_main_connector = self._create_mock_connector("hyperliquid")
        self.mock_hedge_connector = self._create_mock_connector("coinbase")

        # Create risk manager
        self.risk_manager = MagicMock(spec=RiskManager)
        self.risk_manager.calculate_position_size.return_value = (100.0, 5.0)  # (size, leverage)
        self.risk_manager.calculate_hedge_parameters.return_value = (20.0, 2.0)  # (size, leverage)
        self.risk_manager.validate_trade.return_value = (True, "Trade validated")

        # Create trading engine
        self.engine = TradingEngine(
            main_connector=self.mock_main_connector,
            hedge_connector=self.mock_hedge_connector,
            risk_manager=self.risk_manager,
            dry_run=False,  # Set to False to ensure actual execution
            polling_interval=1,
            max_parallel_trades=1,
        )

    def _create_mock_connector(self, name):
        """Create a mock connector with necessary methods for testing."""
        connector = MagicMock()
        connector.name = name
        connector.supports_derivatives = True
        connector.is_connected = True

        # Define market data
        connector.get_ticker.return_value = {
            "symbol": "ETH-USD",
            "last_price": 2000.0,
            "bid": 1995.0,
            "ask": 2005.0,
            "volume": 1000.0,
        }

        # Define account data
        connector.get_account_balance.return_value = {"USD": 10000.0, "ETH": 5.0}

        # Define market types
        connector.market_types = MarketType.PERPETUAL if name == "hyperliquid" else MarketType.SPOT

        # Define position data - initially empty
        connector.get_positions.return_value = []

        # Setup place_order to simulate successful order execution
        connector.place_order.return_value = {
            "order_id": f"{name}_order_123",
            "symbol": "ETH-USD",
            "side": "BUY",
            "size": 1.0,
            "price": 2000.0,
            "status": "FILLED",
        }

        # Setup close_position to simulate successful position closing
        connector.close_position.return_value = {
            "order_id": f"{name}_close_123",
            "symbol": "ETH-USD",
            "side": "SELL",  # Opposite of initial position
            "size": 1.0,
            "price": 2000.0,
            "status": "FILLED",
        }

        return connector

    def test_open_and_close_position(self):
        """Test opening and closing a position works correctly."""
        symbol = "ETH-USD"

        # Setup mock positions after opening a position
        self.mock_main_connector.get_positions.side_effect = [
            [],  # First call returns empty (before position is opened)
            [{   # Second call returns the open position
                "symbol": symbol,
                "side": "LONG",
                "size": 1.0,
                "entry_price": 2000.0,
                "leverage": 5.0,
                "unrealized_pnl": 0.0,
            }],
            []   # Third call returns empty again (after position is closed)
        ]

        # Setup mock positions for hedge connector
        self.mock_hedge_connector.get_positions.side_effect = [
            [],  # First call returns empty (before position is opened)
            [{   # Second call returns the open position (hedge position is opposite)
                "symbol": symbol,
                "side": "SHORT",
                "size": 0.2,
                "entry_price": 2000.0,
                "leverage": 2.0,
                "unrealized_pnl": 0.0,
            }],
            []   # Third call returns empty again (after position is closed)
        ]

        # Step 1: Open a position using place_order directly
        self.engine.active_trades[symbol] = {
            "symbol": symbol,
            "main_position": {
                "side": "LONG",
                "size": 1.0,
                "entry_price": 2000.0,
                "leverage": 5.0,
            },
            "hedge_position": {
                "side": "SHORT",
                "size": 0.2,
                "entry_price": 2000.0,
                "leverage": 2.0,
            }
        }

        # Step 2: Get active trades to verify
        active_trades = self.engine.get_active_trades()
        assert symbol in active_trades
        assert active_trades[symbol]["main_position"]["side"] == "LONG"

        # Step 3: Close the position
        close_result = self.engine.close_all_positions()

        # Verify close result
        assert close_result is True

        # Check if close_position was called correctly on both connectors
        assert self.mock_main_connector.close_position.called
        main_close_args = self.mock_main_connector.close_position.call_args[0]
        assert main_close_args[0] == symbol  # First argument should be symbol

        assert self.mock_hedge_connector.close_position.called
        hedge_close_args = self.mock_hedge_connector.close_position.call_args[0]
        assert hedge_close_args[0] == symbol  # First argument should be symbol

        # Step 4: Verify positions are closed
        assert len(self.engine.get_active_trades()) == 0

    def test_close_all_positions_with_spot_market(self):
        """Test closing all positions in a spot market."""
        symbol = "ETH-USD"

        # Change market type to spot
        self.mock_main_connector.market_types = MarketType.SPOT
        self.mock_main_connector.supports_derivatives = False

        # Add a BUY position to active trades
        self.engine.active_trades[symbol] = {
            "symbol": symbol,
            "market_type": "SPOT",
            "main_position": {
                "side": "BUY",
                "size": 1.0,
                "entry_price": 2000.0,
            }
        }

        # Setup account balance to return ETH balance
        self.mock_main_connector.get_account_balance.return_value = {"USD": 10000.0, "ETH": 1.0}

        # Close all positions
        close_result = self.engine.close_all_positions()

        # Verify close result
        assert close_result is True

        # In spot markets with BUY positions, we should place a SELL order
        assert self.mock_main_connector.place_order.called
        place_order_args, place_order_kwargs = self.mock_main_connector.place_order.call_args
        assert place_order_kwargs["symbol"] == symbol
        assert place_order_kwargs["side"] == OrderSide.SELL
        assert place_order_kwargs["order_type"] == OrderType.MARKET

        # Verify all positions are cleared
        assert len(self.engine.get_active_trades()) == 0

    def test_close_all_positions_with_no_positions(self):
        """Test closing all positions when there are none."""
        # Ensure active_trades is empty
        self.engine.active_trades.clear()

        # Close all positions
        close_result = self.engine.close_all_positions()

        # Verify close result
        assert close_result is True

        # Verify no calls to connectors were made
        assert not self.mock_main_connector.close_position.called
        assert not self.mock_hedge_connector.close_position.called
        assert not self.mock_main_connector.place_order.called
        assert not self.mock_hedge_connector.place_order.called

    def test_error_handling_during_position_closing(self):
        """Test error handling when closing positions fails."""
        symbol = "ETH-USD"

        # Add a position to active trades
        self.engine.active_trades[symbol] = {
            "symbol": symbol,
            "main_position": {
                "side": "LONG",
                "size": 1.0,
                "entry_price": 2000.0,
                "leverage": 5.0,
            }
        }

        # Make close_position fail
        self.mock_main_connector.close_position.side_effect = Exception("Simulated error")

        # Close all positions
        close_result = self.engine.close_all_positions()

        # Even though close_position failed, the method returns False but active_trades is cleared
        assert close_result is False
        assert len(self.engine.get_active_trades()) == 0

        # Verify close_position was called
        assert self.mock_main_connector.close_position.called
