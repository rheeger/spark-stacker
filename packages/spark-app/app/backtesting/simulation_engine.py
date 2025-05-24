import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..connectors.base_connector import OrderSide, OrderStatus, OrderType

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimulatedOrder:
    """Represents a simulated order in the backtesting system."""

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: float,
        price: Optional[float] = None,
        leverage: float = 1.0,
        status: OrderStatus = OrderStatus.OPEN,
        timestamp: Optional[int] = None,
        fees: float = 0.0,
        slippage: float = 0.0,
    ):
        """
        Initialize a simulated order.

        Args:
            symbol: Market symbol (e.g., 'ETH-USD')
            side: Buy or sell
            order_type: Market or limit
            amount: Order amount
            price: Order price (required for limit orders)
            leverage: Leverage multiplier
            status: Order status
            timestamp: Order timestamp in milliseconds
            fees: Fees paid for the order
            slippage: Slippage percentage applied to the order
        """
        self.order_id = str(uuid.uuid4())
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.amount = amount
        self.price = price
        self.leverage = leverage
        self.status = status
        self.timestamp = timestamp or int(datetime.now().timestamp() * 1000)
        self.fees = fees
        self.slippage = slippage
        self.filled_price = None
        self.filled_amount = 0.0
        self.fill_timestamp = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "amount": self.amount,
            "price": self.price,
            "leverage": self.leverage,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "fees": self.fees,
            "slippage": self.slippage,
            "filled_price": self.filled_price,
            "filled_amount": self.filled_amount,
            "fill_timestamp": self.fill_timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulatedOrder":
        """Create an order from dictionary."""
        order = cls(
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["order_type"]),
            amount=data["amount"],
            price=data.get("price"),
            leverage=data.get("leverage", 1.0),
            status=OrderStatus(data["status"]),
            timestamp=data.get("timestamp"),
        )
        order.order_id = data["order_id"]
        order.fees = data.get("fees", 0.0)
        order.slippage = data.get("slippage", 0.0)
        order.filled_price = data.get("filled_price")
        order.filled_amount = data.get("filled_amount", 0.0)
        order.fill_timestamp = data.get("fill_timestamp")
        return order


class SimulatedPosition:
    """Represents a simulated trading position."""

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        entry_price: float,
        amount: float,
        leverage: float = 1.0,
        timestamp: Optional[int] = None,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
    ):
        """
        Initialize a simulated position.

        Args:
            symbol: Market symbol (e.g., 'ETH-USD')
            side: Long (BUY) or short (SELL)
            entry_price: Average entry price
            amount: Position size
            leverage: Leverage multiplier
            timestamp: Position open timestamp
            unrealized_pnl: Unrealized profit/loss
            realized_pnl: Realized profit/loss
        """
        self.position_id = str(uuid.uuid4())
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.amount = amount
        self.leverage = leverage
        self.timestamp = timestamp or int(datetime.now().timestamp() * 1000)
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl
        self.liquidation_price = self._calculate_liquidation_price()

    def _calculate_liquidation_price(self) -> float:
        """Calculate liquidation price based on leverage."""
        if self.leverage <= 1:
            return 0.0 if self.side == OrderSide.BUY else float("inf")

        # Basic liquidation price calculation
        maintenance_margin = 0.05  # 5% maintenance margin requirement
        if self.side == OrderSide.BUY:
            return self.entry_price * (1 - (1 / self.leverage) + maintenance_margin)
        else:
            return self.entry_price * (1 + (1 / self.leverage) - maintenance_margin)

    def update_unrealized_pnl(self, current_price: float) -> float:
        """
        Update and return unrealized PnL based on current price.

        Args:
            current_price: Current market price

        Returns:
            Updated unrealized PnL
        """
        if self.side == OrderSide.BUY:
            price_diff = current_price - self.entry_price
        else:
            price_diff = self.entry_price - current_price

        self.unrealized_pnl = price_diff * self.amount * self.leverage
        return self.unrealized_pnl

    def is_liquidated(self, current_price: float) -> bool:
        """
        Check if position would be liquidated at current price.

        Args:
            current_price: Current market price

        Returns:
            True if position would be liquidated
        """
        if self.leverage <= 1:
            return False

        if self.side == OrderSide.BUY:
            return current_price <= self.liquidation_price
        else:
            return current_price >= self.liquidation_price

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "amount": self.amount,
            "leverage": self.leverage,
            "timestamp": self.timestamp,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "liquidation_price": self.liquidation_price,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulatedPosition":
        """Create a position from dictionary."""
        position = cls(
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            entry_price=data["entry_price"],
            amount=data["amount"],
            leverage=data.get("leverage", 1.0),
            timestamp=data.get("timestamp"),
            unrealized_pnl=data.get("unrealized_pnl", 0.0),
            realized_pnl=data.get("realized_pnl", 0.0),
        )
        position.position_id = data["position_id"]
        position.liquidation_price = data.get(
            "liquidation_price", position.liquidation_price
        )
        return position


class SimulationEngine:
    """
    Engine for simulating trades in backtesting.

    Handles order execution with realistic fees and slippage.
    """

    def __init__(
        self,
        initial_balance: Dict[str, float],
        maker_fee: float = 0.0001,  # 0.01%
        taker_fee: float = 0.0005,  # 0.05%
        slippage_model: str = "random",
    ):
        """
        Initialize the simulation engine.

        Args:
            initial_balance: Starting balances for each asset
            maker_fee: Fee for maker orders (limit orders that don't cross the spread)
            taker_fee: Fee for taker orders (market orders or limit orders that cross the spread)
            slippage_model: Slippage model to use ('fixed', 'random', 'orderbook')
        """
        self.balance = initial_balance.copy()
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_model = slippage_model

        self.orders: List[SimulatedOrder] = []
        self.positions: Dict[str, SimulatedPosition] = {}  # symbol -> position
        self.trade_history: List[Dict[str, Any]] = []

        self.last_prices: Dict[str, float] = {}

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: float,
        price: Optional[float] = None,
        leverage: float = 1.0,
        timestamp: int = None,
        current_candle: Optional[pd.Series] = None,
    ) -> SimulatedOrder:
        """
        Place a simulated order.

        Args:
            symbol: Market symbol (e.g., 'ETH-USD')
            side: Buy or sell
            order_type: Market or limit
            amount: Order amount
            price: Order price (required for limit orders)
            leverage: Leverage multiplier
            timestamp: Order timestamp
            current_candle: Current price candle data

        Returns:
            SimulatedOrder object
        """
        # Validate inputs
        if order_type == OrderType.LIMIT and price is None:
            raise ValueError("Price must be specified for limit orders")

        # Create order
        order = SimulatedOrder(
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
            leverage=leverage,
            timestamp=timestamp,
        )

        # Add to orders list
        self.orders.append(order)

        # If we have current candle data, try to execute the order immediately
        if current_candle is not None:
            self.execute_order(order, current_candle)

        return order

    def execute_order(self, order: SimulatedOrder, candle: pd.Series) -> OrderStatus:
        """
        Attempt to execute a simulated order.

        Args:
            order: The order to execute
            candle: Price candle data

        Returns:
            New order status
        """
        # Skip already filled or canceled orders
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED]:
            return order.status

        # For market orders, always execute
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order, candle)

        # For limit orders, check if price conditions are met
        elif order.order_type == OrderType.LIMIT:
            return self._execute_limit_order(order, candle)

        return order.status

    def _execute_market_order(
        self, order: SimulatedOrder, candle: pd.Series
    ) -> OrderStatus:
        """Execute a market order."""
        # Calculate execution price with slippage
        price = self._calculate_execution_price(
            order.side, order.order_type, candle, order.amount
        )

        # Calculate fees
        fee_rate = self.taker_fee
        fee_amount = price * order.amount * fee_rate

        # Apply slippage based on model
        if self.slippage_model == "fixed":
            slippage = 0.001  # 0.1%
        elif self.slippage_model == "random":
            # Random slippage between 0 and 0.2%
            slippage = np.random.uniform(0, 0.002)
        else:  # orderbook model - more complex simulation
            slippage = 0.0005  # Simplified for now

        # Apply slippage to price
        if order.side == OrderSide.BUY:
            execution_price = price * (1 + slippage)
        else:
            execution_price = price * (1 - slippage)

        # Update order
        order.filled_price = execution_price
        order.filled_amount = order.amount
        order.fill_timestamp = candle.get("timestamp", order.timestamp)
        order.fees = fee_amount
        order.slippage = slippage
        order.status = OrderStatus.FILLED

        # Update position
        self._update_position(order)

        # Update last known price
        self.last_prices[order.symbol] = execution_price

        # Record trade
        self._record_trade(order)

        return OrderStatus.FILLED

    def _execute_limit_order(
        self, order: SimulatedOrder, candle: pd.Series
    ) -> OrderStatus:
        """Execute a limit order if price conditions are met."""
        # Check if limit price is reached
        if order.side == OrderSide.BUY:
            # For buy orders, check if low price <= limit price
            if candle["low"] <= order.price:
                execution_price = min(candle["open"], order.price)

                # Calculate fees (maker fee for limit orders)
                fee_rate = self.maker_fee
                fee_amount = execution_price * order.amount * fee_rate

                # Update order
                order.filled_price = execution_price
                order.filled_amount = order.amount
                order.fill_timestamp = candle.get("timestamp", order.timestamp)
                order.fees = fee_amount
                order.slippage = 0.0  # No slippage for limit orders
                order.status = OrderStatus.FILLED

                # Update position
                self._update_position(order)

                # Update last known price
                self.last_prices[order.symbol] = execution_price

                # Record trade
                self._record_trade(order)

                return OrderStatus.FILLED

        elif order.side == OrderSide.SELL:
            # For sell orders, check if high price >= limit price
            if candle["high"] >= order.price:
                execution_price = max(candle["open"], order.price)

                # Calculate fees (maker fee for limit orders)
                fee_rate = self.maker_fee
                fee_amount = execution_price * order.amount * fee_rate

                # Update order
                order.filled_price = execution_price
                order.filled_amount = order.amount
                order.fill_timestamp = candle.get("timestamp", order.timestamp)
                order.fees = fee_amount
                order.slippage = 0.0  # No slippage for limit orders
                order.status = OrderStatus.FILLED

                # Update position
                self._update_position(order)

                # Update last known price
                self.last_prices[order.symbol] = execution_price

                # Record trade
                self._record_trade(order)

                return OrderStatus.FILLED

        return order.status

    def _calculate_execution_price(
        self, side: OrderSide, order_type: OrderType, candle: pd.Series, amount: float
    ) -> float:
        """
        Calculate execution price for an order.

        Args:
            side: Order side
            order_type: Order type
            candle: Price candle
            amount: Order amount

        Returns:
            Execution price
        """
        # For market orders, use open price of candle
        if order_type == OrderType.MARKET:
            return candle["open"]

        # For limit orders, use the specified price
        return candle[
            "open"
        ]  # Placeholder - actual limit price is checked in _execute_limit_order

    def _update_position(self, order: SimulatedOrder) -> None:
        """
        Update positions after order execution.

        Args:
            order: Executed order
        """
        symbol = order.symbol

        # Get the base and quote currencies - handle both dash and underscore delimiters
        if "-" in symbol:
            base_currency = symbol.split("-")[0]
            quote_currency = symbol.split("-")[1]
        elif "_" in symbol:
            base_currency = symbol.split("_")[0]
            quote_currency = symbol.split("_")[1]
        else:
            # Fallback - assume it's a single asset like BTC
            base_currency = symbol
            quote_currency = "USD"

        # Calculate order value
        order_value = order.filled_price * order.filled_amount

        # Update balance for spot markets (leverage = 1)
        if order.leverage == 1:
            if order.side == OrderSide.BUY:
                # Deduct quote currency, add base currency
                if quote_currency in self.balance:
                    self.balance[quote_currency] -= order_value + order.fees
                    if self.balance[quote_currency] < 0:
                        logger.warning(
                            f"Negative balance for {quote_currency}: {self.balance[quote_currency]}"
                        )

                # Add base currency
                if base_currency not in self.balance:
                    self.balance[base_currency] = 0
                self.balance[base_currency] += order.filled_amount

            else:  # SELL
                # Add quote currency, deduct base currency
                if base_currency in self.balance:
                    self.balance[base_currency] -= order.filled_amount
                    if self.balance[base_currency] < 0:
                        logger.warning(
                            f"Negative balance for {base_currency}: {self.balance[base_currency]}"
                        )

                # Add quote currency
                if quote_currency not in self.balance:
                    self.balance[quote_currency] = 0
                self.balance[quote_currency] += order_value - order.fees

        # Update positions for leveraged trading
        # Check if we already have a position for this symbol
        if symbol in self.positions:
            existing_position = self.positions[symbol]

            # If order is in the same direction as the position, increase position
            if order.side == existing_position.side:
                # Calculate new average entry price
                total_amount = existing_position.amount + order.filled_amount
                new_entry_price = (
                    (existing_position.entry_price * existing_position.amount)
                    + (order.filled_price * order.filled_amount)
                ) / total_amount

                # Update position
                existing_position.entry_price = new_entry_price
                existing_position.amount += order.filled_amount
                existing_position.leverage = max(
                    existing_position.leverage, order.leverage
                )
                existing_position.liquidation_price = (
                    existing_position._calculate_liquidation_price()
                )

            # If order is in the opposite direction, reduce or flip position
            else:
                # Calculate resulting position amount
                net_amount = existing_position.amount - order.filled_amount

                # If net amount is positive, reduce position
                if net_amount > 0:
                    # Calculate realized PnL
                    if existing_position.side == OrderSide.BUY:
                        price_diff = order.filled_price - existing_position.entry_price
                    else:
                        price_diff = existing_position.entry_price - order.filled_price

                    realized_pnl = (
                        price_diff * order.filled_amount * existing_position.leverage
                    )
                    existing_position.realized_pnl += realized_pnl

                    # Add realized PnL to quote currency balance
                    if quote_currency not in self.balance:
                        self.balance[quote_currency] = 0
                    self.balance[quote_currency] += realized_pnl - order.fees

                    # Update position size
                    existing_position.amount = net_amount

                # If net amount is zero, close position
                elif net_amount == 0:
                    # Calculate realized PnL
                    if existing_position.side == OrderSide.BUY:
                        price_diff = order.filled_price - existing_position.entry_price
                    else:
                        price_diff = existing_position.entry_price - order.filled_price

                    realized_pnl = (
                        price_diff
                        * existing_position.amount
                        * existing_position.leverage
                    )

                    # Add realized PnL to quote currency balance
                    if quote_currency not in self.balance:
                        self.balance[quote_currency] = 0
                    self.balance[quote_currency] += realized_pnl - order.fees

                    # Remove position
                    del self.positions[symbol]

                # If net amount is negative, flip position
                else:
                    # Calculate realized PnL for closing existing position
                    if existing_position.side == OrderSide.BUY:
                        price_diff = order.filled_price - existing_position.entry_price
                    else:
                        price_diff = existing_position.entry_price - order.filled_price

                    realized_pnl = (
                        price_diff
                        * existing_position.amount
                        * existing_position.leverage
                    )

                    # Add realized PnL to quote currency balance
                    if quote_currency not in self.balance:
                        self.balance[quote_currency] = 0
                    self.balance[quote_currency] += realized_pnl - order.fees

                    # Create new position in opposite direction
                    new_position = SimulatedPosition(
                        symbol=symbol,
                        side=order.side,
                        entry_price=order.filled_price,
                        amount=abs(net_amount),
                        leverage=order.leverage,
                        timestamp=order.fill_timestamp,
                    )

                    self.positions[symbol] = new_position

        # If no existing position, create a new one
        else:
            # Create new position
            new_position = SimulatedPosition(
                symbol=symbol,
                side=order.side,
                entry_price=order.filled_price,
                amount=order.filled_amount,
                leverage=order.leverage,
                timestamp=order.fill_timestamp,
            )

            self.positions[symbol] = new_position

    def _record_trade(self, order: SimulatedOrder) -> None:
        """
        Record a trade in the trade history.

        Args:
            order: Executed order
        """
        trade = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.order_type.value,
            "amount": order.filled_amount,
            "price": order.filled_price,
            "timestamp": order.fill_timestamp,
            "fees": order.fees,
            "leverage": order.leverage,
        }

        self.trade_history.append(trade)

    def update_positions(self, candle_data: Dict[str, pd.Series]) -> None:
        """
        Update positions with current market data.

        Args:
            candle_data: Dictionary of symbol -> candle data
        """
        # Update positions with new market data
        symbols_to_remove = []

        for symbol, position in self.positions.items():
            # Skip if we don't have data for this symbol
            if symbol not in candle_data:
                continue

            candle = candle_data[symbol]
            current_price = candle["close"]

            # Update last known price
            self.last_prices[symbol] = current_price

            # Check for liquidation
            if position.is_liquidated(current_price):
                logger.warning(
                    f"Position {position.position_id} liquidated at {current_price}"
                )

                # Calculate realized PnL (total loss in this case)
                realized_pnl = -1 * (
                    position.entry_price * position.amount / position.leverage
                )

                # Record liquidation in trade history
                liquidation_trade = {
                    "order_id": "liquidation_" + position.position_id,
                    "symbol": symbol,
                    "side": "LIQUIDATION",
                    "type": "MARKET",
                    "amount": position.amount,
                    "price": position.liquidation_price,
                    "timestamp": candle.get(
                        "timestamp", int(datetime.now().timestamp() * 1000)
                    ),
                    "fees": 0,
                    "leverage": position.leverage,
                    "realized_pnl": realized_pnl,
                }

                self.trade_history.append(liquidation_trade)

                # Mark position for removal
                symbols_to_remove.append(symbol)

            else:
                # Update unrealized PnL
                position.update_unrealized_pnl(current_price)

        # Remove liquidated positions
        for symbol in symbols_to_remove:
            del self.positions[symbol]

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            True if order was canceled, False otherwise
        """
        for order in self.orders:
            if order.order_id == order_id and order.status == OrderStatus.OPEN:
                order.status = OrderStatus.CANCELED
                return True

        return False

    def get_orders(self, symbol: Optional[str] = None) -> List[SimulatedOrder]:
        """
        Get all orders, optionally filtered by symbol.

        Args:
            symbol: Optional symbol to filter by

        Returns:
            List of orders
        """
        if symbol:
            return [order for order in self.orders if order.symbol == symbol]
        else:
            return self.orders.copy()

    def get_positions(self, symbol: Optional[str] = None) -> List[SimulatedPosition]:
        """
        Get all positions, optionally filtered by symbol.

        Args:
            symbol: Optional symbol to filter by

        Returns:
            List of positions
        """
        if symbol:
            return [self.positions[symbol]] if symbol in self.positions else []
        else:
            return list(self.positions.values())

    def get_balance(
        self, asset: Optional[str] = None
    ) -> Union[Dict[str, float], float]:
        """
        Get account balance, optionally for a specific asset.

        Args:
            asset: Optional asset to get balance for

        Returns:
            Balance dictionary or specific asset balance
        """
        if asset:
            return self.balance.get(asset, 0.0)
        else:
            return self.balance.copy()

    def calculate_equity(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total equity including unrealized PnL.

        Args:
            current_prices: Dictionary of symbol -> current price

        Returns:
            Total equity value in USD
        """
        equity = 0.0

        # Convert all asset balances to USD value
        for asset, balance in self.balance.items():
            if balance == 0:
                continue

            if asset == "USD" or asset == "USDT" or asset == "USDC":
                # Already in USD-equivalent
                equity += balance
            else:
                # Convert asset to USD using current price
                # Look for a symbol that matches this asset
                asset_price = None
                for symbol, price in current_prices.items():
                    # Check both dash and underscore separators
                    if "-" in symbol:
                        base_currency, quote_currency = symbol.split("-")
                    elif "_" in symbol:
                        base_currency, quote_currency = symbol.split("_")
                    else:
                        continue

                    # If this symbol's base currency matches our asset and quotes in USD
                    if (base_currency == asset and
                        quote_currency in ["USD", "USDT", "USDC"]):
                        asset_price = price
                        break

                if asset_price is not None:
                    asset_value = balance * asset_price
                    equity += asset_value
                else:
                    # If we can't find a price, warn and treat as zero
                    logger.warning(f"No price found for asset {asset}, treating as zero value")

        # Add unrealized PnL from open positions
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position.update_unrealized_pnl(current_price)
                equity += position.unrealized_pnl

        return equity

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history."""
        return self.trade_history.copy()

    def reset(self, initial_balance: Optional[Dict[str, float]] = None) -> None:
        """
        Reset the simulation engine.

        Args:
            initial_balance: Optional new initial balance
        """
        if initial_balance:
            self.balance = initial_balance.copy()
        else:
            # Reset to empty balance
            self.balance = {}

        self.orders = []
        self.positions = {}
        self.trade_history = []
        self.last_prices = {}
