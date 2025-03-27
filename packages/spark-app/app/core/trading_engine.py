import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import pandas as pd

from app.connectors.base_connector import (
    BaseConnector,
    OrderSide,
    OrderType,
    MarketType,
)
from app.indicators.base_indicator import Signal, SignalDirection
from app.risk_management.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class TradingState(str, Enum):
    """Possible states of the trading engine."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    SHUTTING_DOWN = "SHUTTING_DOWN"


class TradingEngine:
    """
    Core trading engine that coordinates indicators, connectors, and risk management.

    This class is responsible for:
    1. Receiving signals from indicators or webhooks
    2. Validating signals with risk management
    3. Executing trades on exchanges
    4. Monitoring positions and implementing hedging
    5. Managing the lifecycle of trades

    Supports both spot and derivatives markets with appropriate adaptations.
    """

    def __init__(
        self,
        main_connector: BaseConnector,
        hedge_connector: Optional[BaseConnector] = None,
        risk_manager: Optional[RiskManager] = None,
        dry_run: bool = True,
        polling_interval: int = 60,
        max_parallel_trades: int = 1,
        enable_hedging: bool = True,
    ):
        """
        Initialize the trading engine.

        Args:
            main_connector: Primary exchange connector
            hedge_connector: Optional separate connector for hedge positions
            risk_manager: Risk management module
            dry_run: If True, don't execute actual trades (simulation mode)
            polling_interval: Seconds between position update checks
            max_parallel_trades: Maximum number of concurrent trades allowed
            enable_hedging: Whether to use hedging strategies (if False, only main positions are used)
        """
        self.main_connector = main_connector
        self.hedge_connector = (
            hedge_connector or main_connector
        )  # Use main if not provided
        self.risk_manager = risk_manager or RiskManager()
        self.dry_run = dry_run
        self.polling_interval = polling_interval
        self.max_parallel_trades = max_parallel_trades
        self.enable_hedging = enable_hedging

        # Check if hedging is possible with the provided connectors
        self._can_hedge = self._check_hedging_capability()

        # Trading state
        self.state = TradingState.IDLE
        self.active_trades: Dict[str, Dict[str, Any]] = {}  # Symbol -> Trade info
        self.pending_signals: List[Signal] = []

        # Monitoring thread
        self.monitor_thread = None
        self.stop_event = threading.Event()

    def _check_hedging_capability(self) -> bool:
        """
        Check if hedging is possible with the current connectors.

        Returns:
            bool: True if hedging is possible, False otherwise
        """
        # If hedging is disabled in config, return False
        if not self.enable_hedging:
            logger.info("Hedging is disabled in configuration")
            return False

        # If no separate hedge connector is provided, see if main connector supports derivatives
        if self.hedge_connector == self.main_connector:
            if self.main_connector.supports_derivatives:
                logger.info(
                    "Using main connector for both main and hedge positions (derivatives supported)"
                )
                return True
            else:
                logger.warning(
                    "Hedging requires a derivatives-capable connector, using spot-only mode"
                )
                return False
        else:
            # If separate hedge connector is provided, check if it supports derivatives
            if self.hedge_connector.supports_derivatives:
                logger.info("Using separate hedge connector with derivatives support")
                return True
            else:
                logger.warning(
                    "Hedge connector does not support derivatives, using spot-only mode"
                )
                return False

    def start(self) -> bool:
        """
        Start the trading engine.

        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.state == TradingState.RUNNING:
            logger.warning("Trading engine is already running")
            return True

        try:
            # Initialize main connector if not already connected
            if not self.main_connector.is_connected:
                self.main_connector.connect()

            # Initialize hedge connector if it's different and not already connected
            if (
                self.hedge_connector != self.main_connector
                and not self.hedge_connector.is_connected
            ):
                self.hedge_connector.connect()

            # Log market types supported by connectors
            main_market_types = (
                [self.main_connector.market_types]
                if not isinstance(self.main_connector.market_types, list)
                else self.main_connector.market_types
            )
            logger.info(
                f"Main connector market types: {[t.value for t in main_market_types]}"
            )
            if self.hedge_connector != self.main_connector:
                hedge_market_types = (
                    [self.hedge_connector.market_types]
                    if not isinstance(self.hedge_connector.market_types, list)
                    else self.hedge_connector.market_types
                )
                logger.info(
                    f"Hedge connector market types: {[t.value for t in hedge_market_types]}"
                )

            # Start the monitoring thread
            self.stop_event.clear()
            self.monitor_thread = threading.Thread(
                target=self._monitor_positions, daemon=True
            )
            self.monitor_thread.start()

            self.state = TradingState.RUNNING
            logger.info(
                f"Trading engine started (dry_run={self.dry_run}, hedging_enabled={self._can_hedge})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start trading engine: {e}")
            self.state = TradingState.ERROR
            return False

    def stop(self) -> bool:
        """
        Stop the trading engine.

        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if self.state not in [TradingState.RUNNING, TradingState.PAUSED]:
            logger.warning(f"Trading engine is not running (state={self.state})")
            return True

        try:
            self.state = TradingState.SHUTTING_DOWN
            logger.info("Shutting down trading engine...")

            # Signal monitor thread to stop
            self.stop_event.set()

            # Wait for monitor thread to finish
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)

            # Close all positions if requested
            # (This could be made optional based on a parameter)

            self.state = TradingState.IDLE
            logger.info("Trading engine stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping trading engine: {e}")
            self.state = TradingState.ERROR
            return False

    def pause(self) -> bool:
        """
        Pause the trading engine (stops processing new signals).

        Returns:
            bool: True if paused successfully, False otherwise
        """
        if self.state != TradingState.RUNNING:
            logger.warning(
                f"Cannot pause: trading engine is not running (state={self.state})"
            )
            return False

        self.state = TradingState.PAUSED
        logger.info("Trading engine paused")
        return True

    def resume(self) -> bool:
        """
        Resume the trading engine after pausing.

        Returns:
            bool: True if resumed successfully, False otherwise
        """
        if self.state != TradingState.PAUSED:
            logger.warning(
                f"Cannot resume: trading engine is not paused (state={self.state})"
            )
            return False

        self.state = TradingState.RUNNING
        logger.info("Trading engine resumed")
        return True

    def process_signal(self, signal: Signal) -> bool:
        """
        Process a new trading signal.

        Args:
            signal: Trading signal to process

        Returns:
            bool: True if signal was processed, False otherwise
        """
        if self.state not in [TradingState.RUNNING, TradingState.IDLE]:
            logger.warning(
                f"Cannot process signal: engine not in correct state (state={self.state})"
            )
            self.pending_signals.append(signal)
            return False

        try:
            logger.info(f"Processing signal: {signal}")
            logger.debug(f"Signal direction type: {type(signal.direction)}")
            logger.debug(
                f"Signal attributes: direction={signal.direction}, symbol={signal.symbol}"
            )

            # Get symbol
            symbol = signal.symbol

            # Check if we're already trading this symbol
            if symbol in self.active_trades:
                logger.warning(
                    f"Already have an active trade for {symbol}, cannot process new signal"
                )
                return False

            # Check if we've reached max parallel trades
            if len(self.active_trades) >= self.max_parallel_trades:
                logger.warning(
                    f"Maximum parallel trades ({self.max_parallel_trades}) reached, cannot process new signal"
                )
                self.pending_signals.append(signal)
                return False

            # Convert signal direction to order side
            if signal.direction == SignalDirection.BUY:
                logger.debug("Converting BUY signal to BUY main side, SELL hedge side")
                main_side = OrderSide.BUY
                hedge_side = OrderSide.SELL
            elif signal.direction == SignalDirection.SELL:
                logger.debug("Converting SELL signal to SELL main side, BUY hedge side")
                main_side = OrderSide.SELL
                hedge_side = OrderSide.BUY
            else:
                logger.warning(f"Neutral signal received, no action taken")
                return False

            logger.debug(f"Main side: {main_side}, type: {type(main_side)}")
            logger.debug(f"Hedge side: {hedge_side}, type: {type(hedge_side)}")

            # Extract price from signal parameters if available
            price = signal.params.get("price")
            if price:
                logger.info(f"Using price from signal: {price}")
            else:
                # Get the current market price if not provided
                try:
                    ticker_data = self.main_connector.get_ticker(symbol)
                    if (
                        ticker_data
                        and "last_price" in ticker_data
                        and ticker_data["last_price"] > 0
                    ):
                        price = float(ticker_data["last_price"])
                        logger.info(f"Using current market price: {price}")
                except Exception as e:
                    logger.warning(f"Could not fetch current market price: {e}")
                    # Price will remain None, and the exchange will use the current market price

            # Determine if we should use leverage based on market type
            use_leverage = self.main_connector.supports_derivatives
            default_leverage = 10.0 if use_leverage else 1.0

            try:
                # Normal position size calculation
                logger.debug("Calculating position size...")
                (
                    main_size,
                    adjusted_main_leverage,
                ) = self.risk_manager.calculate_position_size(
                    exchange=self.main_connector,
                    symbol=symbol,
                    available_balance=self.main_connector.get_account_balance(),
                    confidence=signal.confidence,
                    signal_side=main_side,
                    leverage=default_leverage,
                    stop_loss_pct=10.0,
                    price=price,
                )
                logger.debug(
                    f"Position size calculation result: size={main_size}, leverage={adjusted_main_leverage}"
                )
            except Exception as e:
                logger.error(f"Error calculating position size: {e}")
                return False

            if main_size <= 0:
                logger.error(f"Invalid position size calculated: {main_size}")
                return False

            try:
                # Validate the trade with risk manager
                logger.debug("Validating trade...")
                is_valid, reason = self.risk_manager.validate_trade(
                    exchange=self.main_connector,
                    symbol=symbol,
                    position_size=main_size,
                    leverage=adjusted_main_leverage,
                    side=main_side,
                )
                logger.debug(
                    f"Trade validation result: valid={is_valid}, reason={reason}"
                )
            except Exception as e:
                logger.error(f"Error validating trade: {e}")
                return False

            if not is_valid:
                logger.warning(f"Trade validation failed: {reason}")
                return False

            # Calculate hedge position parameters if hedging is enabled and possible
            hedge_size = 0.0
            adjusted_hedge_leverage = 1.0

            if self._can_hedge:
                try:
                    (
                        hedge_size,
                        adjusted_hedge_leverage,
                    ) = self.risk_manager.calculate_hedge_parameters(
                        main_position_size=main_size,
                        main_leverage=adjusted_main_leverage,
                        hedge_ratio=0.2,
                        max_hedge_leverage=5.0
                        if self.hedge_connector.supports_derivatives
                        else 1.0,
                    )
                    logger.debug(
                        f"Hedge calculation result: size={hedge_size}, leverage={adjusted_hedge_leverage}"
                    )
                except Exception as e:
                    logger.error(f"Error calculating hedge parameters: {e}")
                    return False

                # Check if the hedge connector has enough balance
                if self.hedge_connector != self.main_connector:
                    hedge_balance = self.hedge_connector.get_account_balance()
                    hedge_available = sum(hedge_balance.values())

                    if hedge_available < hedge_size:
                        logger.error(
                            f"Insufficient balance on hedge exchange: {hedge_available}, needed {hedge_size}"
                        )
                        return False
            else:
                logger.info(
                    "Hedging is not enabled or not possible with current connectors"
                )

            # Log the trade plan
            logger.info(f"Trade plan for {symbol}:")
            logger.info(
                f"  Main position: {main_side.value} {main_size:.2f} @ {adjusted_main_leverage:.1f}x"
            )
            if self._can_hedge:
                logger.info(
                    f"  Hedge position: {hedge_side.value} {hedge_size:.2f} @ {adjusted_hedge_leverage:.1f}x"
                )

            # Execute trades in dry run mode (simulation)
            if self.dry_run:
                logger.info("DRY RUN MODE: No actual trades executed")

                try:
                    # Get current price for the simulation
                    current_price = self.main_connector.get_ticker(symbol).get(
                        "last_price", 0.0
                    )
                    logger.debug(f"Got current price for simulation: {current_price}")

                    # Create the trade record
                    logger.debug(
                        "Creating trade record with: "
                        + f"main_side={main_side}, type={type(main_side)}, "
                        + f"main_side.value={main_side.value}, type={type(main_side.value)}"
                    )

                    # Record the simulated trade in active_trades
                    trade_record = {
                        "symbol": symbol,
                        "timestamp": int(time.time() * 1000),
                        "main_position": {
                            "exchange": "main",
                            "side": main_side.value,  # Use .value to get the string
                            "size": main_size,
                            "leverage": adjusted_main_leverage,
                            "entry_price": current_price,
                            "order_id": f"dry_run_main_{int(time.time())}",
                        },
                        "stop_loss_pct": 10.0,
                        "take_profit_pct": 20.0,
                        "status": "open",
                        "market_type": self.main_connector.market_types[0]
                        if isinstance(self.main_connector.market_types[0], str)
                        else self.main_connector.market_types[0].value,
                    }

                    # Add hedge position if applicable
                    if self._can_hedge:
                        logger.debug(
                            f"Adding hedge position with side={hedge_side.value}"
                        )
                        trade_record["hedge_position"] = {
                            "exchange": "hedge",
                            "side": hedge_side.value,  # Use .value to get the string
                            "size": hedge_size,
                            "leverage": adjusted_hedge_leverage,
                            "order_id": f"dry_run_hedge_{int(time.time())}",
                        }

                    logger.debug(f"Final trade record: {trade_record}")
                    self.active_trades[symbol] = trade_record
                    return True
                except Exception as e:
                    logger.error(
                        f"Error creating simulation trade record: {e}", exc_info=True
                    )
                    return False

            # Execute main position
            # Check if this is a Coinbase connector
            is_coinbase = "coinbase" in self.main_connector.__class__.__name__.lower()

            # For Coinbase, ALWAYS use LIMIT orders because they may enforce limit-only mode
            if is_coinbase:
                # Always need a valid price for Coinbase limit orders
                if not price or price <= 0:
                    try:
                        # Get optimal limit price that should fill immediately
                        optimal_price_data = (
                            self.main_connector.get_optimal_limit_price(
                                symbol=symbol, side=main_side, amount=main_size
                            )
                        )

                        price = optimal_price_data["price"]

                        if not optimal_price_data["enough_liquidity"]:
                            logger.warning(
                                f"Limited liquidity for {symbol}, order may not fill completely"
                            )

                        logger.info(
                            f"Using optimal limit price for immediate fill: {price}"
                        )

                    except Exception as e:
                        logger.warning(
                            f"Error getting optimal limit price: {e}, falling back to ticker price"
                        )
                        # Fallback to getting current price from ticker
                        ticker_data = self.main_connector.get_ticker(symbol)
                        if (
                            ticker_data
                            and "last_price" in ticker_data
                            and ticker_data["last_price"] > 0
                        ):
                            # For BUY orders: add a small buffer to ensure immediate fill (0.1%)
                            # For SELL orders: subtract a small buffer to ensure immediate fill (0.1%)
                            price = float(ticker_data["last_price"])
                            if main_side == OrderSide.BUY:
                                price *= (
                                    1.001  # 0.1% higher than market for immediate fill
                                )
                            else:
                                price *= (
                                    0.999  # 0.1% lower than market for immediate fill
                                )
                            logger.info(
                                f"Using current market price with buffer for limit order: {price}"
                            )
                        else:
                            logger.error(
                                f"Cannot place order: No valid price for {symbol} LIMIT order"
                            )
                            return False

                logger.info(
                    f"Using LIMIT order for Coinbase with price {price} (required by Coinbase)"
                )
                order_type = OrderType.LIMIT
            else:
                order_type = OrderType.MARKET

            # Only pass leverage parameter if derivatives are supported
            order_params = {
                "symbol": symbol,
                "side": main_side,
                "order_type": order_type,
                "amount": main_size,
                "price": price,
            }

            if self.main_connector.supports_derivatives:
                order_params["leverage"] = adjusted_main_leverage

            main_order = self.main_connector.place_order(**order_params)

            if main_order.get("error"):
                logger.error(f"Failed to place main order: {main_order.get('error')}")
                return False

            logger.info(f"Main order placed: {main_order}")

            # Small delay to ensure main order is processed
            time.sleep(1)

            # Execute hedge position if hedge connector is available and hedging is enabled
            hedge_order = None

            if self._can_hedge and hedge_size > 0:
                # Check if this is a Coinbase connector
                is_coinbase_hedge = (
                    "coinbase" in self.hedge_connector.__class__.__name__.lower()
                )

                # For Coinbase, must use LIMIT orders (they enforce limit-only mode)
                if is_coinbase_hedge:
                    # Always need a valid price for Coinbase limit orders
                    if not price or price <= 0:
                        try:
                            # Get optimal limit price that should fill immediately
                            optimal_price_data = (
                                self.hedge_connector.get_optimal_limit_price(
                                    symbol=symbol, side=hedge_side, amount=hedge_size
                                )
                            )

                            price = optimal_price_data["price"]

                            if not optimal_price_data["enough_liquidity"]:
                                logger.warning(
                                    f"Limited liquidity for hedge {symbol}, order may not fill completely"
                                )

                            logger.info(
                                f"Using optimal limit price for immediate hedge fill: {price}"
                            )

                        except Exception as e:
                            logger.warning(
                                f"Error getting optimal limit price for hedge: {e}, falling back to ticker price"
                            )
                            # Fallback to getting current price from ticker
                            ticker_data = self.hedge_connector.get_ticker(symbol)
                            if (
                                ticker_data
                                and "last_price" in ticker_data
                                and ticker_data["last_price"] > 0
                            ):
                                # For BUY orders: add a small buffer to ensure immediate fill (0.1%)
                                # For SELL orders: subtract a small buffer to ensure immediate fill (0.1%)
                                price = float(ticker_data["last_price"])
                                if hedge_side == OrderSide.BUY:
                                    price *= 1.001  # 0.1% higher than market for immediate fill
                                else:
                                    price *= 0.999  # 0.1% lower than market for immediate fill
                                logger.info(
                                    f"Using current market price with buffer for hedge limit order: {price}"
                                )
                            else:
                                logger.error(
                                    f"Cannot place hedge order: No valid price for {symbol} LIMIT order"
                                )
                                return False

                    logger.info(
                        f"Using LIMIT order for Coinbase hedge with price {price} (required by Coinbase)"
                    )
                    hedge_order_type = OrderType.LIMIT
                else:
                    hedge_order_type = OrderType.MARKET

                # Only pass leverage parameter if derivatives are supported
                hedge_order_params = {
                    "symbol": symbol,
                    "side": hedge_side,
                    "order_type": hedge_order_type,
                    "amount": hedge_size,
                    "price": price,
                }

                if self.hedge_connector.supports_derivatives:
                    hedge_order_params["leverage"] = adjusted_hedge_leverage

                hedge_order = self.hedge_connector.place_order(**hedge_order_params)

                if hedge_order.get("error"):
                    logger.error(
                        f"Failed to place hedge order: {hedge_order.get('error')}"
                    )

                    # Try to cancel the main order if hedge fails and we have a valid order_id
                    order_id = main_order.get("order_id")
                    if order_id:
                        logger.info(
                            f"Cancelling main order {order_id} after hedge failure"
                        )
                        self.main_connector.cancel_order(order_id=order_id)
            elif not self._can_hedge:
                logger.info("Skipping hedge position (hedging not enabled)")
            else:
                logger.info(f"Skipping hedge position (size would be {hedge_size})")

            # Record the trade in active_trades
            trade_record = {
                "symbol": symbol,
                "timestamp": int(time.time() * 1000),
                "main_position": {
                    "exchange": "main",
                    "side": main_side.value,
                    "size": main_size,
                    "leverage": adjusted_main_leverage,
                    "order_id": main_order.get("order_id", "unknown_order_id"),
                },
                "stop_loss_pct": 10.0,
                "take_profit_pct": 20.0,
                "status": "open",
                "market_type": self.main_connector.market_types[0]
                if isinstance(self.main_connector.market_types[0], str)
                else self.main_connector.market_types[0].value,
            }

            # Add hedge position if applicable
            if self._can_hedge and hedge_order:
                trade_record["hedge_position"] = {
                    "exchange": "hedge",
                    "side": hedge_side.value,
                    "size": hedge_size,
                    "leverage": adjusted_hedge_leverage,
                    "order_id": hedge_order.get("order_id", "unknown_order_id"),
                }

            self.active_trades[symbol] = trade_record
            return True

        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return False

    def _execute_hedged_trade(
        self,
        symbol: str,
        main_side: OrderSide,
        hedge_side: OrderSide,
        confidence: float,
        main_leverage: float = 10.0,
        hedge_leverage: float = 5.0,
        hedge_ratio: float = 0.2,
        stop_loss_pct: float = 10.0,
        take_profit_pct: float = 20.0,
        price: Optional[float] = None,
        signal: Optional[Signal] = None,
    ) -> bool:
        """
        Execute a hedged trade with a main position and smaller opposite hedge.

        Args:
            symbol: Market symbol (e.g., 'ETH')
            main_side: Side for the main position (BUY or SELL)
            hedge_side: Side for the hedge position (opposite of main)
            confidence: Signal confidence (0-1)
            main_leverage: Leverage for the main position
            hedge_leverage: Leverage for the hedge position
            hedge_ratio: Ratio of hedge notional to main notional (0-1)
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            price: Price for the trade (optional)
            signal: Signal object for the trade

        Returns:
            bool: True if trade was executed, False otherwise
        """
        try:
            # Get account balance for the main connector
            main_balance = self.main_connector.get_account_balance()
            main_available = sum(main_balance.values())

            if main_available <= 0:
                logger.error(f"Insufficient balance on main exchange: {main_available}")
                return False

            # Adjust leverage based on market type
            if not self.main_connector.supports_derivatives:
                main_leverage = 1.0
                logger.info("Main connector is spot-only, using leverage of 1.0")

            if (
                not self.hedge_connector.supports_derivatives
                and self.hedge_connector != self.main_connector
            ):
                hedge_leverage = 1.0
                logger.info("Hedge connector is spot-only, using leverage of 1.0")

            # Calculate position size and leverage for main position
            (
                main_size,
                adjusted_main_leverage,
            ) = self.risk_manager.calculate_position_size(
                exchange=self.main_connector,
                symbol=symbol,
                available_balance=main_available,
                confidence=confidence,
                signal_side=main_side,
                leverage=main_leverage,
                stop_loss_pct=stop_loss_pct,
                price=price,
            )

            if main_size <= 0:
                logger.error(f"Invalid position size calculated: {main_size}")
                return False

            # Validate the trade with risk manager
            is_valid, reason = self.risk_manager.validate_trade(
                exchange=self.main_connector,
                symbol=symbol,
                position_size=main_size,
                leverage=adjusted_main_leverage,
                side=main_side,
            )

            if not is_valid:
                logger.warning(f"Trade validation failed: {reason}")
                return False

            # Skip hedging for spot-only setups if main_side is SELL (already selling the asset)
            hedge_size = 0.0
            adjusted_hedge_leverage = 1.0

            # Only calculate hedge parameters if hedging is enabled and possible
            if self._can_hedge:
                (
                    hedge_size,
                    adjusted_hedge_leverage,
                ) = self.risk_manager.calculate_hedge_parameters(
                    main_position_size=main_size,
                    main_leverage=adjusted_main_leverage,
                    hedge_ratio=hedge_ratio,
                    max_hedge_leverage=hedge_leverage,
                )

                # Check if the hedge connector has enough balance
                if self.hedge_connector != self.main_connector:
                    hedge_balance = self.hedge_connector.get_account_balance()
                    hedge_available = sum(hedge_balance.values())

                    if hedge_available < hedge_size:
                        logger.error(
                            f"Insufficient balance on hedge exchange: {hedge_available}, needed {hedge_size}"
                        )
                        return False
            else:
                logger.info(
                    "Hedging is not enabled or not possible with current connectors"
                )

            # Log the trade plan
            logger.info(f"Trade plan for {symbol}:")
            logger.info(
                f"  Main position: {main_side.value} {main_size:.2f} @ {adjusted_main_leverage:.1f}x"
            )
            if self._can_hedge and hedge_size > 0:
                logger.info(
                    f"  Hedge position: {hedge_side.value} {hedge_size:.2f} @ {adjusted_hedge_leverage:.1f}x"
                )

            # Execute trades in dry run mode (simulation)
            if self.dry_run:
                logger.info("DRY RUN MODE: No actual trades executed")

                # Record the simulated trade in active_trades
                current_price = self.main_connector.get_ticker(symbol).get(
                    "last_price", 0.0
                )

                trade_record = {
                    "symbol": symbol,
                    "timestamp": int(time.time() * 1000),
                    "main_position": {
                        "exchange": "main",
                        "side": main_side.value,
                        "size": main_size,
                        "leverage": adjusted_main_leverage,
                        "entry_price": current_price,
                        "order_id": f"dry_run_main_{int(time.time())}",
                    },
                    "stop_loss_pct": stop_loss_pct,
                    "take_profit_pct": take_profit_pct,
                    "status": "open",
                    "market_type": self.main_connector.market_types[0]
                    if isinstance(self.main_connector.market_types[0], str)
                    else self.main_connector.market_types[0].value,
                }

                # Add hedge position if applicable
                if self._can_hedge and hedge_size > 0:
                    trade_record["hedge_position"] = {
                        "exchange": "hedge",
                        "side": hedge_side.value,
                        "size": hedge_size,
                        "leverage": adjusted_hedge_leverage,
                        "entry_price": current_price,
                        "order_id": f"dry_run_hedge_{int(time.time())}",
                    }

                self.active_trades[symbol] = trade_record
                return True

            # Execute main position
            # Check if this is a Coinbase connector
            is_coinbase = "coinbase" in self.main_connector.__class__.__name__.lower()

            # For Coinbase, ALWAYS use LIMIT orders because they may enforce limit-only mode
            if is_coinbase:
                # Always need a valid price for Coinbase limit orders
                if not price or price <= 0:
                    try:
                        # Get optimal limit price that should fill immediately
                        optimal_price_data = (
                            self.main_connector.get_optimal_limit_price(
                                symbol=symbol, side=main_side, amount=main_size
                            )
                        )

                        price = optimal_price_data["price"]

                        if not optimal_price_data["enough_liquidity"]:
                            logger.warning(
                                f"Limited liquidity for {symbol}, order may not fill completely"
                            )

                        logger.info(
                            f"Using optimal limit price for immediate fill: {price}"
                        )

                    except Exception as e:
                        logger.warning(
                            f"Error getting optimal limit price: {e}, falling back to ticker price"
                        )
                        # Fallback to getting current price from ticker
                        ticker_data = self.main_connector.get_ticker(symbol)
                        if (
                            ticker_data
                            and "last_price" in ticker_data
                            and ticker_data["last_price"] > 0
                        ):
                            # For BUY orders: add a small buffer to ensure immediate fill (0.1%)
                            # For SELL orders: subtract a small buffer to ensure immediate fill (0.1%)
                            price = float(ticker_data["last_price"])
                            if main_side == OrderSide.BUY:
                                price *= (
                                    1.001  # 0.1% higher than market for immediate fill
                                )
                            else:
                                price *= (
                                    0.999  # 0.1% lower than market for immediate fill
                                )
                            logger.info(
                                f"Using current market price with buffer for limit order: {price}"
                            )
                        else:
                            logger.error(
                                f"Cannot place order: No valid price for {symbol} LIMIT order"
                            )
                            return False

                logger.info(
                    f"Using LIMIT order for Coinbase with price {price} (required by Coinbase)"
                )
                order_type = OrderType.LIMIT
            else:
                order_type = OrderType.MARKET

            # Only pass leverage parameter if derivatives are supported
            order_params = {
                "symbol": symbol,
                "side": main_side,
                "order_type": order_type,
                "amount": main_size,
                "price": price,
            }

            if self.main_connector.supports_derivatives:
                order_params["leverage"] = adjusted_main_leverage

            main_order = self.main_connector.place_order(**order_params)

            if main_order.get("error"):
                logger.error(f"Failed to place main order: {main_order.get('error')}")
                return False

            logger.info(f"Main order placed: {main_order}")

            # Small delay to ensure main order is processed
            time.sleep(1)

            # Execute hedge position if hedge connector is available and hedging is enabled
            hedge_order = None
            if self._can_hedge and hedge_size > 0:
                # Check if this is a Coinbase connector
                is_coinbase_hedge = (
                    "coinbase" in self.hedge_connector.__class__.__name__.lower()
                )

                # For Coinbase, must use LIMIT orders (they enforce limit-only mode)
                if is_coinbase_hedge:
                    # Always need a valid price for Coinbase limit orders
                    if not price or price <= 0:
                        try:
                            # Get optimal limit price that should fill immediately
                            optimal_price_data = (
                                self.hedge_connector.get_optimal_limit_price(
                                    symbol=symbol, side=hedge_side, amount=hedge_size
                                )
                            )

                            price = optimal_price_data["price"]

                            if not optimal_price_data["enough_liquidity"]:
                                logger.warning(
                                    f"Limited liquidity for hedge {symbol}, order may not fill completely"
                                )

                            logger.info(
                                f"Using optimal limit price for immediate hedge fill: {price}"
                            )

                        except Exception as e:
                            logger.warning(
                                f"Error getting optimal limit price for hedge: {e}, falling back to ticker price"
                            )
                            # Fallback to getting current price from ticker
                            ticker_data = self.hedge_connector.get_ticker(symbol)
                            if (
                                ticker_data
                                and "last_price" in ticker_data
                                and ticker_data["last_price"] > 0
                            ):
                                # For BUY orders: add a small buffer to ensure immediate fill (0.1%)
                                # For SELL orders: subtract a small buffer to ensure immediate fill (0.1%)
                                price = float(ticker_data["last_price"])
                                if hedge_side == OrderSide.BUY:
                                    price *= 1.001  # 0.1% higher than market for immediate fill
                                else:
                                    price *= 0.999  # 0.1% lower than market for immediate fill
                                logger.info(
                                    f"Using current market price with buffer for hedge limit order: {price}"
                                )
                            else:
                                logger.error(
                                    f"Cannot place hedge order: No valid price for {symbol} LIMIT order"
                                )
                                return False

                    logger.info(
                        f"Using LIMIT order for Coinbase hedge with price {price} (required by Coinbase)"
                    )
                    hedge_order_type = OrderType.LIMIT
                else:
                    hedge_order_type = OrderType.MARKET

                # Only pass leverage parameter if derivatives are supported
                hedge_order_params = {
                    "symbol": symbol,
                    "side": hedge_side,
                    "order_type": hedge_order_type,
                    "amount": hedge_size,
                    "price": price,
                }

                if self.hedge_connector.supports_derivatives:
                    hedge_order_params["leverage"] = adjusted_hedge_leverage

                hedge_order = self.hedge_connector.place_order(**hedge_order_params)

                if hedge_order and hedge_order.get("error"):
                    logger.error(
                        f"Failed to place hedge order: {hedge_order.get('error')}"
                    )

                    # Try to cancel the main order if hedge fails and we have a valid order_id
                    order_id = main_order.get("order_id")
                    if order_id:
                        logger.info(
                            f"Cancelling main order {order_id} after hedge failure"
                        )
                        self.main_connector.cancel_order(order_id=order_id)

            # Record the trade in active_trades
            trade_record = {
                "symbol": symbol,
                "timestamp": int(time.time() * 1000),
                "main_position": {
                    "exchange": "main",
                    "side": main_side.value,
                    "size": main_size,
                    "leverage": adjusted_main_leverage,
                    "order_id": main_order.get("order_id", "unknown_order_id"),
                },
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "status": "open",
                "market_type": self.main_connector.market_types[0]
                if isinstance(self.main_connector.market_types[0], str)
                else self.main_connector.market_types[0].value,
            }

            # Add hedge position if applicable
            if self._can_hedge and hedge_order:
                trade_record["hedge_position"] = {
                    "exchange": "hedge",
                    "side": hedge_side.value,
                    "size": hedge_size,
                    "leverage": adjusted_hedge_leverage,
                    "order_id": hedge_order.get("order_id", "unknown_order_id"),
                }

            self.active_trades[symbol] = trade_record
            return True

        except Exception as e:
            logger.error(f"Error executing hedged trade: {e}")
            return False

    def _monitor_positions(self) -> None:
        """
        Background thread to monitor open positions and manage trade lifecycle.
        """
        logger.info("Position monitoring thread started")

        # Add a last check timestamp to reduce redundant API calls
        last_position_check = 0

        while not self.stop_event.is_set():
            try:
                if self.state not in [TradingState.RUNNING, TradingState.PAUSED]:
                    # If not running or paused, just check if we should stop
                    time.sleep(1)
                    continue

                # Skip position checks if paused
                current_time = time.time()
                should_check_positions = (
                    current_time - last_position_check
                ) >= self.polling_interval

                if self.state != TradingState.PAUSED and should_check_positions:
                    # Get current positions from exchanges
                    main_positions = self.main_connector.get_positions()

                    # Get hedge positions if using a separate connector
                    if self.hedge_connector != self.main_connector:
                        hedge_positions = self.hedge_connector.get_positions()
                    else:
                        hedge_positions = main_positions

                    # Update risk manager with current positions
                    self.risk_manager.update_positions(main_positions + hedge_positions)

                    # Process active trades
                    self._check_active_trades(main_positions, hedge_positions)

                    # Update the last check timestamp
                    last_position_check = current_time

                # Process any pending signals if we have capacity
                if (
                    self.state == TradingState.RUNNING
                    and len(self.active_trades) < self.max_parallel_trades
                    and self.pending_signals
                ):
                    signal = self.pending_signals.pop(0)
                    logger.info(f"Processing pending signal: {signal}")
                    self.process_signal(signal)

                # Sleep for a shorter interval to be responsive to stop events
                time.sleep(min(1, self.polling_interval / 10))

            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                time.sleep(max(1, self.polling_interval // 4))  # Shorter sleep on error

        logger.info("Position monitoring thread stopped")

    def _check_active_trades(
        self, main_positions: List[Dict], hedge_positions: List[Dict]
    ) -> None:
        """
        Check and manage active trades based on current positions.

        Args:
            main_positions: List of positions from main connector
            hedge_positions: List of positions from hedge connector
        """
        # Skip if no active trades
        if not self.active_trades:
            return

        logger.debug(f"Checking {len(self.active_trades)} active trades")

        # Track trades to remove
        trades_to_remove = []

        # Convert positions lists to dictionaries for easier lookup
        main_pos_dict = {pos["symbol"]: pos for pos in main_positions}
        hedge_pos_dict = {pos["symbol"]: pos for pos in hedge_positions}

        # Iterate through active trades
        for symbol, trade in self.active_trades.items():
            try:
                # Get positions for this symbol (if they exist)
                main_pos = main_pos_dict.get(symbol)
                hedge_pos = hedge_pos_dict.get(symbol)

                # Get market type
                market_type = "SPOT"  # Default
                if (
                    isinstance(self.main_connector.market_types, list)
                    and len(self.main_connector.market_types) > 0
                ):
                    if isinstance(self.main_connector.market_types[0], str):
                        market_type = self.main_connector.market_types[0]
                    else:
                        market_type = self.main_connector.market_types[0].value

                # Special handling for spot markets
                if market_type == "SPOT" and not main_pos:
                    try:
                        # For spot markets, position is equal to wallet balance
                        # Get the base asset from the symbol (e.g., BTC from BTCUSD)
                        base_asset = (
                            symbol.split("-")[0]
                            if "-" in symbol
                            else symbol.split("/")[0]
                            if "/" in symbol
                            else symbol
                        )
                        base_asset = base_asset.upper()

                        # Get balances and current price
                        balances = self.main_connector.get_account_balance()
                        ticker_data = self.main_connector.get_ticker(symbol)
                        current_price = ticker_data.get("last_price", 0.0)

                        # Create a synthetic position from our balance
                        main_pos = {
                            "symbol": symbol,
                            "side": "LONG",  # Spot positions are always "long" the asset
                            "size": balances.get(base_asset, 0),
                            "entry_price": trade["main_position"].get(
                                "entry_price", current_price
                            ),
                            "mark_price": current_price,
                            "unrealized_pnl": 0.0,  # Will be calculated below
                        }

                        # Calculate unrealized P&L if we have entry price
                        if main_pos["entry_price"] > 0 and current_price > 0:
                            price_diff = current_price - main_pos["entry_price"]
                            main_pos["unrealized_pnl"] = price_diff * main_pos["size"]
                    except Exception as e:
                        logger.error(f"Error fetching spot position for {symbol}: {e}")

                # Check if positions still exist
                if market_type != "SPOT" and not main_pos and not hedge_pos:
                    logger.info(
                        f"Both positions for {symbol} are closed, removing from active trades"
                    )
                    trades_to_remove.append(symbol)
                    continue
                elif market_type == "SPOT" and (
                    not main_pos or (main_pos.get("size", 0) <= 0)
                ):
                    logger.info(
                        f"Spot position for {symbol} is closed, removing from active trades"
                    )
                    trades_to_remove.append(symbol)
                    continue

                # Update trade record with current position details
                if main_pos:
                    trade["main_position"].update(
                        {
                            "entry_price": main_pos.get("entry_price"),
                            "mark_price": main_pos.get("mark_price"),
                            "size": main_pos.get("size"),
                            "unrealized_pnl": main_pos.get("unrealized_pnl"),
                            "liquidation_price": main_pos.get("liquidation_price"),
                        }
                    )

                if hedge_pos:
                    trade["hedge_position"].update(
                        {
                            "entry_price": hedge_pos.get("entry_price"),
                            "mark_price": hedge_pos.get("mark_price"),
                            "size": hedge_pos.get("size"),
                            "unrealized_pnl": hedge_pos.get("unrealized_pnl"),
                            "liquidation_price": hedge_pos.get("liquidation_price"),
                        }
                    )

                # Check if main position should be closed (stop loss, take profit, etc.)
                if main_pos:
                    try:
                        result = self.risk_manager.should_close_position(
                            main_pos,
                            trade.get("stop_loss_pct", 10.0),
                            trade.get("take_profit_pct", 20.0),
                        )

                        # Handle the case where result might be None or not a tuple
                        if result and isinstance(result, tuple) and len(result) == 2:
                            should_close, reason = result

                            if should_close:
                                logger.info(
                                    f"Closing main position for {symbol}: {reason}"
                                )

                                if not self.dry_run:
                                    # Close position based on market type
                                    self.main_connector.close_position(symbol)

                                trade["main_position"]["status"] = "closing"
                                trade["main_position"]["close_reason"] = reason
                        else:
                            logger.warning(
                                f"Invalid return from should_close_position: {result}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error checking if position should be closed: {e}"
                        )
                        # Continue to next trade rather than breaking

                # Check if hedge position should be adjusted (only for derivative markets)
                if (
                    "hedge_position" in trade
                    and main_pos
                    and hedge_pos
                    and market_type != "SPOT"
                ):
                    (
                        should_adjust,
                        reason,
                        adjustment,
                    ) = self.risk_manager.manage_hedge_position(main_pos, hedge_pos)

                    if should_adjust:
                        logger.info(f"Adjusting hedge for {symbol}: {reason}")
                        action = adjustment.get("action")
                        target = adjustment.get("position")

                        if action == "close":
                            if target == "main" or target == "both":
                                if not self.dry_run:
                                    self.main_connector.close_position(symbol)
                                trade["main_position"]["status"] = "closing"
                                trade["main_position"]["close_reason"] = reason

                            if target == "hedge" or target == "both":
                                if not self.dry_run:
                                    self.hedge_connector.close_position(symbol)
                                trade["hedge_position"]["status"] = "closing"
                                trade["hedge_position"]["close_reason"] = reason

                        elif action == "reduce" and target == "hedge":
                            reduction_pct = adjustment.get("reduction_pct", 50)
                            current_size = hedge_pos.get("size", 0)
                            reduce_size = abs(current_size) * reduction_pct / 100.0

                            logger.info(
                                f"Reducing hedge by {reduction_pct}% ({reduce_size} units)"
                            )

                            if not self.dry_run and reduce_size > 0:
                                # Determine order side (opposite of current position)
                                reduce_side = (
                                    OrderSide.BUY
                                    if hedge_pos.get("side") == "SHORT"
                                    else OrderSide.SELL
                                )

                                # Place a reduce-only order
                                hedge_order_params = {
                                    "symbol": symbol,
                                    "side": reduce_side,
                                    "order_type": OrderType.MARKET,
                                    "amount": reduce_size,
                                }

                                # Only pass leverage if derivatives are supported
                                if self.hedge_connector.supports_derivatives:
                                    hedge_order_params["leverage"] = hedge_pos.get(
                                        "leverage", 1.0
                                    )

                                self.hedge_connector.place_order(**hedge_order_params)

                            trade["hedge_position"]["status"] = "reducing"
                            trade["hedge_position"]["reduce_reason"] = reason

                # Update overall trade status
                main_status = trade.get("main_position", {}).get("status", "open")
                hedge_status = (
                    trade.get("hedge_position", {}).get("status", "open")
                    if "hedge_position" in trade
                    else "none"
                )

                if main_status == "closing" and (
                    hedge_status == "closing" or hedge_status == "none"
                ):
                    trade["status"] = "closing"
                elif main_status != "open" or (
                    hedge_status != "open" and hedge_status != "none"
                ):
                    trade["status"] = "adjusting"
                else:
                    trade["status"] = "open"

            except Exception as e:
                logger.error(f"Error checking trade for {symbol}: {e}")
                logger.exception(e)

        # Remove closed trades
        for symbol in trades_to_remove:
            if symbol in self.active_trades:
                logger.info(f"Removing closed trade for {symbol}")
                del self.active_trades[symbol]

    def get_active_trades(self) -> Dict[str, Dict[str, Any]]:
        """
        Get active trades.

        Returns:
            Dict of active trades by symbol
        """
        return self.active_trades

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get history of closed trades.

        Returns:
            List of trade records
        """
        # In a real implementation, this would fetch from a database
        # For now we'll return an empty list
        return []

    def close_all_positions(self) -> bool:
        """
        Close all open positions.

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Closing all positions")

        if self.dry_run:
            logger.info("DRY RUN MODE: No actual positions closed")
            self.active_trades.clear()
            return True

        success = True

        try:
            # Get all active symbols from our trade records
            active_symbols = list(self.active_trades.keys())

            # Close main positions
            for symbol in active_symbols:
                trade = self.active_trades.get(symbol, {})
                market_type = trade.get("market_type", "SPOT")

                try:
                    # For spot markets with BUY positions, we need to SELL the asset to close
                    if (
                        market_type == "SPOT"
                        and not self.main_connector.supports_derivatives
                    ):
                        main_side = trade.get("main_position", {}).get("side")

                        if main_side == "BUY":
                            # For spot BUY positions, we sell the asset to close
                            logger.info(
                                f"Closing spot position for {symbol} (selling asset)"
                            )

                            # Get our balance of the base asset
                            balances = self.main_connector.get_account_balance()
                            base_asset = symbol.split("-")[0]
                            base_balance = balances.get(base_asset, 0)

                            if base_balance > 0:
                                self.main_connector.place_order(
                                    symbol=symbol,
                                    side=OrderSide.SELL,
                                    order_type=OrderType.MARKET,
                                    amount=base_balance,
                                )
                            else:
                                logger.info(
                                    f"No {base_asset} balance to sell for {symbol}"
                                )
                        else:
                            logger.info(
                                f"No action needed for spot SELL position for {symbol}"
                            )
                    else:
                        # For derivatives markets, use close_position
                        logger.info(f"Closing position for {symbol}")
                        self.main_connector.close_position(symbol)
                except Exception as e:
                    logger.error(f"Failed to close main position for {symbol}: {e}")
                    success = False

            # Close hedge positions if using a separate connector
            if self.hedge_connector != self.main_connector:
                for symbol in active_symbols:
                    trade = self.active_trades.get(symbol, {})

                    # Skip if no hedge position
                    if "hedge_position" not in trade:
                        continue

                    market_type = trade.get("market_type", "SPOT")

                    try:
                        # For spot markets with BUY hedge positions, we need to SELL the asset to close
                        if (
                            market_type == "SPOT"
                            and not self.hedge_connector.supports_derivatives
                        ):
                            hedge_side = trade.get("hedge_position", {}).get("side")

                            if hedge_side == "BUY":
                                # For spot BUY positions, we sell the asset to close
                                logger.info(
                                    f"Closing spot hedge position for {symbol} (selling asset)"
                                )

                                # Get our balance of the base asset
                                balances = self.hedge_connector.get_account_balance()
                                base_asset = symbol.split("-")[0]
                                base_balance = balances.get(base_asset, 0)

                                if base_balance > 0:
                                    self.hedge_connector.place_order(
                                        symbol=symbol,
                                        side=OrderSide.SELL,
                                        order_type=OrderType.MARKET,
                                        amount=base_balance,
                                    )
                                else:
                                    logger.info(
                                        f"No {base_asset} balance to sell for hedge {symbol}"
                                    )
                            else:
                                logger.info(
                                    f"No action needed for spot SELL hedge position for {symbol}"
                                )
                        else:
                            # For derivatives markets, use close_position
                            logger.info(f"Closing hedge position for {symbol}")
                            self.hedge_connector.close_position(symbol)
                    except Exception as e:
                        logger.error(
                            f"Failed to close hedge position for {symbol}: {e}"
                        )
                        success = False

            # Clear active trades
            self.active_trades.clear()

            return success

        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return False
