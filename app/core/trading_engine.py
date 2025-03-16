import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import pandas as pd

from app.connectors.base_connector import BaseConnector, OrderSide, OrderType
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
    """
    
    def __init__(
        self,
        main_connector: BaseConnector,
        hedge_connector: Optional[BaseConnector] = None,
        risk_manager: Optional[RiskManager] = None,
        dry_run: bool = True,
        polling_interval: int = 60,
        max_parallel_trades: int = 1
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
        """
        self.main_connector = main_connector
        self.hedge_connector = hedge_connector or main_connector  # Use main if not provided
        self.risk_manager = risk_manager or RiskManager()
        self.dry_run = dry_run
        self.polling_interval = polling_interval
        self.max_parallel_trades = max_parallel_trades
        
        # Trading state
        self.state = TradingState.IDLE
        self.active_trades: Dict[str, Dict[str, Any]] = {}  # Symbol -> Trade info
        self.pending_signals: List[Signal] = []
        
        # Monitoring thread
        self.monitor_thread = None
        self.stop_event = threading.Event()
    
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
            # Initialize connectors if not already connected
            if not self.main_connector.connect():
                logger.error("Failed to connect to main exchange")
                return False
            
            if self.hedge_connector != self.main_connector and not self.hedge_connector.connect():
                logger.error("Failed to connect to hedge exchange")
                return False
            
            # Start the monitoring thread
            self.stop_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_positions, daemon=True)
            self.monitor_thread.start()
            
            self.state = TradingState.RUNNING
            logger.info(f"Trading engine started (dry_run={self.dry_run})")
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
            logger.warning(f"Cannot pause: trading engine is not running (state={self.state})")
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
            logger.warning(f"Cannot resume: trading engine is not paused (state={self.state})")
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
            logger.warning(f"Cannot process signal: engine not in correct state (state={self.state})")
            self.pending_signals.append(signal)
            return False
        
        try:
            logger.info(f"Processing signal: {signal}")
            
            # Get symbol and check if it's a liveness test
            symbol = signal.symbol
            is_liveness_test = signal and hasattr(signal, 'indicator') and 'liveness' in signal.indicator.lower()
            
            # For liveness test, close any existing positions for this symbol first
            if is_liveness_test and symbol in self.active_trades:
                logger.info(f"Liveness test: Closing existing position for {symbol} before placing a new one")
                
                # If in dry run mode, just remove the trade
                if self.dry_run:
                    del self.active_trades[symbol]
                    logger.info(f"Removed existing dry run trade for {symbol}")
                else:
                    # Attempt to close the real position
                    try:
                        self.main_connector.close_position(symbol)
                        logger.info(f"Closed existing position for {symbol}")
                        # Give the exchange a moment to process the close
                        time.sleep(2)
                        # Remove from active trades
                        del self.active_trades[symbol]
                    except Exception as e:
                        logger.error(f"Error closing existing position for {symbol}: {e}")
                        # Continue anyway for liveness test
            
            # For normal signals (not liveness test), check if we're already trading this symbol
            elif not is_liveness_test and symbol in self.active_trades:
                logger.warning(f"Already have an active trade for {symbol}, cannot process new signal")
                return False
            
            # Check if we've reached max parallel trades (except for liveness test)
            if not is_liveness_test and len(self.active_trades) >= self.max_parallel_trades:
                logger.warning(f"Maximum parallel trades ({self.max_parallel_trades}) reached, cannot process new signal")
                self.pending_signals.append(signal)
                return False
            
            # Convert signal direction to order side
            if signal.direction == SignalDirection.BUY:
                main_side = OrderSide.BUY
                hedge_side = OrderSide.SELL
            elif signal.direction == SignalDirection.SELL:
                main_side = OrderSide.SELL
                hedge_side = OrderSide.BUY
            else:
                logger.warning(f"Neutral signal received, no action taken")
                return False
            
            # Extract price from signal parameters if available
            price = signal.params.get('price')
            if price:
                logger.info(f"Using price from signal: {price}")
            
            # Special case for LivenessTest
            if is_liveness_test:
                logger.info("Using fixed position size for liveness test")
                main_size = 0.001  # Very small position for BTC
                
                # Use a larger size for stablecoins
                if 'PYUSD' in symbol or 'USDC' in symbol or 'USDT' in symbol:
                    main_size = 10.0  # $10 worth of stablecoin
                    logger.info(f"Using stablecoin position size: {main_size}")
                
                adjusted_main_leverage = 1.0
            else:
                # Normal position size calculation
                main_size, adjusted_main_leverage = self.risk_manager.calculate_position_size(
                    exchange=self.main_connector,
                    symbol=symbol,
                    available_balance=self.main_connector.get_account_balance(),
                    confidence=signal.confidence,
                    signal_side=main_side,
                    leverage=10.0,
                    stop_loss_pct=10.0,
                    price=price
                )
            
            if main_size <= 0:
                logger.error(f"Invalid position size calculated: {main_size}")
                return False
            
            # Validate the trade with risk manager (skip for liveness test)
            if not is_liveness_test:
                is_valid, reason = self.risk_manager.validate_trade(
                    exchange=self.main_connector,
                    symbol=symbol,
                    position_size=main_size,
                    leverage=adjusted_main_leverage,
                    side=main_side
                )
                
                if not is_valid:
                    logger.warning(f"Trade validation failed: {reason}")
                    return False
            else:
                logger.info("Bypassing risk validation for liveness test")
                is_valid = True
            
            # Calculate hedge position parameters
            hedge_size, adjusted_hedge_leverage = self.risk_manager.calculate_hedge_parameters(
                main_position_size=main_size,
                main_leverage=adjusted_main_leverage,
                hedge_ratio=0.2,
                max_hedge_leverage=5.0
            )
            
            # Check if the hedge connector has enough balance
            if self.hedge_connector != self.main_connector:
                hedge_balance = self.hedge_connector.get_account_balance()
                hedge_available = sum(hedge_balance.values())
                
                if hedge_available < hedge_size:
                    logger.error(f"Insufficient balance on hedge exchange: {hedge_available}, needed {hedge_size}")
                    return False
            
            # Log the trade plan
            logger.info(f"Trade plan for {symbol}:")
            logger.info(f"  Main position: {main_side.value} {main_size:.2f} @ {adjusted_main_leverage:.1f}x")
            logger.info(f"  Hedge position: {hedge_side.value} {hedge_size:.2f} @ {adjusted_hedge_leverage:.1f}x")
            
            # Execute trades in dry run mode (simulation)
            if self.dry_run:
                logger.info("DRY RUN MODE: No actual trades executed")
                
                # Record the simulated trade in active_trades
                current_price = self.main_connector.get_ticker(symbol).get('last_price', 0.0)
                
                trade_record = {
                    'symbol': symbol,
                    'timestamp': int(time.time() * 1000),
                    'main_position': {
                        'exchange': 'main',
                        'side': main_side.value,
                        'size': main_size,
                        'leverage': adjusted_main_leverage,
                        'entry_price': current_price,
                        'order_id': f"dry_run_main_{int(time.time())}"
                    },
                    'hedge_position': {
                        'exchange': 'hedge',
                        'side': hedge_side.value,
                        'size': hedge_size,
                        'leverage': adjusted_hedge_leverage,
                        'entry_price': current_price,
                        'order_id': f"dry_run_hedge_{int(time.time())}"
                    },
                    'stop_loss_pct': 10.0,
                    'take_profit_pct': 20.0,
                    'status': 'open'
                }
                
                self.active_trades[symbol] = trade_record
                return True
            
            # Execute main position
            # Check if this is a Coinbase connector
            is_coinbase = "coinbase" in self.main_connector.__class__.__name__.lower()
            
            # Always use LIMIT orders for Coinbase
            if is_coinbase:
                if not price or price <= 0:
                    price = 1.0  # Default price for testing
                logger.info(f"Using LIMIT order for Coinbase with price {price}")
                order_type = OrderType.LIMIT
            else:
                order_type = OrderType.MARKET
            
            main_order = self.main_connector.place_order(
                symbol=symbol,
                side=main_side,
                order_type=order_type,
                amount=main_size,
                leverage=adjusted_main_leverage,
                price=price
            )
            
            if main_order.get('error'):
                logger.error(f"Failed to place main order: {main_order.get('error')}")
                return False
            
            logger.info(f"Main order placed: {main_order}")
            
            # Small delay to ensure main order is processed
            time.sleep(1)
            
            # Execute hedge position if hedge connector is available
            if not self.hedge_connector:
                logger.warning("No hedge connector available, skipping hedge position")
            else:
                # Check if this is a Coinbase connector
                is_coinbase_hedge = "coinbase" in self.hedge_connector.__class__.__name__.lower()
                
                # Always use LIMIT orders for Coinbase
                if is_coinbase_hedge:
                    if not price or price <= 0:
                        price = 1.0  # Default price for testing
                    logger.info(f"Using LIMIT order for Coinbase hedge with price {price}")
                    hedge_order_type = OrderType.LIMIT
                else:
                    hedge_order_type = OrderType.MARKET
                
                hedge_order = self.hedge_connector.place_order(
                    symbol=symbol,
                    side=hedge_side,
                    order_type=hedge_order_type,
                    amount=hedge_size,
                    leverage=adjusted_hedge_leverage,
                    price=price
                )
                
                if hedge_order.get('error'):
                    logger.error(f"Failed to place hedge order: {hedge_order.get('error')}")
                    
                    # Try to cancel the main order if hedge fails and we have a valid order_id
                    order_id = main_order.get('order_id')
                    if order_id:
                        logger.info(f"Cancelling main order {order_id} after hedge failure")
                        self.main_connector.cancel_order(symbol=symbol, order_id=order_id)
            
            # Record the trade in active_trades
            trade_record = {
                'symbol': symbol,
                'timestamp': int(time.time() * 1000),
                'main_position': {
                    'exchange': 'main',
                    'side': main_side.value,
                    'size': main_size,
                    'leverage': adjusted_main_leverage,
                    'order_id': main_order.get('order_id', 'unknown_order_id')
                },
                'hedge_position': {
                    'exchange': 'hedge',
                    'side': hedge_side.value,
                    'size': hedge_size,
                    'leverage': adjusted_hedge_leverage,
                    'order_id': hedge_order.get('order_id', 'unknown_order_id') if hedge_order else None
                },
                'stop_loss_pct': 10.0,
                'take_profit_pct': 20.0,
                'status': 'open'
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
        signal: Optional[Signal] = None
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
            
            # Calculate position size and leverage for main position
            # Special case for LivenessTest
            if signal and hasattr(signal, 'indicator') and 'liveness' in signal.indicator.lower():
                logger.info("Using fixed position size for liveness test")
                main_size = 0.001  # Very small position for BTC
                adjusted_main_leverage = 1.0
            else:
                # Normal position size calculation
                main_size, adjusted_main_leverage = self.risk_manager.calculate_position_size(
                    exchange=self.main_connector,
                    symbol=symbol,
                    available_balance=main_available,
                    confidence=confidence,
                    signal_side=main_side,
                    leverage=main_leverage,
                    stop_loss_pct=stop_loss_pct,
                    price=price
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
                side=main_side
            )
            
            if not is_valid:
                logger.warning(f"Trade validation failed: {reason}")
                return False
            
            # Calculate hedge position parameters
            hedge_size, adjusted_hedge_leverage = self.risk_manager.calculate_hedge_parameters(
                main_position_size=main_size,
                main_leverage=adjusted_main_leverage,
                hedge_ratio=hedge_ratio,
                max_hedge_leverage=hedge_leverage
            )
            
            # Check if the hedge connector has enough balance
            if self.hedge_connector != self.main_connector:
                hedge_balance = self.hedge_connector.get_account_balance()
                hedge_available = sum(hedge_balance.values())
                
                if hedge_available < hedge_size:
                    logger.error(f"Insufficient balance on hedge exchange: {hedge_available}, needed {hedge_size}")
                    return False
            
            # Log the trade plan
            logger.info(f"Trade plan for {symbol}:")
            logger.info(f"  Main position: {main_side.value} {main_size:.2f} @ {adjusted_main_leverage:.1f}x")
            logger.info(f"  Hedge position: {hedge_side.value} {hedge_size:.2f} @ {adjusted_hedge_leverage:.1f}x")
            
            # Execute trades in dry run mode (simulation)
            if self.dry_run:
                logger.info("DRY RUN MODE: No actual trades executed")
                
                # Record the simulated trade in active_trades
                current_price = self.main_connector.get_ticker(symbol).get('last_price', 0.0)
                
                trade_record = {
                    'symbol': symbol,
                    'timestamp': int(time.time() * 1000),
                    'main_position': {
                        'exchange': 'main',
                        'side': main_side.value,
                        'size': main_size,
                        'leverage': adjusted_main_leverage,
                        'entry_price': current_price,
                        'order_id': f"dry_run_main_{int(time.time())}"
                    },
                    'hedge_position': {
                        'exchange': 'hedge',
                        'side': hedge_side.value,
                        'size': hedge_size,
                        'leverage': adjusted_hedge_leverage,
                        'entry_price': current_price,
                        'order_id': f"dry_run_hedge_{int(time.time())}"
                    },
                    'stop_loss_pct': stop_loss_pct,
                    'take_profit_pct': take_profit_pct,
                    'status': 'open'
                }
                
                self.active_trades[symbol] = trade_record
                return True
            
            # Execute main position
            # Check if this is a Coinbase connector
            is_coinbase = "coinbase" in self.main_connector.__class__.__name__.lower()
            
            # Always use LIMIT orders for Coinbase
            if is_coinbase:
                if not price or price <= 0:
                    price = 1.0  # Default price for testing
                logger.info(f"Using LIMIT order for Coinbase with price {price}")
                order_type = OrderType.LIMIT
            else:
                order_type = OrderType.MARKET
            
            main_order = self.main_connector.place_order(
                symbol=symbol,
                side=main_side,
                order_type=order_type,
                amount=main_size,
                leverage=adjusted_main_leverage,
                price=price
            )
            
            if main_order.get('error'):
                logger.error(f"Failed to place main order: {main_order.get('error')}")
                return False
            
            logger.info(f"Main order placed: {main_order}")
            
            # Small delay to ensure main order is processed
            time.sleep(1)
            
            # Execute hedge position if hedge connector is available
            if not self.hedge_connector:
                logger.warning("No hedge connector available, skipping hedge position")
            else:
                # Check if this is a Coinbase connector
                is_coinbase_hedge = "coinbase" in self.hedge_connector.__class__.__name__.lower()
                
                # Always use LIMIT orders for Coinbase
                if is_coinbase_hedge:
                    if not price or price <= 0:
                        price = 1.0  # Default price for testing
                    logger.info(f"Using LIMIT order for Coinbase hedge with price {price}")
                    hedge_order_type = OrderType.LIMIT
                else:
                    hedge_order_type = OrderType.MARKET
                
                hedge_order = self.hedge_connector.place_order(
                    symbol=symbol,
                    side=hedge_side,
                    order_type=hedge_order_type,
                    amount=hedge_size,
                    leverage=adjusted_hedge_leverage,
                    price=price
                )
                
                if hedge_order.get('error'):
                    logger.error(f"Failed to place hedge order: {hedge_order.get('error')}")
                    
                    # Try to cancel the main order if hedge fails and we have a valid order_id
                    order_id = main_order.get('order_id')
                    if order_id:
                        logger.info(f"Cancelling main order {order_id} after hedge failure")
                        self.main_connector.cancel_order(symbol=symbol, order_id=order_id)
            
            # Record the trade in active_trades
            trade_record = {
                'symbol': symbol,
                'timestamp': int(time.time() * 1000),
                'main_position': {
                    'exchange': 'main',
                    'side': main_side.value,
                    'size': main_size,
                    'leverage': adjusted_main_leverage,
                    'order_id': main_order.get('order_id', 'unknown_order_id')
                },
                'hedge_position': {
                    'exchange': 'hedge',
                    'side': hedge_side.value,
                    'size': hedge_size,
                    'leverage': adjusted_hedge_leverage,
                    'order_id': hedge_order.get('order_id', 'unknown_order_id') if hedge_order else None
                },
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'status': 'open'
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
        
        while not self.stop_event.is_set():
            try:
                if self.state not in [TradingState.RUNNING, TradingState.PAUSED]:
                    # If not running or paused, just check if we should stop
                    time.sleep(1)
                    continue
                
                # Skip position checks if paused
                if self.state != TradingState.PAUSED:
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
                
                # Process any pending signals if we have capacity
                if self.state == TradingState.RUNNING and len(self.active_trades) < self.max_parallel_trades and self.pending_signals:
                    signal = self.pending_signals.pop(0)
                    logger.info(f"Processing pending signal: {signal}")
                    self.process_signal(signal)
                
                # Sleep until next check
                time.sleep(self.polling_interval)
            
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                time.sleep(max(1, self.polling_interval // 4))  # Shorter sleep on error
        
        logger.info("Position monitoring thread stopped")
    
    def _check_active_trades(
        self,
        main_positions: List[Dict[str, Any]],
        hedge_positions: List[Dict[str, Any]]
    ) -> None:
        """
        Check the status of active trades and manage them.
        
        Args:
            main_positions: List of positions from main exchange
            hedge_positions: List of positions from hedge exchange
        """
        # Convert position lists to dictionary for faster lookup
        main_pos_dict = {p.get('symbol'): p for p in main_positions if p.get('symbol')}
        hedge_pos_dict = {p.get('symbol'): p for p in hedge_positions if p.get('symbol')}
        
        # Check each active trade
        for symbol, trade in list(self.active_trades.items()):
            try:
                main_pos = main_pos_dict.get(symbol)
                hedge_pos = hedge_pos_dict.get(symbol)
                
                # Check if positions still exist
                if not main_pos and not hedge_pos:
                    logger.info(f"Both positions for {symbol} are closed, removing from active trades")
                    del self.active_trades[symbol]
                    continue
                
                # Update trade record with current position details
                if main_pos:
                    trade['main_position'].update({
                        'entry_price': main_pos.get('entry_price'),
                        'mark_price': main_pos.get('mark_price'),
                        'size': main_pos.get('size'),
                        'unrealized_pnl': main_pos.get('unrealized_pnl'),
                        'liquidation_price': main_pos.get('liquidation_price')
                    })
                
                if hedge_pos:
                    trade['hedge_position'].update({
                        'entry_price': hedge_pos.get('entry_price'),
                        'mark_price': hedge_pos.get('mark_price'),
                        'size': hedge_pos.get('size'),
                        'unrealized_pnl': hedge_pos.get('unrealized_pnl'),
                        'liquidation_price': hedge_pos.get('liquidation_price')
                    })
                
                # Check if main position should be closed (stop loss, take profit, etc.)
                if main_pos:
                    should_close, reason = self.risk_manager.should_close_position(
                        main_pos,
                        trade.get('stop_loss_pct', 10.0),
                        trade.get('take_profit_pct', 20.0)
                    )
                    
                    if should_close:
                        logger.info(f"Closing main position for {symbol}: {reason}")
                        
                        if not self.dry_run:
                            self.main_connector.close_position(symbol)
                        
                        trade['main_position']['status'] = 'closing'
                        trade['main_position']['close_reason'] = reason
                
                # Check if hedge position should be adjusted
                if main_pos and hedge_pos:
                    should_adjust, reason, adjustment = self.risk_manager.manage_hedge_position(
                        main_pos, hedge_pos
                    )
                    
                    if should_adjust:
                        logger.info(f"Adjusting hedge for {symbol}: {reason}")
                        action = adjustment.get('action')
                        target = adjustment.get('position')
                        
                        if action == 'close':
                            if target == 'main' or target == 'both':
                                if not self.dry_run:
                                    self.main_connector.close_position(symbol)
                                trade['main_position']['status'] = 'closing'
                                trade['main_position']['close_reason'] = reason
                            
                            if target == 'hedge' or target == 'both':
                                if not self.dry_run:
                                    self.hedge_connector.close_position(symbol)
                                trade['hedge_position']['status'] = 'closing'
                                trade['hedge_position']['close_reason'] = reason
                        
                        elif action == 'reduce' and target == 'hedge':
                            reduction_pct = adjustment.get('reduction_pct', 50)
                            current_size = hedge_pos.get('size', 0)
                            reduce_size = abs(current_size) * reduction_pct / 100.0
                            
                            logger.info(f"Reducing hedge by {reduction_pct}% ({reduce_size} units)")
                            
                            if not self.dry_run and reduce_size > 0:
                                # Determine order side (opposite of current position)
                                reduce_side = OrderSide.BUY if hedge_pos.get('side') == 'SHORT' else OrderSide.SELL
                                
                                # Place a reduce-only order
                                self.hedge_connector.place_order(
                                    symbol=symbol,
                                    side=reduce_side,
                                    order_type=OrderType.MARKET,
                                    amount=reduce_size,
                                    leverage=hedge_pos.get('leverage', 1.0)
                                )
                            
                            trade['hedge_position']['status'] = 'reducing'
                            trade['hedge_position']['reduce_reason'] = reason
                
                # Update overall trade status
                main_status = trade.get('main_position', {}).get('status', 'open')
                hedge_status = trade.get('hedge_position', {}).get('status', 'open')
                
                if main_status == 'closing' and hedge_status == 'closing':
                    trade['status'] = 'closing'
                elif main_status != 'open' or hedge_status != 'open':
                    trade['status'] = 'adjusting'
                else:
                    trade['status'] = 'open'
            
            except Exception as e:
                logger.error(f"Error checking trade for {symbol}: {e}")
    
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
            # Close main positions
            main_positions = self.main_connector.get_positions()
            for position in main_positions:
                symbol = position.get('symbol')
                if not symbol or position.get('size', 0) == 0:
                    continue
                
                try:
                    logger.info(f"Closing main position for {symbol}")
                    self.main_connector.close_position(symbol)
                except Exception as e:
                    logger.error(f"Failed to close main position for {symbol}: {e}")
                    success = False
            
            # Close hedge positions if using a separate connector
            if self.hedge_connector != self.main_connector:
                hedge_positions = self.hedge_connector.get_positions()
                for position in hedge_positions:
                    symbol = position.get('symbol')
                    if not symbol or position.get('size', 0) == 0:
                        continue
                    
                    try:
                        logger.info(f"Closing hedge position for {symbol}")
                        self.hedge_connector.close_position(symbol)
                    except Exception as e:
                        logger.error(f"Failed to close hedge position for {symbol}: {e}")
                        success = False
            
            # Clear active trades
            self.active_trades.clear()
            
            return success
        
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return False
    
    def _generate_test_data(self, symbol: str, intervals: int = 60) -> pd.DataFrame:
        """
        Generate mock price data for testing indicators.
        
        Args:
            symbol: Market symbol
            intervals: Number of data points to generate
            
        Returns:
            DataFrame with OHLCV data
        """
        import random
        import numpy as np
        
        logger.info(f"Generating test data for {symbol} ({intervals} points)")
        
        # Get current price from exchange
        try:
            ticker = self.main_connector.get_ticker(symbol)
            start_price = ticker.get("price") or ticker.get("last_price")
            
            if not start_price and hasattr(self.main_connector, "get_current_price"):
                # Try alternative method if available
                start_price = self.main_connector.get_current_price(symbol)
            
            # Fallback to mock price if needed
            if not start_price:
                start_price = 2000.0  # Default value for testing
                logger.warning(f"Could not get current price for {symbol}, using mock price: {start_price}")
        except Exception as e:
            logger.warning(f"Error getting current price: {e}, using mock price")
            start_price = 2000.0  # Default value for testing
            
        # Generate more volatile price data with a trend
        timestamp = int(time.time()) - intervals * 60  # Start intervals minutes ago
        
        data = []
        price = float(start_price)
        trend = random.choice([-1, 1])  # Random initial trend
        trend_strength = random.uniform(0.3, 0.7)  # How strong the trend is
        
        for i in range(intervals):
            # Occasionally change trend
            if random.random() < 0.1:  # 10% chance to change trend
                trend = -trend
            
            # Generate more volatile price changes (-2% to +2%) with trend bias
            price_change = price * (random.uniform(-0.02, 0.02) + (trend * trend_strength * 0.01))
            price += price_change
            
            # Generate OHLCV data with more volatility
            open_price = price - price * random.uniform(-0.01, 0.01)
            high_price = max(price, open_price) + price * random.uniform(0, 0.015)
            low_price = min(price, open_price) - price * random.uniform(0, 0.015)
            close_price = price + price * random.uniform(-0.01, 0.01)
            volume = random.uniform(10, 1000)  # More realistic volume range
            
            # Create the data point
            data.append({
                "timestamp": (timestamp + i * 60) * 1000,  # Convert to milliseconds
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "symbol": symbol
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def run_test_signal_generation(self, indicator_name: str, symbol: str) -> Optional[Signal]:
        """
        Generate test data and run an indicator to get a signal for testing.
        
        Args:
            indicator_name: Name of the registered indicator
            symbol: Market symbol to use
            
        Returns:
            Signal object if generated, None otherwise
        """
        from app.indicators.indicator_factory import IndicatorFactory
        
        logger.info(f"Running test signal generation for {indicator_name} on {symbol}")
        
        # Get the indicators from the factory
        indicator_config = {
            "name": indicator_name,
            "enabled": True,
            "parameters": {
                "min_spread": 0.0002,
                "max_spread": 0.01
            } if "usdc" in indicator_name.lower() else {
                "short_period": 5,
                "long_period": 20,
                "min_points": 30
            }
        }
        
        indicators = IndicatorFactory.create_indicators_from_config([indicator_config])
        
        if not indicators or indicator_name not in indicators:
            logger.error(f"Indicator {indicator_name} not found or could not be created")
            return None
        
        indicator = indicators[indicator_name]
        logger.info(f"Using indicator: {indicator}")
        
        # Generate test data
        data = self._generate_test_data(symbol, intervals=60)
        
        if data.empty:
            logger.error("Failed to generate test data")
            return None
        
        # Process the data with the indicator
        try:
            processed_data, signal = indicator.process(data)
            
            if signal:
                logger.info(f"Generated signal: {signal}")
                # Process the signal
                self.process_signal(signal)
                return signal
            else:
                logger.info("No signal generated")
                return None
        except Exception as e:
            logger.error(f"Error processing indicator data: {e}")
            return None

    def _execute_fixed_test_trade(
        self,
        symbol: str,
        side: OrderSide,
        confidence: float,
        position_size: float,
        leverage: float,
        price: float
    ) -> bool:
        """
        Execute a fixed test trade with a main position and smaller opposite hedge.
        
        Args:
            symbol: Market symbol (e.g., 'ETH')
            side: Side for the main position (BUY or SELL)
            confidence: Signal confidence (0-1)
            position_size: Fixed position size for the trade
            leverage: Fixed leverage for the trade
            price: Fixed price for the trade
            
        Returns:
            bool: True if trade was executed, False otherwise
        """
        try:
            # Get account balance for the main connector but don't validate for liveness test
            main_balance = self.main_connector.get_account_balance()
            main_available = sum(main_balance.values())
            
            # Skip detailed validation for liveness test - we just want it to work for testing
            logger.info(f"Skipping validation for liveness test trade")
            
            # Log the trade plan
            logger.info(f"Trade plan for {symbol}:")
            logger.info(f"  Main position: {side.value} {position_size:.2f} @ {leverage:.1f}x")
            
            # Execute trades in dry run mode (simulation)
            if self.dry_run:
                logger.info("DRY RUN MODE: No actual trades executed")
                
                # Record the simulated trade in active_trades
                trade_record = {
                    'symbol': symbol,
                    'timestamp': int(time.time() * 1000),
                    'main_position': {
                        'exchange': 'main',
                        'side': side.value,
                        'size': position_size,
                        'leverage': leverage,
                        'entry_price': price,
                        'order_id': f"dry_run_main_{int(time.time())}"
                    },
                    'stop_loss_pct': 10.0,
                    'take_profit_pct': 20.0,
                    'status': 'open'
                }
                
                self.active_trades[symbol] = trade_record
                return True
            
            # Execute main position
            # Check if this is a Coinbase connector
            is_coinbase = "coinbase" in self.main_connector.__class__.__name__.lower()
            
            # Always use LIMIT orders for Coinbase
            if is_coinbase:
                logger.info(f"Using LIMIT order for Coinbase with price {price}")
                order_type = OrderType.LIMIT
            else:
                order_type = OrderType.MARKET
            
            main_order = self.main_connector.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                amount=position_size,
                leverage=leverage,
                price=price
            )
            
            if main_order.get('error'):
                logger.error(f"Failed to place main order: {main_order.get('error')}")
                return False
            
            logger.info(f"Main order placed: {main_order}")
            
            # Record the trade in active_trades (using dictionary access for order_id)
            trade_record = {
                'symbol': symbol,
                'timestamp': int(time.time() * 1000),
                'main_position': {
                    'exchange': 'main',
                    'side': side.value,
                    'size': position_size,
                    'leverage': leverage,
                    'order_id': main_order.get('order_id', 'unknown_order_id')
                },
                'stop_loss_pct': 10.0,
                'take_profit_pct': 20.0,
                'status': 'open'
            }
            
            self.active_trades[symbol] = trade_record
            return True
        
        except Exception as e:
            logger.error(f"Error executing fixed test trade: {e}")
            return False 