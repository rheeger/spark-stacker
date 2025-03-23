import abc
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class OrderStatus(str, Enum):
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"

class TimeInForce(str, Enum):
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date

class MarketType(str, Enum):
    SPOT = "SPOT"
    PERPETUAL = "PERPETUAL"
    FUTURES = "FUTURES"

class BaseConnector(abc.ABC):
    """
    Abstract base class for all exchange connectors.
    
    This class defines the interface that all exchange connectors must implement,
    ensuring consistent interaction with different exchanges.
    
    Supports both spot and derivatives markets with appropriate functionality.
    """
    
    def __init__(self, name: str, exchange_type: str, market_types: List[MarketType] = None):
        """
        Initialize the connector.
        
        Args:
            name: A unique name for this connector instance
            exchange_type: Type of exchange (e.g., 'coinbase', 'hyperliquid', 'synthetix')
            market_types: List of market types supported by this connector instance
        """
        self.name = name
        self.exchange_type = exchange_type
        self.market_types = market_types or [MarketType.SPOT]  # Default to spot markets
        self._is_connected = False  # Track connection state
        
        # Logger attributes for specialized logs
        self.balance_logger = None
        self.markets_logger = None
        self.orders_logger = None
    
    def setup_loggers(self):
        """Set up dedicated loggers for this connector."""
        from app.utils.logging_setup import (
            setup_connector_balance_logger,
            setup_connector_markets_logger,
            setup_connector_orders_logger
        )
        
        # Create dedicated loggers
        self.balance_logger = setup_connector_balance_logger(self.name)
        self.markets_logger = setup_connector_markets_logger(self.name)
        self.orders_logger = setup_connector_orders_logger(self.name)
    
    @property
    def is_connected(self) -> bool:
        """
        Check if the connector is currently connected.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self._is_connected
    
    @property
    def supports_spot(self) -> bool:
        """
        Check if the connector supports spot markets.
        
        Returns:
            bool: True if spot markets are supported
        """
        return MarketType.SPOT in self.market_types
    
    @property
    def supports_derivatives(self) -> bool:
        """
        Check if the connector supports derivative markets.
        
        Returns:
            bool: True if perpetual or futures markets are supported
        """
        return MarketType.PERPETUAL in self.market_types or MarketType.FUTURES in self.market_types
    
    def cleanup(self) -> None:
        """
        Clean up any resources used by the connector.
        
        This should be called when the connector is no longer needed
        to ensure proper resource cleanup and connection termination.
        """
        try:
            # Try to disconnect if we're connected
            if self.is_connected:
                self.disconnect()
                
            # Close any logging handlers
            if hasattr(self, 'balance_logger') and self.balance_logger:
                for handler in self.balance_logger.handlers[:]:
                    handler.close()
                    self.balance_logger.removeHandler(handler)
                    
            if hasattr(self, 'markets_logger') and self.markets_logger:
                for handler in self.markets_logger.handlers[:]:
                    handler.close()
                    self.markets_logger.removeHandler(handler)
                    
            if hasattr(self, 'orders_logger') and self.orders_logger:
                for handler in self.orders_logger.handlers[:]:
                    handler.close()
                    self.orders_logger.removeHandler(handler)
        except Exception as e:
            # Just log the error, don't raise during cleanup
            import logging
            logging.error(f"Error during connector cleanup: {e}")
    
    @abc.abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the exchange API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the exchange API.
        
        Returns:
            bool: True if disconnection is successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def get_markets(self) -> List[Dict[str, Any]]:
        """
        Get available markets/trading pairs from the exchange.
        
        Returns:
            List[Dict[str, Any]]: List of available markets with their details,
                                 including market_type field indicating SPOT or PERPETUAL
        """
        pass
    
    @abc.abstractmethod
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker information for a specific market.
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            
        Returns:
            Dict[str, Any]: Ticker information including price, volume, etc.
        """
        pass
    
    @abc.abstractmethod
    def get_orderbook(self, symbol: str, depth: int = 10) -> Dict[str, List[List[float]]]:
        """
        Get the current order book for a specific market.
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            depth: Number of price levels to retrieve
            
        Returns:
            Dict with 'bids' and 'asks' lists of [price, size] pairs
        """
        pass
    
    @abc.abstractmethod
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balance information.
        
        Returns:
            Dict[str, float]: Asset balances
        """
        pass
    
    @abc.abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current open positions.
        
        For spot markets, this may return an empty list or account holdings.
        For derivative markets, this returns actual leveraged positions.
        
        Returns:
            List[Dict[str, Any]]: List of open positions with details
        """
        pass
    
    @abc.abstractmethod
    def place_order(self, 
                   symbol: str, 
                   side: OrderSide,
                   order_type: OrderType,
                   amount: float,
                   leverage: Optional[float] = None,
                   price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a new order on the exchange.
        
        For spot markets, leverage parameter is ignored.
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            side: Buy or sell
            order_type: Market or limit
            amount: The amount/size to trade
            leverage: Leverage multiplier to use (for derivative markets only)
            price: Limit price (required for limit orders)
            
        Returns:
            Dict[str, Any]: Order details including order ID
        """
        pass
    
    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: The ID of the order to cancel
            
        Returns:
            bool: True if successfully canceled, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id: The ID of the order to check
            
        Returns:
            Dict[str, Any]: Order status and details
        """
        pass
    
    @abc.abstractmethod
    def close_position(self, symbol: str, position_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Close an open position.
        
        For spot markets, this is equivalent to selling the entire holding.
        For derivative markets, this closes the leveraged position.
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            position_id: Optional position ID if the exchange requires it
            
        Returns:
            Dict[str, Any]: Result of the close operation
        """
        pass
    
    @abc.abstractmethod
    def get_historical_candles(self, symbol: str, interval: str, 
                              start_time: Optional[int] = None, 
                              end_time: Optional[int] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical candlestick data.
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            interval: Time interval (e.g., '1m', '5m', '1h')
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Maximum number of candles to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of candlestick data
        """
        pass
    
    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get current funding rate for a perpetual market.
        
        For spot markets, this returns a placeholder as funding rates don't apply.
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            
        Returns:
            Dict[str, Any]: Funding rate information
        """
        if not self.supports_derivatives:
            return {"symbol": symbol, "funding_rate": 0.0, "next_funding_time": 0, "applicable": False}
        else:
            # This should be implemented by derivative connectors
            raise NotImplementedError("This method must be implemented by derivatives connector")
    
    def get_leverage_tiers(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get information about leverage tiers and limits.
        
        For spot markets, this returns empty list as leverage doesn't apply.
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            
        Returns:
            List[Dict[str, Any]]: Information about leverage tiers
        """
        if not self.supports_derivatives:
            return []
        else:
            # This should be implemented by derivative connectors
            raise NotImplementedError("This method must be implemented by derivatives connector")
    
    def calculate_margin_requirement(self, symbol: str, size: float, leverage: float = 1.0) -> Tuple[float, float]:
        """
        Calculate initial and maintenance margin requirements.
        
        For spot markets, the margin is simply the position size (leverage is always 1.0).
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            size: Position size
            leverage: Leverage multiplier (defaults to 1.0 for spot)
            
        Returns:
            Tuple[float, float]: (initial_margin, maintenance_margin)
        """
        if not self.supports_derivatives:
            # For spot markets, margin is simply the position size
            return size, size
        else:
            # Default implementation for derivatives
            initial_margin = size / leverage
            maintenance_margin = initial_margin * 0.5
            return initial_margin, maintenance_margin
    
    @abc.abstractmethod
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get the details of an order.
        
        Args:
            order_id: The ID of the order to retrieve
            
        Returns:
            Dict[str, Any]: Order details
        """
        pass
    
    @abc.abstractmethod
    def set_leverage(self, symbol: str, leverage: float) -> Dict[str, Any]:
        """
        Set leverage for a symbol.
        
        For spot markets, this is a no-op that returns success.
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            leverage: Leverage multiplier to set
            
        Returns:
            Dict[str, Any]: Result of setting leverage
        """
        pass
    
    @abc.abstractmethod
    def get_optimal_limit_price(self, symbol: str, side: OrderSide, amount: float) -> Dict[str, Any]:
        """
        Calculate the optimal limit price for immediate execution based on order book depth.
        
        This method analyzes the order book to determine a limit price that would be likely
        to fill immediately while still getting the best possible price. It also handles
        large orders by calculating slippage and suggesting order batching if necessary.
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            side: Buy or sell
            amount: The amount/size to trade
            
        Returns:
            Dict containing:
                'price': Recommended limit price for immediate execution
                'batches': List of recommended order batches if the amount is large
                'total_cost': Estimated total cost/proceeds of the order
                'slippage': Estimated slippage percentage from best price
                'enough_liquidity': Boolean indicating if there's enough liquidity
        """
        pass 