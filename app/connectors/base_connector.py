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

class BaseConnector(abc.ABC):
    """
    Abstract base class for all exchange connectors.
    
    This class defines the interface that all exchange connectors must implement,
    ensuring consistent interaction with different exchanges.
    """
    
    def __init__(self, name: str, exchange_type: str):
        """
        Initialize the connector.
        
        Args:
            name: A unique name for this connector instance
            exchange_type: Type of exchange (e.g., 'hyperliquid', 'synthetix')
        """
        self.name = name
        self.exchange_type = exchange_type
    
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
            List[Dict[str, Any]]: List of available markets with their details
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
                   leverage: float,
                   price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a new order on the exchange.
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            side: Buy or sell
            order_type: Market or limit
            amount: The amount/size to trade
            leverage: Leverage multiplier to use
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
    
    @abc.abstractmethod
    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get current funding rate for a perpetual market.
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            
        Returns:
            Dict[str, Any]: Funding rate information
        """
        pass
    
    @abc.abstractmethod
    def get_leverage_tiers(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get information about leverage tiers and limits.
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            
        Returns:
            List[Dict[str, Any]]: Information about leverage tiers
        """
        pass
    
    def calculate_margin_requirement(self, symbol: str, size: float, leverage: float) -> Tuple[float, float]:
        """
        Calculate initial and maintenance margin requirements.
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            size: Position size
            leverage: Leverage multiplier
            
        Returns:
            Tuple[float, float]: (initial_margin, maintenance_margin)
        """
        # Default implementation that subclasses can override if needed
        initial_margin = size / leverage
        # Usually maintenance margin is a fraction of initial margin
        # This is a simplified calculation that should be overridden
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
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            leverage: Leverage multiplier to set
            
        Returns:
            Dict[str, Any]: Result of setting leverage
        """
        pass 