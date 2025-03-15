import pytest
from abc import ABC, abstractmethod
from unittest.mock import MagicMock

from app.connectors.base_connector import BaseConnector, OrderSide, OrderType


class TestConnector(BaseConnector):
    """Test implementation of BaseConnector for testing purposes."""
    
    def __init__(self, wallet_address=None, private_key=None, rpc_url=None):
        super().__init__(name="test_connector", exchange_type="test")
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.rpc_url = rpc_url
        self.connected = False
        
        # Mock responses for testing
        self.mock_balance = {"USD": 10000.0, "ETH": 5.0, "BTC": 0.5}
        self.mock_markets = [
            {"symbol": "ETH", "base_currency": "ETH", "quote_currency": "USD"},
            {"symbol": "BTC", "base_currency": "BTC", "quote_currency": "USD"}
        ]
        self.mock_tickers = {
            "ETH": {"symbol": "ETH", "last_price": 1500.0, "bid": 1499.0, "ask": 1501.0},
            "BTC": {"symbol": "BTC", "last_price": 25000.0, "bid": 24990.0, "ask": 25010.0}
        }
        self.mock_positions = [
            {"symbol": "ETH", "size": 1.0, "side": "LONG", "entry_price": 1450.0}
        ]
        self.last_order_id = 0
    
    def connect(self):
        """Connect to the exchange."""
        self.connected = True
        return True
    
    def disconnect(self):
        """Disconnect from the exchange."""
        self.connected = False
        return True
    
    def get_account_balance(self):
        """Get account balance."""
        return self.mock_balance
    
    def get_markets(self):
        """Get available markets."""
        return self.mock_markets
    
    def get_ticker(self, symbol):
        """Get ticker for a symbol."""
        return self.mock_tickers.get(symbol, {})
    
    def get_orderbook(self, symbol, depth=10):
        """Get order book for a symbol."""
        return {
            "bids": [[1490.0, 1.0], [1480.0, 2.0]],
            "asks": [[1510.0, 1.0], [1520.0, 2.0]]
        }
    
    def get_positions(self):
        """Get current positions."""
        return self.mock_positions
    
    def place_order(self, symbol, side, order_type, amount, leverage, price=None):
        """Place an order."""
        self.last_order_id += 1
        return {
            "order_id": f"test_order_{self.last_order_id}",
            "symbol": symbol,
            "side": side.value,
            "type": order_type.value,
            "quantity": amount,
            "price": price,
            "leverage": leverage,
            "status": "filled" if order_type == OrderType.MARKET else "open"
        }
    
    def cancel_order(self, order_id):
        """Cancel an order."""
        return {"order_id": order_id, "status": "cancelled"}
    
    def get_order(self, order_id):
        """Get order details."""
        return {"order_id": order_id, "status": "filled"}
    
    def get_order_status(self, order_id):
        """Get status of an order."""
        return {"order_id": order_id, "status": "filled"}
    
    def close_position(self, symbol, position_id=None):
        """Close a position."""
        return {"symbol": symbol, "status": "closed"}
    
    def set_leverage(self, symbol, leverage):
        """Set leverage for a symbol."""
        return {"symbol": symbol, "leverage": leverage, "status": "success"}
    
    def get_historical_candles(self, symbol, interval, start_time=None, end_time=None, limit=100):
        """Get historical candlestick data."""
        return [
            {"time": 1620000000, "open": 1500.0, "high": 1510.0, "low": 1490.0, "close": 1505.0, "volume": 100.0},
            {"time": 1620001000, "open": 1505.0, "high": 1515.0, "low": 1495.0, "close": 1510.0, "volume": 150.0}
        ]
    
    def get_funding_rate(self, symbol):
        """Get funding rate for a symbol."""
        return {"symbol": symbol, "funding_rate": 0.0001, "next_funding_time": 1620010000}
    
    def get_leverage_tiers(self, symbol):
        """Get leverage tiers for a symbol."""
        return [
            {"tier": 1, "min_notional": 0, "max_notional": 100000, "max_leverage": 50},
            {"tier": 2, "min_notional": 100000, "max_notional": 500000, "max_leverage": 20}
        ]


def test_base_connector_abstract_methods():
    """Test that BaseConnector is an abstract class with required methods."""
    # Verify that BaseConnector is abstract
    assert issubclass(BaseConnector, ABC)
    
    # Create a list of abstract methods that should be defined
    required_methods = [
        'connect', 'disconnect', 'get_account_balance', 'get_markets',
        'get_ticker', 'get_positions', 'place_order', 'cancel_order',
        'get_order', 'close_position', 'set_leverage'
    ]
    
    # Check if each method is marked as abstract in BaseConnector
    for method_name in required_methods:
        method = getattr(BaseConnector, method_name, None)
        assert method is not None, f"Method {method_name} should be defined in BaseConnector"
        assert getattr(method, '__isabstractmethod__', False), f"Method {method_name} should be abstract"


def test_connector_implementation():
    """Test that TestConnector properly implements BaseConnector."""
    # Create an instance of TestConnector
    connector = TestConnector(wallet_address="0x123", private_key="abc", rpc_url="http://test")
    
    # Verify instance attributes
    assert connector.name == "test_connector"
    assert connector.exchange_type == "test"
    assert connector.wallet_address == "0x123"
    assert connector.private_key == "abc"
    assert connector.rpc_url == "http://test"
    assert connector.connected is False
    
    # Test connect method
    result = connector.connect()
    assert result is True
    assert connector.connected is True
    
    # Test disconnect method
    result = connector.disconnect()
    assert result is True
    assert connector.connected is False


def test_get_account_balance():
    """Test getting account balance."""
    connector = TestConnector()
    balance = connector.get_account_balance()
    assert balance == {"USD": 10000.0, "ETH": 5.0, "BTC": 0.5}


def test_get_markets():
    """Test getting markets."""
    connector = TestConnector()
    markets = connector.get_markets()
    assert len(markets) == 2
    assert markets[0]["symbol"] == "ETH"
    assert markets[1]["symbol"] == "BTC"


def test_get_ticker():
    """Test getting ticker information."""
    connector = TestConnector()
    
    # Test valid symbol
    ticker = connector.get_ticker("ETH")
    assert ticker["symbol"] == "ETH"
    assert ticker["last_price"] == 1500.0
    
    # Test invalid symbol
    ticker = connector.get_ticker("DOGE")
    assert ticker == {}


def test_get_positions():
    """Test getting positions."""
    connector = TestConnector()
    positions = connector.get_positions()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "ETH"
    assert positions[0]["size"] == 1.0
    assert positions[0]["side"] == "LONG"


def test_place_order():
    """Test placing orders."""
    connector = TestConnector()
    
    # Test market buy order
    order = connector.place_order(
        symbol="ETH",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=1.0,
        leverage=5.0
    )
    
    assert order["symbol"] == "ETH"
    assert order["side"] == "BUY"
    assert order["type"] == "MARKET"
    assert order["quantity"] == 1.0
    assert order["leverage"] == 5.0
    assert order["status"] == "filled"
    
    # Test limit sell order
    order = connector.place_order(
        symbol="BTC",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        amount=0.5,
        leverage=3.0,
        price=26000.0
    )
    
    assert order["symbol"] == "BTC"
    assert order["side"] == "SELL"
    assert order["type"] == "LIMIT"
    assert order["quantity"] == 0.5
    assert order["price"] == 26000.0
    assert order["leverage"] == 3.0
    assert order["status"] == "open"


def test_cancel_order():
    """Test cancelling an order."""
    connector = TestConnector()
    result = connector.cancel_order("test_order_123")
    assert result["order_id"] == "test_order_123"
    assert result["status"] == "cancelled"


def test_get_order():
    """Test getting order details."""
    connector = TestConnector()
    order = connector.get_order("test_order_456")
    assert order["order_id"] == "test_order_456"
    assert order["status"] == "filled"


def test_close_position():
    """Test closing a position."""
    connector = TestConnector()
    result = connector.close_position("ETH")
    assert result["symbol"] == "ETH"
    assert result["status"] == "closed"


def test_set_leverage():
    """Test setting leverage for a symbol."""
    connector = TestConnector()
    result = connector.set_leverage("ETH", 10.0)
    assert result["symbol"] == "ETH"
    assert result["leverage"] == 10.0
    assert result["status"] == "success"


def test_order_side_enum():
    """Test OrderSide enum values."""
    assert OrderSide.BUY.value == "BUY"
    assert OrderSide.SELL.value == "SELL"


def test_order_type_enum():
    """Test OrderType enum values."""
    assert OrderType.MARKET.value == "MARKET"
    assert OrderType.LIMIT.value == "LIMIT" 