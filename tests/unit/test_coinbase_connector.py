import pytest
import json
from unittest.mock import MagicMock, patch, PropertyMock
import datetime
from decimal import Decimal

from app.connectors.coinbase_connector import CoinbaseConnector
from app.connectors.base_connector import OrderSide, OrderType, OrderStatus


@pytest.fixture
def mock_rest_client():
    """Mock the Coinbase REST client."""
    mock_client = MagicMock()
    
    # Mock account balances
    mock_client.get_accounts.return_value = [
        {"currency": "USD", "balance": "1000.00", "available": "1000.00"},
        {"currency": "ETH", "balance": "5.0", "available": "5.0"},
        {"currency": "BTC", "balance": "0.1", "available": "0.1"}
    ]
    
    # Mock product data
    mock_client.get_products.return_value = [
        {
            "id": "ETH-USD",
            "base_currency": "ETH",
            "quote_currency": "USD",
            "base_min_size": "0.01",
            "base_max_size": "1000.0",
            "quote_increment": "0.01",
            "status": "online"
        },
        {
            "id": "BTC-USD",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "base_min_size": "0.001",
            "base_max_size": "100.0",
            "quote_increment": "0.01",
            "status": "online"
        }
    ]
    
    # Mock ticker data
    mock_client.get_product_ticker.return_value = {
        "price": "2500.00",
        "bid": "2499.00",
        "ask": "2501.00",
        "volume": "1000.0",
        "time": datetime.datetime.now().isoformat()
    }
    
    # Mock order book
    mock_client.get_product_book.return_value = {
        "bids": [
            ["2499.00", "1.0", "1"],
            ["2498.00", "2.0", "1"]
        ],
        "asks": [
            ["2501.00", "1.0", "1"],
            ["2502.00", "2.0", "1"]
        ]
    }
    
    # Mock order placement
    mock_client.create_order.return_value = {
        "id": "test-order-id",
        "product_id": "ETH-USD",
        "side": "buy",
        "type": "limit",
        "size": "1.0",
        "price": "2500.00",
        "status": "pending",
        "created_at": datetime.datetime.now().isoformat()
    }
    
    # Mock order cancellation
    mock_client.cancel_order.return_value = ["test-order-id"]
    
    # Mock order status
    mock_client.get_order.return_value = {
        "id": "test-order-id",
        "product_id": "ETH-USD",
        "side": "buy",
        "type": "limit",
        "size": "1.0",
        "filled_size": "0.5",
        "price": "2500.00",
        "status": "open",
        "created_at": datetime.datetime.now().isoformat()
    }
    
    # Mock orders list
    mock_client.get_orders.return_value = [
        {
            "id": "test-order-id-1",
            "product_id": "ETH-USD",
            "side": "buy",
            "type": "limit",
            "size": "1.0",
            "price": "2500.00",
            "status": "open",
            "created_at": datetime.datetime.now().isoformat()
        }
    ]
    
    # Mock candles data
    mock_client.get_product_candles.return_value = [
        [1617580800, 2000.0, 2100.0, 2050.0, 2075.0, 100.0],
        [1617667200, 2075.0, 2150.0, 2075.0, 2125.0, 200.0]
    ]
    
    return mock_client


@pytest.fixture
def coinbase_connector(mock_rest_client):
    """Create a CoinbaseConnector with a mocked REST client."""
    with patch('coinbase.rest.RESTClient', return_value=mock_rest_client):
        connector = CoinbaseConnector(
            api_key="test_key",
            api_secret="test_secret",
            api_passphrase="test_passphrase",
            use_sandbox=True
        )
        # Mock the client directly to avoid actual connection
        connector.client = mock_rest_client
        return connector


def test_initialization():
    """Test CoinbaseConnector initialization."""
    connector = CoinbaseConnector(
        api_key="test_key",
        api_secret="test_secret",
        api_passphrase="test_passphrase",
        use_sandbox=True
    )
    
    assert connector.name == "coinbase"
    assert connector.exchange_type == "coinbase"
    assert connector.api_key == "test_key"
    assert connector.api_secret == "test_secret"
    assert connector.api_passphrase == "test_passphrase"
    assert connector.use_sandbox is True
    assert connector.api_url == "https://api-public.sandbox.exchange.coinbase.com"
    
    # Test production URL
    connector = CoinbaseConnector(
        api_key="test_key",
        api_secret="test_secret",
        api_passphrase="test_passphrase",
        use_sandbox=False
    )
    assert connector.api_url == "https://api.exchange.coinbase.com"


def test_connect(coinbase_connector, mock_rest_client):
    """Test connecting to the Coinbase API."""
    # Reset the client to simulate reconnection
    coinbase_connector.client = None
    
    with patch('coinbase.rest.RESTClient', return_value=mock_rest_client):
        result = coinbase_connector.connect()
        assert result is True
        assert coinbase_connector.client is not None


def test_connect_failure(coinbase_connector):
    """Test handling of connection failure."""
    # Reset the client to simulate reconnection
    coinbase_connector.client = None
    
    with patch('coinbase.rest.RESTClient', side_effect=Exception("Connection error")):
        result = coinbase_connector.connect()
        assert result is False
        assert coinbase_connector.client is None


def test_disconnect(coinbase_connector):
    """Test disconnecting from the Coinbase API."""
    result = coinbase_connector.disconnect()
    assert result is True
    assert coinbase_connector.client is None


def test_get_markets(coinbase_connector, mock_rest_client):
    """Test retrieving available markets from Coinbase."""
    markets = coinbase_connector.get_markets()
    
    assert len(markets) == 2
    assert markets[0]["symbol"] == "ETH"
    assert markets[0]["base_currency"] == "ETH"
    assert markets[0]["quote_currency"] == "USD"
    assert markets[0]["min_size"] == 0.01
    assert markets[0]["max_size"] == 1000.0
    assert markets[0]["price_increment"] == 0.01
    
    # Verify the method was called
    mock_rest_client.get_products.assert_called_once()


def test_get_ticker(coinbase_connector, mock_rest_client):
    """Test retrieving ticker information for a specific market."""
    ticker = coinbase_connector.get_ticker("ETH")
    
    assert ticker["symbol"] == "ETH"
    assert ticker["price"] == 2500.0
    assert ticker["bid"] == 2499.0
    assert ticker["ask"] == 2501.0
    assert ticker["volume"] == 1000.0
    assert "timestamp" in ticker
    
    # Verify the method was called with correct parameters
    mock_rest_client.get_product_ticker.assert_called_once_with(product_id="ETH-USD")


def test_get_orderbook(coinbase_connector, mock_rest_client):
    """Test retrieving order book for a specific market."""
    orderbook = coinbase_connector.get_orderbook("ETH")
    
    assert "bids" in orderbook
    assert "asks" in orderbook
    assert len(orderbook["bids"]) == 2
    assert len(orderbook["asks"]) == 2
    assert orderbook["bids"][0][0] == 2499.0  # Price
    assert orderbook["bids"][0][1] == 1.0     # Size
    
    # Verify the method was called with correct parameters
    mock_rest_client.get_product_book.assert_called_once_with(product_id="ETH-USD", level=2)


def test_get_account_balance(coinbase_connector, mock_rest_client):
    """Test retrieving account balances from Coinbase."""
    balances = coinbase_connector.get_account_balance()
    
    assert "USD" in balances
    assert "ETH" in balances
    assert "BTC" in balances
    assert balances["USD"] == 1000.0
    assert balances["ETH"] == 5.0
    assert balances["BTC"] == 0.1
    
    # Verify the method was called
    mock_rest_client.get_accounts.assert_called_once()


def test_get_positions(coinbase_connector, mock_rest_client):
    """Test retrieving current positions."""
    positions = coinbase_connector.get_positions()
    
    assert len(positions) == 1
    assert positions[0]["symbol"] == "ETH"
    assert positions[0]["side"] == "LONG"
    assert positions[0]["size"] == 1.0
    assert positions[0]["entry_price"] == 2500.0
    assert positions[0]["mark_price"] == 2500.0
    
    # Verify the methods were called
    mock_rest_client.get_orders.assert_called_once_with(status="open")


def test_place_market_order(coinbase_connector, mock_rest_client):
    """Test placing a market order."""
    # Configure mock to return a market order
    mock_rest_client.create_order.return_value = {
        "id": "test-market-order-id",
        "product_id": "ETH-USD",
        "side": "buy",
        "type": "market",
        "size": "1.0",
        "status": "pending",
        "created_at": datetime.datetime.now().isoformat()
    }
    
    order = coinbase_connector.place_order(
        symbol="ETH",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=1.0,
        leverage=1.0  # Not used for spot
    )
    
    assert order["order_id"] == "test-market-order-id"
    assert order["symbol"] == "ETH"
    assert order["side"] == "BUY"
    assert order["type"] == "MARKET"
    assert order["amount"] == 1.0
    assert order["status"] == "OPEN"
    
    # Verify the method was called with correct parameters
    mock_rest_client.create_order.assert_called_once_with(
        product_id="ETH-USD",
        side="buy",
        type="market",
        size="1.0"
    )


def test_place_limit_order(coinbase_connector, mock_rest_client):
    """Test placing a limit order."""
    order = coinbase_connector.place_order(
        symbol="ETH",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        amount=1.0,
        leverage=1.0,  # Not used for spot
        price=2600.0
    )
    
    assert order["order_id"] == "test-order-id"
    assert order["symbol"] == "ETH"
    assert order["side"] == "SELL"
    assert order["type"] == "LIMIT"
    assert order["amount"] == 1.0
    assert order["price"] == 2600.0
    assert order["status"] == "OPEN"
    
    # Verify the method was called with correct parameters
    mock_rest_client.create_order.assert_called_once_with(
        product_id="ETH-USD",
        side="sell",
        type="limit",
        size="1.0",
        price="2600.0"
    )


def test_cancel_order(coinbase_connector, mock_rest_client):
    """Test canceling an order."""
    result = coinbase_connector.cancel_order("test-order-id")
    
    assert result is True
    
    # Verify the method was called with correct parameters
    mock_rest_client.cancel_order.assert_called_once_with(order_id="test-order-id")


def test_get_order(coinbase_connector, mock_rest_client):
    """Test retrieving order details."""
    order = coinbase_connector.get_order("test-order-id")
    
    assert order["order_id"] == "test-order-id"
    assert order["symbol"] == "ETH"
    assert order["side"] == "BUY"
    assert order["type"] == "LIMIT"
    assert order["amount"] == 1.0
    assert order["filled"] == 0.5
    assert order["price"] == 2500.0
    assert order["status"] == "OPEN"
    
    # Verify the method was called with correct parameters
    mock_rest_client.get_order.assert_called_once_with(order_id="test-order-id")


def test_close_position_by_id(coinbase_connector, mock_rest_client):
    """Test closing a position by order ID."""
    result = coinbase_connector.close_position("ETH", position_id="test-order-id")
    
    assert result["success"] is True
    assert "test-order-id" in result["message"]
    
    # Verify the method was called with correct parameters
    mock_rest_client.cancel_order.assert_called_once_with(order_id="test-order-id")


def test_close_position_by_selling(coinbase_connector, mock_rest_client):
    """Test closing a position by selling the balance."""
    # Config mock to return for market sell
    mock_rest_client.create_order.return_value = {
        "id": "test-sell-order-id",
        "product_id": "ETH-USD",
        "side": "sell",
        "type": "market",
        "size": "5.0",
        "status": "pending",
        "created_at": datetime.datetime.now().isoformat()
    }
    
    result = coinbase_connector.close_position("ETH")
    
    assert result["success"] is True
    assert "order" in result
    
    # Verify the methods were called with correct parameters
    mock_rest_client.get_accounts.assert_called()
    mock_rest_client.create_order.assert_called_with(
        product_id="ETH-USD",
        side="sell",
        type="market",
        size="5.0"
    )


def test_get_historical_candles(coinbase_connector, mock_rest_client):
    """Test retrieving historical candle data."""
    candles = coinbase_connector.get_historical_candles(
        symbol="ETH",
        interval="1h",
        start_time=1617580800000,
        end_time=1617667200000,
        limit=2
    )
    
    assert len(candles) == 2
    assert candles[0]["timestamp"] == 1617580800 * 1000
    assert candles[0]["open"] == 2050.0
    assert candles[0]["high"] == 2100.0
    assert candles[0]["low"] == 2000.0
    assert candles[0]["close"] == 2075.0
    assert candles[0]["volume"] == 100.0
    
    # Verify the method was called with correct parameters
    mock_rest_client.get_product_candles.assert_called_once()


def test_convert_interval_to_granularity(coinbase_connector):
    """Test interval conversion to Coinbase granularity."""
    assert coinbase_connector._convert_interval_to_granularity("1m") == 60
    assert coinbase_connector._convert_interval_to_granularity("5m") == 300
    assert coinbase_connector._convert_interval_to_granularity("15m") == 900
    assert coinbase_connector._convert_interval_to_granularity("1h") == 3600
    assert coinbase_connector._convert_interval_to_granularity("6h") == 21600
    assert coinbase_connector._convert_interval_to_granularity("1d") == 86400
    # Test default fallback
    assert coinbase_connector._convert_interval_to_granularity("unknown") == 60


def test_get_funding_rate(coinbase_connector):
    """Test retrieving funding rate (not applicable for spot)."""
    funding = coinbase_connector.get_funding_rate("ETH")
    
    assert funding["symbol"] == "ETH"
    assert funding["funding_rate"] == 0.0
    assert funding["next_funding_time"] is None
    assert "not applicable" in funding["message"].lower()


def test_get_leverage_tiers(coinbase_connector):
    """Test retrieving leverage tiers (not applicable for spot)."""
    tiers = coinbase_connector.get_leverage_tiers("ETH")
    
    assert len(tiers) == 1
    assert tiers[0]["symbol"] == "ETH"
    assert tiers[0]["tier"] == 1
    assert tiers[0]["max_leverage"] == 1.0
    assert "not applicable" in tiers[0]["message"].lower()


def test_set_leverage(coinbase_connector):
    """Test setting leverage (not applicable for spot)."""
    result = coinbase_connector.set_leverage("ETH", 5.0)
    
    assert result["symbol"] == "ETH"
    assert result["leverage"] == 1.0
    assert result["success"] is False
    assert "not applicable" in result["message"].lower()
