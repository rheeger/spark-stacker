import pytest
import json
from unittest.mock import MagicMock, patch, PropertyMock, ANY
from datetime import datetime, timezone
from decimal import Decimal
import os
from io import StringIO

from app.connectors.coinbase_connector import CoinbaseConnector
from app.connectors.base_connector import OrderSide, OrderType, OrderStatus


@pytest.fixture
def mock_rest_client():
    """Create a mock REST client for testing."""
    mock = MagicMock()

    # Mock account balances
    accounts_response = MagicMock()
    accounts = [
        MagicMock(
            currency="USD",
            available_balance=MagicMock(value="10000.0")
        ),
        MagicMock(
            currency="ETH",
            available_balance=MagicMock(value="5.0")
        ),
        MagicMock(
            currency="BTC",
            available_balance=MagicMock(value="0.5")
        )
    ]
    accounts_response.accounts = accounts
    mock.get_accounts.return_value = accounts_response

    # Mock product data
    eth_product = MagicMock()
    eth_product.product_id = "ETH-USD"
    eth_product.base_min_size = "0.01"
    eth_product.base_max_size = "1000.0"
    eth_product.status = "online"

    btc_product = MagicMock()
    btc_product.product_id = "BTC-USD"
    btc_product.base_min_size = "0.001"
    btc_product.base_max_size = "100.0"
    btc_product.status = "online"

    products_response = MagicMock()
    products_response.products = [eth_product, btc_product]
    mock.get_public_products.return_value = products_response

    # Mock product price data
    product_price = MagicMock()
    product_price.price = "2500.0"
    mock.get_product.return_value = product_price

    # Mock candles data
    candle = MagicMock()
    candle.start = 1617580800000
    candle.low = "2450.0"
    candle.high = "2550.0"
    candle.open = "2500.0"
    candle.close = "2525.0"
    candle.volume = "1000.0"
    mock.get_public_candles.return_value = [candle]

    # Mock order book data
    order_book = MagicMock()
    order_book.bids = [[2000.0, 1.0], [1999.0, 1.0]]
    order_book.asks = [[2001.0, 1.0], [2002.0, 1.0]]
    mock.get_product_book.return_value = order_book

    # Mock market order response
    market_order = MagicMock()
    market_order.order_id = "test-order-id"
    market_order.status = "OPEN"
    market_order.product_id = "ETH-USD"
    mock.market_order_buy.return_value = market_order
    mock.market_order_sell.return_value = market_order

    # Mock limit order response
    limit_order = MagicMock()
    limit_order.order_id = "test-order-id"
    limit_order.status = "OPEN"
    limit_order.product_id = "ETH-USD"
    mock.create_order.return_value = limit_order

    # Mock cancel response
    mock.cancel_orders.return_value = True

    # Mock order status
    order = MagicMock()
    order.order_id = "test-order-id"
    order.status = "OPEN"
    order.product_id = "ETH-USD"
    mock.get_order.return_value = order

    # Mock positions
    position = MagicMock()
    position.product_id = "ETH-USD"
    position.position_size = "1.0"
    position.entry_price = "2500.0"
    mock.get_positions.return_value = [position]

    return mock


@pytest.fixture
def coinbase_connector():
    """Create a Coinbase connector instance for testing."""
    return CoinbaseConnector(
        api_key=os.environ.get("COINBASE_API_KEY", "test_key"),
        api_secret=os.environ.get("COINBASE_API_SECRET", "test_secret"),
        api_passphrase="test_passphrase",
        use_sandbox=True
    )


def test_initialization(coinbase_connector):
    """Test connector initialization."""
    assert coinbase_connector.api_key == os.environ.get("COINBASE_API_KEY", "test_key")
    assert coinbase_connector.api_secret == os.environ.get("COINBASE_API_SECRET", "test_secret")
    assert coinbase_connector.api_passphrase == "test_passphrase"
    assert coinbase_connector.use_sandbox is True


def test_connect(coinbase_connector, mock_rest_client):
    """Test connecting to the Coinbase API."""
    # Reset the client to simulate reconnection
    coinbase_connector.client = None

    # Mock the accounts response
    accounts_response = MagicMock()
    accounts = [
        MagicMock(
            currency="USD",
            available_balance=MagicMock(value="10000.0")
        )
    ]
    accounts_response.accounts = accounts
    mock_rest_client.get_accounts.return_value = accounts_response

    # Mock the RESTClient initialization
    with patch('coinbase.rest.RESTClient', return_value=mock_rest_client):
        result = coinbase_connector.connect()

        # Verify the result
        assert result is True
        assert coinbase_connector.client is not None


def test_connect_failure(coinbase_connector):
    """Test connection failure handling."""
    # Reset the client to simulate reconnection
    coinbase_connector.client = None
    
    # Set the test_connection_fails flag to trigger the test failure path
    coinbase_connector.test_connection_fails = True
    
    # Call connect which should now fail due to our flag
    result = coinbase_connector.connect()
    
    # Clean up the test flag
    del coinbase_connector.test_connection_fails
    
    # Verify the failure
    assert result is False
    assert coinbase_connector.client is None


def test_disconnect(coinbase_connector):
    """Test disconnecting from the API."""
    result = coinbase_connector.disconnect()
    assert result is True


def test_get_markets(coinbase_connector, mock_rest_client):
    """Test retrieving available markets."""
    # Ensure we're working with a new connector instance
    coinbase_connector.client = mock_rest_client
    
    # Force mock return value to be exact match for test expectation
    eth_product = MagicMock()
    eth_product.product_id = "ETH-USD"
    eth_product.base_min_size = "0.01"
    eth_product.base_max_size = "1000.0"
    eth_product.quote_increment = "0.01"
    eth_product.status = "online"

    btc_product = MagicMock()
    btc_product.product_id = "BTC-USD"
    btc_product.base_min_size = "0.001"
    btc_product.base_max_size = "100.0"
    btc_product.quote_increment = "0.01"
    btc_product.status = "online"
    
    products_response = MagicMock()
    products_response.products = [eth_product, btc_product]
    mock_rest_client.get_public_products.return_value = products_response
    
    markets = coinbase_connector.get_markets()
    assert len(markets) == 2
    assert markets[0]["symbol"] == "ETH-USD"
    assert markets[1]["symbol"] == "BTC-USD"


def test_get_ticker(coinbase_connector, mock_rest_client):
    """Test retrieving ticker data."""
    # Ensure we're working with a new connector instance
    coinbase_connector.client = mock_rest_client
    
    # Create a mock product response with a fixed price
    product_price = MagicMock()
    product_price.price = "2500.0"
    mock_rest_client.get_product.return_value = product_price
    
    ticker = coinbase_connector.get_ticker("ETH")
    assert ticker["symbol"] == "ETH-USD"
    assert float(ticker["last_price"]) == 2500.0


def test_get_orderbook(coinbase_connector, mock_rest_client):
    """Test retrieving order book for a specific market."""
    # Ensure we're working with a new connector instance
    coinbase_connector.client = mock_rest_client
    
    # Create a mock order book with exactly 2 bids and 2 asks
    order_book = MagicMock()
    order_book.bids = [
        MagicMock(price="2000.0", size="1.0"),
        MagicMock(price="1999.0", size="1.0")
    ]
    order_book.asks = [
        MagicMock(price="2001.0", size="1.0"),
        MagicMock(price="2002.0", size="1.0")
    ]
    mock_rest_client.get_product_book.return_value = order_book
    
    orderbook = coinbase_connector.get_orderbook("ETH")
    assert "bids" in orderbook
    assert "asks" in orderbook
    assert len(orderbook["bids"]) == 2
    assert len(orderbook["asks"]) == 2


def test_get_account_balance(coinbase_connector, mock_rest_client):
    """Test retrieving account balances from Coinbase."""
    # Ensure we're working with a new connector instance
    coinbase_connector.client = mock_rest_client
    
    # Mock account balances
    accounts_response = MagicMock()
    accounts = [
        MagicMock(
            currency="USD",
            available_balance=MagicMock(value="10000.0")
        ),
        MagicMock(
            currency="ETH",
            available_balance=MagicMock(value="5.0")
        ),
        MagicMock(
            currency="BTC",
            available_balance=MagicMock(value="0.5")
        )
    ]
    accounts_response.accounts = accounts
    mock_rest_client.get_accounts.return_value = accounts_response
    
    balances = coinbase_connector.get_account_balance()
    assert "USD" in balances
    assert "ETH" in balances
    assert "BTC" in balances
    assert float(balances["USD"]) == 10000.0
    assert float(balances["ETH"]) == 5.0
    assert float(balances["BTC"]) == 0.5


def test_get_positions(coinbase_connector, mock_rest_client):
    """Test retrieving current positions."""
    # Ensure we're working with a new connector instance
    coinbase_connector.client = mock_rest_client
    
    # Mock response for account balances with ETH position
    account_eth = MagicMock(
        currency="ETH",
        available_balance=MagicMock(value="5.0")
    )
    accounts_response = MagicMock()
    accounts_response.accounts = [account_eth]
    mock_rest_client.get_accounts.return_value = accounts_response

    positions = coinbase_connector.get_positions()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "ETH"


def test_place_market_order(coinbase_connector, mock_rest_client):
    """Test placing a market order."""
    # Ensure we're working with a new connector instance
    coinbase_connector.client = mock_rest_client
    
    # Mock the market order response
    success_response = MagicMock()
    success_response.order_id = "test-order-id"
    success_response.status = "OPEN"
    
    response = MagicMock()
    response.success_response = success_response
    response.error_response = None
    
    # Set up the mock client to return our response
    mock_rest_client.market_order_buy.return_value = response
    
    order = coinbase_connector.place_order(
        symbol="ETH",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=1.0,
        leverage=1.0  # Not used for spot
    )
    
    assert order["order_id"] == "test-order-id"
    assert order["status"] == OrderStatus.OPEN.value
    assert order["symbol"] == "ETH"


def test_place_limit_order(coinbase_connector, mock_rest_client):
    """Test placing a limit order."""
    # Ensure we're working with a new connector instance
    coinbase_connector.client = mock_rest_client
    
    # Mock the limit order response
    success_response = MagicMock()
    success_response.order_id = "test-order-id"
    success_response.status = "OPEN"
    
    response = MagicMock()
    response.success_response = success_response
    response.error_response = None
    
    # Set up the mock client to return our response
    mock_rest_client.limit_order_sell = MagicMock(return_value=response)
    
    order = coinbase_connector.place_order(
        symbol="ETH",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        amount=1.0,
        leverage=1.0,  # Not used for spot
        price=2600.0
    )
    
    assert order["order_id"] == "test-order-id"
    assert order["status"] == OrderStatus.OPEN.value
    assert order["symbol"] == "ETH"
    assert float(order["price"]) == 2600.0


def test_cancel_order(coinbase_connector, mock_rest_client):
    """Test canceling an order."""
    # Ensure we're working with a new connector instance
    coinbase_connector.client = mock_rest_client
    
    # Mock the cancel order response
    success_response = MagicMock()
    success_response.order_ids = ["test-order-id"]
    
    response = MagicMock()
    response.success_response = success_response
    
    mock_rest_client.cancel_orders.return_value = response
    
    result = coinbase_connector.cancel_order("test-order-id")
    
    assert result is True
    
    # Verify the method was called with correct parameters
    mock_rest_client.cancel_orders.assert_called_once_with(order_ids=["test-order-id"])


def test_get_order(coinbase_connector, mock_rest_client):
    """Test retrieving order details."""
    # Ensure we're working with a new connector instance
    coinbase_connector.client = mock_rest_client
    
    # Mock order response with the right structure
    order_response = MagicMock()
    order_response.order_id = "test-order-id"
    order_response.status = "OPEN"
    order_response.product_id = "ETH-USD"
    # Add fields needed by the implementation
    order_response.side = "BUY"
    
    # Create order configuration for LIMIT orders
    order_config = MagicMock()
    if hasattr(order_config, 'limit_limit_gtc'):
        order_config.limit_limit_gtc.limit_price = "2500.0"
    
    order_response.order_configuration = order_config
    
    response = MagicMock()
    response.order = order_response
    
    mock_rest_client.get_order.return_value = response
    
    order = coinbase_connector.get_order("test-order-id")
    
    assert order["order_id"] == "test-order-id"
    assert order["status"] == OrderStatus.OPEN.value
    assert order["symbol"] == "ETH"  # Symbol should be converted from ETH-USD


def test_close_position_by_selling(coinbase_connector, mock_rest_client):
    """Test closing a position by selling the balance."""
    # Ensure we're working with a new connector instance
    coinbase_connector.client = mock_rest_client
    
    # Mock account response with ETH balance
    account_eth = MagicMock(
        currency="ETH",
        available_balance=MagicMock(value="5.0")
    )
    accounts_response = MagicMock()
    accounts_response.accounts = [account_eth]
    mock_rest_client.get_accounts.return_value = accounts_response
    
    # Mock successful market sell order
    success_response = MagicMock()
    success_response.order_id = "test-order-id"
    success_response.status = "OPEN"
    
    response = MagicMock()
    response.success_response = success_response
    response.error_response = None
    
    mock_rest_client.market_order_sell.return_value = response
    
    result = coinbase_connector.close_position("ETH")
    assert result["success"] is True
    assert "order" in result


def test_get_historical_candles(coinbase_connector, mock_rest_client):
    """Test retrieving historical candle data."""
    # Ensure we're working with a new connector instance
    coinbase_connector.client = mock_rest_client
    
    # Create a mock candle with the expected structure
    candle = MagicMock()
    candle.start = datetime.fromtimestamp(1617580800, tz=timezone.utc)
    candle.low = "2450.0"
    candle.high = "2550.0"
    candle.open = "2500.0"
    candle.close = "2525.0"
    candle.volume = "1000.0"
    
    # Set up the mock response
    candles_response = MagicMock()
    candles_response.candles = [candle]
    mock_rest_client.get_public_candles.return_value = candles_response
    
    candles = coinbase_connector.get_historical_candles(
        symbol="ETH",
        interval="1h",
        start_time=1617580800000,
        end_time=1617667200000,
        limit=2
    )
    
    assert len(candles) == 1
    assert "timestamp" in candles[0]
    assert "open" in candles[0]
    assert "high" in candles[0]
    assert "low" in candles[0]
    assert "close" in candles[0]
    assert "volume" in candles[0]
    
    # Verify the method was called with correct parameters
    mock_rest_client.get_public_candles.assert_called_once()


def test_convert_interval_to_granularity(coinbase_connector):
    """Test converting interval to granularity."""
    granularity = coinbase_connector.convert_interval_to_granularity("1h")
    assert granularity == 3600


def test_get_funding_rate(coinbase_connector):
    """Test getting funding rate (not applicable for spot)."""
    rate = coinbase_connector.get_funding_rate("ETH")
    assert rate == 0.0


def test_get_leverage_tiers(coinbase_connector):
    """Test getting leverage tiers (not applicable for spot)."""
    tiers = coinbase_connector.get_leverage_tiers("ETH")
    assert len(tiers) == 0


def test_set_leverage(coinbase_connector):
    """Test setting leverage (not applicable for spot)."""
    result = coinbase_connector.set_leverage("ETH", 5.0)
    assert result["success"] is False
    assert "not supported" in result["message"].lower()
