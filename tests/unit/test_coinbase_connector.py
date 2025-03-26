import pytest
import json
from unittest.mock import MagicMock, patch, PropertyMock, ANY
from datetime import datetime, timezone
from decimal import Decimal
import os
from io import StringIO
import logging

from app.connectors.coinbase_connector import CoinbaseConnector
from app.connectors.base_connector import OrderSide, OrderType, OrderStatus, MarketType


@pytest.fixture
def mock_rest_client():
    """Create a mock REST client for testing."""
    mock = MagicMock()

    # Mock account balances
    accounts_response = MagicMock()
    accounts = [
        MagicMock(currency="USD", available_balance=MagicMock(value="10000.0")),
        MagicMock(currency="ETH", available_balance=MagicMock(value="5.0")),
        MagicMock(currency="BTC", available_balance=MagicMock(value="0.5")),
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


@pytest.fixture(scope="function")
def coinbase_connector():
    """Create a Coinbase connector instance for testing."""
    connector = CoinbaseConnector(
        api_key=os.environ.get("COINBASE_API_KEY", "test_key"),
        api_secret=os.environ.get("COINBASE_API_SECRET", "test_secret"),
        passphrase="test_passphrase",
        testnet=True,
        market_types=MarketType.SPOT,
    )

    # Set up loggers for testing
    connector.balance_logger = logging.getLogger("test.coinbase.balance")
    connector.markets_logger = logging.getLogger("test.coinbase.markets")
    connector.orders_logger = logging.getLogger("test.coinbase.orders")

    return connector


def test_initialization(coinbase_connector):
    """Test that the Coinbase connector initializes correctly."""
    assert coinbase_connector.api_key == os.environ.get("COINBASE_API_KEY", "test_key")
    assert coinbase_connector.api_secret == os.environ.get(
        "COINBASE_API_SECRET", "test_secret"
    )
    assert coinbase_connector.passphrase == "test_passphrase"
    assert coinbase_connector.testnet is True
    assert isinstance(coinbase_connector.market_types, list)
    assert coinbase_connector.market_types[0] == MarketType.SPOT


def test_connect(coinbase_connector, mock_rest_client):
    """Test connecting to the Coinbase API."""
    # Reset the client to simulate reconnection
    coinbase_connector.client = None

    # Mock the accounts response
    accounts_response = MagicMock()
    accounts = [MagicMock(currency="USD", available_balance=MagicMock(value="10000.0"))]
    accounts_response.accounts = accounts
    mock_rest_client.get_accounts.return_value = accounts_response

    # Mock the RESTClient initialization
    with patch("coinbase.rest.RESTClient", return_value=mock_rest_client):
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


def test_get_markets(coinbase_connector):
    """Test retrieving available markets from the actual API."""
    # Connect to the real API
    if not coinbase_connector.is_connected:
        coinbase_connector.connect()

    # Get markets from the real API
    markets = coinbase_connector.get_markets()

    # Verify we got a substantial number of markets (exact count may vary over time)
    assert len(markets) > 100

    # Verify the basic structure of the markets data
    assert all(
        key in markets[0]
        for key in [
            "symbol",
            "base_asset",
            "quote_asset",
            "price_precision",
            "min_size",
            "tick_size",
            "market_type",
            "active",
        ]
    )

    # Check for some expected markets
    market_symbols = [m["symbol"] for m in markets]
    assert "BTC-USD" in market_symbols
    assert "ETH-USD" in market_symbols


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
        MagicMock(price="1999.0", size="1.0"),
    ]
    order_book.asks = [
        MagicMock(price="2001.0", size="1.0"),
        MagicMock(price="2002.0", size="1.0"),
    ]
    mock_rest_client.get_product_book.return_value = order_book

    orderbook = coinbase_connector.get_orderbook("ETH")
    assert "bids" in orderbook
    assert "asks" in orderbook
    assert len(orderbook["bids"]) == 2
    assert len(orderbook["asks"]) == 2


def test_get_account_balance(coinbase_connector):
    """Test retrieving account balances from Coinbase using the actual API."""
    # Connect to the real API
    if not coinbase_connector.is_connected:
        coinbase_connector.connect()

    # Get account balances from the real API
    balances = coinbase_connector.get_account_balance()

    # Verify we got account data
    assert isinstance(balances, dict)

    # Check the structure - exact balances will vary but we should have the right currencies
    # Check for common currencies (we assume the test account has at least these currencies available)
    common_currencies = ["USD", "BTC", "ETH"]
    found_currencies = [curr for curr in common_currencies if curr in balances]

    # We should find at least one of these currencies (may not have all in test account)
    assert (
        len(found_currencies) > 0
    ), f"None of the expected currencies {common_currencies} found in {list(balances.keys())}"

    # Check that balances are numeric values
    for currency, balance in balances.items():
        assert isinstance(
            balance, (int, float)
        ), f"Balance for {currency} is not a number: {balance}"


def test_get_positions(coinbase_connector, mock_rest_client):
    """Test retrieving current positions."""
    # Ensure we're working with a new connector instance
    coinbase_connector.client = mock_rest_client

    # Mock response for account balances with ETH position
    account_eth = MagicMock(currency="ETH", available_balance=MagicMock(value="5.0"))
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
        leverage=1.0,  # Not used for spot
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
        price=2600.0,
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
    if hasattr(order_config, "limit_limit_gtc"):
        order_config.limit_limit_gtc.limit_price = "2500.0"

    order_response.order_configuration = order_config

    response = MagicMock()
    response.order = order_response

    mock_rest_client.get_order.return_value = response

    order = coinbase_connector.get_order("test-order-id")

    assert order["order_id"] == "test-order-id"
    assert order["status"] == OrderStatus.OPEN.value
    assert order["symbol"] == "ETH"  # Symbol should be converted from ETH-USD


def test_close_position_by_selling(coinbase_connector):
    """Test closing a position by selling - using actual API for structure validation only."""
    # Connect to the real API
    if not coinbase_connector.is_connected:
        coinbase_connector.connect()

    # NOTE: This test doesn't actually execute a sell order on the exchange
    # Instead, we'll verify the method returns the expected structure
    result = coinbase_connector.close_position("ETH", position_id="test-position-id")

    # Check the result structure
    assert isinstance(result, dict)
    assert "success" in result

    # If we have a position_id specified, the implementation should just try to cancel that order
    # and not place a sell order
    if "message" in result:
        assert "test-position-id" in result.get("message", "")

    # If there's an order field, verify its structure (won't always be present)
    if "order" in result:
        assert isinstance(result["order"], dict)
        # If order exists and contains these keys, check the values
        if "symbol" in result["order"]:
            assert result["order"]["symbol"] == "ETH"


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
        limit=2,
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
    # Explicitly test the method directly
    granularity = coinbase_connector.convert_interval_to_granularity("1h")
    assert granularity == 3600


def test_get_funding_rate(coinbase_connector):
    """Test getting funding rate (not applicable for spot)."""
    result = coinbase_connector.get_funding_rate("ETH")
    # Updated to match the actual implementation that returns a dict
    assert isinstance(result, dict)
    assert result["rate"] == 0.0
    assert result["symbol"] == "ETH"
    assert "message" in result
    assert "next_funding_time" in result


def test_get_leverage_tiers(coinbase_connector):
    """Test getting leverage tiers (not applicable for spot)."""
    tiers = coinbase_connector.get_leverage_tiers("ETH")
    # Updated to match the actual implementation
    assert len(tiers) == 0


def test_set_leverage(coinbase_connector):
    """Test setting leverage (not applicable for spot)."""
    result = coinbase_connector.set_leverage("ETH", 5.0)
    assert result["success"] is False
    # Updated to match the exact message returned
    assert "not supported for coinbase spot trading" in result["message"].lower()
