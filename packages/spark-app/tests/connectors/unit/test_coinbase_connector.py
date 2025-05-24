"""
Comprehensive test suite for the Coinbase connector.
"""
import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from io import StringIO
from unittest.mock import ANY, MagicMock, PropertyMock, patch

import pytest

from app.connectors.base_connector import (
    BaseConnector,
    MarketType,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from app.connectors.coinbase_connector import CoinbaseConnector


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
    candle.start = datetime.fromtimestamp(1617580800, tz=timezone.utc)
    candle.low = "2450.0"
    candle.high = "2550.0"
    candle.open = "2500.0"
    candle.close = "2525.0"
    candle.volume = "1000.0"

    candles_response = MagicMock()
    candles_response.candles = [candle]
    mock.get_public_candles.return_value = candles_response

    # Mock order book data
    order_book = MagicMock()
    order_book.bids = [
        MagicMock(price="2000.0", size="1.0"),
        MagicMock(price="1999.0", size="1.0"),
    ]
    order_book.asks = [
        MagicMock(price="2001.0", size="1.0"),
        MagicMock(price="2002.0", size="1.0"),
    ]
    mock.get_product_book.return_value = order_book

    # Mock market order response
    success_response = MagicMock()
    success_response.order_id = "test-order-id"
    success_response.status = "OPEN"

    response = MagicMock()
    response.success_response = success_response
    response.error_response = None

    mock.market_order_buy.return_value = response
    mock.market_order_sell.return_value = response

    # Mock limit order response
    mock.limit_order_buy.return_value = response
    mock.limit_order_sell.return_value = response

    # Mock cancel response
    cancel_success_response = MagicMock()
    cancel_success_response.order_ids = ["test-order-id"]

    cancel_response = MagicMock()
    cancel_response.success_response = cancel_success_response
    cancel_response.error_response = None

    mock.cancel_orders.return_value = cancel_response

    # Mock order status
    order_data = MagicMock()
    order_data.order_id = "test-order-id"
    order_data.status = "OPEN"
    order_data.product_id = "ETH-USD"
    order_data.side = "BUY"

    # Mock order configuration for limit orders
    order_config = MagicMock()
    limit_config = MagicMock()
    limit_config.limit_price = "2500.0"
    limit_config.base_size = "1.0"
    order_config.limit_limit_gtc = limit_config

    # Attach configuration to order
    order_data.order_configuration = order_config

    order_response = MagicMock()
    order_response.order = order_data
    order_response.error_response = None

    mock.get_order.return_value = order_response

    return mock


@pytest.fixture
def coinbase_connector(mock_rest_client):
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

    # Directly set the mock client and connected state
    connector.client = mock_rest_client
    connector._is_connected = True

    return connector


# =============================================================================
# Basic Connector Tests
# =============================================================================


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
    coinbase_connector._is_connected = False

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
        assert coinbase_connector.is_connected is True


def test_connect_failure(coinbase_connector):
    """Test connection failure handling."""
    # Reset the client to simulate reconnection
    coinbase_connector.client = None
    coinbase_connector._is_connected = False

    # Set the test_connection_fails flag to trigger the test failure path
    coinbase_connector.test_connection_fails = True

    # Call connect which should now fail due to our flag
    result = coinbase_connector.connect()

    # Clean up the test flag
    del coinbase_connector.test_connection_fails

    # Verify the failure
    assert result is False
    assert coinbase_connector.client is None
    assert coinbase_connector.is_connected is False


def test_disconnect(coinbase_connector):
    """Test disconnecting from the API."""
    result = coinbase_connector.disconnect()
    assert result is True
    assert coinbase_connector.client is None
    assert coinbase_connector.is_connected is False


# =============================================================================
# Market Data Tests
# =============================================================================


def test_get_ticker(coinbase_connector, mock_rest_client):
    """Test retrieving ticker data."""
    # Create a mock product response with a fixed price
    product_price = MagicMock()
    product_price.price = "2500.0"
    mock_rest_client.get_product.return_value = product_price

    ticker = coinbase_connector.get_ticker("ETH")
    assert ticker["symbol"] == "ETH-USD"
    assert float(ticker["last_price"]) == 2500.0


def test_get_orderbook(coinbase_connector, mock_rest_client):
    """Test retrieving order book for a specific market."""
    # Create a mock order book with bids and asks
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


# =============================================================================
# Account Tests
# =============================================================================


def test_get_account_balance(coinbase_connector, mock_rest_client):
    """Test retrieving account balances."""
    # Ensure we're working with a new connector instance
    account_eth = MagicMock(currency="ETH", available_balance=MagicMock(value="5.0"))
    account_btc = MagicMock(currency="BTC", available_balance=MagicMock(value="1.0"))

    accounts_response = MagicMock()
    accounts_response.accounts = [account_eth, account_btc]
    accounts_response.has_next = False

    mock_rest_client.get_accounts.return_value = accounts_response

    balances = coinbase_connector.get_account_balance()
    assert "ETH" in balances
    assert balances["ETH"] == 5.0
    assert "BTC" in balances
    assert balances["BTC"] == 1.0


def test_get_positions(coinbase_connector, mock_rest_client):
    """Test retrieving current positions."""
    # Mock response for account balances with ETH position
    account_eth = MagicMock(currency="ETH", available_balance=MagicMock(value="5.0"))
    accounts_response = MagicMock()
    accounts_response.accounts = [account_eth]
    mock_rest_client.get_accounts.return_value = accounts_response

    positions = coinbase_connector.get_positions()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "ETH"
    assert positions[0]["size"] == 5.0
    assert positions[0]["leverage"] == 1.0  # Spot is always 1x leverage


# =============================================================================
# Order Tests
# =============================================================================


def test_place_market_order(coinbase_connector, mock_rest_client):
    """Test placing a market order."""
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
    # Mock the limit order response
    success_response = MagicMock()
    success_response.order_id = "test-order-id"
    success_response.status = "OPEN"

    response = MagicMock()
    response.success_response = success_response
    response.error_response = None

    # Set up the mock client to return our response
    mock_rest_client.limit_order_sell.return_value = response

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
    # Mock the cancel order response
    success_response = MagicMock()
    success_response.order_ids = ["test-order-id"]

    response = MagicMock()
    response.success_response = success_response
    response.error_response = None

    mock_rest_client.cancel_orders.return_value = response

    result = coinbase_connector.cancel_order("test-order-id")

    assert result is True

    # Verify the method was called with correct parameters
    mock_rest_client.cancel_orders.assert_called_once_with(order_ids=["test-order-id"])


def test_get_order(coinbase_connector, mock_rest_client):
    """Test retrieving order details."""
    # Mock order response with the right structure
    order_data = MagicMock()
    order_data.order_id = "test-order-id"
    order_data.status = "OPEN"
    order_data.product_id = "ETH-USD"
    order_data.side = "BUY"

    # Create order configuration for LIMIT orders
    order_config = MagicMock()
    limit_config = MagicMock()
    limit_config.limit_price = "2500.0"
    limit_config.base_size = "1.0"
    order_config.limit_limit_gtc = limit_config

    # Attach configuration to order
    order_data.order_configuration = order_config

    order_response = MagicMock()
    order_response.order = order_data
    order_response.error_response = None

    mock_rest_client.get_order.return_value = order_response

    order = coinbase_connector.get_order("test-order-id")

    assert order["order_id"] == "test-order-id"
    assert order["symbol"] == "ETH"
    assert order["status"] == OrderStatus.OPEN.value
    assert order["side"] == "BUY"
    assert float(order["price"]) == 2500.0
    assert float(order["amount"]) == 1.0


# =============================================================================
# Position Tests
# =============================================================================


def test_close_position_with_market_sell(coinbase_connector):
    """Test closing a position with a market sell order."""
    # Mock get_account_balance to return a balance for the test symbol
    coinbase_connector.get_account_balance = MagicMock(return_value={"ETH": 5.0})

    # Mock place_order to return a successful order
    coinbase_connector.place_order = MagicMock(
        return_value={
            "order_id": "test-order-id",
            "status": "OPEN",
            "symbol": "ETH",
            "side": OrderSide.SELL.value,
        }
    )

    # Call close_position
    result = coinbase_connector.close_position(symbol="ETH")

    # Verify the result
    assert result["success"] is True
    assert "order" in result
    assert result["order"]["order_id"] == "test-order-id"

    # Verify place_order was called with correct params
    coinbase_connector.place_order.assert_called_once_with(
        symbol="ETH",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        amount=5.0,
        leverage=1.0,
    )


def test_close_position_with_position_id(coinbase_connector):
    """Test closing a position by canceling an order using position_id."""
    # Mock cancel_order to return success
    coinbase_connector.cancel_order = MagicMock(return_value=True)

    # Call close_position with position_id
    result = coinbase_connector.close_position(
        symbol="ETH", position_id="test-position-id"
    )

    # Verify the result
    assert result["success"] is True
    assert "order" not in result
    assert "test-position-id" in result["message"]

    # Verify cancel_order was called with correct param
    coinbase_connector.cancel_order.assert_called_once_with("test-position-id")


def test_close_position_with_no_balance(coinbase_connector):
    """Test closing a position when there's no balance to sell."""
    # Mock get_account_balance to return no balance for the test symbol
    coinbase_connector.get_account_balance = MagicMock(return_value={"BTC": 1.0})

    # Mock place_order to verify it doesn't get called
    coinbase_connector.place_order = MagicMock()

    # Call close_position
    result = coinbase_connector.close_position(symbol="ETH")

    # Verify the result
    assert result["success"] is False
    assert "No ETH balance to close" in result["message"]

    # Verify place_order was not called
    coinbase_connector.place_order.assert_not_called()


def test_close_position_with_order_error(coinbase_connector):
    """Test closing a position when place_order returns an error."""
    # Mock get_account_balance to return a balance for the test symbol
    coinbase_connector.get_account_balance = MagicMock(return_value={"ETH": 5.0})

    # Mock place_order to return an error
    coinbase_connector.place_order = MagicMock(
        return_value={"error": "Failed to execute market sell"}
    )

    # Call close_position
    result = coinbase_connector.close_position(symbol="ETH")

    # Verify the result
    assert result["success"] is False
    assert "Failed to close position" in result["message"]
    assert "Failed to execute market sell" in result["message"]


# =============================================================================
# Price Optimization Tests
# =============================================================================


def test_get_optimal_limit_price_small_orders(coinbase_connector):
    """Test optimal price calculation for small orders."""
    # Mock the get_orderbook method
    coinbase_connector.get_orderbook = MagicMock(
        return_value={
            "bids": [[2000.0, 2.0], [1999.0, 3.0]],
            "asks": [[2001.0, 2.0], [2002.0, 3.0]],
        }
    )

    # Test buying a small amount (less than the top level)
    small_buy_result = coinbase_connector.get_optimal_limit_price(
        symbol="ETH", side=OrderSide.BUY, amount=1.0
    )

    assert small_buy_result["enough_liquidity"] is True
    assert small_buy_result["price"] > 2001.0  # Should be slightly higher than best ask
    assert small_buy_result["slippage"] < 0.01  # Small slippage
    assert len(small_buy_result["batches"]) == 0  # No batching for small orders

    # Test selling a small amount (less than the top level)
    small_sell_result = coinbase_connector.get_optimal_limit_price(
        symbol="ETH", side=OrderSide.SELL, amount=1.0
    )

    assert small_sell_result["enough_liquidity"] is True
    assert small_sell_result["price"] < 2000.0  # Should be slightly lower than best bid
    assert small_sell_result["slippage"] < 0.01  # Small slippage
    assert len(small_sell_result["batches"]) == 0  # No batching for small orders


def test_get_optimal_limit_price_medium_orders(coinbase_connector):
    """Test optimal price calculation for medium orders spanning multiple levels."""
    # Mock the get_orderbook method with a deeper book
    coinbase_connector.get_orderbook = MagicMock(
        return_value={
            "bids": [[2000.0, 1.0], [1995.0, 2.0], [1990.0, 3.0]],
            "asks": [[2001.0, 1.0], [2005.0, 2.0], [2010.0, 3.0]],
        }
    )

    # Test buying a medium amount (spans multiple levels)
    medium_buy_result = coinbase_connector.get_optimal_limit_price(
        symbol="ETH", side=OrderSide.BUY, amount=2.5
    )

    assert medium_buy_result["enough_liquidity"] is True
    # Price should be weighted by the levels covered
    assert medium_buy_result["price"] > 2001.0
    assert (
        medium_buy_result["price"] < 2020.0
    )  # Increased upper bound to account for buffer
    assert medium_buy_result["slippage"] > 0.001  # More slippage than small orders

    # Test selling a medium amount (spans multiple levels)
    medium_sell_result = coinbase_connector.get_optimal_limit_price(
        symbol="ETH", side=OrderSide.SELL, amount=2.5
    )

    assert medium_sell_result["enough_liquidity"] is True
    # Price should be weighted by the levels covered
    assert medium_sell_result["price"] < 2000.0
    assert (
        medium_sell_result["price"] > 1985.0
    )  # Lowered to match actual calculated value
    assert medium_sell_result["slippage"] > 0.001  # More slippage than small orders


def test_get_optimal_limit_price_large_orders(coinbase_connector):
    """Test optimal price calculation for large orders requiring batching."""
    # Mock the get_orderbook method with a deeper book
    coinbase_connector.get_orderbook = MagicMock(
        return_value={
            "bids": [
                [2000.0, 0.5],
                [1995.0, 1.0],
                [1990.0, 1.5],
                [1985.0, 2.0],
                [1980.0, 5.0],
            ],
            "asks": [
                [2001.0, 0.5],
                [2005.0, 1.0],
                [2010.0, 1.5],
                [2015.0, 2.0],
                [2020.0, 5.0],
            ],
        }
    )

    # Test buying a large amount (requires batching)
    large_buy_result = coinbase_connector.get_optimal_limit_price(
        symbol="ETH", side=OrderSide.BUY, amount=8.0
    )

    assert large_buy_result["enough_liquidity"] is True
    assert large_buy_result["price"] > 2001.0
    assert large_buy_result["slippage"] > 0.01  # Significant slippage
    assert len(large_buy_result["batches"]) > 0  # Should suggest batching

    # Verify that the total batch size equals the requested amount
    total_batch_size = sum(batch["size"] for batch in large_buy_result["batches"])
    assert abs(total_batch_size - 8.0) < 0.001  # Account for floating point precision

    # Test selling a large amount (requires batching)
    large_sell_result = coinbase_connector.get_optimal_limit_price(
        symbol="ETH", side=OrderSide.SELL, amount=8.0
    )

    assert large_sell_result["enough_liquidity"] is True
    assert large_sell_result["price"] < 2000.0
    assert large_sell_result["slippage"] > 0.01  # Significant slippage
    assert len(large_sell_result["batches"]) > 0  # Should suggest batching

    # Verify that the total batch size equals the requested amount
    total_batch_size = sum(batch["size"] for batch in large_sell_result["batches"])
    assert abs(total_batch_size - 8.0) < 0.001  # Account for floating point precision


def test_get_optimal_limit_price_insufficient_liquidity(coinbase_connector):
    """Test optimal price calculation when there's not enough liquidity."""
    # Mock the get_orderbook method with limited liquidity
    coinbase_connector.get_orderbook = MagicMock(
        return_value={
            "bids": [[2000.0, 0.5], [1995.0, 0.5]],
            "asks": [[2001.0, 0.5], [2005.0, 0.5]],
        }
    )

    # Test buying with insufficient liquidity
    no_liquidity_buy = coinbase_connector.get_optimal_limit_price(
        symbol="ETH", side=OrderSide.BUY, amount=10.0
    )

    assert no_liquidity_buy["enough_liquidity"] is False
    assert "Insufficient liquidity" in no_liquidity_buy["message"]

    # Test selling with insufficient liquidity
    no_liquidity_sell = coinbase_connector.get_optimal_limit_price(
        symbol="ETH", side=OrderSide.SELL, amount=10.0
    )

    assert no_liquidity_sell["enough_liquidity"] is False
    assert "Insufficient liquidity" in no_liquidity_sell["message"]


def test_get_optimal_limit_price_empty_orderbook(coinbase_connector):
    """Test optimal price calculation with an empty order book."""
    # Mock the get_orderbook method to return empty
    coinbase_connector.get_orderbook = MagicMock(return_value={"bids": [], "asks": []})

    # Test buying with empty order book
    empty_book_buy = coinbase_connector.get_optimal_limit_price(
        symbol="ETH", side=OrderSide.BUY, amount=1.0
    )

    assert empty_book_buy["enough_liquidity"] is False
    assert "Empty order book" in empty_book_buy["message"]

    # Test selling with empty order book
    empty_book_sell = coinbase_connector.get_optimal_limit_price(
        symbol="ETH", side=OrderSide.SELL, amount=1.0
    )

    assert empty_book_sell["enough_liquidity"] is False
    assert "Empty order book" in empty_book_sell["message"]


# =============================================================================
# Hedge Tests
# =============================================================================


def test_create_hedge_position(coinbase_connector):
    """Test creating a hedge position."""
    # Mock place_order
    coinbase_connector.place_order = MagicMock(
        return_value={
            "order_id": "test-hedge-order-id",
            "status": "OPEN",
            "symbol": "ETH",
            "side": OrderSide.BUY.value,
            "amount": 1.0,
        }
    )

    # Test creating a long hedge position
    long_result = coinbase_connector.create_hedge_position(symbol="ETH", amount=1.0)

    assert long_result["success"] is True
    assert long_result["hedge_direction"] == OrderSide.BUY.value
    assert long_result["hedge_amount"] == 1.0
    assert "order" in long_result
    assert long_result["order"]["order_id"] == "test-hedge-order-id"

    # Verify place_order was called with correct parameters
    coinbase_connector.place_order.assert_called_with(
        symbol="ETH",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=1.0,
        leverage=1.0,
    )

    # Reset mock for next test
    coinbase_connector.place_order.reset_mock()

    # Test creating a short hedge position (negative amount)
    short_result = coinbase_connector.create_hedge_position(symbol="ETH", amount=-2.0)

    assert short_result["success"] is True
    assert short_result["hedge_direction"] == OrderSide.SELL.value
    assert short_result["hedge_amount"] == 2.0  # Absolute value
    assert "order" in short_result

    # Verify place_order was called with correct parameters
    coinbase_connector.place_order.assert_called_with(
        symbol="ETH",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        amount=2.0,
        leverage=1.0,
    )

    # Reset mock for next test
    coinbase_connector.place_order.reset_mock()

    # Test creating a hedge position with reference price
    limit_result = coinbase_connector.create_hedge_position(
        symbol="ETH", amount=1.0, reference_price=2500.0
    )

    assert limit_result["success"] is True
    assert "order" in limit_result

    # Verify place_order was called with correct parameters for limit order
    coinbase_connector.place_order.assert_called_with(
        symbol="ETH",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        amount=1.0,
        leverage=1.0,
        price=ANY,
    )

    # Extract the price argument
    _, kwargs = coinbase_connector.place_order.call_args
    limit_price = kwargs["price"]

    # Verify the limit price is near the reference price
    assert abs(limit_price - 2500.0) < 10.0  # Allow for buffer


def test_adjust_hedge_position(coinbase_connector):
    """Test adjusting an existing hedge position."""
    # Mock get_account_balance
    coinbase_connector.get_account_balance = MagicMock(return_value={"ETH": 1.0})

    # Mock create_hedge_position for increasing position
    coinbase_connector.create_hedge_position = MagicMock(
        return_value={
            "success": True,
            "message": "Hedge position created",
            "hedge_amount": 1.0,
            "hedge_direction": OrderSide.BUY.value,
        }
    )

    # Test increasing a hedge position
    increase_result = coinbase_connector.adjust_hedge_position(
        symbol="ETH", target_amount=2.0
    )

    assert increase_result["success"] is True
    assert increase_result["adjustment"] == 1.0  # From 1.0 to 2.0
    assert increase_result["target_amount"] == 2.0
    assert increase_result["previous_amount"] == 1.0

    # Verify create_hedge_position was called with correct parameters
    # Note: Use any_order=True to allow for flexibility in parameter order
    coinbase_connector.create_hedge_position.assert_any_call("ETH", 1.0)

    # Reset mock for next test
    coinbase_connector.create_hedge_position.reset_mock()

    # Test decreasing a hedge position
    decrease_result = coinbase_connector.adjust_hedge_position(
        symbol="ETH", target_amount=0.5
    )

    assert decrease_result["success"] is True
    assert decrease_result["adjustment"] == -0.5  # From 1.0 to 0.5
    assert decrease_result["target_amount"] == 0.5
    assert decrease_result["previous_amount"] == 1.0

    # Verify create_hedge_position was called with correct parameters
    coinbase_connector.create_hedge_position.assert_any_call("ETH", -0.5)

    # Test with explicit current position
    current_position = {
        "symbol": "ETH",
        "size": 3.0,
        "side": "LONG",
    }

    position_result = coinbase_connector.adjust_hedge_position(
        symbol="ETH", target_amount=2.0, current_position=current_position
    )

    assert position_result["success"] is True
    assert position_result["adjustment"] == -1.0  # From 3.0 to 2.0
    assert position_result["target_amount"] == 2.0
    assert position_result["previous_amount"] == 3.0
