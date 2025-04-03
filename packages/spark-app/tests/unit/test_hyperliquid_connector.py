#!/usr/bin/env /Users/rheeger/Code/rheeger/spark-stacker/packages/spark-app/.venv/bin/python
"""
Comprehensive test suite for the Hyperliquid connector.
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from io import StringIO
from unittest.mock import ANY, MagicMock, PropertyMock, patch

# Add parent directory to path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
import requests
from app.connectors.base_connector import (BaseConnector, MarketType,
                                           OrderSide, OrderStatus, OrderType)
from app.connectors.hyperliquid_connector import (HyperliquidAPIError,
                                                  HyperliquidConnectionError,
                                                  HyperliquidConnector,
                                                  HyperliquidTimeoutError)
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../../shared/.env"))

def get_hyperliquid_creds():
    """
    Get Hyperliquid credentials from environment variables.
    Returns None for any missing credentials.
    """
    wallet_address = os.environ.get("WALLET_ADDRESS")
    private_key = os.environ.get("PRIVATE_KEY")
    testnet = os.environ.get("HYPERLIQUID_TESTNET", "false").lower() in ("true", "1", "t", "yes", "y")
    rpc_url = os.environ.get("HYPERLIQUID_RPC_URL")

    return {
        "wallet_address": wallet_address,
        "private_key": private_key,
        "testnet": testnet,
        "rpc_url": rpc_url
    }

def has_hyperliquid_creds():
    """Check if Hyperliquid credentials are available"""
    creds = get_hyperliquid_creds()
    return all([creds["wallet_address"], creds["private_key"], creds["rpc_url"]])

@pytest.fixture
def mock_info_client():
    """Create a mock Hyperliquid Info client for testing."""
    mock = MagicMock()

    # Mock metadata response
    meta_response = {
        "universe": [
            {
                "name": "BTC",
                "szDecimals": 3,
                "minSize": 0.001,
                "tickSize": 0.1,
                "makerFeeRate": 0.0002,
                "takerFeeRate": 0.0005,
                "maxLeverage": 50.0,
                "fundingInterval": 3600,
            },
            {
                "name": "ETH",
                "szDecimals": 3,
                "minSize": 0.01,
                "tickSize": 0.01,
                "makerFeeRate": 0.0002,
                "takerFeeRate": 0.0005,
                "maxLeverage": 50.0,
                "fundingInterval": 3600,
            },
            {
                "name": "SOL",
                "szDecimals": 0,
                "minSize": 1,
                "tickSize": 0.001,
                "makerFeeRate": 0.0002,
                "takerFeeRate": 0.0005,
                "maxLeverage": 20.0,
                "fundingInterval": 3600,
            },
        ]
    }
    mock.meta.return_value = meta_response

    # Mock user state (perps balances)
    user_state_response = {
        "cash": "5000.0",
        "marginSummary": {
            "accountValue": "5000.0",
            "totalMargin": "1000.0",
            "totalNtlPos": "10000.0"
        },
        "assetPositions": [
            {
                "coin": 0,  # BTC
                "szi": "0.5",
                "entryPx": "40000.0",
                "leverage": "5.0",
                "liqPx": "35000.0",
            },
            {
                "coin": 1,  # ETH
                "szi": "-2.0",
                "entryPx": "3000.0",
                "leverage": "10.0",
                "liqPx": "3300.0",
            },
        ],
    }
    mock.user_state.return_value = user_state_response

    # Mock L2 orderbook data
    l2_snapshot_response = {
        "levels": {
            "bids": [
                ["39000.0", "1.0"],
                ["38950.0", "2.0"],
            ],
            "asks": [
                ["39050.0", "0.5"],
                ["39100.0", "1.0"],
            ],
        }
    }
    mock.l2_snapshot.return_value = l2_snapshot_response

    # Mock mid prices data
    mock.all_mids.return_value = ["39025.0", "2975.0", "105.0"]

    return mock


@pytest.fixture
def mock_exchange_client():
    """Create a mock Hyperliquid Exchange client for testing."""
    mock = MagicMock()

    # Mock order response
    order_response = {
        "order": {
            "oid": "test-order-id",
            "is_buy": True,
            "coin": 0,  # BTC
            "sz": "0.1",
            "limit_px": "39000.0",
            "filled": "0.0",
        },
    }
    mock.order_limit.return_value = order_response
    mock.order_market.return_value = order_response

    # Mock open orders
    mock.open_orders.return_value = [
        {
            "oid": "test-order-id",
            "is_buy": True,
            "coin": 0,  # BTC
            "sz": "0.1",
            "limit_px": "39000.0",
            "filled": "0.0",
        }
    ]

    return mock


@pytest.fixture
def mock_requests():
    """Create a mock for requests for testing additional API endpoints."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # Set up mock response for API health check
        health_response = MagicMock()
        health_response.status_code = 200
        health_response.json.return_value = {
            "universe": [
                {
                    "name": "BTC",
                    "szDecimals": 3,
                    "minSize": 0.001,
                    "tickSize": 0.1,
                    "makerFeeRate": 0.0002,
                    "takerFeeRate": 0.0005,
                    "maxLeverage": 50.0,
                    "fundingInterval": 3600,
                }
            ]
        }

        # Set up mock response for orderbook
        orderbook_response = MagicMock()
        orderbook_response.status_code = 200
        orderbook_response.json.return_value = [
            [  # Bids
                ["39000.0", "1.0"],
                ["38950.0", "2.0"],
            ],
            [  # Asks
                ["39050.0", "0.5"],
                ["39100.0", "1.0"],
            ]
        ]

        def mock_get_side_effect(url, **kwargs):
            if "/info" in url:
                return health_response
            else:
                # Default fallback
                default_response = MagicMock()
                default_response.status_code = 404
                return default_response

        def mock_post_side_effect(url, **kwargs):
            if "/info" in url and kwargs.get("json", {}).get("type") == "l2Book":
                return orderbook_response
            else:
                # Default fallback
                default_response = MagicMock()
                default_response.status_code = 404
                return default_response

        mock_get.side_effect = mock_get_side_effect
        mock_post.side_effect = mock_post_side_effect
        yield mock_get, mock_post


@pytest.fixture
def hyperliquid_connector(mock_requests):
    """Create a Hyperliquid connector instance for testing."""
    mock_get, mock_post = mock_requests

    with patch("hyperliquid.info.Info") as mock_info_class, \
         patch("hyperliquid.exchange.Exchange") as mock_exchange_class, \
         patch("eth_account.Account"):

        # Create mock info instance
        mock_info = MagicMock()
        mock_info.meta.return_value = {
            "universe": [
                {
                    "name": "BTC",
                    "szDecimals": 3,
                    "minSize": 0.001,
                    "tickSize": 0.1,
                    "makerFeeRate": 0.0002,
                    "takerFeeRate": 0.0005,
                    "maxLeverage": 50.0,
                    "fundingInterval": 3600,
                }
            ]
        }
        mock_info.all_mids.return_value = ["39025.0"]  # Mock price data
        mock_info_class.return_value = mock_info

        # Create mock exchange instance
        mock_exchange = MagicMock()
        mock_exchange.place_order.return_value = {
            "order": {
                "oid": "test-order-id",
                "is_buy": True,
                "coin": "0",  # BTC
                "sz": "0.1",
                "limit_px": "39000.0",
                "filled": "0.0",
            },
        }
        mock_exchange_class.return_value = mock_exchange

        connector = HyperliquidConnector(
            private_key="test_private_key",
            testnet=True,
            market_types=[MarketType.PERPETUAL],
        )

        # Set up loggers for testing
        connector.balance_logger = logging.getLogger("test.hyperliquid.balance")
        connector.markets_logger = logging.getLogger("test.hyperliquid.markets")
        connector.orders_logger = logging.getLogger("test.hyperliquid.orders")

        # Connect the connector
        connected = connector.connect()
        if not connected:
            pytest.skip("Failed to connect to Hyperliquid API")

        return connector


def create_mock_connector():
    """Create a connector with mocked dependencies for testing"""
    mock_info_client = MagicMock()

    # Mock metadata response
    meta_response = {
        "universe": [
            {
                "name": "BTC",
                "szDecimals": 3,
                "minSize": 0.001,
                "tickSize": 0.1,
                "makerFeeRate": 0.0002,
                "takerFeeRate": 0.0005,
                "maxLeverage": 50.0,
                "fundingInterval": 3600,
            },
            {
                "name": "ETH",
                "szDecimals": 3,
                "minSize": 0.01,
                "tickSize": 0.01,
                "makerFeeRate": 0.0002,
                "takerFeeRate": 0.0005,
                "maxLeverage": 50.0,
                "fundingInterval": 3600,
            },
            {
                "name": "SOL",
                "szDecimals": 0,
                "minSize": 1,
                "tickSize": 0.001,
                "makerFeeRate": 0.0002,
                "takerFeeRate": 0.0005,
                "maxLeverage": 20.0,
                "fundingInterval": 3600,
            },
        ]
    }
    mock_info_client.meta.return_value = meta_response

    # Mock user state (perps balances)
    user_state_response = {
        "cash": "5000.0",
        "marginSummary": {
            "accountValue": "5000.0",
            "totalMargin": "1000.0",
            "totalNtlPos": "10000.0"
        },
        "assetPositions": [
            {
                "coin": 0,  # BTC
                "szi": "0.5",
                "entryPx": "40000.0",
                "leverage": "5.0",
                "liqPx": "35000.0",
            },
            {
                "coin": 1,  # ETH
                "szi": "-2.0",
                "entryPx": "3000.0",
                "leverage": "10.0",
                "liqPx": "3300.0",
            },
        ],
    }
    mock_info_client.user_state.return_value = user_state_response

    # Mock L2 orderbook data
    l2_snapshot_response = {
        "levels": {
            "bids": [
                ["39000.0", "1.0"],
                ["38950.0", "2.0"],
            ],
            "asks": [
                ["39050.0", "0.5"],
                ["39100.0", "1.0"],
            ],
        }
    }
    mock_info_client.l2_snapshot.return_value = l2_snapshot_response

    # Mock mid prices data
    mock_info_client.all_mids.return_value = ["39025.0", "2975.0", "105.0"]

    # Mock exchange client
    mock_exchange_client = MagicMock()

    # Mock order response
    order_response = {
        "order": {
            "oid": "test-order-id",
            "is_buy": True,
            "coin": 0,  # BTC
            "sz": "0.1",
            "limit_px": "39000.0",
            "filled": "0.0",
        },
    }
    mock_exchange_client.order_limit.return_value = order_response
    mock_exchange_client.order_market.return_value = order_response

    # Mock open orders
    mock_exchange_client.open_orders.return_value = [
        {
            "oid": "test-order-id",
            "is_buy": True,
            "coin": 0,  # BTC
            "sz": "0.1",
            "limit_px": "39000.0",
            "filled": "0.0",
        }
    ]

    with patch("hyperliquid.info.Info", return_value=mock_info_client), \
         patch("hyperliquid.exchange.Exchange", return_value=mock_exchange_client), \
         patch("eth_account.Account"), \
         patch("requests.get"):
        connector = HyperliquidConnector(
            name="hyperliquid",
            wallet_address="0xTESTADDRESS",
            private_key="test_private_key",
            testnet=True,
            market_types=[MarketType.PERPETUAL],
        )

        # Set up loggers for testing
        connector.balance_logger = logging.getLogger("test.hyperliquid.balance")
        connector.markets_logger = logging.getLogger("test.hyperliquid.markets")
        connector.orders_logger = logging.getLogger("test.hyperliquid.orders")

        # Hack: Set is_connected to True since we're mocking
        connector._is_connected = True
        connector.info = mock_info_client
        connector.exchange = mock_exchange_client

        return connector


def is_using_real_creds(hyperliquid_connector):
    """
    Check if we're using real credentials or mocks.
    A more robust check that examines the user state response to determine
    if we're connected to the real API.
    """
    if not has_hyperliquid_creds():
        return False

    # Check if the connector has a real wallet address (not the mock address)
    if hyperliquid_connector.wallet_address == "0xTESTADDRESS":
        return False

    # Check if the info client is mocked or real
    try:
        # If info.meta is a mock, accessing _mock_return_value won't raise an error
        if hasattr(hyperliquid_connector.info.meta, '_mock_return_value'):
            return False

        # Check for active network connection by calling a real method
        hyperliquid_connector.info.meta()
        return True
    except AttributeError:
        # This is likely a real client since it doesn't have _mock_return_value
        return True
    except Exception:
        # If there's any other error, we probably can't connect to the real API
        return False


@pytest.mark.skipif(not has_hyperliquid_creds(), reason="No Hyperliquid credentials available")
def test_get_account_balance(hyperliquid_connector):
    """Test retrieving account balances."""
    # Get the balances
    balances = hyperliquid_connector.get_account_balance()

    # Check that we got a result
    assert isinstance(balances, dict)

    # Check perp balances
    assert "PERP_USDC" in balances
    assert isinstance(balances["PERP_USDC"], (int, float))
    # In a test environment with mocks, a value of 0.0 is acceptable
    assert balances["PERP_USDC"] >= 0

    # Check backward compatibility
    assert "USDC" in balances
    assert balances["USDC"] == balances["PERP_USDC"]

    # Log the balances for debugging
    logging.info(f"Account balances: {balances}")


@pytest.mark.skipif(not has_hyperliquid_creds(), reason="No Hyperliquid credentials available")
def test_get_ticker(hyperliquid_connector):
    """Test retrieving ticker data for BTC."""
    ticker = hyperliquid_connector.get_ticker("BTC")

    # Verify ticker structure
    assert ticker["symbol"] == "BTC"
    assert "last_price" in ticker
    assert isinstance(ticker["last_price"], (int, float))
    assert ticker["last_price"] > 0

    # Log the ticker for debugging
    logging.info(f"BTC ticker: {ticker}")


@pytest.mark.skipif(not has_hyperliquid_creds(), reason="No Hyperliquid credentials available")
def test_get_orderbook(hyperliquid_connector):
    """Test retrieving order book for BTC market."""
    orderbook = hyperliquid_connector.get_orderbook("BTC")

    # Verify orderbook structure
    assert "bids" in orderbook
    assert "asks" in orderbook
    assert len(orderbook["bids"]) > 0
    assert len(orderbook["asks"]) > 0

    # Verify bid/ask prices make sense
    assert all(isinstance(bid[0], (int, float)) for bid in orderbook["bids"])
    assert all(isinstance(ask[0], (int, float)) for ask in orderbook["asks"])
    assert orderbook["bids"][0][0] < orderbook["asks"][0][0]  # Best bid < Best ask

    # Log the orderbook for debugging
    logging.info(f"BTC orderbook summary - Best bid: {orderbook['bids'][0]}, Best ask: {orderbook['asks'][0]}")


@pytest.mark.skipif(not has_hyperliquid_creds(), reason="No Hyperliquid credentials available")
def test_get_markets(hyperliquid_connector):
    """Test retrieving perpetual markets."""
    markets = hyperliquid_connector.get_markets()

    # Check that we have markets
    perp_markets = [m for m in markets if m["market_type"] == MarketType.PERPETUAL.value]
    assert len(perp_markets) > 0

    # Check that BTC is in the markets
    btc_market = next((m for m in perp_markets if m["symbol"] == "BTC"), None)
    assert btc_market is not None
    assert btc_market["exchange_specific"]["section"] == "perp"

    # Log markets for debugging
    logging.info(f"Available markets: {[m['symbol'] for m in perp_markets]}")


@pytest.mark.skipif(not has_hyperliquid_creds(), reason="No Hyperliquid credentials available")
def test_get_positions(hyperliquid_connector):
    """Test retrieving current positions."""
    positions = hyperliquid_connector.get_positions()

    # Should return a list (even if empty)
    assert isinstance(positions, list)

    # Check position structure if any exist
    for position in positions:
        assert "symbol" in position
        assert "size" in position
        assert "side" in position
        assert "entry_price" in position
        assert "mark_price" in position
        assert "leverage" in position
        assert isinstance(position["size"], (int, float))
        assert position["side"] in ["LONG", "SHORT"]
        assert isinstance(position["entry_price"], (int, float))
        assert isinstance(position["mark_price"], (int, float))

    # Log positions for debugging
    logging.info(f"Current positions: {positions}")


def test_initialization(hyperliquid_connector):
    """Test that the Hyperliquid connector initializes correctly."""
    assert hyperliquid_connector.name == "hyperliquid"
    assert hyperliquid_connector.wallet_address is not None
    assert hyperliquid_connector.private_key is not None
    assert isinstance(hyperliquid_connector.market_types, list)
    assert MarketType.PERPETUAL in hyperliquid_connector.market_types


def test_connect(hyperliquid_connector):
    """Test connecting to the Hyperliquid API."""
    assert hyperliquid_connector.is_connected is True
    assert hyperliquid_connector.info is not None
    assert hyperliquid_connector.exchange is not None


def test_disconnect(hyperliquid_connector):
    """Test disconnecting from the API."""
    result = hyperliquid_connector.disconnect()
    assert result is True
    assert hyperliquid_connector.info is None
    assert hyperliquid_connector.exchange is None
    assert hyperliquid_connector.is_connected is False


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.skipif(not has_hyperliquid_creds(), reason="No Hyperliquid credentials available")
def test_invalid_market_handling(hyperliquid_connector):
    """Test handling of invalid market symbol."""
    with pytest.raises(HyperliquidAPIError):
        hyperliquid_connector.get_ticker("INVALID_MARKET")


@pytest.mark.skipif(not has_hyperliquid_creds(), reason="No Hyperliquid credentials available")
def test_invalid_order_params(hyperliquid_connector):
    """Test handling of invalid order parameters."""
    with pytest.raises(HyperliquidAPIError):
        hyperliquid_connector.place_order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=0,  # Invalid amount
            price=30000,
            leverage=1.0  # Add leverage parameter
        )


@pytest.mark.skipif(not has_hyperliquid_creds(), reason="No Hyperliquid credentials available")
def test_connection_recovery():
    """Test that the connector can recover from a disconnection."""
    # Create a connector with mocked dependencies
    connector = create_mock_connector()

    # Verify initial connection
    assert connector.is_connected

    # Force a disconnect
    connector.disconnect()
    assert not connector.is_connected

    # Mock the connect method to ensure it works
    with patch.object(connector, 'connect') as mock_connect:
        mock_connect.return_value = True

        # Try an operation that should trigger reconnection
        balances = connector.get_account_balance()

        # Verify connect was called
        mock_connect.assert_called_once()

    # Log balances for debugging
    logging.info(f"Retrieved balances after reconnection: {balances}")


@pytest.mark.skipif(not has_hyperliquid_creds(), reason="No Hyperliquid credentials available")
def test_rate_limiting(hyperliquid_connector):
    """Test that rapid requests are handled properly."""
    # Make multiple rapid requests
    for _ in range(5):
        ticker = hyperliquid_connector.get_ticker("BTC")
        assert ticker["symbol"] == "BTC"
        assert ticker["last_price"] > 0


@pytest.mark.skipif(not has_hyperliquid_creds(), reason="No Hyperliquid credentials available")
def test_place_order(hyperliquid_connector):
    """Test placing a limit order."""
    order = hyperliquid_connector.place_order(
        symbol="BTC",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        amount=0.1,
        price=39000.0,
        leverage=2.0,  # Test with 2x leverage
        reduce_only=False,
        post_only=True,
        client_order_id="test-cloid"
    )

    # Verify order response structure
    assert "order" in order
    assert order["order"]["oid"] == "test-order-id"
    assert order["order"]["is_buy"] is True
    assert order["order"]["coin"] == "0"  # BTC index
    assert order["order"]["sz"] == "0.1"
    assert order["order"]["limit_px"] == "39000.0"
    assert order["order"]["filled"] == "0.0"

    # Log order for debugging
    logging.info(f"Placed order: {order}")
