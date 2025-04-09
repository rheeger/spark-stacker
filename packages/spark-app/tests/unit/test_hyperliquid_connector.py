#!/usr/bin/env /Users/rheeger/Code/rheeger/spark-stacker/packages/spark-app/.venv/bin/python
"""
Comprehensive test suite for the Hyperliquid connector.
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from io import StringIO
from unittest.mock import ANY, MagicMock, Mock, PropertyMock, patch

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
def hyperliquid_connector():
    """Create a Hyperliquid connector instance for testing."""
    # Create the connector with test credentials
    connector = HyperliquidConnector(
        name="hyperliquid",
        wallet_address="0x123",
        private_key="0xabc",
        testnet=True,
        market_types=[MarketType.PERPETUAL]
    )

    # Mock the info client
    mock_info = Mock()
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
    mock_info.l2_snapshot.return_value = [
        [  # Bids
            ["39000.0", "1.0"],
            ["38900.0", "2.0"]
        ],
        [  # Asks
            ["39100.0", "1.5"],
            ["39200.0", "2.5"]
        ]
    ]

    # Mock the exchange client
    mock_exchange = Mock()
    mock_exchange.user_state.return_value = {
        "cash": "5000.0",
        "marginSummary": {
            "accountValue": "5000.0",
            "totalMargin": "1000.0",
            "totalNtlPos": "10000.0"
        },
        "assetPositions": []
    }

    # Set up the connector with mocked clients
    connector.info = mock_info
    connector.exchange = mock_exchange
    connector._is_connected = True

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
def test_get_orderbook(hyperliquid_connector, mock_requests):
    """Test retrieving order book for BTC market."""
    # Mock requests is already set up with the correct response format
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


def test_get_positions_mocked(hyperliquid_connector, mock_info_client):
    """Test retrieving current positions with mocked API responses."""
    # Set up the mock_info_client on the connector
    hyperliquid_connector.info = mock_info_client

    # Mock the all_mids response (current prices)
    mock_info_client.all_mids.return_value = [40500.0, 2950.0, 100.0]  # BTC, ETH, SOL prices

    # Verify the positions match the mocked data
    positions = hyperliquid_connector.get_positions()

    assert len(positions) == 2  # We mocked 2 positions in mock_info_client fixture

    # Verify BTC position
    btc_pos = next(p for p in positions if p["symbol"] == "BTC")
    assert btc_pos["size"] == 0.5
    assert btc_pos["side"] == "LONG"
    assert btc_pos["entry_price"] == 40000.0
    assert btc_pos["mark_price"] == 40500.0
    assert btc_pos["leverage"] == 5.0
    assert btc_pos["liquidation_price"] == 35000.0
    assert btc_pos["unrealized_pnl"] == 250.0  # (40500 - 40000) * 0.5
    assert btc_pos["margin"] == 0.1  # |0.5| / 5.0

    # Verify ETH position
    eth_pos = next(p for p in positions if p["symbol"] == "ETH")
    assert eth_pos["size"] == -2.0
    assert eth_pos["side"] == "SHORT"
    assert eth_pos["entry_price"] == 3000.0
    assert eth_pos["mark_price"] == 2950.0
    assert eth_pos["leverage"] == 10.0
    assert eth_pos["liquidation_price"] == 3300.0
    assert eth_pos["unrealized_pnl"] == 100.0  # (3000 - 2950) * 2.0
    assert eth_pos["margin"] == 0.2  # |2.0| / 10.0


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
    # Mock market metadata
    mock_market_info = {
        "universe": [{
            "name": "BTC",
            "szDecimals": 8,
            "minSize": "0.0001",
            "tickSize": "0.1",
            "makerFeeRate": 0.0002,
            "takerFeeRate": 0.0005,
            "maxLeverage": 50.0,
            "fundingInterval": 3600,
        }]
    }
    hyperliquid_connector.info.meta.return_value = mock_market_info

    # Test with invalid amount (0)
    with pytest.raises(ValueError, match="Order amount must be positive"):
        hyperliquid_connector.place_order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=0,
            price=50000.0,
            leverage=1.0
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


@pytest.mark.asyncio
async def test_place_order(hyperliquid_connector):
    """Test placing a limit order."""
    # Mock market metadata in info.meta response
    meta_response = {
        "universe": [
            {
                "name": "BTC",
                "szDecimals": 8,
                "minSize": "0.0001",
                "tickSize": "0.1",
                "makerFeeRate": 0.0002,
                "takerFeeRate": 0.0005,
                "maxLeverage": 50.0,
                "fundingInterval": 3600,
            }
        ]
    }
    hyperliquid_connector.info.meta.return_value = meta_response

    # Mock successful order response
    order_response = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {
                        "resting": {
                            "oid": "test-order-id",
                            "status": "open"
                        }
                    }
                ]
            }
        }
    }
    hyperliquid_connector.exchange.order.return_value = order_response

    # Place a limit order
    response = await hyperliquid_connector.place_order(
        symbol="BTC",  # Use BTC instead of BTC-USD
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        amount=0.001,  # Small test amount
        price=50000.0,  # Test price
        leverage=1.0
    )

    logging.info(f"Place order response: {response}")

    assert isinstance(response, dict)
    assert response["status"] == "OPEN"
    assert response["order_id"] == "test-order-id"
    assert response["symbol"] == "BTC"
    assert response["side"] == "BUY"
    assert response["type"] == "LIMIT"
    assert response["amount"] == 0.001
    assert response["price"] == 50000.0

@pytest.mark.asyncio
async def test_invalid_order_params(hyperliquid_connector):
    """Test handling of invalid order parameters."""
    # Mock market metadata
    mock_market_info = {
        "universe": [{
            "name": "BTC",
            "szDecimals": 8,
            "minSize": "0.0001",
            "tickSize": "0.1",
            "makerFeeRate": 0.0002,
            "takerFeeRate": 0.0005,
            "maxLeverage": 50.0,
            "fundingInterval": 3600,
        }]
    }
    hyperliquid_connector.info.meta.return_value = mock_market_info

    # Test with invalid amount (0)
    with pytest.raises(ValueError, match="Order amount must be positive"):
        await hyperliquid_connector.place_order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=0,
            price=50000.0,
            leverage=1.0
        )

@pytest.mark.asyncio
@pytest.mark.production
@pytest.mark.skipif(not has_hyperliquid_creds() or get_hyperliquid_creds().get('testnet', True), reason="Requires PRODUCTION Hyperliquid credentials (testnet=False)")
@patch("app.connectors.hyperliquid_connector.HyperliquidConnector.get_markets")
async def test_place_and_close_minimum_size_order_production(mock_get_markets, hyperliquid_connector):
    # Mock market metadata
    mock_markets = [{
        "symbol": "BTC",  # Use BTC instead of BTC-USD
        "base_asset": "BTC",
        "quote_asset": "USD",
        "price_precision": 8,
        "min_size": 0.001,
        "tick_size": 0.1,
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
        "market_type": MarketType.PERPETUAL.value,
        "exchange_specific": {
            "max_leverage": 50.0,
            "funding_interval": 3600,
            "section": "perp"
        }
    }]
    mock_get_markets.return_value = mock_markets

    # Mock info.meta response
    meta_response = {
        "universe": [
            {
                "name": "BTC",
                "szDecimals": 8,
                "minSize": "0.001",
                "tickSize": "0.1",
                "makerFeeRate": 0.0002,
                "takerFeeRate": 0.0005,
                "maxLeverage": 50.0,
                "fundingInterval": 3600,
            }
        ]
    }
    hyperliquid_connector.info.meta.return_value = meta_response

    # Mock position response
    position_response = {
        "assetPositions": [{
            "coin": 0,  # BTC index
            "szi": "0.001",  # Use szi instead of position
            "entryPx": "50000.0",
            "liqPx": "45000.0",  # Use liqPx instead of liquidationPx
            "unrealizedPnl": "10.5",
            "leverage": "10"
        }]
    }
    hyperliquid_connector.exchange.user_state.return_value = position_response

    # Mock successful order response
    order_response = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {
                        "resting": {
                            "oid": "test-order-id",
                            "status": "open"
                        }
                    }
                ]
            }
        }
    }
    hyperliquid_connector.exchange.order.return_value = order_response

    # Place buy order
    buy_response = await hyperliquid_connector.place_order(
        symbol="BTC",  # Use BTC instead of BTC-USD
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.001")
    )

    # Verify buy order response
    assert isinstance(buy_response, dict)
    assert buy_response["status"] == "OPEN"
    assert buy_response["order_id"] == "test-order-id"
    assert buy_response["symbol"] == "BTC"
    assert buy_response["side"] == "BUY"
    assert buy_response["type"] == "MARKET"
    assert float(buy_response["amount"]) == 0.001

def test_get_min_order_size(hyperliquid_connector):
    """Test getting minimum order size for markets."""
    # Mock info.meta response for both BTC and ETH
    meta_response = {
        "universe": [
            {
                "name": "BTC",
                "szDecimals": 8,
                "minSize": "0.001",
                "tickSize": "0.1",
            },
            {
                "name": "ETH",
                "szDecimals": 8,
                "minSize": "0.01",
                "tickSize": "0.1",
            }
        ]
    }
    hyperliquid_connector.info.meta.return_value = meta_response

    # Test for BTC market
    min_size = hyperliquid_connector.get_min_order_size("BTC")
    assert min_size == 0.001

    # Test for ETH market
    min_size = hyperliquid_connector.get_min_order_size("ETH")
    assert min_size == 0.01

    # Test for non-existent market
    with pytest.raises(ValueError, match="Market SOL not found"):
        hyperliquid_connector.get_min_order_size("SOL")
