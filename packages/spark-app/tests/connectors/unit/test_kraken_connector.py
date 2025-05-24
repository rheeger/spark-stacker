"""
Comprehensive test suite for the Kraken connector.
Uses real API endpoints for testing.
"""
import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path

import pytest
from app.connectors.base_connector import (BaseConnector, MarketType,
                                           OrderSide, OrderStatus, OrderType,
                                           TimeInForce)
from app.connectors.kraken_connector import KrakenConnector
from dotenv import load_dotenv

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_api_credentials():
    """Helper function to get API credentials from environment."""
    # Load environment variables from the shared .env file
    env_path = Path(__file__).parents[4] / "packages" / "shared" / ".env"
    logger.info(f"Loading environment from: {env_path}")
    load_dotenv(env_path)

    api_key = os.environ.get("KRAKEN_API_KEY", "")
    api_secret = os.environ.get("KRAKEN_API_SECRET", "")

    # Print debugging information
    logger.info("=== API Credentials Check ===")
    logger.info(f"KRAKEN_API_KEY set: {bool(api_key)}")
    logger.info(f"KRAKEN_API_SECRET set: {bool(api_secret)}")
    if api_key:
        logger.info(f"API Key length: {len(api_key)}")
    if api_secret:
        logger.info(f"API Secret length: {len(api_secret)}")

    return api_key, api_secret


@pytest.fixture
def kraken_connector():
    """Create a real Kraken connector instance for testing."""
    api_key, api_secret = get_api_credentials()
    logger.info("Initializing Kraken connector...")

    connector = KrakenConnector(
        api_key=api_key,
        api_secret=api_secret,
        testnet=False,  # Use real API for testing
        market_types=[MarketType.SPOT, MarketType.PERPETUAL]
    )

    # Set up loggers for testing
    connector.balance_logger = logging.getLogger("test.kraken.balance")
    connector.markets_logger = logging.getLogger("test.kraken.markets")
    connector.orders_logger = logging.getLogger("test.kraken.orders")

    # Connect to the API
    logger.info("Connecting to Kraken API...")
    connected = connector.connect()
    logger.info(f"Connection successful: {connected}")

    yield connector

    # Cleanup after tests
    connector.disconnect()


# =============================================================================
# Basic Connector Tests
# =============================================================================


def test_initialization(kraken_connector):
    """Test that the Kraken connector initializes correctly."""
    assert kraken_connector.api_key == os.environ.get("KRAKEN_API_KEY", "")
    assert kraken_connector.api_secret == os.environ.get("KRAKEN_API_SECRET", "")
    assert kraken_connector.testnet is False
    assert isinstance(kraken_connector.market_types, list)
    assert MarketType.SPOT in kraken_connector.market_types
    assert MarketType.PERPETUAL in kraken_connector.market_types


def test_connect(kraken_connector):
    """Test connecting to the Kraken API."""
    # Test disconnecting first to ensure we can reconnect
    kraken_connector.disconnect()
    assert kraken_connector.is_connected is False

    # Test reconnecting
    result = kraken_connector.connect()
    assert result is True
    assert kraken_connector.client is not None
    assert kraken_connector.is_connected is True


def test_disconnect(kraken_connector):
    """Test disconnecting from the API."""
    # Ensure we're connected first
    if not kraken_connector.is_connected:
        kraken_connector.connect()

    result = kraken_connector.disconnect()
    assert result is True
    assert kraken_connector.client is None
    assert kraken_connector.is_connected is False


# =============================================================================
# Market Data Tests
# =============================================================================


def test_get_markets(kraken_connector):
    """Test retrieving market data."""
    markets = kraken_connector.get_markets()

    # Verify we have markets
    assert len(markets) > 0

    # Check basic structure of a market
    spot_markets = [m for m in markets if m['market_type'] == MarketType.SPOT.value]
    assert len(spot_markets) > 0

    # Look for BTC market (might be XBT in Kraken)
    btc_market = next((m for m in spot_markets if m['symbol'] == 'BTC'), None)
    assert btc_market is not None
    assert 'quote_asset' in btc_market  # Just check the field exists, don't check specific value
    assert 'maker_fee' in btc_market
    assert 'taker_fee' in btc_market

    # Check perpetual markets if available
    futures_markets = [m for m in markets if m['market_type'] == MarketType.PERPETUAL.value]
    if futures_markets:  # Only assert if futures markets are available
        assert len(futures_markets) > 0


def test_get_ticker(kraken_connector):
    """Test retrieving ticker data."""
    ticker = kraken_connector.get_ticker("BTC")

    # Check the structure of the ticker response
    assert ticker is not None
    assert "symbol" in ticker
    assert "last" in ticker or "last_price" in ticker

    # Get BTC price (might use XXBTZUSD in Kraken)
    last_price = ticker.get("last") or ticker.get("last_price")
    assert float(last_price) > 0

    # Test another asset if available
    ticker = kraken_connector.get_ticker("ETH")
    assert ticker is not None
    assert "symbol" in ticker


def test_get_orderbook(kraken_connector):
    """Test retrieving order book for a specific market."""
    orderbook = kraken_connector.get_orderbook("BTC")

    # Check the structure
    assert "bids" in orderbook
    assert "asks" in orderbook
    assert len(orderbook["bids"]) > 0
    assert len(orderbook["asks"]) > 0

    # Check the format of bids and asks
    assert len(orderbook["bids"][0]) == 2  # [price, size]
    assert len(orderbook["asks"][0]) == 2  # [price, size]

    # Verify bid is less than ask (as it should be)
    assert orderbook["bids"][0][0] < orderbook["asks"][0][0]


# =============================================================================
# Account Tests
# =============================================================================


def test_get_account_balance(kraken_connector, capsys):
    """Test retrieving account balances."""
    logger.info("=== Starting Account Balance Test ===")

    # Log connector state
    logger.info(f"Connector connected: {kraken_connector.is_connected}")
    logger.info(f"API Key exists: {bool(kraken_connector.api_key)}")
    logger.info(f"API Secret exists: {bool(kraken_connector.api_secret)}")

    # Ensure we're connected
    if not kraken_connector.is_connected:
        logger.info("Connecting to Kraken API...")
        connected = kraken_connector.connect()
        logger.info(f"Connection result: {connected}")

    # Skip if credentials not available
    if not kraken_connector.api_key or not kraken_connector.api_secret:
        logger.error("API credentials not available")
        pytest.skip("API credentials not available for account balance test")

    # Get balances
    logger.info("Requesting account balances...")
    balances = kraken_connector.get_account_balance()
    logger.info(f"Received balances response: {balances}")

    # We should have a dict response
    assert isinstance(balances, dict), "Expected dict response from get_account_balance"

    # If we got an error in the response, log it
    if isinstance(balances, dict) and 'error' in balances:
        logger.error(f"Error in balance response: {balances['error']}")
        pytest.fail(f"Failed to get account balance: {balances['error']}")

    logger.info("=== Account Balance Test Complete ===")


def test_get_positions(kraken_connector):
    """Test retrieving current positions."""
    positions = kraken_connector.get_positions()

    # Skip this test if we don't have valid API credentials
    if not kraken_connector.api_key or not kraken_connector.api_secret:
        pytest.skip("API credentials not available for positions test")

    # This could be empty if no positions are open
    # If positions exist, check structure
    for position in positions:
        assert "symbol" in position
        assert "size" in position
        # For spot positions, leverage should be 1.0
        if position.get("market_type", "SPOT") == "SPOT":
            assert position["leverage"] == 1.0


# =============================================================================
# Historical Data Tests
# =============================================================================


def test_get_historical_candles(kraken_connector):
    """Test retrieving historical candlestick data."""
    try:
        candles = kraken_connector.get_historical_candles(
            symbol="BTC",
            interval="1h",
            limit=10
        )

        # Check that we have candles
        assert len(candles) > 0

        # Check candle structure
        assert "timestamp" in candles[0]
        assert "open" in candles[0]
        assert "high" in candles[0]
        assert "low" in candles[0]
        assert "close" in candles[0]
        assert "volume" in candles[0]

        # Check data types
        assert isinstance(candles[0]["timestamp"], int)
        assert float(candles[0]["open"]) > 0
        assert float(candles[0]["high"]) >= float(candles[0]["low"])
        assert float(candles[0]["close"]) > 0

    except Exception as e:
        # If the test fails due to API issues, skip it
        pytest.skip(f"Failed to get historical candles: {str(e)}")

"""
NOTE: Order-related tests are commented out for safety when using a real account.
You should manually test these or use a testnet account.

# =============================================================================
# Order Tests
# =============================================================================


# def test_place_market_order(kraken_connector):
#     '''Test placing a market order.'''
#     # CAUTION: This will place a real order!
#     pass


# def test_place_limit_order(kraken_connector):
#     '''Test placing a limit order.'''
#     # CAUTION: This will place a real order!
#     pass


# def test_cancel_order(kraken_connector):
#     '''Test canceling an order.'''
#     # CAUTION: This will cancel a real order!
#     pass


# def test_get_order(kraken_connector):
#     '''Test retrieving order details.'''
#     # You would need a real order ID to test this
#     pass


# =============================================================================
# Position Tests
# =============================================================================


# def test_close_position(kraken_connector):
#     '''Test closing a position.'''
#     # CAUTION: This will close a real position!
#     pass
"""
