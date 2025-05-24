#!/usr/bin/env python
"""
Test suite for verifying Hyperliquid historical candles API request format.

This test focuses on ensuring our connector uses the correct API request format
for candle data requests after discovering issues in production.
"""
import json
import logging
import os
import time
from unittest.mock import MagicMock, patch

import pytest
import requests
from app.connectors.base_connector import MarketType
from app.connectors.hyperliquid_connector import HyperliquidConnector

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test configuration
USE_TESTNET = os.environ.get('HYPERLIQUID_TESTNET', 'true').lower() in ('true', '1', 't', 'yes', 'y')
SKIP_REAL_API = os.environ.get('SKIP_REAL_API', 'false').lower() in ('true', '1', 't', 'yes', 'y')

def test_historical_candles_request_format():
    """
    Test that we're using the correct format for the historical candles API request.

    This test verifies that the connector is using the required "candleSnapshot" type
    and nesting parameters in a "req" object as expected by the Hyperliquid API.
    """
    connector = HyperliquidConnector(
        name="test_hyperliquid",
        wallet_address="0x123",
        private_key="0xabc",
        testnet=True,
        market_types=[MarketType.PERPETUAL]
    )

    # Create mock universe data directly
    mock_universe = {
        "universe": [
            {
                "name": "BTC",
                "szDecimals": 3,
                "minSize": 0.001,
                "tickSize": 0.1,
            },
            {
                "name": "ETH",
                "szDecimals": 3,
                "minSize": 0.01,
                "tickSize": 0.01,
            },
            {
                "name": "SOL",
                "szDecimals": 0,
                "minSize": 1,
                "tickSize": 0.001,
            },
        ]
    }

    # Create mock candle data
    mock_candles = [
        {
            "t": 1744548945000,
            "T": 1744549004999,
            "s": "ETH",
            "i": "1m",
            "o": "1603.35",
            "c": "1603.38",
            "h": "1603.40",
            "l": "1603.30",
            "v": "10.5",
            "n": 25
        },
        {
            "t": 1744549005000,
            "T": 1744549064999,
            "s": "ETH",
            "i": "1m",
            "o": "1603.38",
            "c": "1603.40",
            "h": "1603.45",
            "l": "1603.35",
            "v": "12.3",
            "n": 30
        }
    ]

    # Mock the universe data to validate the symbol
    with patch.object(connector, 'info') as mock_info:
        mock_info.meta.return_value = mock_universe

        # Mock the POST request to capture the actual request data sent
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_candles
            mock_post.return_value = mock_response

            # Call the function we want to test
            connector.get_historical_candles("ETH-USD", "1m", start_time=1234567890000)

            # Check that the POST call was made with the correct request structure
            mock_post.assert_called()
            args, kwargs = mock_post.call_args

            # Extract and check the request data
            request_data = kwargs.get('json', {})
            assert "type" in request_data, "Missing 'type' in request data"
            assert request_data["type"] == "candleSnapshot", f"Wrong request type: {request_data['type']}"
            assert "req" in request_data, "Missing 'req' object in request data"

            req = request_data["req"]
            assert "coin" in req, "Missing 'coin' in req object"
            assert req["coin"] == "ETH", f"Wrong coin name: {req['coin']}"
            assert "interval" in req, "Missing 'interval' in req object"
            assert req["interval"] == "1m", f"Wrong interval: {req['interval']}"
            assert "startTime" in req, "Missing 'startTime' in req object"
            assert req["startTime"] == 1234567890000, f"Wrong startTime: {req['startTime']}"

def test_historical_candles_response_handling():
    """
    Test that the connector correctly parses the response format from the Hyperliquid API.
    """
    connector = HyperliquidConnector(
        name="test_hyperliquid",
        wallet_address="0x123",
        private_key="0xabc",
        testnet=True,
        market_types=[MarketType.PERPETUAL]
    )

    # Mock universe data
    mock_universe = {
        "universe": [
            {"name": "ETH", "szDecimals": 3}
        ]
    }

    # Create sample candle data in the format returned by Hyperliquid API
    api_response = [
        {
            "t": 1744548945000,  # timestamp
            "T": 1744549004999,  # end timestamp
            "s": "ETH",          # symbol
            "i": "1m",           # interval
            "o": "1603.35",      # open
            "c": "1603.38",      # close
            "h": "1603.40",      # high
            "l": "1603.30",      # low
            "v": "10.5",         # volume
            "n": 25              # number of trades
        },
        {
            "t": 1744549005000,
            "T": 1744549064999,
            "s": "ETH",
            "i": "1m",
            "o": "1603.38",
            "c": "1603.40",
            "h": "1603.45",
            "l": "1603.35",
            "v": "12.3",
            "n": 30
        }
    ]

    # Mock the API interactions
    with patch.object(connector, 'info') as mock_info:
        mock_info.meta.return_value = mock_universe

        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = api_response
            mock_post.return_value = mock_response

            # Call the function we want to test
            candles = connector.get_historical_candles("ETH-USD", "1m", limit=10)

            # Check that the response was parsed correctly
            assert isinstance(candles, list)
            assert len(candles) == 2

            # Verify first candle
            assert candles[0]["timestamp"] == 1744548945000
            assert candles[0]["open"] == 1603.35
            assert candles[0]["high"] == 1603.40
            assert candles[0]["low"] == 1603.30
            assert candles[0]["close"] == 1603.38
            assert candles[0]["volume"] == 10.5
            assert candles[0]["symbol"] == "ETH"
            assert candles[0]["interval"] == "1m"
            assert candles[0]["trades"] == 25

            # Verify second candle
            assert candles[1]["timestamp"] == 1744549005000
            assert candles[1]["open"] == 1603.38
            assert candles[1]["high"] == 1603.45
            assert candles[1]["low"] == 1603.35
            assert candles[1]["close"] == 1603.40
            assert candles[1]["volume"] == 12.3

@pytest.mark.skipif(SKIP_REAL_API, reason="Skipping real API test")
@pytest.mark.slow
def test_real_api_historical_candles():
    """Test getting historical candles from the real Hyperliquid API."""
    connector = HyperliquidConnector(testnet=True)
    connector.connect()

    try:
        # Test different symbols - limit to only ETH for rate limit concerns
        # Reducing to only one symbol since rate limits are causing failures
        symbols = ["ETH-USD"]  # Removed BTC-USD to avoid rate limit issues

        # Fetch historical candles - this should fail with 422 if request format is wrong
        for symbol in symbols:
            interval = "1m"
            end_time = int(time.time() * 1000)
            start_time = end_time - (3600 * 1000)  # 1 hour ago

            # Get candles
            candles = connector.get_historical_candles(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=60
            )

            # Real integration test must return data, not an empty list on error
            assert len(candles) > 0, f"Failed to fetch candles for {symbol} using real API"

            # Verify candle structure
            first_candle = candles[0]
            assert "timestamp" in first_candle, "Missing timestamp in candle data"
            assert "open" in first_candle, "Missing open price in candle data"
            assert "high" in first_candle, "Missing high price in candle data"
            assert "low" in first_candle, "Missing low price in candle data"
            assert "close" in first_candle, "Missing close price in candle data"

            # Log success
            logger.info(f"Successfully fetched {len(candles)} candles from real Hyperliquid API")
            logger.info(f"First candle: {first_candle}")

            # Add a delay to avoid rate limits if testing multiple symbols
            time.sleep(3)

    except requests.exceptions.HTTPError as e:
        if "429" in str(e):  # Rate limit error
            logger.warning(f"Rate limit exceeded during testing: {e}")
            # Skip the test rather than failing when we hit rate limits
            pytest.skip("Rate limit exceeded, skipping test")
        else:
            logger.error(f"HTTP error in test_real_api_historical_candles: {e}")
            assert False, f"HTTP error in test_real_api_historical_candles: {e}"
    except Exception as e:
        logger.error(f"Error in test_real_api_historical_candles: {e}")
        assert False, f"Error in test_real_api_historical_candles: {e}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
