#!/usr/bin/env python
"""
Specialized test suite for Hyperliquid historical candles functionality.

This test focuses on verifying symbol translation and handling for candle data requests.
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests
from app.connectors.base_connector import MarketType
from app.connectors.hyperliquid_connector import (HyperliquidAPIError,
                                                  HyperliquidConnectionError,
                                                  HyperliquidConnector)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test configuration
USE_TESTNET = os.environ.get('HYPERLIQUID_TESTNET', 'true').lower() in ('true', '1', 't', 'yes', 'y')
API_URL = 'https://api.hyperliquid-testnet.xyz' if USE_TESTNET else 'https://api.hyperliquid.xyz'

@pytest.fixture
def mock_universe_data():
    """Return mock universe data with multiple assets."""
    return {
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

@pytest.fixture
def mock_candle_data():
    """Return mock candle data."""
    return [
        {"t": int(time.time() * 1000) - 3600000, "o": "1600.5", "h": "1605.3", "l": "1595.8", "c": "1602.7", "v": "102.5"},
        {"t": int(time.time() * 1000) - 3000000, "o": "1602.7", "h": "1610.2", "l": "1600.1", "c": "1607.9", "v": "98.7"},
        {"t": int(time.time() * 1000) - 2400000, "o": "1607.9", "h": "1612.5", "l": "1605.3", "c": "1609.8", "v": "78.3"}
    ]

@pytest.mark.parametrize("input_symbol,expected", [
    ("ETH-USD", "ETH"),
    ("eth-usd", "ETH"),
    ("ETH", "ETH"),
    ("SOL-PERP", "SOL"),
    ("btc-USD", "BTC")
])
def test_symbol_translation_function(input_symbol, expected):
    """Test that the translate_symbol function works correctly with various formats."""
    connector = HyperliquidConnector(
        name="test_hyperliquid",
        wallet_address="0x123",
        private_key="0xabc",
        testnet=True,
        market_types=[MarketType.PERPETUAL]
    )
    result = connector.translate_symbol(input_symbol)
    assert result == expected, f"Expected {input_symbol} to translate to {expected}, got {result}"

def test_hyperliquid_connector_translate_symbol_validation():
    """Test that the connector validates symbols correctly."""
    connector = HyperliquidConnector(
        name="test_hyperliquid",
        wallet_address="0x123",
        private_key="0xabc",
        testnet=True,
        market_types=[MarketType.PERPETUAL]
    )

    # Test valid symbols
    assert connector.translate_symbol("ETH-USD") == "ETH"
    assert connector.translate_symbol("BTC-USD") == "BTC"
    assert connector.translate_symbol("SOL-PERP") == "SOL"

    # Test with mixed case
    assert connector.translate_symbol("eth-usd") == "ETH"
    assert connector.translate_symbol("Eth-Usd") == "ETH"

    # Test with whitespace
    assert connector.translate_symbol(" ETH-USD ") == "ETH"

    # Test already translated symbol
    assert connector.translate_symbol("ETH") == "ETH"

def test_real_hyperliquid_symbols():
    """
    Test with real Hyperliquid symbol list to ensure our symbol translation works
    with all valid coins in the universe.
    """
    # This is a representative sample of actual Hyperliquid symbols to test against
    real_symbols = [
        ("BTC-USD", "BTC"),
        ("ETH-USD", "ETH"),
        ("SOL-USD", "SOL"),
        ("XRP-USD", "XRP"),
        ("DOGE-USD", "DOGE"),
        ("AVAX-USD", "AVAX"),
        ("SHIB-USD", "SHIB"),
        ("LINK-USD", "LINK"),
        ("DOT-USD", "DOT"),
        ("LTC-USD", "LTC"),
    ]

    connector = HyperliquidConnector(
        name="test_hyperliquid",
        wallet_address="0x123",
        private_key="0xabc",
        testnet=True,
        market_types=[MarketType.PERPETUAL]
    )

    for input_symbol, expected_output in real_symbols:
        result = connector.translate_symbol(input_symbol)
        assert result == expected_output, f"Failed to translate {input_symbol} to {expected_output}, got {result}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
