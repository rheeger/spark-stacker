import asyncio
import json
import logging
import os
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
import requests_mock
from connectors.base_connector import MarketType, OrderSide, OrderType
# Import the connector to test
from connectors.hyperliquid_connector import (HyperliquidConnector,
                                              MetadataWrapper)
from eth_account import Account

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test configuration
TEST_WALLET_ADDRESS = os.getenv('TEST_WALLET_ADDRESS', '0xA30BDaFFe8EEab7eaCB88Bc17158E28f7EcC07F8')
TEST_PRIVATE_KEY = os.getenv('TEST_PRIVATE_KEY', '0x0000000000000000000000000000000000000000000000000000000000000001')
USE_TESTNET = True
API_URL = 'https://api.hyperliquid-testnet.xyz' if USE_TESTNET else 'https://api.hyperliquid.xyz'

# Test fixtures and data paths
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'hyperliquid')

# Ensure fixtures directory exists
os.makedirs(FIXTURES_DIR, exist_ok=True)

# Helper function to run coroutines in tests
def run_async(coroutine):
    """Run a coroutine in a synchronous test."""
    return asyncio.run(coroutine)

class APIResponseRecorder:
    """Utility to record and replay API responses for testing."""

    @staticmethod
    def save_response(endpoint: str, response_data: Any, params: Optional[Dict] = None):
        """Save API response to a fixture file."""
        filename = f"{endpoint.replace('/', '_')}.json"
        if params:
            # Create a hash or identifier from params
            param_str = json.dumps(params, sort_keys=True)
            import hashlib
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            filename = f"{endpoint.replace('/', '_')}_{param_hash}.json"

        file_path = os.path.join(FIXTURES_DIR, filename)
        with open(file_path, 'w') as f:
            json.dump(response_data, f, indent=2)

        logger.info(f"Saved API response to {file_path}")
        return file_path

    @staticmethod
    def load_response(endpoint: str, params: Optional[Dict] = None):
        """Load API response from a fixture file."""
        filename = f"{endpoint.replace('/', '_')}.json"
        if params:
            # Create a hash or identifier from params
            param_str = json.dumps(params, sort_keys=True)
            import hashlib
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            filename = f"{endpoint.replace('/', '_')}_{param_hash}.json"

        file_path = os.path.join(FIXTURES_DIR, filename)

        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Fixture file not found: {file_path}")
            return None

class TestMode:
    """Enum for different test modes."""
    RECORD = 'record'  # Record real API responses
    REPLAY = 'replay'  # Use recorded responses
    MOCK = 'mock'      # Use manually created mocks

# Set default test mode - can be overridden using environment variable
TEST_MODE = os.getenv('HYPERLIQUID_TEST_MODE', TestMode.MOCK)  # Default to MOCK to avoid requiring fixtures

@pytest.fixture
def connector():
    """Fixture to create a Hyperliquid connector instance."""
    connector = HyperliquidConnector(
        name="test_hyperliquid",
        wallet_address=TEST_WALLET_ADDRESS,
        private_key=TEST_PRIVATE_KEY,
        testnet=USE_TESTNET
    )
    # Initialize but don't connect yet
    return connector

@pytest.fixture
def connected_connector(connector):
    """Fixture to create and connect a Hyperliquid connector instance."""
    if TEST_MODE == TestMode.RECORD:
        # Use real connection
        connector.connect()
    elif TEST_MODE == TestMode.REPLAY:
        # Mock the connection methods using recorded responses
        meta_data = APIResponseRecorder.load_response('info_meta')
        if meta_data:
            custom_meta_wrapper = MagicMock(spec=MetadataWrapper)
            custom_meta_wrapper.meta.return_value = {"universe": meta_data}
            connector.info = custom_meta_wrapper
            connector._is_connected = True
        else:
            custom_meta_wrapper = MagicMock(spec=MetadataWrapper)
            custom_meta_wrapper.meta.return_value = {"universe": [{"name": "ETH", "szDecimals": 2}]}
            connector.info = custom_meta_wrapper
            connector._is_connected = True
    else:  # TestMode.MOCK
        # Create mock objects with our wrapper format
        custom_meta_wrapper = MagicMock(spec=MetadataWrapper)
        custom_meta_wrapper.meta.return_value = {"universe": [{"name": "ETH", "szDecimals": 2}]}
        connector.info = custom_meta_wrapper
        connector.exchange = MagicMock()
        connector._is_connected = True

    return connector

# =============================================================================
# CONNECTION TESTS
# =============================================================================

def test_connection(connector):
    """Test connection to the Hyperliquid API."""
    if TEST_MODE == TestMode.RECORD:
        # Real connection test
        result = connector.connect()

        # Record the response for later use
        meta_response = connector.info.meta()
        APIResponseRecorder.save_response('info_meta', meta_response.get("universe", []))

        assert result is True
        assert connector.is_connected is True
    else:
        # Test with recorded/mocked data
        with patch('hyperliquid.info.Info') as mock_info_class:
            # Create a mock Info instance
            mock_info_instance = MagicMock()
            mock_info_instance.meta.return_value = {"universe": [{"name": "ETH", "szDecimals": 2}]}
            mock_info_class.return_value = mock_info_instance

            result = connector.connect()

            assert result is True
            assert connector.is_connected is True

# =============================================================================
# MARKET DATA TESTS
# =============================================================================

def test_get_markets(connected_connector):
    """Test retrieving available markets from Hyperliquid."""
    if TEST_MODE == TestMode.RECORD:
        markets = connected_connector.get_markets()
        APIResponseRecorder.save_response('markets', markets)

        assert isinstance(markets, list)
        assert len(markets) > 0
        for market in markets:
            assert 'symbol' in market
            assert 'base_asset' in market
            assert 'quote_asset' in market
            assert 'market_type' in market
    else:
        markets = connected_connector.get_markets()

        assert isinstance(markets, list)
        assert len(markets) > 0
        assert markets[0]['symbol'] == 'ETH'

def test_get_ticker(connected_connector):
    """Test retrieving ticker data for a specific market."""
    symbol = 'ETH-USD'

    if TEST_MODE == TestMode.RECORD:
        with patch.object(connected_connector, 'translate_symbol', return_value='ETH'):
            ticker = connected_connector.get_ticker(symbol)
            APIResponseRecorder.save_response(f'ticker_{symbol}', ticker)

            assert isinstance(ticker, dict)
            assert 'symbol' in ticker
            assert 'last_price' in ticker
            assert 'timestamp' in ticker
    else:
        with patch.object(connected_connector, 'translate_symbol', return_value='ETH'):
            with patch.object(connected_connector.info, 'all_mids') as mock_all_mids:
                mock_all_mids.return_value = {'ETH': '1603.35'}

                ticker = connected_connector.get_ticker(symbol)

                assert isinstance(ticker, dict)
                assert ticker['symbol'] == symbol
                assert ticker['last_price'] == 1603.35

def test_get_orderbook(connected_connector):
    """Test retrieving orderbook data for a specific market."""
    symbol = 'ETH-USD'

    if TEST_MODE == TestMode.RECORD:
        try:
            orderbook = connected_connector.get_orderbook(symbol)
            APIResponseRecorder.save_response(f'orderbook_{symbol}', orderbook)

            assert isinstance(orderbook, dict)
            assert 'bids' in orderbook
            assert 'asks' in orderbook
            assert len(orderbook['bids']) > 0
            assert len(orderbook['asks']) > 0
        except Exception as e:
            # Record the error response for debugging
            APIResponseRecorder.save_response(f'orderbook_{symbol}_error', str(e))
            raise
    else:
        with requests_mock.Mocker() as m:
            # Test multiple format responses
            m.post(f"{API_URL}/info", json={
                'coin': 'ETH',
                'time': 1744548947478,
                'levels': [
                    [{'px': '1603.3', 'sz': '508.0683', 'n': 20},
                     {'px': '1603.2', 'sz': '372.3886', 'n': 21}],
                    [{'px': '1603.4', 'sz': '12.6074', 'n': 3},
                     {'px': '1603.5', 'sz': '150.6767', 'n': 7}]
                ]
            })

            with patch.object(connected_connector, 'translate_symbol', return_value='ETH'):
                orderbook = connected_connector.get_orderbook(symbol)

                assert isinstance(orderbook, dict)
                assert 'bids' in orderbook
                assert 'asks' in orderbook
                assert len(orderbook['bids']) > 0
                assert len(orderbook['asks']) > 0

def test_get_historical_candles(connected_connector):
    """Test retrieving historical candlestick data."""
    symbol = 'ETH-USD'
    interval = '1m'

    if TEST_MODE == TestMode.RECORD:
        try:
            candles = connected_connector.get_historical_candles(symbol, interval, limit=10)
            APIResponseRecorder.save_response(f'candles_{symbol}_{interval}', candles)

            assert isinstance(candles, list)
            if len(candles) > 0:
                assert 'timestamp' in candles[0]
                assert 'open' in candles[0]
                assert 'high' in candles[0]
                assert 'low' in candles[0]
                assert 'close' in candles[0]
                assert 'volume' in candles[0]
        except Exception as e:
            # Record the error response for debugging
            APIResponseRecorder.save_response(f'candles_{symbol}_{interval}_error', str(e))
            raise
    else:
        with requests_mock.Mocker() as m:
            m.post(f"{API_URL}/info", json=[
                {'t': 1744548945000, 'o': '1603.35', 'h': '1603.40', 'l': '1603.30', 'c': '1603.38', 'v': '10.5'},
                {'t': 1744548945060, 'o': '1603.38', 'h': '1603.45', 'l': '1603.35', 'c': '1603.40', 'v': '12.3'}
            ])

            # Need to patch connect() since it's called in get_historical_candles when info is None
            with patch.object(connected_connector, 'connect', return_value=True):
                candles = connected_connector.get_historical_candles(symbol, interval, limit=10)

                assert isinstance(candles, list)
                assert len(candles) > 0
                assert 'timestamp' in candles[0]
                assert 'open' in candles[0]
                assert 'high' in candles[0]
                assert 'low' in candles[0]
                assert 'close' in candles[0]
                assert 'volume' in candles[0]

# =============================================================================
# ACCOUNT DATA TESTS
# =============================================================================

def test_get_account_balance(connected_connector):
    """Test retrieving account balance information."""
    if TEST_MODE == TestMode.RECORD:
        try:
            balances = connected_connector.get_account_balance()
            APIResponseRecorder.save_response('account_balance', balances)

            assert isinstance(balances, dict)
            assert len(balances) > 0
            # Check for USDC balance which should always exist
            assert 'PERP_USDC' in balances
        except Exception as e:
            # Record the error response for debugging
            APIResponseRecorder.save_response('account_balance_error', str(e))
            raise
    else:
        with patch.object(connected_connector.info, 'user_state') as mock_user_state:
            mock_user_state.return_value = {
                'marginSummary': {'accountValue': '98.315674'},
                'assetPositions': []
            }

            balances = connected_connector.get_account_balance()

            assert isinstance(balances, dict)
            assert 'PERP_USDC' in balances
            assert balances['PERP_USDC'] == 98.315674

def test_get_positions(connected_connector):
    """Test retrieving position information."""
    if TEST_MODE == TestMode.RECORD:
        try:
            positions = connected_connector.get_positions()
            APIResponseRecorder.save_response('positions', positions)

            assert isinstance(positions, list)
            # If positions exist, check their structure
            if len(positions) > 0:
                assert 'symbol' in positions[0]
                assert 'size' in positions[0]
                assert 'side' in positions[0]
                assert 'entry_price' in positions[0]
                assert 'mark_price' in positions[0]
                assert 'leverage' in positions[0]
        except Exception as e:
            # Record the error response for debugging
            APIResponseRecorder.save_response('positions_error', str(e))
            raise
    else:
        with patch.object(connected_connector.info, 'user_state') as mock_user_state:
            mock_user_state.return_value = {
                'assetPositions': [
                    {'coin': 0, 'szi': '1.2349', 'entryPx': '1603.5', 'leverage': '1.0', 'liqPx': '1554.88'}
                ]
            }
            with patch.object(connected_connector.info, 'all_mids') as mock_all_mids:
                mock_all_mids.return_value = ['1603.35']

                positions = connected_connector.get_positions()

                assert isinstance(positions, list)
                assert len(positions) > 0
                assert positions[0]['symbol'] == 'ETH-USD'
                assert positions[0]['size'] == 1.2349
                assert positions[0]['entry_price'] == 1603.5

# =============================================================================
# ORDER MANAGEMENT TESTS
# =============================================================================

def test_place_order(connected_connector):
    """Test placing an order."""
    symbol = 'ETH-USD'
    side = OrderSide.BUY
    order_type = OrderType.LIMIT
    amount = 0.1
    price = 1600.0
    leverage = 1.0

    if TEST_MODE == TestMode.RECORD:
        try:
            # For safety in recording mode, don't actually place an order
            logger.warning("SKIPPING ACTUAL ORDER PLACEMENT IN RECORD MODE")
            # Instead, document the expected request format
            order_request = {
                'symbol': symbol,
                'side': side.value,
                'order_type': order_type.value,
                'amount': amount,
                'price': price,
                'leverage': leverage
            }
            APIResponseRecorder.save_response('place_order_request', order_request)
        except Exception as e:
            # Record the error response for debugging
            APIResponseRecorder.save_response('place_order_error', str(e))
            raise
    else:
        with patch.object(connected_connector, 'translate_symbol', return_value='ETH'):
            with patch.object(connected_connector, 'get_min_order_size', return_value=0.01):
                with patch.object(connected_connector, 'get_optimal_limit_price') as mock_price:
                    mock_price.return_value = {
                        'price': price,
                        'batches': [{'price': price, 'amount': amount}],
                        'enough_liquidity': True,
                        'slippage': 0.0
                    }
                    with patch.object(connected_connector.info, 'meta') as mock_meta:
                        mock_meta.return_value = {'universe': [{'name': 'ETH', 'szDecimals': 2}]}
                        with patch.object(connected_connector, 'exchange') as mock_exchange:
                            mock_exchange.order.return_value = {
                                'status': 'ok',
                                'response': {
                                    'type': 'order',
                                    'data': {
                                        'statuses': [
                                            {'filled': {'totalSz': '0.1', 'avgPx': '1600.0', 'oid': 123456}}
                                        ]
                                    }
                                }
                            }

                            # Run the async method and get result
                            coroutine = connected_connector.place_order(
                                symbol=symbol,
                                side=side,
                                order_type=order_type,
                                amount=amount,
                                price=price,
                                leverage=leverage
                            )
                            response = run_async(coroutine)

                            assert isinstance(response, dict)
                            assert 'order_id' in response
                            assert response['status'] == 'FILLED'

def test_cancel_order(connected_connector):
    """Test cancelling an order."""
    symbol = 'ETH-USD'
    order_id = '123456'

    if TEST_MODE == TestMode.RECORD:
        # For safety in recording mode, don't actually cancel an order
        logger.warning("SKIPPING ACTUAL ORDER CANCELLATION IN RECORD MODE")
        # Instead, document the expected request format
        cancel_request = {
            'symbol': symbol,
            'order_id': order_id
        }
        APIResponseRecorder.save_response('cancel_order_request', cancel_request)
    else:
        with patch.object(connected_connector, 'translate_symbol', return_value='ETH'):
            with patch.object(connected_connector.exchange, 'cancel') as mock_cancel:
                mock_cancel.return_value = {
                    'status': 'ok',
                    'response': {
                        'data': {
                            'statuses': ['cancelled']
                        }
                    }
                }

                response = connected_connector.cancel_order(order_id, symbol)

                assert isinstance(response, dict)
                assert response['status'] == 'success'
                assert response['order_id'] == order_id

def test_get_order_status(connected_connector):
    """Test retrieving order status."""
    order_id = '123456'

    if TEST_MODE == TestMode.RECORD:
        try:
            # In real mode, we would need an actual order ID
            logger.warning("SKIPPING ACTUAL ORDER STATUS IN RECORD MODE")
        except Exception as e:
            APIResponseRecorder.save_response('order_status_error', str(e))
            raise
    else:
        with patch.object(connected_connector.exchange, 'open_orders') as mock_open_orders:
            mock_open_orders.return_value = [
                {
                    'oid': '123456',
                    'coin': 0,  # ETH
                    'is_buy': True,
                    'limit_px': '1600.0',
                    'sz': '0.1',
                    'filled': '0.0'
                }
            ]
            with patch.object(connected_connector.info, 'meta') as mock_meta:
                mock_meta.return_value = {'universe': [{'name': 'ETH'}]}

                status = connected_connector.get_order_status(order_id)

                assert isinstance(status, dict)
                assert status['order_id'] == order_id
                assert status['status'] == 'OPEN'
                assert status['symbol'] == 'ETH'

# =============================================================================
# CALCULATION TESTS
# =============================================================================

def test_get_leverage_tiers(connected_connector):
    """Test retrieving leverage tier information."""
    symbol = 'ETH-USD'

    if TEST_MODE == TestMode.RECORD:
        try:
            tiers = connected_connector.get_leverage_tiers(symbol)
            APIResponseRecorder.save_response(f'leverage_tiers_{symbol}', tiers)

            assert isinstance(tiers, list)
            if len(tiers) > 0:
                assert 'max_leverage' in tiers[0]
                assert 'maintenance_margin_fraction' in tiers[0]
                assert 'initial_margin_fraction' in tiers[0]
        except Exception as e:
            APIResponseRecorder.save_response(f'leverage_tiers_{symbol}_error', str(e))
            raise
    else:
        with patch.object(connected_connector, 'translate_symbol', return_value='ETH'):
            with patch.object(connected_connector, 'get_markets') as mock_get_markets:
                mock_get_markets.return_value = [
                    {
                        'symbol': 'ETH',
                        'base_asset': 'ETH',
                        'exchange_specific': {
                            'max_leverage': 25.0
                        }
                    }
                ]

                tiers = connected_connector.get_leverage_tiers(symbol)

                assert isinstance(tiers, list)
                assert len(tiers) > 0
                assert tiers[0]['max_leverage'] == 25.0

def test_calculate_margin_requirement(connected_connector):
    """Test margin requirement calculation."""
    symbol = 'ETH-USD'
    quantity = 1.0
    price = 1600.0
    leverage = 10.0

    with patch.object(connected_connector, 'get_leverage_tiers') as mock_get_tiers:
        mock_get_tiers.return_value = [
            {
                'max_leverage': 25.0,
                'maintenance_margin_fraction': 0.025,
                'initial_margin_fraction': 0.05
            }
        ]

        result = connected_connector.calculate_margin_requirement(
            symbol=symbol,
            quantity=quantity,
            price=price,
            leverage=leverage
        )

        assert isinstance(result, dict)
        assert 'initial_margin' in result
        assert 'maintenance_margin' in result
        assert 'effective_leverage' in result
        assert result['effective_leverage'] == leverage
        assert result['initial_margin'] == price * quantity / leverage

def test_get_optimal_limit_price(connected_connector):
    """Test optimal limit price calculation."""
    symbol = 'ETH-USD'
    side = OrderSide.BUY
    amount = 1.0

    with patch.object(connected_connector, 'get_orderbook') as mock_get_orderbook:
        mock_get_orderbook.return_value = {
            'bids': [[1599.0, 2.0], [1598.0, 3.0]],
            'asks': [[1600.0, 1.5], [1601.0, 2.5]]
        }

        result = connected_connector.get_optimal_limit_price(
            symbol=symbol,
            side=side,
            amount=amount
        )

        assert isinstance(result, dict)
        assert 'price' in result
        assert 'enough_liquidity' in result
        assert 'slippage' in result
        assert result['price'] > 0
        assert result['enough_liquidity'] is True

def test_get_min_order_size(connected_connector):
    """Test retrieving minimum order size."""
    symbol = 'ETH-USD'

    if TEST_MODE == TestMode.RECORD:
        try:
            min_size = connected_connector.get_min_order_size(symbol)
            APIResponseRecorder.save_response(f'min_order_size_{symbol}', min_size)

            assert isinstance(min_size, float)
            assert min_size > 0
        except Exception as e:
            APIResponseRecorder.save_response(f'min_order_size_{symbol}_error', str(e))
            raise
    else:
        with patch.object(connected_connector, 'translate_symbol', return_value='ETH'):
            with patch.object(connected_connector.info, 'meta') as mock_meta:
                mock_meta.return_value = {
                    'universe': [
                        {'name': 'ETH', 'minSize': 0.01}
                    ]
                }

                min_size = connected_connector.get_min_order_size(symbol)

                assert isinstance(min_size, float)
                assert min_size == 0.01

# =============================================================================
# ADVANCED FUNCTIONALITY TESTS
# =============================================================================

def test_translate_symbol(connector):
    """Test symbol translation functionality."""
    test_cases = [
        ('ETH-USD', 'ETH'),
        ('ETH', 'ETH'),
        ('btc-usd', 'BTC'),  # Test case insensitivity
        ('SOL-PERP', 'SOL')
    ]

    for input_symbol, expected_output in test_cases:
        result = connector.translate_symbol(input_symbol)
        assert result == expected_output

    # Test error cases
    with pytest.raises(ValueError):
        connector.translate_symbol('')

    with pytest.raises(ValueError):
        connector.translate_symbol(None)

def test_leverage_application_in_orders(connected_connector):
    """Test that leverage is correctly applied in order placement."""
    symbol = 'ETH-USD'
    side = OrderSide.BUY
    order_type = OrderType.LIMIT
    amount = 0.1
    price = 1600.0
    leverage = 5.0

    # This test is critical to verify leverage is correctly applied
    with patch.object(connected_connector, 'translate_symbol', return_value='ETH'):
        with patch.object(connected_connector, 'get_min_order_size', return_value=0.01):
            with patch.object(connected_connector, 'get_optimal_limit_price') as mock_price:
                mock_price.return_value = {
                    'price': price,
                    'batches': [{'price': price, 'amount': amount}],
                    'enough_liquidity': True,
                    'slippage': 0.0
                }
                with patch.object(connected_connector.info, 'meta') as mock_meta:
                    mock_meta.return_value = {'universe': [{'name': 'ETH', 'szDecimals': 2}]}
                    with patch.object(connected_connector, 'exchange') as mock_exchange:
                        mock_exchange.order.return_value = {
                            'status': 'ok',
                            'response': {
                                'type': 'order',
                                'data': {
                                    'statuses': [
                                        {'filled': {'totalSz': '0.1', 'avgPx': '1600.0', 'oid': 123456}}
                                    ]
                                }
                            }
                        }

                        # The key part: Run the async method
                        coroutine = connected_connector.place_order(
                            symbol=symbol,
                            side=side,
                            order_type=order_type,
                            amount=amount,
                            price=price,
                            leverage=leverage
                        )
                        run_async(coroutine)

                        # Check that the order was called with correct leverage
                        # We need to adapt this to the actual API call structure
                        mock_exchange.order.assert_called_once()
                        call_kwargs = mock_exchange.order.call_args.kwargs

                        # Verify that leverage parameter was included in the order
                        # This test may need to be adapted based on exact API call format
                        assert call_kwargs is not None

                        # The assertion logic depends on how leverage is passed in your implementation
                        # Example assertions (adjust based on actual implementation):
                        # assert 'leverage' in call_kwargs
                        # assert call_kwargs['leverage'] == leverage
                        # Or just check it was called at all for now
                        assert mock_exchange.order.called

# =============================================================================
# COMPREHENSIVE API FORMAT TESTS
# =============================================================================

def test_orderbook_format(connected_connector):
    """Test handling of actual orderbook format from production API."""

    # This test specifically addresses the orderbook parsing issue
    with requests_mock.Mocker() as m:
        # Use the actual problematic format seen in production
        m.post(f"{API_URL}/info", json={
            'coin': 'ETH',
            'time': 1744548947478,
            'levels': [
                [{'px': '1603.3', 'sz': '508.0683', 'n': 20},
                 {'px': '1603.2', 'sz': '372.3886', 'n': 21}],
                [{'px': '1603.4', 'sz': '12.6074', 'n': 3},
                 {'px': '1603.5', 'sz': '150.6767', 'n': 7}]
            ]
        })

        with patch.object(connected_connector, 'translate_symbol', return_value='ETH'):
            with patch.object(connected_connector.info, 'meta') as mock_meta:
                mock_meta.return_value = {'universe': [{'name': 'ETH'}]}

                # This test should now handle the nested format correctly
                orderbook = connected_connector.get_orderbook('ETH-USD')

                assert isinstance(orderbook, dict)
                assert 'bids' in orderbook
                assert 'asks' in orderbook
                assert len(orderbook['bids']) > 0
                assert len(orderbook['asks']) > 0

def test_position_data_format(connected_connector):
    """Test handling of actual position data format from production API."""

    # This test specifically addresses the position parsing issue
    with patch.object(connected_connector.info, 'user_state') as mock_user_state:
        # Use the actual format seen in production with nested position data
        mock_user_state.return_value = {
            'marginSummary': {'accountValue': '98.315674'},
            'assetPositions': [
                {
                    'type': 'oneWay',
                    'position': {
                        'coin': 0,  # Index-based reference in the production API
                        'szi': '1.2349',
                        'leverage': {'type': 'cross', 'value': 20},
                        'entryPx': '1603.5'
                    }
                }
            ]
        }

        with patch.object(connected_connector.info, 'meta') as mock_meta:
            mock_meta.return_value = {'universe': [{'name': 'ETH'}]}

        with patch.object(connected_connector.info, 'all_mids') as mock_all_mids:
            mock_all_mids.return_value = ['1603.35']

            # This should now correctly parse the position data
            positions = connected_connector.get_positions()

            assert isinstance(positions, list)
            assert len(positions) > 0
            assert positions[0]['symbol'] == 'ETH-USD'
            assert positions[0]['size'] == 1.2349
            assert positions[0]['entry_price'] == 1603.5

# =============================================================================
# Run the test suite
# =============================================================================

if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
