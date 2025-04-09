import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Import Kraken API client libraries
import krakenex
import requests
# Import the BaseConnector interface
from connectors.base_connector import (BaseConnector, MarketType, OrderSide,
                                       OrderStatus, OrderType, TimeInForce)
# Import decorators
from metrics.decorators import track_api_latency, update_rate_limit
from pykrakenapi import KrakenAPI
from utils.logging_setup import (setup_connector_balance_logger,
                                 setup_connector_markets_logger,
                                 setup_connector_orders_logger)

# Get the main logger
logger = logging.getLogger(__name__)

class KrakenConnector(BaseConnector):
    """
    Connector for Kraken Exchange using their official Python client.

    This class implements the BaseConnector interface for Kraken.
    """

    def __init__(
        self,
        name: str = "kraken",
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
        market_types: Optional[List[MarketType]] = None,
    ):
        """
        Initialize the Kraken connector.

        Args:
            name: Custom name for this connector instance
            api_key: Kraken API key
            api_secret: Kraken API secret
            testnet: Whether to use testnet (demo.kraken.com)
            market_types: List of market types this connector supports
                          (defaults to [SPOT, PERPETUAL] for Kraken)
        """
        # Default market types for Kraken if none provided
        if market_types is None:
            market_types = [MarketType.SPOT, MarketType.PERPETUAL]
        elif not isinstance(market_types, list):
            market_types = [market_types]

        # Call the parent class constructor
        super().__init__(name=name, exchange_type="kraken", market_types=market_types)

        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self.k = None  # Will hold the KrakenAPI wrapper

        # Set API URLs based on testnet flag
        self.api_url = "https://api.kraken.com/0" if not testnet else "https://api.demo.kraken.com/0"
        self.futures_api_url = "https://futures.kraken.com/derivatives/api/v3" if not testnet else "https://demo-futures.kraken.com/derivatives/api/v3"

        # Log the supported market types
        logger.info(
            f"Initialized KrakenConnector with market types: {[mt.value for mt in self.market_types]}"
        )

        # Map of Kraken asset pairs to our standard symbols
        # Note: Kraken uses XBT instead of BTC and adds special prefixes to asset pairs
        self.symbol_map = {
            "ETH": "XETHZUSD",
            "BTC": "XXBTZUSD",
            "SOL": "SOLUSD",
            "USDC": "USDCUSD",
            "USDT": "USDTZUSD",
            # Add more mappings as needed
        }

        # Futures symbol mapping is different
        self.futures_symbol_map = {
            "ETH": "PI_ETHUSD",  # Perpetual
            "BTC": "PI_XBTUSD",  # Perpetual
            "SOL": "PI_SOLUSD",  # Perpetual
            # Add more as needed
        }

        # Reverse mapping
        self.reverse_symbol_map = {}
        for k, v in self.symbol_map.items():
            self.reverse_symbol_map[v] = k

        # Reverse futures mapping
        self.reverse_futures_symbol_map = {}
        for k, v in self.futures_symbol_map.items():
            self.reverse_futures_symbol_map[v] = k

        # Cache for market data to reduce API calls
        self._markets_cache = None
        self._futures_markets_cache = None
        self._last_markets_update = 0
        self._markets_cache_ttl = 3600  # 1 hour in seconds

    def _get_product_id(self, symbol: str, market_type: MarketType = MarketType.SPOT) -> str:
        """Convert our standard symbol to Kraken product ID."""
        if market_type == MarketType.SPOT:
            if symbol in self.symbol_map:
                return self.symbol_map[symbol]
            # Handle pairs not in the map
            return symbol
        else:
            # For futures/perpetual
            if symbol in self.futures_symbol_map:
                return self.futures_symbol_map[symbol]
            # Handle pairs not in the map
            return symbol

    def _get_symbol(self, product_id: str, market_type: MarketType = MarketType.SPOT) -> str:
        """Convert Kraken product ID to our standard symbol."""
        if market_type == MarketType.SPOT:
            return self.reverse_symbol_map.get(product_id, product_id)
        else:
            return self.reverse_futures_symbol_map.get(product_id, product_id)

    @track_api_latency(exchange="kraken", endpoint="connect")
    def connect(self) -> bool:
        """
        Establish connection to the Kraken API.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        # If already connected, return True without reconnecting
        if self.is_connected and self.client is not None:
            logger.debug("Already connected to Kraken API, reusing existing connection")
            return True

        try:
            logger.info("=== Starting Kraken API connection ===")
            logger.info(f"API URL: {self.api_url}")
            logger.info(f"Using testnet: {self.testnet}")

            # Log API key details before initialization
            logger.info(f"API Key exists: {bool(self.api_key)}")
            logger.info(f"API Secret exists: {bool(self.api_secret)}")

            # Initialize the Kraken API client
            api = krakenex.API(key=self.api_key, secret=self.api_secret)
            logger.info("Successfully created krakenex API instance")

            # Initialize pykrakenapi wrapper for easier usage
            self.k = KrakenAPI(api)
            self.client = api
            logger.info("Successfully initialized pykrakenapi wrapper")

            # Test connection with a simple public request
            try:
                logger.info("Testing connection with Time endpoint...")
                # Try to get server time - a simple unauthenticated endpoint
                response = self.client.query_public('Time')
                logger.info(f"Time endpoint response: {response}")

                if 'error' in response and response['error']:
                    logger.error(f"Failed to connect to Kraken API: {response['error']}")
                    self.client = None
                    self.k = None
                    self._is_connected = False
                    return False

                # Try a simple private endpoint to test authentication
                logger.info("Testing private endpoint authentication...")
                try:
                    account_response = self.client.query_private('GetWebSocketsToken')
                    logger.info("Successfully authenticated with private endpoint")
                    logger.debug(f"WebSocket token response: {account_response}")
                except Exception as auth_error:
                    logger.error(f"Failed to authenticate with private endpoint: {auth_error}")
                    if hasattr(auth_error, 'response'):
                        logger.error(f"Response status: {auth_error.response.status_code}")
                        logger.error(f"Response body: {auth_error.response.text}")

                self._is_connected = True
                logger.info("Successfully connected to Kraken API")
                return True

            except Exception as connection_error:
                logger.error(f"Failed to connect to Kraken API during test request: {connection_error}")
                self.client = None
                self.k = None
                self._is_connected = False
                return False

        except Exception as e:
            logger.error(f"Failed to create Kraken API client: {e}")
            self.client = None
            self.k = None
            self._is_connected = False
            return False

    @track_api_latency(exchange="kraken", endpoint="disconnect")
    def disconnect(self) -> bool:
        """
        Disconnect from the Kraken API.

        Returns:
            bool: True if successfully disconnected, False otherwise
        """
        try:
            # Clean up resources if needed
            self.client = None
            self.k = None
            self._is_connected = False
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Kraken API: {e}")
            return False

    def _create_futures_signature(self, endpoint: str, nonce: str, params: Dict) -> str:
        """
        Create signature for futures API authentication.

        Args:
            endpoint: API endpoint
            nonce: Unique nonce value
            params: Request parameters

        Returns:
            str: Base64-encoded signature
        """
        try:
            import base64
            import hashlib
            import hmac
            import urllib.parse

            # Convert params to query string
            if params:
                query_string = urllib.parse.urlencode(params)
            else:
                query_string = ""

            # Create message string
            # Format: nonce + endpoint + query_string
            message = nonce + endpoint
            if query_string:
                message += "?" + query_string

            # Create signature using HMAC-SHA512
            signature = hmac.new(
                base64.b64decode(self.api_secret),
                message.encode('utf-8'),
                hashlib.sha512
            )

            # Return base64 encoded signature
            return base64.b64encode(signature.digest()).decode('utf-8')

        except Exception as e:
            logger.error(f"Error creating futures signature: {e}")
            return ""

    def _make_futures_request(self, method: str, endpoint: str, params: Dict = None, auth: bool = False) -> Dict:
        """Make a request to the Kraken Futures API"""
        if params is None:
            params = {}

        url = f"{self.futures_api_url}/{endpoint}"

        try:
            headers = {
                'Content-Type': 'application/json',
            }

            if auth:
                if not self.api_key or not self.api_secret:
                    return {'error': 'API key and secret required for authenticated endpoints'}

                # Add authentication headers
                nonce = str(int(time.time() * 1000))
                headers.update({
                    'APIKey': self.api_key,
                    'Nonce': nonce,
                })

                # Create signature
                signature = self._create_futures_signature(endpoint, nonce, params)
                if not signature:
                    return {'error': 'Failed to create signature'}

                headers['Signature'] = signature

            # Make the request
            if method.upper() == 'GET':
                response = requests.get(url, params=params, headers=headers)
            elif method.upper() == 'POST':
                response = requests.post(url, json=params, headers=headers)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return {'error': f"Unsupported HTTP method: {method}"}

            # Handle response
            if response.status_code == 200:
                result = response.json()
                error = self._handle_error_response(result)
                if error:
                    logger.error(f"Futures API error: {error}")
                    return {'error': error}
                return result
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Futures API request failed: {error_msg}")
                return {'error': error_msg}

        except Exception as e:
            logger.error(f"Error making futures API request: {e}")
            return {'error': str(e)}

    def _handle_error_response(self, response: Dict) -> Optional[str]:
        """
        Handle Kraken API error responses.
        Returns error message if there's an error, None otherwise.
        """
        if not isinstance(response, dict):
            return "Invalid response format"

        if 'error' in response:
            errors = response['error']
            if isinstance(errors, list) and errors:
                # Join multiple errors if present
                return "; ".join(str(err) for err in errors)
            elif isinstance(errors, str):
                return errors
            elif isinstance(errors, dict):
                return json.dumps(errors)

        return None

    @track_api_latency(exchange="kraken", endpoint="get_markets")
    def get_markets(self) -> List[Dict[str, Any]]:
        """
        Get available markets from Kraken.

        Returns:
            List of market details
        """
        if not self._is_connected:
            self.connect()

        try:
            markets = []

            # Get spot markets
            if MarketType.SPOT in self.market_types:
                try:
                    # Use the Asset Pairs endpoint to get trading pairs info
                    response = self.client.query_public('AssetPairs')

                    if 'error' in response and response['error']:
                        logger.error(f"Error fetching spot markets: {response['error']}")
                    else:
                        spot_pairs = response.get('result', {})

                        # Process each pair
                        for pair_name, pair_data in spot_pairs.items():
                            # Extract relevant information
                            base_asset = pair_data.get('base', '')  # Keep original base asset (e.g., XXBT)
                            quote_asset = pair_data.get('quote', '').replace('Z', '')  # Remove Z prefix from quote

                            # Skip if base or quote is empty
                            if not base_asset or not quote_asset:
                                continue

                            # For symbol field, convert XXBT to BTC
                            symbol = base_asset
                            if base_asset == 'XXBT':
                                symbol = 'BTC'
                            else:
                                symbol = base_asset.replace('X', '').replace('Z', '')

                            market_info = {
                                'symbol': symbol,  # Use BTC in symbol
                                'base_asset': base_asset,  # Keep original Kraken base asset (XXBT)
                                'quote_asset': quote_asset,
                                'price_precision': pair_data.get('pair_decimals', 8),
                                'min_size': float(pair_data.get('lot_decimals', 0.0001)),
                                'tick_size': float(pair_data.get('tick_size', 0.00001)),
                                'maker_fee': float(pair_data.get('fees_maker', [[0, 0.0016]])[0][1]) if 'fees_maker' in pair_data else 0.0016,
                                'taker_fee': float(pair_data.get('fees', [[0, 0.0026]])[0][1]) if 'fees' in pair_data else 0.0026,
                                'market_type': MarketType.SPOT.value,
                                'active': True,
                                'kraken_symbol': pair_name,  # Store Kraken's internal symbol
                            }

                            markets.append(market_info)
                except Exception as spot_err:
                    logger.error(f"Error processing spot markets: {spot_err}")

            # Get futures/perpetual markets
            if MarketType.PERPETUAL in self.market_types or MarketType.FUTURES in self.market_types:
                try:
                    # Request instruments from the futures API
                    response = self._make_futures_request('GET', 'instruments')

                    if 'error' in response and response['error']:
                        logger.error(f"Error fetching futures markets: {response['error']}")
                    else:
                        futures_instruments = response.get('instruments', [])

                        for instrument in futures_instruments:
                            # Determine market type
                            is_perpetual = instrument.get('isTradeable', False) and 'perpetual' in instrument.get('symbol', '').lower()
                            market_type = MarketType.PERPETUAL.value if is_perpetual else MarketType.FUTURES.value

                            # Skip if it's not a type we support
                            if (market_type == MarketType.PERPETUAL.value and MarketType.PERPETUAL not in self.market_types) or \
                               (market_type == MarketType.FUTURES.value and MarketType.FUTURES not in self.market_types):
                                continue

                            # Extract symbol components
                            symbol = instrument.get('symbol', '')
                            base_asset = instrument.get('underlying', {}).get('symbol', '')  # Keep XBT for futures
                            quote_asset = 'USD'  # Most Kraken futures are quoted in USD

                            market_info = {
                                'symbol': base_asset.replace('XBT', 'BTC'),  # Use BTC in symbol
                                'base_asset': base_asset,  # Keep XBT in base_asset
                                'quote_asset': quote_asset,
                                'price_precision': int(instrument.get('tickSize', 0.01) * 100),
                                'min_size': float(instrument.get('minTradeSize', 0.0001)),
                                'tick_size': float(instrument.get('tickSize', 0.01)),
                                'maker_fee': float(instrument.get('makerFee', 0.0002)),
                                'taker_fee': float(instrument.get('takerFee', 0.0005)),
                                'market_type': market_type,
                                'active': instrument.get('isTradeable', False),
                                'kraken_symbol': symbol,  # Store Kraken's internal symbol
                            }

                            markets.append(market_info)
                except Exception as futures_err:
                    logger.error(f"Error processing futures markets: {futures_err}")

            logger.info(f"Retrieved {len(markets)} markets from Kraken")
            return markets

        except Exception as e:
            logger.error(f"Error getting markets: {e}")
            return []

    @track_api_latency(exchange="kraken", endpoint="get_ticker")
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker information for a specific market.

        Args:
            symbol: The market symbol (e.g., 'BTC')

        Returns:
            Dict containing ticker information
        """
        try:
            if not self.client:
                self.connect()

            # Convert BTC to XBT for Kraken API if needed
            kraken_symbol = symbol
            if symbol == 'BTC':
                kraken_symbol = 'XBT'

            # Get the full Kraken symbol format
            if MarketType.SPOT in self.market_types:
                # For spot markets, we need to find the right kraken pair id
                if kraken_symbol == 'XBT':
                    product_id = 'XXBTZUSD'
                elif kraken_symbol == 'ETH':
                    product_id = 'XETHZUSD'
                else:
                    # Try to get the mapped symbol or use original with USD
                    product_id = self.symbol_map.get(symbol, f"{symbol}USD")

                # Query the ticker API
                response = self.client.query_public('Ticker', {'pair': product_id})

                if 'error' in response and response['error']:
                    self.markets_logger.error(f"Error fetching ticker: {response['error']}")
                    return {}

                ticker_data = response.get('result', {}).get(product_id)
                if not ticker_data:
                    self.markets_logger.warning(f"No ticker data found for {product_id}")
                    return {}

                # Format the response consistently
                return {
                    'symbol': product_id,
                    'bid': float(ticker_data['b'][0]),
                    'ask': float(ticker_data['a'][0]),
                    'last_price': float(ticker_data['c'][0]),  # Use 'last_price' for consistency
                    'volume': float(ticker_data['v'][1]),  # 24h volume
                    'high': float(ticker_data['h'][1]),    # 24h high
                    'low': float(ticker_data['l'][1]),     # 24h low
                }

            # For futures markets
            elif MarketType.PERPETUAL in self.market_types:
                # Use futures API ticker endpoint
                response = self._make_futures_request('GET', 'tickers')

                if not response or 'tickers' not in response:
                    return {}

                # Find the perpetual contract for the symbol (with XBT substitution)
                ticker = next(
                    (t for t in response['tickers'] if t.get('symbol', '').startswith(f"PI_{kraken_symbol}")),
                    None
                )

                if not ticker:
                    return {}

                return {
                    'symbol': ticker['symbol'],
                    'bid': float(ticker.get('bid', 0)),
                    'ask': float(ticker.get('ask', 0)),
                    'last_price': float(ticker.get('last', 0)),  # Use 'last_price' for consistency
                    'volume': float(ticker.get('vol24h', 0)),
                    'high': float(ticker.get('high24h', 0)),
                    'low': float(ticker.get('low24h', 0)),
                }

            return {}

        except Exception as e:
            self.markets_logger.error(f"Error getting ticker for {symbol}: {e}")
            return {}

    @track_api_latency(exchange="kraken", endpoint="get_orderbook")
    def get_orderbook(
        self, symbol: str, depth: int = 10
    ) -> Dict[str, List[List[float]]]:
        """
        Get the current order book for a specific market.

        Args:
            symbol: The market symbol (e.g., 'ETH')
            depth: Number of price levels to retrieve

        Returns:
            Dict with 'bids' and 'asks' lists of [price, size] pairs
        """
        try:
            if not self.client:
                self.connect()

            # Convert BTC to XBT for Kraken API if needed
            kraken_symbol = symbol
            if symbol == 'BTC':
                kraken_symbol = 'XBT'

            # For spot markets
            # Use the correct Kraken symbol format
            if kraken_symbol == 'XBT':
                product_id = 'XXBTZUSD'
            elif kraken_symbol == 'ETH':
                product_id = 'XETHZUSD'
            else:
                # Try to get from symbol map or construct
                product_id = self.symbol_map.get(symbol, f"{symbol}USD")

            # Get orderbook from spot API
            response = self.client.query_public('Depth', {'pair': product_id, 'count': depth})

            if 'error' in response and response['error']:
                logger.error(f"Error fetching spot orderbook for {symbol}: {response['error']}")
                return {"bids": [], "asks": []}

            # Process the spot orderbook
            orderbook_data = response.get('result', {}).get(product_id, {})
            bids = []
            asks = []

            # Process bids
            for bid in orderbook_data.get('bids', [])[:depth]:
                if len(bid) >= 2:
                    bids.append([float(bid[0]), float(bid[1])])

            # Process asks
            for ask in orderbook_data.get('asks', [])[:depth]:
                if len(ask) >= 2:
                    asks.append([float(ask[0]), float(ask[1])])

            # Return if we got data
            if bids or asks:
                return {"bids": bids, "asks": asks}

            # If we didn't get data from spot, try futures
            # This is a futures symbol, use futures API
            if MarketType.PERPETUAL in self.market_types:
                product_id = self._get_product_id(symbol, MarketType.PERPETUAL)
                response = self._make_futures_request('GET', f'orderbook', {'symbol': product_id})

                if 'error' in response:
                    logger.error(f"Error fetching futures orderbook for {symbol}: {response['error']}")
                    return {"bids": [], "asks": []}

                # Process the futures orderbook
                orderbook = response.get('orderBook', {})
                bids = []
                asks = []

                # Process bids
                for bid in orderbook.get('bids', [])[:depth]:
                    if len(bid) >= 2:
                        bids.append([float(bid[0]), float(bid[1])])

                # Process asks
                for ask in orderbook.get('asks', [])[:depth]:
                    if len(ask) >= 2:
                        asks.append([float(ask[0]), float(ask[1])])

                return {"bids": bids, "asks": asks}

            # Return empty orderbook if no data found
            return {"bids": [], "asks": []}

        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol} from Kraken: {e}")
            # Return empty order book on error
            return {"bids": [], "asks": []}

    @track_api_latency(exchange="kraken", endpoint="get_account_balance")
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balances for all assets.

        Returns:
            Dict[str, float]: Dictionary mapping asset names to their balances
        """
        try:
            if not self.client:
                self.balance_logger.error("Client not connected")
                return {}

            # Enhanced debug logging
            self.balance_logger.info("=== Starting get_account_balance request ===")
            self.balance_logger.info(f"API Key length: {len(self.api_key) if self.api_key else 0}")
            self.balance_logger.info(f"API Secret length: {len(self.api_secret) if self.api_secret else 0}")

            # Check if API key looks like base64
            import base64
            try:
                if self.api_key:
                    base64.b64decode(self.api_key)
                    self.balance_logger.info("API Key appears to be base64 encoded")
                if self.api_secret:
                    base64.b64decode(self.api_secret)
                    self.balance_logger.info("API Secret appears to be base64 encoded")
            except Exception as e:
                self.balance_logger.warning(f"API credentials are not base64 encoded: {e}")

            # Check if we have valid credentials
            if not self.api_key or not self.api_secret:
                self.balance_logger.error("Missing API credentials")
                return {}

            # Authenticate and get balances
            try:
                self.balance_logger.info("Attempting to query private Balance endpoint...")
                response = self.client.query_private('Balance')
                self.balance_logger.info(f"Raw API Response: {response}")
            except Exception as auth_error:
                self.balance_logger.error(f"Authentication error details: {str(auth_error)}")
                # Try to get more details about the error
                if hasattr(auth_error, 'response'):
                    self.balance_logger.error(f"Response status code: {auth_error.response.status_code}")
                    self.balance_logger.error(f"Response headers: {auth_error.response.headers}")
                    self.balance_logger.error(f"Response body: {auth_error.response.text}")
                return {}

            if 'error' in response and response['error']:
                self.balance_logger.error(f"Error fetching balances: {response['error']}")
                return {}

            balances = {}

            # Process the spot balances
            for asset, balance_str in response.get('result', {}).items():
                try:
                    # Handle special cases for BTC
                    if asset == 'XXBT':
                        clean_asset = 'BTC'
                    elif asset.startswith('X'):
                        clean_asset = asset[1:]  # Remove X prefix
                    else:
                        clean_asset = asset.replace('Z', '')  # Remove Z prefix

                    # Convert to float and add to our balances dict
                    balance = float(balance_str)
                    if balance > 0:
                        balances[clean_asset] = balance
                        self.balance_logger.debug(f"Found non-zero balance for {clean_asset}: {balance}")
                except (ValueError, TypeError) as e:
                    self.balance_logger.warning(f"Error converting balance for {asset}: {e}")

            # If we have futures enabled, get futures balances too
            if MarketType.PERPETUAL in self.market_types or MarketType.FUTURES in self.market_types:
                try:
                    # Different endpoint for futures API
                    futures_response = self._make_futures_request('GET', 'accounts', auth=True)

                    if 'accounts' in futures_response:
                        for account in futures_response.get('accounts', []):
                            currency = account.get('currency', '')
                            if currency == 'XBT':
                                currency = 'BTC'
                            balance = float(account.get('cash', 0.0))

                            # Combine with existing balance or add new
                            if currency in balances:
                                balances[currency] += balance
                            elif balance > 0:
                                balances[currency] = balance
                except Exception as futures_err:
                    self.balance_logger.error(f"Error fetching futures balances: {futures_err}")

            if not balances:
                self.balance_logger.warning("No non-zero balances found in Kraken account")
            else:
                self.balance_logger.info(f"Retrieved {len(balances)} non-zero balances from Kraken")

            return balances

        except Exception as e:
            self.balance_logger.error(f"Failed to get account balances: {e}")
            return {}

    @track_api_latency(exchange="kraken", endpoint="get_positions")
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current open positions.

        Returns:
            List[Dict[str, Any]]: List of open positions with details
        """
        try:
            if not self.client:
                logger.error("Client not connected")
                return []

            positions = []

            # For spot trading, get balances and treat non-zero balances as positions
            balances = self.get_account_balance()

            # Format spot balances as positions
            for asset, balance in balances.items():
                if balance > 0 and asset != 'USD':  # Skip USD balance
                    position = {
                        "symbol": asset,
                        "size": balance,
                        "entry_price": 0.0,  # Not available in the API
                        "mark_price": 0.0,  # Not available in the API
                        "pnl": 0.0,  # Not available in the API
                        "liquidation_price": 0.0,  # Not available in the API
                        "leverage": 1.0,  # Spot trading is always 1x
                        "collateral": balance,
                        "position_value": balance,
                    }
                    positions.append(position)

            # If we have futures enabled, get futures positions too
            if MarketType.PERPETUAL in self.market_types or MarketType.FUTURES in self.market_types:
                try:
                    # Get open positions from futures API
                    response = self.client.query_private('OpenPositions')

                    if 'error' in response and response['error']:
                        logger.error(f"Error fetching positions: {response['error']}")
                    else:
                        # Process each position
                        for pos_id, pos_data in response.get('result', {}).items():
                            try:
                                # Extract symbol
                                pair = pos_data.get('pair', '')
                                symbol = self._get_symbol(pair, MarketType.PERPETUAL)

                                # Determine position size and direction
                                size = float(pos_data.get('vol', '0'))
                                if pos_data.get('type') == 'sell':
                                    size = -size

                                # Get other position details
                                leverage = float(pos_data.get('leverage', '1'))
                                cost = float(pos_data.get('cost', '0'))
                                value = float(pos_data.get('value', '0'))

                                # Create position info
                                futures_position = {
                                    "symbol": symbol,
                                    "size": size,
                                    "entry_price": float(pos_data.get('avg_price', '0')),
                                    "mark_price": value / size if size != 0 else 0.0,
                                    "pnl": float(pos_data.get('net', '0')),
                                    "liquidation_price": 0.0,  # Not directly available
                                    "leverage": leverage,
                                    "collateral": cost,
                                    "position_value": value,
                                    "position_id": pos_id,
                                }

                                positions.append(futures_position)
                            except Exception as pos_err:
                                logger.error(f"Error processing position {pos_id}: {pos_err}")
                except Exception as futures_err:
                    logger.error(f"Error fetching futures positions: {futures_err}")

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    @track_api_latency(exchange="kraken", endpoint="place_order")
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: float,
        leverage: Optional[float] = None,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place a new order on Kraken.

        Args:
            symbol: The market symbol (e.g., 'BTC')
            side: Buy or sell
            order_type: Market or limit
            amount: The amount/size to trade
            leverage: Leverage multiplier (for margin/futures trading)
            price: Limit price (required for limit orders)

        Returns:
            Dict[str, Any]: Order details including order ID
        """
        try:
            if not self.client:
                self.connect()

            # Determine if this is a spot or futures order
            is_futures = symbol in self.futures_symbol_map

            # Get the product ID
            if is_futures:
                product_id = self._get_product_id(symbol, MarketType.PERPETUAL)
            else:
                product_id = self._get_product_id(symbol)

            # Convert our enum values to Kraken's expected values
            k_side = "buy" if side == OrderSide.BUY else "sell"

            # Log order details before placement
            self.orders_logger.info(
                f"Placing {order_type.value} {k_side} order for {amount} {symbol} "
                + (f"at price {price}" if price else "at market price")
            )

            # Handle futures orders
            if is_futures:
                return self._place_futures_order(
                    product_id, side, order_type, amount, leverage, price
                )

            # Handle spot orders
            else:
                return self._place_spot_order(
                    product_id, side, order_type, amount, leverage, price
                )

        except Exception as e:
            error_msg = f"Error placing order: {e}"
            logger.error(error_msg)
            self.orders_logger.error(error_msg)
            return {"error": error_msg}

    def _place_spot_order(
        self,
        product_id: str,
        side: OrderSide,
        order_type: OrderType,
        amount: float,
        leverage: Optional[float] = None,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place a spot order on Kraken"""
        try:
            # Prepare order parameters
            params = {
                'pair': product_id,
                'type': 'buy' if side == OrderSide.BUY else 'sell',
                'volume': str(amount),
            }

            # Set order type
            if order_type == OrderType.MARKET:
                params['ordertype'] = 'market'
            elif order_type == OrderType.LIMIT:
                if price is None:
                    return {"error": "Price is required for LIMIT orders"}
                params['ordertype'] = 'limit'
                params['price'] = str(price)

            # Add leverage if specified and > 1
            if leverage and leverage > 1:
                params['leverage'] = str(leverage)

            # Submit the order
            response = self.client.query_private('AddOrder', params)

            if 'error' in response and response['error']:
                error_msg = f"Error placing order: {response['error']}"
                self.orders_logger.error(error_msg)
                return {"error": error_msg}

            # Extract order ID from response
            result = response.get('result', {})
            order_ids = result.get('txid', [])

            if not order_ids:
                error_msg = "No order ID returned from Kraken"
                self.orders_logger.error(error_msg)
                return {"error": error_msg}

            # Use the first order ID (usually only one is returned)
            order_id = order_ids[0]

            # Return order details
            return {
                "order_id": order_id,
                "client_order_id": "",  # Kraken doesn't support client order IDs for spot
                "symbol": self._get_symbol(product_id),
                "side": side.value,
                "order_type": order_type.value,
                "price": price if price else 0.0,
                "amount": amount,
                "status": OrderStatus.OPEN.value,
                "timestamp": int(time.time() * 1000),
            }

        except Exception as e:
            error_msg = f"Error placing spot order: {e}"
            self.orders_logger.error(error_msg)
            return {"error": error_msg}

    def _place_futures_order(
        self,
        product_id: str,
        side: OrderSide,
        order_type: OrderType,
        amount: float,
        leverage: Optional[float] = None,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place a futures order on Kraken"""
        try:
            # Prepare order parameters
            params = {
                'symbol': product_id,
                'side': 'buy' if side == OrderSide.BUY else 'sell',
                'size': str(amount),
            }

            # Set order type and price
            if order_type == OrderType.MARKET:
                params['orderType'] = 'market'
            elif order_type == OrderType.LIMIT:
                if price is None:
                    return {"error": "Price is required for LIMIT orders"}
                params['orderType'] = 'limit'
                params['limitPrice'] = str(price)

            # Add leverage if specified
            if leverage is not None and leverage > 0:
                params['leverage'] = str(leverage)

            # Submit the order
            response = self._make_futures_request('POST', 'orders', params, auth=True)

            error = self._handle_error_response(response)
            if error:
                logger.error(f"Error placing futures order: {error}")
                return {"error": error}

            # Extract order details from response
            order_data = response.get('order', {})
            if not order_data:
                return {"error": "No order data in response"}

            # Map response to our standard format
            return {
                "order_id": order_data.get('orderId', ''),
                "client_order_id": order_data.get('cliOrdId', ''),
                "symbol": self._get_symbol(product_id, MarketType.PERPETUAL),
                "side": side.value,
                "order_type": order_type.value,
                "price": float(order_data.get('price', 0.0)),
                "amount": float(order_data.get('size', 0.0)),
                "status": self._map_futures_order_status(order_data.get('status', '')),
                "timestamp": int(time.time() * 1000),
            }

        except Exception as e:
            error_msg = f"Error placing futures order: {e}"
            logger.error(error_msg)
            return {"error": error_msg}

    def _map_futures_order_status(self, status: str) -> str:
        """Map Kraken futures order status to our standard OrderStatus enum values."""
        status_map = {
            'untriggered': OrderStatus.OPEN.value,
            'triggered': OrderStatus.OPEN.value,
            'filled': OrderStatus.FILLED.value,
            'cancelled': OrderStatus.CANCELED.value,
            'rejected': OrderStatus.REJECTED.value,
        }
        return status_map.get(status.lower(), OrderStatus.OPEN.value)

    def _map_order_status(self, kraken_status: str) -> str:
        """Map Kraken order status to our standard OrderStatus enum values."""
        status_map = {
            "open": OrderStatus.OPEN.value,
            "pending": OrderStatus.OPEN.value,
            "closed": OrderStatus.FILLED.value,
            "canceled": OrderStatus.CANCELED.value,
            "expired": OrderStatus.CANCELED.value,
        }
        return status_map.get(kraken_status.lower(), OrderStatus.OPEN.value)

    @track_api_latency(exchange="kraken", endpoint="cancel_order")
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: The ID of the order to cancel

        Returns:
            bool: True if successfully canceled, False otherwise
        """
        try:
            if not self.client:
                self.connect()

            # Log the cancel attempt
            self.orders_logger.info(f"Attempting to cancel order {order_id}")

            # Determine if this is a futures order
            is_futures = order_id.startswith("mock-futures-order-")

            if is_futures:
                # This is a placeholder for futures order cancellation
                self.orders_logger.info(f"Cancelling futures order {order_id}")
                return True

            # Cancel spot order
            response = self.client.query_private('CancelOrder', {'txid': order_id})

            if 'error' in response and response['error']:
                error_msg = f"Failed to cancel order {order_id}: {response['error']}"
                self.orders_logger.error(error_msg)
                return False

            # Check if the cancellation was successful
            result = response.get('result', {})
            count = result.get('count', 0)

            if count > 0:
                self.orders_logger.info(f"Successfully canceled order {order_id}")
                return True
            else:
                self.orders_logger.warning(f"Order {order_id} not found or already canceled")
                return False

        except Exception as e:
            error_msg = f"Error canceling order {order_id}: {e}"
            self.orders_logger.error(error_msg)
            logger.error(error_msg)
            return False

    @track_api_latency(exchange="kraken", endpoint="get_order_status")
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.

        Args:
            order_id: The ID of the order to check

        Returns:
            Dict[str, Any]: Order status and details
        """
        # This is essentially the same as get_order for Kraken
        return self.get_order(order_id)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get the details of an order from Kraken.

        Args:
            order_id: The ID of the order to retrieve

        Returns:
            Dict[str, Any]: Order details
        """
        try:
            if not self.client:
                self.connect()

            # Determine if this is a futures order
            is_futures = order_id.startswith("mock-futures-order-")

            if is_futures:
                # This is a placeholder for futures order details
                self.orders_logger.info(f"Getting futures order {order_id}")

                # Return mock order details
                return {
                    "order_id": order_id,
                    "client_order_id": "",
                    "symbol": "BTC",  # Placeholder
                    "side": "BUY",    # Placeholder
                    "order_type": "LIMIT",  # Placeholder
                    "price": 40000.0,  # Placeholder
                    "amount": 1.0,    # Placeholder
                    "status": OrderStatus.OPEN.value,
                    "timestamp": int(time.time() * 1000),
                }

            # Get spot order details
            response = self.client.query_private('QueryOrders', {'txid': order_id, 'trades': True})

            if 'error' in response and response['error']:
                error_msg = f"Error retrieving order {order_id}: {response['error']}"
                logger.error(error_msg)
                return {"error": error_msg}

            # Extract order details
            order_data = response.get('result', {}).get(order_id, {})

            if not order_data:
                error_msg = f"Order {order_id} not found"
                logger.error(error_msg)
                return {"error": error_msg}

            # Map status
            status = self._map_order_status(order_data.get('status', 'open'))

            # Get price and volume
            price = float(order_data.get('price', 0.0))
            volume = float(order_data.get('vol', 0.0))
            filled = float(order_data.get('vol_exec', 0.0))

            # Get pair and map to symbol
            pair = order_data.get('descr', {}).get('pair', '')
            symbol = self._get_symbol(pair)

            # Get side
            side = order_data.get('descr', {}).get('type', 'buy').upper()

            # Get order type
            order_type = order_data.get('descr', {}).get('ordertype', 'limit').upper()

            # Return order details
            return {
                "order_id": order_id,
                "client_order_id": "",  # Kraken doesn't support client order IDs for spot
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "price": price,
                "amount": volume,
                "filled": filled,
                "status": status,
                "timestamp": int(float(order_data.get('opentm', time.time())) * 1000),
            }

        except Exception as e:
            error_msg = f"Exception when retrieving order {order_id}: {e}"
            logger.error(error_msg)
            return {"error": error_msg}

    @track_api_latency(exchange="kraken", endpoint="close_position")
    def close_position(
        self, symbol: str, position_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Close an open position.

        Args:
            symbol: The market symbol (e.g., 'BTC')
            position_id: Optional position ID (required for futures)

        Returns:
            Dict[str, Any]: Result of the close operation
        """
        try:
            # Ensure client is connected
            if not self.client:
                self.orders_logger.info("Client not connected, attempting to connect...")
                if not self.connect():
                    return {
                        "success": False,
                        "message": "Failed to connect to exchange",
                    }

            # If position_id is provided, try to close that specific position
            if position_id:
                self.orders_logger.info(f"Attempting to close position {position_id}")

                # Determine if this is a futures position
                is_futures = position_id.startswith("mock-futures-order-")

                if is_futures:
                    # This is a placeholder for futures position closing
                    self.orders_logger.info(f"Closing futures position {position_id}")
                    return {
                        "success": True,
                        "message": f"Closed futures position {position_id}",
                    }

                # Try to cancel the order for spot positions
                try:
                    cancel_result = self.cancel_order(position_id)
                    return {
                        "success": cancel_result,
                        "message": f"Canceled order {position_id}" if cancel_result else f"Failed to cancel order {position_id}",
                    }
                except Exception as cancel_err:
                    error_msg = f"Error canceling order {position_id}: {cancel_err}"
                    self.orders_logger.error(error_msg)
                    return {
                        "success": False,
                        "message": error_msg,
                    }

            # For spot market, we need to get the current balance of the asset
            try:
                balances = self.get_account_balance()
                asset_balance = balances.get(symbol, 0.0)
            except Exception as balance_err:
                error_msg = f"Error retrieving balance for {symbol}: {balance_err}"
                self.orders_logger.error(error_msg)
                return {
                    "success": False,
                    "message": error_msg,
                }

            # If no balance, nothing to close
            if asset_balance <= 0:
                self.orders_logger.info(f"No {symbol} balance to close")
                return {
                    "success": False,
                    "message": f"No {symbol} balance to close",
                }

            # Place a market sell order for the entire balance
            self.orders_logger.info(f"Closing position by selling {asset_balance} {symbol}")
            try:
                order_result = self.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    amount=asset_balance,
                    leverage=1.0,  # Default for spot
                )
            except Exception as order_err:
                error_msg = f"Error placing market sell order for {symbol}: {order_err}"
                self.orders_logger.error(error_msg)
                return {
                    "success": False,
                    "message": error_msg,
                }

            if "error" in order_result:
                self.orders_logger.error(f"Failed to close {symbol} position: {order_result['error']}")
                return {
                    "success": False,
                    "message": f"Failed to close position: {order_result['error']}",
                }

            self.orders_logger.info(f"Successfully closed {symbol} position with order {order_result['order_id']}")
            return {
                "success": True,
                "message": f"Position closed with market sell order",
                "order": order_result,
            }

        except Exception as e:
            error_msg = f"Error closing position for {symbol}: {e}"
            self.orders_logger.error(error_msg)
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
            }

    def get_optimal_limit_price(
        self, symbol: str, side: OrderSide, amount: float
    ) -> Dict[str, Any]:
        """
        Calculate the optimal limit price for a given order based on the current order book.

        This analyzes the order book to find a price that balances execution probability
        with price efficiency. For large orders, it may suggest batching.

        Args:
            symbol: Market symbol (e.g., 'BTC')
            side: Buy or sell
            amount: Order size

        Returns:
            Dict with optimal price, batching suggestion, slippage estimation, etc.
        """
        try:
            if not self.client:
                self.connect()

            # Get order book for depth analysis
            order_book = self.get_orderbook(symbol, depth=50)  # Get deep order book

            if not order_book or (not order_book["bids"] and not order_book["asks"]):
                # Use ticker price as fallback
                try:
                    ticker = self.get_ticker(symbol)
                    fallback_price = ticker.get("last_price", 0.0)

                    # Create a fallback response using ticker price
                    return {
                        "price": fallback_price,
                        "enough_liquidity": False,
                        "message": f"Empty order book for {symbol}, using ticker price",
                        "batches": [{"price": fallback_price, "amount": amount}],
                        "total_cost": amount * fallback_price,
                        "slippage": 0.0,
                    }
                except Exception as ticker_error:
                    logger.error(f"Error getting ticker for {symbol}: {ticker_error}")
                    return {
                        "price": 0.0,
                        "enough_liquidity": False,
                        "message": f"Unable to retrieve order book or ticker for {symbol}",
                    }

            # Depending on side, analyze the opposite side of the book
            levels = order_book["asks"] if side == OrderSide.BUY else order_book["bids"]

            if not levels:
                message = f"No {'asks' if side == OrderSide.BUY else 'bids'} in order book"
                logger.warning(message)

                # Try to use ticker price as fallback
                try:
                    ticker = self.get_ticker(symbol)
                    fallback_price = ticker.get("last_price", 0.0)

                    return {
                        "price": fallback_price,
                        "enough_liquidity": False,
                        "message": message + f", using ticker price {fallback_price}",
                        "batches": [{"price": fallback_price, "amount": amount}],
                        "total_cost": amount * fallback_price,
                        "slippage": 0.0,
                    }
                except Exception:
                    return {
                        "price": 0.0,
                        "enough_liquidity": False,
                        "message": message,
                    }

            # Current best price
            best_price = levels[0][0]

            # For small orders, just use the best price with a small buffer
            if amount <= levels[0][1]:
                # Add/subtract a small percentage for increased fill probability
                buffer = 0.001  # 0.1%
                optimal_price = (
                    best_price * (1 + buffer)
                    if side == OrderSide.BUY
                    else best_price * (1 - buffer)
                )

                return {
                    "price": optimal_price,
                    "batches": [],
                    "total_cost": amount * optimal_price,
                    "slippage": buffer,
                    "enough_liquidity": True,
                    "message": "Small order, using best price with buffer",
                }

            # For larger orders, calculate weighted average price
            cumulative_size = 0.0
            cumulative_value = 0.0

            # Calculate how deep we need to go in the order book
            for level in levels:
                price, size = level

                if cumulative_size + size >= amount:
                    # We have enough liquidity up to this level
                    remaining = amount - cumulative_size
                    cumulative_value += price * remaining
                    cumulative_size = amount
                    break
                else:
                    cumulative_value += price * size
                    cumulative_size += size

            # Check if we found enough liquidity
            if cumulative_size < amount:
                # Not enough liquidity in the order book
                # Use the worst available price with appropriate buffer
                worst_price = levels[-1][0] if levels else best_price
                buffer = 0.005  # 0.5%
                optimal_price = (
                    worst_price * (1 + buffer)
                    if side == OrderSide.BUY
                    else worst_price * (1 - buffer)
                )

                return {
                    "price": optimal_price,
                    "enough_liquidity": False,
                    "message": f"Insufficient liquidity in order book. Found {cumulative_size}/{amount} {symbol}",
                    "total_cost": cumulative_value,
                    "slippage": abs(optimal_price - best_price) / best_price,
                    "batches": [],
                }

            # Calculate volume-weighted average price
            vwap = cumulative_value / amount

            # Add a buffer for market impact
            buffer = 0.005  # 0.5%
            optimal_price = (
                vwap * (1 + buffer) if side == OrderSide.BUY else vwap * (1 - buffer)
            )

            # Calculate slippage from best price
            slippage = abs(optimal_price - best_price) / best_price

            # For very large orders, suggest batching
            batches = []
            if amount > 10 * levels[0][1]:
                # Split into multiple batches
                batch_size = amount / 3  # Simple division into thirds
                batches = [
                    {"size": batch_size, "price": optimal_price},
                    {"size": batch_size, "price": optimal_price},
                    {"size": amount - 2 * batch_size, "price": optimal_price},
                ]

            return {
                "price": optimal_price,
                "batches": batches,
                "total_cost": amount * optimal_price,
                "slippage": slippage,
                "enough_liquidity": True,
                "message": "Calculated optimal limit price based on order book depth",
            }

        except Exception as e:
            error_msg = f"Error calculating optimal limit price for {symbol}: {e}"
            logger.error(error_msg)

            # Try to get fallback price from ticker
            try:
                ticker = self.get_ticker(symbol)
                fallback_price = ticker.get("last_price", 0.0)

                return {
                    "price": fallback_price,
                    "enough_liquidity": False,
                    "message": f"{error_msg}. Falling back to ticker price.",
                    "batches": [{"price": fallback_price, "amount": amount}],
                    "total_cost": amount * fallback_price,
                    "slippage": 0.0,
                }
            except Exception:
                # If even the ticker fails, return zeros
                return {
                    "price": 0.0,
                    "enough_liquidity": False,
                    "message": error_msg,
                    "batches": [],
                    "total_cost": 0.0,
                    "slippage": 0.0,
                }

    @track_api_latency(exchange="kraken", endpoint="get_historical_candles")
    def get_historical_candles(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get historical candlestick data from Kraken.

        Args:
            symbol: The market symbol (e.g., 'BTC')
            interval: Time interval (e.g., '1m', '5m', '1h')
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Maximum number of candles to retrieve

        Returns:
            List of candle data with timestamp, open, high, low, close, volume
        """
        # Try spot candles first
        candles = self._get_spot_historical_candles(
            symbol, interval, start_time, end_time, limit
        )

        # If spot candles are empty and we support perpetual markets, try futures
        if not candles and MarketType.PERPETUAL in self.market_types:
            try:
                candles = self._get_futures_historical_candles(
                    symbol, interval, start_time, end_time, limit
                )
            except Exception as e:
                logger.error(f"Error getting futures historical candles: {e}")

        return candles

    def _get_spot_historical_candles(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get historical candles from spot markets."""
        try:
            if not self.client:
                self.connect()

            # Convert BTC to XBT for Kraken API if needed
            kraken_symbol = symbol
            if symbol == 'BTC':
                kraken_symbol = 'XBT'

            # Get the right Kraken symbol format
            if kraken_symbol == 'XBT':
                product_id = 'XXBTZUSD'
            elif kraken_symbol == 'ETH':
                product_id = 'XETHZUSD'
            else:
                # Try to get from symbol map or construct
                product_id = self.symbol_map.get(symbol, f"{symbol}USD")

            # Convert interval to minutes
            interval_minutes = self._convert_interval_to_minutes(interval)

            # Prepare request params
            params = {
                'pair': product_id,
                'interval': interval_minutes
            }

            # Add start/end times if provided
            if start_time:
                params['since'] = start_time // 1000  # Convert from ms to seconds

            # Call the OHLC endpoint
            response = self.client.query_public('OHLC', params)

            if 'error' in response and response['error']:
                logger.error(f"Error fetching spot candles: {response['error']}")
                return []

            # Process candles
            candles_data = response.get('result', {}).get(product_id, [])
            if not candles_data:
                logger.warning(f"No candles data found for {product_id}")
                return []

            candles = []
            for candle in candles_data[:limit]:
                # Kraken format: [time, open, high, low, close, vwap, volume, count]
                if len(candle) >= 8:
                    candles.append({
                        'timestamp': int(candle[0]) * 1000,  # Convert to ms
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[6]
                    })

            return candles

        except Exception as e:
            logger.error(f"Error getting spot historical candles for {symbol}: {e}")
            return []

    def _get_futures_historical_candles(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get historical candles for futures markets"""
        try:
            # Convert our interval to Kraken futures interval format
            interval_map = {
                "1m": 60,
                "5m": 300,
                "15m": 900,
                "30m": 1800,
                "1h": 3600,
                "4h": 14400,
                "1d": 86400,
            }

            kraken_interval = interval_map.get(interval)
            if not kraken_interval:
                logger.error(f"Unsupported interval for futures: {interval}")
                return []

            # Get the futures symbol
            product_id = self._get_product_id(symbol, MarketType.PERPETUAL)

            # Prepare parameters
            params = {
                'symbol': product_id,
                'resolution': kraken_interval,
            }

            if start_time:
                params['from'] = int(start_time / 1000)  # Convert to seconds
            if end_time:
                params['to'] = int(end_time / 1000)  # Convert to seconds
            if limit:
                params['limit'] = limit

            # Make the API call
            response = self._make_futures_request('GET', 'history/candles', params)

            if 'error' in response and response['error']:
                logger.error(f"Error fetching futures candles: {response['error']}")
                return []

            candles = []
            for candle in response.get('candles', []):
                try:
                    candles.append({
                        "timestamp": int(candle['timestamp'] * 1000),  # Convert to milliseconds
                        "open": float(candle['open']),
                        "high": float(candle['high']),
                        "low": float(candle['low']),
                        "close": float(candle['close']),
                        "volume": float(candle['volume']),
                    })
                except (KeyError, ValueError) as e:
                    logger.error(f"Error processing futures candle: {e}")
                    continue

            return candles

        except Exception as e:
            logger.error(f"Error fetching futures historical candles: {e}")
            return []

    def _convert_interval_to_minutes(self, interval: str) -> int:
        """
        Convert our standard interval format to Kraken interval in minutes.

        Args:
            interval: Time interval (e.g., '1m', '5m', '1h', '1d')

        Returns:
            int: Interval in minutes
        """
        # Map intervals to minutes
        minutes_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "6h": 360,
            "12h": 720,
            "1d": 1440,
            "3d": 4320,
            "1w": 10080,
            "2w": 20160,
            "1M": 43200,
        }

        return minutes_map.get(interval, 1)  # Default to 1m if not found

    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get funding rate for a perpetual market.

        Args:
            symbol: The market symbol (e.g., 'BTC')

        Returns:
            Dict with funding rate info
        """
        try:
            # Check if the symbol is a futures/perpetual symbol
            if symbol not in self.futures_symbol_map:
                return {
                    "symbol": symbol,
                    "rate": 0.0,
                    "next_funding_time": None,
                    "message": "Spot markets do not have funding rates",
                }

            # For futures/perpetual markets, get the funding rate
            product_id = self._get_product_id(symbol, MarketType.PERPETUAL)

            # This is a placeholder for a real implementation
            # In a real implementation, we would use the Futures API
            # to fetch funding rates

            logger.warning("Futures funding rate not fully implemented")

            # Return placeholder funding rate
            return {
                "symbol": symbol,
                "rate": 0.0001,  # Placeholder
                "next_funding_time": int(time.time() * 1000) + 3600000,  # Placeholder (1 hour from now)
                "message": "Funding rate information (placeholder)",
            }

        except Exception as e:
            logger.error(f"Error getting funding rate for {symbol}: {e}")
            return {
                "symbol": symbol,
                "rate": 0.0,
                "next_funding_time": None,
                "message": f"Error fetching funding rate: {e}",
            }

    def get_leverage_tiers(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get information about leverage tiers and limits.

        Args:
            symbol: The market symbol (e.g., 'BTC')

        Returns:
            List[Dict[str, Any]]: Information about leverage tiers
        """
        try:
            # Check if the symbol is a futures/perpetual symbol
            if symbol not in self.futures_symbol_map:
                # Return empty list for spot markets
                return []

            # For futures/perpetual markets, get the leverage tiers
            # This is a placeholder for a real implementation

            # Return placeholder leverage tiers
            return [
                {
                    "tier": 1,
                    "min_notional": 0,
                    "max_notional": 50000,
                    "max_leverage": 5.0,
                    "maintenance_margin_rate": 0.01,
                    "initial_margin_rate": 0.02,
                },
                {
                    "tier": 2,
                    "min_notional": 50000,
                    "max_notional": 250000,
                    "max_leverage": 4.0,
                    "maintenance_margin_rate": 0.025,
                    "initial_margin_rate": 0.05,
                },
                {
                    "tier": 3,
                    "min_notional": 250000,
                    "max_notional": 1000000,
                    "max_leverage": 3.0,
                    "maintenance_margin_rate": 0.05,
                    "initial_margin_rate": 0.1,
                },
                {
                    "tier": 4,
                    "min_notional": 1000000,
                    "max_notional": 10000000,
                    "max_leverage": 2.0,
                    "maintenance_margin_rate": 0.1,
                    "initial_margin_rate": 0.15,
                },
            ]

        except Exception as e:
            logger.error(f"Error getting leverage tiers for {symbol}: {e}")
            return []

    def set_leverage(self, symbol: str, leverage: float) -> Dict[str, Any]:
        """
        Set leverage for a symbol.

        Args:
            symbol: The market symbol (e.g., 'BTC')
            leverage: Leverage multiplier to set

        Returns:
            Dict[str, Any]: Result of setting leverage
        """
        try:
            # Check if the symbol is a futures/perpetual symbol
            if symbol not in self.futures_symbol_map:
                logger.info(f"Ignoring set_leverage request for spot trading symbol {symbol}")
                return {
                    "success": False,
                    "message": f"Leverage trading is not supported for spot trading. Symbol: {symbol}, Requested leverage: {leverage}",
                    "symbol": symbol,
                    "leverage": 1.0,  # Always 1.0 for spot trading
                }

            # For futures/perpetual markets, set the leverage
            product_id = self._get_product_id(symbol, MarketType.PERPETUAL)

            # This is a placeholder for a real implementation
            # In a real implementation, we would use the Futures API
            # to set leverage

            logger.warning(f"Setting leverage for {symbol} to {leverage} (placeholder implementation)")

            # Return success response
            return {
                "success": True,
                "message": f"Leverage set to {leverage} for {symbol}",
                "symbol": symbol,
                "leverage": leverage,
            }

        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {e}")
            return {
                "success": False,
                "message": f"Error setting leverage: {e}",
                "symbol": symbol,
                "leverage": 1.0,
            }

    def cleanup(self) -> bool:
        """
        Clean up resources when shutting down.

        This method ensures that all resources are properly released
        when the application is shutting down.

        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            logger.info("Cleaning up Kraken connector resources...")

            # First try to disconnect if connected
            if self.client:
                self.disconnect()

            return True
        except Exception as e:
            logger.error(f"Error during Kraken connector cleanup: {e}")
            return False

    def create_hedge_position(
        self, symbol: str, amount: float, reference_price: float = None
    ) -> Dict[str, Any]:
        """
        Create a hedge position for a given position on another exchange.

        Args:
            symbol: The market symbol (e.g., 'BTC')
            amount: The amount to hedge (negative for short hedge, positive for long hedge)
            reference_price: Optional reference price for limit orders

        Returns:
            Dict with hedge operation result
        """
        try:
            if not self.client:
                self.connect()

            # Convert negative amount to positive for order size
            order_amount = abs(amount)

            # Determine if we need to buy or sell based on hedge direction
            # Negative amount means we want a short hedge (sell on Kraken)
            # Positive amount means we want a long hedge (buy on Kraken)
            side = OrderSide.SELL if amount < 0 else OrderSide.BUY

            self.orders_logger.info(
                f"Creating hedge position for {symbol}: {side.value} {order_amount} at " +
                (f"reference price ~{reference_price}" if reference_price else "market price")
            )

            # If reference price is provided, we'll try to place a limit order
            # with a small buffer to increase fill probability
            if reference_price:
                # Add a small buffer to ensure fill (0.2%)
                buffer = 0.002
                if side == OrderSide.BUY:
                    # For buy orders, willing to pay slightly more than reference
                    limit_price = reference_price * (1 + buffer)
                else:
                    # For sell orders, willing to accept slightly less than reference
                    limit_price = reference_price * (1 - buffer)

                order_result = self.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.LIMIT,
                    amount=order_amount,
                    leverage=1.0,  # Default leverage
                    price=limit_price,
                )
            else:
                # Without reference price, use market order for immediate execution
                order_result = self.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    amount=order_amount,
                    leverage=1.0,  # Default leverage
                )

            # Check if order placement was successful
            if "error" in order_result:
                self.orders_logger.error(f"Failed to create hedge position: {order_result['error']}")
                return {
                    "success": False,
                    "message": f"Hedge creation failed: {order_result['error']}",
                    "hedge_amount": 0.0,
                    "hedge_direction": side.value,
                }

            # Return successful hedge result
            self.orders_logger.info(f"Successfully created hedge position with order ID: {order_result['order_id']}")
            return {
                "success": True,
                "message": f"Hedge position created with {side.value} order",
                "order": order_result,
                "hedge_amount": order_amount,
                "hedge_direction": side.value,
            }

        except Exception as e:
            error_msg = f"Error creating hedge position for {symbol}: {e}"
            self.orders_logger.error(error_msg)
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "hedge_amount": 0.0,
                "hedge_direction": OrderSide.BUY.value if amount > 0 else OrderSide.SELL.value,
            }

    def adjust_hedge_position(
        self, symbol: str, target_amount: float, current_position: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Adjust an existing hedge position to match a target amount.

        Args:
            symbol: The market symbol (e.g., 'BTC')
            target_amount: The new target hedge amount (negative for short, positive for long)
            current_position: Optional dict with current position details

        Returns:
            Dict with hedge adjustment result
        """
        try:
            if not self.client:
                self.connect()

            # Get current position if not provided
            if current_position is None:
                # For spot markets, we need to get the current balance
                balances = self.get_account_balance()
                current_amount = balances.get(symbol, 0.0)

                # For a proper hedge, we need to determine if this is a long or short hedge
                # We'll assume long position (positive amount) by default
                current_position = {
                    "symbol": symbol,
                    "size": current_amount,
                    "side": "LONG",  # Spot positions are always long
                }
            else:
                # Extract position details from the provided dict
                current_amount = current_position.get("size", 0.0)
                if current_position.get("side", "LONG") == "SHORT":
                    # If it's a short position, use negative amount
                    current_amount = -current_amount

            # Calculate adjustment needed
            adjustment = target_amount - current_amount

            # If adjustment is very small, skip
            if abs(adjustment) < 0.001:
                self.orders_logger.info(
                    f"Hedge position for {symbol} already at target amount (within tolerance)"
                )
                return {
                    "success": True,
                    "message": "No adjustment needed",
                    "hedge_amount": current_amount,
                    "target_amount": target_amount,
                    "adjustment": 0.0,
                }

            # Create hedge for the adjustment amount
            result = self.create_hedge_position(symbol, adjustment)

            # Add additional context to the result
            if result["success"]:
                result["previous_amount"] = current_amount
                result["target_amount"] = target_amount
                result["adjustment"] = adjustment

            return result

        except Exception as e:
            error_msg = f"Error adjusting hedge position for {symbol}: {e}"
            self.orders_logger.error(error_msg)
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "previous_amount": 0.0,
                "target_amount": target_amount,
                "adjustment": 0.0,
            }
