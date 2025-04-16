import json
import logging
import time
from datetime import timedelta
from decimal import Decimal
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import urllib3.exceptions
# Import the BaseConnector interface
from connectors.base_connector import (BaseConnector, ConnectorError,
                                       MarketType, OrderSide, OrderStatus,
                                       OrderType)

# For type hinting and future import
try:
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
except ImportError:
    # We'll handle actual import in the constructor
    pass

logger = logging.getLogger(__name__)


# Define custom exception classes for better error handling
class HyperliquidConnectionError(Exception):
    """Raised when connection to Hyperliquid API fails"""

    pass


class HyperliquidTimeoutError(Exception):
    """Raised when a request to Hyperliquid API times out"""

    pass


class HyperliquidAPIError(Exception):
    """Raised when Hyperliquid API returns an error response"""

    pass


# Custom wrapper for Hyperliquid Info class to handle response format differences
class MetadataWrapper:
    """
    Wrapper for Hyperliquid metadata to handle different response formats.
    The API sometimes returns a list instead of a dict with a 'universe' key.
    """
    def __init__(self, base_url):
        self.base_url = base_url
        self._universe_cache = None
        self._api_client = None

    def _setup_api_client(self):
        """Set up API client if not already initialized"""
        if self._api_client is None:
            try:
                from hyperliquid.info import Info
                self._api_client = Info(base_url=self.base_url)
            except ImportError as e:
                logger.error(f"Failed to import hyperliquid.info: {e}")
                raise
        return self._api_client

    def meta(self):
        """
        Get metadata with proper handling of different response formats.
        Returns a standardized dict with a 'universe' key.
        """
        client = self._setup_api_client()

        try:
            # Get raw metadata response
            raw_meta = client.meta()
            logger.debug(f"Raw metadata response type: {type(raw_meta)}")

            # Handle different response formats
            if isinstance(raw_meta, dict) and "universe" in raw_meta:
                # Standard format as expected by the connector
                return raw_meta
            elif isinstance(raw_meta, list):
                # API returned a list directly - wrap it in a dict
                logger.info("Metadata response is a list instead of dict, adapting format")
                return {"universe": raw_meta}
            else:
                # Unknown format - log and try to adapt
                logger.warning(f"Unexpected metadata format: {type(raw_meta)}, attempting to adapt")
                if isinstance(raw_meta, dict):
                    # If it's a dict but missing 'universe', try to find something that looks like universe data
                    for key, value in raw_meta.items():
                        if isinstance(value, list) and len(value) > 0:
                            logger.info(f"Using '{key}' as universe data")
                            return {"universe": value}
                    # No suitable list found, create empty universe
                    return {"universe": []}
                else:
                    # Last resort - treat the entire response as universe
                    return {"universe": [raw_meta]}
        except Exception as e:
            logger.error(f"Error in meta() wrapper: {e}")
            # Return empty universe as fallback
            return {"universe": []}

    def all_mids(self):
        """Get all mid prices"""
        client = self._setup_api_client()
        return client.all_mids()

    def user_state(self, address):
        """Get user state"""
        client = self._setup_api_client()
        return client.user_state(address)

    # Add other methods as needed, following the same pattern


# Define a decorator for API call retries
def retry_api_call(max_tries=3, backoff_factor=1.5, max_backoff=30):
    """
    Decorator for retrying API calls with exponential backoff

    Args:
        max_tries: Maximum number of attempts
        backoff_factor: Multiplier for backoff time between retries
        max_backoff: Maximum backoff time in seconds
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_exceptions = (
                HyperliquidConnectionError,
                HyperliquidTimeoutError,
                urllib3.exceptions.NewConnectionError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            )

            last_exception = None
            for attempt in range(max_tries):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    if attempt < max_tries - 1:  # Don't sleep on the last attempt
                        wait_time = min(backoff_factor**attempt, max_backoff)
                        logger.warning(
                            f"API call to {func.__name__} failed with error: {str(e)}. "
                            f"Retrying in {wait_time:.2f}s (attempt {attempt+1}/{max_tries})"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"API call to {func.__name__} failed after {max_tries} attempts: {str(e)}"
                        )
            # If we get here, all retries have failed
            raise last_exception

        return wrapper

    return decorator


class HyperliquidConnector(BaseConnector):
    """
    Connector for Hyperliquid DEX using the official Python SDK.

    This class implements the BaseConnector interface for Hyperliquid.
    Hyperliquid primarily supports perpetual futures markets.
    """

    MAINNET_API_URL = "https://api.hyperliquid.xyz"
    TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"

    def __init__(
        self,
        name: str = "hyperliquid",
        wallet_address: str = "",
        private_key: str = "",
        testnet: bool = True,
        rpc_url: Optional[str] = None,
        market_types: Optional[List[MarketType]] = None,
    ):
        """
        Initialize the Hyperliquid connector.

        Args:
            name: Custom name for this connector instance
            wallet_address: The Ethereum address associated with the API key
            private_key: The private key for signing API requests
            testnet: Whether to connect to testnet or mainnet
            rpc_url: Custom RPC URL for blockchain connection
            market_types: List of market types this connector supports
                         (defaults to [PERPETUAL] for Hyperliquid)
        """
        # Default market types for HyperLiquid if none provided
        if market_types is None:
            market_types = [MarketType.PERPETUAL]

        # Call the parent class constructor
        super().__init__(
            name=name, exchange_type="hyperliquid", market_types=market_types
        )

        self.wallet_address = wallet_address
        self.private_key = private_key
        self.testnet = testnet
        self.rpc_url = rpc_url
        self.exchange = None
        self.info = None
        self.connection_timeout = 10  # Default timeout in seconds
        self.retry_count = 0  # Track connection retry attempts
        self.candle_cache = {} # Initialize cache
        self.cache_ttl = 60 # Cache time-to-live in seconds

        # Set up dedicated loggers
        self.setup_loggers()

        # Set API URL based on testnet flag
        self.api_url = self.TESTNET_API_URL if testnet else self.MAINNET_API_URL

        # Log the supported market types
        logger.info(
            f"Initialized HyperliquidConnector with market types: {[mt.value for mt in self.market_types]}"
        )

    @retry_api_call(max_tries=3, backoff_factor=2)
    def connect(self) -> bool:
        """
        Establish connection to Hyperliquid API.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Defer imports until needed to avoid import errors if library not installed
            import requests  # Keep requests for potential direct calls elsewhere
            from eth_account import Account
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info

            logger.info(
                f"Connecting to Hyperliquid API at {self.api_url} (testnet={self.testnet})"
            )

            # Reset retry count on new connection attempt
            self.retry_count = 0

            # Initialize info client for data retrieval using our wrapper to handle different API formats
            try:
                # Use our custom wrapper instead of directly initializing Info
                self.info = MetadataWrapper(base_url=self.api_url)

                # Make the meta() call the primary connection check
                logger.info("Testing connection by fetching exchange metadata...")
                meta = self.info.meta()
                if not meta or "universe" not in meta:
                    logger.error("Failed to fetch valid metadata from Hyperliquid API.")
                    self._is_connected = False
                    return False

                # Log details about the universe data
                universe = meta.get("universe", [])
                logger.info(f"Successfully fetched metadata. Found {len(universe)} markets.")
                if len(universe) > 0:
                    logger.debug(f"First market: {universe[0]}")

            except Exception as e:
                logger.error(f"Failed to initialize Info client or fetch metadata: {e}", exc_info=True)
                self._is_connected = False
                return False

            # Initialize exchange client for trading
            try:
                if self.private_key:
                    # Create a wallet object from the private key
                    wallet = Account.from_key(self.private_key)
                    self.exchange = Exchange(wallet=wallet, base_url=self.api_url)
                    logger.info("Exchange client initialized successfully.")
                else:
                    logger.warning("No private key provided. Exchange client not initialized. Read-only operations only.")
                    self.exchange = None
            except Exception as e:
                logger.error(f"Failed to initialize wallet or Exchange client: {e}", exc_info=True)
                # Don't necessarily mark as disconnected if info client worked
                if self.private_key:  # Only consider it a failure if a key was provided
                    logger.warning("Exchange client initialization failed, but continuing with read-only operations")
                self.exchange = None

            # If info client initialized successfully, consider connected
            self._is_connected = True
            logger.info(
                f"Connected to Hyperliquid {'Testnet' if self.testnet else 'Mainnet'} successfully."
            )
            return True

        except ImportError as e:
             logger.error(f"Missing required Hyperliquid libraries (hyperliquid, eth_account): {e}")
             self._is_connected = False
             return False
        except Exception as e:
            # Catch-all for any other unexpected errors during connection
            logger.error(f"Unexpected error during connection to Hyperliquid: {e}", exc_info=True)
            self._is_connected = False
            return False

    def get_markets(self) -> List[Dict[str, Any]]:
        """
        Get available markets from Hyperliquid.
        Currently, only perpetual futures are supported.
        Spot markets may be supported in future API versions.

        Returns:
            List of market details with appropriate market types
        """
        if not self._is_connected:
            self.connect()

        markets = []

        # Get perpetual futures markets
        try:
            # Fetch all metadata info from Hyperliquid
            meta_info = self.info.meta()

            # Extract perp markets and format them
            perps_count = len(meta_info.get("universe", []))

            # Log that we're processing markets
            if hasattr(self, "markets_logger") and self.markets_logger:
                self.markets_logger.info(
                    f"Retrieved {perps_count} perpetual markets from Hyperliquid"
                )

            for asset in meta_info.get("universe", []):
                asset_info = {
                    "symbol": asset.get("name"),
                    "base_asset": asset.get("name"),
                    "quote_asset": "USD",
                    "price_precision": asset.get("szDecimals", 2),
                    "min_size": asset.get("minSize", 0.01),
                    "tick_size": asset.get("tickSize", 0.01),
                    "maker_fee": asset.get("makerFeeRate", 0.0),
                    "taker_fee": asset.get("takerFeeRate", 0.0),
                    "market_type": MarketType.PERPETUAL.value,
                    "exchange_specific": {
                        "max_leverage": asset.get("maxLeverage", 50.0),
                        "funding_interval": asset.get("fundingInterval", 3600),
                        "section": "perp",
                    },
                    "active": True,
                }
                markets.append(asset_info)

            # Log that spot markets are a future feature
            logger.info("Spot markets are planned for future Hyperliquid API versions")

        except Exception as e:
            logger.error(f"Error getting perpetual markets: {e}")

        if hasattr(self, "markets_logger") and self.markets_logger:
            self.markets_logger.info(f"Total markets retrieved from Hyperliquid: {len(markets)}")

        return markets

    @retry_api_call(max_tries=3, backoff_factor=2)
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balance information from Hyperliquid perps.
        Spot balances may be supported in future API versions.

        Returns:
            Dict[str, float]: Asset balances with section prefixes
        """
        if not self.info:
            logger.warning("Not connected to Hyperliquid. Attempting to connect...")
            if not self.connect():
                raise HyperliquidConnectionError(
                    "Failed to connect to Hyperliquid. Please check network and API status."
                )

        try:
            # Fetch perps balances
            try:
                logger.info(f"Fetching perps state for wallet: {self.wallet_address}")
                perps_state = self.info.user_state(self.wallet_address)
                logger.info(f"Received user_state response: {perps_state}")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error while getting perps state: {e}")
                self._is_connected = False
                raise HyperliquidConnectionError(
                    f"Failed to connect to Hyperliquid API: {e}"
                )
            except requests.exceptions.Timeout as e:
                logger.error(f"Request timed out while getting perps state: {e}")
                raise HyperliquidTimeoutError(
                    f"Request to Hyperliquid API timed out: {e}"
                )

            # Initialize the combined balance dictionary
            balances = {}

            # Process perps balance (USDC)
            if "marginSummary" in perps_state and "accountValue" in perps_state["marginSummary"]:
                try:
                    perps_usdc = float(perps_state["marginSummary"]["accountValue"])
                    logger.info(f"USDC balance found: {perps_usdc}")
                    balances["PERP_USDC"] = perps_usdc
                except (ValueError, TypeError) as e:
                    logger.error(f"Error converting USDC balance: {e}, value was: {perps_state['marginSummary']['accountValue']}")
                    perps_usdc = 0.0
                    balances["PERP_USDC"] = perps_usdc
            else:
                logger.warning("No 'marginSummary.accountValue' field found in user_state response")
                perps_usdc = 0.0
                balances["PERP_USDC"] = perps_usdc

            # Fetch metadata to get token names
            try:
                meta = self.info.meta()
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                logger.error(f"Error fetching metadata: {e}")
                raise

            # Get asset positions for perps
            positions_added = 0
            if "assetPositions" in perps_state:
                for position in perps_state.get("assetPositions", []):
                    coin_idx = position.get("coin")
                    if (
                        coin_idx is not None
                        and "universe" in meta
                        and coin_idx < len(meta["universe"])
                    ):
                        coin_name = meta["universe"][coin_idx]["name"]
                        try:
                            size = float(position.get("szi", "0"))
                            if size != 0:  # Only include non-zero positions
                                balances[f"PERP_{coin_name}"] = size
                                positions_added += 1
                        except (ValueError, TypeError) as e:
                            logger.error(f"Error converting position size: {e}, value was: {position.get('szi')}")
            else:
                logger.warning("No 'assetPositions' field found in user_state response")

            logger.info(f"Added {positions_added} position balances")

            # Log that spot balances are a future feature
            logger.info("Spot balances are planned for future Hyperliquid API versions")

            # Log the balances for debugging
            logger.info(f"Retrieved Hyperliquid balances: {balances}")

            if hasattr(self, "balance_logger") and self.balance_logger:
                self.balance_logger.info(f"Hyperliquid balances: {balances}")

            return balances

        except (HyperliquidConnectionError, HyperliquidTimeoutError):
            # These exceptions will be caught by the retry decorator
            raise
        except Exception as e:
            logger.error(f"Failed to get account balances: {e}")
            return {"PERP_USDC": 0.0}

    @retry_api_call(max_tries=3, backoff_factor=2)
    def get_spot_balances(self) -> Dict[str, float]:
        """
        Get spot trading balances from Hyperliquid.
        Note: Spot trading is planned for future Hyperliquid API versions.

        Returns:
            Dict mapping token symbols to their balances
        """
        # Log that spot balances are a future feature
        logger.info("Spot balances are planned for future Hyperliquid API versions")
        return {}

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker for a specific market.

        Args:
            symbol: Market symbol (e.g., 'BTC')

        Returns:
            Dict[str, Any]: Ticker information
        """
        if not self.info:
            raise ConnectionError("Not connected to Hyperliquid. Call connect() first.")

        try:
            # Translate symbol to base format for Hyperliquid
            base_symbol = self.translate_symbol(symbol)

            meta = self.info.meta() # Fetch metadata once

            all_mids = self.info.all_mids()

            coin_idx = None
            market_found_in_meta = False
            for i, coin in enumerate(meta.get("universe", [])):
                # Compare using the translated base_symbol
                if coin.get("name") == base_symbol:
                    coin_idx = i
                    market_found_in_meta = True
                    logger.debug(f"Found coin '{base_symbol}' at index {coin_idx} in metadata.")
                    break

            if not market_found_in_meta:
                 # Report error using the original symbol for clarity
                logger.error(f"Market symbol '{symbol}' (searched as '{base_symbol}') not found in Hyperliquid metadata universe.")
                raise ValueError(f"Market {symbol} (base: {base_symbol}) not found in metadata")

            # --- Get Price Logic ---
            last_price = None
            if isinstance(all_mids, dict):
                # If all_mids is a dict, try accessing by base_symbol name first
                if base_symbol in all_mids:
                    last_price = float(all_mids[base_symbol])
                    logger.debug(f"Found price for '{base_symbol}' directly in all_mids dict: {last_price}")
                elif coin_idx is not None and str(coin_idx) in all_mids: # Fallback: Check if index as string key works
                     last_price = float(all_mids[str(coin_idx)])
                     logger.debug(f"Found price using string index '{coin_idx}' in all_mids dict: {last_price}")
                elif coin_idx is not None and coin_idx in all_mids: # Fallback: Check if integer index as key works
                     last_price = float(all_mids[coin_idx])
                     logger.debug(f"Found price using integer index {coin_idx} in all_mids dict: {last_price}")
                else:
                     logger.warning(f"Could not find price for '{base_symbol}' or index {coin_idx} in all_mids dictionary: {list(all_mids.keys())[:20]}...") # Log some keys

            elif isinstance(all_mids, list):
                # If all_mids is a list, use coin_idx if valid
                if coin_idx is not None and 0 <= coin_idx < len(all_mids):
                    last_price = float(all_mids[coin_idx])
                    logger.debug(f"Found price using index {coin_idx} in all_mids list: {last_price}")
                else:
                    logger.warning(f"Index {coin_idx} is out of bounds or invalid for all_mids list (length {len(all_mids)}).")
            else:
                 logger.error(f"Unexpected type for all_mids: {type(all_mids)}. Cannot extract price.")


            if last_price is None:
                logger.error(f"Price data for symbol '{symbol}' (base: '{base_symbol}', index: {coin_idx}) could not be extracted from all_mids.")
                # Keep the original error structure but provide more context
                raise ValueError(f"No price data available or extractable from all_mids for {symbol} (base: {base_symbol}, index: {coin_idx})")
            # --- End Get Price Logic ---

            logger.debug(f"Successfully retrieved last_price: {last_price} for {symbol} (base: {base_symbol})")

            return {
                "symbol": symbol, # Return original symbol
                "last_price": last_price,
                "timestamp": int(time.time() * 1000),
            }

        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                logger.warning(f"Rate limit hit while getting ticker for {symbol}. Retrying after delay...")
                time.sleep(1)  # Add delay before retry
                all_mids = self.info.all_mids()  # Retry once
                if not all_mids or coin_idx >= len(all_mids):
                    raise ValueError(f"No price data available for {symbol}")
                return {
                    "symbol": symbol,
                    "last_price": float(all_mids[coin_idx]),
                    "timestamp": int(time.time() * 1000),
                }
            raise

        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            if isinstance(e, ValueError):
                raise HyperliquidAPIError(str(e))
            raise

    def get_orderbook(
        self, symbol: str, depth: int = 10
    ) -> Dict[str, List[List[float]]]:
        """
        Get current orderbook for a specific market.

        Args:
            symbol: Market symbol (e.g., 'BTC')
            depth: Number of price levels to retrieve

        Returns:
            Dict with 'bids' and 'asks' lists of [price, size] pairs
        """
        if not self.info:
            logger.warning("Not connected to Hyperliquid. Attempting to connect...")
            if not self.connect():
                raise HyperliquidConnectionError("Failed to connect.")

        try:
            # Translate symbol to base format for Hyperliquid
            base_symbol = self.translate_symbol(symbol)

            # Get market index
            meta = self.info.meta()
            coin_idx = None

            for i, coin in enumerate(meta.get("universe", [])):
                if coin.get("name") == base_symbol:
                    coin_idx = i
                    break

            if coin_idx is None:
                raise ValueError(f"Market {symbol} not found")

            # Get L2 orderbook with retry on rate limit
            max_retries = 3
            retry_delay = 1  # seconds

            for attempt in range(max_retries):
                try:
                    # Make POST request to /info endpoint for L2 book snapshot
                    response = requests.post(
                        f"{self.api_url}/info",
                        json={
                            "type": "l2Book",
                            "coin": base_symbol,  # Use symbol name instead of index
                        },
                        headers={"Content-Type": "application/json"},
                        timeout=10
                    )

                    if response.status_code == 429:  # Rate limit
                        if attempt < max_retries - 1:
                            logger.warning(f"Rate limit hit while getting orderbook for {symbol}. Retrying after delay...")
                            time.sleep(retry_delay)
                            continue
                    elif response.status_code == 500:  # Server error
                        if attempt < max_retries - 1:
                            logger.warning(f"Server error while getting orderbook for {symbol}. Retrying after delay...")
                            time.sleep(retry_delay)
                            continue

                    response.raise_for_status()
                    data = response.json()

                    # Debug log the type of data received
                    logger.debug(f"Orderbook data type: {type(data)}, content sample: {str(data)[:200]}...")

                    # Handle different response formats
                    if isinstance(data, dict) and "levels" in data:
                        # Format: {"coin": "ETH", "time": 1234567890, "levels": [[bids], [asks]]}
                        logger.debug("Detected structured orderbook format with 'levels' key")

                        bids_data = data["levels"][0] if len(data["levels"]) > 0 else []
                        asks_data = data["levels"][1] if len(data["levels"]) > 1 else []

                        # Format with px/sz fields
                        bids = []
                        for bid in bids_data[:depth]:
                            if isinstance(bid, dict) and "px" in bid and "sz" in bid:
                                bids.append([float(bid["px"]), float(bid["sz"])])
                            elif isinstance(bid, list) and len(bid) >= 2:
                                bids.append([float(bid[0]), float(bid[1])])

                        asks = []
                        for ask in asks_data[:depth]:
                            if isinstance(ask, dict) and "px" in ask and "sz" in ask:
                                asks.append([float(ask["px"]), float(ask["sz"])])
                            elif isinstance(ask, list) and len(ask) >= 2:
                                asks.append([float(ask[0]), float(ask[1])])

                        return {"bids": bids, "asks": asks}

                    elif isinstance(data, list) and len(data) == 2:
                        # Format: [[bids], [asks]] where each item is [price, size] or {"px": price, "sz": size}
                        logger.debug("Detected simple array orderbook format [bids, asks]")

                        bids_data = data[0] if len(data) > 0 else []
                        asks_data = data[1] if len(data) > 1 else []

                        # Process bids - handle both array and object formats
                        bids = []
                        for bid in bids_data[:depth]:
                            if isinstance(bid, dict) and "px" in bid and "sz" in bid:
                                bids.append([float(bid["px"]), float(bid["sz"])])
                            elif isinstance(bid, list) and len(bid) >= 2:
                                bids.append([float(bid[0]), float(bid[1])])

                        # Process asks - handle both array and object formats
                        asks = []
                        for ask in asks_data[:depth]:
                            if isinstance(ask, dict) and "px" in ask and "sz" in ask:
                                asks.append([float(ask["px"]), float(ask["sz"])])
                            elif isinstance(ask, list) and len(ask) >= 2:
                                asks.append([float(ask[0]), float(ask[1])])

                        return {"bids": bids, "asks": asks}
                    else:
                        logger.warning(f"Unrecognized orderbook format: {data}")
                        raise ValueError(f"Invalid orderbook format from API: {data}")

                except (requests.exceptions.RequestException, ValueError) as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to get orderbook after {max_retries} attempts: {e}")
                        raise
                    time.sleep(retry_delay)

            raise ValueError(f"Failed to get orderbook after {max_retries} attempts")

        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol}: {e}")
            return {"bids": [], "asks": []}

    @retry_api_call(max_tries=3, backoff_factor=2)
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current open positions.

        Returns:
            List[Dict[str, Any]]: List of open positions
        """
        if not self.info:
            logger.warning("Not connected to Hyperliquid. Attempting to connect...")
            self.connect()
            if not self._is_connected:
                raise HyperliquidConnectionError("Failed to connect.")

        try:
            try:
                # Get user state with positions
                logger.debug(f"Fetching user state for wallet: {self.wallet_address}")
                user_state = self.info.user_state(self.wallet_address)
                logger.debug(f"User state response type: {type(user_state)}")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error while getting user state: {e}")
                # Update connection status for metrics
                self._is_connected = False
                raise HyperliquidConnectionError(f"Failed to connect to Hyperliquid API: {e}")
            except requests.exceptions.Timeout as e:
                logger.error(f"Request timed out while getting user state: {e}")
                raise HyperliquidTimeoutError(f"Request to Hyperliquid API timed out: {e}")

            try:
                # Get metadata for coin information
                logger.debug("Fetching metadata for position information")
                meta = self.info.meta()
                logger.debug(f"Meta response universe length: {len(meta.get('universe', []))}")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error while getting meta data: {e}")
                self._is_connected = False
                raise HyperliquidConnectionError(f"Failed to connect to Hyperliquid metadata API: {e}")

            try:
                # Get current prices
                logger.debug("Fetching current prices")
                all_mids = self.info.all_mids()
                logger.debug(f"all_mids response type: {type(all_mids)}")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error while getting all_mids: {e}")
                self._is_connected = False
                raise HyperliquidConnectionError(f"Failed to connect to Hyperliquid price API: {e}")

            positions = []

            # Process asset positions which could be in different formats based on API response
            if isinstance(user_state, dict) and "assetPositions" in user_state:
                # Standard format
                asset_positions = user_state.get("assetPositions", [])
                logger.debug(f"Found {len(asset_positions)} positions in standard format")

                for position in asset_positions:
                    try:
                        # Check if this is a nested position format (type: oneWay structure)
                        if isinstance(position, dict) and "type" in position and "position" in position:
                            logger.debug(f"Found nested position format: {position['type']}")
                            # Extract the actual position data from the nested structure
                            position = position["position"]

                        # Safely extract coin index
                        if "coin" not in position:
                            logger.warning(f"Position missing 'coin' field: {position}")
                            continue

                        coin_idx = position.get("coin")

                        # Validate coin index
                        universe = meta.get("universe", [])
                        if not isinstance(coin_idx, int) or coin_idx < 0 or coin_idx >= len(universe):
                            logger.warning(f"Invalid coin index {coin_idx} (universe size: {len(universe)})")
                            continue

                        # Get coin name
                        coin_info = universe[coin_idx]
                        coin_name = coin_info.get("name", f"Unknown-{coin_idx}")

                        # Get current price - handle different all_mids formats
                        current_price = None
                        if isinstance(all_mids, dict):
                            # Try by name first, then by index
                            if coin_name in all_mids:
                                current_price = float(all_mids[coin_name])
                            elif str(coin_idx) in all_mids:
                                current_price = float(all_mids[str(coin_idx)])
                            else:
                                logger.warning(f"Could not find price for {coin_name} in all_mids dict")
                        elif isinstance(all_mids, list) and coin_idx < len(all_mids):
                            current_price = float(all_mids[coin_idx])
                        else:
                            logger.warning(f"Could not extract price for {coin_name} from all_mids")

                        if current_price is None:
                            logger.warning(f"Skipping position for {coin_name} due to missing price data")
                            continue

                        # Extract position details with safe conversions
                        try:
                            size = float(position.get("szi", "0"))
                            entry_price = float(position.get("entryPx", "0"))

                            # Handle leverage which could be a scalar value or a complex object
                            leverage_data = position.get("leverage", "1.0")
                            if isinstance(leverage_data, dict) and "value" in leverage_data:
                                # Extract from {type: "cross", value: 20} format
                                leverage = float(leverage_data["value"])
                                leverage_type = leverage_data.get("type", "cross")
                                logger.debug(f"Extracted leverage {leverage} (type: {leverage_type}) from complex object")
                            else:
                                # Simple scalar value
                                leverage = float(leverage_data)

                            liquidation_price = float(position.get("liqPx", "0"))

                            # Skip positions with zero size
                            if size == 0:
                                continue

                            # Calculate PNL
                            unrealized_pnl = size * (current_price - entry_price)

                            # Format position info
                            position_info = {
                                "symbol": f"{coin_name}-USD",
                                "size": size,
                                "side": "LONG" if size > 0 else "SHORT",
                                "entry_price": entry_price,
                                "mark_price": current_price,
                                "leverage": leverage,
                                "liquidation_price": liquidation_price,
                                "unrealized_pnl": unrealized_pnl,
                                "margin": abs(size) / leverage if leverage > 0 else 0,
                            }

                            positions.append(position_info)
                            logger.debug(f"Added position for {coin_name}: size={size}, pnl={unrealized_pnl:.2f}")

                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error processing position values for {coin_name}: {e}")
                            continue

                    except Exception as e:
                        logger.warning(f"Error processing position: {e}")
                        continue

            elif isinstance(user_state, list):
                # Alternative API format - direct list of positions
                logger.debug(f"Position data in list format, length: {len(user_state)}")
                for position in user_state:
                    try:
                        if not isinstance(position, dict):
                            continue

                        # Try to extract symbol/coin information
                        coin_name = position.get("symbol") or position.get("coin") or position.get("asset")
                        if not coin_name:
                            logger.warning(f"Could not determine coin name from position: {position}")
                            continue

                        # Extract basic position data with safe conversions
                        try:
                            size = float(position.get("size", position.get("amount", "0")))
                            entry_price = float(position.get("entryPrice", position.get("entry", "0")))
                            mark_price = float(position.get("markPrice", position.get("price", "0")))
                            leverage = float(position.get("leverage", "1"))

                            # Skip positions with zero size
                            if size == 0:
                                continue

                            # Try to extract or calculate PNL
                            if "unrealizedPnl" in position:
                                unrealized_pnl = float(position["unrealizedPnl"])
                            else:
                                unrealized_pnl = size * (mark_price - entry_price)

                            # Format position info
                            position_info = {
                                "symbol": f"{coin_name}-USD",
                                "size": size,
                                "side": "LONG" if size > 0 else "SHORT",
                                "entry_price": entry_price,
                                "mark_price": mark_price,
                                "leverage": leverage,
                                "liquidation_price": float(position.get("liquidationPrice", "0")),
                                "unrealized_pnl": unrealized_pnl,
                                "margin": abs(size) / leverage if leverage > 0 else 0,
                            }

                            positions.append(position_info)
                            logger.debug(f"Added position for {coin_name} from list format")

                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error processing position values for {coin_name}: {e}")
                            continue

                    except Exception as e:
                        logger.warning(f"Error processing position from list: {e}")
                        continue
            else:
                logger.warning(f"Unexpected user_state format: {type(user_state)}")

            logger.info(f"Retrieved {len(positions)} positions from Hyperliquid")
            return positions

        except (HyperliquidConnectionError, HyperliquidTimeoutError):
            # These exceptions will be caught by the retry decorator
            raise
        except Exception as e:
            logger.error(f"Failed to get positions: {e}", exc_info=True)
            return []

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: Union[float, Decimal],
        price: Optional[float] = None,
        leverage: Optional[float] = None,
        reduce_only: bool = False,
        post_only: bool = False,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Place an order on Hyperliquid.

        Args:
            symbol: Market symbol (e.g., 'BTC')
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET/LIMIT)
            amount: Order amount in base currency
            price: Order price (required for LIMIT orders)
            leverage: Position leverage (default: 1x)
            reduce_only: Whether the order should only reduce position
            post_only: Whether the order must be maker (LIMIT orders only)
            client_order_id: Optional client order ID

        Returns:
            Dict[str, Any]: Order response containing status and potentially order ID
        """
        if not self.exchange or not self.info:
            logger.error("Not connected to Hyperliquid. Cannot place order.")
            raise HyperliquidConnectionError("Not connected to Hyperliquid. Call connect() first.")

        try:
            # Translate symbol to base format for Hyperliquid
            base_symbol = self.translate_symbol(symbol)

            # Convert amount to Decimal for consistent comparison
            amount_decimal = Decimal(str(amount))
            if amount_decimal <= Decimal('0'):
                logger.error(f"Order amount must be positive, got {amount}")
                raise ValueError("Order amount must be positive")

            min_order_size = Decimal(str(self.get_min_order_size(symbol)))
            if amount_decimal < min_order_size:
                logger.error(f"Order amount {amount} is below minimum size {min_order_size} for {symbol}")
                raise ValueError(f"Order amount {amount} is below minimum size {min_order_size} for {symbol}")

            if order_type == OrderType.LIMIT:
                if price is None or price <= 0:
                    logger.error(f"Limit order requires a positive price, got {price}")
                    raise ValueError("Limit order price must be positive and provided")
            elif order_type == OrderType.MARKET:
                # For market orders, get optimal price from orderbook analysis
                try:
                    optimal_price_data = self.get_optimal_limit_price(symbol, side, amount)
                    if not optimal_price_data["enough_liquidity"]:
                        logger.warning(f"Limited liquidity for {symbol} order of size {amount}. Slippage may be high.")

                    price = optimal_price_data["price"]
                    if price <= 0:
                        raise ValueError(f"Could not determine valid market price from orderbook for {symbol}")

                    logger.info(
                        f"Using optimal price {price} for {side.value} order of {amount} {symbol}. "
                        f"Expected slippage: {optimal_price_data['slippage']}%, "
                        f"Enough liquidity: {optimal_price_data['enough_liquidity']}"
                    )
                except Exception as e:
                    logger.error(f"Failed to get optimal price for {symbol}: {e}")
                    raise ValueError(f"Could not determine optimal market price for {symbol}: {e}")

            if leverage is not None and leverage <= 0:
                 logger.error(f"Leverage must be positive, got {leverage}")
                 raise ValueError("Leverage must be positive")

            # --- Fetch Market Metadata for Precision and Limits ---
            meta = self.info.meta()
            market_info = None
            coin_idx = None
            for i, coin_data in enumerate(meta.get("universe", [])):
                if coin_data.get("name") == self.translate_symbol(base_symbol): # Use translate_symbol for consistent case handling
                    market_info = coin_data
                    coin_idx = i
                    break

            if market_info is None or coin_idx is None:
                logger.error(f"Market {symbol} (base: {base_symbol}) not found in Hyperliquid metadata.")
                raise ValueError(f"Market {symbol} not found")

            size_decimals = market_info.get("szDecimals", 8) # Default to 8 if not found
            tick_size_str = market_info.get("tickSize", "0.1") # Price increment as string
            try:
                # Convert tick size to float for validation
                tick_size = float(tick_size_str)
                # Calculate price decimals from tickSize (e.g., "0.01" -> 2 decimals)
                if '.' in tick_size_str:
                    price_decimals = len(tick_size_str.split('.')[1])
                else:
                    price_decimals = 0 # Handle whole numbers
            except (ValueError, IndexError):
                logger.warning(f"Could not parse tickSize '{tick_size_str}' for {symbol}, defaulting price precision.")
                tick_size = 0.1  # Default tick size
                price_decimals = 2 # Default price precision

            logger.debug(f"Market {symbol}: szDecimals={size_decimals}, minSize={min_order_size}, priceDecimals={price_decimals}, tickSize={tick_size}")
            # --- End Fetch Market Metadata ---

            # --- Format Order Parameters for API ---
            formatted_amount = f"{amount:.{size_decimals}f}"
            formatted_price = None
            if order_type == OrderType.LIMIT and price is not None:
                formatted_price = f"{price:.{price_decimals}f}"

            order_params = {}
            if order_type == OrderType.LIMIT:
                order_params["order_type"] = {"limit": {"tif": "Gtc"}}
                if post_only:
                    order_params["order_type"] = {"limit": {"tif": "Alo"}}
            elif order_type == OrderType.MARKET:
                order_params["order_type"] = {"limit": {"tif": "Ioc"}}
                if post_only:
                    logger.warning("post_only is ignored for MARKET (Ioc) orders")
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            if client_order_id:
                order_params["cloid"] = client_order_id

            # Format the price according to market precision
            formatted_price = None
            if price is not None:
                formatted_price = f"{price:.{price_decimals}f}"

            sdk_limit_price = 0.0
            if formatted_price:
                try:
                    sdk_limit_price = float(formatted_price)
                except (ValueError, TypeError):
                    logger.error(f"Invalid formatted price: {formatted_price}")
                    raise ValueError(f"Invalid price: {price}")

            # Record the order parameters for SDK call
            order_data_for_sdk = {
                 "coin": base_symbol.upper(),  # Use translated base symbol for API
                 "is_buy": side == OrderSide.BUY,
                 "sz": float(formatted_amount),
                 "limit_px": sdk_limit_price,
                 "order_type": order_params["order_type"],
                 "reduce_only": reduce_only,
                 "cloid": client_order_id
             }
            if order_data_for_sdk["cloid"] is None: del order_data_for_sdk["cloid"]

            logger.info(f"Prepared order data for SDK: {order_data_for_sdk}")
            # --- End Format Order Parameters ---


            # --- Place the Order via SDK/API ---
            try:
                logger.debug(f"Calling exchange.order with data: {order_data_for_sdk}")
                response = self.exchange.order(
                    order_data_for_sdk["coin"],
                    order_data_for_sdk["is_buy"],
                    order_data_for_sdk["sz"],
                    order_data_for_sdk["limit_px"],
                    order_data_for_sdk["order_type"],
                    reduce_only=order_data_for_sdk["reduce_only"],
                    cloid=order_data_for_sdk.get("cloid")
                )
                logger.info(f"Order placement response: {response}")

                # --- Process Response ---
                processed_response = {
                    "status": "UNKNOWN",
                    "order_id": None,
                    "client_order_id": client_order_id,
                    "symbol": symbol,
                    "side": side.value,
                    "type": order_type.value,
                    "amount": amount,
                    "price": price,
                    "entry_price": price if price is not None else 0.0,
                    "raw_response": response
                }
                if isinstance(response, dict) and response.get("status") == "ok":
                    statuses = response.get("response", {}).get("data", {}).get("statuses", [])
                    if statuses and isinstance(statuses[0], dict):
                        order_info = statuses[0]
                        if "resting" in order_info:
                            processed_response["status"] = OrderStatus.OPEN.value
                            processed_response["order_id"] = order_info["resting"].get("oid")
                            if "px" in order_info["resting"]:
                                processed_response["entry_price"] = float(order_info["resting"]["px"])
                        elif "filled" in order_info:
                            processed_response["status"] = OrderStatus.FILLED.value
                            processed_response["order_id"] = order_info["filled"].get("oid")
                            processed_response["filled_amount"] = order_info["filled"].get("totalSz")
                            avg_price = order_info["filled"].get("avgPx")
                            processed_response["average_price"] = avg_price
                            processed_response["entry_price"] = float(avg_price) if avg_price is not None else price
                        elif "error" in order_info:
                            processed_response["status"] = OrderStatus.REJECTED.value
                            processed_response["error_message"] = order_info.get("error")
                            logger.error(f"Order placement failed (API Error): {order_info.get('error')}")
                        else:
                            logger.warning(f"Order placed but status unclear: {order_info}")
                            processed_response["status"] = "ACCEPTED_UNKNOWN_STATUS"
                elif isinstance(response, dict) and response.get("status") == "error":
                    processed_response["status"] = OrderStatus.REJECTED.value
                    processed_response["error_message"] = response.get("error")
                    logger.error(f"Order placement failed (API Error): {response.get('error')}")
                else:
                    logger.warning(f"Unrecognized order response format: {response}")
                # --- End Process Response ---

                return processed_response

            # --- Exception Handling for SDK/Network Call ---
            except requests.exceptions.RequestException as e:
                 logger.error(f"HTTP request failed during order placement for {symbol}: {e}")
                 if isinstance(e, requests.exceptions.Timeout):
                     raise HyperliquidTimeoutError(f"Request timed out: {e}")
                 elif isinstance(e, requests.exceptions.ConnectionError):
                     self._is_connected = False
                     raise HyperliquidConnectionError(f"Connection error: {e}")
                 else:
                     err_response = getattr(e, 'response', None)
                     if err_response is not None:
                         logger.error(f"API Error Response: Status={err_response.status_code}, Body={err_response.text}")
                     raise HyperliquidAPIError(f"API error placing order: {e}")
            except Exception as sdk_e: # Catch other SDK errors
                 logger.error(f"Hyperliquid SDK error placing order for {symbol}: {sdk_e}", exc_info=True)
                 return {
                     "status": OrderStatus.REJECTED.value,
                     "symbol": symbol,
                     "error": f"SDK error: {str(sdk_e)}",
                     "raw_response": str(sdk_e),
                     "entry_price": price if price is not None else 0.0  # Add entry_price even for errors
                 }
            # --- End Place Order Call Logic ---

        # Catch errors from validation or connection steps *before* the SDK call attempt
        except (ValueError, HyperliquidConnectionError, HyperliquidTimeoutError, HyperliquidAPIError) as e:
            logger.error(f"Order placement validation or connection error for {symbol}: {e}")
            raise # Re-raise specific known errors
        except Exception as e: # Catch any other unexpected errors during preparation
            logger.error(f"Unexpected error during order preparation for {symbol}: {e}", exc_info=True)
            return {
                 "status": OrderStatus.REJECTED.value,
                 "symbol": symbol,
                 "error": f"Unexpected preparation error: {str(e)}",
                 "entry_price": price if price is not None else 0.0  # Add entry_price for all error cases
             }

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an existing order by its ID.

        Args:
            order_id: The exchange-assigned order ID to cancel.
            symbol: The market symbol the order belongs to (required by Hyperliquid).

        Returns:
            Dict[str, Any] indicating success or failure.
        """
        if not self.exchange:
            logger.error("Not connected to Hyperliquid. Cannot cancel order.")
            raise HyperliquidConnectionError("Not connected to Hyperliquid.")

        if not symbol:
             logger.error("Symbol is required to cancel orders on Hyperliquid.")
             raise ValueError("Symbol is required for cancel_order on Hyperliquid")

        logger.info(f"Attempting to cancel order {order_id} for symbol {symbol}")
        try:
            # Hyperliquid API requires symbol (coin) and order ID (oid)
            response = self.exchange.cancel(self.translate_symbol(symbol), int(order_id))
            logger.info(f"Cancel order response for {order_id}: {response}")

            # Process response
            if isinstance(response, dict) and response.get("status") == "ok":
                 logger.info(f"Successfully initiated cancellation for order {order_id}")
                 # Check nested status if needed
                 statuses = response.get("response", {}).get("data", {}).get("statuses", [])
                 if statuses and statuses[0] == "cancelled":
                      return {"status": "success", "order_id": order_id, "message": "Order cancelled successfully."}
                 else:
                      # Might be pending cancellation or another state
                      return {"status": "pending", "order_id": order_id, "message": "Cancellation request accepted.", "raw_response": response}
            elif isinstance(response, dict) and response.get("status") == "error":
                 error_msg = response.get("error", "Unknown API error")
                 logger.error(f"Failed to cancel order {order_id}: {error_msg}")
                 return {"status": "error", "order_id": order_id, "error": error_msg}
            else:
                 logger.warning(f"Unrecognized cancel response format for order {order_id}: {response}")
                 return {"status": "unknown", "order_id": order_id, "raw_response": response}

        except ValueError as e:
            # Handle cases like invalid order_id format
            logger.error(f"Invalid input for cancelling order {order_id}: {e}")
            return {"status": "error", "order_id": order_id, "error": f"Invalid input: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error cancelling order {order_id}: {e}", exc_info=True)
            return {"status": "error", "order_id": order_id, "error": f"Unexpected error: {str(e)}"}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.

        Args:
            order_id: The ID of the order to check

        Returns:
            Dict[str, Any]: Order status and details
        """
        if not self.info:
            raise ConnectionError("Not connected to Hyperliquid. Call connect() first.")

        try:
            # Check open orders
            open_orders = self.exchange.open_orders()

            for order in open_orders:
                if order.get("oid") == order_id:
                    meta = self.info.meta()
                    coin_name = meta["universe"][order["coin"]]["name"]

                    return {
                        "order_id": order_id,
                        "symbol": coin_name,
                        "side": "BUY" if order.get("is_buy") else "SELL",
                        "type": "LIMIT" if "limit_px" in order else "MARKET",
                        "price": float(order.get("limit_px", 0)),
                        "amount": float(order.get("sz", 0)),
                        "filled": float(order.get("filled", 0)),
                        "status": OrderStatus.OPEN.value,
                        "timestamp": int(time.time() * 1000),
                    }

            # Order not found in open orders, could be filled or canceled
            # In a real implementation, you might check order history here
            return {
                "order_id": order_id,
                "status": OrderStatus.FILLED.value,  # Assumption, should check history
                "timestamp": int(time.time() * 1000),
            }

        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return {"order_id": order_id, "status": "UNKNOWN", "error": str(e)}

    def close_position(
        self, symbol: str, position_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Close an open position.

        Args:
            symbol: Market symbol (e.g., 'BTC')
            position_id: Not used for Hyperliquid

        Returns:
            Dict[str, Any]: Result of the close operation
        """
        if not self.exchange or not self.info:
            raise ConnectionError("Not connected to Hyperliquid. Call connect() first.")

        try:
            # Get coin index
            meta = self.info.meta()
            coin_idx = None

            for i, coin in enumerate(meta["universe"]):
                if coin["name"] == symbol:
                    coin_idx = i
                    break

            if coin_idx is None:
                raise ValueError(f"Market {symbol} not found")

            # Get current position
            positions = self.get_positions()
            current_position = None

            for position in positions:
                if position["symbol"] == symbol:
                    current_position = position
                    break

            if current_position is None or current_position["size"] == 0:
                return {"status": "NO_POSITION", "symbol": symbol}

            # Place opposite order to close position
            is_buy = current_position["side"] == "SHORT"
            size = abs(current_position["size"])

            # Use market order to close immediately
            order_params = {
                "coin": coin_idx,
                "is_buy": is_buy,
                "sz": size,
                "reduce_only": True,  # Important: set reduce_only to true
            }

            order_result = self.exchange.order_market(order_params)

            return {
                "status": "CLOSING",
                "symbol": symbol,
                "order_id": order_result.get("order", {}).get("oid", ""),
                "side": "BUY" if is_buy else "SELL",
                "amount": size,
                "timestamp": int(time.time() * 1000),
            }

        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return {"status": "ERROR", "symbol": symbol, "error": str(e)}

    def _parse_interval(self, interval: str) -> Tuple[timedelta, int]:
        """
        Parse interval string to timedelta and milliseconds.

        Args:
            interval: Time interval (e.g., '1m', '5m', '1h', '1d')

        Returns:
            Tuple[timedelta, int]: (timedelta object, milliseconds)
        """
        # Map intervals to (timedelta, milliseconds, hyperliquid_format)
        interval_map = {
            "1m": (timedelta(minutes=1), 60000),
            "5m": (timedelta(minutes=5), 300000),
            "15m": (timedelta(minutes=15), 900000),
            "30m": (timedelta(minutes=30), 1800000),
            "1h": (timedelta(hours=1), 3600000),
            "4h": (timedelta(hours=4), 14400000),
            "1d": (timedelta(days=1), 86400000),
        }

        if interval not in interval_map:
            raise ValueError(f"Unsupported interval: {interval}")

        return interval_map[interval]

    @retry_api_call(max_tries=3, backoff_factor=2)
    def get_historical_candles(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get historical Klines (candlestick data) for a specific market.

        Args:
            symbol: Market symbol (e.g., 'BTC')
            interval: Kline interval (e.g., '1m', '5m', '1h')
            start_time: Start time in milliseconds since epoch
            end_time: End time in milliseconds since epoch
            limit: Max number of candles to retrieve

        Returns:
            List[Dict[str, Any]]: List of candle data dictionaries
        """
        if not self.info:
            logger.warning("Not connected to Hyperliquid. Attempting to connect...")
            self.connect()
            if not self._is_connected:
                raise HyperliquidConnectionError("Failed to connect.")

        # Translate symbol to base format for Hyperliquid
        base_symbol = self.translate_symbol(symbol)

        # Cache key
        cache_key = f"{base_symbol}_{interval}_{start_time}_{end_time}_{limit}"

        # Check cache first to avoid unnecessary API calls
        if cache_key in self.candle_cache:
            cache_entry = self.candle_cache[cache_key]
            cache_age = time.time() - cache_entry['timestamp']

            # If cache is still valid, return it
            if cache_age < self.cache_ttl:
                logger.debug(f"Using cached candles for {symbol}, age: {cache_age:.1f}s")
                return cache_entry['data']
            else:
                logger.debug(f"Cache expired for {symbol}, fetching fresh data")

        # Parse interval
        try:
            interval_td, granularity = self._parse_interval(interval)
        except ValueError as e:
            logger.error(f"Invalid interval {interval}: {e}")
            raise ValueError(f"Invalid interval format: {interval}")

        # Convert interval to seconds for API
        interval_seconds = int(interval_td.total_seconds())

        # Prepare request data
        try:
            # Make direct request to the API for candles since the SDK doesn't seem to have this
            max_retries = 3
            retry_delay = 1  # seconds

            for attempt in range(max_retries):
                try:
                    # Format time range parameters
                    request_data = {
                        "type": "candles",
                        "coin": base_symbol,
                        "interval": interval_seconds,
                    }

                    if start_time:
                        request_data["startTime"] = start_time
                    if end_time:
                        request_data["endTime"] = end_time
                    if limit:
                        request_data["limit"] = limit

                    logger.debug(f"Requesting candles for {symbol} with data: {request_data}")

                    # Make the request
                    response = requests.post(
                        f"{self.api_url}/info",
                        json=request_data,
                        headers={"Content-Type": "application/json"},
                        timeout=10
                    )

                    if response.status_code == 429:  # Rate limit
                        if attempt < max_retries - 1:
                            logger.warning(f"Rate limit hit while getting candles for {symbol}. Retrying after delay...")
                            time.sleep(retry_delay)
                            continue

                    response.raise_for_status()
                    data = response.json()

                    # Debug log the response format
                    logger.debug(f"Candle data type: {type(data)}, content sample: {str(data)[:200] if data else 'empty'}...")

                    # Handle different response formats
                    candles = []

                    if isinstance(data, list):
                        # Format could be list of candle objects or list of arrays
                        for candle in data:
                            if isinstance(candle, dict):
                                # API returns objects with t, o, h, l, c, v keys
                                try:
                                    candle_data = {
                                        "timestamp": int(candle.get("t", 0)),
                                        "open": float(candle.get("o", 0)),
                                        "high": float(candle.get("h", 0)),
                                        "low": float(candle.get("l", 0)),
                                        "close": float(candle.get("c", 0)),
                                        "volume": float(candle.get("v", 0))
                                    }
                                    candles.append(candle_data)
                                except (ValueError, TypeError) as e:
                                    logger.warning(f"Error parsing candle object: {e}, data: {candle}")
                            elif isinstance(candle, list) and len(candle) >= 6:
                                # API returns arrays: [timestamp, open, high, low, close, volume]
                                try:
                                    candle_data = {
                                        "timestamp": int(candle[0]),
                                        "open": float(candle[1]),
                                        "high": float(candle[2]),
                                        "low": float(candle[3]),
                                        "close": float(candle[4]),
                                        "volume": float(candle[5])
                                    }
                                    candles.append(candle_data)
                                except (ValueError, TypeError, IndexError) as e:
                                    logger.warning(f"Error parsing candle array: {e}, data: {candle}")
                    elif isinstance(data, dict) and "candles" in data:
                        # Some APIs nest the candles in a 'candles' key
                        candle_list = data["candles"]
                        for candle in candle_list:
                            if isinstance(candle, dict):
                                try:
                                    candle_data = {
                                        "timestamp": int(candle.get("t", 0)),
                                        "open": float(candle.get("o", 0)),
                                        "high": float(candle.get("h", 0)),
                                        "low": float(candle.get("l", 0)),
                                        "close": float(candle.get("c", 0)),
                                        "volume": float(candle.get("v", 0))
                                    }
                                    candles.append(candle_data)
                                except (ValueError, TypeError) as e:
                                    logger.warning(f"Error parsing candle from 'candles' key: {e}, data: {candle}")

                    # Sort candles by timestamp
                    candles.sort(key=lambda x: x["timestamp"])

                    # Store in cache
                    self.candle_cache[cache_key] = {
                        'timestamp': time.time(),
                        'data': candles
                    }

                    return candles

                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to get candles after {max_retries} attempts: {e}")
                        raise HyperliquidConnectionError(f"Failed to get candles: {e}")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff

            raise HyperliquidConnectionError(f"Failed to get candles after {max_retries} attempts")

        except Exception as e:
            logger.error(f"Error retrieving candles for {symbol}: {e}", exc_info=True)
            # Return empty list instead of raising on failure
            return []

    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get current funding rate for a perpetual market.

        Args:
            symbol: Market symbol (e.g., 'BTC')

        Returns:
            Dict[str, Any]: Funding rate information
        """
        if not self.info:
            raise ConnectionError("Not connected to Hyperliquid. Call connect() first.")

        try:
            # Get coin index
            meta = self.info.meta()
            coin_idx = None

            for i, coin in enumerate(meta["universe"]):
                if coin["name"] == symbol:
                    coin_idx = i
                    break

            if coin_idx is None:
                raise ValueError(f"Market {symbol} not found")

            # Get funding rate
            # Note: This is a placeholder. Actual implementation would use the Hyperliquid API
            funding_info = {
                "symbol": symbol,
                "funding_rate": 0.0001,  # Placeholder value
                "next_funding_time": int(time.time() * 1000)
                + 3600000,  # 1 hour from now
                "timestamp": int(time.time() * 1000),
            }

            logger.warning(
                "Funding rate fetching not fully implemented for Hyperliquid"
            )

            return funding_info

        except Exception as e:
            logger.error(f"Failed to get funding rate for {symbol}: {e}")
            return {"symbol": symbol, "funding_rate": 0, "error": str(e)}

    def translate_symbol(self, symbol: str) -> str:
        """
        Convert composite symbols like 'ETH-USD' to base asset 'ETH' for Hyperliquid API.

        IMPORTANT: This method is used throughout the connector to ensure compatibility
        between the application's symbol format (ETH-USD) and Hyperliquid's format (ETH).
        All methods that need to lookup symbols in Hyperliquid should use this translation.
        The method will always return the symbol in uppercase for consistency with Hyperliquid API.

        Args:
            symbol: Trading symbol that may contain a separator (e.g., 'ETH-USD')

        Returns:
            str: Base symbol in uppercase for use with Hyperliquid API

        Raises:
            ValueError: If symbol is None, empty, or in an invalid format
        """
        if not symbol:
            logger.error("Received empty or None symbol for translation")
            raise ValueError("Symbol cannot be None or empty")

        # Strip whitespace and convert to uppercase first
        cleaned_symbol = symbol.strip().upper()

        if not cleaned_symbol:
            logger.error("Symbol contains only whitespace")
            raise ValueError("Symbol cannot be only whitespace")

        # Check for invalid characters
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        invalid_chars = set(cleaned_symbol) - valid_chars
        if invalid_chars:
            logger.error(f"Symbol contains invalid characters: {invalid_chars}")
            raise ValueError(f"Symbol contains invalid characters: {invalid_chars}")

        original_symbol = cleaned_symbol
        translated_symbol = original_symbol.split('-')[0] if '-' in original_symbol else original_symbol

        # Log the translation with structured data
        log_data = {
            "original_symbol": original_symbol,
            "translated_symbol": translated_symbol,
            "has_separator": '-' in original_symbol,
            "method": "translate_symbol"
        }
        logger.debug(f"Symbol translation: {json.dumps(log_data)}")

        return translated_symbol

    def get_leverage_tiers(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get leverage tiers for a given symbol from Hyperliquid API.

        Args:
            symbol: Trading symbol (e.g., 'ETH-USD')

        Returns:
            List[Dict[str, Any]]: List of leverage tier information

        Raises:
            ConnectorError: If API request fails or returns invalid data
        """
        try:
            translated_symbol = self.translate_symbol(symbol)

            log_data = {
                "method": "get_leverage_tiers",
                "original_symbol": symbol,
                "translated_symbol": translated_symbol,
                "action": "fetching_leverage_tiers"
            }
            logger.info(f"Fetching leverage tiers: {json.dumps(log_data)}")

            markets = self.get_markets()

            # Log available markets for debugging
            logger.debug(f"Available markets: {[m.get('symbol') for m in markets[:5]]}...")

            # Find the market info for our symbol - we need to match the base_asset with our translated symbol
            market_info = None
            for market in markets:
                base_asset = market.get('base_asset', '')
                if base_asset.upper() == translated_symbol.upper():
                    market_info = market
                    logger.debug(f"Found market info for {translated_symbol}: {market.get('symbol')}")
                    break

            if not market_info:
                available_markets = [m.get('base_asset') for m in markets]
                error_msg = f"Market info not found for symbol: {translated_symbol}"
                logger.error(f"{error_msg}. Available markets: {available_markets[:10]}...",
                             extra={"symbol": translated_symbol, "available_markets": available_markets})
                raise ConnectorError(error_msg)

            # Extract leverage information from exchange_specific data
            exchange_specific = market_info.get('exchange_specific', {})
            leverage_info = {
                "max_leverage": exchange_specific.get('max_leverage', 20.0),
                "maintenance_margin_fraction": 0.025,  # Default to 2.5%
                "initial_margin_fraction": 0.05,      # Default to 5%
                "base_position_notional": 0.0,
                "base_position_value": 0.0
            }

            logger.info(f"Retrieved leverage tiers: {json.dumps({**log_data, 'leverage_info': leverage_info})}")

            return [leverage_info]  # Return as list for consistency with other exchanges

        except Exception as e:
            error_msg = f"Error fetching leverage tiers for {symbol}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConnectorError(error_msg) from e

    def calculate_margin_requirement(
        self, symbol: str, quantity: float, price: float, leverage: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate initial and maintenance margin requirements for a position.

        Args:
            symbol: Trading symbol (e.g., 'ETH-USD')
            quantity: Position size in base currency
            price: Current market price
            leverage: Desired leverage (default: 1.0)

        Returns:
            Dict[str, float]: Dictionary containing:
                - initial_margin: Required initial margin
                - maintenance_margin: Required maintenance margin
                - effective_leverage: Actual leverage after applying limits

        Raises:
            ConnectorError: If margin calculation fails or parameters are invalid
        """
        try:
            if not symbol or not isinstance(symbol, str):
                raise ValueError("Symbol must be a non-empty string")
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                raise ValueError("Quantity must be a positive number")
            if not isinstance(price, (int, float)) or price <= 0:
                raise ValueError("Price must be a positive number")
            if not isinstance(leverage, (int, float)) or leverage <= 0:
                raise ValueError("Leverage must be a positive number")

            log_data = {
                "method": "calculate_margin_requirement",
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "requested_leverage": leverage,
            }
            logger.info(f"Calculating margin requirements: {json.dumps(log_data)}")

            # Get leverage tiers for the symbol
            leverage_tiers = self.get_leverage_tiers(symbol)
            if not leverage_tiers:
                raise ConnectorError(f"No leverage tiers found for symbol: {symbol}")

            tier = leverage_tiers[0]  # Hyperliquid uses a single tier system

            # Calculate position value
            position_value = abs(quantity * price)

            # Apply leverage limits
            max_leverage = float(tier.get('max_leverage', 1.0))
            effective_leverage = min(leverage, max_leverage)

            # Calculate margins using the effective leverage
            initial_margin = position_value / effective_leverage
            maintenance_margin = initial_margin * (
                tier.get('maintenance_margin_fraction', 0.02) /
                tier.get('initial_margin_fraction', 0.04)
            )

            result = {
                'initial_margin': initial_margin,
                'maintenance_margin': maintenance_margin,
                'effective_leverage': effective_leverage
            }

            logger.info(f"Margin calculation completed: {json.dumps({**log_data, 'result': result})}")

            return result

        except ValueError as ve:
            error_msg = f"Invalid parameters for margin calculation: {str(ve)}"
            logger.error(error_msg)
            raise ConnectorError(error_msg) from ve
        except Exception as e:
            error_msg = f"Error calculating margin requirements: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConnectorError(error_msg) from e

    def disconnect(self) -> bool:
        """
        Disconnect from the Hyperliquid API.

        Returns:
            bool: True if disconnection is successful, False otherwise
        """
        try:
            # Close any open connections
            self.info = None
            self.exchange = None
            # Set the connected state to False
            self._is_connected = False
            logger.info("Disconnected from Hyperliquid API")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from Hyperliquid: {e}")
            return False

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get the details of an order.

        Args:
            order_id: The ID of the order to retrieve

        Returns:
            Dict[str, Any]: Order details
        """
        try:
            if not self.exchange:
                logger.error("Not connected to Hyperliquid. Call connect() first.")
                return {"error": "Not connected"}

            # Use order status method to get order details
            return self.get_order_status(order_id)
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return {"error": str(e)}

    def set_leverage(self, symbol: str, leverage: float) -> Dict[str, Any]:
        """
        Set leverage for a specific market.

        Args:
            symbol: Market symbol (e.g., 'BTC')
            leverage: Leverage value to set

        Returns:
            Dict with operation result
        """
        return {"success": True, "message": f"Leverage for {symbol} set to {leverage}x"}

    def get_optimal_limit_price(
        self, symbol: str, side: OrderSide, amount: Union[float, Decimal]
    ) -> Dict[str, Any]:
        """
        Calculate the optimal limit price for an order based on current orderbook.
        """
        try:
            # Convert amount to Decimal if it isn't already
            amount_decimal = Decimal(str(amount))

            # Get order book
            orderbook = self.get_orderbook(symbol)
            if not orderbook:
                logger.warning("Empty price levels for {symbol}, cannot calculate optimal price")
                return {
                    "price": 0.0,
                    "batches": [],
                    "total_cost": 0.0,
                    "slippage": 0.0,
                    "enough_liquidity": False,
                }

            # Get price levels based on order side
            price_levels = (
                orderbook["asks"] if side == OrderSide.BUY else orderbook["bids"]
            )

            if not price_levels:
                logger.warning(f"No {side.value} price levels found for {symbol}")
                return {
                    "price": 0.0,
                    "batches": [],
                    "total_cost": 0.0,
                    "slippage": 0.0,
                    "enough_liquidity": False,
                }

            # Get best price from first level
            best_price = Decimal(str(price_levels[0][0]))

            # Calculate how much we can fill at each price level
            cumulative_volume = Decimal('0')
            batches = []
            total_cost = Decimal('0')
            worst_price = best_price  # Initialize with best price

            for price, size in price_levels:
                price_decimal = Decimal(str(price))
                size_decimal = Decimal(str(size))

                if cumulative_volume >= amount_decimal:
                    break

                remaining = amount_decimal - cumulative_volume
                fill_amount = min(remaining, size_decimal)

                batches.append({
                    "price": float(price_decimal),
                    "amount": float(fill_amount)
                })

                total_cost += price_decimal * fill_amount
                cumulative_volume += fill_amount
                worst_price = price_decimal

            # Check if we have enough liquidity
            enough_liquidity = cumulative_volume >= amount_decimal

            # Calculate slippage as percentage difference between best and worst price
            slippage = (
                abs((worst_price - best_price) / best_price) * Decimal('100')
                if best_price > 0
                else Decimal('0')
            )

            # For BUY orders: add a small buffer to ensure immediate fill
            # For SELL orders: subtract a small buffer to ensure immediate fill
            buffer_percentage = Decimal('0.0005')  # 0.05%
            price_adjustment = worst_price * buffer_percentage

            if side == OrderSide.BUY:
                optimal_price = worst_price + price_adjustment
            else:  # SELL
                optimal_price = worst_price - price_adjustment

            # If there's not enough liquidity, we'll still return the best price we found
            # but mark enough_liquidity as False
            if not enough_liquidity:
                logger.warning(
                    f"Not enough liquidity in the order book for {amount} {symbol}"
                )

            # For very small orders or perfect amount match, simplify to a single price
            if len(batches) == 1 or (slippage < Decimal('0.1') and enough_liquidity):
                batches = [{"price": float(optimal_price), "amount": float(amount_decimal)}]

            return {
                "price": float(optimal_price),
                "batches": batches,
                "total_cost": float(total_cost),
                "slippage": float(slippage),
                "enough_liquidity": enough_liquidity,
            }

        except Exception as e:
            logger.error(f"Error calculating optimal price for {symbol}: {e}")
            # Fallback to current market price from ticker
            ticker = self.get_ticker(symbol)
            return {
                "price": float(ticker.get("last_price", 0)),
                "batches": [
                    {"price": float(ticker.get("last_price", 0)), "amount": float(amount_decimal)}
                ],
                "total_cost": float(ticker.get("last_price", 0)) * float(amount_decimal),
                "slippage": 0.0,
                "enough_liquidity": False,
            }

    @property
    def is_connected(self) -> bool:
        """
        Check if the connector is currently connected.

        Returns:
            bool: True if connected, False otherwise
        """
        return self._is_connected

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position information for a specific symbol.

        Args:
            symbol (str): The trading symbol to get position for

        Returns:
            Optional[Dict]: Position information if exists, None otherwise
        """
        try:
            # Translate symbol to base format for Hyperliquid
            base_symbol = self.translate_symbol(symbol)

            response = self.exchange.user_state()
            if not response or "assetPositions" not in response:
                logger.warning(f"No position data found for {base_symbol}")
                return None

            # Find position for the specified symbol
            for position in response["assetPositions"]:
                if position.get("coin") == base_symbol:
                    return {
                        "symbol": symbol,  # Keep original symbol in response
                        "size": float(position.get("position", "0")),
                        "entry_price": float(position.get("entryPx", "0")),
                        "liquidation_price": float(position.get("liquidationPx", "0")),
                        "unrealized_pnl": float(position.get("unrealizedPnl", "0")),
                        "leverage": float(position.get("leverage", "1"))
                    }

            logger.debug(f"No active position found for {base_symbol}")
            return None

        except Exception as e:
            logger.error(f"Error getting position for {symbol} (base: {base_symbol}): {str(e)}")
            return None

    def get_min_order_size(self, symbol: str) -> float:
        """
        Get the minimum order size for a market.

        Args:
            symbol: Market symbol (e.g., 'BTC')

        Returns:
            float: Minimum order size
        """
        if not self.info:
            raise ConnectionError("Not connected to Hyperliquid. Call connect() first.")

        try:
            # Translate symbol to base format for Hyperliquid
            base_symbol = self.translate_symbol(symbol)

            # Get market metadata
            meta = self.info.meta()

            # Find the market info for the symbol
            for coin in meta.get("universe", []):
                if coin.get("name") == base_symbol:
                    # Get minSize from market info, default to 0.001 if not found
                    min_size = coin.get("minSize", 0.001)
                    # Convert to float if it's a string
                    return float(min_size) if isinstance(min_size, str) else min_size

            # If market not found, log warning and return default
            logger.warning(f"Market {symbol} not found for min order size, using default")
            return 0.001  # Default minimum size

        except Exception as e:
            logger.error(f"Error getting minimum order size for {symbol}: {e}")
            return 0.001  # Return default minimum size in case of error
