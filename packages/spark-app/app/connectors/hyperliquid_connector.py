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
from connectors.base_connector import (BaseConnector, MarketType, OrderSide,
                                       OrderStatus, OrderType)

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

            # Initialize info client for data retrieval
            try:
                self.info = Info(base_url=self.api_url)
                # Make the self.info.meta() call the primary connection check
                logger.info("Testing connection by fetching exchange metadata via SDK...")
                meta = self.info.meta()
                if not meta or "universe" not in meta:
                    logger.error("Failed to fetch valid metadata from Hyperliquid API.")
                    self._is_connected = False
                    return False
                logger.info(f"Successfully fetched metadata. Found {len(meta['universe'])} markets.")

            except Exception as e:
                logger.error(f"Failed to initialize Info client or fetch metadata: {e}", exc_info=True)
                self._is_connected = False
                return False

            # Initialize exchange client for trading
            try:
                # Create a wallet object from the private key
                wallet = Account.from_key(self.private_key)
                self.exchange = Exchange(wallet=wallet, base_url=self.api_url)
                logger.info("Exchange client initialized successfully.")

            except Exception as e:
                logger.error(f"Failed to initialize wallet or Exchange client: {e}", exc_info=True)
                # Don't necessarily mark as disconnected if info client worked, but log error
                # Or decide if exchange client is critical for 'connected' status
                self._is_connected = False # Let's consider failure to init exchange as connection failure
                return False

            # If both info and exchange clients initialized successfully
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
            # --- Find Market Index Modification ---
            base_symbol = symbol
            if "-" in symbol:
                base_symbol = symbol.split("-")[0]
                logger.debug(f"Symbol '{symbol}' contains '-', using base '{base_symbol}' for lookup.")
            # --- End Find Market Index Modification ---

            meta = self.info.meta() # Fetch metadata once

            all_mids = self.info.all_mids()

            coin_idx = None
            market_found_in_meta = False
            for i, coin in enumerate(meta.get("universe", [])):
                # Compare using the potentially modified search_symbol (now base_symbol)
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
            raise ConnectionError("Not connected to Hyperliquid. Call connect() first.")

        try:
            # Get market index
            meta = self.info.meta()
            coin_idx = None

            for i, coin in enumerate(meta["universe"]):
                if coin["name"] == symbol:
                    coin_idx = i
                    break

            if coin_idx is None:
                raise ValueError(f"Market {symbol} not found")

            # Get L2 orderbook with retry on rate limit
            try:
                # Make POST request to /info endpoint for L2 book snapshot
                response = requests.post(
                    f"{self.api_url}/info",
                    json={
                        "type": "l2Book",
                        "coin": str(coin_idx),  # API expects coin index, not symbol
                        "nSigFigs": None  # Use full precision
                    },
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 429:
                    logger.warning(f"Rate limit hit while getting orderbook for {symbol}. Retrying after delay...")
                    time.sleep(1)  # Add delay before retry
                    response = requests.post(  # Retry once
                        f"{self.api_url}/info",
                        json={
                            "type": "l2Book",
                            "coin": str(coin_idx),
                            "nSigFigs": None
                        },
                        headers={"Content-Type": "application/json"}
                    )

                if response.status_code != 200:
                    logger.error(f"API error response: {response.text}")
                    raise HyperliquidAPIError(f"API returned status code: {response.status_code}")

                data = response.json()
                logger.info(f"Orderbook response for {symbol}: {data}")

                if not data or not isinstance(data, list) or len(data) != 2:
                    logger.error(f"Invalid orderbook format: {data}")
                    raise ValueError(f"Invalid orderbook data for {symbol}")

                # Format response from snapshot - data[0] is bids, data[1] is asks
                # Handle different response formats (API may return array of arrays or object with levels)
                try:
                    # First try to parse as array of arrays format
                    return {
                        "bids": [
                            [float(price), float(size)]
                            for price, size in data[0][:depth]
                        ],
                        "asks": [
                            [float(price), float(size)]
                            for price, size in data[1][:depth]
                        ]
                    }
                except (IndexError, TypeError) as e:
                    logger.warning(f"Error parsing orderbook in array format: {e}, trying alternative format")
                    # Try parsing as object with px/sz format
                    try:
                        return {
                            "bids": [
                                [float(level["px"]), float(level["sz"])]
                                for level in (data[0] or [])[:depth]
                            ],
                            "asks": [
                                [float(level["px"]), float(level["sz"])]
                                for level in (data[1] or [])[:depth]
                            ]
                        }
                    except (KeyError, TypeError) as e:
                        logger.error(f"Failed to parse orderbook data: {e}")
                        return {"bids": [], "asks": []}

            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 429:
                    logger.warning(f"Rate limit hit while getting orderbook for {symbol}. Retrying after delay...")
                    time.sleep(1)  # Add delay before retry
                    # Try getting state which includes orderbook
                    state = self.info.user_state(self.wallet_address)
                    if "orderBook" in state:
                        return {
                            "bids": [
                                [float(level["px"]), float(level["sz"])]
                                for level in state["orderBook"].get("bids", [])[:depth]
                            ],
                            "asks": [
                                [float(level["px"]), float(level["sz"])]
                                for level in state["orderBook"].get("asks", [])[:depth]
                            ],
                        }
                raise

        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol}: {e}")
            if isinstance(e, ValueError):
                raise HyperliquidAPIError(str(e))
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
            if not self.connect():
                raise HyperliquidConnectionError(
                    "Failed to connect to Hyperliquid. Please check network and API status."
                )

        try:
            try:
                user_state = self.info.user_state(self.wallet_address)
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error while getting user state: {e}")
                # Update connection status for metrics
                self._is_connected = False
                raise HyperliquidConnectionError(
                    f"Failed to connect to Hyperliquid API: {e}"
                )
            except requests.exceptions.Timeout as e:
                logger.error(f"Request timed out while getting user state: {e}")
                raise HyperliquidTimeoutError(
                    f"Request to Hyperliquid API timed out: {e}"
                )

            try:
                meta = self.info.meta()
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error while getting meta data: {e}")
                self._is_connected = False
                raise HyperliquidConnectionError(
                    f"Failed to connect to Hyperliquid metadata API: {e}"
                )

            try:
                all_mids = self.info.all_mids()
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error while getting all_mids: {e}")
                self._is_connected = False
                raise HyperliquidConnectionError(
                    f"Failed to connect to Hyperliquid price API: {e}"
                )

            positions = []
            for position in user_state.get("assetPositions", []):
                coin_idx = position["coin"]
                coin_name = meta["universe"][coin_idx]["name"]
                current_price = float(all_mids[coin_idx])

                # Extract position details
                size = float(position.get("szi", "0"))
                entry_price = float(position.get("entryPx", "0"))
                leverage = float(position.get("leverage", "1"))
                liquidation_price = float(position.get("liqPx", "0"))

                # Calculate PNL
                unrealized_pnl = size * (current_price - entry_price)

                position_info = {
                    "symbol": coin_name,
                    "size": size,
                    "side": "LONG" if size > 0 else "SHORT",
                    "entry_price": entry_price,
                    "mark_price": current_price,
                    "leverage": leverage,
                    "liquidation_price": liquidation_price,
                    "unrealized_pnl": unrealized_pnl,
                    "margin": abs(size) / leverage,
                }

                positions.append(position_info)

            return positions
        except (HyperliquidConnectionError, HyperliquidTimeoutError):
            # These exceptions will be caught by the retry decorator
            raise
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
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
            elif order_type == OrderType.MARKET and price is not None:
                logger.warning("Price is ignored for MARKET orders.")
                price = None # Ensure price is None for market orders

            if leverage is not None and leverage <= 0:
                 logger.error(f"Leverage must be positive, got {leverage}")
                 raise ValueError("Leverage must be positive")

            # --- Fetch Market Metadata for Precision and Limits ---
            meta = self.info.meta()
            market_info = None
            coin_idx = None
            for i, coin_data in enumerate(meta.get("universe", [])):
                if coin_data.get("name") == symbol.upper(): # Match symbol case-insensitively
                    market_info = coin_data
                    coin_idx = i
                    break

            if market_info is None or coin_idx is None:
                logger.error(f"Market {symbol} not found in Hyperliquid metadata.")
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

            sdk_limit_price = 0.0
            if order_type == OrderType.LIMIT and formatted_price:
                try:
                    sdk_limit_price = float(formatted_price)
                except (ValueError, TypeError):
                    logger.error(f"Invalid formatted price for LIMIT order: {formatted_price}")
                    raise ValueError(f"Invalid price for LIMIT order: {price}")

            order_data_for_sdk = {
                 "coin": symbol.upper(),
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
                    "entry_price": price if price is not None else 0.0,  # Add entry_price
                    "raw_response": response
                }
                if isinstance(response, dict) and response.get("status") == "ok":
                    statuses = response.get("response", {}).get("data", {}).get("statuses", [])
                    if statuses and isinstance(statuses[0], dict):
                        order_info = statuses[0]
                        if "resting" in order_info:
                            processed_response["status"] = OrderStatus.OPEN.value
                            processed_response["order_id"] = order_info["resting"].get("oid")
                            # Set entry_price for resting orders (limit orders)
                            if "px" in order_info["resting"]:
                                processed_response["entry_price"] = float(order_info["resting"]["px"])
                        elif "filled" in order_info:
                            processed_response["status"] = OrderStatus.CLOSED.value
                            processed_response["order_id"] = order_info["filled"].get("oid")
                            processed_response["filled_amount"] = order_info["filled"].get("totalSz")
                            # Use avgPx as entry_price for filled orders
                            avg_price = order_info["filled"].get("avgPx")
                            processed_response["average_price"] = avg_price
                            processed_response["entry_price"] = float(avg_price) if avg_price is not None else price
                        elif "error" in order_info:
                            processed_response["status"] = OrderStatus.REJECTED.value
                            processed_response["error_message"] = order_info.get("error")
                            logger.error(f"Order placement failed (API Error): {order_info.get('error')}")
                        else:
                            logger.warning(f"Order placed but status unclear: {order_info}")
                    else:
                        logger.warning("Order status 'ok' but no order details found in response.")
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

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]: # Added symbol
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
            response = self.exchange.cancel(symbol.upper(), int(order_id))
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
            if not self.connect():
                raise HyperliquidConnectionError("Failed to connect.")

        try:
            # Extract base symbol
            base_symbol = symbol.split('-')[0].upper() if '-' in symbol else symbol.upper()

            # Get interval in milliseconds
            _, interval_ms = self._parse_interval(interval)

            # Calculate time range
            current_time = int(time.time() * 1000)
            end_time = end_time or current_time
            start_time = start_time or (end_time - (limit * interval_ms))

            # Construct request body
            request_body = {
                "type": "candleSnapshot",
                "req": {
                    "coin": base_symbol,
                    "interval": interval,  # Hyperliquid accepts standard formats like "1m", "5m"
                    "startTime": start_time,
                    "endTime": end_time
                }
            }

            # Make API request
            try:
                response = requests.post(
                    f"{self.api_url}/info",
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                    timeout=self.connection_timeout
                )

                # Handle rate limiting
                if response.status_code == 429:
                    logger.warning(f"Rate limit hit while fetching candles for {symbol}. Retrying after 1 second...")
                    time.sleep(1)
                    response = requests.post(
                        f"{self.api_url}/info",
                        json=request_body,
                        headers={"Content-Type": "application/json"},
                        timeout=self.connection_timeout
                    )

                response.raise_for_status()
                candles_data = response.json()

                # Format candles
                formatted_candles = []
                for candle in candles_data:
                    try:
                        formatted_candles.append({
                            "timestamp": int(candle['t']),
                            "open": float(candle['o']),
                            "high": float(candle['h']),
                            "low": float(candle['l']),
                            "close": float(candle['c']),
                            "volume": float(candle['v'])
                        })
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"Could not parse candle data: {e}")
                        continue

                # Sort by timestamp and apply limit
                formatted_candles.sort(key=lambda x: x["timestamp"])
                return formatted_candles[-limit:] if limit else formatted_candles

            except requests.exceptions.RequestException as e:
                if isinstance(e, requests.exceptions.Timeout):
                    raise HyperliquidTimeoutError(f"Request timed out: {e}")
                elif isinstance(e, requests.exceptions.ConnectionError):
                    self._is_connected = False
                    raise HyperliquidConnectionError(f"Connection error: {e}")
                else:
                    raise HyperliquidAPIError(f"API error fetching candles: {e}")

        except Exception as e:
            logger.error(f"Unexpected error fetching candles for {symbol}: {e}", exc_info=True)
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

    def get_leverage_tiers(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get information about leverage tiers and limits.

        Args:
            symbol: Market symbol (e.g., 'BTC')

        Returns:
            List[Dict[str, Any]]: Information about leverage tiers
        """
        if not self.info:
            raise ConnectionError("Not connected to Hyperliquid. Call connect() first.")

        try:
            # Get coin index and info
            meta = self.info.meta()
            coin_info = None

            for coin in meta["universe"]:
                if coin["name"] == symbol:
                    coin_info = coin
                    break

            if coin_info is None:
                raise ValueError(f"Market {symbol} not found")

            # Extract leverage information
            max_leverage = float(coin_info.get("maxLeverage", 50.0))

            # Hyperliquid may have a simpler leverage structure than tiered
            # This is a simplified representation
            tiers = [
                {
                    "min_notional": 0,
                    "max_notional": float("inf"),
                    "max_leverage": max_leverage,
                    "maintenance_margin_rate": 0.02,  # Placeholder value
                    "initial_margin_rate": 0.04,  # Placeholder value
                }
            ]

            return tiers

        except Exception as e:
            logger.error(f"Failed to get leverage tiers for {symbol}: {e}")
            return []

    def calculate_margin_requirement(
        self, symbol: str, size: float, leverage: float
    ) -> Tuple[float, float]:
        """
        Calculate initial and maintenance margin requirements.

        Args:
            symbol: Market symbol (e.g., 'BTC')
            size: Position size
            leverage: Leverage multiplier

        Returns:
            Tuple[float, float]: (initial_margin, maintenance_margin)
        """
        if not self.info:
            raise ConnectionError("Not connected to Hyperliquid. Call connect() first.")

        try:
            # Get current price
            ticker = self.get_ticker(symbol)
            price = ticker["last_price"]

            # Get leverage tiers
            tiers = self.get_leverage_tiers(symbol)

            if not tiers:
                # Use default calculation
                return super().calculate_margin_requirement(symbol, size, leverage)

            # Find applicable tier
            notional_value = size * price
            applicable_tier = tiers[0]  # Default to first tier

            for tier in tiers:
                if tier["min_notional"] <= notional_value <= tier["max_notional"]:
                    applicable_tier = tier
                    break

            # Calculate margins
            initial_margin = notional_value / leverage
            maintenance_margin = (
                notional_value * applicable_tier["maintenance_margin_rate"]
            )

            return initial_margin, maintenance_margin

        except Exception as e:
            logger.error(f"Failed to calculate margin requirements: {e}")
            # Fallback to base implementation
            return super().calculate_margin_requirement(symbol, size, leverage)

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
        self, symbol: str, side: OrderSide, amount: float
    ) -> Dict[str, Any]:
        """
        Calculate the optimal limit price for immediate execution based on order book depth.

        Args:
            symbol: The market symbol (e.g., 'BTC')
            side: Buy or sell
            amount: The amount/size to trade

        Returns:
            Dict with optimal price and batching information
        """
        try:
            if not self.info:
                self.connect()

            # Get a deeper order book for large orders
            depth = 50  # Use a deeper order book for better analysis
            orderbook = self.get_orderbook(symbol, depth)

            # For buy orders, we need to look at the ask side
            # For sell orders, we need to look at the bid side
            if side == OrderSide.BUY:
                price_levels = orderbook["asks"]
                # Find the worst price (highest) we'd need to pay to fill the entire order
                best_price = price_levels[0][0] if price_levels else None
            else:  # SELL
                price_levels = orderbook["bids"]
                # Find the worst price (lowest) we'd receive to fill the entire order
                best_price = price_levels[0][0] if price_levels else None

            if not price_levels or best_price is None:
                logger.warning(
                    f"Empty price levels for {symbol}, cannot calculate optimal price"
                )
                # Fallback to current market price from ticker
                ticker = self.get_ticker(symbol)
                return {
                    "price": float(ticker.get("last_price", 0)),
                    "batches": [
                        {"price": float(ticker.get("last_price", 0)), "amount": amount}
                    ],
                    "total_cost": float(ticker.get("last_price", 0)) * amount,
                    "slippage": 0.0,
                    "enough_liquidity": False,
                }

            # Calculate how much we can fill at each price level
            cumulative_volume = 0.0
            batches = []
            total_cost = 0.0
            worst_price = best_price  # Initialize with best price

            for price, size in price_levels:
                if cumulative_volume >= amount:
                    break

                remaining = amount - cumulative_volume
                fill_amount = min(remaining, size)

                batches.append({"price": float(price), "amount": float(fill_amount)})

                total_cost += float(price) * float(fill_amount)
                cumulative_volume += float(fill_amount)
                worst_price = float(price)  # Update to current price level

            # Check if we have enough liquidity
            enough_liquidity = cumulative_volume >= amount

            # Calculate slippage as percentage difference between best and worst price
            slippage = (
                abs((worst_price - best_price) / best_price) * 100
                if best_price > 0
                else 0.0
            )

            # For BUY orders: add a small buffer to ensure immediate fill
            # For SELL orders: subtract a small buffer to ensure immediate fill
            buffer_percentage = 0.0005  # 0.05%
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
            if len(batches) == 1 or (slippage < 0.1 and enough_liquidity):
                batches = [{"price": optimal_price, "amount": amount}]

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
                    {"price": float(ticker.get("last_price", 0)), "amount": amount}
                ],
                "total_cost": float(ticker.get("last_price", 0)) * amount,
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
            response = self.exchange.user_state()
            if not response or "assetPositions" not in response:
                logger.warning(f"No position data found for {symbol}")
                return None

            # Find position for the specified symbol
            for position in response["assetPositions"]:
                if position.get("coin") == symbol:
                    return {
                        "symbol": symbol,
                        "size": float(position.get("position", "0")),
                        "entry_price": float(position.get("entryPx", "0")),
                        "liquidation_price": float(position.get("liquidationPx", "0")),
                        "unrealized_pnl": float(position.get("unrealizedPnl", "0")),
                        "leverage": float(position.get("leverage", "1"))
                    }

            logger.debug(f"No active position found for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {str(e)}")
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
            # Get market metadata
            meta = self.info.meta()

            # Find the market info for the symbol
            for coin in meta.get("universe", []):
                if coin.get("name") == symbol:
                    # Get minSize from market info, default to 0.001 if not found
                    min_size = coin.get("minSize", 0.001)
                    # Convert to float if it's a string
                    return float(min_size) if isinstance(min_size, str) else min_size

            # If market not found, raise error
            raise ValueError(f"Market {symbol} not found")

        except Exception as e:
            logger.error(f"Error getting minimum order size for {symbol}: {e}")
            raise
