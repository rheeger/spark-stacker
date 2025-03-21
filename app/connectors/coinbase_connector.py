import logging
import time
from typing import Dict, Any, List, Optional

# Import the BaseConnector interface
from app.connectors.base_connector import BaseConnector, OrderSide, OrderType, OrderStatus

# Import Coinbase Advanced API client
from coinbase.rest import RESTClient

logger = logging.getLogger(__name__)

class CoinbaseConnector(BaseConnector):
    """
    Connector for Coinbase Exchange using the official Python client.
    
    This class implements the BaseConnector interface for Coinbase.
    """
    
    def __init__(self, api_key: str, api_secret: str, api_passphrase: Optional[str] = None, use_sandbox: bool = True):
        """
        Initialize the Coinbase connector.
        
        Args:
            api_key: The API key for authentication (format: organizations/{org_id}/apiKeys/{key_id})
            api_secret: The API secret for request signing (EC private key)
            api_passphrase: Not used in the Advanced API
            use_sandbox: Whether to use the sandbox environment or production
        """
        # Call the parent class constructor
        super().__init__(name="coinbase", exchange_type="coinbase")
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_sandbox = use_sandbox
        self.client = None
            
        # Map of Coinbase product IDs to our standard symbols
        self.symbol_map = {
            "ETH": "ETH-USD",
            "BTC": "BTC-USD",
            "SOL": "SOL-USD",
            "USDC": "USDC-USD",
            "USDT": "USDT-USD",
            # Add more mappings as needed
        }
        
        # Reverse mapping
        self.reverse_symbol_map = {v: k for k, v in self.symbol_map.items()}
        
    def _get_product_id(self, symbol: str) -> str:
        """Convert our standard symbol to Coinbase product ID."""
        return self.symbol_map.get(symbol, symbol)
    
    def _get_symbol(self, product_id: str) -> str:
        """Convert Coinbase product ID to our standard symbol."""
        return self.reverse_symbol_map.get(product_id, product_id)
    
    def _debug_response(self, response, name: str = "Response"):
        """
        Helper method to debug API responses by logging their structure.
        
        Args:
            response: The API response object
            name: A descriptive name for the response
        """
        try:
            attrs = []
            for attr in dir(response):
                if not attr.startswith('_') and not callable(getattr(response, attr)):
                    value = getattr(response, attr)
                    if hasattr(value, '__dict__'):
                        attrs.append(f"{attr}: [Object]")
                    else:
                        attrs.append(f"{attr}: {value}")
            
            logger.debug(f"{name} structure: {', '.join(attrs)}")
        except Exception as e:
            logger.debug(f"Failed to debug {name}: {e}")
    
    def connect(self) -> bool:
        """
        Establish connection to the Coinbase Advanced API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Create the client with proper configuration
            self.client = RESTClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                verbose=False  # Disable verbose mode for production
            )
            
            # Test connection with a simple request
            try:
                # Use get_accounts instead of get_accounts
                accounts = self.client.get_accounts()
                if accounts:
                    logger.info(f"Successfully connected to Coinbase Advanced API")
                    
                    # Get sample data for debugging
                    try:
                        # Debug get_products response structure
                        products = self.client.get_public_products()
                        if products and hasattr(products, 'products') and products.products:
                            logger.debug(f"Product count: {len(products.products)}")
                            if len(products.products) > 0:
                                self._debug_response(products.products[0], "Product")
                                
                        # Debug best_bid_ask response structure
                        if self.symbol_map:
                            sample_product = next(iter(self.symbol_map.values()))
                            bid_ask = self.client.get_product_book(product_id=sample_product)
                            self._debug_response(bid_ask, "ProductBook")
                    except Exception as debug_error:
                        logger.debug(f"Debug API structure failed: {debug_error}")
                    
                    return True
                return False
            except Exception as connection_error:
                logger.error(f"Failed to connect to Coinbase API during test request: {connection_error}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to create Coinbase API client: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the Coinbase API.
        
        Returns:
            bool: True if successfully disconnected, False otherwise
        """
        try:
            if self.client:
                # Close any underlying session/connection
                if hasattr(self.client, 'session') and self.client.session:
                    self.client.session.close()
                # Reset the client
                self.client = None
                logger.info("Disconnected from Coinbase API")
                return True
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Coinbase API: {e}")
            return False
    
    def get_markets(self) -> List[Dict[str, Any]]:
        """
        Get available markets/trading pairs from Coinbase.
        Uses /api/v3/brokerage/products endpoint.
        
        Returns:
            List[Dict[str, Any]]: List of available markets with their details
        """
        try:
            if not self.client:
                self.connect()
                
            # Use get_public_products() to get the product list
            products_response = self.client.get_public_products()
            result = []
            
            if not products_response or not hasattr(products_response, 'products'):
                logger.error("No products returned from Coinbase API")
                return []
            
            for product in products_response.products:
                # Extract currency pair from product_id (e.g., "BTC-USD" -> "BTC", "USD")
                base_currency = None
                quote_currency = None
                
                if '-' in product.product_id:
                    parts = product.product_id.split('-')
                    base_currency = parts[0]
                    quote_currency = parts[1]
                
                # Safely access attributes with fallbacks
                min_size = 0.0
                max_size = None
                price_increment = 0.0
                
                if hasattr(product, 'base_min_size'):
                    min_size = float(product.base_min_size)
                
                if hasattr(product, 'base_max_size') and product.base_max_size:
                    max_size = float(product.base_max_size)
                
                if hasattr(product, 'quote_increment'):
                    price_increment = float(product.quote_increment)
                
                result.append({
                    "symbol": product.product_id,
                    "base_currency": base_currency,
                    "quote_currency": quote_currency,
                    "min_size": min_size,
                    "max_size": max_size,
                    "price_increment": price_increment,
                    "status": product.status if hasattr(product, 'status') else "unknown"
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching markets from Coinbase: {e}")
            return []
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker data for a symbol.
        
        Args:
            symbol: Market symbol (e.g., 'BTC-USD')
            
        Returns:
            Dictionary with ticker data
        """
        try:
            if not self.client:
                self.connect()
            
            # Ensure symbol format is correct (BTC-USD)
            if '/' in symbol:
                symbol = symbol.replace('/', '-')
            
            # Try to find the product
            product_id = self._get_product_id(symbol)
            
            if not product_id:
                logger.warning(f"Product ID not found for {symbol}, using symbol directly")
                product_id = symbol
            
            # Get ticker data
            ticker = self.client.get_product(product_id=product_id)
            
            # For some reason, the get_product method may not return price data
            # so we need to make a separate call to get the latest price
            price_data = None
            try:
                # Try to get the latest candle for price data
                candles = self.client.get_candles(
                    product_id=product_id,
                    granularity="ONE_MINUTE", 
                    start=None,
                    end=None,
                    limit=1
                )
                
                if candles and len(candles) > 0:
                    # Latest candle should have closing price
                    price_data = candles[0].close
                    logger.info(f"Got price data from candles for {symbol}: {price_data}")
            except Exception as e:
                logger.warning(f"Could not get candle data for {symbol}: {e}")
            
            # As a fallback, try to get product ticker data
            if not price_data:
                try:
                    ticker_data = self.client.get_product_ticker(product_id=product_id)
                    if ticker_data:
                        price_data = float(ticker_data.price)
                        logger.info(f"Got price data from ticker for {symbol}: {price_data}")
                except Exception as e:
                    logger.warning(f"Could not get ticker data for {symbol}: {e}")
            
            # If still no price, use a default for stablecoins
            if not price_data and ('USDC' in symbol or 'USDT' in symbol or 'PYUSD' in symbol):
                price_data = 1.0
                logger.warning(f"Using default price of 1.0 for stablecoin {symbol}")
            
            return {
                "symbol": symbol,
                "last_price": price_data or 0.0,
                "bid": getattr(ticker, "bid", 0.0),
                "ask": getattr(ticker, "ask", 0.0),
                "volume": getattr(ticker, "volume", 0.0),
                "timestamp": int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            
            # For stablecoins, use a default price of 1.0
            if 'USDC' in symbol or 'USDT' in symbol or 'PYUSD' in symbol:
                return {
                    "symbol": symbol,
                    "last_price": 1.0,
                    "bid": 1.0,
                    "ask": 1.0,
                    "volume": 0.0,
                    "timestamp": int(time.time() * 1000)
                }
            
            return {
                "symbol": symbol,
                "last_price": 0.0,
                "bid": 0.0,
                "ask": 0.0,
                "volume": 0.0,
                "timestamp": int(time.time() * 1000)
            }
    
    def get_orderbook(self, symbol: str, depth: int = 10) -> Dict[str, List[List[float]]]:
        """
        Get the current order book for a specific market.
        Uses /api/v3/brokerage/product_book endpoint.
        
        Args:
            symbol: The market symbol (e.g., 'ETH')
            depth: Number of price levels to retrieve
            
        Returns:
            Dict with 'bids' and 'asks' lists of [price, size] pairs
        """
        try:
            if not self.client:
                self.connect()
                
            product_id = self._get_product_id(symbol)
            # Get the order book with specified depth
            book = self.client.get_product_book(product_id=product_id, limit=depth)
            
            bids = []
            asks = []
            
            # Handle different response structures
            if hasattr(book, 'bids'):
                # Direct bids attribute
                for bid in book.bids[:depth]:
                    if hasattr(bid, 'price') and hasattr(bid, 'size'):
                        bids.append([float(bid.price), float(bid.size)])
            elif hasattr(book, 'pricebook') and hasattr(book.pricebook, 'bids'):
                # Nested in pricebook
                for bid in book.pricebook.bids[:depth]:
                    if hasattr(bid, 'price') and hasattr(bid, 'size'):
                        bids.append([float(bid.price), float(bid.size)])
                
            if hasattr(book, 'asks'):
                # Direct asks attribute
                for ask in book.asks[:depth]:
                    if hasattr(ask, 'price') and hasattr(ask, 'size'):
                        asks.append([float(ask.price), float(ask.size)])
            elif hasattr(book, 'pricebook') and hasattr(book.pricebook, 'asks'):
                # Nested in pricebook
                for ask in book.pricebook.asks[:depth]:
                    if hasattr(ask, 'price') and hasattr(ask, 'size'):
                        asks.append([float(ask.price), float(ask.size)])
            
            if not bids and not asks:
                logger.warning(f"Empty or invalid order book response for {symbol}")
            
            return {
                "bids": bids,
                "asks": asks
            }
            
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol} from Coinbase: {e}")
            # Return mock order book data for testing
            mock_price = 2000.0
            return {
                "bids": [[mock_price - i, 1.0] for i in range(depth)],
                "asks": [[mock_price + i, 1.0] for i in range(depth)]
            }
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balances for all assets.
        Uses /api/v3/brokerage/accounts endpoint.
        
        Returns:
            Dict[str, float]: Dictionary mapping asset names to their balances
        """
        try:
            if not self.client:
                logger.error("Client not connected")
                return {}
            
            if self.use_sandbox:
                # Return mock balances for sandbox testing
                return {
                    "USD": 10000.0,
                    "USDC": 10000.0,
                    "BTC": 1.0,
                    "ETH": 10.0
                }
            
            # Get all accounts using get_accounts
            response = self.client.get_accounts()
            
            if not response or not hasattr(response, 'accounts'):
                logger.error("Invalid response from Coinbase accounts endpoint")
                return {}
            
            balances = {}
            for account in response.accounts:
                currency = account.currency
                # Get the available balance
                if hasattr(account, 'available_balance') and hasattr(account.available_balance, 'value'):
                    available = float(account.available_balance.value)
                    balances[currency] = available
            
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get account balances: {e}")
            if self.use_sandbox:
                # Return mock balances on error in sandbox
                return {
                    "USD": 10000.0,
                    "USDC": 10000.0,
                    "BTC": 1.0,
                    "ETH": 10.0
                }
            return {}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current open positions.
        
        Returns:
            List[Dict[str, Any]]: List of position data dictionaries
        """
        try:
            if not self.client:
                logger.error("Client not connected")
                return []
            
            # The Advanced API doesn't have a direct position endpoint
            # For spot trading, we'll use the accounts endpoint to get balances
            response = self.client.get_accounts()
            
            if not response or not hasattr(response, 'accounts'):
                return []
            
            positions = []
            for account in response.accounts:
                # Handle different response formats
                available = 0.0
                
                if hasattr(account, 'available_balance'):
                    if hasattr(account.available_balance, 'value'):
                        available = float(account.available_balance.value)
                    elif isinstance(account.available_balance, dict) and 'value' in account.available_balance:
                        available = float(account.available_balance['value'])
                    else:
                        available = float(account.available_balance)
                elif hasattr(account, 'balance'):
                    if isinstance(account.balance, dict) and 'value' in account.balance:
                        available = float(account.balance['value'])
                    else:
                        available = float(account.balance)
                else:
                    logger.warning(f"Could not find balance for {account.currency}")
                    continue
                
                if available > 0:
                    position = {
                        "symbol": account.currency,
                        "size": available,
                        "entry_price": 0.0,  # Not available in the API
                        "mark_price": 0.0,   # Not available in the API
                        "pnl": 0.0,          # Not available in the API
                        "liquidation_price": 0.0,  # Not available in the API
                        "leverage": 1.0,     # Spot trading is always 1x
                        "collateral": available,
                        "position_value": available
                    }
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def place_order(self, 
                   symbol: str, 
                   side: OrderSide,
                   order_type: OrderType,
                   amount: float,
                   leverage: float,
                   price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a new order on Coinbase.
        Uses /api/v3/brokerage/orders endpoint.
        
        Args:
            symbol: The market symbol (e.g., 'ETH')
            side: Buy or sell
            order_type: Market or limit
            amount: The amount/size to trade
            leverage: Leverage multiplier to use (not applicable for spot)
            price: Limit price (required for limit orders)
            
        Returns:
            Dict[str, Any]: Order details including order ID
        """
        try:
            if not self.client:
                self.connect()
                
            product_id = self._get_product_id(symbol)
            
            # Convert our enum values to Coinbase's expected values
            cb_side = "BUY" if side == OrderSide.BUY else "SELL"
            
            # Generate a unique client order ID
            client_order_id = str(int(time.time() * 1000))
            
            # Place the order based on order type
            if order_type == OrderType.MARKET:
                if side == OrderSide.BUY:
                    # For buy market orders, use quote_size (USD amount)
                    response = self.client.market_order_buy(
                        client_order_id=client_order_id,
                        product_id=product_id,
                        quote_size=str(amount)
                    )
                else:
                    # For sell market orders, use base_size (crypto amount)
                    response = self.client.market_order_sell(
                        client_order_id=client_order_id,
                        product_id=product_id,
                        base_size=str(amount)
                    )
            else:  # LIMIT order
                if price is None:
                    logger.error("Price is required for LIMIT orders but was not provided")
                    return {"error": "Price is required for LIMIT orders"}
                
                # Ensure price is a valid float
                try:
                    price_float = float(price)
                    if price_float <= 0:
                        logger.error(f"Invalid price for LIMIT order: {price_float}")
                        return {"error": f"Invalid price for LIMIT order: {price_float}"}
                    
                    # Format price with proper precision
                    price_str = str(price_float)
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid price format: {price}")
                    return {"error": f"Invalid price format: {price}"}
                    
                try:
                    if side == OrderSide.BUY:
                        response = self.client.limit_order_buy(
                            client_order_id=client_order_id,
                            product_id=product_id,
                            base_size=str(amount),
                            limit_price=price_str
                        )
                    else:
                        response = self.client.limit_order_sell(
                            client_order_id=client_order_id,
                            product_id=product_id,
                            base_size=str(amount),
                            limit_price=price_str
                        )
                except AttributeError:
                    # If limit_order_buy/sell don't exist, try to use create_order with limit type
                    try:
                        order_config = {
                            'client_order_id': client_order_id,
                            'product_id': product_id,
                            'side': cb_side,
                            'order_configuration': {
                                'limit_limit_gtc': {
                                    'base_size': str(amount),
                                    'limit_price': price_str,
                                    'post_only': False
                                }
                            }
                        }
                        logger.info(f"Placing limit order with config: {order_config}")
                        response = self.client.create_order(**order_config)
                    except Exception as create_order_error:
                        logger.error(f"Error using fallback create_order for LIMIT: {str(create_order_error)}")
                        return {"error": f"Error placing LIMIT order via create_order: {str(create_order_error)}"}
                except Exception as e:
                    logger.error(f"Error placing LIMIT order: {str(e)}")
                    return {"error": f"Error placing LIMIT order: {str(e)}"}
            
            if not response or not hasattr(response, 'success_response'):
                error_msg = "Invalid order response from Coinbase"
                if hasattr(response, 'error_response'):
                    error_msg = f"Coinbase API error: {response.error_response}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            order_details = response.success_response
            
            # Extract order_id from the response
            order_id = None
            if hasattr(order_details, 'order_id'):
                order_id = order_details.order_id
            
            return {
                "order_id": order_id,
                "client_order_id": getattr(order_details, 'client_order_id', None),
                "symbol": symbol,
                "side": side.value,
                "type": order_type.value,
                "amount": amount,
                "price": price,
                "status": self._map_order_status(getattr(order_details, 'status', 'open')),
                "timestamp": getattr(order_details, 'created_time', int(time.time() * 1000))
            }
            
        except Exception as e:
            logger.error(f"Error placing order on Coinbase: {e}")
            raise
    
    def _map_order_status(self, coinbase_status: str) -> str:
        """Map Coinbase order status to our standard OrderStatus enum values."""
        status_map = {
            "open": OrderStatus.OPEN.value,
            "pending": OrderStatus.OPEN.value,
            "active": OrderStatus.OPEN.value,
            "done": OrderStatus.FILLED.value,
            "settled": OrderStatus.FILLED.value,
            "canceled": OrderStatus.CANCELED.value,
            "rejected": OrderStatus.REJECTED.value
        }
        return status_map.get(coinbase_status.lower(), OrderStatus.OPEN.value)
    
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
                
            response = self.client.cancel_orders(order_ids=[order_id])
            
            if not response:
                return False
                
            # Check if the order ID is in the succeeded orders
            return order_id in response.success_response.order_ids
            
        except Exception as e:
            logger.error(f"Error canceling order {order_id} on Coinbase: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id: The ID of the order to check
            
        Returns:
            Dict[str, Any]: Order status and details
        """
        return self.get_order(order_id)
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get the details of an order.
        
        Args:
            order_id: The ID of the order to retrieve
            
        Returns:
            Dict[str, Any]: Order details
        """
        try:
            if not self.client:
                self.connect()
                
            response = self.client.get_order(order_id=order_id)
            
            if not response or not hasattr(response, 'order'):
                raise ValueError(f"Invalid response for order {order_id}")
                
            order = response.order
            
            product_id = order.product_id
            symbol = self._get_symbol(product_id)
            
            # Extract order side and type
            order_side = OrderSide.BUY.value if order.side.upper() == "BUY" else OrderSide.SELL.value
            order_type = OrderType.MARKET.value
            if hasattr(order, 'order_configuration'):
                if hasattr(order.order_configuration, 'limit_limit_gtc') or hasattr(order.order_configuration, 'limit_limit_gtd'):
                    order_type = OrderType.LIMIT.value
            
            # Extract order size and price
            size = 0.0
            price = None
            
            if hasattr(order, 'filled_size'):
                size = float(order.filled_size)
            
            if order_type == OrderType.LIMIT.value and hasattr(order.order_configuration, 'limit_limit_gtc'):
                if hasattr(order.order_configuration.limit_limit_gtc, 'limit_price'):
                    price = float(order.order_configuration.limit_limit_gtc.limit_price)
            
            return {
                "order_id": order.order_id,
                "symbol": symbol,
                "side": order_side,
                "type": order_type,
                "amount": size,
                "filled": size,
                "price": price,
                "status": self._map_order_status(order.status),
                "timestamp": order.created_time
            }
            
        except Exception as e:
            logger.error(f"Error fetching order {order_id} from Coinbase: {e}")
            return {"order_id": order_id, "status": OrderStatus.REJECTED.value}
    
    def close_position(self, symbol: str, position_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Close an open position.
        
        Args:
            symbol: The market symbol (e.g., 'ETH')
            position_id: Optional position ID if the exchange requires it
            
        Returns:
            Dict[str, Any]: Result of the close operation
        """
        try:
            if not self.client:
                self.connect()
                
            # For spot trading, we need to determine what position to close
            # This is a simplified approach - in practice you'd need to track your positions
            if position_id:
                # If we have a specific order ID, cancel it if it's open
                self.cancel_order(position_id)
                return {"success": True, "message": f"Order {position_id} canceled"}
            
            # Otherwise, for spot trading, closing a position would involve 
            # selling what you've bought or buying back what you've shorted
            balances = self.get_account_balance()
            
            # Extract the base currency from the symbol (e.g., 'ETH' from 'ETH-USD')
            base_currency = symbol
            if "-" in self._get_product_id(symbol):
                base_currency = self._get_product_id(symbol).split("-")[0]
            
            # Check if we have a balance in this currency
            if base_currency in balances and balances[base_currency] > 0:
                # Sell the entire balance
                product_id = self._get_product_id(symbol)
                order = self.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    amount=balances[base_currency],
                    leverage=1.0
                )
                return {"success": True, "order": order}
            
            return {"success": False, "message": f"No position found for {symbol}"}
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol} on Coinbase: {e}")
            return {"success": False, "error": str(e)}
    
    def get_historical_candles(self, symbol: str, interval: str, 
                              start_time: Optional[int] = None, 
                              end_time: Optional[int] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical candlestick data.
        
        Args:
            symbol: The market symbol (e.g., 'ETH')
            interval: Time interval (e.g., '1m', '5m', '1h')
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Maximum number of candles to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of candlestick data
        """
        try:
            if not self.client:
                self.connect()
                
            # Convert our interval format to Coinbase's granularity (in seconds)
            granularity = self._convert_interval_to_granularity(interval)
            
            # Convert milliseconds to ISO 8601 if provided
            start_iso = None
            end_iso = None
            
            if start_time:
                start_iso = self._milliseconds_to_iso(start_time)
            if end_time:
                end_iso = self._milliseconds_to_iso(end_time)
            
            product_id = self._get_product_id(symbol)
            
            # Fetch the candles using the get_candles method
            candles = self.client.get_public_candles(
                product_id=product_id,
                granularity=granularity,
                start=start_iso,
                end=end_iso
            )
            
            # Format the response
            result = []
            if hasattr(candles, 'candles'):
                for candle in candles.candles[:limit]:
                    result.append({
                        "timestamp": int(candle.start.timestamp() * 1000),  # Convert to milliseconds
                        "open": float(candle.open),
                        "high": float(candle.high),
                        "low": float(candle.low),
                        "close": float(candle.close),
                        "volume": float(candle.volume)
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching historical candles for {symbol} from Coinbase: {e}")
            return []
    
    def _convert_interval_to_granularity(self, interval: str) -> int:
        """Convert interval string to Coinbase granularity in seconds."""
        # Coinbase supports 60, 300, 900, 3600, 21600, 86400 seconds
        interval_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "6h": 21600,
            "1d": 86400
        }
        
        return interval_map.get(interval, 60)  # Default to 1m
    
    def _milliseconds_to_iso(self, timestamp_ms: int) -> str:
        """Convert timestamp in milliseconds to ISO 8601 format."""
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    
    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get current funding rate for a perpetual market.
        
        This is not applicable for spot trading on Coinbase. If using Coinbase futures,
        this would need to be implemented using their futures API.
        
        Args:
            symbol: The market symbol (e.g., 'ETH')
            
        Returns:
            Dict[str, Any]: Funding rate information
        """
        # Coinbase spot trading doesn't have funding rates
        # This would need to be updated for futures
        return {
            "symbol": symbol,
            "funding_rate": 0.0,
            "next_funding_time": None,
            "message": "Funding rates not applicable for Coinbase spot trading"
        }
    
    def get_leverage_tiers(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get information about leverage tiers and limits.
        
        Not applicable for spot trading on Coinbase. For futures,
        this would need specific implementation.
        
        Args:
            symbol: The market symbol (e.g., 'ETH')
            
        Returns:
            List[Dict[str, Any]]: Information about leverage tiers
        """
        # Coinbase spot trading doesn't support leverage
        return [{
            "symbol": symbol,
            "tier": 1,
            "max_leverage": 1.0,
            "min_notional": 0.0,
            "max_notional": float('inf'),
            "maintenance_margin_rate": 0.0,
            "message": "Leverage not applicable for Coinbase spot trading"
        }]
    
    def set_leverage(self, symbol: str, leverage: float) -> Dict[str, Any]:
        """
        Set leverage for a symbol.
        
        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            leverage: Leverage multiplier to set
            
        Returns:
            Dict[str, Any]: Result of setting leverage
        """
        logger.warning("Coinbase Advanced Trading does not support leverage trading")
        return {"error": "Not supported", "message": "Coinbase Advanced Trading does not support leverage trading"}
        
    def cleanup(self) -> bool:
        """
        Clean up resources when shutting down.
        
        This method ensures that all resources are properly released
        when the application is shutting down.
        
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            logger.info("Cleaning up Coinbase connector resources...")
            
            # First try to disconnect if connected
            if self.client:
                self.disconnect()
                
            return True
        except Exception as e:
            logger.error(f"Error during Coinbase connector cleanup: {e}")
            return False
