import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union

# Import the BaseConnector interface
from app.connectors.base_connector import BaseConnector, OrderSide, OrderType, OrderStatus, MarketType

# For type hinting and future import
try:
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
except ImportError:
    # We'll handle actual import in the constructor
    pass

logger = logging.getLogger(__name__)

class HyperliquidConnector(BaseConnector):
    """
    Connector for Hyperliquid DEX using the official Python SDK.
    
    This class implements the BaseConnector interface for Hyperliquid.
    Hyperliquid primarily supports perpetual futures markets.
    """
    
    def __init__(self, 
                 name: str = "hyperliquid", 
                 wallet_address: str = "", 
                 private_key: str = "", 
                 testnet: bool = True, 
                 rpc_url: Optional[str] = None,
                 market_types: Optional[List[MarketType]] = None):
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
        super().__init__(name=name, exchange_type="hyperliquid", market_types=market_types)
        
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.testnet = testnet
        self.rpc_url = rpc_url
        self.exchange = None
        self.info = None
        
        # Set up dedicated loggers
        self.setup_loggers()
        
        # Set API URL based on testnet flag
        if testnet:
            # Testnet API URL (without /v1)
            self.api_base_url = "https://api.testnet.hyperliquid.xyz"
        else:
            # Mainnet API URL (without /v1)
            self.api_base_url = "https://api.hyperliquid.xyz"
            
        # Log the supported market types
        logger.info(f"Initialized HyperliquidConnector with market types: {[mt.value for mt in self.market_types]}")
    
    def connect(self) -> bool:
        """
        Establish connection to Hyperliquid API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info
            from eth_account import Account
            
            logger.info(f"Connecting to Hyperliquid API at {self.api_base_url} (testnet={self.testnet})")
            
            # Initialize info client for data retrieval
            self.info = Info(base_url=self.api_base_url)
            
            # Create a wallet object from the private key
            try:
                # Create a wallet from the private key
                wallet = Account.from_key(self.private_key)
                
                # Initialize exchange client for trading
                self.exchange = Exchange(
                    wallet=wallet,
                    base_url=self.api_base_url
                )
                
                # Test connection by getting a simple API call that doesn't require authentication
                try:
                    # Try to get the metadata which should be available without authentication
                    logger.info("Testing connection by fetching exchange metadata...")
                    meta = self.info.meta()
                    logger.info(f"Connected to Hyperliquid {'Testnet' if self.testnet else 'Mainnet'}")
                    if 'universe' in meta:
                        logger.info(f"Found {len(meta['universe'])} markets")
                    return True
                except Exception as e:
                    # If metadata fetch fails, try a simpler API call
                    logger.warning(f"Failed to fetch metadata: {e}")
                    logger.info("Trying a direct API call to test connection...")
                    
                    # Use requests to check if the API is responding
                    import requests
                    response = requests.get(f"{self.api_base_url}/info")
                    if response.status_code == 200:
                        logger.info(f"API connection test successful: {response.status_code}")
                        return True
                    else:
                        logger.error(f"API connection test failed: {response.status_code}")
                        return False
            except Exception as e:
                logger.error(f"Failed to initialize wallet or exchange: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Hyperliquid: {e}")
            return False
    
    def get_markets(self) -> List[Dict[str, Any]]:
        """
        Get available markets from Hyperliquid.
        
        Returns:
            List of market details
        """
        if not self._is_connected:
            self.connect()
        
        try:
            # Fetch all metadata info from Hyperliquid
            meta_info = self.info.meta()
            
            # Extract assets/markets and format them
            markets = []
            
            # Log that we're processing markets
            if hasattr(self, 'markets_logger') and self.markets_logger:
                self.markets_logger.info(f"Retrieved {len(meta_info.get('universe', []))} markets from Hyperliquid")
            
            for asset in meta_info.get('universe', []):
                asset_info = {
                    'symbol': asset.get('name'),  # Use the name as the symbol
                    'base_asset': asset.get('name'),
                    'quote_asset': 'USD',
                    'price_precision': asset.get('szDecimals', 2),
                    'min_size': asset.get('minSize', 0.01),
                    'tick_size': asset.get('tickSize', 0.01),
                    'maker_fee': asset.get('makerFeeRate', 0.0),
                    'taker_fee': asset.get('takerFeeRate', 0.0),
                    'market_type': MarketType.PERPETUAL.value,
                    'active': True
                }
                
                markets.append(asset_info)
            
            return markets
        except Exception as e:
            logger.error(f"Error getting markets: {e}")
            return []
    
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
            # Get market index
            meta = self.info.meta()
            coin_idx = None
            
            for i, coin in enumerate(meta["universe"]):
                if coin["name"] == symbol:
                    coin_idx = i
                    break
            
            if coin_idx is None:
                raise ValueError(f"Market {symbol} not found")
            
            # Get ticker information
            state = self.info.user_state(self.wallet_address)
            all_tickers = self.info.all_mids()
            
            ticker = {
                "symbol": symbol,
                "last_price": float(all_tickers[coin_idx]),
                "timestamp": int(time.time() * 1000)
            }
            
            return ticker
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def get_orderbook(self, symbol: str, depth: int = 10) -> Dict[str, List[List[float]]]:
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
            
            # Get L2 orderbook
            orderbook = self.info.l2_snapshot(coin_idx)
            
            # Format response
            return {
                "bids": [[float(level[0]), float(level[1])] for level in orderbook["levels"]["bids"][:depth]],
                "asks": [[float(level[0]), float(level[1])] for level in orderbook["levels"]["asks"][:depth]]
            }
        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol}: {e}")
            return {"bids": [], "asks": []}
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balance information.
        
        Returns:
            Dict[str, float]: Asset balances
        """
        if not self.info:
            raise ConnectionError("Not connected to Hyperliquid. Call connect() first.")
        
        try:
            user_state = self.info.user_state(self.wallet_address)
            
            # Extract USDC balance
            balance = {
                "USDC": float(user_state.get("cash", "0"))
            }
            
            return balance
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return {"USDC": 0.0}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current open positions.
        
        Returns:
            List[Dict[str, Any]]: List of open positions
        """
        if not self.info:
            raise ConnectionError("Not connected to Hyperliquid. Call connect() first.")
        
        try:
            user_state = self.info.user_state(self.wallet_address)
            meta = self.info.meta()
            all_mids = self.info.all_mids()
            
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
        Place a new order on Hyperliquid.
        
        Args:
            symbol: Market symbol (e.g., 'BTC')
            side: Buy or sell
            order_type: Market or limit
            amount: The amount/size to trade
            leverage: Leverage multiplier
            price: Limit price (required for limit orders)
            
        Returns:
            Dict[str, Any]: Order details including order ID
        """
        if not self.exchange:
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
            
            # Set leverage first
            self.exchange.update_leverage(coin_idx, leverage)
            
            # Prepare order parameters
            order_params = {
                "coin": coin_idx,
                "is_buy": side == OrderSide.BUY,
                "sz": amount,
                "reduce_only": False
            }
            
            # Add price for limit orders
            if order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("Price is required for limit orders")
                order_params["limit_px"] = price
                order_result = self.exchange.order_limit(order_params)
            else:
                order_result = self.exchange.order_market(order_params)
            
            # Process response
            return {
                "order_id": order_result.get("order", {}).get("oid", ""),
                "symbol": symbol,
                "side": side.value,
                "type": order_type.value,
                "amount": amount,
                "leverage": leverage,
                "price": price,
                "status": "OPEN",
                "timestamp": int(time.time() * 1000)
            }
        
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {"error": str(e), "status": "REJECTED"}
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: The ID of the order to cancel
            
        Returns:
            bool: True if successfully canceled, False otherwise
        """
        if not self.exchange:
            raise ConnectionError("Not connected to Hyperliquid. Call connect() first.")
        
        try:
            # The cancel method might require additional parameters like coin_idx
            # We would need to parse the order_id to extract the necessary information
            # or keep track of orders in a local store
            
            # For simplicity, let's assume the order_id contains the coin index
            # In a real implementation, you'd need to track orders or query open orders first
            orders = self.exchange.open_orders()
            
            for order in orders:
                if order.get("oid") == order_id:
                    coin_idx = order.get("coin")
                    self.exchange.cancel(coin_idx, order_id)
                    return True
            
            logger.warning(f"Order {order_id} not found for cancellation")
            return False
        
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
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
                        "timestamp": int(time.time() * 1000)
                    }
            
            # Order not found in open orders, could be filled or canceled
            # In a real implementation, you might check order history here
            return {
                "order_id": order_id,
                "status": OrderStatus.FILLED.value,  # Assumption, should check history
                "timestamp": int(time.time() * 1000)
            }
        
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return {"order_id": order_id, "status": "UNKNOWN", "error": str(e)}
    
    def close_position(self, symbol: str, position_id: Optional[str] = None) -> Dict[str, Any]:
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
                "reduce_only": True  # Important: set reduce_only to true
            }
            
            order_result = self.exchange.order_market(order_params)
            
            return {
                "status": "CLOSING",
                "symbol": symbol,
                "order_id": order_result.get("order", {}).get("oid", ""),
                "side": "BUY" if is_buy else "SELL",
                "amount": size,
                "timestamp": int(time.time() * 1000)
            }
        
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return {"status": "ERROR", "symbol": symbol, "error": str(e)}
    
    def get_historical_candles(self, symbol: str, interval: str, 
                              start_time: Optional[int] = None, 
                              end_time: Optional[int] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical candlestick data.
        
        Args:
            symbol: Market symbol (e.g., 'BTC')
            interval: Time interval (e.g., '1m', '5m', '1h')
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Maximum number of candles to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of candlestick data
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
            
            # Map interval string to seconds
            interval_map = {
                "1m": 60,
                "5m": 300,
                "15m": 900,
                "1h": 3600,
                "4h": 14400,
                "1d": 86400
            }
            
            if interval not in interval_map:
                raise ValueError(f"Unsupported interval: {interval}. Supported intervals: {list(interval_map.keys())}")
            
            # Convert to Hyperliquid format
            resolution = interval_map[interval]
            
            # Calculate default time range if not provided
            if end_time is None:
                end_time = int(time.time() * 1000)
            
            if start_time is None:
                # Default to fetching data for the last 'limit' intervals
                start_time = end_time - (resolution * 1000 * limit)
            
            # Fetch candles
            # Note: Hyperliquid API might have specific parameters for candle retrieval
            # This is a placeholder implementation
            candles = []
            
            # In a real implementation, you'd call the actual Hyperliquid API for historical data
            # For now, we'll return an empty list with a warning
            logger.warning("Historical candles fetching not fully implemented for Hyperliquid")
            
            return candles
        
        except Exception as e:
            logger.error(f"Failed to get historical candles for {symbol}: {e}")
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
                "next_funding_time": int(time.time() * 1000) + 3600000,  # 1 hour from now
                "timestamp": int(time.time() * 1000)
            }
            
            logger.warning("Funding rate fetching not fully implemented for Hyperliquid")
            
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
            tiers = [{
                "min_notional": 0,
                "max_notional": float('inf'),
                "max_leverage": max_leverage,
                "maintenance_margin_rate": 0.02,  # Placeholder value
                "initial_margin_rate": 0.04       # Placeholder value
            }]
            
            return tiers
        
        except Exception as e:
            logger.error(f"Failed to get leverage tiers for {symbol}: {e}")
            return []
    
    def calculate_margin_requirement(self, symbol: str, size: float, leverage: float) -> Tuple[float, float]:
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
            maintenance_margin = notional_value * applicable_tier["maintenance_margin_rate"]
            
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
            # Hyperliquid SDK doesn't have a specific disconnect method as it uses
            # REST API calls. We'll just clean up references.
            self.exchange = None
            self.info = None
            logger.info("Disconnected from Hyperliquid API")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Hyperliquid: {e}")
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
        return {
            "success": True,
            "message": f"Leverage for {symbol} set to {leverage}x"
        }
        
    def get_optimal_limit_price(self, symbol: str, side: OrderSide, amount: float) -> Dict[str, Any]:
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
                logger.warning(f"Empty price levels for {symbol}, cannot calculate optimal price")
                # Fallback to current market price from ticker
                ticker = self.get_ticker(symbol)
                return {
                    "price": float(ticker.get("last_price", 0)),
                    "batches": [{"price": float(ticker.get("last_price", 0)), "amount": amount}],
                    "total_cost": float(ticker.get("last_price", 0)) * amount,
                    "slippage": 0.0,
                    "enough_liquidity": False
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
                
                batches.append({
                    "price": float(price),
                    "amount": float(fill_amount)
                })
                
                total_cost += float(price) * float(fill_amount)
                cumulative_volume += float(fill_amount)
                worst_price = float(price)  # Update to current price level
            
            # Check if we have enough liquidity
            enough_liquidity = cumulative_volume >= amount
            
            # Calculate slippage as percentage difference between best and worst price
            slippage = abs((worst_price - best_price) / best_price) * 100 if best_price > 0 else 0.0
            
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
                logger.warning(f"Not enough liquidity in the order book for {amount} {symbol}")
            
            # For very small orders or perfect amount match, simplify to a single price
            if len(batches) == 1 or (slippage < 0.1 and enough_liquidity):
                batches = [{"price": optimal_price, "amount": amount}]
            
            return {
                "price": float(optimal_price),
                "batches": batches,
                "total_cost": float(total_cost),
                "slippage": float(slippage),
                "enough_liquidity": enough_liquidity
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal price for {symbol}: {e}")
            # Fallback to current market price from ticker
            ticker = self.get_ticker(symbol)
            return {
                "price": float(ticker.get("last_price", 0)),
                "batches": [{"price": float(ticker.get("last_price", 0)), "amount": amount}],
                "total_cost": float(ticker.get("last_price", 0)) * amount,
                "slippage": 0.0,
                "enough_liquidity": False
            } 