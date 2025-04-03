# Exchange Connectors

This document provides information about the exchange connectors available in Spark Stacker, their
capabilities, configuration options, and usage examples.

## Overview

Exchange connectors in Spark Stacker provide a standardized interface for interacting with different
cryptocurrency exchanges. Each connector implements the `BaseConnector` interface, ensuring
consistent functionality across exchanges while handling the specific details of each exchange's
API.

## Available Connectors

Currently, Spark Stacker supports the following exchange connectors:

1. **Hyperliquid** - On-chain perpetual futures trading and vaults
2. **Coinbase** - Spot trading with Advanced Trade API
3. **Kraken** - Spot and perpetual futures trading

## Common Interface

All connectors implement these core methods:

- `connect()` - Establish connection to the exchange API
- `disconnect()` - Disconnect from the exchange API
- `get_markets()` - Get available markets/trading pairs
- `get_ticker()` - Get current ticker information
- `get_orderbook()` - Get current order book data
- `get_account_balance()` - Get account balances
- `get_positions()` - Get current open positions
- `place_order()` - Place a new order
- `cancel_order()` - Cancel an existing order
- `get_order_status()` - Get the status of an order
- `get_order()` - Get details of an order
- `close_position()` - Close an open position
- `get_historical_candles()` - Get historical price data
- `set_leverage()` - Set leverage for a symbol (for derivatives)
- `get_funding_rate()` - Get funding rate (for perpetuals)
- `get_leverage_tiers()` - Get leverage tier information
- `get_optimal_limit_price()` - Calculate optimal limit price for an order

## Market Types

Connectors support one or more of the following market types:

- `SPOT` - Spot trading markets
- `PERPETUAL` - Perpetual futures markets
- `FUTURES` - Fixed-expiry futures markets
- `VAULT` - Yield-generating vaults (e.g., Hyperliquid vaults)

## Configuration

Exchange connectors are typically configured via the `.env` file or by providing configuration
objects to the `ConnectorFactory`. Here's how to configure each supported exchange:

### Hyperliquid

```env
# Hyperliquid Configuration
HYPERLIQUID_WALLET_ADDRESS=0x...
HYPERLIQUID_PRIVATE_KEY=your_private_key
HYPERLIQUID_TESTNET=true
HYPERLIQUID_RPC_URL=optional_custom_rpc_url
```

### Coinbase

```env
# Coinbase Configuration
COINBASE_API_KEY=your_api_key
COINBASE_API_SECRET=your_api_secret
COINBASE_PASSPHRASE=your_passphrase
COINBASE_SANDBOX=true
```

### Kraken

```env
# Kraken Configuration
KRAKEN_API_KEY=your_api_key
KRAKEN_API_SECRET=your_api_secret
KRAKEN_TESTNET=true
```

## Creating Connectors

Connectors can be created using the `ConnectorFactory`:

```python
from app.connectors.connector_factory import ConnectorFactory
from app.connectors.base_connector import MarketType

# Create a Kraken connector
kraken_connector = ConnectorFactory.create_connector(
    exchange_type="kraken",
    name="kraken_main",
    api_key="your_api_key",
    api_secret="your_api_secret",
    testnet=True,
    market_types=[MarketType.SPOT, MarketType.PERPETUAL]
)

# Connect to the exchange
if kraken_connector.connect():
    print("Connected to Kraken!")
else:
    print("Failed to connect to Kraken.")
```

## Connector Specifics

### Hyperliquid Connector

Hyperliquid is an on-chain perpetual futures exchange that uses blockchain transactions for trading.
It requires a wallet address and private key for authentication.

**Market Types:** `PERPETUAL`, `VAULT`

**Features:**

- On-chain perpetual futures trading
- Up to 50x leverage
- Transparent on-chain order book and settlement
- Robust error handling with automatic retries
- WebSocket support for real-time data
- Support for vault deposits (future feature)

**Error Handling:**

The Hyperliquid connector implements robust error handling with these custom exceptions:

- `HyperliquidConnectionError` - Raised when connection to API fails
- `HyperliquidTimeoutError` - Raised when a request times out
- `HyperliquidAPIError` - Raised when the API returns an error response

API calls are automatically retried with exponential backoff using the `@retry_api_call` decorator.

### Coinbase Connector

Coinbase connector uses the Advanced Trade API to provide access to Coinbase's spot markets.

**Market Types:** `SPOT`

**Features:**

- Spot trading on all Coinbase markets
- Market and limit orders
- Real-time order book data
- Historical price data

### Kraken Connector

Kraken connector supports both spot markets via the standard Kraken API and futures markets via the
Kraken Futures API.

**Market Types:** `SPOT`, `PERPETUAL`

**Features:**

- Spot trading on all Kraken markets
- Perpetual futures trading on Kraken Futures
- Market and limit orders
- Advanced order types
- Margin trading with adjustable leverage
- Real-time order book data
- Historical price data

**Symbol Handling:** Kraken uses special prefixes for some assets (XBT instead of BTC, XXBT instead
of BTC, ZUSD instead of USD). The connector handles these conversions internally, so you can use
standard symbols like "BTC" and "ETH" in your code.

**API Limitations:**

- Rate limits apply to API requests (Tier 3: 20 calls per second)
- Some endpoints require API key permissions to be set correctly in your Kraken account

## Usage Examples

### Basic Order Placement

```python
from app.connectors.base_connector import OrderSide, OrderType

# Place a market buy order
order = connector.place_order(
    symbol="BTC",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    amount=0.01,  # Buy 0.01 BTC
)

print(f"Order placed with ID: {order['order_id']}")
```

### Working with Order Books

```python
# Get order book data
orderbook = connector.get_orderbook("ETH", depth=10)

# Access best bid and ask
best_bid = orderbook["bids"][0][0] if orderbook["bids"] else None
best_ask = orderbook["asks"][0][0] if orderbook["asks"] else None

print(f"Best bid: {best_bid}, Best ask: {best_ask}")
```

### Setting Leverage (Futures/Perpetuals)

```python
# Set leverage for BTC (only works on exchanges supporting leveraged trading)
result = connector.set_leverage("BTC", 5.0)  # 5x leverage

if result["success"]:
    print(f"Leverage set to {result['leverage']}x for {result['symbol']}")
else:
    print(f"Failed to set leverage: {result['message']}")
```

### Getting Optimal Limit Price

```python
# Calculate optimal limit price for a large order
price_info = connector.get_optimal_limit_price(
    symbol="BTC",
    side=OrderSide.BUY,
    amount=1.0  # 1 BTC (large order)
)

if price_info["enough_liquidity"]:
    print(f"Recommended limit price: {price_info['price']}")
    print(f"Estimated slippage: {price_info['slippage'] * 100:.2f}%")

    # Check if order should be split into batches
    if price_info["batches"]:
        print("Large order detected, consider splitting into these batches:")
        for i, batch in enumerate(price_info["batches"], 1):
            print(f"Batch {i}: {batch['size']} BTC at {batch['price']}")
else:
    print(f"Insufficient liquidity: {price_info['message']}")
```

### Error Handling with Retries

The connectors implement automatic retries for transient errors. You can also implement your own
retry logic:

```python
import time
from app.connectors.hyperliquid_connector import HyperliquidConnectionError, HyperliquidTimeoutError

# Try an operation with custom retry logic
max_retries = 3
retry_count = 0

while retry_count < max_retries:
    try:
        balance = connector.get_account_balance()
        print(f"Account balance: {balance}")
        break  # Success, exit the loop
    except (HyperliquidConnectionError, HyperliquidTimeoutError) as e:
        retry_count += 1
        wait_time = 2 ** retry_count  # Exponential backoff
        print(f"Error: {e}. Retrying in {wait_time} seconds...")
        time.sleep(wait_time)
        if retry_count == max_retries:
            print("Maximum retries reached. Operation failed.")
```

## Best Practices

1. **Error Handling**: Always check for errors when calling connector methods, as network issues or
   API errors can occur.

2. **Connection Management**: Use the `connect()` and `disconnect()` methods to properly manage
   connections.

3. **Market Type Awareness**: Be aware of which market types each connector supports and use them
   appropriately.

4. **Rate Limiting**: Avoid making too many API calls in a short period to prevent being
   rate-limited by exchanges.

5. **Testnet First**: Always test on testnet before using real funds on mainnet.

## Extending with New Connectors

To add a new exchange connector to Spark Stacker:

1. Create a new class that inherits from `BaseConnector`
2. Implement all required methods
3. Register the connector in `ConnectorFactory`
4. Add appropriate test coverage

See the existing connectors for examples of implementation patterns.
