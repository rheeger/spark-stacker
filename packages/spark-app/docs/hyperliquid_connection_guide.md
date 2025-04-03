# Hyperliquid Connection Handling Guide

## Overview

This document provides guidance on handling connection issues with the Hyperliquid API in the Spark Stacker application, as well as information about Hyperliquid's different market types and balance sections.

## Hyperliquid Market Types

Hyperliquid offers three main sections, each with its own balance and market types:

1. **Perpetual Futures (PERP)**: Leveraged trading products for cryptocurrencies
2. **Spot Markets (SPOT)**: Direct token trading without leverage
3. **Vaults (VAULT)**: Yield-generating opportunities

Our connector recognizes all three types and provides specific methods to interact with each section.

## Balance Structure

When using the `get_account_balance()` method, you'll receive balances from all three sections with appropriate prefixes:

- **PERP_USDC**: USDC balance in perps trading
- **PERP_[TOKEN]**: Token position sizes in perpetual markets
- **SPOT_[TOKEN]**: Token balances in spot trading
- **VAULT_[TOKEN]**: Token balances in vaults

For backward compatibility, the total USDC balance from perps section is also returned with the key "USDC".

## Specialized Balance Methods

For more detailed interactions with specific sections, use these methods:

- `get_spot_balances()`: Get detailed spot market balances
- `get_vault_balances()`: Get detailed vault balances with additional metadata like APY

## Common Issues

The most common Hyperliquid connection errors include:

1. **Connection Refused**:

   ```
   HTTPSConnectionPool(host='api.hyperliquid.xyz', port=443): Max retries exceeded with url: /info (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object>: Failed to establish a new connection: [Errno 111] Connection refused'))
   ```

2. **Timeout Errors**:

   ```
   HTTPSConnectionPool(host='api.hyperliquid.xyz', port=443): Read timed out
   ```

3. **API Rate Limits**:

   ```
   429 Too Many Requests
   ```

## Implemented Solutions

The connector now includes:

1. **Automatic Retry Mechanism**: Using a custom retry decorator with exponential backoff
2. **Better Exception Handling**: With custom exceptions for different error types
3. **Connection Status Tracking**: For metrics and monitoring
4. **Multi-Section Support**: Handling all three Hyperliquid sections (perps, spot, and vaults)

## Troubleshooting Connection Issues

When encountering connection issues with Hyperliquid:

### 1. Check API Status

First, verify if the Hyperliquid API is functioning properly:

- Visit [Hyperliquid Status Page](https://status.hyperliquid.xyz) (if available)
- Try simple API calls with cURL:

  ```
  # Check perps API
  curl -v https://api.hyperliquid.xyz/info

  # Check spot API
  curl -v https://api.hyperliquid.xyz/spot/markets

  # Check vault API
  curl -v https://api.hyperliquid.xyz/vault/vaults
  ```

### 2. Check Network Connectivity

- Ensure the application has internet access
- Check if there are any firewall/proxy restrictions
- Verify DNS resolution is working correctly

### 3. Review Application Logs

Look for specific error patterns in the logs:

- `ConnectionError`: Network connectivity issues
- `TimeoutError`: API is slow to respond
- `APIError`: API returned an error response

### 4. Verify API Configuration

- Ensure you're using the correct API endpoint (testnet vs mainnet)
- Verify authentication credentials are correct and haven't expired

## Working with Different Market Types

### Perpetual Futures

Perpetual futures are the primary trading product on Hyperliquid, allowing leveraged trading:

```python
# Get perpetual markets
markets = connector.get_markets()
perp_markets = [m for m in markets if m['market_type'] == 'PERPETUAL']

# Place a trade on a perpetual market
connector.place_order(
    symbol="BTC",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    amount=0.1,
    leverage=5.0
)
```

### Spot Trading

Spot markets allow direct token trading:

```python
# Get spot balances
spot_balances = connector.get_spot_balances()

# Get spot markets
markets = connector.get_markets()
spot_markets = [m for m in markets if m['market_type'] == 'SPOT']
```

### Vaults

Vaults provide yield-generating opportunities:

```python
# Get vault balances with details
vault_balances = connector.get_vault_balances()

# Get vault markets
markets = connector.get_markets()
vault_markets = [m for m in markets if m['market_type'] == 'VAULT']
```

## Extending Error Handling

To add support for additional error types:

1. Add a new exception class in `hyperliquid_connector.py`:

   ```python
   class HyperliquidRateLimitError(Exception):
       """Raised when Hyperliquid API rate limit is exceeded"""
       pass
   ```

2. Update the retry decorator's exception list to include the new exception type:

   ```python
   retry_exceptions = (
       HyperliquidConnectionError,
       HyperliquidTimeoutError,
       HyperliquidRateLimitError,
       ...
   )
   ```

3. Add specific handling for the new error condition in the relevant methods.

## Monitoring Connection Health

The connector tracks:

- Number of connection retries
- Connection status
- API latency

These metrics can be visualized in the monitoring dashboard to quickly identify issues with the Hyperliquid API.

## Additional Resources

- [Hyperliquid API Documentation](https://docs.hyperliquid.xyz/technical-documentation/)
- [Python Requests Documentation](https://docs.python-requests.org/en/latest/)
