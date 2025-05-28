# Symbol Conversion Guide - Spark Stacker Trading System

This guide explains how the Spark Stacker system handles symbol conversion between different
exchanges and maintains consistency across the trading pipeline.

## Overview

Different exchanges use different symbol formats for the same trading pairs. The Spark Stacker
system uses a **standardized internal format** and automatically converts symbols when communicating
with specific exchanges.

### Symbol Format Standardization

**Internal Standard Format**: `BASE-QUOTE`

- Examples: `ETH-USD`, `BTC-USD`, `AVAX-USD`, `SOL-USD`
- Always uppercase
- Always uses hyphen separator
- Always includes quote currency (usually USD)

### Exchange-Specific Formats

| Exchange    | Format       | Example   | Internal Format |
| ----------- | ------------ | --------- | --------------- |
| Hyperliquid | `BASE`       | `ETH`     | `ETH-USD`       |
| Coinbase    | `BASE-QUOTE` | `ETH-USD` | `ETH-USD`       |
| Binance     | `BASEUSD`    | `ETHUSD`  | `ETH-USD`       |

## Symbol Conversion Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                  CONFIGURATION LAYER                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Strategy: "market": "ETH-USD"                           │   │
│  │ Indicator: processes data for "ETH-USD"                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ uses standard format
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 SYMBOL CONVERTER                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ convert_symbol_for_exchange(symbol, exchange)           │   │
│  │ • "ETH-USD" + "hyperliquid" → "ETH"                    │   │
│  │ • "ETH-USD" + "coinbase" → "ETH-USD"                   │   │
│  │ • Handles validation and error cases                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ converts per exchange
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                EXCHANGE CONNECTORS                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Hyperliquid  │  │   Coinbase   │  │    Binance   │          │
│  │ Uses: "ETH"  │  │ Uses: "ETH-  │  │ Uses: "ETHUSD│          │
│  │              │  │ USD"         │  │ "            │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Symbol Converter Implementation

### Core Functions

The symbol converter is implemented in `packages/spark-app/app/core/symbol_converter.py`:

```python
def convert_symbol_for_exchange(symbol: str, exchange: str) -> str:
    """
    Convert standard symbol format to exchange-specific format.

    Args:
        symbol: Standard format symbol (e.g., "ETH-USD")
        exchange: Exchange name (e.g., "hyperliquid", "coinbase")

    Returns:
        Exchange-specific symbol format

    Examples:
        convert_symbol_for_exchange("ETH-USD", "hyperliquid") -> "ETH"
        convert_symbol_for_exchange("ETH-USD", "coinbase") -> "ETH-USD"
    """
```

### Exchange-Specific Conversion Rules

#### Hyperliquid

- **Format**: Base currency only (no quote currency)
- **Examples**:
  - `ETH-USD` → `ETH`
  - `BTC-USD` → `BTC`
  - `AVAX-USD` → `AVAX`

```python
def convert_to_hyperliquid(symbol: str) -> str:
    """Convert to Hyperliquid format (base currency only)."""
    if "-USD" in symbol:
        return symbol.replace("-USD", "")
    elif "-" in symbol:
        # Handle other quote currencies
        base = symbol.split("-")[0]
        return base
    return symbol
```

#### Coinbase

- **Format**: Standard BASE-QUOTE format
- **Examples**:
  - `ETH-USD` → `ETH-USD`
  - `BTC-USD` → `BTC-USD`

```python
def convert_to_coinbase(symbol: str) -> str:
    """Convert to Coinbase format (same as standard)."""
    return symbol  # No conversion needed
```

#### Binance

- **Format**: Concatenated base and quote (no separator)
- **Examples**:
  - `ETH-USD` → `ETHUSD`
  - `BTC-USD` → `BTCUSD`

```python
def convert_to_binance(symbol: str) -> str:
    """Convert to Binance format (no separator)."""
    return symbol.replace("-", "")
```

## Usage in Trading Pipeline

### Strategy Manager

The strategy manager uses standard format internally and converts when needed:

```python
class StrategyManager:
    def _prepare_indicator_data(self, market: str, timeframe: str, indicator: BaseIndicator) -> pd.DataFrame:
        """Prepare data for an indicator on a specific market and timeframe."""
        cache_key = f"{market}_{timeframe}"

        # market is in standard format (e.g., "ETH-USD")
        # Convert for exchange-specific data fetching
        exchange_symbol = convert_symbol_for_exchange(market, self.exchange_name)

        # Fetch data using exchange-specific symbol
        historical_data = self._fetch_historical_data(
            symbol=exchange_symbol,  # "ETH" for Hyperliquid, "ETH-USD" for Coinbase
            interval=timeframe,
            limit=self.data_window_size
        )
```

### Trading Engine

The trading engine converts symbols before executing trades:

```python
class TradingEngine:
    async def process_signal(self, signal: Signal) -> bool:
        """Process a trading signal with proper symbol conversion."""
        # Signal contains standard format market symbol
        market = signal.market  # e.g., "ETH-USD"
        exchange_name = signal.exchange  # e.g., "hyperliquid"

        # Convert to exchange-specific format
        exchange_symbol = convert_symbol_for_exchange(market, exchange_name)

        # Execute trade using converted symbol
        return await self._execute_trade(signal, connector, exchange_symbol)
```

### Exchange Connectors

Each connector receives exchange-specific symbols:

```python
class HyperliquidConnector:
    async def get_market_data(self, symbol: str, timeframe: str):
        """
        Fetch market data for symbol.

        Args:
            symbol: Hyperliquid format symbol (e.g., "ETH")
        """
        # symbol is already in Hyperliquid format
        return await self.api.get_candles(symbol, timeframe)

class CoinbaseConnector:
    async def get_market_data(self, symbol: str, timeframe: str):
        """
        Fetch market data for symbol.

        Args:
            symbol: Coinbase format symbol (e.g., "ETH-USD")
        """
        # symbol is already in Coinbase format
        return await self.api.get_candles(symbol, timeframe)
```

## Configuration Best Practices

### Strategy Configuration

Always use standard format in configuration:

```json
{
  "strategies": [
    {
      "name": "eth_momentum_strategy",
      "market": "ETH-USD", // ✅ Standard format
      "exchange": "hyperliquid", // ✅ Exchange specified
      "indicators": ["rsi_4h"]
    },
    {
      "name": "btc_trend_strategy",
      "market": "BTC-USD", // ✅ Standard format
      "exchange": "coinbase", // ✅ Different exchange
      "indicators": ["rsi_4h"]
    }
  ]
}
```

### Indicator Configuration

Indicators don't need to know about exchange formats:

```json
{
  "indicators": [
    {
      "name": "rsi_4h",
      "type": "rsi",
      "timeframe": "4h",
      // ✅ No symbol field needed - provided by strategy
      "parameters": {
        "period": 14
      }
    }
  ]
}
```

## Symbol Validation

### Format Validation

The system validates symbol formats at multiple levels:

```python
def validate_symbol_format(symbol: str) -> bool:
    """Validate symbol is in standard format."""
    if not symbol:
        return False

    if "-" not in symbol:
        logger.warning(f"Symbol '{symbol}' missing quote currency separator")
        return False

    parts = symbol.split("-")
    if len(parts) != 2:
        logger.warning(f"Symbol '{symbol}' has invalid format")
        return False

    base, quote = parts
    if not base or not quote:
        logger.warning(f"Symbol '{symbol}' has empty base or quote")
        return False

    return True
```

### Strategy Validation

Strategies are validated during configuration loading:

```python
class StrategyConfig:
    def __post_init__(self):
        """Validate strategy configuration."""
        if not validate_symbol_format(self.market):
            raise ValueError(f"Invalid market symbol '{self.market}'. Use format 'ETH-USD'")

        if not self.exchange:
            raise ValueError("Exchange field is required")
```

## Error Handling

### Common Symbol Errors

**1. Using Exchange Format in Configuration**

❌ **Incorrect**:

```json
{
  "strategies": [
    {
      "name": "eth_strategy",
      "market": "ETH", // ❌ Hyperliquid format in config
      "exchange": "hyperliquid"
    }
  ]
}
```

✅ **Correct**:

```json
{
  "strategies": [
    {
      "name": "eth_strategy",
      "market": "ETH-USD", // ✅ Standard format in config
      "exchange": "hyperliquid"
    }
  ]
}
```

**2. Treating Indicator Names as Symbols**

❌ **Incorrect**:

```json
{
  "strategies": [
    {
      "name": "rsi_strategy",
      "market": "RSI-4H", // ❌ Indicator name, not market symbol
      "exchange": "hyperliquid"
    }
  ]
}
```

✅ **Correct**:

```json
{
  "strategies": [
    {
      "name": "eth_rsi_strategy",
      "market": "ETH-USD", // ✅ Market symbol
      "indicators": ["rsi_4h"], // ✅ Indicator reference
      "exchange": "hyperliquid"
    }
  ]
}
```

### Error Recovery

The system provides helpful error messages for symbol issues:

```python
def convert_symbol_for_exchange(symbol: str, exchange: str) -> str:
    """Convert symbol with comprehensive error handling."""
    try:
        if not validate_symbol_format(symbol):
            raise ValueError(f"Invalid symbol format: '{symbol}'. Use 'BASE-QUOTE' format.")

        if exchange == "hyperliquid":
            return convert_to_hyperliquid(symbol)
        elif exchange == "coinbase":
            return convert_to_coinbase(symbol)
        else:
            logger.warning(f"Unknown exchange '{exchange}', using symbol as-is")
            return symbol

    except Exception as e:
        logger.error(f"Symbol conversion failed: {symbol} -> {exchange}: {e}")
        raise
```

## Testing Symbol Conversion

### Unit Tests

Test symbol conversion with various scenarios:

```python
def test_symbol_conversion():
    """Test symbol conversion for all supported exchanges."""

    # Test Hyperliquid conversion
    assert convert_symbol_for_exchange("ETH-USD", "hyperliquid") == "ETH"
    assert convert_symbol_for_exchange("BTC-USD", "hyperliquid") == "BTC"

    # Test Coinbase conversion
    assert convert_symbol_for_exchange("ETH-USD", "coinbase") == "ETH-USD"
    assert convert_symbol_for_exchange("BTC-USD", "coinbase") == "BTC-USD"

    # Test error cases
    with pytest.raises(ValueError):
        convert_symbol_for_exchange("ETH", "hyperliquid")  # Missing quote

    with pytest.raises(ValueError):
        convert_symbol_for_exchange("", "hyperliquid")  # Empty symbol
```

### Integration Tests

Test symbol conversion in the complete pipeline:

```python
def test_strategy_symbol_conversion():
    """Test symbol conversion in strategy execution."""
    config = {
        "strategies": [{
            "name": "test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "indicators": ["rsi_4h"]
        }]
    }

    strategy_manager = StrategyManager(strategies=config["strategies"])

    # Verify internal symbol handling
    assert strategy_manager.strategies[0]["market"] == "ETH-USD"

    # Verify exchange conversion happens during data fetching
    # (would be tested with mocked exchange connector)
```

## Multi-Exchange Trading

### Cross-Exchange Strategies

Trade the same market on different exchanges:

```json
{
  "strategies": [
    {
      "name": "eth_hyperliquid_long",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "indicators": ["rsi_4h"]
    },
    {
      "name": "eth_coinbase_hedge",
      "market": "ETH-USD",
      "exchange": "coinbase",
      "indicators": ["rsi_4h"]
    }
  ]
}
```

### Arbitrage Strategies

Exploit price differences between exchanges:

```json
{
  "strategies": [
    {
      "name": "eth_arbitrage_strategy",
      "market": "ETH-USD",
      "exchange": "arbitrage",
      "indicators": ["eth_price_diff_monitor"],
      "arbitrage_config": {
        "primary_exchange": "hyperliquid",
        "secondary_exchange": "coinbase",
        "min_spread_pct": 0.1
      }
    }
  ]
}
```

## Best Practices

### Configuration Guidelines

1. **Always use standard format** in configuration files
2. **Specify exchange explicitly** for each strategy
3. **Let the system handle conversion** automatically
4. **Don't hardcode exchange formats** in configuration
5. **Use descriptive strategy names** that include market info

### Development Guidelines

1. **Use symbol converter functions** instead of manual conversion
2. **Validate symbols** at configuration load time
3. **Handle conversion errors** gracefully with logging
4. **Test symbol conversion** for all supported exchanges
5. **Document exchange-specific requirements** clearly

### Monitoring Guidelines

1. **Log symbol conversions** for debugging
2. **Monitor for conversion errors** in production
3. **Track exchange-specific symbol usage** in metrics
4. **Alert on unsupported symbol formats**

## Adding New Exchanges

### Implementation Steps

To add a new exchange, implement conversion logic:

```python
def convert_to_new_exchange(symbol: str) -> str:
    """Convert to NewExchange format."""
    # Implement exchange-specific conversion logic
    pass

def convert_symbol_for_exchange(symbol: str, exchange: str) -> str:
    """Updated conversion function with new exchange."""
    if exchange == "hyperliquid":
        return convert_to_hyperliquid(symbol)
    elif exchange == "coinbase":
        return convert_to_coinbase(symbol)
    elif exchange == "new_exchange":
        return convert_to_new_exchange(symbol)
    else:
        logger.warning(f"Unknown exchange '{exchange}', using symbol as-is")
        return symbol
```

### Testing New Exchanges

Add comprehensive tests for new exchange formats:

```python
def test_new_exchange_conversion():
    """Test symbol conversion for new exchange."""
    assert convert_symbol_for_exchange("ETH-USD", "new_exchange") == "expected_format"
    assert convert_symbol_for_exchange("BTC-USD", "new_exchange") == "expected_format"
```

## Troubleshooting

### Common Issues

**Issue**: "Market ETH not found on exchange" **Cause**: Exchange connector receiving standard
format instead of exchange format **Solution**: Verify symbol conversion is being called before
exchange API calls

**Issue**: "Invalid symbol format in configuration" **Cause**: Using exchange-specific format in
config instead of standard format **Solution**: Update configuration to use standard format (e.g.,
"ETH-USD" not "ETH")

**Issue**: "Symbol conversion failed for unknown exchange" **Cause**: Strategy specifies unsupported
exchange name **Solution**: Add exchange support or verify exchange name spelling

### Debugging

Enable detailed symbol conversion logging:

```python
import logging
logging.getLogger("app.core.symbol_converter").setLevel(logging.DEBUG)
```

Check symbol conversion in logs:

```bash
grep "Symbol conversion" packages/spark-app/_logs/spark_stacker.log
grep "convert_symbol_for_exchange" packages/spark-app/_logs/spark_stacker.log
```

This symbol conversion guide ensures consistent symbol handling across all exchanges while
maintaining clean separation between configuration, business logic, and exchange-specific
implementations.
