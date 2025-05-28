# Configuration Guide - Spark Stacker Trading System

This guide explains how to properly configure the Spark Stacker trading system, with clear
explanations of how strategies, indicators, markets, timeframes, and **position sizing** work
together.

## Architecture Overview

Understanding the relationship between components is essential for proper configuration:

### Component Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                        STRATEGY                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ market: "ETH-USD"           # What to trade (exchange   │   │
│  │ exchange: "hyperliquid"     # symbol on specific       │   │
│  │ timeframe: "4h"             # exchange)                │   │
│  │ indicators: [               # Default timeframe        │   │
│  │   "trend_4h",               # Which indicators to use  │   │
│  │   "entry_1h"                # (by name reference)      │   │
│  │ ]                                                       │   │
│  │ position_sizing: {...}      # Strategy-specific sizing │   │
│  │ risk_params: {...}          # Position sizing, stops   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
           │                                │
           │ references                     │ references
           ▼                                ▼
┌──────────────────────┐         ┌──────────────────────┐
│     INDICATOR        │         │     INDICATOR        │
│ name: "trend_4h"     │         │ name: "entry_1h"     │
│ type: "rsi"          │         │ type: "macd"         │
│ timeframe: "4h"      │         │ timeframe: "1h"      │
│ parameters: {...}    │         │ parameters: {...}    │
└──────────────────────┘         └──────────────────────┘
           │                                │
           │ fetches data                   │ fetches data
           ▼                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXCHANGE CONNECTOR                           │
│  Fetches market data for: "ETH-USD"                            │
│  Provides timeframes: 1m, 5m, 15m, 1h, 4h, 1d, etc.          │
│  Executes trades on: Hyperliquid/Coinbase/etc.                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Strategies define WHAT to trade**: Market symbol, exchange, which indicators to use, risk
   parameters, and position sizing
2. **Indicators define HOW to analyze**: Algorithm type, timeframe, and specific parameters
3. **Markets use full exchange symbols**: "ETH-USD", "BTC-USD", not abbreviated forms
4. **Timeframes are set at indicator level**: Each indicator specifies its own data timeframe
5. **Strategy-indicator connection**: Strategies reference indicators by name in the `indicators`
   array
6. **Position sizing is strategy-specific**: Each strategy can have its own position sizing method
   and parameters

## Configuration File Structure

### Complete Example

```json
{
  "log_level": "INFO",
  "webhook_enabled": false,
  "polling_interval": 30,
  "dry_run": true,
  "max_parallel_trades": 1,

  "exchanges": [
    {
      "name": "hyperliquid",
      "exchange_type": "hyperliquid",
      "wallet_address": "${WALLET_ADDRESS}",
      "private_key": "${PRIVATE_KEY}",
      "testnet": true,
      "enabled": true,
      "use_as_main": true
    }
  ],

  "position_sizing": {
    "method": "fixed_usd",
    "fixed_amount_usd": 1000,
    "risk_per_trade_pct": 0.02,
    "max_position_size_usd": 5000
  },

  "strategies": [
    {
      "name": "eth_multi_timeframe_strategy",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "enabled": true,
      "timeframe": "4h",
      "indicators": ["trend_daily", "momentum_4h", "entry_1h"],
      "main_leverage": 1.0,
      "stop_loss_pct": 5.0,
      "take_profit_pct": 10.0,
      "max_position_size": 0.1,
      "risk_per_trade_pct": 0.02,
      "position_sizing": {
        "method": "risk_based",
        "risk_per_trade_pct": 0.03,
        "max_position_size_usd": 3000
      }
    },
    {
      "name": "btc_conservative_strategy",
      "market": "BTC-USD",
      "exchange": "hyperliquid",
      "enabled": true,
      "indicators": ["btc_rsi_4h"],
      "position_sizing": {
        "method": "percent_equity",
        "percent_equity": 0.05,
        "max_position_size_usd": 2000
      }
    }
  ],

  "indicators": [
    {
      "name": "trend_daily",
      "type": "ma",
      "enabled": true,
      "timeframe": "1d",
      "parameters": {
        "short_period": 20,
        "long_period": 50,
        "ma_type": "sma"
      }
    },
    {
      "name": "momentum_4h",
      "type": "rsi",
      "enabled": true,
      "timeframe": "4h",
      "parameters": {
        "period": 14,
        "overbought": 70,
        "oversold": 30,
        "signal_period": 1
      }
    },
    {
      "name": "entry_1h",
      "type": "macd",
      "enabled": true,
      "timeframe": "1h",
      "parameters": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
      }
    },
    {
      "name": "btc_rsi_4h",
      "type": "rsi",
      "enabled": true,
      "timeframe": "4h",
      "parameters": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
      }
    }
  ]
}
```

## Strategy Configuration

### Required Fields

| Field        | Type   | Description                    | Example                   |
| ------------ | ------ | ------------------------------ | ------------------------- |
| `name`       | string | Unique strategy identifier     | `"eth_momentum_strategy"` |
| `market`     | string | Full exchange symbol           | `"ETH-USD"`               |
| `exchange`   | string | Exchange to use                | `"hyperliquid"`           |
| `indicators` | array  | List of indicator names to use | `["rsi_4h", "macd_1h"]`   |

### Optional Fields

| Field                | Type    | Description                              | Default |
| -------------------- | ------- | ---------------------------------------- | ------- |
| `enabled`            | boolean | Whether strategy is active               | `true`  |
| `timeframe`          | string  | Default timeframe for strategy           | `"1h"`  |
| `main_leverage`      | number  | Leverage multiplier                      | `1.0`   |
| `stop_loss_pct`      | number  | Stop loss percentage                     | `5.0`   |
| `take_profit_pct`    | number  | Take profit percentage                   | `10.0`  |
| `max_position_size`  | number  | Maximum position size                    | `0.1`   |
| `risk_per_trade_pct` | number  | Risk per trade as % of capital           | `0.02`  |
| `position_sizing`    | object  | Strategy-specific position sizing config | `null`  |

### Position Sizing Field

The `position_sizing` object allows each strategy to override global position sizing defaults:

```json
{
  "position_sizing": {
    "method": "risk_based|fixed_usd|percent_equity",
    "fixed_amount_usd": 1000, // For fixed_usd method
    "risk_per_trade_pct": 0.02, // For risk_based method
    "percent_equity": 0.1, // For percent_equity method
    "max_position_size_usd": 5000 // Universal maximum limit
  }
}
```

### Strategy Examples

**Single Timeframe Strategy:**

```json
{
  "name": "btc_4h_rsi",
  "market": "BTC-USD",
  "exchange": "hyperliquid",
  "timeframe": "4h",
  "indicators": ["rsi_4h"],
  "main_leverage": 2.0
}
```

**Multi-Timeframe Strategy:**

```json
{
  "name": "eth_comprehensive",
  "market": "ETH-USD",
  "exchange": "hyperliquid",
  "indicators": [
    "trend_daily", // 1d timeframe
    "momentum_4h", // 4h timeframe
    "entry_1h" // 1h timeframe
  ],
  "main_leverage": 1.5
}
```

**Multi-Market Portfolio:**

```json
{
  "strategies": [
    {
      "name": "eth_strategy",
      "market": "ETH-USD",
      "indicators": ["rsi_4h"]
    },
    {
      "name": "btc_strategy",
      "market": "BTC-USD",
      "indicators": ["macd_1h"]
    },
    {
      "name": "avax_strategy",
      "market": "AVAX-USD",
      "indicators": ["bb_4h"]
    }
  ]
}
```

## Indicator Configuration

### Required Fields

| Field       | Type   | Description                 | Example                                  |
| ----------- | ------ | --------------------------- | ---------------------------------------- |
| `name`      | string | Unique indicator identifier | `"rsi_4h"`                               |
| `type`      | string | Algorithm type              | `"rsi"`, `"macd"`, `"bollinger"`, `"ma"` |
| `timeframe` | string | Data timeframe              | `"4h"`, `"1h"`, `"1d"`                   |

### Optional Fields

| Field        | Type    | Description                 | Default        |
| ------------ | ------- | --------------------------- | -------------- |
| `enabled`    | boolean | Whether indicator is active | `true`         |
| `parameters` | object  | Algorithm-specific settings | varies by type |

### Supported Timeframes

| Category    | Timeframes                     | Use Cases                         |
| ----------- | ------------------------------ | --------------------------------- |
| **Minutes** | `1m`, `3m`, `5m`, `15m`, `30m` | Scalping, high-frequency          |
| **Hours**   | `1h`, `2h`, `4h`, `6h`, `12h`  | Day trading, swing trading        |
| **Days**    | `1d`, `3d`                     | Position trading, trend following |
| **Weeks**   | `1w`                           | Long-term investment              |

### Indicator Types & Parameters

#### RSI (Relative Strength Index)

```json
{
  "name": "rsi_4h",
  "type": "rsi",
  "timeframe": "4h",
  "parameters": {
    "period": 14, // Calculation period
    "overbought": 70, // Overbought threshold
    "oversold": 30, // Oversold threshold
    "signal_period": 1 // Signal smoothing
  }
}
```

#### MACD (Moving Average Convergence Divergence)

```json
{
  "name": "macd_1h",
  "type": "macd",
  "timeframe": "1h",
  "parameters": {
    "fast_period": 12, // Fast EMA period
    "slow_period": 26, // Slow EMA period
    "signal_period": 9, // Signal line EMA period
    "trigger_threshold": 0 // Minimum signal strength
  }
}
```

#### Bollinger Bands

```json
{
  "name": "bb_4h",
  "type": "bollinger",
  "timeframe": "4h",
  "parameters": {
    "period": 20, // Moving average period
    "std_dev": 2.0 // Standard deviation multiplier
  }
}
```

#### Moving Average

```json
{
  "name": "ma_1d",
  "type": "ma",
  "timeframe": "1d",
  "parameters": {
    "short_period": 20, // Short MA period
    "long_period": 50, // Long MA period
    "ma_type": "sma" // Type: "sma", "ema", "wma"
  }
}
```

## Market Symbol Mapping

### Correct Symbol Format

Always use full exchange symbols with the quote currency:

✅ **Correct:**

- `"ETH-USD"` - Ethereum vs US Dollar
- `"BTC-USD"` - Bitcoin vs US Dollar
- `"AVAX-USD"` - Avalanche vs US Dollar
- `"SOL-USD"` - Solana vs US Dollar

❌ **Incorrect:**

- `"ETH"` - Missing quote currency, ambiguous
- `"RSI-4H"` - This is an indicator name, not a market symbol!
- `"ETHEREUM"` - Use standardized symbols
- `"eth"` - Use proper case

### Exchange-Specific Symbols

Different exchanges may use different formats:

**Hyperliquid:**

- `"ETH"` (exchange format) → `"ETH-USD"` (config format)
- `"BTC"` (exchange format) → `"BTC-USD"` (config format)

**Coinbase:**

- `"ETH-USD"` (both exchange and config format)
- `"BTC-USD"` (both exchange and config format)

The system automatically handles exchange-specific symbol conversion.

## Common Configuration Patterns

### Pattern 1: Simple Single-Asset Strategy

For beginners or focused trading:

```json
{
  "strategies": [
    {
      "name": "eth_simple_rsi",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "indicators": ["rsi_4h"],
      "main_leverage": 1.0
    }
  ],
  "indicators": [
    {
      "name": "rsi_4h",
      "type": "rsi",
      "timeframe": "4h",
      "parameters": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
      }
    }
  ]
}
```

### Pattern 2: Multi-Timeframe Analysis

Combining different timeframes for comprehensive analysis:

```json
{
  "strategies": [
    {
      "name": "eth_multi_tf",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "indicators": [
        "trend_daily", // Daily trend
        "momentum_4h", // 4-hour momentum
        "entry_1h" // 1-hour entry timing
      ]
    }
  ],
  "indicators": [
    {
      "name": "trend_daily",
      "type": "ma",
      "timeframe": "1d",
      "parameters": { "short_period": 20, "long_period": 50 }
    },
    {
      "name": "momentum_4h",
      "type": "rsi",
      "timeframe": "4h",
      "parameters": { "period": 14 }
    },
    {
      "name": "entry_1h",
      "type": "macd",
      "timeframe": "1h",
      "parameters": { "fast_period": 12, "slow_period": 26 }
    }
  ]
}
```

### Pattern 3: Portfolio Diversification

Trading multiple assets with different strategies:

```json
{
  "strategies": [
    {
      "name": "eth_momentum",
      "market": "ETH-USD",
      "indicators": ["rsi_4h", "macd_1h"]
    },
    {
      "name": "btc_trend",
      "market": "BTC-USD",
      "indicators": ["ma_1d", "bb_4h"]
    },
    {
      "name": "avax_scalp",
      "market": "AVAX-USD",
      "indicators": ["rsi_4h"]
    }
  ],
  "indicators": [
    { "name": "rsi_4h", "type": "rsi", "timeframe": "4h" },
    { "name": "macd_1h", "type": "macd", "timeframe": "1h" },
    { "name": "ma_1d", "type": "ma", "timeframe": "1d" },
    { "name": "bb_4h", "type": "bollinger", "timeframe": "4h" },
    { "name": "rsi_4h", "type": "rsi", "timeframe": "4h" }
  ]
}
```

## Naming Conventions

### Strategy Names

Format: `{market}_{strategy_type}[_{timeframe}]`

Examples:

- `"eth_momentum_strategy"`
- `"btc_trend_following"`
- `"avax_scalping_1h"`
- `"multi_asset_portfolio"`

### Indicator Names

Format: `{market}_{indicator_type}_{timeframe}`

Examples:

- `"rsi_4h"`
- `"macd_1h"`
- `"bb_4h"`
- `"ma_1d"`

### Benefits of Good Naming

1. **Clarity**: Immediately understand what each component does
2. **Organization**: Easy to find and group related components
3. **Debugging**: Logs and errors reference clear names
4. **Maintenance**: Easy to modify and extend configurations

## Configuration Validation

### Common Errors & Solutions

**Error: "Market RSI-4H not found"**

```
Cause: Using indicator name as market symbol
Fix: Check strategy uses proper market like "ETH-USD"
```

**Error: "No indicators registered"**

```
Cause: Indicators not enabled or misconfigured
Fix: Verify "enabled": true and proper parameters
```

**Error: "Could not parse symbol from indicator name"**

```
Cause: Legacy symbol parsing attempt
Fix: Use proper strategy.indicators array instead
```

**Error: "Indicator 'rsi_4h' not found"**

```
Cause: Strategy references non-existent indicator
Fix: Ensure indicator name matches exactly
```

### Validation Checklist

Before running the system:

1. ✅ All strategy `market` fields use full symbols ("ETH-USD")
2. ✅ All strategy `indicators` arrays reference existing indicator names
3. ✅ All indicators have valid `type` and `timeframe` values
4. ✅ All required parameters are provided for each indicator type
5. ✅ Exchange configurations match strategy `exchange` fields
6. ✅ No duplicate names in strategies or indicators

## Testing Configuration

### Dry Run Testing

Always test new configurations with `"dry_run": true`:

```json
{
  "dry_run": true,
  "log_level": "DEBUG",
  "polling_interval": 10
}
```

### Configuration Validation Script

```bash
cd packages/spark-app
.venv/bin/python -c "
import json
from app.indicators.indicator_factory import IndicatorFactory
from app.core.config_validator import ConfigValidator

# Load config
with open('../shared/config.json') as f:
    config = json.load(f)

# Validate structure
validator = ConfigValidator()
if validator.validate(config):
    print('✅ Configuration is valid')
else:
    print('❌ Configuration errors:', validator.errors)

# Test indicator creation
indicators = IndicatorFactory.create_indicators_from_config(
    config.get('indicators', [])
)
print(f'✅ Created {len(indicators)} indicators')
"
```

### Backtesting Validation

Test each strategy configuration:

```bash
cd packages/spark-app
.venv/bin/python -m tests._utils.cli backtest-indicator \
  --indicator rsi_4h \
  --symbol ETH-USD \
  --timeframe 4h \
  --start-date 2024-01-01 \
  --end-date 2024-12-01
```

## Best Practices

### 1. Start Simple

Begin with single-asset, single-indicator strategies:

```json
{
  "strategies": [{ "name": "eth_simple", "market": "ETH-USD", "indicators": ["rsi_4h"] }],
  "indicators": [{ "name": "rsi_4h", "type": "rsi", "timeframe": "4h" }]
}
```

### 2. Use Descriptive Names

Prefer clear, descriptive names over short abbreviations:

✅ Good: `"momentum_4h"`, `"btc_trend_daily"` ❌ Avoid: `"ind1"`, `"strat_a"`, `"rsi"`

### 3. Group Related Components

Organize indicators by market and timeframe:

```json
{
  "indicators": [
    // ETH indicators
    { "name": "trend_daily", "type": "ma", "timeframe": "1d" },
    { "name": "momentum_4h", "type": "rsi", "timeframe": "4h" },
    { "name": "entry_1h", "type": "macd", "timeframe": "1h" },

    // BTC indicators
    { "name": "btc_trend_daily", "type": "ma", "timeframe": "1d" },
    { "name": "btc_momentum_4h", "type": "rsi", "timeframe": "4h" }
  ]
}
```

### 4. Document Your Strategy

Add comments to explain your strategy logic:

```json
{
  // ETH multi-timeframe momentum strategy
  // Uses daily MA for trend, 4h RSI for momentum, 1h MACD for entry
  "name": "eth_momentum_strategy",
  "market": "ETH-USD",
  "indicators": ["trend_daily", "momentum_4h", "entry_1h"]
}
```

### 5. Version Your Configurations

Keep backups of working configurations:

```
configs/
├── config-v1.0-simple-rsi.json
├── config-v1.1-multi-timeframe.json
├── config-v1.2-portfolio.json
└── config.json (current)
```

## Position Sizing Configuration

### Global Position Sizing

Define default position sizing behavior in the root `position_sizing` section:

```json
{
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_amount_usd": 1000,
    "risk_per_trade_pct": 0.02,
    "max_position_size_usd": 5000
  }
}
```

### Strategy-Specific Position Sizing

Override global defaults with strategy-specific position sizing:

```json
{
  "name": "eth_aggressive_strategy",
  "market": "ETH-USD",
  "indicators": ["rsi_4h"],
  "position_sizing": {
    "method": "risk_based",
    "risk_per_trade_pct": 0.03,
    "max_position_size_usd": 3000
  }
}
```

### Position Sizing Methods

#### 1. Fixed USD Amount

Trade a fixed dollar amount for each signal:

```json
{
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_amount_usd": 1000,
    "max_position_size_usd": 5000
  }
}
```

**Parameters:**

- `fixed_amount_usd`: Fixed dollar amount per trade
- `max_position_size_usd`: Maximum position size limit

**Use Cases:**

- Conservative trading
- Consistent position sizes regardless of market conditions
- Beginner strategies

#### 2. Risk-Based Sizing

Size positions based on risk per trade percentage:

```json
{
  "position_sizing": {
    "method": "risk_based",
    "risk_per_trade_pct": 0.02,
    "max_position_size_usd": 10000
  }
}
```

**Parameters:**

- `risk_per_trade_pct`: Percentage of portfolio to risk per trade (e.g., 0.02 = 2%)
- `max_position_size_usd`: Maximum position size limit

**Use Cases:**

- Professional risk management
- Scaling position size with portfolio growth
- Consistent risk exposure

#### 3. Percent Equity

Size positions as a percentage of total equity:

```json
{
  "position_sizing": {
    "method": "percent_equity",
    "percent_equity": 0.1,
    "max_position_size_usd": 15000
  }
}
```

**Parameters:**

- `percent_equity`: Percentage of total equity per position (e.g., 0.1 = 10%)
- `max_position_size_usd`: Maximum position size limit

**Use Cases:**

- Growth-oriented strategies
- Scaling with account size
- Diversified portfolio allocation

### Position Sizing Inheritance

Strategy-specific position sizing inherits from global config and overrides specific parameters:

```json
{
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_amount_usd": 1000,
    "risk_per_trade_pct": 0.02,
    "max_position_size_usd": 5000
  },
  "strategies": [
    {
      "name": "eth_conservative",
      "market": "ETH-USD",
      "indicators": ["rsi_4h"]
      // Uses global position sizing: fixed_usd with $1000
    },
    {
      "name": "eth_aggressive",
      "market": "ETH-USD",
      "indicators": ["macd_1h"],
      "position_sizing": {
        "method": "risk_based",
        "risk_per_trade_pct": 0.03
        // Inherits max_position_size_usd: 5000 from global
      }
    },
    {
      "name": "btc_scalping",
      "market": "BTC-USD",
      "indicators": ["rsi_15m"],
      "position_sizing": {
        "fixed_amount_usd": 500
        // Inherits method: "fixed_usd" and max_position_size_usd: 5000 from global
      }
    }
  ]
}
```

### Advanced Position Sizing Examples

#### Multi-Strategy Portfolio with Different Risk Profiles

```json
{
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_amount_usd": 1000,
    "max_position_size_usd": 5000
  },
  "strategies": [
    {
      "name": "eth_trend_following",
      "market": "ETH-USD",
      "indicators": ["ma_1d", "rsi_4h"],
      "position_sizing": {
        "method": "percent_equity",
        "percent_equity": 0.15,
        "max_position_size_usd": 8000
      }
    },
    {
      "name": "btc_scalping",
      "market": "BTC-USD",
      "indicators": ["rsi_15m"],
      "position_sizing": {
        "method": "fixed_usd",
        "fixed_amount_usd": 500,
        "max_position_size_usd": 2000
      }
    },
    {
      "name": "avax_momentum",
      "market": "AVAX-USD",
      "indicators": ["macd_1h"],
      "position_sizing": {
        "method": "risk_based",
        "risk_per_trade_pct": 0.025,
        "max_position_size_usd": 3000
      }
    }
  ]
}
```

#### Exchange-Specific Position Sizing

```json
{
  "strategies": [
    {
      "name": "hyperliquid_eth_strategy",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "indicators": ["rsi_4h"],
      "position_sizing": {
        "method": "risk_based",
        "risk_per_trade_pct": 0.03,
        "max_position_size_usd": 5000
      }
    },
    {
      "name": "coinbase_eth_hedge",
      "market": "ETH-USD",
      "exchange": "coinbase",
      "indicators": ["ma_1h"],
      "position_sizing": {
        "method": "fixed_usd",
        "fixed_amount_usd": 1000,
        "max_position_size_usd": 3000
      }
    }
  ]
}
```

This configuration guide provides the foundation for properly setting up the Spark Stacker trading
system. The key insight is understanding the clear separation of concerns: strategies define WHAT to
trade, indicators define HOW to analyze, and the system handles connecting them together.
