# User Guide: Spark Stacker Trading System

## System Overview

The **Spark Stacker Trading System** is designed to execute high-leverage trades on decentralized
perpetual futures exchanges while implementing sophisticated hedging strategies to protect your
capital. The system follows technical indicators to enter trades and automatically manages risk
through position sizing, stop-losses, and strategic hedge positions.

## Architecture Overview

Understanding the relationship between the core components is essential for proper configuration:

### Core Components Relationship

```
STRATEGY (defines what to trade)
├── market: "ETH-USD"              # Actual exchange symbol
├── timeframe: "4h"                # Default timeframe for the strategy
├── indicators: ["rsi_4h", "macd_1h"]  # Which indicators to use
├── position_sizing: {...}         # Strategy-specific position sizing (optional)
└── risk_params: {...}             # Position sizing, stop-loss, etc.

INDICATORS (define how to analyze)
├── name: "rsi_4h"             # Unique identifier
├── type: "rsi"                    # Indicator algorithm
├── timeframe: "4h"                # Data timeframe for this indicator
├── parameters: {...}              # Algorithm-specific settings
└── enabled: true                  # Whether to run this indicator

EXCHANGE CONNECTOR (provides data)
├── Fetches data for market: "ETH-USD"
├── Provides timeframe-specific candles: "4h", "1h", etc.
└── Executes trades on the actual exchange
```

### Key Principles

1. **Strategies define WHAT to trade**: The market symbol (like "ETH-USD"), which indicators to use,
   risk parameters, and position sizing methods
2. **Indicators define HOW to analyze**: The specific algorithm, timeframe, and parameters for
   analysis
3. **Markets are actual exchange symbols**: Use full symbols like "ETH-USD", "BTC-USD", not just
   "ETH"
4. **Timeframes can be set at multiple levels**: Strategy-level defaults and indicator-level
   specifics
5. **Position sizing can be strategy-specific**: Each strategy can use different position sizing
   methods while inheriting from global defaults

## 1. **Setup & Installation**

### 1.1 System Requirements

- Python 3.11 or higher
- Node.js 20+ (for the NX monorepo)
- Access to exchange APIs (Hyperliquid, Coinbase)
- Yarn package manager

### 1.2 Installation Process

1. Clone the repository:

   ```bash
   git clone https://github.com/user/spark-stacker.git
   cd spark-stacker
   ```

2. Install dependencies:

   ```bash
   # Install Node dependencies for the monorepo
   yarn install

   # Set up Python virtual environment
   cd packages/spark-app
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   # Copy the example environment file
   cp packages/shared/.env.example packages/shared/.env
   # Edit the .env file with your API keys and settings
   ```

## 2. **Configuration Setup**

### 2.1 Proper Configuration Structure

The configuration follows a clear hierarchy. Here's a complete example:

```json
{
  "log_level": "INFO",
  "webhook_enabled": false,
  "polling_interval": 30,
  "dry_run": true,
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
      "market": "ETH-USD", // ← ACTUAL EXCHANGE SYMBOL
      "exchange": "hyperliquid", // ← WHICH EXCHANGE TO USE
      "enabled": true,
      "timeframe": "4h", // ← DEFAULT TIMEFRAME
      "indicators": ["trend_4h", "entry_1h"], // ← WHICH INDICATORS
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
    }
  ],
  "indicators": [
    {
      "name": "trend_4h", // ← UNIQUE IDENTIFIER
      "type": "rsi", // ← ALGORITHM TYPE
      "enabled": true,
      "timeframe": "4h", // ← DATA TIMEFRAME
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
      "timeframe": "1h", // ← DIFFERENT TIMEFRAME
      "parameters": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
      }
    }
  ]
}
```

### 2.2 Strategy Configuration Explained

**Required Fields:**

- `market`: Full exchange symbol (e.g., "ETH-USD", "BTC-USD")
- `exchange`: Which exchange to use ("hyperliquid", "coinbase")
- `indicators`: Array of indicator names to use with this strategy

**Optional Fields:**

- `timeframe`: Default timeframe (overridden by indicator-specific timeframes)
- Risk management parameters (leverage, stop-loss, etc.)

**Example Multi-Timeframe Strategy:**

```json
{
  "name": "btc_comprehensive_strategy",
  "market": "BTC-USD",
  "exchange": "hyperliquid",
  "enabled": true,
  "timeframe": "1h", // Default
  "indicators": [
    "btc_daily_trend", // Uses 1d timeframe
    "btc_4h_momentum", // Uses 4h timeframe
    "btc_1h_entry" // Uses 1h timeframe
  ],
  "main_leverage": 2.0,
  "stop_loss_pct": 3.0,
  "take_profit_pct": 6.0
}
```

### 2.3 Indicator Configuration Explained

**Required Fields:**

- `name`: Unique identifier used by strategies
- `type`: Algorithm type ("rsi", "macd", "bollinger", "ma")
- `timeframe`: Data timeframe for this indicator

**Timeframe Examples:**

- `"1m"`, `"5m"`, `"15m"`, `"30m"` - Minutes
- `"1h"`, `"4h"`, `"12h"` - Hours
- `"1d"`, `"1w"` - Days/Weeks

**Multi-Timeframe Indicator Setup:**

```json
{
  "indicators": [
    {
      "name": "btc_daily_trend",
      "type": "ma",
      "timeframe": "1d", // Daily trend
      "parameters": {
        "short_period": 20,
        "long_period": 50
      }
    },
    {
      "name": "btc_4h_momentum",
      "type": "rsi",
      "timeframe": "4h", // 4-hour momentum
      "parameters": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
      }
    },
    {
      "name": "btc_1h_entry",
      "type": "macd",
      "timeframe": "1h", // 1-hour entry timing
      "parameters": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
      }
    }
  ]
}
```

### 2.4 Market Symbol Mapping

**Important**: Always use full exchange symbols, not abbreviated forms:

✅ **Correct:**

- `"ETH-USD"` - Ethereum vs US Dollar
- `"BTC-USD"` - Bitcoin vs US Dollar
- `"AVAX-USD"` - Avalanche vs US Dollar

❌ **Incorrect:**

- `"ETH"` - Ambiguous, what's the quote currency?
- `"RSI-4H"` - This is an indicator name, not a market!
- `"ETHEREUM"` - Use standard symbols

### 2.5 Common Configuration Patterns

**Pattern 1: Single Timeframe Strategy**

```json
{
  "strategies": [
    {
      "name": "rsi_4h",
      "market": "ETH-USD",
      "timeframe": "4h",
      "indicators": ["rsi_4h"]
    }
  ],
  "indicators": [
    {
      "name": "rsi_4h",
      "type": "rsi",
      "timeframe": "4h"
    }
  ]
}
```

**Pattern 2: Multi-Indicator Strategy**

```json
{
  "strategies": [
    {
      "name": "eth_combined",
      "market": "ETH-USD",
      "indicators": ["rsi_4h", "macd_1h", "bb_15m"]
    }
  ],
  "indicators": [
    { "name": "rsi_4h", "type": "rsi", "timeframe": "4h" },
    { "name": "macd_1h", "type": "macd", "timeframe": "1h" },
    { "name": "bb_15m", "type": "bollinger", "timeframe": "15m" }
  ]
}
```

**Pattern 3: Multiple Market Strategy**

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
      "indicators": ["btc_macd_1h"]
    }
  ],
  "indicators": [
    { "name": "rsi_4h", "type": "rsi", "timeframe": "4h" },
    { "name": "btc_macd_1h", "type": "macd", "timeframe": "1h" }
  ]
}
```

### 2.6 Position Sizing Configuration

The system supports both global and strategy-specific position sizing, allowing you to customize
position sizes based on your risk tolerance and strategy characteristics.

#### 2.6.1 Global Position Sizing

Set default position sizing behavior in the root `position_sizing` section:

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

#### 2.6.2 Strategy-Specific Position Sizing

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

#### 2.6.3 Position Sizing Methods

**1. Fixed USD Amount (`fixed_usd`)**

Trade a consistent dollar amount for each signal:

```json
{
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_amount_usd": 1000,
    "max_position_size_usd": 5000
  }
}
```

- **Best for**: Conservative trading, beginners, consistent exposure
- **Example**: Always trade $1000 worth, regardless of account size

**2. Risk-Based Sizing (`risk_based`)**

Size positions based on percentage of portfolio at risk:

```json
{
  "position_sizing": {
    "method": "risk_based",
    "risk_per_trade_pct": 0.02,
    "max_position_size_usd": 10000
  }
}
```

- **Best for**: Professional risk management, scaling with portfolio
- **Example**: Risk 2% of portfolio per trade, adjusting position size accordingly

**3. Percent Equity (`percent_equity`)**

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

- **Best for**: Growth strategies, scaling with account size
- **Example**: Use 10% of total equity per position

#### 2.6.4 Multi-Strategy Position Sizing Example

```json
{
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_amount_usd": 1000,
    "max_position_size_usd": 5000
  },
  "strategies": [
    {
      "name": "eth_conservative",
      "market": "ETH-USD",
      "indicators": ["rsi_4h"]
      // Uses global: fixed_usd $1000
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
      "indicators": ["btc_rsi_15m"],
      "position_sizing": {
        "fixed_amount_usd": 500
        // Inherits method: "fixed_usd" from global
      }
    }
  ]
}
```

#### 2.6.5 Position Sizing Best Practices

1. **Start Conservative**: Begin with fixed USD amounts or low risk percentages
2. **Test Different Methods**: Backtest each method to understand performance
3. **Consider Strategy Type**:
   - Scalping strategies → Fixed USD or small percentages
   - Swing trading → Risk-based sizing
   - Long-term trends → Percent equity
4. **Set Maximum Limits**: Always use `max_position_size_usd` to prevent oversized positions
5. **Monitor Performance**: Track how different sizing methods affect your returns

## 3. **Exchange Connections**

### 3.1 Connecting to Hyperliquid

Configure Hyperliquid in your `.env` file:

```env
WALLET_ADDRESS=0x...
PRIVATE_KEY=0x...
HYPERLIQUID_TESTNET=true  # Set to false for mainnet
```

Verify connection:

```bash
cd packages/spark-app
.venv/bin/python -c "
from app.connectors.hyperliquid_connector import HyperliquidConnector
connector = HyperliquidConnector(testnet=True)
print('Available markets:', connector.get_available_markets())
"
```

### 3.2 Connecting to Coinbase

Configure Coinbase in your `.env` file:

```env
COINBASE_API_KEY=your_key
COINBASE_API_SECRET=your_secret
COINBASE_PASSPHRASE=your_passphrase
```

## 4. **Running the Trading System**

### 4.1 Development Mode

Start with dry run mode enabled:

```bash
cd packages/spark-app
.venv/bin/python app/main.py
```

Monitor the logs to ensure:

- Indicators are properly loaded
- Market data is being fetched
- Strategies are running without errors

### 4.2 Backtesting

Test your strategy configuration:

```bash
cd packages/spark-app
.venv/bin/python -m tests._utils.cli backtest-indicator \
  --indicator rsi_4h \
  --symbol ETH-USD \
  --timeframe 4h \
  --start-date 2024-01-01 \
  --end-date 2024-12-01
```

### 4.3 Live Trading (Production)

Only after successful backtesting and dry run validation:

1. Set `"dry_run": false` in your config
2. Start with small position sizes
3. Monitor closely for the first few hours

```bash
cd packages/spark-app
.venv/bin/python app/main.py
```

## 5. **Monitoring & Troubleshooting**

### 5.1 Log Analysis

Check logs for common issues:

```bash
# Check if strategies are loading correctly
grep "strategy" packages/spark-app/_logs/spark_stacker.log

# Check if indicators are being assigned markets
grep "Assigned symbol" packages/spark-app/_logs/spark_stacker.log

# Check for market data fetching
grep "Fetching.*historical.*candles" packages/spark-app/_logs/spark_stacker.log
```

### 5.2 Common Errors & Solutions

**Error: "Market RSI-4H not found"**

- **Cause**: Indicator name being used as market symbol
- **Fix**: Check strategy configuration uses proper market symbols like "ETH-USD"

**Error: "No indicators registered"**

- **Cause**: Indicators not properly configured or enabled
- **Fix**: Verify indicator configuration and `"enabled": true`

**Error: "Could not parse symbol from indicator name"**

- **Cause**: Legacy symbol parsing logic
- **Fix**: Use proper strategy-indicator relationships instead

### 5.3 Grafana Monitoring

Access monitoring dashboards:

```bash
cd packages/monitoring
make start-monitoring
# Open http://localhost:3000
```

View timeframe-specific metrics:

```promql
# 4-hour RSI data
spark_stacker_candle{market="ETH-USD", timeframe="4h", field="close"}

# 1-hour MACD data
spark_stacker_macd{market="ETH-USD", timeframe="1h", component="macd_line"}
```

## 6. **Best Practices**

### 6.1 Configuration Best Practices

1. **Use descriptive names**: `"rsi_4h"` instead of `"rsi1"`
2. **Match timeframes to strategy**: Daily strategies use daily indicators
3. **Start simple**: Single indicator, single timeframe, then expand
4. **Test thoroughly**: Always backtest before live trading

### 6.2 Naming Conventions

**Indicators:**

- Format: `{market}_{type}_{timeframe}`
- Examples: `"rsi_4h"`, `"btc_macd_1h"`, `"avax_bb_15m"`

**Strategies:**

- Format: `{market}_{strategy_type}`
- Examples: `"eth_momentum_strategy"`, `"btc_scalping_strategy"`

### 6.3 Risk Management

1. **Start with conservative parameters**: Low leverage, wide stops
2. **Use multiple timeframes**: Combine trend, momentum, and timing
3. **Diversify across markets**: Don't put all capital in one asset
4. **Monitor performance**: Track win rates, drawdown, Sharpe ratio

This user guide provides the foundation for properly configuring and running the Spark Stacker
trading system. The key is understanding the clear separation between strategies (what to trade),
indicators (how to analyze), and markets (actual exchange symbols).
