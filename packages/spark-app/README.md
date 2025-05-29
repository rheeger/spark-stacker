# Spark Stacker Trading Application

The core trading application for Spark Stacker, built with Python 3.11+ and designed for multi-exchange algorithmic trading with strategy-driven architecture.

## Overview

This package contains the main trading engine that:

- Executes trading strategies across multiple exchanges
- Processes technical indicators for signal generation
- Manages risk and position sizing per strategy
- Provides real-time monitoring and logging
- Supports both live trading and backtesting

## Architecture

### Strategy-Driven Design

The application follows a clear separation between **strategies** (WHAT to trade) and **indicators** (HOW to analyze):

```
Strategy → Indicators → Signals → Trading Engine → Exchange
```

- **Strategies** define the trading plan (market, exchange, indicators, risk parameters)
- **Indicators** analyze market data and generate signals
- **Trading Engine** executes trades based on signals and strategy rules
- **Risk Manager** handles position sizing and risk controls per strategy

## Quick Start

### Prerequisites

- Python 3.11+
- Virtual environment setup

### Installation

```bash
cd packages/spark-app

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Configuration

1. **Configure environment variables** in `packages/shared/.env`:

```bash
# Hyperliquid
WALLET_ADDRESS=your_wallet_address
PRIVATE_KEY=your_private_key
HYPERLIQUID_TESTNET=true

# Coinbase (optional)
COINBASE_API_KEY=your_api_key
COINBASE_API_SECRET=your_api_secret
COINBASE_USE_SANDBOX=true
```

2. **Configure trading settings** in `packages/shared/config.json`:

```json
{
  "dry_run": true,
  "log_level": "INFO",
  "polling_interval": 60,

  "strategies": [
    {
      "name": "my_first_strategy",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "enabled": true,
      "indicators": ["rsi_4h"],
      "stop_loss_pct": 5.0,
      "take_profit_pct": 10.0
    }
  ],

  "indicators": [
    {
      "name": "rsi_4h",
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

3. **Run the application**:

```bash
python app/main.py
```

## Configuration Guide

### Strategy Configuration

Each strategy must specify:

| Field             | Required | Description                         | Example                   |
| ----------------- | -------- | ----------------------------------- | ------------------------- |
| `name`            | ✅       | Unique strategy identifier          | `"eth_momentum_strategy"` |
| `market`          | ✅       | Trading pair in "SYMBOL-USD" format | `"ETH-USD"`               |
| `exchange`        | ✅       | Which exchange to use               | `"hyperliquid"`           |
| `enabled`         | ✅       | Whether strategy is active          | `true`                    |
| `indicators`      | ✅       | List of indicator names to use      | `["rsi_4h", "macd_1h"]`   |
| `timeframe`       | ❌       | Default timeframe (optional)        | `"4h"`                    |
| `stop_loss_pct`   | ❌       | Stop loss percentage                | `5.0`                     |
| `take_profit_pct` | ❌       | Take profit percentage              | `10.0`                    |
| `position_sizing` | ❌       | Strategy-specific position sizing   | See below                 |

### Position Sizing Configuration

#### Global Position Sizing (applies to all strategies by default)

```json
{
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_usd_amount": 100.0,
    "max_position_size_usd": 1000.0,
    "min_position_size_usd": 50.0
  }
}
```

#### Strategy-Specific Position Sizing

Override global settings for individual strategies:

```json
{
  "strategies": [
    {
      "name": "conservative_btc",
      "market": "BTC-USD",
      "exchange": "hyperliquid",
      "indicators": ["rsi_4h"],
      "position_sizing": {
        "method": "fixed_usd",
        "fixed_usd_amount": 50.0
      }
    },
    {
      "name": "aggressive_eth",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "indicators": ["macd_1h"],
      "position_sizing": {
        "method": "risk_based",
        "risk_per_trade_pct": 0.03,
        "default_stop_loss_pct": 0.05,
        "max_position_size_usd": 2000.0
      }
    },
    {
      "name": "percent_equity_sol",
      "market": "SOL-USD",
      "exchange": "hyperliquid",
      "indicators": ["rsi_1h"],
      "position_sizing": {
        "method": "percent_equity",
        "equity_percentage": 0.05,
        "max_position_size_usd": 1500.0
      }
    }
  ]
}
```

### Position Sizing Methods

#### 1. Fixed USD Amount

```json
{
  "method": "fixed_usd",
  "fixed_usd_amount": 100.0,
  "max_position_size_usd": 1000.0
}
```

- Trades a fixed dollar amount each time
- Simple and predictable
- Good for testing and conservative strategies

#### 2. Percentage of Equity

```json
{
  "method": "percent_equity",
  "equity_percentage": 0.05,
  "max_position_size_usd": 2000.0
}
```

- Trades a percentage of account balance
- Position size grows/shrinks with account
- Good for compounding strategies

#### 3. Risk-Based Sizing

```json
{
  "method": "risk_based",
  "risk_per_trade_pct": 0.02,
  "default_stop_loss_pct": 0.05,
  "max_position_size_usd": 1500.0
}
```

- Position size based on stop loss and risk tolerance
- `risk_per_trade_pct`: How much of account to risk per trade (2%)
- `default_stop_loss_pct`: Expected stop loss distance (5%)
- Position size = (Account \* Risk%) / Stop Loss%

#### 4. Kelly Criterion

```json
{
  "method": "kelly",
  "kelly_win_rate": 0.55,
  "kelly_avg_win": 0.08,
  "kelly_avg_loss": 0.04,
  "kelly_max_position_pct": 0.25,
  "max_position_size_usd": 3000.0
}
```

- Optimal position sizing based on historical performance
- Requires backtesting data to set parameters
- Can be aggressive - use `kelly_max_position_pct` to limit

### Indicator Configuration

```json
{
  "indicators": [
    {
      "name": "rsi_4h",
      "type": "rsi",
      "enabled": true,
      "timeframe": "4h",
      "parameters": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
      }
    },
    {
      "name": "macd_daily",
      "type": "macd",
      "enabled": true,
      "timeframe": "1d",
      "parameters": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
      }
    }
  ]
}
```

### Exchange Configuration

```json
{
  "exchanges": [
    {
      "name": "hyperliquid",
      "exchange_type": "hyperliquid",
      "wallet_address": "${WALLET_ADDRESS}",
      "private_key": "${PRIVATE_KEY}",
      "testnet": true,
      "enabled": true,
      "use_as_main": true
    },
    {
      "name": "coinbase",
      "exchange_type": "coinbase",
      "api_key": "${COINBASE_API_KEY}",
      "api_secret": "${COINBASE_API_SECRET}",
      "use_sandbox": true,
      "enabled": false,
      "use_as_main": false
    }
  ]
}
```

## Complete Configuration Examples

### Single Strategy, Single Indicator

```json
{
  "dry_run": true,
  "log_level": "INFO",
  "polling_interval": 60,

  "position_sizing": {
    "method": "fixed_usd",
    "fixed_usd_amount": 100.0
  },

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

  "strategies": [
    {
      "name": "eth_rsi_strategy",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "enabled": true,
      "indicators": ["rsi_4h"],
      "stop_loss_pct": 5.0,
      "take_profit_pct": 10.0
    }
  ],

  "indicators": [
    {
      "name": "rsi_4h",
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

### Multi-Strategy, Multi-Timeframe

```json
{
  "dry_run": true,
  "log_level": "INFO",
  "polling_interval": 30,

  "position_sizing": {
    "method": "fixed_usd",
    "fixed_usd_amount": 100.0,
    "max_position_size_usd": 1000.0
  },

  "strategies": [
    {
      "name": "btc_conservative",
      "market": "BTC-USD",
      "exchange": "hyperliquid",
      "enabled": true,
      "indicators": ["rsi_daily"],
      "stop_loss_pct": 3.0,
      "take_profit_pct": 8.0,
      "position_sizing": {
        "method": "fixed_usd",
        "fixed_usd_amount": 200.0
      }
    },
    {
      "name": "eth_aggressive",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "enabled": true,
      "indicators": ["rsi_4h", "macd_1h"],
      "stop_loss_pct": 5.0,
      "take_profit_pct": 15.0,
      "position_sizing": {
        "method": "percent_equity",
        "equity_percentage": 0.1,
        "max_position_size_usd": 2000.0
      }
    },
    {
      "name": "sol_momentum",
      "market": "SOL-USD",
      "exchange": "hyperliquid",
      "enabled": true,
      "indicators": ["macd_4h"],
      "stop_loss_pct": 7.0,
      "take_profit_pct": 20.0,
      "position_sizing": {
        "method": "risk_based",
        "risk_per_trade_pct": 0.02,
        "default_stop_loss_pct": 0.07
      }
    }
  ],

  "indicators": [
    {
      "name": "rsi_daily",
      "type": "rsi",
      "timeframe": "1d",
      "parameters": {
        "period": 14,
        "overbought": 75,
        "oversold": 25
      }
    },
    {
      "name": "rsi_4h",
      "type": "rsi",
      "timeframe": "4h",
      "parameters": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
      }
    },
    {
      "name": "macd_1h",
      "type": "macd",
      "timeframe": "1h",
      "parameters": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
      }
    },
    {
      "name": "macd_4h",
      "type": "macd",
      "timeframe": "4h",
      "parameters": {
        "fast_period": 8,
        "slow_period": 21,
        "signal_period": 7
      }
    }
  ]
}
```

## Running the Application

### Development Mode

```bash
cd packages/spark-app

# Activate virtual environment
source .venv/bin/activate

# Run with dry run enabled
python app/main.py

# Run with specific log level
python app/main.py --log-level DEBUG

# Run single strategy
python app/main.py --strategy eth_rsi_strategy
```

### Production Mode

```bash
# Set dry_run to false in config.json first!
python app/main.py --log-level INFO
```

### Testing Configuration

```bash
# Test configuration loading
.venv/bin/python -c "
import json
from app.core.strategy_config import StrategyConfigLoader
from app.indicators.indicator_factory import IndicatorFactory

with open('../shared/config.json') as f:
    config = json.load(f)

# Test strategy loading
strategies = StrategyConfigLoader.load_strategies(config['strategies'])
print(f'✅ Loaded {len(strategies)} strategies')

# Test indicator loading
indicators = IndicatorFactory.create_indicators_from_config(config['indicators'])
print(f'✅ Loaded {len(indicators)} indicators')

# Test validation
StrategyConfigLoader.validate_indicators(strategies, indicators)
print('✅ Strategy-indicator validation passed')
"

# Test position sizing
.venv/bin/python -c "
import json
from app.risk_management.risk_manager import RiskManager

with open('../shared/config.json') as f:
    config = json.load(f)

risk_manager = RiskManager.from_config(config)
print('✅ Risk manager created successfully')
"
```

## Monitoring and Logging

### Log Files

Logs are stored in `_logs/` directory:

- `spark_stacker.log` - Main application log
- `balance.log` - Account balance changes
- `orders.log` - Order executions
- `markets.log` - Market data updates

### Log Levels

Set in `config.json`:

- `DEBUG` - Detailed information for debugging
- `INFO` - General information (recommended)
- `WARNING` - Important warnings
- `ERROR` - Error conditions only

### Metrics

Application exposes Prometheus metrics on port 9000:

- `spark_stacker_trades_total` - Number of trades executed
- `spark_stacker_active_positions` - Current open positions
- `spark_stacker_account_balance` - Account balance
- `spark_stacker_strategy_signals` - Signals generated per strategy

## Troubleshooting

### Common Issues

#### 1. "Market RSI-4H not found" Error

**Cause**: Indicator name used as market symbol

**Solution**: Check strategy configuration:

```json
// ❌ Wrong
{
  "market": "RSI-4H"
}

// ✅ Correct
{
  "market": "ETH-USD",
  "indicators": ["rsi_4h"]
}
```

#### 2. Strategy Not Executing

**Checklist**:

- [ ] Strategy `enabled: true`
- [ ] Exchange `enabled: true`
- [ ] All indicators exist and enabled
- [ ] Market symbol format correct ("ETH-USD", not "ETH")
- [ ] Exchange field specified
- [ ] Valid API credentials in `.env`

#### 3. Position Sizing Errors

**Check**:

- Required parameters for chosen method
- Positive numeric values
- `max_position_size_usd` > `min_position_size_usd`
- Risk percentages as decimals (0.02 = 2%)

#### 4. Exchange Connection Issues

**Verify**:

- API credentials in `.env`
- Network connectivity
- Exchange-specific symbol formats
- Rate limiting compliance

### Debug Commands

```bash
# Test exchange connectivity
.venv/bin/python -c "
from app.connectors.connector_factory import ConnectorFactory
import json

with open('../shared/config.json') as f:
    config = json.load(f)

connector = ConnectorFactory.create_connector(config['exchanges'][0])
balance = connector.get_account_info()
print(f'✅ Connected to exchange: {balance}')
"

# Test indicator processing
.venv/bin/python -c "
from app.indicators.indicator_factory import IndicatorFactory
import json

with open('../shared/config.json') as f:
    config = json.load(f)

indicator = IndicatorFactory.create_indicator(config['indicators'][0])
print(f'✅ Indicator created: {indicator}')
"

# Run with verbose logging
python app/main.py --log-level DEBUG
```

## Development

### Adding New Strategies

1. Add strategy configuration to `config.json`
2. Reference existing indicators or create new ones
3. Test configuration loading
4. Run in dry-run mode first

### Adding New Indicators

1. Create new indicator class in `app/indicators/`
2. Extend `BaseIndicator` class
3. Implement `process()` method
4. Add to `IndicatorFactory`
5. Add configuration to `config.json`

### Testing

```bash
# Run all tests
.venv/bin/python -m pytest tests/ --cov=app

# Run specific test category
.venv/bin/python -m pytest tests/core/ -v
.venv/bin/python -m pytest tests/indicators/ -v
.venv/bin/python -m pytest tests/risk_management/ -v

# Run integration tests
.venv/bin/python -m pytest tests/integration/ -v
```

## Performance Optimization

### Configuration Tips

1. **Polling Interval**: Balance between responsiveness and API limits

   ```json
   "polling_interval": 60  // Seconds between strategy cycles
   ```

2. **Parallel Trades**: Limit concurrent operations

   ```json
   "max_parallel_trades": 1  // Start with 1, increase carefully
   ```

3. **Log Level**: Use INFO or WARNING in production

   ```json
   "log_level": "INFO"  // Avoid DEBUG in production
   ```

4. **Timeframe Selection**: Longer timeframes = fewer API calls
   ```json
   "timeframe": "4h"  // vs "1m" for fewer updates
   ```

### Monitoring Performance

- Check log files for warnings/errors
- Monitor API rate limits
- Track position execution times
- Use Grafana dashboards for real-time metrics

## Security Best Practices

1. **Environment Variables**: Never commit API keys
2. **Testnet First**: Always test on testnet/sandbox
3. **Position Limits**: Set reasonable `max_position_size_usd`
4. **Stop Losses**: Always configure stop loss percentages
5. **Dry Run**: Test new strategies with `dry_run: true`
6. **Log Security**: Don't log sensitive data

## Support

For additional help:

- Check the main [README.md](../../README.md) for general information
- Review [troubleshooting documentation](../shared/docs/)
- Run configuration validation scripts
- Enable debug logging for detailed information
