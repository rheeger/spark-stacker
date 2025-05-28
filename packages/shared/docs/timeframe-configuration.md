# Timeframe Configuration Guide

This document explains how to configure timeframes for indicators, strategies, and backtesting in
the Spark Stacker trading system.

## Overview

The Spark Stacker system now supports unified timeframe configuration across all components:

- **Indicators**: Each indicator can run on its own specified timeframe
- **Strategies**: Can have default timeframes that apply to their indicators
- **Backtesting**: Uses the same configuration for consistency between live trading and backtesting

## Configuration Levels

### 1. Indicator-Level Timeframes (Highest Priority)

Configure specific timeframes for individual indicators in `config.json`:

```json
{
  "indicators": [
    {
      "name": "eth_rsi_4h",
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
      "name": "eth_macd_1h",
      "type": "macd",
      "enabled": true,
      "timeframe": "1h",
      "parameters": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
      }
    }
  ]
}
```

### 2. Strategy-Level Timeframes

Set default timeframes for strategies:

```json
{
  "strategies": [
    {
      "name": "eth_strategy",
      "market": "ETH",
      "enabled": true,
      "timeframe": "4h",
      "main_leverage": 1.0,
      "stop_loss_pct": 5.0,
      "take_profit_pct": 10.0
    }
  ]
}
```

### 3. Backtesting Configuration

Configure global defaults and supported timeframes:

```json
{
  "backtesting": {
    "default_timeframe": "1h",
    "enable_multi_timeframe": true,
    "supported_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
    "default_data_source": "default",
    "default_leverage": 1.0
  }
}
```

## Supported Timeframes

The system supports the following timeframe formats:

| Timeframe                      | Description |
| ------------------------------ | ----------- |
| `1m`, `3m`, `5m`, `15m`, `30m` | Minutes     |
| `1h`, `2h`, `4h`, `6h`, `12h`  | Hours       |
| `1d`, `3d`                     | Days        |
| `1w`                           | Week        |

## Timeframe Resolution Hierarchy

When determining which timeframe to use, the system follows this priority order:

1. **CLI Parameter** (highest priority) - for backtesting commands
2. **Indicator-level timeframe** - from `config.json`
3. **Strategy-level timeframe** - from `config.json`
4. **Global default** - from backtesting config or system default (`1h`)

## Multi-Timeframe Trading

You can run multiple indicators on different timeframes simultaneously:

```json
{
  "indicators": [
    {
      "name": "eth_rsi_daily",
      "type": "rsi",
      "enabled": true,
      "timeframe": "1d",
      "parameters": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
      }
    },
    {
      "name": "eth_macd_4h",
      "type": "macd",
      "enabled": true,
      "timeframe": "4h",
      "parameters": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
      }
    },
    {
      "name": "eth_bollinger_1h",
      "type": "bollinger",
      "enabled": true,
      "timeframe": "1h",
      "parameters": {
        "period": 20,
        "std_dev": 2.0
      }
    }
  ]
}
```

## Backtesting with Timeframes

### CLI Usage

All backtesting CLI commands now respect the configured timeframes:

```bash
# Run backtest with specific timeframe (overrides config)
packages/spark-app/.venv/bin/python -m tests._utils.cli backtest-indicator \
  --indicator MACD \
  --symbol ETH-USD \
  --timeframe 4h \
  --start-date 2024-01-01 \
  --end-date 2024-12-01

# Run backtest using config timeframes
packages/spark-app/.venv/bin/python -m tests._utils.cli backtest-indicator \
  --indicator eth_macd_1h \
  --symbol ETH-USD
```

### Programmatic Usage

```python
from app.backtesting.indicator_backtest_manager import IndicatorBacktestManager
from app.backtesting.backtest_engine import BacktestEngine
from app.backtesting.data_manager import DataManager

# Initialize backtesting components
data_manager = DataManager()
backtest_engine = BacktestEngine(data_manager)
backtest_manager = IndicatorBacktestManager(backtest_engine)

# Run backtest with configured timeframe
result = backtest_manager.run_indicator_backtest(
    indicator_name="eth_macd_1h",  # Uses timeframe from config
    symbol="ETH-USD",
    timeframe="1h",  # Can override config timeframe
    start_date="2024-01-01",
    end_date="2024-12-01"
)
```

## Live Trading Behavior

### Data Fetching

Each indicator automatically fetches historical data for its configured timeframe:

- RSI on 4h timeframe fetches 4-hour candles
- MACD on 1h timeframe fetches 1-hour candles
- Multiple indicators can run simultaneously on different timeframes

### Metrics Collection

Prometheus metrics are collected using the correct timeframes:

```python
# Metrics are labeled with the indicator's actual timeframe
update_candle_data(market="ETH-USD", timeframe="4h", field="close", value=2500.0)
update_macd_indicator(market="ETH-USD", timeframe="1h", component="macd_line", value=15.2)
```

## Monitoring and Visualization

### Grafana Dashboards

When viewing data in Grafana, filter by timeframe to see the correct data:

```promql
# View 4-hour RSI data
spark_stacker_candle{market="ETH-USD", timeframe="4h", field="close"}

# View 1-hour MACD data
spark_stacker_macd{market="ETH-USD", timeframe="1h", component="macd_line"}
```

### Log Analysis

Logs now include timeframe information:

```
2024-12-19 10:15:30 - strategy_manager - INFO - Found target MACD indicator: eth_macd_1h for ETH-USD on 1h timeframe
2024-12-19 10:15:31 - strategy_manager - DEBUG - Updated MACD metrics for ETH-USD on 1h timeframe
```

## Migration from Previous Versions

### Updating Existing Configurations

If you have existing indicator configurations without timeframes:

**Before:**

```json
{
  "name": "eth_rsi",
  "type": "rsi",
  "enabled": true,
  "parameters": {
    "period": 14
  }
}
```

**After:**

```json
{
  "name": "eth_rsi_4h",
  "type": "rsi",
  "enabled": true,
  "timeframe": "4h",
  "parameters": {
    "period": 14
  }
}
```

### Backward Compatibility

The system maintains backward compatibility:

- Indicators without explicit timeframes default to `1h`
- Existing naming conventions continue to work
- Legacy code using `interval` attribute is supported

## Best Practices

### 1. Naming Conventions

Include timeframes in indicator names for clarity:

```json
{
  "name": "btc_macd_4h", // Good: Clear timeframe
  "name": "eth_rsi_daily", // Good: Descriptive timeframe
  "name": "macd_indicator" // Avoid: No timeframe information
}
```

### 2. Timeframe Selection

Choose appropriate timeframes for different strategies:

- **Scalping**: 1m, 5m
- **Day Trading**: 15m, 1h, 4h
- **Swing Trading**: 4h, 1d
- **Position Trading**: 1d, 1w

### 3. Multi-Timeframe Analysis

Combine multiple timeframes for comprehensive analysis:

```json
{
  "indicators": [
    {
      "name": "trend_daily",
      "type": "ma",
      "timeframe": "1d",
      "parameters": { "period": 50 }
    },
    {
      "name": "entry_4h",
      "type": "rsi",
      "timeframe": "4h",
      "parameters": { "period": 14 }
    },
    {
      "name": "timing_1h",
      "type": "macd",
      "timeframe": "1h",
      "parameters": { "fast_period": 12, "slow_period": 26 }
    }
  ]
}
```

## Troubleshooting

### Common Issues

1. **Indicator not receiving data**

   - Check that the timeframe is supported
   - Verify data is available for the specified timeframe
   - Check logs for data fetching errors

2. **Metrics not appearing in Grafana**

   - Ensure timeframe labels match in queries
   - Check Prometheus targets are scraping correctly
   - Verify metric names include timeframe labels

3. **Backtesting errors**
   - Confirm data source supports the timeframe
   - Check date ranges have sufficient data
   - Verify indicator requirements are met

### Log Analysis

Monitor logs for timeframe-related information:

```bash
# Check indicator initialization
grep "timeframe" packages/spark-app/_logs/spark_stacker.log

# Monitor data fetching
grep "Fetching.*historical.*candles" packages/spark-app/_logs/spark_stacker.log

# Check metric updates
grep "Updated.*metrics.*timeframe" packages/spark-app/_logs/spark_stacker.log
```

## API Reference

### BaseIndicator Methods

```python
class BaseIndicator:
    def get_effective_timeframe(self) -> str:
        """Get the effective timeframe for this indicator."""

    def set_timeframe(self, timeframe: str) -> None:
        """Set the timeframe for this indicator."""
```

### Configuration Schema

```typescript
interface IndicatorConfig {
  name: string;
  type: string;
  enabled: boolean;
  timeframe?: string; // Optional, defaults to "1h"
  parameters: Record<string, any>;
}

interface StrategyConfig {
  name: string;
  market: string;
  enabled: boolean;
  timeframe?: string; // Optional default for strategy indicators
  // ... other strategy fields
}

interface BacktestingConfig {
  default_timeframe: string;
  enable_multi_timeframe: boolean;
  supported_timeframes: string[];
  // ... other backtesting fields
}
```

This unified timeframe system provides flexible, consistent configuration across all components
while maintaining backward compatibility and enabling powerful multi-timeframe analysis strategies.
