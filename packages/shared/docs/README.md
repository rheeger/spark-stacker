# Spark Stacker Documentation

Welcome to the Spark Stacker trading system documentation.

## 🚨 **URGENT: Architecture Fixes Required**

- [**⚡ IMMEDIATE FIXES SUMMARY**](./IMMEDIATE-FIXES-SUMMARY.md) - **Quick fixes for "Market RSI-4H
  not found" error**
- [**🔧 Architectural Fixes**](./architectural-fixes.md) - **Complete code changes needed**

## Table of Contents

- [**🏗️ Configuration Guide**](./configuration.md) - **Complete system configuration with
  architecture explanations**
- [User Guide](./userguide.md) - Step-by-step setup and usage guide
- [**🕒 Timeframe Configuration**](./timeframe-configuration.md) - **Configure indicators and
  strategies with different timeframes**
- [Position Sizing](./position-sizing.md) - Risk management and position sizing strategies
- [Connectors](./connectors.md) - Exchange connector documentation
- [Indicators](./indicators.md) - Technical indicator documentation
- [Logging](./logging.md) - Logging configuration and best practices
- [Checklists](./checklists/) - Development phase checklists

## 🕒 New: Multi-Timeframe Trading

The Spark Stacker system now supports **unified timeframe configuration** across all components:

### Quick Start

1. **Configure indicators with specific timeframes:**

```json
{
  "indicators": [
    {
      "name": "rsi_4h",
      "type": "rsi",
      "enabled": true,
      "timeframe": "4h",
      "parameters": { "period": 14 }
    },
    {
      "name": "macd_1h",
      "type": "macd",
      "enabled": true,
      "timeframe": "1h",
      "parameters": { "fast_period": 12, "slow_period": 26 }
    }
  ]
}
```

2. **Run multi-timeframe backtests:**

```bash
packages/spark-app/.venv/bin/python -m tests._utils.cli backtest-indicator \
  --indicator rsi_4h \
  --symbol ETH-USD \
  --timeframe 1h
```

3. **Monitor timeframe-specific metrics in Grafana:**

```promql
spark_stacker_candle{market="ETH-USD", timeframe="4h", field="close"}
spark_stacker_macd{market="ETH-USD", timeframe="1h", component="macd_line"}
```

### Key Features

- ✅ **Indicator-level timeframes** - Each indicator can run on its own timeframe
- ✅ **Strategy-level defaults** - Set default timeframes for strategies
- ✅ **Backtesting alignment** - Same configuration for live trading and backtesting
- ✅ **Multi-timeframe analysis** - Run multiple indicators on different timeframes simultaneously
- ✅ **Backward compatibility** - Existing configurations continue to work

### Migration

If you have existing indicators without timeframes, simply add the `timeframe` field:

```json
{
  "name": "rsi_4h",
  "type": "rsi",
  "enabled": true,
  "timeframe": "4h", // ← Add this line
  "parameters": { "period": 14 }
}
```

See the [complete timeframe configuration guide](./timeframe-configuration.md) for detailed
information and examples.

## Example Configurations

- [Multi-Timeframe Trading](../examples/multi-timeframe-config.json) - Complete example with
  multiple timeframes
- [Basic Configuration](../config.json) - Simple single-timeframe setup

## Getting Started

1. Review the [timeframe configuration guide](./timeframe-configuration.md)
2. Copy and modify the [multi-timeframe example](../examples/multi-timeframe-config.json)
3. Test your configuration with backtesting before going live
4. Monitor metrics in Grafana with timeframe-specific queries

For detailed setup instructions, see the specific documentation sections above.
