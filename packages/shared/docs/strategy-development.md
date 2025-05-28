# Strategy Development Guide - Spark Stacker Trading System

This guide explains how to develop and add new trading strategies to the Spark Stacker system,
covering everything from configuration to testing and deployment.

## Overview

In the Spark Stacker architecture, strategies are **configuration-driven** rather than code-driven.
This means you can create new strategies by:

1. **Adding strategy configuration** to `config.json`
2. **Creating required indicators** (if they don't exist)
3. **Testing the strategy** with backtesting
4. **Deploying** with proper monitoring

## Strategy Architecture

### Core Concepts

```
┌─────────────────────────────────────────────────────────────────┐
│                     STRATEGY LAYER                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Configuration-driven (JSON)                             │   │
│  │ • Market selection (ETH-USD, BTC-USD)                   │   │
│  │ • Indicator selection (RSI, MACD, MA)                   │   │
│  │ • Risk management (position sizing, stops)              │   │
│  │ • Timeframe coordination                                │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ references
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INDICATOR LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ RSI Module   │  │ MACD Module  │  │  MA Module   │          │
│  │ (Code-based) │  │ (Code-based) │  │ (Code-based) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ processes
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                 │
│  Market data from exchanges (Hyperliquid, Coinbase, etc.)      │
└─────────────────────────────────────────────────────────────────┘
```

### Strategy Types

Strategies can be categorized by their approach:

1. **Single-Timeframe Strategies**: Use one timeframe for all analysis
2. **Multi-Timeframe Strategies**: Combine multiple timeframes (trend + momentum + entry)
3. **Multi-Indicator Strategies**: Use multiple indicators on same timeframe
4. **Market-Neutral Strategies**: Trade across different markets
5. **Exchange-Arbitrage Strategies**: Exploit price differences between exchanges

## Creating a New Strategy

### Step 1: Strategy Planning

Before configuration, plan your strategy:

```markdown
# Example: ETH Momentum Strategy

**Objective**: Capture ETH momentum moves using multi-timeframe analysis

**Timeframes**:

- Daily (1d): Overall trend direction (MA crossover)
- 4-hour (4h): Momentum confirmation (RSI)
- 1-hour (1h): Entry timing (MACD)

**Logic**:

1. Daily MA confirms uptrend (short MA > long MA)
2. 4h RSI shows momentum (> 50 but < 70)
3. 1h MACD shows bullish crossover

**Risk Management**:

- Position sizing: Risk-based (2% per trade)
- Stop loss: 3% below entry
- Take profit: 6% above entry
- Max position: $5000
```

### Step 2: Indicator Setup

Ensure required indicators exist or create them:

```json
{
  "indicators": [
    {
      "name": "eth_trend_1d",
      "type": "ma",
      "timeframe": "1d",
      "enabled": true,
      "parameters": {
        "short_period": 20,
        "long_period": 50,
        "ma_type": "sma"
      }
    },
    {
      "name": "eth_momentum_4h",
      "type": "rsi",
      "timeframe": "4h",
      "enabled": true,
      "parameters": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
      }
    },
    {
      "name": "eth_entry_1h",
      "type": "macd",
      "timeframe": "1h",
      "enabled": true,
      "parameters": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
      }
    }
  ]
}
```

### Step 3: Strategy Configuration

Add the strategy to your configuration:

```json
{
  "strategies": [
    {
      "name": "eth_momentum_strategy",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "enabled": true,
      "timeframe": "1h",
      "indicators": ["eth_trend_1d", "eth_momentum_4h", "eth_entry_1h"],
      "main_leverage": 2.0,
      "stop_loss_pct": 3.0,
      "take_profit_pct": 6.0,
      "max_position_size": 0.1,
      "risk_per_trade_pct": 0.02,
      "position_sizing": {
        "method": "risk_based",
        "risk_per_trade_pct": 0.02,
        "max_position_size_usd": 5000
      }
    }
  ]
}
```

### Step 4: Configuration Validation

Validate your configuration before testing:

```bash
cd packages/spark-app
.venv/bin/python -c "
import json
from app.core.strategy_config import StrategyConfigLoader
from app.indicators.indicator_factory import IndicatorFactory

# Load configuration
with open('../shared/config.json') as f:
    config = json.load(f)

# Validate strategy loading
strategies = StrategyConfigLoader.load_strategies(config['strategies'])
print(f'✅ Loaded {len(strategies)} strategies')

# Validate indicator loading
indicators = IndicatorFactory.create_indicators_from_config(config['indicators'])
print(f'✅ Created {len(indicators)} indicators')

# Validate strategy-indicator relationships
StrategyConfigLoader.validate_indicators(strategies, indicators)
print('✅ All strategy-indicator relationships valid')

# Test specific strategy
eth_strategy = next(s for s in strategies if s.name == 'eth_momentum_strategy')
print(f'✅ Strategy {eth_strategy.name} configured for {eth_strategy.market}')
print(f'   Indicators: {eth_strategy.indicators}')
print(f'   Position sizing: {eth_strategy.position_sizing}')
"
```

## Advanced Strategy Patterns

### Pattern 1: Market Regime Strategy

Adapt behavior based on market conditions:

```json
{
  "name": "eth_adaptive_strategy",
  "market": "ETH-USD",
  "exchange": "hyperliquid",
  "indicators": ["eth_volatility_1d", "eth_trend_strength_4h", "eth_momentum_1h"],
  "position_sizing": {
    "method": "risk_based",
    "risk_per_trade_pct": 0.015
  }
}
```

### Pattern 2: Cross-Market Strategy

Trade multiple markets with shared logic:

```json
{
  "strategies": [
    {
      "name": "crypto_momentum_eth",
      "market": "ETH-USD",
      "indicators": ["crypto_momentum_4h", "eth_entry_1h"]
    },
    {
      "name": "crypto_momentum_btc",
      "market": "BTC-USD",
      "indicators": ["crypto_momentum_4h", "btc_entry_1h"]
    }
  ]
}
```

### Pattern 3: Exchange Arbitrage Strategy

Exploit price differences between exchanges:

```json
{
  "strategies": [
    {
      "name": "eth_hyperliquid_long",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "indicators": ["eth_arbitrage_signal"]
    },
    {
      "name": "eth_coinbase_short",
      "market": "ETH-USD",
      "exchange": "coinbase",
      "indicators": ["eth_arbitrage_signal"]
    }
  ]
}
```

### Pattern 4: Dynamic Position Sizing

Adjust position sizing based on strategy performance:

```json
{
  "name": "eth_dynamic_strategy",
  "market": "ETH-USD",
  "indicators": ["eth_rsi_4h"],
  "position_sizing": {
    "method": "risk_based",
    "risk_per_trade_pct": 0.02,
    "max_position_size_usd": 10000
  }
}
```

## Testing Strategies

### Backtesting Individual Indicators

Test each indicator used in your strategy:

```bash
cd packages/spark-app
.venv/bin/python -m tests._utils.cli backtest-indicator \
  --indicator eth_momentum_4h \
  --symbol ETH-USD \
  --timeframe 4h \
  --start-date 2024-01-01 \
  --end-date 2024-12-01
```

### Strategy Integration Testing

Test the complete strategy configuration:

```bash
cd packages/spark-app
.venv/bin/python -c "
from app.core.strategy_manager import StrategyManager
from app.core.trading_engine import TradingEngine
import json

# Load configuration
with open('../shared/config.json') as f:
    config = json.load(f)

# Test strategy manager initialization
trading_engine = TradingEngine()
strategy_manager = StrategyManager(
    trading_engine=trading_engine,
    strategies=config['strategies'],
    config=config
)

print('✅ Strategy manager initialized successfully')
print(f'Loaded strategies: {[s[\"name\"] for s in config[\"strategies\"]]}')
"
```

### Dry Run Testing

Test with live market data but without real trades:

```json
{
  "dry_run": true,
  "log_level": "DEBUG",
  "strategies": [
    {
      "name": "eth_momentum_strategy",
      "enabled": true
    }
  ]
}
```

```bash
cd packages/spark-app
.venv/bin/python app/main.py
```

## Strategy Naming Conventions

### Naming Best Practices

**Strategy Names:**

- Format: `{market}_{strategy_type}[_{variant}]`
- Examples:
  - `eth_momentum_strategy`
  - `btc_scalping_15m`
  - `multi_asset_arbitrage`
  - `eth_conservative_long`

**Indicator Names:**

- Format: `{market}_{indicator_type}_{timeframe}`
- Examples:
  - `eth_rsi_4h`
  - `btc_macd_1h`
  - `avax_bb_15m`
  - `crypto_volatility_1d`

### Organization Structure

```json
{
  "strategies": [
    // ETH Strategies
    {
      "name": "eth_momentum_strategy",
      "market": "ETH-USD"
    },
    {
      "name": "eth_scalping_strategy",
      "market": "ETH-USD"
    },

    // BTC Strategies
    {
      "name": "btc_trend_strategy",
      "market": "BTC-USD"
    },

    // Multi-Asset Strategies
    {
      "name": "crypto_momentum_portfolio",
      "market": "MULTI"
    }
  ],

  "indicators": [
    // ETH Indicators
    { "name": "eth_rsi_4h", "type": "rsi", "timeframe": "4h" },
    { "name": "eth_macd_1h", "type": "macd", "timeframe": "1h" },

    // BTC Indicators
    { "name": "btc_ma_1d", "type": "ma", "timeframe": "1d" },

    // Shared Indicators
    { "name": "crypto_volatility_1d", "type": "volatility", "timeframe": "1d" }
  ]
}
```

## Strategy Performance Monitoring

### Key Metrics to Track

1. **Signal Generation Rate**: How often the strategy generates signals
2. **Win Rate**: Percentage of profitable trades
3. **Risk-Adjusted Returns**: Sharpe ratio, Calmar ratio
4. **Maximum Drawdown**: Largest peak-to-trough decline
5. **Position Sizing Effectiveness**: Actual vs. target position sizes

### Monitoring Setup

```bash
# Start monitoring infrastructure
cd packages/monitoring
make start-monitoring

# View strategy-specific metrics in Grafana
# http://localhost:3000/d/strategy-performance
```

### Custom Metrics

Add strategy-specific metrics to track performance:

```python
# In your strategy configuration or custom indicators
metrics = {
    "strategy_name": "eth_momentum_strategy",
    "signals_generated": 15,
    "win_rate": 0.67,
    "avg_return_pct": 2.3,
    "max_drawdown_pct": 4.1
}
```

## Advanced Strategy Development

### Custom Indicator Development

If existing indicators don't meet your needs, create custom ones:

```python
# packages/spark-app/app/indicators/custom_momentum_indicator.py
from app.indicators.base_indicator import BaseIndicator, Signal, SignalDirection

class CustomMomentumIndicator(BaseIndicator):
    def __init__(self, period: int = 14, threshold: float = 0.02):
        super().__init__()
        self.period = period
        self.threshold = threshold

    def process(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Signal]]:
        # Custom momentum calculation
        momentum = data['close'].pct_change(self.period)

        # Generate signal based on momentum threshold
        if momentum.iloc[-1] > self.threshold:
            signal = Signal(
                direction=SignalDirection.LONG,
                strength=min(momentum.iloc[-1] / self.threshold, 2.0),
                symbol=self.symbol,
                timestamp=data.index[-1],
                indicator_name=self.name
            )
            return data, signal

        return data, None
```

### Strategy Optimization

Use backtesting to optimize strategy parameters:

```python
# Strategy parameter optimization
optimization_results = {}

for rsi_period in [10, 14, 18, 22]:
    for rsi_threshold in [60, 65, 70, 75]:
        # Update strategy configuration
        strategy_config = {
            "name": f"eth_rsi_opt_{rsi_period}_{rsi_threshold}",
            "market": "ETH-USD",
            "indicators": [f"eth_rsi_{rsi_period}_{rsi_threshold}"]
        }

        # Run backtest
        results = run_backtest(strategy_config)
        optimization_results[(rsi_period, rsi_threshold)] = results

# Find best parameters
best_params = max(optimization_results.items(),
                 key=lambda x: x[1]['sharpe_ratio'])
```

## Deployment Best Practices

### Pre-Deployment Checklist

- [ ] Strategy configuration validated
- [ ] All indicators tested individually
- [ ] Backtesting completed with positive results
- [ ] Dry run testing successful
- [ ] Position sizing appropriate for account size
- [ ] Risk management parameters set
- [ ] Monitoring dashboards configured
- [ ] Alert thresholds defined

### Gradual Deployment

1. **Start Small**: Use minimal position sizes
2. **Monitor Closely**: Watch first few trades carefully
3. **Scale Gradually**: Increase position sizes as confidence grows
4. **Document Performance**: Track actual vs. backtested results

### Risk Management

```json
{
  "name": "new_strategy",
  "enabled": true,
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_amount_usd": 100, // Start very small
    "max_position_size_usd": 1000
  },
  "stop_loss_pct": 2.0, // Tight stops initially
  "take_profit_pct": 4.0
}
```

## Troubleshooting Common Issues

### Configuration Errors

**"Strategy references unknown indicator"**

```bash
# Check indicator names match exactly
grep -A 5 -B 5 "indicator_name" packages/shared/config.json
```

**"Invalid market symbol format"**

```bash
# Ensure market uses format like "ETH-USD"
# Not: "ETH", "RSI-4H", or other formats
```

### Performance Issues

**Strategy not generating signals**

```bash
# Check if indicators are enabled and processing
grep "Processing indicator" packages/spark-app/_logs/spark_stacker.log

# Verify market data is being fetched
grep "Fetching.*historical.*candles" packages/spark-app/_logs/spark_stacker.log
```

**Position sizing errors**

```bash
# Validate position sizing configuration
python -c "
from app.risk_management.risk_manager import RiskManager
risk_manager = RiskManager.from_config(config)
print('Position sizing methods:', list(risk_manager.strategy_position_sizers.keys()))
"
```

This strategy development guide provides a comprehensive framework for creating, testing, and
deploying new trading strategies in the Spark Stacker system. The key is to start simple, test
thoroughly, and scale gradually while maintaining proper risk management.
