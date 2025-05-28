# Strategy-Specific Position Sizing Guide - Spark Stacker Trading System

This guide explains how to configure and use strategy-specific position sizing in the Spark Stacker
system, allowing each trading strategy to use different position sizing methods and parameters while
maintaining proper risk management.

## Overview

The Spark Stacker system supports **strategy-specific position sizing**, which means each trading
strategy can have its own position sizing method and parameters. This allows for sophisticated
portfolio management where different strategies can use different risk approaches based on their
characteristics.

### Key Benefits

1. **Risk Diversification**: Different strategies can have different risk profiles
2. **Strategy Optimization**: Position sizing can be tailored to each strategy's characteristics
3. **Flexible Risk Management**: Conservative strategies can use smaller positions, aggressive
   strategies can use larger ones
4. **Inheritance**: Strategies inherit global defaults but can override specific parameters
5. **Dynamic Allocation**: Position sizes can be adjusted per strategy without affecting others

## Position Sizing Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    GLOBAL CONFIGURATION                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ position_sizing: {                                     │   │
│  │   method: "fixed_usd",                                 │   │
│  │   fixed_amount_usd: 1000,                              │   │
│  │   max_position_size_usd: 5000                          │   │
│  │ }                                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ inherits defaults
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STRATEGY-SPECIFIC OVERRIDES                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Strategy A      │  │ Strategy B      │  │ Strategy C      │ │
│  │ position_sizing:│  │ position_sizing:│  │ (uses global)   │ │
│  │ {               │  │ {               │  │                 │ │
│  │  method:"risk_  │  │  method:"percent│  │                 │ │
│  │  based",        │  │  _equity",      │  │                 │ │
│  │  risk_per_trade:│  │  percent_equity:│  │                 │ │
│  │  0.03           │  │  0.15           │  │                 │ │
│  │ }               │  │ }               │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ creates position sizers
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RISK MANAGER                              │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ strategy_position_sizers: {                          │      │
│  │   "strategy_a": RiskBasedPositionSizer(...),         │      │
│  │   "strategy_b": PercentEquityPositionSizer(...),     │      │
│  │   "strategy_c": FixedUSDPositionSizer(...)           │      │
│  │ }                                                     │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Structure

### Global Position Sizing

Define default position sizing behavior at the root level:

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

### Strategy-Specific Overrides

Override global defaults in individual strategies:

```json
{
  "strategies": [
    {
      "name": "eth_conservative_strategy",
      "market": "ETH-USD",
      "indicators": ["rsi_4h"]
      // Uses global position sizing (fixed_usd $1000)
    },
    {
      "name": "eth_aggressive_strategy",
      "market": "ETH-USD",
      "indicators": ["macd_1h"],
      "position_sizing": {
        "method": "risk_based",
        "risk_per_trade_pct": 0.03,
        "max_position_size_usd": 8000
      }
    }
  ]
}
```

## Position Sizing Methods

### 1. Fixed USD Amount (`fixed_usd`)

Trade a consistent dollar amount regardless of account size or market conditions.

**Configuration:**

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

- Conservative trading strategies
- Beginner strategies with predictable exposure
- Strategies where consistent position size is important
- Testing new strategies with limited capital at risk

**Example Strategy:**

```json
{
  "name": "btc_scalping_strategy",
  "market": "BTC-USD",
  "indicators": ["rsi_15m"],
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_amount_usd": 500,
    "max_position_size_usd": 2000
  }
}
```

### 2. Risk-Based Sizing (`risk_based`)

Size positions based on a percentage of total portfolio value at risk.

**Configuration:**

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

**Calculation:**

```
Position Size = (Portfolio Value × Risk Per Trade %) / Stop Loss Distance
```

**Use Cases:**

- Professional risk management
- Strategies that should scale with account growth
- Maintaining consistent risk exposure across different market conditions
- Long-term growth strategies

**Example Strategy:**

```json
{
  "name": "eth_momentum_strategy",
  "market": "ETH-USD",
  "indicators": ["rsi_4h", "macd_1h"],
  "position_sizing": {
    "method": "risk_based",
    "risk_per_trade_pct": 0.025,
    "max_position_size_usd": 15000
  },
  "stop_loss_pct": 3.0
}
```

### 3. Percent Equity (`percent_equity`)

Size positions as a fixed percentage of total account equity.

**Configuration:**

```json
{
  "position_sizing": {
    "method": "percent_equity",
    "percent_equity": 0.1,
    "max_position_size_usd": 20000
  }
}
```

**Parameters:**

- `percent_equity`: Percentage of total equity per position (e.g., 0.1 = 10%)
- `max_position_size_usd`: Maximum position size limit

**Calculation:**

```
Position Size = Account Equity × Percent Equity
```

**Use Cases:**

- Growth-oriented strategies
- Strategies that should scale with account size
- Portfolio allocation strategies
- High-conviction strategies

**Example Strategy:**

```json
{
  "name": "avax_trend_following",
  "market": "AVAX-USD",
  "indicators": ["avax_ma_1d", "avax_rsi_4h"],
  "position_sizing": {
    "method": "percent_equity",
    "percent_equity": 0.15,
    "max_position_size_usd": 25000
  }
}
```

## Parameter Inheritance

### How Inheritance Works

Strategy-specific position sizing inherits missing parameters from global configuration:

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
      "name": "strategy_a",
      "position_sizing": {
        "method": "risk_based"
        // Inherits: risk_per_trade_pct: 0.02, max_position_size_usd: 5000
      }
    },
    {
      "name": "strategy_b",
      "position_sizing": {
        "fixed_amount_usd": 2000
        // Inherits: method: "fixed_usd", max_position_size_usd: 5000
      }
    },
    {
      "name": "strategy_c"
      // Inherits entire global position_sizing config
    }
  ]
}
```

### Inheritance Rules

1. **Method inheritance**: If strategy doesn't specify method, uses global method
2. **Parameter inheritance**: Missing parameters are inherited from global config
3. **Override priority**: Strategy-specific parameters always override global ones
4. **Validation**: Final configuration is validated for completeness

### Example: Mixed Inheritance

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
      "name": "conservative_strategy"
      // Uses: method="fixed_usd", fixed_amount_usd=1000, max_position_size_usd=5000
    },
    {
      "name": "moderate_strategy",
      "position_sizing": {
        "method": "risk_based"
        // Uses: method="risk_based", risk_per_trade_pct=0.02, max_position_size_usd=5000
      }
    },
    {
      "name": "aggressive_strategy",
      "position_sizing": {
        "method": "percent_equity",
        "percent_equity": 0.2,
        "max_position_size_usd": 15000
        // Uses: method="percent_equity", percent_equity=0.2, max_position_size_usd=15000
      }
    }
  ]
}
```

## Advanced Configuration Patterns

### Pattern 1: Risk-Tiered Portfolio

Different strategies with increasing risk levels:

```json
{
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_amount_usd": 1000,
    "max_position_size_usd": 5000
  },
  "strategies": [
    {
      "name": "conservative_btc",
      "market": "BTC-USD",
      "position_sizing": {
        "method": "fixed_usd",
        "fixed_amount_usd": 500
      }
    },
    {
      "name": "moderate_eth",
      "market": "ETH-USD",
      "position_sizing": {
        "method": "risk_based",
        "risk_per_trade_pct": 0.02
      }
    },
    {
      "name": "aggressive_avax",
      "market": "AVAX-USD",
      "position_sizing": {
        "method": "percent_equity",
        "percent_equity": 0.1
      }
    }
  ]
}
```

### Pattern 2: Timeframe-Based Sizing

Different position sizes based on strategy timeframes:

```json
{
  "strategies": [
    {
      "name": "btc_scalping_1m",
      "market": "BTC-USD",
      "timeframe": "1m",
      "position_sizing": {
        "method": "fixed_usd",
        "fixed_amount_usd": 200,
        "max_position_size_usd": 1000
      }
    },
    {
      "name": "swing_4h",
      "market": "ETH-USD",
      "timeframe": "4h",
      "position_sizing": {
        "method": "risk_based",
        "risk_per_trade_pct": 0.03,
        "max_position_size_usd": 5000
      }
    },
    {
      "name": "btc_position_1d",
      "market": "BTC-USD",
      "timeframe": "1d",
      "position_sizing": {
        "method": "percent_equity",
        "percent_equity": 0.2,
        "max_position_size_usd": 20000
      }
    }
  ]
}
```

### Pattern 3: Exchange-Specific Sizing

Different position sizes for different exchanges:

```json
{
  "strategies": [
    {
      "name": "eth_hyperliquid_main",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "position_sizing": {
        "method": "percent_equity",
        "percent_equity": 0.15,
        "max_position_size_usd": 10000
      }
    },
    {
      "name": "eth_coinbase_hedge",
      "market": "ETH-USD",
      "exchange": "coinbase",
      "position_sizing": {
        "method": "fixed_usd",
        "fixed_amount_usd": 2000,
        "max_position_size_usd": 5000
      }
    }
  ]
}
```

### Pattern 4: Dynamic Risk Adjustment

Adjust risk based on strategy performance (configuration-based approach):

```json
{
  "strategies": [
    {
      "name": "eth_adaptive_strategy",
      "market": "ETH-USD",
      "position_sizing": {
        "method": "risk_based",
        "risk_per_trade_pct": 0.02,
        "max_position_size_usd": 8000
      },
      "risk_adjustment": {
        "enabled": true,
        "performance_window": 30,
        "min_risk_pct": 0.01,
        "max_risk_pct": 0.04
      }
    }
  ]
}
```

## Implementation Details

### Position Sizer Creation

The system creates position sizers for each strategy during initialization:

```python
class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.global_position_sizing = config.get("position_sizing", {})
        self.strategy_position_sizers: Dict[str, PositionSizer] = {}

    def _create_strategy_position_sizers(self, strategies: List[StrategyConfig]) -> None:
        """Create position sizers for each strategy."""
        for strategy in strategies:
            position_sizer = self._create_position_sizer_for_strategy(strategy)
            self.strategy_position_sizers[strategy.name] = position_sizer

    def _create_position_sizer_for_strategy(self, strategy: StrategyConfig) -> PositionSizer:
        """Create position sizer for a specific strategy with inheritance."""
        # Start with global defaults
        config = self.global_position_sizing.copy()

        # Override with strategy-specific settings
        if hasattr(strategy, 'position_sizing') and strategy.position_sizing:
            config.update(strategy.position_sizing)

        # Create position sizer based on method
        method = config.get("method", "fixed_usd")
        if method == "fixed_usd":
            return FixedUSDPositionSizer(config)
        elif method == "risk_based":
            return RiskBasedPositionSizer(config)
        elif method == "percent_equity":
            return PercentEquityPositionSizer(config)
        else:
            raise ValueError(f"Unknown position sizing method: {method}")
```

### Strategy Context in Trading

Position sizing is called with strategy context during trade execution:

```python
class TradingEngine:
    async def _execute_trade(self, signal: Signal, connector: BaseConnector, exchange_symbol: str) -> bool:
        """Execute trade with strategy-specific position sizing."""
        # Get strategy-specific position size
        position_size = self.risk_manager.calculate_position_size(
            signal=signal,
            current_price=current_price,
            strategy_name=signal.strategy_name  # Strategy context
        )

        # Execute trade with calculated position size
        return await connector.execute_trade(
            symbol=exchange_symbol,
            side=signal.direction,
            size=position_size,
            leverage=leverage
        )
```

## Testing Strategy Position Sizing

### Configuration Validation

Test that position sizing configurations are valid:

```bash
cd packages/spark-app
.venv/bin/python -c "
import json
from app.core.strategy_config import StrategyConfigLoader
from app.risk_management.risk_manager import RiskManager

# Load configuration
with open('../shared/config.json') as f:
    config = json.load(f)

# Test strategy loading
strategies = StrategyConfigLoader.load_strategies(config['strategies'])
print(f'✅ Loaded {len(strategies)} strategies')

# Test risk manager creation
risk_manager = RiskManager.from_config(config)
print(f'✅ Created risk manager with {len(risk_manager.strategy_position_sizers)} strategy position sizers')

# Test position sizer creation for each strategy
for strategy in strategies:
    sizer = risk_manager.strategy_position_sizers.get(strategy.name)
    print(f'✅ Strategy {strategy.name}: {sizer.config.method.value} position sizing')
"
```

### Position Size Calculation Testing

Test position size calculations for different methods:

```bash
cd packages/spark-app
.venv/bin/python -c "
from app.risk_management.risk_manager import RiskManager
from app.indicators.base_indicator import Signal, SignalDirection
import json

# Load configuration
with open('../shared/config.json') as f:
    config = json.load(f)

risk_manager = RiskManager.from_config(config)

# Create test signal
signal = Signal(
    direction=SignalDirection.LONG,
    strength=1.0,
    strategy_name='eth_momentum_strategy',
    market='ETH-USD'
)

# Test position size calculation
position_size = risk_manager.calculate_position_size(
    signal=signal,
    current_price=3000.0,
    strategy_name='eth_momentum_strategy'
)

print(f'✅ Calculated position size: ${position_size:.2f}')
"
```

### Backtesting with Strategy Position Sizing

Test strategies with different position sizing methods:

```bash
cd packages/spark-app
.venv/bin/python -m tests._utils.cli backtest-strategy \
  --strategy eth_momentum_strategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-01 \
  --include-position-sizing
```

## Monitoring and Performance

### Key Metrics to Track

1. **Position Size Distribution**: How position sizes vary across strategies
2. **Risk Utilization**: Actual risk taken vs. target risk per strategy
3. **Capital Allocation**: How capital is distributed across strategies
4. **Position Size Efficiency**: Whether position sizes are appropriate for returns

### Grafana Monitoring

Monitor strategy-specific position sizing:

```promql
# Position sizes by strategy
spark_stacker_position_size{strategy_name=~".*"}

# Risk utilization by strategy
spark_stacker_risk_utilized{strategy_name=~".*"} / spark_stacker_target_risk{strategy_name=~".*"}

# Capital allocation across strategies
sum by (strategy_name) (spark_stacker_position_size{strategy_name=~".*"})
```

### Performance Analysis

Compare performance across different position sizing methods:

```python
# Analyze position sizing effectiveness
position_sizing_analysis = {
    "fixed_usd_strategies": {
        "avg_return_pct": 2.1,
        "max_drawdown_pct": 5.2,
        "sharpe_ratio": 1.3
    },
    "risk_based_strategies": {
        "avg_return_pct": 3.4,
        "max_drawdown_pct": 8.1,
        "sharpe_ratio": 1.7
    },
    "percent_equity_strategies": {
        "avg_return_pct": 4.2,
        "max_drawdown_pct": 12.3,
        "sharpe_ratio": 1.4
    }
}
```

## Best Practices

### Configuration Guidelines

1. **Start Conservative**: Begin with fixed USD amounts or low percentages
2. **Test Thoroughly**: Backtest each position sizing method before going live
3. **Set Maximum Limits**: Always specify `max_position_size_usd` to prevent oversized positions
4. **Use Inheritance**: Leverage global defaults to reduce configuration complexity
5. **Document Rationale**: Comment on why specific position sizing methods were chosen

### Risk Management Guidelines

1. **Diversify Methods**: Use different position sizing methods for different strategy types
2. **Monitor Correlation**: Ensure strategies with large positions aren't highly correlated
3. **Regular Review**: Periodically review and adjust position sizing parameters
4. **Performance Tracking**: Monitor how position sizing affects strategy performance
5. **Emergency Limits**: Have circuit breakers for extreme market conditions

### Development Guidelines

1. **Validate Configurations**: Test position sizing configurations before deployment
2. **Handle Edge Cases**: Account for zero balance, extreme prices, etc.
3. **Log Position Calculations**: Log position size calculations for debugging
4. **Test Inheritance**: Verify parameter inheritance works correctly
5. **Monitor Performance**: Track position sizing effectiveness over time

## Troubleshooting

### Common Issues

**Issue**: "Position size calculation failed for strategy X" **Cause**: Missing or invalid position
sizing configuration **Solution**: Verify strategy has valid position sizing config or falls back to
global

**Issue**: "Position size exceeds maximum limit" **Cause**: Calculated position size is larger than
`max_position_size_usd` **Solution**: Adjust position sizing parameters or increase maximum limit

**Issue**: "Strategy not found in position sizers" **Cause**: Strategy not properly registered
during risk manager initialization **Solution**: Verify strategy configuration is valid and position
sizer is created

### Debugging

Enable position sizing logging:

```python
import logging
logging.getLogger("app.risk_management.risk_manager").setLevel(logging.DEBUG)
```

Check position sizing calculations in logs:

```bash
grep "Position size calculated" packages/spark-app/_logs/spark_stacker.log
grep "strategy_position_sizers" packages/spark-app/_logs/spark_stacker.log
```

## Migration Guide

### Migrating from Global to Strategy-Specific

If you currently use only global position sizing and want to add strategy-specific sizing:

1. **Keep Global Config**: Maintain existing global position sizing as defaults
2. **Add Strategy Overrides**: Add position sizing to specific strategies that need different
   behavior
3. **Test Gradually**: Start with one strategy override and verify it works correctly
4. **Monitor Changes**: Watch for changes in position sizes and performance

**Example Migration:**

**Before (Global Only):**

```json
{
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_amount_usd": 1000
  },
  "strategies": [{ "name": "strategy_a" }, { "name": "strategy_b" }, { "name": "strategy_c" }]
}
```

**After (Mixed):**

```json
{
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_amount_usd": 1000
  },
  "strategies": [
    { "name": "strategy_a" },
    {
      "name": "strategy_b",
      "position_sizing": {
        "method": "risk_based",
        "risk_per_trade_pct": 0.02
      }
    },
    { "name": "strategy_c" }
  ]
}
```

This strategy-specific position sizing guide provides a comprehensive framework for implementing
sophisticated risk management across multiple trading strategies while maintaining simplicity and
flexibility.
