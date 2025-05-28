# Risk Management Guide - Spark Stacker Trading System

This guide explains the comprehensive risk management framework in the Spark Stacker system,
covering position sizing, stop losses, leverage management, and strategy-specific risk controls.

## Overview

The Spark Stacker system implements a multi-layered risk management approach that operates at both
global and strategy-specific levels. This allows for sophisticated risk control while maintaining
flexibility for different trading strategies.

### Risk Management Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    GLOBAL RISK CONTROLS                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • Maximum portfolio risk                                │   │
│  │ • Overall position size limits                          │   │
│  │ • Default position sizing method                        │   │
│  │ • Emergency stop mechanisms                             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ inherits and overrides
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                STRATEGY-SPECIFIC RISK CONTROLS                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Strategy A      │  │ Strategy B      │  │ Strategy C      │ │
│  │ • Position      │  │ • Position      │  │ • Position      │ │
│  │   sizing method │  │   sizing method │  │   sizing method │ │
│  │ • Stop loss %   │  │ • Stop loss %   │  │ • Stop loss %   │ │
│  │ • Max leverage  │  │ • Max leverage  │  │ • Max leverage  │ │
│  │ • Risk per trade│  │ • Risk per trade│  │ • Risk per trade│ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ implements
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRADE-LEVEL CONTROLS                        │
│  • Pre-trade risk validation                                   │
│  • Position size calculation                                   │
│  • Stop loss placement                                         │
│  • Leverage adjustment                                         │
│  • Real-time monitoring                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Core Risk Management Components

### 1. Position Sizing

The foundation of risk management is proper position sizing, which determines how much capital to
allocate to each trade.

#### Global Position Sizing Configuration

Set default position sizing behavior:

```json
{
  "position_sizing": {
    "method": "risk_based",
    "risk_per_trade_pct": 0.02,
    "max_position_size_usd": 10000,
    "max_portfolio_risk_pct": 0.1
  }
}
```

#### Strategy-Specific Position Sizing

Override global defaults for specific strategies:

```json
{
  "strategies": [
    {
      "name": "conservative_btc_strategy",
      "market": "BTC-USD",
      "position_sizing": {
        "method": "fixed_usd",
        "fixed_amount_usd": 1000,
        "max_position_size_usd": 3000
      }
    },
    {
      "name": "aggressive_eth_strategy",
      "market": "ETH-USD",
      "position_sizing": {
        "method": "percent_equity",
        "percent_equity": 0.15,
        "max_position_size_usd": 20000
      }
    }
  ]
}
```

### 2. Stop Loss Management

Implement systematic stop losses to limit downside risk:

#### Global Stop Loss Configuration

```json
{
  "risk_management": {
    "default_stop_loss_pct": 3.0,
    "trailing_stop_enabled": true,
    "trailing_stop_distance_pct": 2.0,
    "max_stop_loss_pct": 10.0
  }
}
```

#### Strategy-Specific Stop Losses

Different strategies can have different stop loss requirements:

```json
{
  "strategies": [
    {
      "name": "scalping_strategy",
      "stop_loss_pct": 1.0,
      "take_profit_pct": 2.0,
      "trailing_stop_enabled": false
    },
    {
      "name": "swing_strategy",
      "stop_loss_pct": 5.0,
      "take_profit_pct": 10.0,
      "trailing_stop_enabled": true,
      "trailing_stop_distance_pct": 3.0
    }
  ]
}
```

### 3. Leverage Management

Control leverage usage to manage risk exposure:

#### Global Leverage Limits

```json
{
  "risk_management": {
    "max_leverage": 3.0,
    "default_leverage": 1.0,
    "leverage_scaling_enabled": true
  }
}
```

#### Strategy-Specific Leverage

```json
{
  "strategies": [
    {
      "name": "high_conviction_strategy",
      "main_leverage": 2.5,
      "max_leverage": 3.0
    },
    {
      "name": "conservative_strategy",
      "main_leverage": 1.0,
      "max_leverage": 1.5
    }
  ]
}
```

## Position Sizing Methods

### 1. Fixed USD Method

**Best for**: Conservative strategies, beginners, testing

```json
{
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_amount_usd": 1000,
    "max_position_size_usd": 5000
  }
}
```

**Risk Characteristics**:

- Consistent exposure regardless of account size
- Predictable capital allocation
- Limited scalability with account growth

### 2. Risk-Based Method

**Best for**: Professional trading, consistent risk management

```json
{
  "position_sizing": {
    "method": "risk_based",
    "risk_per_trade_pct": 0.02,
    "max_position_size_usd": 15000
  }
}
```

**Risk Characteristics**:

- Consistent risk percentage per trade
- Scales with account size
- Adapts to stop loss distance

**Calculation**:

```
Position Size = (Account Value × Risk %) / Stop Loss Distance %
```

### 3. Percent Equity Method

**Best for**: Growth strategies, portfolio allocation

```json
{
  "position_sizing": {
    "method": "percent_equity",
    "percent_equity": 0.1,
    "max_position_size_usd": 25000
  }
}
```

**Risk Characteristics**:

- Fixed percentage of account per position
- Natural diversification
- Scales with account growth

## Multi-Strategy Risk Management

### Portfolio-Level Risk Controls

Monitor risk across all strategies:

```json
{
  "portfolio_risk": {
    "max_total_exposure_pct": 0.8,
    "max_correlated_exposure_pct": 0.3,
    "rebalancing_threshold_pct": 0.1,
    "emergency_liquidation_threshold_pct": 0.15
  }
}
```

### Strategy Correlation Management

Prevent over-concentration in correlated positions:

```json
{
  "correlation_limits": {
    "max_btc_exposure_pct": 0.4,
    "max_eth_exposure_pct": 0.4,
    "max_altcoin_exposure_pct": 0.3,
    "max_single_timeframe_exposure_pct": 0.5
  }
}
```

### Risk Budgeting

Allocate risk budget across strategies:

```json
{
  "risk_budget": {
    "conservative_strategies": 0.3,
    "moderate_strategies": 0.5,
    "aggressive_strategies": 0.2
  }
}
```

## Advanced Risk Controls

### 1. Dynamic Position Sizing

Adjust position sizes based on market conditions:

```json
{
  "dynamic_sizing": {
    "enabled": true,
    "volatility_adjustment": true,
    "performance_adjustment": true,
    "drawdown_reduction": true,
    "market_regime_adjustment": true
  }
}
```

### 2. Drawdown Protection

Implement circuit breakers for excessive losses:

```json
{
  "drawdown_protection": {
    "max_daily_loss_pct": 5.0,
    "max_weekly_loss_pct": 10.0,
    "max_monthly_loss_pct": 15.0,
    "auto_reduce_positions": true,
    "halt_trading_threshold_pct": 20.0
  }
}
```

### 3. Volatility-Based Adjustments

Adjust risk based on market volatility:

```json
{
  "volatility_adjustment": {
    "enabled": true,
    "lookback_period": 20,
    "min_volatility_multiplier": 0.5,
    "max_volatility_multiplier": 2.0,
    "adjustment_smoothing": 0.1
  }
}
```

## Implementation Examples

### Conservative Multi-Strategy Portfolio

```json
{
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_amount_usd": 1000,
    "max_position_size_usd": 3000
  },
  "risk_management": {
    "default_stop_loss_pct": 2.0,
    "max_leverage": 1.5,
    "max_portfolio_risk_pct": 0.05
  },
  "strategies": [
    {
      "name": "btc_trend_conservative",
      "market": "BTC-USD",
      "stop_loss_pct": 2.0,
      "main_leverage": 1.0,
      "position_sizing": {
        "fixed_amount_usd": 800
      }
    },
    {
      "name": "eth_momentum_conservative",
      "market": "ETH-USD",
      "stop_loss_pct": 2.5,
      "main_leverage": 1.0,
      "position_sizing": {
        "fixed_amount_usd": 800
      }
    }
  ]
}
```

### Balanced Risk Portfolio

```json
{
  "position_sizing": {
    "method": "risk_based",
    "risk_per_trade_pct": 0.02,
    "max_position_size_usd": 10000
  },
  "risk_management": {
    "default_stop_loss_pct": 3.0,
    "max_leverage": 2.0,
    "max_portfolio_risk_pct": 0.08
  },
  "strategies": [
    {
      "name": "btc_swing_moderate",
      "market": "BTC-USD",
      "stop_loss_pct": 4.0,
      "main_leverage": 1.5,
      "position_sizing": {
        "risk_per_trade_pct": 0.025
      }
    },
    {
      "name": "eth_scalping_moderate",
      "market": "ETH-USD",
      "stop_loss_pct": 1.5,
      "main_leverage": 2.0,
      "position_sizing": {
        "risk_per_trade_pct": 0.015
      }
    }
  ]
}
```

### Aggressive Growth Portfolio

```json
{
  "position_sizing": {
    "method": "percent_equity",
    "percent_equity": 0.1,
    "max_position_size_usd": 25000
  },
  "risk_management": {
    "default_stop_loss_pct": 5.0,
    "max_leverage": 3.0,
    "max_portfolio_risk_pct": 0.15
  },
  "strategies": [
    {
      "name": "crypto_momentum_aggressive",
      "market": "ETH-USD",
      "stop_loss_pct": 6.0,
      "main_leverage": 2.5,
      "position_sizing": {
        "percent_equity": 0.15
      }
    },
    {
      "name": "altcoin_breakout_aggressive",
      "market": "AVAX-USD",
      "stop_loss_pct": 8.0,
      "main_leverage": 3.0,
      "position_sizing": {
        "percent_equity": 0.12
      }
    }
  ]
}
```

## Risk Monitoring and Alerts

### Real-Time Risk Metrics

Monitor key risk metrics continuously:

```json
{
  "risk_monitoring": {
    "enabled": true,
    "update_frequency_seconds": 30,
    "metrics": [
      "total_portfolio_risk",
      "individual_position_risk",
      "correlation_exposure",
      "leverage_utilization",
      "drawdown_levels"
    ]
  }
}
```

### Alert Configuration

Set up alerts for risk threshold breaches:

```json
{
  "risk_alerts": {
    "portfolio_risk_threshold_pct": 0.12,
    "individual_position_threshold_pct": 0.05,
    "correlation_threshold_pct": 0.4,
    "drawdown_threshold_pct": 0.08,
    "leverage_threshold": 2.5,
    "alert_channels": ["email", "webhook", "dashboard"]
  }
}
```

### Grafana Dashboard Metrics

Monitor risk through Grafana dashboards:

```promql
# Portfolio risk utilization
sum(spark_stacker_position_risk) / spark_stacker_portfolio_value

# Individual strategy risk
spark_stacker_strategy_risk{strategy_name=~".*"}

# Leverage utilization by strategy
spark_stacker_leverage_used{strategy_name=~".*"} / spark_stacker_max_leverage{strategy_name=~".*"}

# Correlation exposure
spark_stacker_correlated_exposure{asset_class=~".*"}
```

## Risk Management Best Practices

### 1. Position Sizing Guidelines

- **Start Conservative**: Begin with smaller position sizes and gradually increase
- **Diversify Methods**: Use different position sizing methods for different strategy types
- **Set Maximum Limits**: Always specify maximum position size limits
- **Regular Review**: Periodically review and adjust position sizing parameters
- **Account for Correlation**: Reduce position sizes for highly correlated strategies

### 2. Stop Loss Management

- **Consistent Application**: Use stop losses on all positions
- **Appropriate Distance**: Set stop losses based on strategy timeframe and volatility
- **Trailing Stops**: Use trailing stops for trend-following strategies
- **Emergency Stops**: Have hard stops for extreme market events
- **Regular Adjustment**: Adjust stop losses as positions move in favor

### 3. Leverage Guidelines

- **Conservative Approach**: Start with lower leverage and increase gradually
- **Strategy-Specific**: Adjust leverage based on strategy characteristics
- **Market Conditions**: Reduce leverage in volatile or uncertain markets
- **Risk Scaling**: Scale leverage inversely with position size
- **Regular Monitoring**: Monitor leverage usage continuously

### 4. Portfolio Management

- **Diversification**: Spread risk across multiple strategies and markets
- **Correlation Monitoring**: Track correlation between positions
- **Risk Budgeting**: Allocate risk budget across different strategy types
- **Rebalancing**: Regularly rebalance portfolio allocations
- **Performance Review**: Analyze risk-adjusted returns regularly

## Testing and Validation

### Risk Configuration Testing

Test risk management configurations before deployment:

```bash
cd packages/spark-app
.venv/bin/python -c "
import json
from app.risk_management.risk_manager import RiskManager
from app.core.strategy_config import StrategyConfigLoader

# Load configuration
with open('../shared/config.json') as f:
    config = json.load(f)

# Test risk manager creation
risk_manager = RiskManager.from_config(config)
print(f'✅ Risk manager created successfully')

# Test strategy position sizers
strategies = StrategyConfigLoader.load_strategies(config['strategies'])
for strategy in strategies:
    sizer = risk_manager.strategy_position_sizers.get(strategy.name)
    print(f'✅ Strategy {strategy.name}: {sizer.config.method.value} position sizing')

# Test risk calculations
total_risk = risk_manager.calculate_portfolio_risk()
print(f'✅ Current portfolio risk: {total_risk:.2%}')
"
```

### Backtesting with Risk Controls

Test strategies with realistic risk management:

```bash
cd packages/spark-app
.venv/bin/python -m tests._utils.cli backtest-strategy \
  --strategy eth_momentum_strategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-01 \
  --include-risk-management \
  --max-drawdown 0.15 \
  --risk-per-trade 0.02
```

### Stress Testing

Test risk management under extreme scenarios:

```python
# Stress test scenarios
stress_scenarios = [
    {"name": "market_crash", "price_drop_pct": 0.3, "volatility_multiplier": 3.0},
    {"name": "flash_crash", "price_drop_pct": 0.15, "duration_minutes": 10},
    {"name": "extended_bear", "price_decline_days": 60, "decline_pct": 0.5},
    {"name": "high_volatility", "volatility_multiplier": 5.0, "duration_days": 7}
]

for scenario in stress_scenarios:
    result = run_stress_test(scenario)
    print(f"Scenario {scenario['name']}: Max drawdown {result['max_drawdown']:.2%}")
```

## Risk Management Evolution

### Continuous Improvement

1. **Performance Analysis**: Regularly analyze risk-adjusted returns
2. **Parameter Optimization**: Optimize risk parameters based on historical performance
3. **Market Adaptation**: Adjust risk management for changing market conditions
4. **Strategy Evolution**: Update risk controls as strategies evolve
5. **Technology Updates**: Implement new risk management techniques

### Advanced Features

Future enhancements to consider:

1. **Machine Learning Risk Models**: Use ML to predict and manage risk
2. **Dynamic Correlation Models**: Real-time correlation tracking and adjustment
3. **Options-Based Hedging**: Use options for portfolio protection
4. **Cross-Exchange Risk Management**: Manage risk across multiple exchanges
5. **Regulatory Compliance**: Implement regulatory risk controls

This comprehensive risk management guide provides the framework for safe and profitable trading
while maintaining the flexibility needed for different strategy types and market conditions. The key
is to start conservative, test thoroughly, and gradually optimize based on performance and market
experience.
