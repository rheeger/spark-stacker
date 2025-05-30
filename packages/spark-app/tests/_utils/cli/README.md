# Modular CLI Architecture

This directory contains the new modular CLI architecture for Spark Stacker, providing a clean, maintainable, and extensible command-line interface for backtesting and strategy analysis.

## 🏗️ Architecture Overview

The CLI is organized into focused, single-responsibility modules with clear separation of concerns:

```
cli/
├── main.py                       # Main CLI entry point
├── __init__.py                   # Package initialization
├── commands/                     # Command handler modules
│   ├── strategy_commands.py      # Strategy backtesting commands
│   ├── indicator_commands.py     # Legacy indicator commands
│   ├── comparison_commands.py    # Strategy/indicator comparison commands
│   ├── list_commands.py          # List strategies/indicators commands
│   └── utility_commands.py       # Config validation, migration commands
├── core/                         # Core business logic modules
│   ├── config_manager.py         # Configuration loading and validation
│   ├── data_manager.py           # Data fetching and caching
│   ├── backtest_orchestrator.py  # Coordinates backtesting workflow
│   └── scenario_manager.py       # Multi-scenario testing coordination
├── managers/                     # Specialized manager classes
│   ├── strategy_backtest_manager.py    # Strategy-specific backtesting
│   ├── indicator_backtest_manager.py   # Legacy indicator backtesting
│   ├── scenario_backtest_manager.py    # Multi-scenario execution
│   └── comparison_manager.py           # Strategy comparison logic
├── reporting/                    # Report generation modules
│   ├── strategy_reporter.py      # Strategy-specific reporting
│   ├── comparison_reporter.py    # Strategy comparison reports
│   ├── scenario_reporter.py      # Multi-scenario reporting
│   └── interactive_reporter.py   # Interactive trade selection features
├── validation/                   # Validation and error handling
│   ├── config_validator.py       # Configuration validation
│   ├── strategy_validator.py     # Strategy-specific validation
│   └── data_validator.py         # Data quality validation
└── utils/                        # Utility functions and helpers
    ├── cli_helpers.py            # CLI utility functions
    ├── output_formatters.py      # Console output formatting
    └── progress_trackers.py      # Progress tracking utilities
```

## 🚀 Quick Start

### Using the New Modular CLI

```bash
# Navigate to the spark-app directory
cd packages/spark-app

# Run commands using the new modular CLI
python tests/_utils/cli/main.py --help
python tests/_utils/cli/main.py list-strategies
python tests/_utils/cli/main.py strategy my_strategy_name
```

### Backward Compatibility

The old CLI location still works through a compatibility shim:

```bash
# Legacy location (shows deprecation warning)
python tests/_utils/cli.py --help
```

**⚠️ Migration Notice**: The legacy `cli.py` file is deprecated. Please update your scripts to use the new modular location.

## 📋 Available Commands

### Strategy Commands

| Command              | Description                                          | Example                                              |
| -------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| `strategy`           | Run strategy backtesting with multi-scenario testing | `python main.py strategy eth_momentum_strategy`      |
| `compare-strategies` | Compare multiple strategies across scenarios         | `python main.py compare-strategies --all-strategies` |

### List Commands

| Command           | Description                           | Example                          |
| ----------------- | ------------------------------------- | -------------------------------- |
| `list-strategies` | Show available strategies from config | `python main.py list-strategies` |
| `list-indicators` | Show available indicators             | `python main.py list-indicators` |

### Utility Commands

| Command           | Description                  | Example                          |
| ----------------- | ---------------------------- | -------------------------------- |
| `validate-config` | Validate configuration files | `python main.py validate-config` |
| `migrate-config`  | Migrate old config formats   | `python main.py migrate-config`  |
| `clean-cache`     | Clear cached data            | `python main.py clean-cache`     |

### Legacy Commands (Deprecated)

| Command     | Description        | Migration Path                              |
| ----------- | ------------------ | ------------------------------------------- |
| `demo`      | Run indicator demo | Use `strategy` command with strategy config |
| `real-data` | Run with real data | Use `strategy --use-real-data`              |
| `compare`   | Compare indicators | Use `compare-strategies`                    |

## 🔧 Configuration Integration

The CLI integrates seamlessly with `config.json` files:

```bash
# Use default config location
python main.py list-strategies

# Use custom config file
python main.py --config /path/to/config.json list-strategies

# Validate configuration
python main.py validate-config --config /path/to/config.json
```

### Strategy Configuration Example

```json
{
  "strategy_configs": {
    "eth_momentum_strategy": {
      "name": "ETH Momentum Strategy",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "timeframe": "1h",
      "indicators": {
        "rsi": {
          "class": "RSIIndicator",
          "timeframe": "1h",
          "window": 14
        },
        "macd": {
          "class": "MACDIndicator",
          "timeframe": "4h",
          "fast_period": 12,
          "slow_period": 26
        }
      },
      "position_sizing": {
        "method": "fixed_usd",
        "amount": 100
      },
      "enabled": true
    }
  }
}
```

## 🎯 Multi-Scenario Testing

The new CLI runs comprehensive multi-scenario testing by default:

### Synthetic Market Scenarios

1. **Bull Market** - Consistent uptrend with 60-80% up days
2. **Bear Market** - Consistent downtrend with 60-80% down days
3. **Sideways** - Range-bound oscillating within 5-10% range
4. **High Volatility** - Large daily swings, 15-25% moves
5. **Low Volatility** - Minimal daily changes, <2% moves
6. **Choppy Market** - Frequent direction changes, whipsaws
7. **Gap-Heavy** - Frequent price gaps, simulating news events

### Real Data Testing

8. **Real Market Data** - Actual historical data for comparison

### Usage Examples

```bash
# Run all scenarios (default behavior)
python main.py strategy eth_momentum_strategy --days 30

# Run specific scenarios only
python main.py strategy eth_momentum_strategy --scenarios "bull,bear,real" --days 30

# Run single scenario for quick testing
python main.py strategy eth_momentum_strategy --scenario-only bull --days 30

# Export scenario data for analysis
python main.py strategy eth_momentum_strategy --days 14 --export-data
```

## 📊 Enhanced Reporting

### Strategy Reports

- **Configuration Display** - Full strategy settings and parameters
- **Multi-Scenario Performance** - Results across all market conditions
- **Indicator Breakdown** - Individual indicator performance within strategy
- **Position Sizing Analysis** - Impact of position sizing on results
- **Interactive Trade Selection** - Click trades to highlight on charts

### Strategy Comparison Reports

- **Side-by-Side Comparison** - Multiple strategies compared directly
- **Robustness Analysis** - Performance consistency across scenarios
- **Risk-Adjusted Metrics** - Sharpe ratio, max drawdown comparison
- **Portfolio Optimization** - Strategy combination recommendations

### Interactive Features

- **Clickable Trade Lists** - Select trades to highlight on charts
- **Scenario Tabs** - Switch between different market scenarios
- **Trade Filtering** - Search and filter trades by criteria
- **Zoom Controls** - Focus on specific time periods
- **Export Functionality** - Save results and data

## 🧩 Module System

### Core Modules

#### ConfigManager

```python
from core.config_manager import ConfigManager

config_mgr = ConfigManager(config_path="config.json")
strategies = config_mgr.list_strategies()
strategy_config = config_mgr.get_strategy_config("my_strategy")
```

#### DataManager

```python
from core.data_manager import DataManager

data_mgr = DataManager(cache_dir="./cache")
data = data_mgr.fetch_real_data("ETH-USD", "hyperliquid", "1h", days=7)
synthetic_data = data_mgr.generate_synthetic_data("bull", 30, "1h")
```

### Manager Classes

#### StrategyBacktestManager

```python
from managers.strategy_backtest_manager import StrategyBacktestManager

manager = StrategyBacktestManager(config_mgr, data_mgr)
results = manager.run_strategy_backtest("my_strategy", days=30)
```

#### ComparisonManager

```python
from managers.comparison_manager import ComparisonManager

comp_mgr = ComparisonManager(config_mgr, data_mgr)
comparison = comp_mgr.compare_strategies(["strategy1", "strategy2"])
```

### Command Handlers

Commands are organized by functionality:

- **Strategy Commands** - `/commands/strategy_commands.py`
- **Indicator Commands** - `/commands/indicator_commands.py`
- **List Commands** - `/commands/list_commands.py`
- **Utility Commands** - `/commands/utility_commands.py`

### Reporting Modules

- **Strategy Reporter** - Individual strategy reporting
- **Comparison Reporter** - Multi-strategy comparison
- **Scenario Reporter** - Multi-scenario analysis
- **Interactive Reporter** - JavaScript components for interactivity

## 🔌 Extension Points

### Adding New Commands

1. Create command handler in appropriate `/commands/` module
2. Add command setup function
3. Register in `main.py`

```python
# commands/my_commands.py
import click

def setup_my_commands(cli_group):
    @cli_group.command("my-command")
    @click.option("--param", help="My parameter")
    def my_command(param):
        """My custom command."""
        # Implementation here
        pass
```

### Adding New Managers

1. Create manager class in `/managers/`
2. Integrate with core modules
3. Add tests in `/test_modules/`

```python
# managers/my_manager.py
from ..core.config_manager import ConfigManager
from ..core.data_manager import DataManager

class MyManager:
    def __init__(self, config_mgr: ConfigManager, data_mgr: DataManager):
        self.config_mgr = config_mgr
        self.data_mgr = data_mgr

    def my_operation(self):
        # Implementation here
        pass
```

### Adding New Validation

1. Create validator in `/validation/`
2. Integrate with existing validation workflow
3. Add comprehensive tests

```python
# validation/my_validator.py
def validate_my_feature(config):
    """Validate my feature configuration."""
    errors = []
    # Validation logic here
    return len(errors) == 0, errors
```

## 🧪 Testing

### Module Tests

```bash
# Run all module tests
cd packages/spark-app
python -m pytest tests/_utils/cli/test_modules/ -v

# Run specific module tests
python -m pytest tests/_utils/cli/test_modules/test_core_modules/ -v
python -m pytest tests/_utils/cli/test_modules/test_integration/ -v
```

### Integration Tests

```bash
# Test CLI integration
python -m pytest tests/backtesting/integration/test_cli.py -v

# Test backward compatibility
python -m pytest tests/backtesting/integration/test_cli.py::TestCLIBackwardCompatibility -v

# Test new modular CLI
python -m pytest tests/backtesting/integration/test_cli.py::TestModularCLI -v
```

### Performance Tests

```bash
# Test module performance
python -m pytest tests/_utils/cli/test_modules/test_performance/ -v
```

## ⚠️ Migration Guide

### From Legacy CLI

**Old Usage:**

```bash
python tests/_utils/cli.py demo --indicator RSI --market ETH-USD
python tests/_utils/cli.py real-data --indicator MACD --days 7
python tests/_utils/cli.py compare --indicators "RSI,MACD" --market ETH-USD
```

**New Usage:**

```bash
# Create strategy config first, then:
python tests/_utils/cli/main.py strategy eth_rsi_strategy
python tests/_utils/cli/main.py strategy eth_macd_strategy --use-real-data --days 7
python tests/_utils/cli/main.py compare-strategies --strategy-names "eth_rsi_strategy,eth_macd_strategy"
```

### Updating Scripts

**Old Script:**

```bash
#!/bin/bash
python tests/_utils/cli.py demo --indicator RSI
```

**New Script:**

```bash
#!/bin/bash
python tests/_utils/cli/main.py strategy rsi_strategy
```

### Configuration Migration

Use the migration command to update old configurations:

```bash
python tests/_utils/cli/main.py migrate-config --input old_config.json --output new_config.json
```

## 🔍 Troubleshooting

### Common Issues

#### Import Errors

```bash
# If you see import errors, ensure you're in the correct directory
cd packages/spark-app
python tests/_utils/cli/main.py --help
```

#### Module Not Found

```bash
# Check Python path and virtual environment
which python
source .venv/bin/activate  # If using virtual environment
```

#### Configuration Issues

```bash
# Validate your configuration
python tests/_utils/cli/main.py validate-config

# Check configuration path
python tests/_utils/cli/main.py validate-config --config /full/path/to/config.json
```

### Performance Issues

#### Slow Startup

- Check cache directory permissions
- Clear old cache: `python main.py clean-cache`
- Reduce log level in configuration

#### Memory Usage

- Limit concurrent scenarios: `--scenarios "bull,bear"`
- Reduce backtest duration: `--days 7`
- Enable data caching for repeated runs

### Getting Help

```bash
# General help
python tests/_utils/cli/main.py --help

# Command-specific help
python tests/_utils/cli/main.py strategy --help
python tests/_utils/cli/main.py compare-strategies --help

# Verbose output for debugging
python tests/_utils/cli/main.py strategy my_strategy --verbose
```

## 📈 Performance Optimizations

### Caching Strategy

- **Configuration Caching** - Configs cached across runs
- **Data Caching** - Market data cached with intelligent expiration
- **Result Caching** - Backtest results cached for comparison
- **Parallel Execution** - Multiple scenarios run concurrently

### Resource Management

- **Memory Optimization** - Efficient DataFrame handling
- **Disk Space** - Automatic cleanup of old cache files
- **Network Efficiency** - Batch API calls, connection pooling

## 🔄 Future Enhancements

### Planned Features

- **WebSocket Data Feeds** - Real-time data integration
- **Cloud Storage** - S3/GCS integration for data and results
- **Distributed Computing** - Multi-machine scenario execution
- **ML Integration** - Machine learning model backtesting
- **Portfolio Management** - Multi-strategy portfolio optimization

### Plugin System

Future versions will support plugin architecture:

```python
# Example plugin structure
plugins/
├── my_plugin/
│   ├── __init__.py
│   ├── commands.py
│   ├── managers.py
│   └── validators.py
```

## 📚 Additional Resources

- [Phase 3.5.3 Checklist](../../../../shared/docs/checklists/phase3.5.3-backtesting-improvements.md) - Implementation roadmap
- [Strategy Development Guide](../../../../shared/docs/strategy-development.md) - Strategy creation
- [Configuration Guide](../../../../shared/docs/configuration.md) - Config file format
- [API Documentation](../../../app/README.md) - Core application APIs

---

**Need Help?** Check the troubleshooting section above or refer to the comprehensive test suite in `test_modules/` for usage examples.
