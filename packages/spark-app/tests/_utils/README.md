# Test Utilities

This directory contains utility scripts for testing the Spark App package. These scripts are primarily used for test preparation, execution, and cleanup.

## Overview

### Core Utilities

- `run_tests.py`: Main test runner script that configures and runs pytest with appropriate options
- `test_system.py`: Comprehensive test script that handles data preparation and test execution
- `refresh_test_market_data.py`: Utility to refresh market data cache for testing
- `run_eth_macd_backtest.py`: Wrapper around the CLI to run MACD backtest demos for ETH
- `run_tests_with_venv.sh`: Shell script to ensure tests run within the correct virtual environment
- `cleanup_logs.py`: Utility to clean up log directories

### CLI Architecture (Migrated to Modular Structure)

The CLI has been migrated to a modular architecture for better maintainability and extensibility:

#### New Modular CLI Location: `/cli/`

- **Main Entry Point**: `cli/main.py`
- **Modular Architecture**: Organized into focused modules (commands, core, managers, reporting, validation, utils)
- **Enhanced Features**: Multi-scenario testing, strategy-driven backtesting, interactive reporting

#### Legacy CLI (Deprecated): `cli.py`

- **Status**: ⚠️ **DEPRECATED** - Maintained for backward compatibility only
- **Functionality**: Compatibility shim that redirects to new modular CLI
- **Migration Path**: Update scripts to use `cli/main.py` instead of `cli.py`

**📖 For detailed CLI documentation, see: [`cli/README.md`](cli/README.md)**

## Usage

### Running Tests

To run all tests with coverage:

```bash
cd packages/spark-app
python -m tests.utils.run_tests
```

For quick tests (excluding slow ones):

```bash
cd packages/spark-app
python -m tests.utils.run_tests --quick
```

### CLI Usage (New Modular Architecture)

**✅ Recommended Usage (New Modular CLI):**

```bash
cd packages/spark-app

# Strategy backtesting
python tests/_utils/cli/main.py strategy my_strategy_name
python tests/_utils/cli/main.py compare-strategies --all-strategies

# List available options
python tests/_utils/cli/main.py list-strategies
python tests/_utils/cli/main.py list-indicators

# Utility commands
python tests/_utils/cli/main.py validate-config
python tests/_utils/cli/main.py clean-cache
```

**⚠️ Legacy Usage (Deprecated but still works):**

```bash
cd packages/spark-app

# These commands still work but show deprecation warnings
python tests/_utils/cli.py demo --indicator RSI
python tests/_utils/cli.py real-data --indicator MACD --days 7
```

### Migration Guide

**Updating Existing Scripts:**

```bash
# OLD (deprecated)
python tests/_utils/cli.py demo --indicator RSI --market ETH-USD

# NEW (recommended)
# 1. Create strategy config with RSI indicator
# 2. Run: python tests/_utils/cli/main.py strategy eth_rsi_strategy
```

### Refreshing Test Market Data

```bash
cd packages/spark-app
python -m tests.utils.refresh_test_market_data
```

### Running MACD Backtest Demo

**Using New CLI (Recommended):**

```bash
cd packages/spark-app
# Create MACD strategy in config.json first, then:
python tests/_utils/cli/main.py strategy eth_macd_strategy
```

**Using Legacy CLI (Deprecated):**

```bash
cd packages/spark-app
python -m tests.utils.run_eth_macd_backtest
```

## CLI Features Comparison

### Legacy CLI (`cli.py`) - Deprecated

- ❌ Indicator-focused commands
- ❌ Single scenario testing only
- ❌ Limited configuration integration
- ❌ Basic reporting
- ⚠️ Maintenance mode only

### New Modular CLI (`cli/main.py`) - Recommended

- ✅ Strategy-driven architecture
- ✅ Multi-scenario testing (7 synthetic + real data)
- ✅ Full config.json integration
- ✅ Interactive reporting with trade selection
- ✅ Comprehensive validation
- ✅ Performance optimizations
- ✅ Extensible modular design

## Architecture Benefits

### Modular Design

- **Single Responsibility**: Each module has a focused purpose
- **Easy Testing**: Modules can be tested independently
- **Extensibility**: New features can be added without modifying existing code
- **Maintainability**: Clear separation of concerns

### Enhanced Functionality

- **Multi-Scenario Testing**: Automatically tests strategies across different market conditions
- **Strategy Focus**: Built around complete trading strategies rather than individual indicators
- **Better Reporting**: Interactive HTML reports with trade selection features
- **Configuration Integration**: Seamless integration with strategy configurations

## Migration Timeline

### Phase 1: Dual Support (Current)

- ✅ New modular CLI fully functional
- ✅ Legacy CLI shows deprecation warnings but still works
- ✅ Comprehensive backward compatibility testing

### Phase 2: Migration Encouragement (Future)

- 📋 Documentation updated to prioritize new CLI
- 📋 Examples and tutorials use new CLI
- 📋 Legacy CLI warnings become more prominent

### Phase 3: Legacy Removal (Future)

- 📋 Legacy `cli.py` file removed
- 📋 Only modular CLI remains
- 📋 Clean up compatibility shims

## Testing the CLI Migration

### Backward Compatibility Tests

```bash
cd packages/spark-app
python -m pytest tests/backtesting/integration/test_cli.py::TestCLIBackwardCompatibility -v
```

### New Modular CLI Tests

```bash
cd packages/spark-app
python -m pytest tests/backtesting/integration/test_cli.py::TestModularCLI -v
```

### Module-Specific Tests

```bash
cd packages/spark-app
python -m pytest tests/_utils/cli/test_modules/ -v
```

## Getting Help

### For New Modular CLI

```bash
python tests/_utils/cli/main.py --help
python tests/_utils/cli/main.py strategy --help
python tests/_utils/cli/main.py compare-strategies --help
```

### For Legacy CLI (Deprecated)

```bash
python tests/_utils/cli.py --help  # Shows deprecation warning
```

### Documentation

- **Detailed CLI Guide**: [`cli/README.md`](cli/README.md)
- **Migration Examples**: See CLI README migration section
- **Architecture Overview**: [`cli/README.md#architecture-overview`](cli/README.md#🏗️-architecture-overview)

## Notes

These utilities were moved from the `scripts/` directory as part of the backtesting suite refactor (Phase 3.5.1, task 4.5) to better organize test-related scripts.

The CLI migration (Phase 3.5.3) represents a significant architectural improvement, moving from indicator-focused to strategy-driven backtesting with enhanced reporting and multi-scenario testing capabilities.

**For the most up-to-date CLI information, always refer to [`cli/README.md`](cli/README.md).**
