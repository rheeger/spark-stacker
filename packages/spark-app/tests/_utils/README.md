# Test Utilities

This directory contains utility scripts for testing the Spark App package. These scripts are primarily used for test preparation, execution, and cleanup.

## Overview

- `run_tests.py`: Main test runner script that configures and runs pytest with appropriate options
- `test_system.py`: Comprehensive test script that handles data preparation and test execution
- `refresh_test_market_data.py`: Utility to refresh market data cache for testing
- `run_eth_macd_backtest.py`: Wrapper around the CLI to run MACD backtest demos for ETH
- `run_tests_with_venv.sh`: Shell script to ensure tests run within the correct virtual environment
- `cleanup_logs.py`: Utility to clean up log directories

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

### Refreshing Test Market Data

```bash
cd packages/spark-app
python -m tests.utils.refresh_test_market_data
```

### Running MACD Backtest Demo

```bash
cd packages/spark-app
python -m tests.utils.run_eth_macd_backtest
```

## Notes

These utilities were moved from the `scripts/` directory as part of the backtesting suite refactor (Phase 3.5.1, task 4.5) to better organize test-related scripts.
