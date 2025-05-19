# Spark Stacker Testing Suite

This directory contains the test suite for the Spark Stacker trading system. The tests are organized
to validate all aspects of the system, including unit tests for individual components and
integration tests for the entire system.

## Directory Structure

- `conftest.py`: Common test fixtures used across all tests
- `unit/`: Unit tests for individual components
  - `test_rsi_indicator.py`: Tests for the RSI indicator
  - `test_indicator_factory.py`: Tests for the indicator factory
  - `test_risk_manager.py`: Tests for the risk management module
  - `test_webhook_server.py`: Tests for the webhook server
  - `test_base_connector.py`: Tests for the base exchange connector
  - `test_connector_factory.py`: Tests for the connector factory
  - `test_trading_engine.py`: Tests for the trading engine
- `integration/`: Integration tests for combined components
- `simulation/`: Simulation tests for strategy validation

## Running Tests

### Using pytest directly

⚠️ **IMPORTANT: Before running tests, you must run the data refresh script:**

```bash
python scripts/refresh_test_market_data.py
```

Run all tests:

```bash
pytest
```

Run tests with verbose output:

```bash
pytest -v
```

Run a specific test file:

```bash
pytest tests/unit/test_rsi_indicator.py
```

Run tests matching a specific name:

```bash
pytest -k "test_rsi"
```

Allow tests to use synthetic data when cached data isn't available:

```bash
pytest --allow-synthetic-data
```

### Using the test runner script (Recommended)

The project includes a test runner script that provides additional features like coverage reporting,
linting, and **automatic market data refresh**:

```bash
# Run all tests with coverage report (auto-refreshes market data if needed)
./scripts/run_tests.py

# Run specific tests
./scripts/run_tests.py --path tests/unit/test_rsi_indicator.py

# Run with verbose output
./scripts/run_tests.py --verbose

# Run without coverage
./scripts/run_tests.py --no-coverage

# Force refresh market data
./scripts/run_tests.py --refresh-data

# Skip market data refresh
./scripts/run_tests.py --skip-data-refresh

# Allow tests to use synthetic data if cache is missing
./scripts/run_tests.py --allow-synthetic-data

# Run linting checks
./scripts/run_tests.py --lint

# Run type checking
./scripts/run_tests.py --type-check

# Run all checks (tests, linting, type checking)
./scripts/run_tests.py --all-checks
```

The test runner script automatically checks if the market data cache is older than 24 hours and
refreshes it if needed. This ensures tests always have recent data without manual intervention.

### Using the file watcher

For continuous testing during development, you can use the file watcher script:

```bash
# Watch app and tests directories
./scripts/file_watcher.py

# Watch specific directories
./scripts/file_watcher.py app/indicators tests/unit
```

The file watcher will automatically run the relevant tests when files are modified.

## Writing Tests

When writing new tests, follow these guidelines:

1. Create test files with the `test_` prefix
2. Use the fixtures provided in `conftest.py` where appropriate
3. Use descriptive test names that indicate what is being tested
4. Include assertions that validate the expected behavior
5. Add type annotations to improve code readability
6. Document complex test cases with docstrings

## Test Coverage

To view a coverage report:

```bash
pytest --cov=app --cov-report=html
```

This will generate an HTML coverage report in the `htmlcov/` directory, which you can open in a
browser to see which parts of the code are covered by tests.

## Mocking External Dependencies

For tests that depend on external services like exchanges, use the mock objects provided in
`conftest.py` to avoid making actual API calls during testing.

# Testing Framework with Real Market Data

This testing framework uses cached real market data from exchange connectors (Hyperliquid and
Coinbase) for reliable and realistic tests.

## Overview

The testing framework now supports:

1. **Cached real market data** from exchange connectors (no API calls during tests)
2. **Parameterizable data sources** to test with different exchanges, symbols, and timeframes
3. **Visual inspection** of test results with automatically generated charts
4. **Optional synthetic data fallback** with the `--allow-synthetic-data` flag
5. **Automatic data refresh** when using the run_tests.py script

## Market Data Refresh

The market data cache is automatically refreshed in the following scenarios:

1. When running `./scripts/run_tests.py` and the cache is older than 24 hours
2. When running `./scripts/run_tests.py --refresh-data` (forces refresh regardless of cache age)
3. When manually running `python scripts/refresh_test_market_data.py`

You can skip the automatic refresh by using the `--skip-data-refresh` flag:

```bash
./scripts/run_tests.py --skip-data-refresh
```

## How to Use

### Basic Usage

The default `sample_price_data` fixture now provides cached real market data:

```python
def test_my_indicator(sample_price_data):
    # sample_price_data is a DataFrame with real OHLCV data
    my_indicator = MyIndicator()
    result = my_indicator.generate_signals(sample_price_data)
    assert result is not None
```

### Testing with Different Data Sources

Use the `real_market_data` fixture with parametrization to test with different data sources:

```python
@pytest.mark.parametrize('market_data_params', [
    {'exchange': 'hyperliquid', 'symbol': 'BTC-USD', 'timeframe': '1h'},
    {'exchange': 'coinbase', 'symbol': 'ETH-USD', 'timeframe': '15m'},
], indirect=True)
def test_with_different_markets(real_market_data, market_data_params):
    # real_market_data will contain data for the specified exchange, symbol, and timeframe
    assert real_market_data is not None

    # You can access the parameters if needed
    symbol = market_data_params['symbol']
    timeframe = market_data_params['timeframe']
```

### Testing with Real Connectors

Use the `real_connector` fixture to test with a real exchange connector:

```python
@pytest.mark.parametrize('real_connector', ['hyperliquid'], indirect=True)
def test_with_real_connector(real_connector):
    # real_connector is an instance of the specified exchange connector
    markets = real_connector.get_markets()
    assert markets is not None
```

## Market Data Cache

All market data is cached in the `tests/test_data/market_data` directory. This caching strategy:

1. Makes tests faster and more reliable by eliminating API calls during test runs
2. Ensures consistent data across test runs for reproducible results
3. Prevents hitting API rate limits during intensive test sessions
4. Makes tests work in CI/CD environments without API access

## Configuration

Default settings are defined in `tests/conftest.py`:

```python
DEFAULT_TEST_EXCHANGE = "hyperliquid"  # Change to coinbase if preferred
DEFAULT_TEST_SYMBOL = "ETH-USD"        # Use format that works with your connector
DEFAULT_TEST_TIMEFRAME = "1h"
DEFAULT_DATA_DAYS = 30
```

You can modify these constants to change the default data source.

The cache refresh age threshold (24 hours) is defined in `scripts/run_tests.py`:

```python
CACHE_MAX_AGE_HOURS = 24  # Refresh data if older than this many hours
```

## Visual Inspection

Tests that generate signals (like in `test_macd_indicator_with_real_data.py`) will automatically
create charts for visual inspection. These charts are saved in the `tests/test_results` directory.

## Best Practices

1. **Use the run_tests.py script**: This ensures your market data is always up to date.
2. **Add required data files**: Update the `REQUIRED_DATA_FILES` set in `scripts/run_tests.py` when
   adding new tests that depend on specific data files.
3. **Use parametrization**: Test your components with different markets and timeframes to ensure
   robustness.
4. **Add new data types when needed**: Update the refresh script when testing new market pairs or
   timeframes.
5. **Version control your test results**: Consider committing visual test results to track algorithm
   changes over time.
6. **Automate data refresh in CI/CD**: Your CI/CD pipeline should use the run_tests.py script to
   ensure data is refreshed.

## Adding New Data Sources

To add a new data source:

1. Update the `EXCHANGES` and `SYMBOLS` dictionaries in `scripts/refresh_test_market_data.py`
2. Update the `REQUIRED_DATA_FILES` set in `scripts/run_tests.py` if you need these files for all
   tests
3. Add appropriate parametrization in your tests
4. Run the refresh script to cache the new data

## Troubleshooting

- **Missing data errors**: Ensure the run_tests.py script is running with proper permissions to
  refresh the data.
- **API errors during refresh**: Check connector authentication settings.
- **Using synthetic data**: In development, you can use `--allow-synthetic-data` to bypass the
  requirement for cached data (not recommended for CI/CD).

# Spark App Testing

This directory contains tests for the Spark App package. The tests are organized into different categories:

- `backtesting/`: Tests for the backtesting engine and related components
- `indicators/`: Tests for trading indicators
- `connectors/`: Tests for exchange connectors
- `_fixtures/`: Test fixtures and mock data
- `_helpers/`: Helper functions and utilities for testing

## Running Tests

To run the tests, make sure you are in the `packages/spark-app` directory and have the virtual environment activated:

```bash
cd packages/spark-app
source .venv/bin/activate  # On Unix/Mac
# OR
.venv\Scripts\activate     # On Windows
```

Run all tests:

```bash
python -m pytest
```

Run specific test files:

```bash
python -m pytest tests/backtesting/unit/test_backtest_engine.py
```

Run tests with coverage:

```bash
python -m pytest --cov=app
```

Generate coverage report:

```bash
python -m pytest --cov=app --cov-report=html
```

Run only quick tests (skip slow tests):

```bash
python -m pytest -m "not slow"
```

## Viewing Local Artifacts

### Temporary Test Artifacts

Many tests generate artifacts (reports, plots, JSON files) in temporary directories that are automatically cleaned up after the test completes. By default, these temporary directories are created using Python's `tempfile.TemporaryDirectory()` function.

To view these artifacts during a test run, you can add a sleep or breakpoint in your test after they are generated, or modify the test to print the path to the temporary directory:

```python
print(f"Test artifacts are in: {results_dir}")
import time
time.sleep(60)  # Keep the directory around for 60 seconds
```

### Persistent Artifacts

For persistent artifacts that aren't cleaned up automatically, check these locations:

1. **Default Test Results**: `packages/spark-app/tests/__test_results__/`

   - This directory contains results generated by the indicator test harness when run without a specific output path.

2. **Coverage Reports**: `packages/spark-app/htmlcov/`

   - HTML coverage reports generated with `--cov-report=html`

3. **Test Data**: `packages/spark-app/tests/__test_data__/`
   - Test data files used by tests

### Using the `results_dir` Fixture

Most tests that generate artifacts use the `results_dir` fixture, which creates a temporary directory that is automatically cleaned up after the test completes. To preserve these artifacts, you can modify your tests to use a specific directory instead:

```python
def test_something(results_dir):
    # By default, results_dir is a temporary directory
    # You can override it for debugging:
    import os
    os.environ['TEST_RESULTS_DIR'] = '/path/to/your/debug/directory'
    # Now results will be saved to that directory and not cleaned up
```

## Cleaning Up Test Artifacts

To clean up all test artifacts, you can use the `make clean-results` command from the project root:

```bash
make clean-results
```

This will remove all generated test results, but preserve test data.
