# Spark Stacker Testing Suite

This directory contains the test suite for the Spark Stacker trading system. The tests are organized to validate all aspects of the system, including unit tests for individual components and integration tests for the entire system.

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

### Using the test runner script

The project includes a test runner script that provides additional features like coverage reporting and linting:

```bash
# Run all tests with coverage report
./scripts/run_tests.py

# Run specific tests
./scripts/run_tests.py --path tests/unit/test_rsi_indicator.py

# Run with verbose output
./scripts/run_tests.py --verbose

# Run without coverage
./scripts/run_tests.py --no-coverage

# Run linting checks
./scripts/run_tests.py --lint

# Run type checking
./scripts/run_tests.py --type-check

# Run all checks (tests, linting, type checking)
./scripts/run_tests.py --all-checks
```

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

This will generate an HTML coverage report in the `htmlcov/` directory, which you can open in a browser to see which parts of the code are covered by tests.

## Mocking External Dependencies

For tests that depend on external services like exchanges, use the mock objects provided in `conftest.py` to avoid making actual API calls during testing. 