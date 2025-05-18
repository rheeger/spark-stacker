from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from app.backtesting.backtest_engine import BacktestEngine
from app.backtesting.data_manager import DataManager

# Shared fixtures
@pytest.fixture
def price_dataframe() -> pd.DataFrame:
    """
    Creates a sample price dataframe with deterministic values for testing.

    Returns:
        pd.DataFrame: A DataFrame with OHLCV price data
    """
    ...

@pytest.fixture
def temp_csv_dir() -> Path:
    """
    Creates a temporary directory for CSV files and cleans up after the test.

    Yields:
        Path: Path to the temporary directory
    """
    ...

@pytest.fixture
def backtest_env(price_dataframe: pd.DataFrame, temp_csv_dir: Path) -> Tuple[BacktestEngine, DataManager, str, str]:
    """
    Sets up a BacktestEngine with a DataManager and sample data.

    Args:
        price_dataframe: Sample price data
        temp_csv_dir: Temporary directory for CSV files

    Returns:
        tuple: (BacktestEngine, DataManager, symbol, interval)
    """
    ...

@pytest.fixture
def results_dir() -> Path:
    """
    Creates a temporary directory for test results and cleans up after the test.

    Yields:
        Path: Path to the temporary directory for test results
    """
    ...

# Existing fixtures from conftest.py
@pytest.fixture
def sample_price_data(request: FixtureRequest) -> pd.DataFrame:
    """
    Sample price data for indicator testing from cache.
    Falls back to synthetic data only if pytest is run with --allow-synthetic-data flag.
    """
    ...

@pytest.fixture
def real_market_data(request: FixtureRequest) -> pd.DataFrame:
    """
    Parametrizable fixture to get market data for different symbols and timeframes.
    """
    ...

@pytest.fixture
def market_data_cache_dir() -> Path:
    """Return the path to the market data cache directory."""
    ...

@pytest.fixture
def sample_config() -> Any:
    """Sample application configuration."""
    ...

@pytest.fixture
def mock_connector() -> Any:
    """Mock exchange connector."""
    ...

@pytest.fixture
def real_connector(request: FixtureRequest) -> Any:
    """
    Parametrizable fixture to get a real connector instance.
    """
    ...

@pytest.fixture
def mock_risk_manager() -> Any:
    """Mock risk manager."""
    ...

@pytest.fixture
def trading_engine(mock_connector: Any, mock_risk_manager: Any) -> Any:
    """Trading engine instance with mock components."""
    ...

@pytest.fixture
def sample_signal() -> Any:
    """Sample trading signal."""
    ...

@pytest.fixture
def market_data_params(request: FixtureRequest) -> Dict[str, Any]:
    """
    Fixture to provide market data parameters to the real_market_data fixture.
    """
    ...

@pytest.fixture(scope="session")
def app_path() -> str:
    """Return the absolute path to the app directory."""
    ...
