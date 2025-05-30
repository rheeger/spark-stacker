import os

# Set PYTEST_RUNNING environment variable before any imports
os.environ['PYTEST_RUNNING'] = 'True'

import importlib.util
import json
import logging
import sys
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from app.backtesting.backtest_engine import BacktestEngine
from app.backtesting.data_manager import CSVDataSource, DataManager

# Configure pytest-asyncio
pytest.mark.asyncio.apply_to_all = True
pytest_asyncio.default_fixture_loop_scope = "function"  # Set default fixture loop scope

# Filter out the specific pandas_ta pkg_resources deprecation warning
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=DeprecationWarning,
    module="pandas_ta"
)

# Also filter using an alternative pattern to catch all instances
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources is deprecated as an API.*",
    category=DeprecationWarning
)

# Filter numpy deprecation warning
warnings.filterwarnings(
    "ignore",
    message=".*np.find_common_type is deprecated.*",
    category=DeprecationWarning
)

# Check if we're running in CI
IN_CI = os.environ.get("CI", "").lower() == "true"
if IN_CI:
    print("Running in CI environment")

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(f"Project root: {project_root}")

# Add the project root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to sys.path")

# Also add the app directory to the Python path to enable direct imports
app_path = os.path.join(project_root, "app")
if app_path not in sys.path:
    sys.path.insert(0, app_path)
    print(f"Added {app_path} to sys.path")

# For troubleshooting, print PYTHONPATH and sys.path
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"sys.path[0:3]: {sys.path[0:3]}")

# Patch pandas_ta module to work with newer numpy versions
try:
    import numpy
    if not hasattr(numpy, 'NaN'):
        numpy.NaN = numpy.nan
except:
    pass

# Import metrics first to ensure proper initialization
try:
    from app.metrics.metrics import clear_metrics
except ImportError as e:
    print(f"Error importing app.metrics.metrics: {e}")
    # Try a fallback import without app prefix
    try:
        from metrics.metrics import clear_metrics
        print("Successfully imported metrics.metrics directly")
    except ImportError as e2:
        print(f"Error importing metrics.metrics directly: {e2}")
        # Define a stub for clear_metrics if we can't import it
        def clear_metrics():
            print("Using stub clear_metrics function")
            pass


@pytest.fixture(autouse=True)
def clear_metrics_before_test():
    """Clear all metrics before each test."""
    clear_metrics()
    yield

# Import necessary modules with error handling
try:
    from app.connectors.base_connector import (BaseConnector, MarketType,
                                               OrderSide, OrderType)
    from app.connectors.connector_factory import ConnectorFactory
    from app.core.trading_engine import TradingEngine
    from app.indicators.base_indicator import (BaseIndicator, Signal,
                                               SignalDirection)
    from app.risk_management.risk_manager import RiskManager
    from app.utils.config import (AppConfig, ExchangeConfig, IndicatorConfig,
                                  TradingStrategyConfig)
except ImportError as e:
    print(f"Import error in conftest.py: {e}")
    # Try imports without app prefix
    try:
        print("Attempting imports without 'app.' prefix...")
        import_path = "."
        if IN_CI:
            # In CI, we need to use the absolute path
            print("Setting sys.path for direct imports in CI")
            sys.path.insert(0, os.path.abspath(os.path.join(project_root, "app")))
            import_path = os.path.abspath(os.path.join(project_root, "app"))
            print(f"Import path: {import_path}")

        # Show all directories in the import path
        print(f"Directories in app path:")
        for item in os.listdir(import_path):
            if os.path.isdir(os.path.join(import_path, item)):
                print(f"  - {item}")

        # Try direct imports
        from connectors.base_connector import (BaseConnector, MarketType,
                                               OrderSide, OrderType)
        from connectors.connector_factory import ConnectorFactory
        from core.trading_engine import TradingEngine
        from indicators.base_indicator import (BaseIndicator, Signal,
                                               SignalDirection)
        from risk_management.risk_manager import RiskManager
        from utils.config import (AppConfig, ExchangeConfig, IndicatorConfig,
                                  TradingStrategyConfig)
        print("Successfully imported modules without 'app.' prefix")
    except ImportError as e2:
        print(f"Second import attempt failed: {e2}")
        # If this also fails, define stub classes
        print("Using stub classes for testing")

        class BaseConnector:
            pass

        class MarketType:
            PERPETUAL = "PERPETUAL"

        class OrderSide:
            BUY = "BUY"
            SELL = "SELL"

        class OrderType:
            MARKET = "MARKET"
            LIMIT = "LIMIT"

        class ConnectorFactory:
            @staticmethod
            def create_connector(name):
                return BaseConnector()

        class TradingEngine:
            pass

        class BaseIndicator:
            pass

        class Signal:
            pass

        class SignalDirection:
            BUY = "BUY"
            SELL = "SELL"
            NEUTRAL = "NEUTRAL"

        class RiskManager:
            pass

        class AppConfig:
            pass

        class ExchangeConfig:
            pass

        class IndicatorConfig:
            pass

        class TradingStrategyConfig:
            pass

# Set up logging for conftest
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants for market data
DEFAULT_TEST_EXCHANGE = "hyperliquid"  # Change to coinbase if preferred
DEFAULT_TEST_SYMBOL = "ETH-USD"  # Use format that works with your connector
DEFAULT_TEST_TIMEFRAME = "1h"
DEFAULT_DATA_DAYS = 30
MARKET_DATA_CACHE_DIR = Path(__file__).parent / "__test_data__" / "market_data"

# Create cache directory if it doesn't exist
MARKET_DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cached_market_data(
    exchange_name: str, symbol: str, timeframe: str, days: int = DEFAULT_DATA_DAYS
) -> pd.DataFrame:
    """
    Get market data from cache only. Does not fetch data from exchanges.

    Args:
        exchange_name: Name of the exchange connector
        symbol: Trading pair symbol
        timeframe: Candle timeframe (1m, 5m, 15m, 1h, etc.)
        days: Not used for fetching, just for cache filename consistency

    Returns:
        DataFrame with OHLCV data

    Raises:
        FileNotFoundError: If the cached data file doesn't exist
    """
    symbol_normalized = symbol.replace("/", "_").replace("-", "_")
    cache_file = (
        MARKET_DATA_CACHE_DIR / f"{exchange_name}_{symbol_normalized}_{timeframe}.csv"
    )

    # Return cached data if available
    if cache_file.exists():
        logger.info(f"Loading cached market data from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    # Raise error with helpful message if cache doesn't exist
    error_msg = (
        f"Cached market data not found: {cache_file}. "
        f"Please run 'python tests/_utils/refresh_test_market_data.py' first to populate the cache."
    )
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)


def generate_sample_price_data(symbol: str = "ETH") -> pd.DataFrame:
    """Generate synthetic price data for testing when real data is unavailable."""
    logger.warning(
        f"Using synthetic data for {symbol} as real data could not be fetched"
    )

    # Create a DataFrame with typical OHLCV data
    data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1h"),
        "symbol": symbol,
        "open": np.random.normal(1500, 50, 100),
        "high": np.random.normal(1550, 50, 100),
        "low": np.random.normal(1450, 50, 100),
        "close": np.random.normal(1500, 50, 100),
        "volume": np.random.normal(1000, 200, 100),
    }

    # Make sure high is actually the highest and low is the lowest
    for i in range(len(data["open"])):
        data["high"][i] = max(data["open"][i], data["close"][i], data["high"][i])
        data["low"][i] = min(data["open"][i], data["close"][i], data["low"][i])

    return pd.DataFrame(data)


@pytest.fixture
def sample_price_data(request):
    """
    Sample price data for indicator testing from cache.
    Falls back to synthetic data only if pytest is run with --allow-synthetic-data flag.
    """
    try:
        return get_cached_market_data(
            exchange_name=DEFAULT_TEST_EXCHANGE,
            symbol=DEFAULT_TEST_SYMBOL,
            timeframe=DEFAULT_TEST_TIMEFRAME,
        )
    except FileNotFoundError as e:
        # Check if we should allow synthetic data fallback
        if request.config.getoption("--allow-synthetic-data", default=False):
            logger.warning(
                "Using synthetic data because cached data is missing and --allow-synthetic-data flag is present"
            )
            return generate_sample_price_data(DEFAULT_TEST_SYMBOL)
        else:
            # Re-raise with a clear message about running the refresh script
            pytest.fail(f"Cached market data required: {e}")


@pytest.fixture
def real_market_data(request):
    """
    Parametrizable fixture to get market data for different symbols and timeframes.

    Usage:
        @pytest.mark.parametrize('market_data_params', [
            {'exchange': 'hyperliquid', 'symbol': 'BTC-USD', 'timeframe': '1h'},
            {'exchange': 'coinbase', 'symbol': 'ETH-USD', 'timeframe': '15m'},
        ], indirect=True)
        def test_with_different_data(real_market_data):
            # Use real_market_data as a DataFrame with OHLCV data
            ...
    """
    params = getattr(request, "param", {})
    exchange = params.get("exchange", DEFAULT_TEST_EXCHANGE)
    symbol = params.get("symbol", DEFAULT_TEST_SYMBOL)
    timeframe = params.get("timeframe", DEFAULT_TEST_TIMEFRAME)
    days = params.get("days", DEFAULT_DATA_DAYS)

    return get_cached_market_data(exchange, symbol, timeframe, days)


@pytest.fixture
def market_data_cache_dir():
    """Return the path to the market data cache directory."""
    return MARKET_DATA_CACHE_DIR


@pytest.fixture
def sample_config():
    """Sample application configuration."""
    config = AppConfig(
        log_level="INFO",
        webhook_enabled=False,
        webhook_port=8080,
        webhook_host="0.0.0.0",
        max_parallel_trades=1,
        polling_interval=60,
        dry_run=True,
        exchanges=[
            ExchangeConfig(
                name="mock_exchange",
                wallet_address="0x123",
                private_key="0xabc",
                testnet=True,
                use_as_main=True,
                use_as_hedge=True,
            )
        ],
        strategies=[
            TradingStrategyConfig(
                name="test_strategy",
                market="ETH",
                enabled=True,
                main_leverage=5.0,
                hedge_leverage=2.0,
                hedge_ratio=0.2,
                stop_loss_pct=10.0,
                take_profit_pct=20.0,
                max_position_size=100.0,
            )
        ],
        indicators=[
            IndicatorConfig(
                name="test_rsi",
                type="rsi",
                enabled=True,
                parameters={"period": 14, "overbought": 70, "oversold": 30},
            )
        ],
    )
    return config


@pytest.fixture
def mock_connector():
    """Mock exchange connector."""
    connector = MagicMock(spec=BaseConnector)

    # Set up basic methods
    connector.connect.return_value = True
    connector.get_markets.return_value = [
        {"symbol": "ETH", "base_asset": "ETH", "quote_asset": "USD", "min_size": 0.001}
    ]
    connector.get_ticker.return_value = {
        "symbol": "ETH",
        "last_price": 1500.0,
        "bid": 1499.0,
        "ask": 1501.0,
        "volume": 1000.0,
    }
    connector.get_account_balance.return_value = {"USD": 10000.0}
    connector.get_positions.return_value = []
    connector.close_position = MagicMock(return_value={"status": "success"})

    # Create an async mock for place_order
    async def mock_place_order(*args, **kwargs):
        return {
            "order_id": "mock_order_123",
            "symbol": "ETH",
            "side": "BUY",
            "size": 1.0,
            "price": 1500.0,
            "entry_price": 1500.0,
            "status": "FILLED",
            "leverage": 5.0,
            "position_id": "mock_position_1",
            "unrealized_pnl": 0.0,
            "liquidation_price": 0.0,
            "margin": 300.0,
            "timestamp": int(time.time() * 1000)
        }

    connector.place_order = mock_place_order

    # Add supports_derivatives property
    connector.supports_derivatives = True

    # Add market_types property
    connector.market_types = [MarketType.PERPETUAL]

    return connector


@pytest.fixture
def real_connector(request):
    """
    Parametrizable fixture to get a real connector instance.

    Usage:
        @pytest.mark.parametrize('real_connector', ['hyperliquid'], indirect=True)
        def test_with_real_connector(real_connector):
            # Use real_connector instance
            ...
    """
    connector_name = getattr(request, "param", DEFAULT_TEST_EXCHANGE)
    return ConnectorFactory.create_connector(connector_name)


@pytest.fixture
def mock_risk_manager():
    """Mock risk manager."""
    risk_manager = MagicMock(spec=RiskManager)

    # Set up basic methods
    risk_manager.calculate_position_size.return_value = (100.0, 5.0)  # size, leverage
    risk_manager.calculate_hedge_parameters.return_value = (20.0, 2.0)  # size, leverage
    risk_manager.validate_trade.return_value = (True, "Trade validated")

    return risk_manager


@pytest.fixture
def trading_engine(mock_connector, mock_risk_manager):
    """Trading engine instance with mock components."""
    engine = TradingEngine(
        main_connector=mock_connector,
        hedge_connector=mock_connector,  # Use the same mock for both
        risk_manager=mock_risk_manager,
        dry_run=True,
        polling_interval=1,
        max_parallel_trades=1,
    )
    return engine


@pytest.fixture
def sample_signal():
    """Sample trading signal."""
    return Signal(
        direction=SignalDirection.BUY,
        symbol="ETH",
        indicator="test_indicator",
        confidence=0.8,
        timestamp=1630000000000,
        params={"reason": "test signal"},
    )


# Add the command line option for allowing synthetic data fallback
def pytest_addoption(parser):
    parser.addoption(
        "--allow-synthetic-data",
        action="store_true",
        default=False,
        help="Allow using synthetic data when cached data is missing",
    )


@pytest.fixture
def market_data_params(request):
    """
    Fixture to provide market data parameters to the real_market_data fixture.

    This enables parametrization of the real_market_data fixture by passing
    parameters through this intermediary fixture.
    """
    return request.param


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "production: mark test that should only run in production")

    # Configure asyncio
    try:
        import pytest_asyncio
        config.option.asyncio_mode = "auto"
    except ImportError:
        pass

# Adjust the Python path to include the app directory
# This ensures that 'app' can be imported directly in the tests
APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app"))
sys.path.insert(0, os.path.dirname(APP_PATH))

print(f"conftest.py: Added to sys.path: {os.path.dirname(APP_PATH)}")
print(f"conftest.py: sys.path: {sys.path[:3]}")


@pytest.fixture(scope="session")
def app_path():
    """Return the absolute path to the app directory."""
    return APP_PATH

# Define a deterministic seed for reproducible tests
DETERMINISTIC_SEED = 42

@pytest.fixture
def price_dataframe():
    """
    Creates a sample price dataframe with deterministic values for testing.
    Uses the make_price_dataframe factory with pattern="trend".

    Returns:
        pd.DataFrame: A DataFrame with OHLCV price data
    """
    from tests._helpers.data_factory import make_price_dataframe

    # Create a price dataframe with a trending pattern
    return make_price_dataframe(rows=100, pattern="trend", noise=0.5, seed=DETERMINISTIC_SEED)

@pytest.fixture
def temp_csv_dir():
    """
    Creates a temporary directory for CSV files and cleans up after the test.

    Yields:
        Path: Path to the temporary directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def backtest_env(price_dataframe, temp_csv_dir):
    """
    Sets up a BacktestEngine with a DataManager and sample data.

    Args:
        price_dataframe: Sample price data
        temp_csv_dir: Temporary directory for CSV files

    Returns:
        tuple: (BacktestEngine, DataManager, symbol, interval)
    """
    # Save sample data to CSV
    symbol = "ETH-USD"
    interval = "1d"
    csv_path = temp_csv_dir / f"{symbol}_{interval}.csv"
    price_dataframe.to_csv(csv_path, index=False)

    # Create a DataManager and register a CSV data source
    data_manager = DataManager(data_dir=str(temp_csv_dir))
    data_manager.register_data_source("csv", CSVDataSource(str(temp_csv_dir)))

    # Create a BacktestEngine
    engine = BacktestEngine(
        data_manager=data_manager,
        initial_balance={"USD": 10000.0},
        maker_fee=0.001,
        taker_fee=0.002,
        slippage_model="fixed",
    )

    return (engine, data_manager, symbol, interval)

@pytest.fixture
def results_dir():
    """
    Creates a persistent timestamped directory for test results.
    These results will remain after test completion to allow viewing of reports.
    Use the `make clean-test-results` command to clean up old test results.

    Returns:
        Path: Path to the permanent directory for test results
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Use the correct path within the tests directory
    results_path = Path(__file__).parent / "__test_results__" / f"run_{timestamp}"
    results_path.mkdir(parents=True, exist_ok=True)
    print(f"Test results will be saved in: {results_path}")
    return results_path
