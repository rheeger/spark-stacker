import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add the package root to the Python path so imports work correctly
# when running pytest from the command line
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from app.connectors.base_connector import (
    BaseConnector,
    MarketType,
    OrderSide,
    OrderType,
)
from app.connectors.connector_factory import ConnectorFactory
from app.core.trading_engine import TradingEngine
from app.indicators.base_indicator import BaseIndicator, Signal, SignalDirection
from app.risk_management.risk_manager import RiskManager

# Import project components
from app.utils.config import (
    AppConfig,
    ExchangeConfig,
    IndicatorConfig,
    TradingStrategyConfig,
)

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
MARKET_DATA_CACHE_DIR = Path(__file__).parent / "test_data" / "market_data"

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
        f"Please run 'python scripts/refresh_test_market_data.py' first to populate the cache."
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
def sample_price_data():
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
        if pytest.config.getoption("--allow-synthetic-data", default=False):
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

    # Mock the place_order method to return a successful order
    connector.place_order.return_value = {
        "order_id": "mock_order_123",
        "symbol": "ETH",
        "side": "BUY",
        "size": 1.0,
        "price": 1500.0,
        "status": "FILLED",
    }

    # Add market_types property
    connector.market_types = MarketType.SPOT

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
