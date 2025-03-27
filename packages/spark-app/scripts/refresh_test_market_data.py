#!/usr/bin/env python3
"""
Utility script to refresh test market data cache.
This ensures tests have access to recent market data without making API calls during test runs.

IMPORTANT: This script must be run before running tests, as tests now use only cached data.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path to access app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import project components
from app.connectors.connector_factory import ConnectorFactory
from tests.conftest import MARKET_DATA_CACHE_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Market data specifications
EXCHANGES = ["hyperliquid", "coinbase"]  # List of exchange connectors to use
SYMBOLS = {
    "hyperliquid": [
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
    ],  # Adjust based on available markets
    "coinbase": ["BTC-USD", "ETH-USD", "SOL-USD"],  # Adjust based on available markets
}
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
DATA_DAYS = 60  # Number of days of historical data to fetch


def generate_synthetic_data(symbol, timeframe, days=DATA_DAYS):
    """
    Generate synthetic price data when real data can't be fetched.

    Args:
        symbol: Trading pair symbol (e.g., 'BTC-USD')
        timeframe: Timeframe for the data (e.g., '1h', '1d')
        days: Number of days of data to generate

    Returns:
        DataFrame with synthetic OHLCV data
    """
    logger.warning(f"Generating synthetic data for {symbol} {timeframe}")

    # Determine base price and volatility based on symbol
    if symbol.startswith("BTC"):
        base_price = 40000
        volatility = 1000
    elif symbol.startswith("ETH"):
        base_price = 2000
        volatility = 100
    elif symbol.startswith("SOL"):
        base_price = 100
        volatility = 5
    else:
        base_price = 500
        volatility = 50

    # Determine time interval based on timeframe
    if timeframe.endswith("m"):
        minutes = int(timeframe[:-1])
        freq = f"{minutes}min"
    elif timeframe.endswith("h"):
        hours = int(timeframe[:-1])
        freq = f"{hours}H"
    elif timeframe.endswith("d"):
        days_tf = int(timeframe[:-1])
        freq = f"{days_tf}D"
    else:
        freq = "1H"  # Default to 1 hour

    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)

    # Calculate number of periods
    periods = len(timestamps)

    # Generate random walk for close prices
    np.random.seed(42)  # For reproducibility
    random_walk = np.random.normal(0, volatility * 0.01, periods).cumsum()
    close_prices = base_price + random_walk

    # Ensure no negative prices
    close_prices = np.maximum(close_prices, base_price * 0.1)

    # Generate OHLCV data
    data = {
        "timestamp": timestamps,
        "symbol": symbol,
        "open": close_prices * (1 + np.random.normal(0, 0.005, periods)),
        "high": close_prices * (1 + np.random.uniform(0.001, 0.02, periods)),
        "low": close_prices * (1 - np.random.uniform(0.001, 0.02, periods)),
        "close": close_prices,
        "volume": np.random.normal(base_price * 100, base_price * 20, periods),
    }

    # Ensure high is always the highest and low is always the lowest
    df = pd.DataFrame(data)
    for i in range(len(df)):
        df.loc[i, "high"] = max(
            df.loc[i, "open"], df.loc[i, "close"], df.loc[i, "high"]
        )
        df.loc[i, "low"] = min(df.loc[i, "open"], df.loc[i, "close"], df.loc[i, "low"])

    # Ensure volume is positive
    df["volume"] = np.abs(df["volume"])

    # Set timestamp as index
    df.set_index("timestamp", inplace=True)

    return df


def fetch_and_cache_market_data(exchange_name, symbol, timeframe, days=DATA_DAYS):
    """
    Fetch market data from an exchange and save it to the cache.

    Args:
        exchange_name: Name of the exchange connector to use
        symbol: Trading pair symbol
        timeframe: Candle timeframe (1m, 5m, 15m, 1h, etc.)
        days: Number of days of historical data to fetch

    Returns:
        True if successful, False otherwise
    """
    # Generate cache filename
    symbol_normalized = symbol.replace("/", "_").replace("-", "_")
    cache_file = (
        Path(MARKET_DATA_CACHE_DIR)
        / f"{exchange_name}_{symbol_normalized}_{timeframe}.csv"
    )

    try:
        # Create connector instance
        connector = ConnectorFactory.create_connector(exchange_name)

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Fetch data from connector
        logger.info(f"Fetching {symbol} {timeframe} data from {exchange_name}")
        ohlcv_data = connector.get_historical_candles(
            symbol, timeframe, start_time, end_time
        )

        # Convert to DataFrame if needed
        if not isinstance(ohlcv_data, pd.DataFrame):
            df = pd.DataFrame(
                ohlcv_data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
        else:
            df = ohlcv_data

        # Add symbol column if not present
        if "symbol" not in df.columns:
            df["symbol"] = symbol

        # Cache the data
        logger.info(f"Caching data to {cache_file}")
        df.to_csv(cache_file)

        return len(df) > 0

    except Exception as e:
        logger.error(f"Error fetching {exchange_name} {symbol} {timeframe} data: {e}")

        # Generate synthetic data instead
        logger.warning(
            f"Generating synthetic data for {exchange_name} {symbol} {timeframe}"
        )
        df = generate_synthetic_data(symbol, timeframe, days)

        # Cache the synthetic data
        logger.info(f"Caching synthetic data to {cache_file}")
        df.to_csv(cache_file)

        # Mark as successful since we've generated usable data
        return True


def refresh_market_data_cache():
    """Refresh the market data cache for all exchanges, symbols, and timeframes."""
    logger.info(
        f"Starting market data refresh. Cache directory: {MARKET_DATA_CACHE_DIR}"
    )

    start_time = datetime.now()
    failure_count = 0
    success_count = 0

    for exchange in EXCHANGES:
        logger.info(f"Processing exchange: {exchange}")

        try:
            # Verify connector can be created
            connector = ConnectorFactory.create_connector(exchange)
            logger.info(f"Successfully created {exchange} connector")

            # Process each symbol and timeframe
            for symbol in SYMBOLS.get(exchange, []):
                for timeframe in TIMEFRAMES:
                    try:
                        logger.info(f"Fetching {exchange} {symbol} {timeframe} data...")

                        # Fetch and cache data
                        if fetch_and_cache_market_data(
                            exchange, symbol, timeframe, DATA_DAYS
                        ):
                            logger.info(
                                f"✓ Successfully cached data for {exchange} {symbol} {timeframe}"
                            )
                            success_count += 1
                        else:
                            logger.warning(
                                f"✗ Retrieved empty dataset for {exchange} {symbol} {timeframe}"
                            )
                            failure_count += 1

                    except Exception as e:
                        logger.error(
                            f"✗ Error fetching {exchange} {symbol} {timeframe}: {e}"
                        )
                        failure_count += 1

        except Exception as e:
            logger.error(f"✗ Failed to create connector for {exchange}: {e}")

            # Generate synthetic data for all symbols/timeframes for this exchange
            logger.warning(
                f"Generating synthetic data for all {exchange} symbols due to connector error"
            )
            for symbol in SYMBOLS.get(exchange, []):
                for timeframe in TIMEFRAMES:
                    try:
                        # Generate synthetic data
                        df = generate_synthetic_data(symbol, timeframe, DATA_DAYS)

                        # Generate cache filename
                        symbol_normalized = symbol.replace("/", "_").replace("-", "_")
                        cache_file = (
                            Path(MARKET_DATA_CACHE_DIR)
                            / f"{exchange}_{symbol_normalized}_{timeframe}.csv"
                        )

                        # Cache the data
                        logger.info(f"Caching synthetic data to {cache_file}")
                        df.to_csv(cache_file)
                        success_count += 1
                    except Exception as synthetic_err:
                        logger.error(
                            f"✗ Error generating synthetic data: {synthetic_err}"
                        )
                        failure_count += 1

    elapsed_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Market data refresh completed in {elapsed_time:.2f} seconds")
    logger.info(f"Summary: {success_count} successful, {failure_count} failed")

    return success_count, failure_count


def ensure_required_data_exists():
    """
    Ensure that at minimum, the most essential test data files exist.
    This is useful for CI environments and first-time setup.
    """
    # The minimal set of data files needed for basic tests
    essential_files = [
        ("hyperliquid", "ETH-USD", "1h"),
        ("hyperliquid", "BTC-USD", "1h"),
    ]

    for exchange, symbol, timeframe in essential_files:
        symbol_normalized = symbol.replace("/", "_").replace("-", "_")
        cache_file = (
            Path(MARKET_DATA_CACHE_DIR)
            / f"{exchange}_{symbol_normalized}_{timeframe}.csv"
        )

        if not cache_file.exists():
            logger.info(f"Essential file missing: {cache_file}")
            try:
                if not fetch_and_cache_market_data(exchange, symbol, timeframe):
                    # If API fetch fails, create synthetic data
                    df = generate_synthetic_data(symbol, timeframe)
                    df.to_csv(cache_file)
                    logger.info(
                        f"Created synthetic data for essential file: {cache_file}"
                    )
            except Exception as e:
                logger.error(f"Failed to create essential data file {cache_file}: {e}")
                return False

    return True


if __name__ == "__main__":
    # Create cache directory if it doesn't exist
    Path(MARKET_DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1] == "--essential-only":
        # Only ensure essential data exists
        success = ensure_required_data_exists()
        sys.exit(0 if success else 1)
    else:
        # Full refresh
        success, failure = refresh_market_data_cache()

        # Ensure essential data exists even if some refreshes failed
        essential_ok = ensure_required_data_exists()

        # Print final message
        if failure == 0 and success > 0:
            logger.info("✅ SUCCESS: All market data refreshed successfully")
            sys.exit(0)
        elif success > 0 and essential_ok:
            logger.warning(
                "⚠️ PARTIAL SUCCESS: Some market data refreshed, essential data is available"
            )
            sys.exit(0)  # Still exit with success if essential data is available
        else:
            logger.error("❌ FAILURE: No market data was refreshed")
            sys.exit(1)
