#!/usr/bin/env python3
"""
Spark-App CLI - Unified command line interface for backtest operations
"""
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import click
import pandas as pd

# Add the app directory to the path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two levels: utils -> tests -> spark-app (where app directory is)
spark_app_dir = os.path.dirname(os.path.dirname(current_dir))
# Path to the tests directory
tests_dir = os.path.dirname(current_dir)
sys.path.insert(0, spark_app_dir)

# Now use absolute imports with correct file names
from app.backtesting.backtest_engine import BacktestEngine
from app.backtesting.data_manager import CSVDataSource, DataManager
from app.backtesting.indicator_backtest_manager import IndicatorBacktestManager
from app.connectors.hyperliquid_connector import HyperliquidConnector
from app.indicators.indicator_factory import IndicatorFactory
from app.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

def get_default_output_dir() -> str:
    """
    Get the default output directory for CLI results.
    Creates a timestamped directory within tests/__test_results__/

    Returns:
        str: Path to the output directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(tests_dir, "__test_results__", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool):
    """Spark-App CLI for backtesting and indicator management."""
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level=log_level)
    logger.debug("Debug logging enabled")


@cli.command()
@click.option("--symbol", required=True, help="Trading symbol (e.g., BTC/USDT)")
@click.option("--timeframe", default="1h", help="Timeframe for analysis (e.g., 1h, 4h, 1d)")
@click.option("--data-file", help="Path to OHLCV CSV data file (optional)")
@click.option("--indicator", required=True, help="Indicator name from factory")
@click.option("--config-file", help="Path to indicator config YAML file")
@click.option("--start-date", help="Start date for backtest (YYYY-MM-DD)")
@click.option("--end-date", help="End date for backtest (YYYY-MM-DD)")
@click.option("--output-dir", help="Directory to save backtest results")
def backtest(
    symbol: str,
    timeframe: str,
    data_file: Optional[str],
    indicator: str,
    config_file: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    output_dir: Optional[str],
):
    """Run a backtest for a specific indicator against historical data."""
    logger.info(f"Running backtest for {indicator} on {symbol} {timeframe}")

    try:
        # Create indicator from factory
        indicator_instance = IndicatorFactory.create(indicator)

        if config_file and os.path.exists(config_file):
            logger.info(f"Loading config from {config_file}")
            indicator_instance.load_config(config_file)

        # Initialize backtest engine
        engine = BacktestEngine(symbol=symbol, timeframe=timeframe)

        if data_file:
            if not os.path.exists(data_file):
                logger.error(f"Data file not found: {data_file}")
                sys.exit(1)
            engine.load_csv_data(data_file)
        else:
            logger.info("No data file provided, will use default data source")
            # Add logic to fetch data if needed

        # Set date range if provided
        if start_date:
            engine.set_start_date(start_date)
        if end_date:
            engine.set_end_date(end_date)

        # Run backtest
        results = engine.run_backtest(indicator_instance)

        # Save results - use default directory if none specified
        if not output_dir:
            output_dir = get_default_output_dir()
            logger.info(f"Using default output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, f"{symbol}_{timeframe}_{indicator}_results.json")
        results.save_to_json(result_file)
        logger.info(f"Results saved to {result_file}")

        # Display summary
        print("\nBacktest Summary:")
        print(f"Total trades: {len(results.trades)}")
        print(f"Win rate: {results.win_rate:.2f}%")
        print(f"Profit factor: {results.profit_factor:.2f}")
        print(f"Max drawdown: {results.max_drawdown:.2f}%")

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument("indicator_name", required=True)
@click.option("--symbol", default="ETH-USD", help="Trading symbol")
@click.option("--timeframe", default="1h", help="Timeframe for analysis")
@click.option("--output-dir", help="Directory to save demo results")
def demo(
    indicator_name: str,
    symbol: str,
    timeframe: str,
    output_dir: Optional[str],
):
    """Run a demonstration backtest with preset configurations."""
    logger.info(f"Running demo for {indicator_name} on {symbol} {timeframe}")

    try:
        # Create data directory if it doesn't exist
        data_dir = os.path.join(spark_app_dir, "data", "market_datasets", "demo")
        os.makedirs(data_dir, exist_ok=True)

        # Standardize symbol format - replace / or - with _ for filenames
        file_symbol = symbol.replace("/", "_").replace("-", "_")

        # Create a synthetic data file for the demo
        create_demo_data_file(data_dir, file_symbol, timeframe)

        # Initialize data manager and create default data source for demo data
        data_manager = DataManager(data_dir=data_dir)

        # Initialize demo data source
        csv_data_source = CSVDataSource(data_dir=data_dir)
        data_manager.register_data_source("default", csv_data_source)

        # Initialize backtest engine with data manager
        backtest_engine = BacktestEngine(
            data_manager=data_manager,
            initial_balance={"USD": 10000.0},
            maker_fee=0.0001,
            taker_fee=0.0005
        )

        # Create manager with the backtest engine
        manager = IndicatorBacktestManager(backtest_engine=backtest_engine)

        # Set output directory - use default if none provided
        if not output_dir:
            output_dir = get_default_output_dir()
            logger.info(f"Using default output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        manager.set_output_directory(output_dir)

        # Run indicator backtest with default settings
        result_paths = manager.run_indicator_backtest(
            indicator_name=indicator_name,
            symbol=file_symbol,  # Use the same symbol format as the file
            timeframe=timeframe,
            generate_report=True
        )

        # Print paths to results
        if result_paths:
            print("\nDemo completed successfully")
            print(f"Report: {result_paths.get('report_path')}")
            print(f"JSON results: {result_paths.get('json_path')}")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        sys.exit(1)


@cli.command(name="real-data")
@click.argument("indicator_name", required=True)
@click.option("--symbol", default="ETH-USD", help="Trading symbol")
@click.option("--timeframe", default="1m", help="Timeframe for analysis")
@click.option("--days", default=10, help="Number of days of historical data to fetch")
@click.option("--output-dir", help="Directory to save backtest results")
@click.option("--testnet", is_flag=True, default=True, help="Use Hyperliquid testnet")
def real_data(
    indicator_name: str,
    symbol: str,
    timeframe: str,
    days: int,
    output_dir: Optional[str],
    testnet: bool,
):
    """Run a backtest with real market data from Hyperliquid."""
    logger.info(f"Running {indicator_name} backtest with real Hyperliquid data")
    logger.info(f"Parameters: {symbol} {timeframe}, {days} days, testnet={testnet}")

    try:
        # Create data directory for real data
        data_dir = os.path.join(spark_app_dir, "data", "market_datasets", "real")
        os.makedirs(data_dir, exist_ok=True)

        # Fetch real data from Hyperliquid
        data_file_path = fetch_hyperliquid_data(
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            data_dir=data_dir,
            testnet=testnet
        )

        if not data_file_path or not os.path.exists(data_file_path):
            logger.error("Failed to fetch real market data")
            sys.exit(1)

        # Initialize data manager with real data
        data_manager = DataManager(data_dir=data_dir)
        csv_data_source = CSVDataSource(data_dir=data_dir)
        data_manager.register_data_source("default", csv_data_source)

        # Initialize backtest engine with data manager
        backtest_engine = BacktestEngine(
            data_manager=data_manager,
            initial_balance={"USD": 10000.0},
            maker_fee=0.0001,
            taker_fee=0.0005
        )

        # Create manager with the backtest engine
        manager = IndicatorBacktestManager(backtest_engine=backtest_engine)

        # Set output directory - use default if none provided
        if not output_dir:
            output_dir = get_default_output_dir()
            logger.info(f"Using default output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        manager.set_output_directory(output_dir)

        # Standardize symbol format for filename matching
        file_symbol = symbol.replace("/", "_").replace("-", "_")

        # Run indicator backtest with real data
        result_paths = manager.run_indicator_backtest(
            indicator_name=indicator_name,
            symbol=file_symbol,
            timeframe=timeframe,
            generate_report=True
        )

        # Print results
        if result_paths:
            print(f"\n🎯 Real data backtest completed successfully!")
            print(f"📊 Data source: Hyperliquid {'testnet' if testnet else 'mainnet'}")
            print(f"📈 Symbol: {symbol}")
            print(f"⏱️  Timeframe: {timeframe}")
            print(f"📅 Days of data: {days}")
            print(f"📋 Report: {result_paths.get('report_path')}")
            print(f"📄 JSON results: {result_paths.get('json_path')}")
            print(f"💾 Data file: {data_file_path}")
        else:
            print("❌ Backtest completed but no results generated")

    except Exception as e:
        import traceback
        logger.error(f"Real data backtest failed: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


def fetch_hyperliquid_data(
    symbol: str,
    timeframe: str,
    days: int,
    data_dir: str,
    testnet: bool = True
) -> Optional[str]:
    """
    Fetch real market data from Hyperliquid and save it as CSV.

    Args:
        symbol: Trading symbol (e.g., 'ETH-USD')
        timeframe: Timeframe interval (e.g., '1m')
        days: Number of days of historical data
        data_dir: Directory to save the data
        testnet: Whether to use testnet

    Returns:
        Path to the saved CSV file or None if failed
    """
    try:
        logger.info(f"Initializing Hyperliquid connector (testnet={testnet})...")

        # Initialize Hyperliquid connector
        connector = HyperliquidConnector(
            name="cli_hyperliquid",
            testnet=testnet
        )

        # Connect to Hyperliquid
        logger.info("Connecting to Hyperliquid...")
        if not connector.connect():
            logger.error("Failed to connect to Hyperliquid")
            return None

        # Calculate time range
        end_time = int(time.time() * 1000)  # Current time in milliseconds
        start_time = end_time - (days * 24 * 60 * 60 * 1000)  # days ago

        logger.info(f"Fetching {days} days of {timeframe} data for {symbol}...")
        logger.info(f"Time range: {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")

        # Fetch historical candles
        candles = connector.get_historical_candles(
            symbol=symbol,
            interval=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=days * 24 * 60 if timeframe == "1m" else 1000  # Adjust limit based on timeframe
        )

        if not candles:
            logger.error(f"No candles received for {symbol}")
            return None

        logger.info(f"Received {len(candles)} candles from Hyperliquid")

        # Convert to DataFrame with the expected format
        df = pd.DataFrame(candles)

        # Ensure we have the required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return None

        # Sort by timestamp
        df = df.sort_values("timestamp")

        # Convert columns to proper types
        df["timestamp"] = df["timestamp"].astype(int)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Create filename
        file_symbol = symbol.replace("/", "_").replace("-", "_")
        filename = f"{file_symbol}_{timeframe}.csv"
        file_path = os.path.join(data_dir, filename)

        # Save to CSV
        df[required_columns].to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} candles to {file_path}")

        # Log data summary
        if len(df) > 0:
            logger.info(f"Data summary:")
            logger.info(f"  - Time range: {datetime.fromtimestamp(df['timestamp'].min()/1000)} to {datetime.fromtimestamp(df['timestamp'].max()/1000)}")
            logger.info(f"  - Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            logger.info(f"  - Average volume: {df['volume'].mean():.2f}")

        return file_path

    except Exception as e:
        logger.error(f"Error fetching Hyperliquid data: {str(e)}")
        return None


def create_demo_data_file(data_dir: str, symbol: str, timeframe: str) -> str:
    """
    Create a synthetic data file for demo purposes.

    Args:
        data_dir: Directory to save the data file
        symbol: Trading symbol (already formatted for filename)
        timeframe: Timeframe interval

    Returns:
        Path to the created file
    """
    # Use simple filename format
    filename = f"{symbol}_{timeframe}.csv"
    file_path = os.path.join(data_dir, filename)

    # Check if file already exists
    if os.path.exists(file_path):
        logger.info(f"Demo data file already exists: {file_path}")
        return file_path

    # Create synthetic OHLCV data
    import numpy as np
    import pandas as pd

    # Create timestamps for the last 365 days
    end_time = pd.Timestamp.now()
    if timeframe == "1h":
        # Hourly data for the past year
        timestamps = pd.date_range(end=end_time, periods=24*365, freq="H")
    elif timeframe == "1d":
        # Daily data for the past year
        timestamps = pd.date_range(end=end_time, periods=365, freq="D")
    else:
        # Default to daily data
        timestamps = pd.date_range(end=end_time, periods=365, freq="D")

    # Convert timestamps to milliseconds
    timestamps_ms = [int(ts.timestamp() * 1000) for ts in timestamps]

    # Create price data (starting at 1000 with random walk)
    np.random.seed(42)  # For reproducibility
    n = len(timestamps)

    # Generate starting values
    price = 1000.0

    # Initialize arrays with the right length
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)
    volumes = np.zeros(n)

    # Set initial values
    opens[0] = price
    highs[0] = price * 1.01
    lows[0] = price * 0.99
    closes[0] = price
    volumes[0] = np.random.uniform(100, 1000)

    # Generate OHLCV data with a random walk
    for i in range(1, n):
        # Random price change (-2% to +2%)
        change = np.random.uniform(-0.02, 0.02)
        price *= (1 + change)

        # Generate OHLCV values
        opens[i] = price
        highs[i] = price * np.random.uniform(1.0, 1.03)
        lows[i] = price * np.random.uniform(0.97, 1.0)
        closes[i] = np.random.uniform(lows[i], highs[i])
        volumes[i] = np.random.uniform(100, 1000)

    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps_ms,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })

    # Save to CSV
    df.to_csv(file_path, index=False)
    logger.info(f"Created synthetic data file: {file_path}")

    return file_path


@cli.command(name="demo-macd")
@click.option("--symbol", default="ETH/USDT", help="Trading symbol")
@click.option("--timeframe", default="1h", help="Timeframe for analysis")
@click.option("--output-dir", help="Directory to save demo results")
def demo_macd(
    symbol: str,
    timeframe: str,
    output_dir: Optional[str],
):
    """Run a demonstration backtest with MACD indicator."""
    # This is equivalent to calling demo with indicator_name=MACD
    ctx = click.get_current_context()
    ctx.invoke(demo, indicator_name="MACD", symbol=symbol, timeframe=timeframe, output_dir=output_dir)


@cli.command()
def list_indicators():
    """List all available indicators from the factory."""
    indicators = IndicatorFactory.get_available_indicators()

    print("\nAvailable Indicators:")
    for idx, indicator_name in enumerate(indicators, 1):
        # Display indicator name in uppercase to match the test expectation
        print(f"{idx}. {indicator_name.upper()}")


if __name__ == "__main__":
    cli()
