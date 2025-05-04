import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from app.backtesting.timeframe_manager import TimeframeManager


@pytest.fixture
def mock_data():
    """Create mock OHLCV data for testing."""
    # Create sample data for 1-minute timeframe
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(120)]  # 2 hours of data

    df_1m = pd.DataFrame({
        "timestamp": [int(ts.timestamp() * 1000) for ts in timestamps],
        "open": [100 + i * 0.1 for i in range(len(timestamps))],
        "high": [101 + i * 0.1 for i in range(len(timestamps))],
        "low": [99 + i * 0.1 for i in range(len(timestamps))],
        "close": [100.5 + i * 0.1 for i in range(len(timestamps))],
        "volume": [1000 + i * 10 for i in range(len(timestamps))]
    })

    # Create sample data for 5-minute timeframe
    timestamps_5m = [start_time + timedelta(minutes=i*5) for i in range(24)]  # 2 hours of data

    df_5m = pd.DataFrame({
        "timestamp": [int(ts.timestamp() * 1000) for ts in timestamps_5m],
        "open": [100 + i * 0.5 for i in range(len(timestamps_5m))],
        "high": [101 + i * 0.5 for i in range(len(timestamps_5m))],
        "low": [99 + i * 0.5 for i in range(len(timestamps_5m))],
        "close": [100.5 + i * 0.5 for i in range(len(timestamps_5m))],
        "volume": [5000 + i * 50 for i in range(len(timestamps_5m))]
    })

    # Create sample data for 1-hour timeframe
    timestamps_1h = [start_time + timedelta(hours=i) for i in range(2)]

    df_1h = pd.DataFrame({
        "timestamp": [int(ts.timestamp() * 1000) for ts in timestamps_1h],
        "open": [100, 106],
        "high": [106, 112],
        "low": [99, 105],
        "close": [105, 111],
        "volume": [60000, 62000]
    })

    return {"1m": df_1m, "5m": df_5m, "1h": df_1h}


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()

    # Create market regime subdirectories
    for regime in ["bull", "bear", "sideways", "normalized"]:
        os.makedirs(os.path.join(temp_dir, regime), exist_ok=True)

    yield temp_dir

    # Clean up after tests
    shutil.rmtree(temp_dir)


def create_test_datasets(data_dir, mock_data):
    """
    Create test dataset files in the specified directory.

    Args:
        data_dir: Directory to create test files in
        mock_data: Dictionary of mock dataframes
    """
    # Save datasets for different market regimes
    regimes = ["bull", "bear", "sideways"]
    symbols = ["BTC", "ETH"]

    for regime in regimes:
        regime_dir = os.path.join(data_dir, regime)

        for symbol in symbols:
            for timeframe, df in mock_data.items():
                # Create first dataset
                filename = f"{symbol}_{timeframe}_{regime}_1.csv"
                filepath = os.path.join(regime_dir, filename)
                df.to_csv(filepath, index=False)

                # Create second dataset for some combinations
                if timeframe in ["1h", "4h"] and regime == "bull":
                    filename = f"{symbol}_{timeframe}_{regime}_2.csv"
                    filepath = os.path.join(regime_dir, filename)
                    df.to_csv(filepath, index=False)


@pytest.fixture
def populated_temp_dir(temp_data_dir, mock_data):
    """Create a temporary directory with test dataset files."""
    create_test_datasets(temp_data_dir, mock_data)
    return temp_data_dir


def test_timeframe_manager_init():
    """Test TimeframeManager initialization."""
    manager = TimeframeManager(data_dir="test_data_dir")
    assert manager.data_dir == "test_data_dir"
    assert hasattr(manager, "data_manager")
    assert hasattr(manager, "timeframe_data")


def test_get_available_timeframes(populated_temp_dir):
    """Test getting available timeframes for a symbol."""
    manager = TimeframeManager(data_dir=populated_temp_dir)

    # Test without market regime filter
    timeframes = manager.get_available_timeframes("BTC")
    assert set(timeframes) == {"1m", "5m", "1h"}

    # Test with market regime filter
    bull_timeframes = manager.get_available_timeframes("BTC", market_regime="bull")
    assert set(bull_timeframes) == {"1m", "5m", "1h"}

    # Test symbol that doesn't exist
    nonexistent_timeframes = manager.get_available_timeframes("XRP")
    assert nonexistent_timeframes == []


def test_load_dataset(populated_temp_dir, mock_data):
    """Test loading a dataset from file."""
    manager = TimeframeManager(data_dir=populated_temp_dir)

    # Test loading existing file
    filepath = os.path.join(populated_temp_dir, "bull", "BTC_1h_bull_1.csv")
    df = manager.load_dataset(filepath)

    assert not df.empty
    assert "timestamp" in df.columns
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    assert "datetime" in df.columns  # Added by the manager

    # Test caching behavior
    manager.timeframe_data = {}  # Clear cache

    # Mock the pd.read_csv function to track calls
    with patch("pandas.read_csv", return_value=mock_data["1h"]) as mock_read_csv:
        # First call should read the file
        df1 = manager.load_dataset(filepath)
        mock_read_csv.assert_called_once()

        # Second call should use cached data
        mock_read_csv.reset_mock()
        df2 = manager.load_dataset(filepath)
        mock_read_csv.assert_not_called()

        # Force reload should read again
        mock_read_csv.reset_mock()
        df3 = manager.load_dataset(filepath, force_reload=True)
        mock_read_csv.assert_called_once()


def test_load_multi_timeframe_data(populated_temp_dir):
    """Test loading data for multiple timeframes."""
    manager = TimeframeManager(data_dir=populated_temp_dir)

    # Test loading multiple timeframes
    timeframes = ["1m", "5m", "1h"]
    data = manager.load_multi_timeframe_data(
        symbol="BTC",
        timeframes=timeframes,
        market_regime="bull",
        dataset_index=1
    )

    assert len(data) == 3
    assert all(tf in data for tf in timeframes)
    assert all(not data[tf].empty for tf in timeframes)

    # Test with nonexistent timeframe
    timeframes = ["1m", "5m", "1h", "4h"]  # 4h might not exist
    data = manager.load_multi_timeframe_data(
        symbol="BTC",
        timeframes=timeframes,
        market_regime="bull",
        dataset_index=1
    )

    # Should only contain existing timeframes
    assert len(data) <= 4
    assert all(not data[tf].empty for tf in data.keys())


def test_align_timeframes(populated_temp_dir, mock_data):
    """Test aligning multiple timeframe data to a common timeline."""
    manager = TimeframeManager(data_dir=populated_temp_dir)

    # Pre-process dataframes to add datetime column
    timeframe_data = {}
    for tf, df in mock_data.items():
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        timeframe_data[tf] = df

    # Test alignment to 1h timeframe
    aligned_data = manager.align_timeframes(timeframe_data, base_timeframe="1h")

    assert len(aligned_data) == 3
    assert "1m" in aligned_data
    assert "5m" in aligned_data
    assert "1h" in aligned_data

    # Check that all dataframes cover the same time period
    for tf in ["1m", "5m"]:
        assert aligned_data[tf]["datetime"].min() >= aligned_data["1h"]["datetime"].min()
        assert aligned_data[tf]["datetime"].max() <= aligned_data["1h"]["datetime"].max()


def test_get_current_candle(populated_temp_dir, mock_data):
    """Test getting the current candle for a specific timeframe at a given time."""
    manager = TimeframeManager(data_dir=populated_temp_dir)

    # Pre-process dataframes to add datetime column
    timeframe_data = {}
    for tf, df in mock_data.items():
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        timeframe_data[tf] = df

    # Test getting current 1h candle
    time_1h_30m = pd.Timestamp("2023-01-01 01:30:00")
    candle = manager.get_current_candle(timeframe_data, "1h", time_1h_30m)

    assert candle is not None
    assert candle["open"] == 106
    assert candle["high"] == 112
    assert candle["low"] == 105
    assert candle["close"] == 111

    # Test getting current 5m candle
    time_0h_17m = pd.Timestamp("2023-01-01 00:17:00")
    candle = manager.get_current_candle(timeframe_data, "5m", time_0h_17m)

    assert candle is not None
    assert candle["datetime"] <= time_0h_17m


def test_resample_on_the_fly(populated_temp_dir, mock_data):
    """Test resampling timeframe data on the fly during backtesting."""
    manager = TimeframeManager(data_dir=populated_temp_dir)

    # Mock the data_manager's resample_timeframe method
    manager.data_manager.resample_timeframe = MagicMock(return_value=mock_data["1h"])

    # Test resampling
    resampled = manager.resample_on_the_fly(mock_data["1m"], "1m", "1h")

    # Verify that the mocked method was called with correct arguments
    manager.data_manager.resample_timeframe.assert_called_once_with(mock_data["1m"], "1h")
    assert resampled is mock_data["1h"]
