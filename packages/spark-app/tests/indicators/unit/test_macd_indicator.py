import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from app.indicators.base_indicator import SignalDirection
from app.indicators.macd_indicator import MACDIndicator

# Initialize logger for the test file
logger = logging.getLogger(__name__)

def test_macd_initialization():
    """Test MACD indicator initialization with default and custom parameters."""
    # Test with default parameters
    macd = MACDIndicator(name="test_macd")
    assert macd.name == "test_macd"
    assert macd.fast_period == 12
    assert macd.slow_period == 26
    assert macd.signal_period == 9
    assert macd.trigger_threshold == 0

    # Test with custom parameters
    custom_params = {
        "fast_period": 8,
        "slow_period": 21,
        "signal_period": 5,
        "trigger_threshold": 0.001,
    }
    macd_custom = MACDIndicator(name="custom_macd", params=custom_params)
    assert macd_custom.name == "custom_macd"
    assert macd_custom.fast_period == 8
    assert macd_custom.slow_period == 21
    assert macd_custom.signal_period == 5
    assert macd_custom.trigger_threshold == 0.001


def test_macd_calculation(sample_price_data):
    """Test MACD calculation using sample price data."""
    macd = MACDIndicator(name="test_macd")
    result = macd.calculate(sample_price_data)

    # Verify MACD columns were added
    required_columns = ["macd", "macd_signal", "macd_histogram"]
    for col in required_columns:
        assert col in result.columns

    # After slow_period + signal_period rows, values should be valid
    min_periods = macd.slow_period + macd.signal_period
    for col in required_columns:
        assert not pd.isna(result[col].iloc[min_periods])

    # Verify crossover detection columns
    assert "macd_crosses_above_signal" in result.columns
    assert "macd_crosses_below_signal" in result.columns
    assert "macd_crosses_above_zero" in result.columns
    assert "macd_crosses_below_zero" in result.columns


def test_macd_signal_generation():
    """Test signal generation based on MACD values."""
    # Create a mock DataFrame with MACD values
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH-USD",
            "close": [1500, 1550, 1600, 1650, 1600],
            "macd": [-2, -1, 0, 1, 0.5],
            "macd_signal": [-1, -1.2, -0.5, 0.5, 0.8],
            "macd_histogram": [-1, 0.2, 0.5, 0.5, -0.3],
            "macd_crosses_above_signal": [False, True, False, False, False],
            "macd_crosses_below_signal": [False, False, False, False, True],
            "macd_crosses_above_zero": [False, False, True, False, False],
            "macd_crosses_below_zero": [False, False, False, False, False],
        }
    )

    macd = MACDIndicator(name="test_macd")

    # Test buy signal (MACD crosses above signal line)
    signal = macd.generate_signal(data.iloc[[0, 1]])
    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.symbol == "ETH-USD"
    assert signal.indicator == "test_macd"
    assert signal.confidence > 0.5
    assert signal.params.get("trigger") == "crosses_above_signal"

    # Test sell signal (MACD crosses below signal line)
    signal = macd.generate_signal(data.iloc[[3, 4]])
    assert signal is not None
    assert signal.direction == SignalDirection.SELL
    assert signal.symbol == "ETH-USD"
    assert signal.indicator == "test_macd"
    assert signal.confidence > 0.5
    assert signal.params.get("trigger") == "crosses_below_signal"

    # Test no signal (no crossover conditions)
    neutral_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
            "symbol": "ETH-USD",
            "close": [1500, 1550],
            "macd": [1, 1.2],
            "macd_signal": [0.5, 0.7],
            "macd_histogram": [0.5, 0.5],
            "macd_crosses_above_signal": [False, False],
            "macd_crosses_below_signal": [False, False],
            "macd_crosses_above_zero": [False, False],
            "macd_crosses_below_zero": [False, False],
        }
    )

    signal = macd.generate_signal(neutral_data)
    assert signal is None


def test_macd_process_method(sample_price_data):
    """Test the combined process method."""
    macd = MACDIndicator(
        name="test_macd",
        params={"fast_period": 5, "slow_period": 10, "signal_period": 3},
    )  # Shorter periods for quicker signals

    # Use a larger dataset to ensure we get enough data points
    df = sample_price_data.copy()

    # Get actual row indices - we'll use the timestamp index
    timestamps = df.index.tolist()

    # Choose points at 15% and 25% through the data
    down_trend_start_idx = int(len(timestamps) * 0.15)
    up_trend_start_idx = int(len(timestamps) * 0.25)

    down_trend_periods = 20
    up_trend_periods = 20

    # Make sure we don't go beyond the dataframe bounds
    if up_trend_start_idx + up_trend_periods >= len(timestamps):
        # Adjust to ensure we have enough room for our modifications
        down_trend_start_idx = 20
        up_trend_start_idx = 40
        down_trend_periods = min(20, up_trend_start_idx - down_trend_start_idx)
        up_trend_periods = min(20, len(timestamps) - up_trend_start_idx)

    # Create a downtrend
    current_price = df.iloc[down_trend_start_idx]["close"]
    for i in range(down_trend_periods):
        idx = down_trend_start_idx + i
        if idx < len(timestamps):
            current_price = current_price * 0.99  # 1% drop
            df.loc[timestamps[idx], "close"] = current_price

    # Create an uptrend
    current_price = df.iloc[up_trend_start_idx]["close"]
    for i in range(up_trend_periods):
        idx = up_trend_start_idx + i
        if idx < len(timestamps):
            current_price = current_price * 1.01  # 1% increase
            df.loc[timestamps[idx], "close"] = current_price

    processed_data, signal = macd.process(df)

    # Verify the data was processed
    assert "macd" in processed_data.columns
    assert "macd_signal" in processed_data.columns
    assert "macd_histogram" in processed_data.columns

    # We may or may not get a signal depending on the data patterns
    if signal is not None:
        assert signal.symbol == "ETH-USD"
        assert signal.indicator == "test_macd"
        assert signal.direction in [SignalDirection.BUY, SignalDirection.SELL]


def test_macd_error_handling():
    """Test error handling in MACD indicator."""
    macd = MACDIndicator(name="test_macd")

    # Test with invalid DataFrame (missing close column)
    invalid_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH-USD",
            "invalid_column": [1, 2, 3, 4, 5],
        }
    )

    with pytest.raises(ValueError, match="must contain a 'close' or 'c' price column"):
        macd.calculate(invalid_df)

    # Test with insufficient data points
    insufficient_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH-USD",
            "close": [1500, 1550, 1600, 1650, 1700],
        }
    )

    # This should not raise an error but log a warning and return the data with NaN MACD values
    result = macd.calculate(insufficient_df)
    assert "macd" in result.columns
    assert pd.isna(result["macd"]).all()
    assert pd.isna(result["macd_signal"]).all()
    assert pd.isna(result["macd_histogram"]).all()

    # Test generate_signal with missing MACD column
    missing_macd_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH-USD",
            "close": [1500, 1550, 1600, 1650, 1700],
        }
    )

    # Should return None if MACD column is missing
    assert macd.generate_signal(missing_macd_df) is None


def test_macd_mvp_parameters_signal_on_sample_data(sample_price_data):
    """Test MACD calculation and signal generation with MVP parameters (8, 21, 5) on sample data."""
    # Initialize with MVP parameters
    mvp_params = {
        "fast_period": 8,
        "slow_period": 21,
        "signal_period": 5,
    }
    macd_mvp = MACDIndicator(name="macd_mvp", params=mvp_params)

    # Use the fixture data
    df = sample_price_data.copy()

    # Generate signals for the entire dataframe
    result_df = macd_mvp.generate_signals(df)

    # 1. Verify MACD columns were added
    required_columns = ["macd", "macd_signal", "macd_histogram", "signal", "confidence"]
    for col in required_columns:
        assert col in result_df.columns, f"Column '{col}' missing in result DataFrame"

    # 2. Check that calculations ran (values are not all NaN after initial period)
    min_periods = macd_mvp.slow_period + macd_mvp.signal_period
    assert not result_df['macd'].iloc[min_periods:].isna().all(), "MACD column is all NaN after initial period"
    assert not result_df['macd_signal'].iloc[min_periods:].isna().all(), "MACD Signal column is all NaN after initial period"
    assert not result_df['macd_histogram'].iloc[min_periods:].isna().all(), "MACD Histogram column is all NaN after initial period"

    # 3. Verify that both BUY and SELL signals were generated somewhere in the data
    assert not result_df["signal"].isna().all(), "Signal column is all NaN"
    signal_values = result_df["signal"].dropna().unique()

    # Use explicit comparison instead of 'in' operator
    assert any(
        s == SignalDirection.BUY for s in signal_values
    ), f"No BUY signals generated with MVP parameters on sample data. Signals found: {signal_values}"
    assert any(
        s == SignalDirection.SELL for s in signal_values
    ), f"No SELL signals generated with MVP parameters on sample data. Signals found: {signal_values}"

    # 4. Verify confidence is populated for signals
    buy_signals_df = result_df[result_df["signal"] == SignalDirection.BUY]
    sell_signals_df = result_df[result_df["signal"] == SignalDirection.SELL]

    assert not buy_signals_df.empty, "No BUY signals found to check confidence"
    assert (buy_signals_df["confidence"] > 0).all(), "Confidence for BUY signals should be positive"

    assert not sell_signals_df.empty, "No SELL signals found to check confidence"
    assert (sell_signals_df["confidence"] > 0).all(), "Confidence for SELL signals should be positive"

    logger.info(f"MACD MVP test generated {len(buy_signals_df)} BUY and {len(sell_signals_df)} SELL signals.")
