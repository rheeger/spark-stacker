from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.indicators.base_indicator import SignalDirection
from app.indicators.macd_indicator import MACDIndicator


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
        "trigger_threshold": 0.001
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
            "symbol": "ETH",
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
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_macd"
    assert signal.confidence > 0.5
    assert signal.params.get("trigger") == "crosses_above_signal"

    # Test sell signal (MACD crosses below signal line)
    signal = macd.generate_signal(data.iloc[[3, 4]])
    assert signal is not None
    assert signal.direction == SignalDirection.SELL
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_macd"
    assert signal.confidence > 0.5
    assert signal.params.get("trigger") == "crosses_below_signal"

    # Test no signal (no crossover conditions)
    neutral_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
            "symbol": "ETH",
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
        params={"fast_period": 5, "slow_period": 10, "signal_period": 3}
    )  # Shorter periods for quicker signals

    # Use a larger dataset to ensure we get enough data points
    df = sample_price_data.copy()

    # Manipulate data to create a crossover scenario
    # First make price trend down, then reverse upward
    down_trend_start = 20
    up_trend_start = 40

    # Create a downtrend
    for i in range(down_trend_start, up_trend_start):
        df.loc[i, "close"] = df.loc[i-1, "close"] * 0.99  # 1% drop

    # Create an uptrend
    for i in range(up_trend_start, up_trend_start + 20):
        df.loc[i, "close"] = df.loc[i-1, "close"] * 1.01  # 1% increase

    processed_data, signal = macd.process(df)

    # Verify the data was processed
    assert "macd" in processed_data.columns
    assert "macd_signal" in processed_data.columns
    assert "macd_histogram" in processed_data.columns

    # We may or may not get a signal depending on the data patterns
    if signal is not None:
        assert signal.symbol == "ETH"
        assert signal.indicator == "test_macd"
        assert signal.direction in [SignalDirection.BUY, SignalDirection.SELL]


def test_macd_error_handling():
    """Test error handling in MACD indicator."""
    macd = MACDIndicator(name="test_macd")

    # Test with invalid DataFrame (missing close column)
    invalid_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "invalid_column": [1, 2, 3, 4, 5],
        }
    )

    with pytest.raises(ValueError, match="must contain a 'close' price column"):
        macd.calculate(invalid_df)

    # Test with insufficient data points
    insufficient_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
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
            "symbol": "ETH",
            "close": [1500, 1550, 1600, 1650, 1700],
        }
    )

    # Should return None if MACD column is missing
    assert macd.generate_signal(missing_macd_df) is None
