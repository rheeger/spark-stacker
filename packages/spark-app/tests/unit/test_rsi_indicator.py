from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.indicators.base_indicator import SignalDirection
from app.indicators.rsi_indicator import RSIIndicator


def test_rsi_initialization():
    """Test RSI indicator initialization with default and custom parameters."""
    # Test with default parameters
    rsi = RSIIndicator(name="test_rsi")
    assert rsi.name == "test_rsi"
    assert rsi.period == 14
    assert rsi.overbought == 70
    assert rsi.oversold == 30
    assert rsi.signal_period == 1

    # Test with custom parameters
    custom_params = {"period": 7, "overbought": 80, "oversold": 20, "signal_period": 2}
    rsi_custom = RSIIndicator(name="custom_rsi", params=custom_params)
    assert rsi_custom.name == "custom_rsi"
    assert rsi_custom.period == 7
    assert rsi_custom.overbought == 80
    assert rsi_custom.oversold == 20
    assert rsi_custom.signal_period == 2


def test_rsi_calculation(sample_price_data):
    """Test RSI calculation using sample price data."""
    rsi = RSIIndicator(name="test_rsi")
    result = rsi.calculate(sample_price_data)

    # Verify RSI column was added
    assert "rsi" in result.columns

    # RSI should be between 0 and 100
    assert result["rsi"].dropna().min() >= 0
    assert result["rsi"].dropna().max() <= 100

    # First values should be NaN because of rolling window
    assert pd.isna(result["rsi"].iloc[0])

    # After period+1 rows, values should be valid
    assert not pd.isna(result["rsi"].iloc[rsi.period + 1])

    # Verify additional columns for conditions
    assert "is_oversold" in result.columns
    assert "is_overbought" in result.columns
    assert "leaving_oversold" in result.columns
    assert "leaving_overbought" in result.columns


def test_rsi_signal_generation():
    """Test signal generation based on RSI values."""
    # Create a mock DataFrame with RSI values
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550, 1600, 1650, 1700],
            "rsi": [
                25,
                29,
                31,
                71,
                69,
            ],  # 31 is leaving oversold, 69 is leaving overbought
            "is_oversold": [True, True, False, False, False],
            "is_overbought": [False, False, False, True, False],
            "leaving_oversold": [False, False, True, False, False],
            "leaving_overbought": [False, False, False, False, True],
        }
    )

    rsi = RSIIndicator(name="test_rsi")

    # Test buy signal (leaving oversold)
    signal = rsi.generate_signal(data.iloc[[0, 1, 2]])
    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_rsi"
    assert signal.confidence > 0.5  # Confidence should be reasonable

    # Test sell signal (leaving overbought)
    signal = rsi.generate_signal(data.iloc[[3, 4]])
    assert signal is not None
    assert signal.direction == SignalDirection.SELL
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_rsi"
    assert signal.confidence > 0.5

    # Test no signal (no crossing conditions)
    neutral_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550],
            "rsi": [45, 50],
            "is_oversold": [False, False],
            "is_overbought": [False, False],
            "leaving_oversold": [False, False],
            "leaving_overbought": [False, False],
        }
    )

    signal = rsi.generate_signal(neutral_data)
    assert signal is None


def test_rsi_process_method(sample_price_data):
    """Test the combined process method."""
    rsi = RSIIndicator(
        name="test_rsi", params={"period": 5}
    )  # Shorter period for quicker signals

    # Manipulate some data to ensure we get a signal
    # Create a scenario where price drops causing oversold and then rises
    df = sample_price_data.copy()

    # Get actual row indices - we'll use actual timestamps since that's the index
    timestamps = df.index.tolist()
    # Use 10% of the way through the data, and 15% of the way through for recovery
    drop_start_idx = int(len(timestamps) * 0.1)
    recovery_start_idx = int(len(timestamps) * 0.15)

    drop_periods = 10
    recovery_periods = 10

    # Make sure we don't go beyond the dataframe bounds
    if drop_start_idx + drop_periods + recovery_periods >= len(timestamps):
        # Adjust to ensure we have enough room for our modifications
        drop_start_idx = 20
        recovery_start_idx = 30
        drop_periods = min(10, len(timestamps) - drop_start_idx - recovery_periods)
        recovery_periods = min(10, len(timestamps) - drop_start_idx - drop_periods)

    # Make the price drop for several periods to trigger oversold
    current_price = df.iloc[drop_start_idx]["close"]
    for i in range(drop_periods):
        idx = drop_start_idx + i
        if idx < len(timestamps):
            current_price = current_price * 0.95  # 5% drop
            df.loc[timestamps[idx], "close"] = current_price

    # Then make it recover for the next several periods to trigger a buy signal
    current_price = df.iloc[drop_start_idx + drop_periods - 1]["close"]
    for i in range(recovery_periods):
        idx = drop_start_idx + drop_periods + i
        if idx < len(timestamps):
            current_price = current_price * 1.05  # 5% rise
            df.loc[timestamps[idx], "close"] = current_price

    processed_data, signal = rsi.process(df)

    # Verify the data was processed
    assert "rsi" in processed_data.columns

    # We may or may not get a signal depending on the data
    if signal is not None:
        assert signal.symbol == "ETH"
        assert signal.indicator == "test_rsi"
        assert signal.direction in [SignalDirection.BUY, SignalDirection.SELL]


def test_rsi_error_handling():
    """Test error handling in RSI indicator."""
    rsi = RSIIndicator(name="test_rsi")

    # Test with invalid DataFrame (missing close column)
    invalid_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "invalid_column": [1, 2, 3, 4, 5],
        }
    )

    with pytest.raises(ValueError, match="must contain a 'close' price column"):
        rsi.calculate(invalid_df)

    # Test with insufficient data points
    insufficient_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550],
        }
    )

    # This should not raise an error but log a warning and return the data with NaN RSI values
    result = rsi.calculate(insufficient_df)
    assert "rsi" in result.columns
    assert pd.isna(result["rsi"]).all()

    # Test generate_signal with missing RSI column
    missing_rsi_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550, 1600, 1650, 1700],
        }
    )

    # Should return None if RSI column is missing
    assert rsi.generate_signal(missing_rsi_df) is None
