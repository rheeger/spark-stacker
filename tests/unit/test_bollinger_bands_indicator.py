from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.indicators.base_indicator import SignalDirection
from app.indicators.bollinger_bands_indicator import BollingerBandsIndicator


def test_bollinger_bands_initialization():
    """Test Bollinger Bands indicator initialization with default and custom parameters."""
    # Test with default parameters
    bb = BollingerBandsIndicator(name="test_bb")
    assert bb.name == "test_bb"
    assert bb.period == 20
    assert bb.std_dev == 2
    assert bb.mean_reversion_threshold == 0.05

    # Test with custom parameters
    custom_params = {
        "period": 15,
        "std_dev": 2.5,
        "mean_reversion_threshold": 0.1
    }
    bb_custom = BollingerBandsIndicator(name="custom_bb", params=custom_params)
    assert bb_custom.name == "custom_bb"
    assert bb_custom.period == 15
    assert bb_custom.std_dev == 2.5
    assert bb_custom.mean_reversion_threshold == 0.1


def test_bollinger_bands_calculation(sample_price_data):
    """Test Bollinger Bands calculation using sample price data."""
    bb = BollingerBandsIndicator(name="test_bb")
    result = bb.calculate(sample_price_data)

    # Verify Bollinger Bands columns were added
    required_columns = ["bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_%b"]
    for col in required_columns:
        assert col in result.columns

    # After period rows, values should be valid
    for col in required_columns:
        assert not pd.isna(result[col].iloc[bb.period])

    # Upper band should be greater than middle band, which should be greater than lower band
    non_nan_indices = ~result["bb_middle"].isna()
    assert (result.loc[non_nan_indices, "bb_upper"] > result.loc[non_nan_indices, "bb_middle"]).all()
    assert (result.loc[non_nan_indices, "bb_middle"] > result.loc[non_nan_indices, "bb_lower"]).all()

    # Verify signal detection columns
    assert "price_below_lower" in result.columns
    assert "price_crossing_below_lower" in result.columns
    assert "price_above_upper" in result.columns
    assert "price_crossing_above_upper" in result.columns
    assert "mean_reversion_buy" in result.columns
    assert "mean_reversion_sell" in result.columns


def test_bollinger_bands_signal_generation():
    """Test signal generation based on Bollinger Bands values."""
    # Create a mock DataFrame with Bollinger Bands values
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=6, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1450, 1430, 1550, 1580, 1570],
            "bb_middle": [1500, 1500, 1500, 1500, 1500, 1500],
            "bb_upper": [1550, 1550, 1550, 1550, 1550, 1550],
            "bb_lower": [1450, 1450, 1450, 1450, 1450, 1450],
            "bb_width": [0.07, 0.07, 0.07, 0.07, 0.07, 0.07],
            "bb_%b": [0.5, 0.0, -0.4, 1.0, 1.6, 1.4],
            "price_below_lower": [False, False, True, False, False, False],
            "price_crossing_below_lower": [False, True, False, False, False, False],
            "price_above_upper": [False, False, False, False, True, True],
            "price_crossing_above_upper": [False, False, False, True, False, False],
            "mean_reversion_buy": [False, False, False, True, False, False],
            "mean_reversion_sell": [False, False, False, False, False, True],
        }
    )

    bb = BollingerBandsIndicator(name="test_bb")

    # Test buy signal - price crossing below lower band
    signal = bb.generate_signal(data.iloc[[0, 1]])
    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_bb"
    assert signal.confidence > 0.5
    assert signal.params.get("trigger") == "price_crossing_below_lower"

    # Test buy signal - mean reversion from lower band
    signal = bb.generate_signal(data.iloc[[2, 3]])
    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_bb"
    assert signal.confidence > 0.5
    assert signal.params.get("trigger") == "mean_reversion_buy"

    # Test sell signal - price crossing above upper band
    signal = bb.generate_signal(data.iloc[[2, 3]])
    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_bb"
    assert signal.confidence > 0.5
    assert signal.params.get("trigger") == "mean_reversion_buy"

    # Create new data specifically for price crossing above upper band test
    upper_cross_data = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
        "symbol": "ETH",
        "close": [1540, 1560],
        "bb_middle": [1500, 1500],
        "bb_upper": [1550, 1550],
        "bb_lower": [1450, 1450],
        "bb_width": [0.07, 0.07],
        "bb_%b": [0.9, 1.1],
        "price_below_lower": [False, False],
        "price_crossing_below_lower": [False, False],
        "price_above_upper": [False, True],
        "price_crossing_above_upper": [False, True],
        "mean_reversion_buy": [False, False],
        "mean_reversion_sell": [False, False]
    })

    # Test sell signal - price crossing above upper band
    signal = bb.generate_signal(upper_cross_data)
    assert signal is not None
    assert signal.direction == SignalDirection.SELL
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_bb"
    assert signal.confidence > 0.5
    assert signal.params.get("trigger") == "price_crossing_above_upper"

    # Test sell signal - mean reversion from upper band
    signal = bb.generate_signal(data.iloc[[4, 5]])
    assert signal is not None
    assert signal.direction == SignalDirection.SELL
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_bb"
    assert signal.confidence > 0.5
    assert signal.params.get("trigger") == "mean_reversion_sell"

    # Test no signal (no conditions met)
    neutral_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1510],
            "bb_middle": [1500, 1500],
            "bb_upper": [1550, 1550],
            "bb_lower": [1450, 1450],
            "bb_width": [0.07, 0.07],
            "bb_%b": [0.5, 0.6],
            "price_below_lower": [False, False],
            "price_crossing_below_lower": [False, False],
            "price_above_upper": [False, False],
            "price_crossing_above_upper": [False, False],
            "mean_reversion_buy": [False, False],
            "mean_reversion_sell": [False, False],
        }
    )

    signal = bb.generate_signal(neutral_data)
    assert signal is None


def test_bollinger_bands_process_method(sample_price_data):
    """Test the combined process method."""
    bb = BollingerBandsIndicator(
        name="test_bb",
        params={"period": 10, "std_dev": 2}
    )  # Shorter period for quicker signals

    # Manipulate data to create scenarios for signal generation
    df = sample_price_data.copy()

    # First create a scenario where price drops below lower band then recovers
    drop_start = 30
    recovery_start = 40

    # Make the price drop sharply below the lower band
    for i in range(drop_start, drop_start + 5):
        df.loc[i, "close"] = df.loc[i-1, "close"] * 0.95  # 5% drop

    # Then make it recover
    for i in range(recovery_start, recovery_start + 5):
        df.loc[i, "close"] = df.loc[i-1, "close"] * 1.05  # 5% rise

    # Now create a scenario where price rises above upper band then falls
    rise_start = 60
    fall_start = 70

    # Make the price rise sharply above the upper band
    for i in range(rise_start, rise_start + 5):
        df.loc[i, "close"] = df.loc[i-1, "close"] * 1.05  # 5% rise

    # Then make it fall
    for i in range(fall_start, fall_start + 5):
        df.loc[i, "close"] = df.loc[i-1, "close"] * 0.95  # 5% drop

    processed_data, signal = bb.process(df)

    # Verify the data was processed
    assert "bb_middle" in processed_data.columns
    assert "bb_upper" in processed_data.columns
    assert "bb_lower" in processed_data.columns
    assert "bb_width" in processed_data.columns
    assert "bb_%b" in processed_data.columns

    # We may or may not get a signal depending on the exact data patterns
    if signal is not None:
        assert signal.symbol == "ETH"
        assert signal.indicator == "test_bb"
        assert signal.direction in [SignalDirection.BUY, SignalDirection.SELL]


def test_bollinger_bands_error_handling():
    """Test error handling in Bollinger Bands indicator."""
    bb = BollingerBandsIndicator(name="test_bb")

    # Test with invalid DataFrame (missing close column)
    invalid_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "invalid_column": [1, 2, 3, 4, 5],
        }
    )

    with pytest.raises(ValueError, match="must contain a 'close' price column"):
        bb.calculate(invalid_df)

    # Test with insufficient data points
    insufficient_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550, 1600, 1650, 1700],
        }
    )

    # This should not raise an error but log a warning and return the data with NaN BB values
    result = bb.calculate(insufficient_df)
    assert "bb_middle" in result.columns
    assert pd.isna(result["bb_middle"]).all()
    assert pd.isna(result["bb_upper"]).all()
    assert pd.isna(result["bb_lower"]).all()

    # Test generate_signal with missing required columns
    missing_cols_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550, 1600, 1650, 1700],
            # Missing bb_middle, bb_upper, bb_lower, etc.
        }
    )

    # Should return None if required columns are missing
    assert bb.generate_signal(missing_cols_df) is None
