from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from app.indicators.base_indicator import SignalDirection
from app.indicators.moving_average_indicator import MovingAverageIndicator


def test_moving_average_initialization():
    """Test Moving Average indicator initialization with default and custom parameters."""
    # Test with default parameters
    ma = MovingAverageIndicator(name="test_ma")
    assert ma.name == "test_ma"
    assert ma.fast_period == 10
    assert ma.slow_period == 30
    assert ma.ma_type == "sma"
    assert ma.signal_threshold == 0.001

    # Test with custom parameters
    custom_params = {
        "fast_period": 5,
        "slow_period": 20,
        "ma_type": "ema",
        "signal_threshold": 0.002,
    }
    ma_custom = MovingAverageIndicator(name="custom_ma", params=custom_params)
    assert ma_custom.name == "custom_ma"
    assert ma_custom.fast_period == 5
    assert ma_custom.slow_period == 20
    assert ma_custom.ma_type == "ema"
    assert ma_custom.signal_threshold == 0.002


def test_moving_average_parameter_validation():
    """Test parameter validation in Moving Average indicator."""
    # Test fast_period >= slow_period correction
    invalid_params = {"fast_period": 30, "slow_period": 20}
    ma = MovingAverageIndicator(name="invalid_ma", params=invalid_params)
    assert ma.fast_period < ma.slow_period  # Should be corrected automatically

    # Test invalid ma_type correction
    invalid_type_params = {"ma_type": "invalid"}
    ma_type = MovingAverageIndicator(name="invalid_type", params=invalid_type_params)
    assert ma_type.ma_type == "sma"  # Should default to "sma"


def test_moving_average_calculation(sample_price_data):
    """Test Moving Average calculation using sample price data."""
    ma = MovingAverageIndicator(name="test_ma")
    result = ma.calculate(sample_price_data)

    # Verify MA columns were added
    required_columns = ["fast_ma", "slow_ma", "ma_diff", "ma_ratio"]
    for col in required_columns:
        assert col in result.columns

    # After slow_period rows, values should be valid
    for col in required_columns:
        assert not pd.isna(result[col].iloc[ma.slow_period])

    # Verify crossover detection columns
    assert "ma_crosses_above" in result.columns
    assert "ma_crosses_below" in result.columns
    assert "price_above_slow_ma" in result.columns
    assert "price_below_slow_ma" in result.columns
    assert "price_crosses_above_slow_ma" in result.columns
    assert "price_crosses_below_slow_ma" in result.columns

    # Test both SMA and EMA types
    ma_ema = MovingAverageIndicator(name="test_ema", params={"ma_type": "ema"})
    result_ema = ma_ema.calculate(sample_price_data)

    # Both should have the same columns
    for col in required_columns:
        assert col in result_ema.columns


def test_moving_average_signal_generation():
    """Test signal generation based on Moving Average values."""
    # Create a mock DataFrame with Moving Average values
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=6, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550, 1600, 1550, 1500, 1450],
            "fast_ma": [1480, 1510, 1540, 1545, 1530, 1510],
            "slow_ma": [1500, 1505, 1510, 1520, 1525, 1520],
            "ma_diff": [-20, 5, 30, 25, 5, -10],
            "ma_ratio": [0.987, 1.001, 1.02, 1.016, 1.003, 0.993],
            "ma_crosses_above": [False, True, False, False, False, False],
            "ma_crosses_below": [False, False, False, False, False, True],
            "price_above_slow_ma": [False, True, True, True, False, False],
            "price_below_slow_ma": [True, False, False, False, True, True],
            "price_crosses_above_slow_ma": [False, True, False, False, False, False],
            "price_crosses_below_slow_ma": [False, False, False, False, True, False],
        }
    )

    ma = MovingAverageIndicator(name="test_ma")

    # Test buy signal - Golden Cross (fast MA crosses above slow MA)
    signal = ma.generate_signal(data.iloc[[0, 1]])
    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_ma"
    assert signal.confidence > 0.5
    assert signal.params.get("trigger") == "golden_cross"

    # Test buy signal - Price crosses above slow MA
    signal = ma.generate_signal(
        pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
                "symbol": "ETH",
                "close": [1490, 1530],
                "fast_ma": [1500, 1505],
                "slow_ma": [1500, 1510],
                "ma_diff": [0, -5],
                "ma_ratio": [1.0, 0.995],
                "ma_crosses_above": [False, False],
                "ma_crosses_below": [False, False],
                "price_above_slow_ma": [False, True],
                "price_below_slow_ma": [True, False],
                "price_crosses_above_slow_ma": [False, True],
                "price_crosses_below_slow_ma": [False, False],
            }
        )
    )
    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_ma"
    assert signal.confidence > 0.5
    assert signal.params.get("trigger") == "price_crosses_above_ma"

    # Test sell signal - Death Cross (fast MA crosses below slow MA)
    signal = ma.generate_signal(data.iloc[[4, 5]])
    assert signal is not None
    assert signal.direction == SignalDirection.SELL
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_ma"
    assert signal.confidence > 0.5
    assert signal.params.get("trigger") == "death_cross"

    # Test sell signal - Price crosses below slow MA
    signal = ma.generate_signal(data.iloc[[3, 4]])
    assert signal is not None
    assert signal.direction == SignalDirection.SELL
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_ma"
    assert signal.confidence > 0.5
    assert signal.params.get("trigger") == "price_crosses_below_ma"

    # Test no signal (no crossover conditions)
    neutral_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550],
            "fast_ma": [1480, 1490],
            "slow_ma": [1450, 1460],
            "ma_diff": [30, 30],
            "ma_ratio": [1.02, 1.02],
            "ma_crosses_above": [False, False],
            "ma_crosses_below": [False, False],
            "price_above_slow_ma": [True, True],
            "price_below_slow_ma": [False, False],
            "price_crosses_above_slow_ma": [False, False],
            "price_crosses_below_slow_ma": [False, False],
        }
    )

    signal = ma.generate_signal(neutral_data)
    assert signal is None


def test_moving_average_process_method(sample_price_data):
    """Test the combined process method."""
    # Create indicator with shorter periods for testing
    ma = MovingAverageIndicator(
        name="test_ma", params={"fast_period": 5, "slow_period": 15, "ma_type": "ema"}
    )

    # Use a simple approach - create a copy of the data to work with
    df = sample_price_data.copy()

    # First, process the data to get moving averages
    processed_data, _ = ma.process(df)

    # Verify the data was processed correctly
    assert "fast_ma" in processed_data.columns
    assert "slow_ma" in processed_data.columns
    assert "ma_diff" in processed_data.columns
    assert "ma_ratio" in processed_data.columns

    # Create a mock DataFrame with clear Golden Cross signal conditions
    mock_golden_cross = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
            "symbol": "ETH-USD",
            "close": [1500, 1550],
            "fast_ma": [1490, 1530],
            "slow_ma": [1500, 1510],
            "ma_diff": [-10, 20],
            "ma_ratio": [0.993, 1.02],
            "ma_crosses_above": [False, True],
            "ma_crosses_below": [False, False],
            "price_above_slow_ma": [False, True],
            "price_below_slow_ma": [True, False],
            "price_crosses_above_slow_ma": [False, False],
            "price_crosses_below_slow_ma": [False, False],
        }
    )

    # Test golden cross buy signal
    golden_cross_signal = ma.generate_signal(mock_golden_cross)
    assert golden_cross_signal is not None
    assert golden_cross_signal.direction == SignalDirection.BUY
    assert golden_cross_signal.symbol == "ETH-USD"
    assert golden_cross_signal.indicator == "test_ma"
    assert golden_cross_signal.params.get("trigger") == "golden_cross"

    # Create a mock DataFrame with clear Death Cross signal conditions
    mock_death_cross = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
            "symbol": "ETH-USD",
            "close": [1500, 1460],
            "fast_ma": [1510, 1490],
            "slow_ma": [1500, 1500],
            "ma_diff": [10, -10],
            "ma_ratio": [1.007, 0.993],
            "ma_crosses_above": [False, False],
            "ma_crosses_below": [False, True],
            "price_above_slow_ma": [False, False],
            "price_below_slow_ma": [False, True],
            "price_crosses_above_slow_ma": [False, False],
            "price_crosses_below_slow_ma": [False, False],
        }
    )

    # Test death cross sell signal
    death_cross_signal = ma.generate_signal(mock_death_cross)
    assert death_cross_signal is not None
    assert death_cross_signal.direction == SignalDirection.SELL
    assert death_cross_signal.symbol == "ETH-USD"
    assert death_cross_signal.indicator == "test_ma"
    assert death_cross_signal.params.get("trigger") == "death_cross"

    # Create a mock DataFrame with price crossing above MA
    mock_price_above = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
            "symbol": "ETH-USD",
            "close": [1490, 1520],
            "fast_ma": [1510, 1515],
            "slow_ma": [1500, 1505],
            "ma_diff": [10, 10],
            "ma_ratio": [1.007, 1.01],
            "ma_crosses_above": [False, False],
            "ma_crosses_below": [False, False],
            "price_above_slow_ma": [False, True],
            "price_below_slow_ma": [True, False],
            "price_crosses_above_slow_ma": [False, True],
            "price_crosses_below_slow_ma": [False, False],
        }
    )

    # Test price crosses above MA buy signal
    price_above_signal = ma.generate_signal(mock_price_above)
    assert price_above_signal is not None
    assert price_above_signal.direction == SignalDirection.BUY
    assert price_above_signal.symbol == "ETH-USD"
    assert price_above_signal.indicator == "test_ma"
    assert price_above_signal.params.get("trigger") == "price_crosses_above_ma"

    # Create a mock DataFrame with price crossing below MA
    mock_price_below = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
            "symbol": "ETH-USD",
            "close": [1520, 1490],
            "fast_ma": [1510, 1515],
            "slow_ma": [1500, 1505],
            "ma_diff": [10, 10],
            "ma_ratio": [1.007, 1.01],
            "ma_crosses_above": [False, False],
            "ma_crosses_below": [False, False],
            "price_above_slow_ma": [True, False],
            "price_below_slow_ma": [False, True],
            "price_crosses_above_slow_ma": [False, False],
            "price_crosses_below_slow_ma": [False, True],
        }
    )

    # Test price crosses below MA sell signal
    price_below_signal = ma.generate_signal(mock_price_below)
    assert price_below_signal is not None
    assert price_below_signal.direction == SignalDirection.SELL
    assert price_below_signal.symbol == "ETH-USD"
    assert price_below_signal.indicator == "test_ma"
    assert price_below_signal.params.get("trigger") == "price_crosses_below_ma"

    # Create a mock DataFrame with no signal conditions
    mock_no_signal = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
            "symbol": "ETH-USD",
            "close": [1520, 1525],
            "fast_ma": [1510, 1515],
            "slow_ma": [1500, 1505],
            "ma_diff": [10, 10],
            "ma_ratio": [1.007, 1.01],
            "ma_crosses_above": [False, False],
            "ma_crosses_below": [False, False],
            "price_above_slow_ma": [True, True],
            "price_below_slow_ma": [False, False],
            "price_crosses_above_slow_ma": [False, False],
            "price_crosses_below_slow_ma": [False, False],
        }
    )

    # Test no signal with neutral conditions
    no_signal = ma.generate_signal(mock_no_signal)
    assert no_signal is None


def test_moving_average_error_handling():
    """Test error handling in Moving Average indicator."""
    ma = MovingAverageIndicator(name="test_ma")

    # Test with invalid DataFrame (missing close column)
    invalid_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "invalid_column": [1, 2, 3, 4, 5],
        }
    )

    with pytest.raises(ValueError, match="must contain a 'close' price column"):
        ma.calculate(invalid_df)

    # Test with insufficient data points
    insufficient_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550, 1600, 1650, 1700],
        }
    )

    # This should not raise an error but log a warning and return the data with NaN values
    result = ma.calculate(insufficient_df)
    assert "fast_ma" in result.columns
    assert pd.isna(result["fast_ma"]).all()
    assert pd.isna(result["slow_ma"]).all()
    assert pd.isna(result["ma_diff"]).all()

    # Test generate_signal with missing MA columns
    missing_ma_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550, 1600, 1650, 1700],
        }
    )

    # Should return None if required columns are missing
    assert ma.generate_signal(missing_ma_df) is None
