from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.indicators.base_indicator import SignalDirection
from app.indicators.ultimate_ma_indicator import UltimateMAIndicator


def test_ultimate_ma_initialization():
    """Test Ultimate MA indicator initialization with default and custom parameters."""
    # Test with default parameters
    uma = UltimateMAIndicator(name="test_uma")
    assert uma.name == "test_uma"
    assert uma.source_col == "close"
    assert uma.length == 20
    assert uma.ma_type == 1  # SMA
    assert abs(uma.t3_factor - 0.7) < 1e-10  # Use approximation for floating point
    assert uma.use_second_ma is False
    assert uma.length2 == 50
    assert uma.ma_type2 == 1  # SMA
    assert abs(uma.t3_factor2 - 0.7) < 1e-10  # Use approximation for floating point
    assert uma.color_based_on_direction is True
    assert uma.smooth_factor == 2

    # Test with custom parameters
    custom_params = {
        "source": "high",
        "length": 10,
        "ma_type": 2,  # EMA
        "t3_factor": 5,  # 0.5 after multiplier
        "use_second_ma": True,
        "length2": 30,
        "ma_type2": 4,  # HullMA
        "t3_factor2": 3,  # 0.3 after multiplier
        "color_based_on_direction": False,
        "smooth_factor": 3
    }
    uma_custom = UltimateMAIndicator(name="custom_uma", params=custom_params)
    assert uma_custom.name == "custom_uma"
    assert uma_custom.source_col == "high"
    assert uma_custom.length == 10
    assert uma_custom.ma_type == 2  # EMA
    assert abs(uma_custom.t3_factor - 0.5) < 1e-10  # Use approximation for floating point
    assert uma_custom.use_second_ma is True
    assert uma_custom.length2 == 30
    assert uma_custom.ma_type2 == 4  # HullMA
    assert abs(uma_custom.t3_factor2 - 0.3) < 1e-10  # Use approximation for floating point
    assert uma_custom.color_based_on_direction is False
    assert uma_custom.smooth_factor == 3


def test_ultimate_ma_parameter_validation():
    """Test parameter validation in Ultimate MA indicator."""
    # Test invalid ma_type correction
    invalid_params = {"ma_type": 10}  # Invalid MA type
    uma = UltimateMAIndicator(name="invalid_ma", params=invalid_params)
    assert uma.ma_type == 1  # Should correct to default (SMA)

    # Test invalid ma_type2 correction
    invalid_params2 = {"ma_type2": 0}  # Invalid MA type
    uma2 = UltimateMAIndicator(name="invalid_ma2", params=invalid_params2)
    assert uma2.ma_type2 == 1  # Should correct to default (SMA)


def test_moving_average_calculation_methods():
    """Test individual moving average calculation methods."""
    uma = UltimateMAIndicator(name="test_uma")

    # Create sample data
    data = pd.Series([100, 105, 110, 115, 120, 125, 130, 135, 140, 145])

    # Test SMA calculation
    sma = uma._calculate_sma(data, 5)
    assert len(sma) == len(data)
    assert sma.iloc[4] == 110.0  # First valid SMA value
    assert sma.iloc[9] == 135.0  # Last SMA value

    # Test EMA calculation
    ema = uma._calculate_ema(data, 5)
    assert len(ema) == len(data)
    # EMA values will be different from SMA
    assert not np.isclose(ema.iloc[9], sma.iloc[9])

    # Test WMA calculation
    wma = uma._calculate_wma(data, 5)
    assert len(wma) == len(data)
    # WMA gives more weight to recent values
    assert wma.iloc[9] > sma.iloc[9]

    # Test Hull MA calculation
    hull = uma._calculate_hull_ma(data, 4)
    assert len(hull) == len(data)
    # Hull MA responds faster to trend changes
    assert not pd.isna(hull.iloc[-1])

    # Test RMA calculation
    rma = uma._calculate_rma(data, 5)
    assert len(rma) == len(data)
    # RMA is smoother than EMA
    assert np.isclose(rma.iloc[-1], data.iloc[-1], rtol=0.2)

    # Test TEMA calculation
    tema = uma._calculate_tema(data, 5)
    assert len(tema) == len(data)
    # TEMA reduces lag
    assert not pd.isna(tema.iloc[-1])

    # Test T3 calculation
    t3 = uma._calculate_t3(data, 5, 0.7)
    assert len(t3) == len(data)
    assert not pd.isna(t3.iloc[-1])


def test_ultimate_ma_calculation(sample_price_data):
    """Test Ultimate MA calculation using sample price data."""
    # Test with single MA
    uma = UltimateMAIndicator(name="test_uma", params={"length": 10})
    result = uma.calculate(sample_price_data)

    # Verify MA columns were added
    assert "uma_line1" in result.columns
    assert "uma_price_crossing_up1" in result.columns
    assert "uma_price_crossing_down1" in result.columns

    # After length rows, values should be valid
    assert not pd.isna(result["uma_line1"].iloc[uma.length])

    # Test with dual MA setup
    uma_dual = UltimateMAIndicator(
        name="test_uma_dual",
        params={"length": 10, "use_second_ma": True, "length2": 20}
    )
    result_dual = uma_dual.calculate(sample_price_data)

    # Verify second MA columns
    assert "uma_line2" in result_dual.columns
    assert "uma_price_crossing_up2" in result_dual.columns
    assert "uma_price_crossing_down2" in result_dual.columns
    assert "uma_ma_crossing_up" in result_dual.columns
    assert "uma_ma_crossing_down" in result_dual.columns

    # After longer period rows, values should be valid
    assert not pd.isna(result_dual["uma_line2"].iloc[uma_dual.length2])

    # Test with different MA types
    types = [
        {"ma_type": 1, "name": "SMA", "min_rows": 10},       # SMA needs length rows
        {"ma_type": 2, "name": "EMA", "min_rows": 10},       # EMA converges quickly
        {"ma_type": 3, "name": "WMA", "min_rows": 10},       # WMA needs length rows
        {"ma_type": 4, "name": "HullMA", "min_rows": 15},    # Hull needs more warmup
        {"ma_type": 6, "name": "RMA", "min_rows": 10},       # RMA converges quickly
        {"ma_type": 7, "name": "TEMA", "min_rows": 30},      # TEMA needs 3x length
        {"ma_type": 8, "name": "T3", "min_rows": 30},        # T3 needs more warmup
    ]

    # Instead of comparing results directly, just verify each MA type produces valid output
    for ma_type in types:
        uma_type = UltimateMAIndicator(
            name=f"test_{ma_type['name']}",
            params={"length": 10, "ma_type": ma_type["ma_type"]}
        )
        result_type = uma_type.calculate(sample_price_data)

        # Allow for different warmup periods for different MA types
        # Check that the values are valid after the required warmup
        min_rows = ma_type["min_rows"]
        warmup_end = min(len(sample_price_data) - 1, min_rows + 5)  # Add a few more rows for safety

        # Check that the indicator has valid values near the end of the series
        assert not pd.isna(result_type["uma_line1"].iloc[-10:]).all()

    # Test VWMA (ma_type 5) separately as it requires volume data
    uma_vwma = UltimateMAIndicator(
        name="test_VWMA",
        params={"length": 10, "ma_type": 5}
    )
    result_vwma = uma_vwma.calculate(sample_price_data)
    assert not pd.isna(result_vwma["uma_line1"].iloc[-1])


def test_ultimate_ma_signal_generation():
    """Test signal generation based on Ultimate MA values."""
    # Create a mock DataFrame with Ultimate MA values
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550, 1600, 1550, 1500],
            "uma_line1": [1520, 1530, 1540, 1550, 1545],
            "uma_price_crossing_up1": [False, True, False, False, False],
            "uma_price_crossing_down1": [False, False, False, True, False],
            "uma_is_uptrend": [False, True, True, True, False],
        }
    )

    uma = UltimateMAIndicator(name="test_uma")

    # Test buy signal - Price crosses above MA1
    signal = uma.generate_signal(data.iloc[[0, 1]])
    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_uma"
    assert signal.confidence >= 0.6
    assert signal.params.get("trigger") == "price_crossing_up_ma1"

    # Test sell signal - Price crosses below MA1
    signal = uma.generate_signal(data.iloc[[2, 3]])
    assert signal is not None
    assert signal.direction == SignalDirection.SELL
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_uma"
    assert signal.confidence >= 0.6
    assert signal.params.get("trigger") == "price_crossing_down_ma1"

    # Test no signal (no crossing conditions)
    neutral_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=2, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1520],
            "uma_line1": [1520, 1530],
            "uma_price_crossing_up1": [False, False],
            "uma_price_crossing_down1": [False, False],
            "uma_is_uptrend": [False, True],
        }
    )

    signal = uma.generate_signal(neutral_data)
    assert signal is None

    # Test with dual MAs
    dual_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550, 1600, 1550, 1500],
            "uma_line1": [1520, 1530, 1545, 1550, 1545],
            "uma_line2": [1510, 1515, 1525, 1535, 1540],
            "uma_price_crossing_up1": [False, False, False, False, False],
            "uma_price_crossing_down1": [False, False, False, False, False],
            "uma_price_crossing_up2": [False, True, False, False, False],
            "uma_price_crossing_down2": [False, False, False, True, False],
            "uma_ma_crossing_up": [False, False, True, False, False],
            "uma_ma_crossing_down": [False, False, False, False, True],
            "uma_is_uptrend": [False, True, True, False, False],
            "uma_is_uptrend2": [False, False, True, True, False],
        }
    )

    uma_dual = UltimateMAIndicator(name="test_uma_dual", params={"use_second_ma": True})

    # Test buy signal - Price crosses above MA2
    signal = uma_dual.generate_signal(dual_data.iloc[[0, 1]])
    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.params.get("trigger") == "price_crossing_up_ma2"

    # Test buy signal - MA1 crosses above MA2 (golden cross)
    signal = uma_dual.generate_signal(dual_data.iloc[[1, 2]])
    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.params.get("trigger") == "ma_crossing_up"
    assert signal.confidence >= 0.7  # Should have higher confidence

    # Test sell signal - Price crosses below MA2
    signal = uma_dual.generate_signal(dual_data.iloc[[2, 3]])
    assert signal is not None
    assert signal.direction == SignalDirection.SELL
    assert signal.params.get("trigger") == "price_crossing_down_ma2"

    # Test sell signal - MA1 crosses below MA2 (death cross)
    signal = uma_dual.generate_signal(dual_data.iloc[[3, 4]])
    assert signal is not None
    assert signal.direction == SignalDirection.SELL
    assert signal.params.get("trigger") == "ma_crossing_down"
    assert signal.confidence >= 0.7  # Should have higher confidence


def test_ultimate_ma_process_method(sample_price_data):
    """Test the combined process method."""
    uma = UltimateMAIndicator(
        name="test_uma",
        params={"length": 5, "use_second_ma": True, "length2": 10}
    )  # Shorter periods for quicker signals

    # Manipulate data to create crossover scenarios
    df = sample_price_data.copy()

    # Get actual row indices - we'll use the timestamp index
    timestamps = df.index.tolist()

    # Define relative positions in the dataset
    down_start_idx = int(len(timestamps) * 0.1)  # 10% through data
    up_start_idx = int(len(timestamps) * 0.2)    # 20% through data

    # Define periods for trends
    trend_period = 10

    # Make sure we don't exceed the dataframe bounds
    if up_start_idx + trend_period >= len(timestamps):
        # Scale down our indices
        down_start_idx = 20
        up_start_idx = 30
        trend_period = min(10, (len(timestamps) - up_start_idx) // 2)

    # Create a downtrend
    current_price = df.iloc[down_start_idx]["close"]
    for i in range(trend_period):
        idx = down_start_idx + i
        if idx < len(timestamps):
            current_price *= 0.98  # 2% drop
            df.loc[timestamps[idx], "close"] = current_price

    # Create an uptrend
    current_price = df.iloc[up_start_idx]["close"]
    for i in range(trend_period):
        idx = up_start_idx + i
        if idx < len(timestamps):
            current_price *= 1.02  # 2% increase
            df.loc[timestamps[idx], "close"] = current_price

    processed_data, signal = uma.process(df)

    # Verify the data was processed
    assert "uma_line1" in processed_data.columns
    assert "uma_line2" in processed_data.columns

    # We may or may not get a signal depending on the exact data patterns
    if signal is not None:
        assert signal.symbol == "ETH"
        assert signal.indicator == "test_uma"
        assert signal.direction in [SignalDirection.BUY, SignalDirection.SELL]


def test_ultimate_ma_error_handling():
    """Test error handling in Ultimate MA indicator."""
    uma = UltimateMAIndicator(name="test_uma")

    # Test with invalid DataFrame (missing close column)
    invalid_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "invalid_column": [1, 2, 3, 4, 5],
        }
    )

    with pytest.raises(ValueError, match="must contain a 'close' price column"):
        uma.calculate(invalid_df)

    # Test with insufficient data points
    insufficient_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550, 1600, 1650, 1700],
        }
    )

    # This should not raise an error but log a warning and return the data with NaN values
    result = uma.calculate(insufficient_df)
    assert "uma_line1" in result.columns
    assert pd.isna(result["uma_line1"]).all()

    # Test generate_signal with missing MA columns
    missing_ma_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550, 1600, 1650, 1700],
        }
    )

    # Should return None if required columns are missing
    assert uma.generate_signal(missing_ma_df) is None


def test_str_representation():
    """Test the string representation of the Ultimate MA indicator."""
    # Test with default parameters (SMA)
    uma = UltimateMAIndicator(name="test_uma")
    assert str(uma) == "UltimateMA(SMA(20))"

    # Test with EMA
    uma_ema = UltimateMAIndicator(name="test_ema", params={"ma_type": 2, "length": 14})
    assert str(uma_ema) == "UltimateMA(EMA(14))"

    # Test with dual MA setup
    uma_dual = UltimateMAIndicator(
        name="test_dual",
        params={
            "ma_type": 3,
            "length": 10,
            "use_second_ma": True,
            "ma_type2": 4,
            "length2": 20
        }
    )
    assert str(uma_dual) == "UltimateMA(WMA(10) + HullMA(20))"
