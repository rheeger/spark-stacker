from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.indicators.adaptive_trend_finder_indicator import AdaptiveTrendFinderIndicator
from app.indicators.base_indicator import SignalDirection


def test_adaptive_trend_finder_initialization():
    """Test Adaptive Trend Finder indicator initialization with default and custom parameters."""
    # Test with default parameters
    atf = AdaptiveTrendFinderIndicator(name="test_atf")
    assert atf.name == "test_atf"
    assert atf.use_long_term == False
    assert atf.dev_multiplier == 2.0
    assert atf.source_col == "close"
    assert len(atf.periods) > 0
    assert atf.periods[0] == 20  # Short-term first period

    # Test with custom parameters
    custom_params = {"use_long_term": True, "dev_multiplier": 3.0, "source": "high"}
    atf_custom = AdaptiveTrendFinderIndicator(name="custom_atf", params=custom_params)
    assert atf_custom.name == "custom_atf"
    assert atf_custom.use_long_term == True
    assert atf_custom.dev_multiplier == 3.0
    assert atf_custom.source_col == "high"
    assert len(atf_custom.periods) > 0
    assert atf_custom.periods[0] == 300  # Long-term first period


def test_confidence_mapping():
    """Test the confidence mapping function."""
    atf = AdaptiveTrendFinderIndicator(name="test_atf")

    # Test various Pearson R values
    assert atf._calculate_confidence(0.1) == "Extremely Weak"
    assert atf._calculate_confidence(0.25) == "Very Weak"
    assert atf._calculate_confidence(0.35) == "Weak"
    assert atf._calculate_confidence(0.45) == "Mostly Weak"
    assert atf._calculate_confidence(0.55) == "Somewhat Weak"
    assert atf._calculate_confidence(0.65) == "Moderately Weak"
    assert atf._calculate_confidence(0.75) == "Moderate"
    assert atf._calculate_confidence(0.85) == "Moderately Strong"
    assert atf._calculate_confidence(0.91) == "Mostly Strong"
    assert atf._calculate_confidence(0.93) == "Strong"
    assert atf._calculate_confidence(0.95) == "Very Strong"
    assert atf._calculate_confidence(0.97) == "Exceptionally Strong"
    assert atf._calculate_confidence(0.99) == "Ultra Strong"

    # Test confidence to value mapping
    assert atf._confidence_to_value("Extremely Weak") == 0.5
    assert atf._confidence_to_value("Moderate") == 0.7
    assert atf._confidence_to_value("Ultra Strong") == 1.0


def test_log_regression_calculation():
    """Test the logarithmic regression calculation."""
    atf = AdaptiveTrendFinderIndicator(name="test_atf")

    # Create an uptrend series for testing
    prices = np.array([100, 102, 105, 107, 110, 112, 115, 118, 120, 123])
    log_prices = pd.Series(np.log(prices))

    std_dev, pearson_r, slope, intercept = atf._calc_log_regression(
        log_prices, len(log_prices)
    )

    # Verify results
    assert std_dev > 0  # Standard deviation should be positive
    assert pearson_r > 0.9  # Correlation should be high for this clear trend
    assert slope != 0  # Slope should not be zero


def test_adaptive_trend_finder_calculation(sample_price_data):
    """Test Adaptive Trend Finder calculation with sample price data."""
    atf = AdaptiveTrendFinderIndicator(name="test_atf")

    # Ensure we have enough data for the test
    if len(sample_price_data) < 50:
        # Create a larger dataset with a clear trend if needed
        index = pd.date_range("2023-01-01", periods=200, freq="1h")
        data = pd.DataFrame(
            {
                "timestamp": index,
                "symbol": "ETH",
                "open": np.linspace(1000, 2000, 200) + np.random.normal(0, 50, 200),
                "high": np.linspace(1020, 2020, 200) + np.random.normal(0, 50, 200),
                "low": np.linspace(980, 1980, 200) + np.random.normal(0, 50, 200),
                "close": np.linspace(1000, 2000, 200) + np.random.normal(0, 30, 200),
                "volume": np.random.normal(1000, 200, 200),
            }
        )
        # Make sure high is actually highest and low is lowest
        for i in range(len(data)):
            data.loc[i, "high"] = max(
                data.loc[i, "open"], data.loc[i, "close"], data.loc[i, "high"]
            )
            data.loc[i, "low"] = min(
                data.loc[i, "open"], data.loc[i, "close"], data.loc[i, "low"]
            )
    else:
        data = sample_price_data.copy()

    # Run the calculation
    result = atf.calculate(data)

    # Verify ATF columns were added
    assert "atf_period" in result.columns
    assert "atf_pearson_r" in result.columns
    assert "atf_confidence" in result.columns
    assert "atf_slope" in result.columns
    assert "atf_midline" in result.columns
    assert "atf_upper" in result.columns
    assert "atf_lower" in result.columns
    assert "atf_channel_position" in result.columns

    # Verify basic calculations
    assert not pd.isna(result["atf_pearson_r"].iloc[-1])
    assert not pd.isna(result["atf_midline"].iloc[-1])

    # Channel position can be outside 0-100 range when price is outside bands
    valid_positions = result["atf_channel_position"].dropna()
    assert len(valid_positions) > 0


def test_adaptive_trend_finder_signal_generation():
    """Test signal generation based on Adaptive Trend Finder values."""
    atf = AdaptiveTrendFinderIndicator(name="test_atf")

    # Create mock data with a buy signal scenario (price returning from lower band)
    buy_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1450, 1400, 1380, 1420],  # Price bouncing from lower band
            "atf_period": [50, 50, 50, 50, 50],
            "atf_pearson_r": [0.92, 0.92, 0.92, 0.92, 0.92],
            "atf_confidence": [
                "Mostly Strong",
                "Mostly Strong",
                "Mostly Strong",
                "Mostly Strong",
                "Mostly Strong",
            ],
            "atf_slope": [
                -0.01,
                -0.01,
                -0.01,
                -0.01,
                -0.01,
            ],  # Negative slope = uptrend in log regression
            "atf_midline": [1500, 1500, 1500, 1500, 1500],
            "atf_upper": [1600, 1600, 1600, 1600, 1600],
            "atf_lower": [1400, 1400, 1400, 1400, 1400],
            "atf_returning_from_lower": [False, False, False, False, True],
            "atf_returning_from_upper": [False, False, False, False, False],
            "atf_channel_position": [50, 25, 0, -10, 10],
        }
    )

    # Generate buy signal
    signal = atf.generate_signal(buy_data)
    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_atf"
    assert signal.confidence == 0.8  # Confidence for "Mostly Strong"
    assert signal.params["trigger"] == "returning_from_lower"

    # Create mock data with a sell signal scenario (price returning from upper band)
    sell_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1550, 1600, 1620, 1580],  # Price bouncing from upper band
            "atf_period": [50, 50, 50, 50, 50],
            "atf_pearson_r": [0.95, 0.95, 0.95, 0.95, 0.95],
            "atf_confidence": [
                "Very Strong",
                "Very Strong",
                "Very Strong",
                "Very Strong",
                "Very Strong",
            ],
            "atf_slope": [
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
            ],  # Positive slope = downtrend in log regression
            "atf_midline": [1500, 1500, 1500, 1500, 1500],
            "atf_upper": [1600, 1600, 1600, 1600, 1600],
            "atf_lower": [1400, 1400, 1400, 1400, 1400],
            "atf_returning_from_lower": [False, False, False, False, False],
            "atf_returning_from_upper": [False, False, False, False, True],
            "atf_channel_position": [50, 75, 100, 110, 90],
        }
    )

    # Generate sell signal
    signal = atf.generate_signal(sell_data)
    assert signal is not None
    assert signal.direction == SignalDirection.SELL
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_atf"
    assert signal.confidence == 0.9  # Confidence for "Very Strong"
    assert signal.params["trigger"] == "returning_from_upper"

    # Test with no signal conditions
    neutral_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=3, freq="1h"),
            "symbol": "ETH",
            "close": [1500, 1510, 1520],
            "atf_period": [50, 50, 50],
            "atf_pearson_r": [0.92, 0.92, 0.92],
            "atf_confidence": ["Mostly Strong", "Mostly Strong", "Mostly Strong"],
            "atf_slope": [-0.01, -0.01, -0.01],
            "atf_midline": [1500, 1500, 1500],
            "atf_upper": [1600, 1600, 1600],
            "atf_lower": [1400, 1400, 1400],
            "atf_returning_from_lower": [False, False, False],
            "atf_returning_from_upper": [False, False, False],
            "atf_channel_position": [50, 55, 60],
        }
    )

    signal = atf.generate_signal(neutral_data)
    assert signal is None


def test_adaptive_trend_finder_insufficient_data():
    """Test error handling with insufficient data."""
    atf = AdaptiveTrendFinderIndicator(name="test_atf")

    # Test with insufficient data points
    insufficient_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="1h"),
            "symbol": "ETH",
            "close": np.linspace(1000, 1100, 10),
        }
    )

    # Should still process but return empty indicator values
    result = atf.calculate(insufficient_data)
    assert "atf_period" in result.columns
    assert pd.isna(result["atf_period"].iloc[0])

    # Test with missing required column
    invalid_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=50, freq="1h"),
            "symbol": "ETH",
            # Missing 'close' column
        }
    )

    with pytest.raises(ValueError, match="must contain a 'close' price column"):
        atf.calculate(invalid_data)
