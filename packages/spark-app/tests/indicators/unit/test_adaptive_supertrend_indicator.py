from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.indicators.adaptive_supertrend_indicator import AdaptiveSupertrendIndicator
from app.indicators.base_indicator import SignalDirection


def test_adaptive_supertrend_initialization():
    """Test Adaptive SuperTrend indicator initialization with default and custom parameters."""
    # Test with default parameters
    ast = AdaptiveSupertrendIndicator(name="test_ast")
    assert ast.name == "test_ast"
    assert ast.atr_length == 10
    assert ast.factor == 3.0
    assert ast.training_length == 100
    assert ast.high_vol_percentile == 0.75
    assert ast.medium_vol_percentile == 0.50
    assert ast.low_vol_percentile == 0.25
    assert ast.max_iterations == 10

    # Test with custom parameters
    custom_params = {
        "atr_length": 14,
        "factor": 2.5,
        "training_length": 200,
        "high_vol_percentile": 0.8,
        "medium_vol_percentile": 0.6,
        "low_vol_percentile": 0.3,
        "max_iterations": 15,
    }
    ast_custom = AdaptiveSupertrendIndicator(name="custom_ast", params=custom_params)
    assert ast_custom.name == "custom_ast"
    assert ast_custom.atr_length == 14
    assert ast_custom.factor == 2.5
    assert ast_custom.training_length == 200
    assert ast_custom.high_vol_percentile == 0.8
    assert ast_custom.medium_vol_percentile == 0.6
    assert ast_custom.low_vol_percentile == 0.3
    assert ast_custom.max_iterations == 15


def test_atr_calculation():
    """Test the ATR calculation function."""
    ast = AdaptiveSupertrendIndicator(name="test_ast")

    # Create test data with a known pattern
    data = pd.DataFrame(
        {
            "high": [110, 120, 130, 140, 150],
            "low": [90, 80, 70, 60, 50],
            "close": [100, 100, 100, 100, 100],
        }
    )

    atr = ast._calculate_atr(data, 3)

    # Validate calculations
    assert len(atr) == 5
    assert pd.isna(atr.iloc[0])  # First values should be NaN
    assert pd.isna(atr.iloc[1])  # First values should be NaN

    # Only check the last value for simplicity
    # True range for last 3 periods before the current period would be:
    # TR[2] = max(high[2]-low[2], abs(high[2]-close[1]), abs(low[2]-close[1])) = max(130-70, |130-100|, |70-100|) = max(60, 30, 30) = 60
    # TR[3] = max(high[3]-low[3], abs(high[3]-close[2]), abs(low[3]-close[2])) = max(140-60, |140-100|, |60-100|) = max(80, 40, 40) = 80
    # TR[4] = max(high[4]-low[4], abs(high[4]-close[3]), abs(low[4]-close[3])) = max(150-50, |150-100|, |50-100|) = max(100, 50, 50) = 100
    # ATR[4] = (60 + 80 + 100) / 3 = 80
    assert abs(atr.iloc[4] - 80.0) < 0.01  # Allow for small floating point differences


def test_kmeans_clustering():
    """Test the K-means clustering implementation."""
    ast = AdaptiveSupertrendIndicator(name="test_ast", params={"max_iterations": 5})

    # Create synthetic volatility data with clear clusters
    volatility = pd.Series(
        [
            # Low volatility points
            1.2,
            1.3,
            1.1,
            1.4,
            1.5,
            1.2,
            1.3,
            1.0,
            1.4,
            1.3,
            # Medium volatility points
            3.7,
            3.8,
            3.6,
            3.9,
            4.0,
            3.7,
            3.8,
            3.5,
            3.9,
            3.8,
            # High volatility points
            7.5,
            7.6,
            7.4,
            7.7,
            7.8,
            7.5,
            7.6,
            7.3,
            7.7,
            7.6,
            # End with a high volatility point for the current value
            7.7,
        ]
    )

    (
        high_centroid,
        medium_centroid,
        low_centroid,
        current_cluster,
    ) = ast._kmeans_clustering(volatility, 30)

    # Verify centroids are in the correct ranges
    assert 7.0 < high_centroid < 8.0  # Should be around 7.5-7.6
    assert 3.5 < medium_centroid < 4.5  # Should be around 3.7-3.8
    assert 1.0 < low_centroid < 2.0  # Should be around 1.2-1.3

    # Current value is in the high volatility cluster
    assert current_cluster == 0


def test_supertrend_calculation():
    """Test the SuperTrend calculation."""
    ast = AdaptiveSupertrendIndicator(name="test_ast")

    # Create test data for a basic trend
    data = pd.DataFrame(
        {
            "high": [110, 120, 130, 125, 115, 105, 95, 90, 100, 110],
            "low": [90, 100, 110, 105, 95, 85, 75, 70, 80, 90],
            "close": [100, 110, 120, 115, 105, 95, 85, 80, 90, 100],
        }
    )

    # Use a constant ATR for simplicity
    atr = pd.Series([10] * len(data))

    supertrend, direction = ast._calculate_supertrend(data, 3.0, atr)

    # Check that the direction vector has reasonable changes
    # In this test case, we should see at least one direction change
    assert not all(d == direction.iloc[0] for d in direction)

    # Verify some key relationships
    for i in range(1, len(data)):
        # If direction is -1 (bullish), supertrend should be below close
        # If direction is 1 (bearish), supertrend should be above close
        if direction.iloc[i] == -1:
            assert supertrend.iloc[i] <= data["close"].iloc[i]
        elif direction.iloc[i] == 1:
            assert supertrend.iloc[i] >= data["close"].iloc[i]


def test_adaptive_supertrend_calculation(sample_price_data):
    """Test the Adaptive SuperTrend calculation with sample price data."""
    ast = AdaptiveSupertrendIndicator(name="test_ast", params={"training_length": 50})

    # Ensure we have enough data for testing
    if len(sample_price_data) < 150:
        # Create synthetic data with trend and volatility changes
        index = pd.date_range("2023-01-01", periods=200, freq="1h")
        close_prices = np.linspace(1000, 1500, 200)

        # Add volatility regimes
        volatility = np.ones(200) * 10  # Base volatility

        # High volatility in the beginning
        volatility[:50] *= 3

        # Medium volatility in the middle
        volatility[50:100] *= 2

        # Low volatility at the end
        # (no change needed)

        # Create price data with the volatility pattern
        noise = np.random.normal(0, volatility, 200)
        close_with_noise = close_prices + noise

        data = pd.DataFrame(
            {
                "timestamp": index,
                "symbol": "ETH",
                "open": close_with_noise - np.random.normal(0, 5, 200),
                "high": close_with_noise + np.random.normal(10, 5, 200),
                "low": close_with_noise - np.random.normal(10, 5, 200),
                "close": close_with_noise,
                "volume": np.random.normal(1000, 200, 200),
            }
        )

        # Ensure high/low constraints
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
    result = ast.calculate(data)

    # Verify columns were added
    assert "ast_atr" in result.columns
    assert "ast_high_centroid" in result.columns
    assert "ast_medium_centroid" in result.columns
    assert "ast_low_centroid" in result.columns
    assert "ast_volatility_cluster" in result.columns
    assert "ast_supertrend" in result.columns
    assert "ast_direction" in result.columns
    assert "ast_trend_change_up" in result.columns
    assert "ast_trend_change_down" in result.columns

    # Check that the values make sense
    assert not pd.isna(result["ast_supertrend"].iloc[-1])
    assert result["ast_direction"].iloc[-1] in [-1, 1]
    assert result["ast_volatility_cluster"].iloc[-1] in [0, 1, 2, -1]

    # Verify that there is at least one trend change in the data
    assert (result["ast_trend_change_up"].sum() > 0) or (
        result["ast_trend_change_down"].sum() > 0
    )


def test_adaptive_supertrend_signal_generation():
    """Test signal generation based on Adaptive SuperTrend values."""
    ast = AdaptiveSupertrendIndicator(name="test_ast")

    # Create test data for a buy signal (trend changing from bearish to bullish)
    buy_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "open": [1500, 1450, 1400, 1380, 1420],
            "high": [1520, 1470, 1420, 1400, 1440],
            "low": [1480, 1430, 1380, 1360, 1400],
            "close": [1500, 1450, 1400, 1380, 1420],
            "ast_atr": [20, 20, 20, 20, 20],
            "ast_high_centroid": [30, 30, 30, 30, 30],
            "ast_medium_centroid": [20, 20, 20, 20, 20],
            "ast_low_centroid": [10, 10, 10, 10, 10],
            "ast_volatility_cluster": [1, 1, 1, 1, 1],  # Medium volatility
            "ast_supertrend": [1520, 1470, 1420, 1400, 1380],
            "ast_direction": [1, 1, 1, 1, -1],  # Last bar changes to bullish
            "ast_trend_change_up": [False, False, False, False, True],  # Bullish signal
            "ast_trend_change_down": [False, False, False, False, False],
            "ast_is_bullish": [False, False, False, False, True],
            "ast_is_bearish": [True, True, True, True, False],
        }
    )

    # Generate buy signal
    signal = ast.generate_signal(buy_data)
    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_ast"
    assert 0.65 <= signal.confidence <= 0.75  # Medium volatility confidence
    assert signal.params["trigger"] == "trend_change_bullish"
    assert signal.params["volatility_regime"] == "Medium"

    # Create test data for a sell signal (trend changing from bullish to bearish)
    sell_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1h"),
            "symbol": "ETH",
            "open": [1500, 1520, 1540, 1560, 1520],
            "high": [1520, 1540, 1560, 1580, 1540],
            "low": [1480, 1500, 1520, 1540, 1500],
            "close": [1500, 1520, 1540, 1560, 1520],
            "ast_atr": [15, 15, 15, 15, 15],
            "ast_high_centroid": [30, 30, 30, 30, 30],
            "ast_medium_centroid": [20, 20, 20, 20, 20],
            "ast_low_centroid": [10, 10, 10, 10, 10],
            "ast_volatility_cluster": [2, 2, 2, 2, 2],  # Low volatility
            "ast_supertrend": [1480, 1500, 1520, 1540, 1560],
            "ast_direction": [-1, -1, -1, -1, 1],  # Last bar changes to bearish
            "ast_trend_change_up": [False, False, False, False, False],
            "ast_trend_change_down": [
                False,
                False,
                False,
                False,
                True,
            ],  # Bearish signal
            "ast_is_bullish": [True, True, True, True, False],
            "ast_is_bearish": [False, False, False, False, True],
        }
    )

    # Generate sell signal
    signal = ast.generate_signal(sell_data)
    assert signal is not None
    assert signal.direction == SignalDirection.SELL
    assert signal.symbol == "ETH"
    assert signal.indicator == "test_ast"
    assert 0.75 <= signal.confidence <= 0.85  # Low volatility confidence is higher
    assert signal.params["trigger"] == "trend_change_bearish"
    assert signal.params["volatility_regime"] == "Low"

    # Test with no signal conditions
    neutral_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=3, freq="1h"),
            "symbol": "ETH",
            "open": [1500, 1520, 1540],
            "high": [1520, 1540, 1560],
            "low": [1480, 1500, 1520],
            "close": [1500, 1520, 1540],
            "ast_atr": [15, 15, 15],
            "ast_high_centroid": [30, 30, 30],
            "ast_medium_centroid": [20, 20, 20],
            "ast_low_centroid": [10, 10, 10],
            "ast_volatility_cluster": [1, 1, 1],
            "ast_supertrend": [1480, 1500, 1520],
            "ast_direction": [-1, -1, -1],  # No change in direction
            "ast_trend_change_up": [False, False, False],
            "ast_trend_change_down": [False, False, False],
            "ast_is_bullish": [True, True, True],
            "ast_is_bearish": [False, False, False],
        }
    )

    signal = ast.generate_signal(neutral_data)
    assert signal is None


def test_adaptive_supertrend_error_handling():
    """Test error handling for Adaptive SuperTrend indicator."""
    ast = AdaptiveSupertrendIndicator(name="test_ast")

    # Test with insufficient data
    insufficient_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=20, freq="1h"),
            "symbol": "ETH",
            "open": np.linspace(1000, 1100, 20),
            "high": np.linspace(1020, 1120, 20),
            "low": np.linspace(980, 1080, 20),
            "close": np.linspace(1000, 1100, 20),
        }
    )

    # Should process but return empty columns due to insufficient data
    result = ast.calculate(insufficient_data)
    assert "ast_supertrend" in result.columns
    assert pd.isna(result["ast_supertrend"].iloc[0])

    # Test with missing required columns
    invalid_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=150, freq="1h"),
            "symbol": "ETH",
            "close": np.linspace(1000, 1500, 150),
            # Missing 'open', 'high', 'low' columns
        }
    )

    with pytest.raises(ValueError):
        ast.calculate(invalid_data)
