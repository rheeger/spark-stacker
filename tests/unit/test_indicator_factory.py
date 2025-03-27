from unittest.mock import MagicMock, patch

import pytest

from app.indicators.adaptive_supertrend_indicator import \
    AdaptiveSupertrendIndicator
from app.indicators.adaptive_trend_finder_indicator import \
    AdaptiveTrendFinderIndicator
from app.indicators.base_indicator import BaseIndicator
from app.indicators.bollinger_bands_indicator import BollingerBandsIndicator
from app.indicators.indicator_factory import IndicatorFactory
from app.indicators.macd_indicator import MACDIndicator
from app.indicators.moving_average_indicator import MovingAverageIndicator
from app.indicators.rsi_indicator import RSIIndicator
from app.indicators.ultimate_ma_indicator import UltimateMAIndicator


def test_indicator_registry():
    """Test the indicator registry initialization."""
    # Verify all registered indicators
    assert "rsi" in IndicatorFactory._indicator_registry
    assert IndicatorFactory._indicator_registry["rsi"] == RSIIndicator

    # Verify new indicators are registered
    assert "macd" in IndicatorFactory._indicator_registry
    assert "bollinger" in IndicatorFactory._indicator_registry
    assert "ma" in IndicatorFactory._indicator_registry
    assert "adaptive_supertrend" in IndicatorFactory._indicator_registry
    assert "adaptive_trend_finder" in IndicatorFactory._indicator_registry
    assert "ultimate_ma" in IndicatorFactory._indicator_registry

    # Verify indicator classes
    assert IndicatorFactory._indicator_registry["macd"] == MACDIndicator
    assert IndicatorFactory._indicator_registry["bollinger"] == BollingerBandsIndicator
    assert IndicatorFactory._indicator_registry["ma"] == MovingAverageIndicator
    assert IndicatorFactory._indicator_registry["adaptive_supertrend"] == AdaptiveSupertrendIndicator
    assert IndicatorFactory._indicator_registry["adaptive_trend_finder"] == AdaptiveTrendFinderIndicator
    assert IndicatorFactory._indicator_registry["ultimate_ma"] == UltimateMAIndicator


def test_create_indicator():
    """Test creating an indicator instance."""
    # Test creating RSI indicator
    rsi = IndicatorFactory.create_indicator(
        name="test_rsi",
        indicator_type="rsi",
        params={"period": 10, "overbought": 75, "oversold": 25},
    )

    assert rsi is not None
    assert isinstance(rsi, RSIIndicator)
    assert rsi.name == "test_rsi"
    assert rsi.period == 10
    assert rsi.overbought == 75
    assert rsi.oversold == 25

    # Test creating Adaptive SuperTrend indicator
    ast = IndicatorFactory.create_indicator(
        name="test_ast",
        indicator_type="adaptive_supertrend",
        params={"atr_length": 14, "factor": 2.5, "training_length": 150},
    )

    assert ast is not None
    assert isinstance(ast, AdaptiveSupertrendIndicator)
    assert ast.name == "test_ast"
    assert ast.atr_length == 14
    assert ast.factor == 2.5
    assert ast.training_length == 150

    # Test with invalid indicator type
    invalid_indicator = IndicatorFactory.create_indicator(
        name="test_invalid", indicator_type="invalid_type"
    )

    assert invalid_indicator is None


def test_create_indicators_from_config():
    """Test creating multiple indicators from configuration."""
    configs = [
        {
            "name": "rsi_eth",
            "type": "rsi",
            "enabled": True,
            "parameters": {"period": 14, "overbought": 70, "oversold": 30},
        },
        {
            "name": "macd_eth",
            "type": "macd",
            "enabled": True,
            "parameters": {"fast_period": 8, "slow_period": 17, "signal_period": 9},
        },
        {
            "name": "bollinger_eth",
            "type": "bollinger",
            "enabled": True,
            "parameters": {"period": 15, "std_dev": 2.5},
        },
        {
            "name": "ma_eth",
            "type": "ma",
            "enabled": True,
            "parameters": {"fast_period": 5, "slow_period": 20, "ma_type": "ema"},
        },
        {
            "name": "adaptive_supertrend_eth",
            "type": "adaptive_supertrend",
            "enabled": True,
            "parameters": {"atr_length": 14, "factor": 2.5, "training_length": 150},
        },
        {
            "name": "adaptive_trend_finder_eth",
            "type": "adaptive_trend_finder",
            "enabled": True,
            "parameters": {"period": 20, "smooth_factor": 2.0},
        },
        {
            "name": "ultimate_ma_eth",
            "type": "ultimate_ma",
            "enabled": True,
            "parameters": {"length": 20, "ma_type": 2, "use_second_ma": True, "length2": 50},
        },
        {
            "name": "disabled_indicator",
            "type": "rsi",
            "enabled": False,
            "parameters": {},
        },
        {
            "name": "invalid_indicator",
            "type": "invalid_type",
            "enabled": True,
            "parameters": {},
        },
    ]

    indicators = IndicatorFactory.create_indicators_from_config(configs)

    # Should have seven valid indicators
    assert len(indicators) == 7

    # Check the RSI indicator
    assert "rsi_eth" in indicators
    rsi = indicators["rsi_eth"]
    assert isinstance(rsi, RSIIndicator)
    assert rsi.name == "rsi_eth"
    assert rsi.period == 14

    # Check the MACD indicator
    assert "macd_eth" in indicators
    macd = indicators["macd_eth"]
    assert isinstance(macd, MACDIndicator)
    assert macd.name == "macd_eth"
    assert macd.fast_period == 8
    assert macd.slow_period == 17
    assert macd.signal_period == 9

    # Check the Bollinger Bands indicator
    assert "bollinger_eth" in indicators
    bb = indicators["bollinger_eth"]
    assert isinstance(bb, BollingerBandsIndicator)
    assert bb.name == "bollinger_eth"
    assert bb.period == 15
    assert bb.std_dev == 2.5

    # Check the Moving Average indicator
    assert "ma_eth" in indicators
    ma = indicators["ma_eth"]
    assert isinstance(ma, MovingAverageIndicator)
    assert ma.name == "ma_eth"
    assert ma.fast_period == 5
    assert ma.slow_period == 20
    assert ma.ma_type == "ema"

    # Check the Adaptive SuperTrend indicator
    assert "adaptive_supertrend_eth" in indicators
    ast = indicators["adaptive_supertrend_eth"]
    assert isinstance(ast, AdaptiveSupertrendIndicator)
    assert ast.name == "adaptive_supertrend_eth"
    assert ast.atr_length == 14
    assert ast.factor == 2.5
    assert ast.training_length == 150

    # Check the Adaptive Trend Finder indicator
    assert "adaptive_trend_finder_eth" in indicators
    atf = indicators["adaptive_trend_finder_eth"]
    assert isinstance(atf, AdaptiveTrendFinderIndicator)
    assert atf.name == "adaptive_trend_finder_eth"
    assert atf.period == 20

    # Check the Ultimate MA indicator
    assert "ultimate_ma_eth" in indicators
    uma = indicators["ultimate_ma_eth"]
    assert isinstance(uma, UltimateMAIndicator)
    assert uma.name == "ultimate_ma_eth"
    assert uma.length == 20
    assert uma.ma_type == 2
    assert uma.use_second_ma == True
    assert uma.length2 == 50

    # Disabled and invalid indicators should be excluded
    assert "disabled_indicator" not in indicators
    assert "invalid_indicator" not in indicators


def test_register_indicator():
    """Test registering a new indicator type."""

    # Create a proper indicator class instead of a mock
    class MockIndicator(BaseIndicator):
        def calculate(self, data):
            return data

        def generate_signal(self, data):
            return None

    # Register it
    IndicatorFactory.register_indicator("mock", MockIndicator)

    # Verify it was added to the registry
    assert "mock" in IndicatorFactory._indicator_registry
    assert IndicatorFactory._indicator_registry["mock"] == MockIndicator

    # Clean up by removing the mock indicator from the registry
    del IndicatorFactory._indicator_registry["mock"]

    # Test registering an invalid class (not a subclass of BaseIndicator)
    with pytest.raises(TypeError):
        IndicatorFactory.register_indicator("invalid", object)


def test_get_available_indicators():
    """Test getting available indicator types."""
    available = IndicatorFactory.get_available_indicators()

    # Should include all our indicator types
    expected_indicators = ["rsi", "macd", "bollinger", "ma", "adaptive_supertrend", "adaptive_trend_finder", "ultimate_ma"]
    for indicator in expected_indicators:
        assert indicator in available

    # Add a temporary indicator
    IndicatorFactory._indicator_registry["temp_indicator"] = MagicMock(
        spec=BaseIndicator
    )

    # Should now include the new indicator
    available = IndicatorFactory.get_available_indicators()
    assert "temp_indicator" in available

    # Clean up
    del IndicatorFactory._indicator_registry["temp_indicator"]
