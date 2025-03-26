import pytest
from unittest.mock import patch, MagicMock

from app.indicators.indicator_factory import IndicatorFactory
from app.indicators.base_indicator import BaseIndicator
from app.indicators.rsi_indicator import RSIIndicator


def test_indicator_registry():
    """Test the indicator registry initialization."""
    # Verify RSI indicator is registered
    assert "rsi" in IndicatorFactory._indicator_registry
    assert IndicatorFactory._indicator_registry["rsi"] == RSIIndicator


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
            "name": "rsi_btc",
            "type": "rsi",
            "enabled": True,
            "parameters": {"period": 7, "overbought": 80, "oversold": 20},
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

    # Should have two valid indicators
    assert len(indicators) == 2

    # Check the first indicator
    assert "rsi_eth" in indicators
    eth_rsi = indicators["rsi_eth"]
    assert isinstance(eth_rsi, RSIIndicator)
    assert eth_rsi.name == "rsi_eth"
    assert eth_rsi.period == 14

    # Check the second indicator
    assert "rsi_btc" in indicators
    btc_rsi = indicators["rsi_btc"]
    assert isinstance(btc_rsi, RSIIndicator)
    assert btc_rsi.name == "rsi_btc"
    assert btc_rsi.period == 7

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

    # Should include RSI at minimum
    assert "rsi" in available

    # Add a temporary indicator
    IndicatorFactory._indicator_registry["temp_indicator"] = MagicMock(
        spec=BaseIndicator
    )

    # Should now include the new indicator
    available = IndicatorFactory.get_available_indicators()
    assert "temp_indicator" in available

    # Clean up
    del IndicatorFactory._indicator_registry["temp_indicator"]
