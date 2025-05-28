"""
Unit tests for BaseIndicator and Signal class enhancements.

Tests the new strategy context fields and backward compatibility.
"""

import time
from typing import Optional
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from app.indicators.base_indicator import (BaseIndicator, Signal,
                                           SignalDirection)


class TestSignal:
    """Test cases for the enhanced Signal class."""

    def test_signal_creation_with_strategy_context(self):
        """Test Signal creation with all strategy context fields."""
        timestamp = int(time.time() * 1000)

        signal = Signal(
            direction=SignalDirection.BUY,
            symbol="ETH",
            indicator="RSI",
            confidence=0.8,
            timestamp=timestamp,
            params={"rsi_value": 25.5},
            strategy_name="eth_multi_timeframe_strategy",
            market="ETH-USD",
            exchange="hyperliquid",
            timeframe="4h"
        )

        # Test all basic fields
        assert signal.direction == SignalDirection.BUY
        assert signal.symbol == "ETH"
        assert signal.indicator == "RSI"
        assert signal.confidence == 0.8
        assert signal.timestamp == timestamp
        assert signal.params == {"rsi_value": 25.5}

        # Test strategy context fields
        assert signal.strategy_name == "eth_multi_timeframe_strategy"
        assert signal.market == "ETH-USD"
        assert signal.exchange == "hyperliquid"
        assert signal.timeframe == "4h"

    def test_signal_creation_without_strategy_context(self):
        """Test Signal creation with minimal fields (backward compatibility)."""
        signal = Signal(
            direction=SignalDirection.SELL,
            symbol="BTC",
            indicator="MACD"
        )

        # Test basic fields
        assert signal.direction == SignalDirection.SELL
        assert signal.symbol == "BTC"
        assert signal.indicator == "MACD"
        assert signal.confidence == 0.5  # Default
        assert isinstance(signal.timestamp, int)
        assert signal.params == {}

        # Test strategy context fields are None
        assert signal.strategy_name is None
        assert signal.market is None
        assert signal.exchange is None
        assert signal.timeframe is None

    def test_signal_confidence_clamping(self):
        """Test that confidence values are clamped between 0 and 1."""
        # Test confidence > 1
        signal1 = Signal(SignalDirection.BUY, "ETH", "RSI", confidence=1.5)
        assert signal1.confidence == 1.0

        # Test confidence < 0
        signal2 = Signal(SignalDirection.SELL, "BTC", "MACD", confidence=-0.3)
        assert signal2.confidence == 0.0

        # Test valid confidence
        signal3 = Signal(SignalDirection.NEUTRAL, "DOT", "BB", confidence=0.7)
        assert signal3.confidence == 0.7

    def test_signal_to_dict_with_strategy_context(self):
        """Test Signal.to_dict() includes strategy context when present."""
        signal = Signal(
            direction=SignalDirection.BUY,
            symbol="ETH",
            indicator="RSI",
            confidence=0.8,
            timestamp=1234567890,
            params={"test": "value"},
            strategy_name="test_strategy",
            market="ETH-USD",
            exchange="hyperliquid",
            timeframe="1h"
        )

        result = signal.to_dict()

        # Test basic fields
        assert result["direction"] == SignalDirection.BUY
        assert result["symbol"] == "ETH"
        assert result["indicator"] == "RSI"
        assert result["confidence"] == 0.8
        assert result["timestamp"] == 1234567890
        assert result["params"] == {"test": "value"}

        # Test strategy context fields
        assert result["strategy_name"] == "test_strategy"
        assert result["market"] == "ETH-USD"
        assert result["exchange"] == "hyperliquid"
        assert result["timeframe"] == "1h"

    def test_signal_to_dict_without_strategy_context(self):
        """Test Signal.to_dict() excludes None strategy context fields."""
        signal = Signal(
            direction=SignalDirection.SELL,
            symbol="BTC",
            indicator="MACD",
            confidence=0.6,
            timestamp=1234567890,
            params={"test": "value"}
        )

        result = signal.to_dict()

        # Test basic fields are present
        assert result["direction"] == SignalDirection.SELL
        assert result["symbol"] == "BTC"
        assert result["indicator"] == "MACD"
        assert result["confidence"] == 0.6
        assert result["timestamp"] == 1234567890
        assert result["params"] == {"test": "value"}

        # Test strategy context fields are not in dict when None
        assert "strategy_name" not in result
        assert "market" not in result
        assert "exchange" not in result
        assert "timeframe" not in result

    def test_signal_from_dict_with_strategy_context(self):
        """Test Signal.from_dict() creates signal with strategy context."""
        data = {
            "direction": "BUY",
            "symbol": "ETH",
            "indicator": "RSI",
            "confidence": 0.8,
            "timestamp": 1234567890,
            "params": {"test": "value"},
            "strategy_name": "test_strategy",
            "market": "ETH-USD",
            "exchange": "hyperliquid",
            "timeframe": "4h"
        }

        signal = Signal.from_dict(data)

        # Test basic fields
        assert signal.direction == SignalDirection.BUY
        assert signal.symbol == "ETH"
        assert signal.indicator == "RSI"
        assert signal.confidence == 0.8
        assert signal.timestamp == 1234567890
        assert signal.params == {"test": "value"}

        # Test strategy context fields
        assert signal.strategy_name == "test_strategy"
        assert signal.market == "ETH-USD"
        assert signal.exchange == "hyperliquid"
        assert signal.timeframe == "4h"

    def test_signal_from_dict_without_strategy_context(self):
        """Test Signal.from_dict() handles missing strategy context fields."""
        data = {
            "direction": "SELL",
            "symbol": "BTC",
            "indicator": "MACD"
        }

        signal = Signal.from_dict(data)

        # Test basic fields
        assert signal.direction == SignalDirection.SELL
        assert signal.symbol == "BTC"
        assert signal.indicator == "MACD"
        assert signal.confidence == 0.5  # Default
        assert isinstance(signal.timestamp, int)  # Timestamp auto-generated when not provided
        assert signal.params == {}

        # Test strategy context fields default to None
        assert signal.strategy_name is None
        assert signal.market is None
        assert signal.exchange is None
        assert signal.timeframe is None

    def test_signal_str_representation_with_strategy_context(self):
        """Test Signal.__str__() includes strategy context when present."""
        signal = Signal(
            direction=SignalDirection.BUY,
            symbol="ETH",
            indicator="RSI",
            confidence=0.85,
            timestamp=1234567890,
            strategy_name="test_strategy",
            timeframe="4h"
        )

        str_repr = str(signal)

        # Test basic format
        assert "Signal(BUY, ETH, RSI" in str_repr
        assert "confidence=0.85" in str_repr
        assert "timestamp=1234567890" in str_repr

        # Test strategy context included
        assert "strategy=test_strategy" in str_repr
        assert "timeframe=4h" in str_repr

    def test_signal_str_representation_without_strategy_context(self):
        """Test Signal.__str__() without strategy context (backward compatibility)."""
        signal = Signal(
            direction=SignalDirection.SELL,
            symbol="BTC",
            indicator="MACD",
            confidence=0.6,
            timestamp=1234567890
        )

        str_repr = str(signal)

        # Test basic format
        assert "Signal(SELL, BTC, MACD" in str_repr
        assert "confidence=0.60" in str_repr
        assert "timestamp=1234567890" in str_repr

        # Test no strategy context included
        assert "strategy=" not in str_repr
        assert "timeframe=" not in str_repr

    def test_signal_backward_compatibility(self):
        """Test that existing code using Signal still works."""
        # Test old-style signal creation
        signal = Signal(SignalDirection.NEUTRAL, "MATIC", "StochRSI", 0.4)

        assert signal.direction == SignalDirection.NEUTRAL
        assert signal.symbol == "MATIC"
        assert signal.indicator == "StochRSI"
        assert signal.confidence == 0.4

        # Test to_dict() backward compatibility
        signal_dict = signal.to_dict()
        assert "direction" in signal_dict
        assert "symbol" in signal_dict
        assert "indicator" in signal_dict
        assert "confidence" in signal_dict

        # Test from_dict() backward compatibility
        reconstructed = Signal.from_dict(signal_dict)
        assert reconstructed.direction == signal.direction
        assert reconstructed.symbol == signal.symbol
        assert reconstructed.indicator == signal.indicator
        assert reconstructed.confidence == signal.confidence


class TestBaseIndicatorSignalEnhancements:
    """Test BaseIndicator integration with enhanced Signal class."""

    def test_indicator_process_adds_timeframe_to_signal(self):
        """Test that BaseIndicator.process() adds timeframe context to generated signals."""

        class MockIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
                return data

            def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
                return Signal(
                    direction=SignalDirection.BUY,
                    symbol="ETH",
                    indicator=self.name
                )

        indicator = MockIndicator("test_indicator")
        mock_data = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })

        # Test with strategy timeframe
        _, signal = indicator.process(mock_data, strategy_timeframe="4h")

        assert signal is not None
        assert signal.timeframe == "4h"

    def test_indicator_process_without_strategy_timeframe(self):
        """Test BaseIndicator.process() without strategy timeframe uses indicator default."""

        class MockIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
                return data

            def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
                return Signal(
                    direction=SignalDirection.SELL,
                    symbol="BTC",
                    indicator=self.name
                )

        indicator = MockIndicator("test_indicator", params={"timeframe": "1h"})
        mock_data = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })

        # Test without strategy timeframe
        _, signal = indicator.process(mock_data)

        assert signal is not None
        assert signal.timeframe == "1h"  # Should use indicator's default

    def test_indicator_process_no_signal_generated(self):
        """Test BaseIndicator.process() when no signal is generated."""

        class MockIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
                return data

            def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
                return None  # No signal generated

        indicator = MockIndicator("test_indicator")
        mock_data = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })

        processed_data, signal = indicator.process(mock_data, strategy_timeframe="4h")

        assert signal is None
        assert processed_data is not None

    def test_multiple_signals_with_different_strategy_contexts(self):
        """Test creating multiple signals with different strategy contexts."""
        signals = [
            Signal(
                direction=SignalDirection.BUY,
                symbol="ETH",
                indicator="RSI",
                strategy_name="strategy_1",
                market="ETH-USD",
                exchange="hyperliquid",
                timeframe="1h"
            ),
            Signal(
                direction=SignalDirection.SELL,
                symbol="BTC",
                indicator="MACD",
                strategy_name="strategy_2",
                market="BTC-USD",
                exchange="coinbase",
                timeframe="4h"
            ),
            Signal(
                direction=SignalDirection.NEUTRAL,
                symbol="DOT",
                indicator="BB",
                # No strategy context
            )
        ]

        # Test first signal
        assert signals[0].strategy_name == "strategy_1"
        assert signals[0].market == "ETH-USD"
        assert signals[0].exchange == "hyperliquid"
        assert signals[0].timeframe == "1h"

        # Test second signal
        assert signals[1].strategy_name == "strategy_2"
        assert signals[1].market == "BTC-USD"
        assert signals[1].exchange == "coinbase"
        assert signals[1].timeframe == "4h"

        # Test third signal (no strategy context)
        assert signals[2].strategy_name is None
        assert signals[2].market is None
        assert signals[2].exchange is None
        assert signals[2].timeframe is None
