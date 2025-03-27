import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.indicators.base_indicator import (BaseIndicator, Signal,
                                           SignalDirection)

logger = logging.getLogger(__name__)


class MACDIndicator(BaseIndicator):
    """
    Moving Average Convergence Divergence (MACD) indicator implementation.

    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the MACD indicator.

        Args:
            name: Indicator name
            params: Parameters for the MACD indicator
                fast_period: The fast EMA period (default: 12)
                slow_period: The slow EMA period (default: 26)
                signal_period: The signal line period (default: 9)
                trigger_threshold: The threshold for signal generation (default: 0)
        """
        super().__init__(name, params)
        self.fast_period = self.params.get("fast_period", 12)
        self.slow_period = self.params.get("slow_period", 26)
        self.signal_period = self.params.get("signal_period", 9)
        self.trigger_threshold = self.params.get("trigger_threshold", 0)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD values for the provided price data.

        Args:
            data: Price data as pandas DataFrame with at least 'close' column

        Returns:
            DataFrame with MACD values added as new columns 'macd', 'macd_signal', 'macd_histogram'
        """
        if "close" not in data.columns:
            raise ValueError("DataFrame must contain a 'close' price column")

        if len(data) < self.slow_period + self.signal_period:
            logger.warning(
                f"Not enough data points for MACD calculation. Need at least {self.slow_period + self.signal_period}, got {len(data)}"
            )
            # Add empty MACD columns
            data["macd"] = np.nan
            data["macd_signal"] = np.nan
            data["macd_histogram"] = np.nan
            return data

        # Create a copy of the dataframe to avoid modifying the original
        df = data.copy()

        # Calculate fast and slow EMAs
        fast_ema = df["close"].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df["close"].ewm(span=self.slow_period, adjust=False).mean()

        # Calculate MACD line
        df["macd"] = fast_ema - slow_ema

        # Calculate signal line (EMA of MACD)
        df["macd_signal"] = df["macd"].ewm(span=self.signal_period, adjust=False).mean()

        # Calculate histogram (MACD - Signal)
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Calculate crossovers
        df["macd_crosses_above_signal"] = (
            (df["macd"] > df["macd_signal"]) &
            (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        )
        df["macd_crosses_below_signal"] = (
            (df["macd"] < df["macd_signal"]) &
            (df["macd"].shift(1) >= df["macd_signal"].shift(1))
        )

        # Zero line crossovers
        df["macd_crosses_above_zero"] = (
            (df["macd"] > 0) &
            (df["macd"].shift(1) <= 0)
        )
        df["macd_crosses_below_zero"] = (
            (df["macd"] < 0) &
            (df["macd"].shift(1) >= 0)
        )

        return df

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate a trading signal based on MACD values.

        Buy signals are generated when the MACD line crosses above the signal line.
        Sell signals are generated when the MACD line crosses below the signal line.

        Args:
            data: Price data with MACD values

        Returns:
            Signal object if conditions are met, None otherwise
        """
        if "macd" not in data.columns or "macd_signal" not in data.columns:
            logger.warning("MACD or signal columns missing in data, cannot generate signal")
            return None

        if len(data) < 2:
            logger.warning("Not enough data points to generate a signal")
            return None

        # Get the latest data point
        latest = data.iloc[-1]
        symbol = latest.get("symbol", "UNKNOWN")

        # Check for buy signal (MACD crosses above signal line)
        if latest.get("macd_crosses_above_signal", False):
            # Calculate confidence based on histogram value and zero line position
            histogram = latest["macd_histogram"]
            macd_value = latest["macd"]

            # Higher confidence if MACD is also crossing above zero or already positive
            if latest.get("macd_crosses_above_zero", False) or macd_value > 0:
                confidence = min(1.0, 0.7 + abs(histogram) / 5)
            else:
                confidence = min(1.0, 0.5 + abs(histogram) / 5)

            return Signal(
                direction=SignalDirection.BUY,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params={
                    "macd": macd_value,
                    "signal": latest["macd_signal"],
                    "histogram": histogram,
                    "trigger": "crosses_above_signal",
                },
            )

        # Check for sell signal (MACD crosses below signal line)
        elif latest.get("macd_crosses_below_signal", False):
            # Calculate confidence based on histogram value and zero line position
            histogram = latest["macd_histogram"]
            macd_value = latest["macd"]

            # Higher confidence if MACD is also crossing below zero or already negative
            if latest.get("macd_crosses_below_zero", False) or macd_value < 0:
                confidence = min(1.0, 0.7 + abs(histogram) / 5)
            else:
                confidence = min(1.0, 0.5 + abs(histogram) / 5)

            return Signal(
                direction=SignalDirection.SELL,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params={
                    "macd": macd_value,
                    "signal": latest["macd_signal"],
                    "histogram": histogram,
                    "trigger": "crosses_below_signal",
                },
            )

        # No signal
        return None

    def __str__(self) -> str:
        """String representation of the indicator."""
        return f"MACD(fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period})"
