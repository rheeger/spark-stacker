import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.indicators.base_indicator import BaseIndicator, Signal, SignalDirection

logger = logging.getLogger(__name__)


class MACDIndicator(BaseIndicator):
    """
    Moving Average Convergence Divergence (MACD) indicator implementation.

    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price.
    """

    def __init__(
        self,
        name: str = "macd",
        params: Optional[Dict[str, Any]] = None,
        fast_period: int = None,
        slow_period: int = None,
        signal_period: int = None,
    ):
        """
        Initialize the MACD indicator.

        Args:
            name: Indicator name
            params: Parameters for the MACD indicator
                fast_period: The fast EMA period (default: 12)
                slow_period: The slow EMA period (default: 26)
                signal_period: The signal line period (default: 9)
                trigger_threshold: The threshold for signal generation (default: 0)
            fast_period: Direct parameter for fast EMA period (overrides params)
            slow_period: Direct parameter for slow EMA period (overrides params)
            signal_period: Direct parameter for signal line period (overrides params)
        """
        super().__init__(name, params or {})

        # Allow parameters to be passed directly or via params dict
        self.fast_period = fast_period or self.params.get("fast_period", 12)
        self.slow_period = slow_period or self.params.get("slow_period", 26)
        self.signal_period = signal_period or self.params.get("signal_period", 9)
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
        df["macd_crosses_above_signal"] = (df["macd"] > df["macd_signal"]) & (
            df["macd"].shift(1) <= df["macd_signal"].shift(1)
        )
        df["macd_crosses_below_signal"] = (df["macd"] < df["macd_signal"]) & (
            df["macd"].shift(1) >= df["macd_signal"].shift(1)
        )

        # Zero line crossovers
        df["macd_crosses_above_zero"] = (df["macd"] > 0) & (df["macd"].shift(1) <= 0)
        df["macd_crosses_below_zero"] = (df["macd"] < 0) & (df["macd"].shift(1) >= 0)

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
            logger.warning(
                "MACD or signal columns missing in data, cannot generate signal"
            )
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

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process an entire DataFrame and return all signals in a new DataFrame.

        Args:
            data: Price data as pandas DataFrame with at least 'close' column

        Returns:
            DataFrame with MACD values and signal information
        """
        # Calculate MACD values
        result = self.calculate(data.copy())

        # Create signals column with None as default
        result["signal"] = None
        result["confidence"] = 0.0

        # Detect buy signals (MACD crosses above signal line)
        buy_signals = result["macd_crosses_above_signal"]
        result.loc[buy_signals, "signal"] = SignalDirection.BUY

        # Detect sell signals (MACD crosses below signal line)
        sell_signals = result["macd_crosses_below_signal"]
        result.loc[sell_signals, "signal"] = SignalDirection.SELL

        # Calculate confidence based on histogram value and MACD value
        # For buy signals
        buy_indices = result.index[buy_signals]
        for idx in buy_indices:
            histogram = result.loc[idx, "macd_histogram"]
            macd_value = result.loc[idx, "macd"]

            # Higher confidence if MACD is also crossing above zero or already positive
            if result.loc[idx, "macd_crosses_above_zero"] or macd_value > 0:
                confidence = min(1.0, 0.7 + abs(histogram) / 5)
            else:
                confidence = min(1.0, 0.5 + abs(histogram) / 5)

            result.loc[idx, "confidence"] = confidence

        # For sell signals
        sell_indices = result.index[sell_signals]
        for idx in sell_indices:
            histogram = result.loc[idx, "macd_histogram"]
            macd_value = result.loc[idx, "macd"]

            # Higher confidence if MACD is also crossing below zero or already negative
            if result.loc[idx, "macd_crosses_below_zero"] or macd_value < 0:
                confidence = min(1.0, 0.7 + abs(histogram) / 5)
            else:
                confidence = min(1.0, 0.5 + abs(histogram) / 5)

            result.loc[idx, "confidence"] = confidence

        # Return the result with signals
        return result
