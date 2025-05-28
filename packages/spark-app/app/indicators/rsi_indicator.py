import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from app.indicators.base_indicator import (BaseIndicator, Signal,
                                           SignalDirection)

logger = logging.getLogger(__name__)


class RSIIndicator(BaseIndicator):
    """
    Relative Strength Index (RSI) indicator implementation.

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions in the price of an asset.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the RSI indicator.

        Args:
            name: Indicator name
            params: Parameters for the RSI indicator
                period: The period for RSI calculation (default: 14)
                overbought: The overbought threshold (default: 70)
                oversold: The oversold threshold (default: 30)
                signal_period: Minimum number of periods to confirm signal (default: 1)
        """
        super().__init__(name, params)
        self.period = self.params.get("period", 14)
        self.overbought = self.params.get("overbought", 70)
        self.oversold = self.params.get("oversold", 30)
        self.signal_period = self.params.get("signal_period", 1)

        # Update params with actual values being used (including defaults)
        self.params.update({
            "period": self.period,
            "overbought": self.overbought,
            "oversold": self.oversold,
            "signal_period": self.signal_period
        })

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI values for the provided price data.

        Args:
            data: Price data as pandas DataFrame with at least 'close' column

        Returns:
            DataFrame with RSI values added as a new column 'rsi'
        """
        if "close" not in data.columns:
            raise ValueError("DataFrame must contain a 'close' price column")

        if len(data) < self.period + 1:
            logger.warning(
                f"Not enough data points for RSI calculation. Need at least {self.period + 1}, got {len(data)}"
            )
            # Add empty RSI column
            data["rsi"] = np.nan
            return data

        # Create a copy of the dataframe to avoid modifying the original
        df = data.copy()

        # Calculate price changes
        delta = df["close"].diff()

        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Calculate average gains and losses
        avg_gain = gain.rolling(window=self.period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=1).mean()

        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss

        # Calculate RSI
        df["rsi"] = 100 - (100 / (1 + rs))

        # Calculate conditions for buy/sell signals
        df["is_oversold"] = df["rsi"] < self.oversold
        df["is_overbought"] = df["rsi"] > self.overbought

        # Detect crossovers
        df["leaving_oversold"] = (df["rsi"] > self.oversold) & (
            df["rsi"].shift(1) <= self.oversold
        )
        df["leaving_overbought"] = (df["rsi"] < self.overbought) & (
            df["rsi"].shift(1) >= self.overbought
        )

        return df

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate a trading signal based on RSI values.

        Buy signals are generated when RSI moves above the oversold threshold.
        Sell signals are generated when RSI moves below the overbought threshold.

        Args:
            data: Price data with RSI values

        Returns:
            Signal object if conditions are met, None otherwise
        """
        if "rsi" not in data.columns:
            logger.warning("No RSI column in data, cannot generate signal")
            return None

        if len(data) < 2:
            logger.warning("Not enough data points to generate a signal")
            return None

        # Get the latest data point
        latest = data.iloc[-1]
        symbol = latest.get("symbol", "UNKNOWN")

        # Check for buy signal (leaving oversold)
        if latest.get("leaving_oversold", False):
            # Calculate confidence based on how deep RSI was in oversold territory
            prev_rsi = data.iloc[-2]["rsi"]
            confidence = min(1.0, (self.oversold - prev_rsi) / self.oversold + 0.5)

            return Signal(
                direction=SignalDirection.BUY,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params={
                    "rsi": latest["rsi"],
                    "trigger": "leaving_oversold",
                    "prev_rsi": prev_rsi,
                },
            )

        # Check for sell signal (leaving overbought)
        elif latest.get("leaving_overbought", False):
            # Calculate confidence based on how high RSI was in overbought territory
            prev_rsi = data.iloc[-2]["rsi"]
            confidence = min(
                1.0, (prev_rsi - self.overbought) / (100 - self.overbought) + 0.5
            )

            return Signal(
                direction=SignalDirection.SELL,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params={
                    "rsi": latest["rsi"],
                    "trigger": "leaving_overbought",
                    "prev_rsi": prev_rsi,
                },
            )

        # No signal
        return None

    def __str__(self) -> str:
        """String representation of the indicator."""
        return f"RSI(period={self.period}, overbought={self.overbought}, oversold={self.oversold})"
