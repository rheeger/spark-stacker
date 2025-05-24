import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from app.indicators.base_indicator import (BaseIndicator, Signal,
                                           SignalDirection)

logger = logging.getLogger(__name__)


class BollingerBandsIndicator(BaseIndicator):
    """
    Bollinger Bands indicator implementation.

    Bollinger Bands measure volatility by plotting bands that are standard deviations
    away from a simple moving average of the price.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Bollinger Bands indicator.

        Args:
            name: Indicator name
            params: Parameters for the Bollinger Bands indicator
                period: The period for SMA calculation (default: 20)
                std_dev: Number of standard deviations for bands (default: 2)
                mean_reversion_threshold: Threshold for mean reversion signals (default: 0.05)
        """
        super().__init__(name, params)
        self.period = self.params.get("period", 20)
        self.std_dev = self.params.get("std_dev", 2)
        self.mean_reversion_threshold = self.params.get(
            "mean_reversion_threshold", 0.05
        )

        # Update params with actual values being used (including defaults)
        self.params.update({
            "period": self.period,
            "std_dev": self.std_dev,
            "mean_reversion_threshold": self.mean_reversion_threshold
        })

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands values for the provided price data.

        Args:
            data: Price data as pandas DataFrame with at least 'close' column

        Returns:
            DataFrame with Bollinger Bands values added as new columns 'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_%b'
        """
        if "close" not in data.columns:
            raise ValueError("DataFrame must contain a 'close' price column")

        if len(data) < self.period:
            logger.warning(
                f"Not enough data points for Bollinger Bands calculation. Need at least {self.period}, got {len(data)}"
            )
            # Add empty Bollinger Bands columns
            data["bb_middle"] = np.nan
            data["bb_upper"] = np.nan
            data["bb_lower"] = np.nan
            data["bb_width"] = np.nan
            data["bb_%b"] = np.nan
            return data

        # Create a copy of the dataframe to avoid modifying the original
        df = data.copy()

        # Calculate middle band (SMA)
        df["bb_middle"] = df["close"].rolling(window=self.period).mean()

        # Calculate standard deviation
        rolling_std = df["close"].rolling(window=self.period).std()

        # Calculate upper and lower bands
        df["bb_upper"] = df["bb_middle"] + (rolling_std * self.std_dev)
        df["bb_lower"] = df["bb_middle"] - (rolling_std * self.std_dev)

        # Calculate Bollinger Bandwidth (normalized measure of width)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # Calculate %B (shows where price is in relation to the bands)
        df["bb_%b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Calculate conditions for buy/sell signals
        # Price crossing below lower band (potential buy)
        df["price_below_lower"] = df["close"] < df["bb_lower"]
        df["price_crossing_below_lower"] = (df["close"] < df["bb_lower"]) & (
            df["close"].shift(1) >= df["bb_lower"].shift(1)
        )

        # Price crossing above upper band (potential sell)
        df["price_above_upper"] = df["close"] > df["bb_upper"]
        df["price_crossing_above_upper"] = (df["close"] > df["bb_upper"]) & (
            df["close"].shift(1) <= df["bb_upper"].shift(1)
        )

        # Mean reversion signals (price moving back from bands toward middle)
        df["mean_reversion_buy"] = (
            df["bb_%b"].shift(1) < self.mean_reversion_threshold
        ) & (df["bb_%b"] > df["bb_%b"].shift(1) + self.mean_reversion_threshold)

        df["mean_reversion_sell"] = (
            df["bb_%b"].shift(1) > (1 - self.mean_reversion_threshold)
        ) & (df["bb_%b"] < df["bb_%b"].shift(1) - self.mean_reversion_threshold)

        return df

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate a trading signal based on Bollinger Bands values.

        Buy signals are generated when:
        - Price crosses below the lower band (oversold condition)
        - Price was below lower band and starts moving toward middle (mean reversion)

        Sell signals are generated when:
        - Price crosses above the upper band (overbought condition)
        - Price was above upper band and starts moving toward middle (mean reversion)

        Args:
            data: Price data with Bollinger Bands values

        Returns:
            Signal object if conditions are met, None otherwise
        """
        required_cols = ["bb_middle", "bb_upper", "bb_lower", "bb_%b", "bb_width"]
        for col in required_cols:
            if col not in data.columns:
                logger.warning(
                    f"Required column {col} missing in data, cannot generate signal"
                )
                return None

        if len(data) < 2:
            logger.warning("Not enough data points to generate a signal")
            return None

        # Get the latest data point
        latest = data.iloc[-1]
        symbol = latest.get("symbol", "UNKNOWN")

        # Check for buy signal - price crossing below lower band
        if latest.get("price_crossing_below_lower", False):
            # Calculate confidence based on bandwidth and distance from middle
            width = latest["bb_width"]
            percent_b = latest["bb_%b"]
            confidence = min(1.0, 0.6 + (width * 5) + (1 - percent_b) * 0.4)

            return Signal(
                direction=SignalDirection.BUY,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params={
                    "price": latest["close"],
                    "lower_band": latest["bb_lower"],
                    "middle_band": latest["bb_middle"],
                    "percent_b": percent_b,
                    "width": width,
                    "trigger": "price_crossing_below_lower",
                },
            )

        # Check for buy signal - mean reversion from lower band
        elif latest.get("mean_reversion_buy", False):
            # Calculate confidence based on %B movement
            percent_b = latest["bb_%b"]
            prev_percent_b = data.iloc[-2]["bb_%b"]
            percent_b_change = percent_b - prev_percent_b
            confidence = min(1.0, 0.5 + percent_b_change * 5)

            return Signal(
                direction=SignalDirection.BUY,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params={
                    "price": latest["close"],
                    "lower_band": latest["bb_lower"],
                    "middle_band": latest["bb_middle"],
                    "percent_b": percent_b,
                    "prev_percent_b": prev_percent_b,
                    "trigger": "mean_reversion_buy",
                },
            )

        # Check for sell signal - price crossing above upper band
        elif latest.get("price_crossing_above_upper", False):
            # Calculate confidence based on bandwidth and distance from middle
            width = latest["bb_width"]
            percent_b = latest["bb_%b"]
            confidence = min(1.0, 0.6 + (width * 5) + percent_b * 0.4)

            return Signal(
                direction=SignalDirection.SELL,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params={
                    "price": latest["close"],
                    "upper_band": latest["bb_upper"],
                    "middle_band": latest["bb_middle"],
                    "percent_b": percent_b,
                    "width": width,
                    "trigger": "price_crossing_above_upper",
                },
            )

        # Check for sell signal - mean reversion from upper band
        elif latest.get("mean_reversion_sell", False):
            # Calculate confidence based on %B movement
            percent_b = latest["bb_%b"]
            prev_percent_b = data.iloc[-2]["bb_%b"]
            percent_b_change = prev_percent_b - percent_b
            confidence = min(1.0, 0.5 + percent_b_change * 5)

            return Signal(
                direction=SignalDirection.SELL,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params={
                    "price": latest["close"],
                    "upper_band": latest["bb_upper"],
                    "middle_band": latest["bb_middle"],
                    "percent_b": percent_b,
                    "prev_percent_b": prev_percent_b,
                    "trigger": "mean_reversion_sell",
                },
            )

        # No signal
        return None

    def __str__(self) -> str:
        """String representation of the indicator."""
        return f"BollingerBands(period={self.period}, std_dev={self.std_dev})"
