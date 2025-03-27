import logging
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

from app.indicators.base_indicator import (BaseIndicator, Signal,
                                           SignalDirection)

logger = logging.getLogger(__name__)


class MovingAverageIndicator(BaseIndicator):
    """
    Moving Average indicator implementation.

    This indicator can use either Simple Moving Average (SMA) or
    Exponential Moving Average (EMA) and generate signals based on
    crossovers between fast and slow moving averages.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Moving Average indicator.

        Args:
            name: Indicator name
            params: Parameters for the Moving Average indicator
                fast_period: The fast MA period (default: 10)
                slow_period: The slow MA period (default: 30)
                ma_type: Type of moving average - 'sma' or 'ema' (default: 'sma')
                signal_threshold: Minimum percentage difference for signal generation (default: 0.001)
        """
        super().__init__(name, params)
        self.fast_period = self.params.get("fast_period", 10)
        self.slow_period = self.params.get("slow_period", 30)
        self.ma_type = self.params.get("ma_type", "sma")
        self.signal_threshold = self.params.get("signal_threshold", 0.001)

        # Validate parameters
        if self.fast_period >= self.slow_period:
            logger.warning(
                f"Fast period ({self.fast_period}) should be less than slow period ({self.slow_period}), adjusting"
            )
            self.fast_period = max(1, self.slow_period // 2)

        if self.ma_type not in ["sma", "ema"]:
            logger.warning(f"Invalid MA type '{self.ma_type}', defaulting to 'sma'")
            self.ma_type = "sma"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Moving Average values for the provided price data.

        Args:
            data: Price data as pandas DataFrame with at least 'close' column

        Returns:
            DataFrame with moving average values added as new columns 'fast_ma', 'slow_ma', 'ma_diff'
        """
        if "close" not in data.columns:
            raise ValueError("DataFrame must contain a 'close' price column")

        if len(data) < self.slow_period:
            logger.warning(
                f"Not enough data points for Moving Average calculation. Need at least {self.slow_period}, got {len(data)}"
            )
            # Add empty MA columns
            data["fast_ma"] = np.nan
            data["slow_ma"] = np.nan
            data["ma_diff"] = np.nan
            data["ma_ratio"] = np.nan
            return data

        # Create a copy of the dataframe to avoid modifying the original
        df = data.copy()

        # Calculate moving averages
        if self.ma_type == "sma":
            df["fast_ma"] = df["close"].rolling(window=self.fast_period).mean()
            df["slow_ma"] = df["close"].rolling(window=self.slow_period).mean()
        else:  # ema
            df["fast_ma"] = df["close"].ewm(span=self.fast_period, adjust=False).mean()
            df["slow_ma"] = df["close"].ewm(span=self.slow_period, adjust=False).mean()

        # Calculate difference and ratio between fast and slow MAs
        df["ma_diff"] = df["fast_ma"] - df["slow_ma"]
        df["ma_ratio"] = df["fast_ma"] / df["slow_ma"]

        # Calculate crossovers
        df["ma_crosses_above"] = (
            (df["ma_diff"] > 0) & (df["ma_diff"].shift(1) <= 0)
        )
        df["ma_crosses_below"] = (
            (df["ma_diff"] < 0) & (df["ma_diff"].shift(1) >= 0)
        )

        # Calculate price relative to MAs
        df["price_above_slow_ma"] = df["close"] > df["slow_ma"]
        df["price_below_slow_ma"] = df["close"] < df["slow_ma"]
        df["price_crosses_above_slow_ma"] = (
            (df["close"] > df["slow_ma"]) & (df["close"].shift(1) <= df["slow_ma"].shift(1))
        )
        df["price_crosses_below_slow_ma"] = (
            (df["close"] < df["slow_ma"]) & (df["close"].shift(1) >= df["slow_ma"].shift(1))
        )

        return df

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate a trading signal based on Moving Average values.

        Buy signals are generated when:
        - Fast MA crosses above slow MA (golden cross)
        - Price crosses above slow MA (support level)

        Sell signals are generated when:
        - Fast MA crosses below slow MA (death cross)
        - Price crosses below slow MA (resistance break)

        Args:
            data: Price data with Moving Average values

        Returns:
            Signal object if conditions are met, None otherwise
        """
        required_cols = ["fast_ma", "slow_ma", "ma_diff", "ma_ratio"]
        for col in required_cols:
            if col not in data.columns:
                logger.warning(f"Required column {col} missing in data, cannot generate signal")
                return None

        if len(data) < 2:
            logger.warning("Not enough data points to generate a signal")
            return None

        # Get the latest data point
        latest = data.iloc[-1]
        symbol = latest.get("symbol", "UNKNOWN")

        # Check for buy signal - Golden Cross (fast MA crosses above slow MA)
        if latest.get("ma_crosses_above", False):
            # Calculate confidence based on the magnitude of the crossover
            ma_ratio = latest["ma_ratio"]
            confidence = min(1.0, 0.6 + (ma_ratio - 1) * 10)

            return Signal(
                direction=SignalDirection.BUY,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params={
                    "fast_ma": latest["fast_ma"],
                    "slow_ma": latest["slow_ma"],
                    "ma_diff": latest["ma_diff"],
                    "ratio": ma_ratio,
                    "trigger": "golden_cross",
                },
            )

        # Check for buy signal - Price crosses above slow MA
        elif latest.get("price_crosses_above_slow_ma", False):
            # Calculate confidence based on price vs. MA
            slow_ma = latest["slow_ma"]
            price = latest["close"]
            price_to_ma_ratio = price / slow_ma
            confidence = min(1.0, 0.5 + (price_to_ma_ratio - 1) * 10)

            return Signal(
                direction=SignalDirection.BUY,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params={
                    "price": price,
                    "slow_ma": slow_ma,
                    "ratio": price_to_ma_ratio,
                    "trigger": "price_crosses_above_ma",
                },
            )

        # Check for sell signal - Death Cross (fast MA crosses below slow MA)
        elif latest.get("ma_crosses_below", False):
            # Calculate confidence based on the magnitude of the crossover
            ma_ratio = latest["ma_ratio"]
            confidence = min(1.0, 0.6 + (1 - ma_ratio) * 10)

            return Signal(
                direction=SignalDirection.SELL,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params={
                    "fast_ma": latest["fast_ma"],
                    "slow_ma": latest["slow_ma"],
                    "ma_diff": latest["ma_diff"],
                    "ratio": ma_ratio,
                    "trigger": "death_cross",
                },
            )

        # Check for sell signal - Price crosses below slow MA
        elif latest.get("price_crosses_below_slow_ma", False):
            # Calculate confidence based on price vs. MA
            slow_ma = latest["slow_ma"]
            price = latest["close"]
            price_to_ma_ratio = price / slow_ma
            confidence = min(1.0, 0.5 + (1 - price_to_ma_ratio) * 10)

            return Signal(
                direction=SignalDirection.SELL,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params={
                    "price": price,
                    "slow_ma": slow_ma,
                    "ratio": price_to_ma_ratio,
                    "trigger": "price_crosses_below_ma",
                },
            )

        # No signal
        return None

    def __str__(self) -> str:
        """String representation of the indicator."""
        return f"MovingAverage(type={self.ma_type}, fast={self.fast_period}, slow={self.slow_period})"
