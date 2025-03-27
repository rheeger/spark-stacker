import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.indicators.base_indicator import BaseIndicator, Signal, SignalDirection

logger = logging.getLogger(__name__)


class AdaptiveTrendFinderIndicator(BaseIndicator):
    """
    Adaptive Trend Finder indicator implementation.

    This indicator identifies the strongest trend by analyzing multiple lookback periods
    and selecting the one with the highest correlation (Pearson's R). It then creates
    a logarithmic regression channel to visualize the trend.

    Original pine script by Julien Eche.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Adaptive Trend Finder indicator.

        Args:
            name: Indicator name
            params: Parameters for the indicator
                use_long_term: Whether to use long-term channel mode (default: False)
                dev_multiplier: Deviation multiplier for channel width (default: 2.0)
                source: Price source to use (default: "close")
        """
        super().__init__(name, params)
        self.use_long_term = self.params.get("use_long_term", False)
        self.dev_multiplier = self.params.get("dev_multiplier", 2.0)
        self.source_col = self.params.get("source", "close")

        # Define periods to analyze based on mode
        if self.use_long_term:
            self.periods = [
                300,
                350,
                400,
                450,
                500,
                550,
                600,
                650,
                700,
                750,
                800,
                850,
                900,
                950,
                1000,
                1050,
                1100,
                1150,
                1200,
            ]
        else:
            self.periods = [
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                110,
                120,
                130,
                140,
                150,
                160,
                170,
                180,
                190,
                200,
            ]

    def _calc_log_regression(
        self, log_prices: pd.Series, length: int
    ) -> Tuple[float, float, float, float]:
        """
        Calculate logarithmic regression and Pearson's R correlation for the given period.

        Args:
            log_prices: Logarithm of prices
            length: Period to analyze

        Returns:
            Tuple of (std_dev, pearson_r, slope, intercept)
        """
        if len(log_prices) < length:
            return np.nan, np.nan, np.nan, np.nan

        # Create X values (indices: 1, 2, 3, ...)
        x = np.arange(1, length + 1)
        # Get the required log prices (most recent length values)
        y = log_prices.iloc[-length:].values

        # Calculate components for linear regression
        sum_x = np.sum(x)
        sum_xx = np.sum(x * x)
        sum_y = np.sum(y)
        sum_yx = np.sum(y * x)

        # Calculate slope and intercept of the regression line
        slope = (length * sum_yx - sum_x * sum_y) / (length * sum_xx - sum_x * sum_x)
        average = sum_y / length
        intercept = average - slope * sum_x / length + slope

        # Calculate the regression values
        period_1 = length - 1
        regres = intercept + slope * period_1 * 0.5
        sum_slp = intercept

        # Calculate deviations and correlation components
        sum_dev = 0.0
        sum_dxx = 0.0
        sum_dyy = 0.0
        sum_dyx = 0.0

        for i in range(length):
            l_src = y[i]
            dxt = l_src - average
            dyt = sum_slp - regres
            l_src = l_src - sum_slp
            sum_slp += slope
            sum_dxx += dxt * dxt
            sum_dyy += dyt * dyt
            sum_dyx += dxt * dyt
            sum_dev += l_src * l_src

        # Calculate unbiased standard deviation and Pearson's R
        un_std_dev = math.sqrt(sum_dev / period_1)  # unbiased std dev
        divisor = sum_dxx * sum_dyy
        pearson_r = sum_dyx / math.sqrt(divisor) if divisor > 0 else 0

        return un_std_dev, pearson_r, slope, intercept

    def _calculate_confidence(self, pearson_r: float) -> str:
        """
        Map Pearson's R correlation to trend strength confidence.

        Args:
            pearson_r: Pearson's R correlation coefficient

        Returns:
            String description of trend strength
        """
        if pearson_r < 0.2:
            return "Extremely Weak"
        elif pearson_r < 0.3:
            return "Very Weak"
        elif pearson_r < 0.4:
            return "Weak"
        elif pearson_r < 0.5:
            return "Mostly Weak"
        elif pearson_r < 0.6:
            return "Somewhat Weak"
        elif pearson_r < 0.7:
            return "Moderately Weak"
        elif pearson_r < 0.8:
            return "Moderate"
        elif pearson_r < 0.9:
            return "Moderately Strong"
        elif pearson_r < 0.92:
            return "Mostly Strong"
        elif pearson_r < 0.94:
            return "Strong"
        elif pearson_r < 0.96:
            return "Very Strong"
        elif pearson_r < 0.98:
            return "Exceptionally Strong"
        else:
            return "Ultra Strong"

    def _confidence_to_value(self, confidence_str: str) -> float:
        """
        Convert text confidence to numeric value between 0.5 and 1.0.

        Args:
            confidence_str: Text description of trend strength

        Returns:
            Numeric confidence value between 0.5 and 1.0
        """
        confidence_map = {
            "Extremely Weak": 0.5,
            "Very Weak": 0.53,
            "Weak": 0.56,
            "Mostly Weak": 0.59,
            "Somewhat Weak": 0.62,
            "Moderately Weak": 0.65,
            "Moderate": 0.7,
            "Moderately Strong": 0.75,
            "Mostly Strong": 0.8,
            "Strong": 0.85,
            "Very Strong": 0.9,
            "Exceptionally Strong": 0.95,
            "Ultra Strong": 1.0,
        }
        return confidence_map.get(confidence_str, 0.5)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Adaptive Trend Finder values for the provided price data.

        Args:
            data: Price data as pandas DataFrame with at least 'close' column

        Returns:
            DataFrame with Adaptive Trend Finder values added as new columns
        """
        if self.source_col not in data.columns:
            raise ValueError(
                f"DataFrame must contain a '{self.source_col}' price column"
            )

        # Check if we have enough data for minimum period
        min_period = min(self.periods)
        if len(data) < min_period:
            logger.warning(
                f"Not enough data points for Adaptive Trend Finder calculation. Need at least {min_period}, got {len(data)}"
            )
            return self._add_empty_columns(data)

        # Create a copy of the dataframe to avoid modifying the original
        df = data.copy()

        # Calculate log of prices
        log_prices = np.log(df[self.source_col])

        # Calculate regression for each period and find the one with highest correlation
        best_period_idx = 0
        highest_pearson_r = -1
        results = []

        for i, period in enumerate(self.periods):
            if len(df) >= period:
                std_dev, pearson_r, slope, intercept = self._calc_log_regression(
                    log_prices, period
                )
                results.append((std_dev, pearson_r, slope, intercept, period))

                if not np.isnan(pearson_r) and pearson_r > highest_pearson_r:
                    highest_pearson_r = pearson_r
                    best_period_idx = i
            else:
                results.append((np.nan, np.nan, np.nan, np.nan, period))

        # Get the best period's parameters
        if len(results) > 0 and not np.isnan(highest_pearson_r):
            std_dev, pearson_r, slope, intercept, period = results[best_period_idx]

            # Calculate regression line and bands
            df["atf_period"] = period
            df["atf_pearson_r"] = abs(pearson_r)
            df["atf_confidence"] = self._calculate_confidence(abs(pearson_r))
            df["atf_slope"] = slope

            # Calculate middle, upper, and lower regression lines at current bar
            time_index = np.arange(1, len(df) + 1)
            log_midline = intercept + slope * (period - time_index)
            df["atf_midline"] = np.exp(log_midline)

            upper_dev = self.dev_multiplier * std_dev
            lower_dev = self.dev_multiplier * std_dev

            df["atf_upper"] = df["atf_midline"] * np.exp(upper_dev)
            df["atf_lower"] = df["atf_midline"] / np.exp(lower_dev)

            # Calculate price position relative to the channel
            df["atf_above_upper"] = df[self.source_col] > df["atf_upper"]
            df["atf_below_lower"] = df[self.source_col] < df["atf_lower"]
            df["atf_crossing_upper"] = (df[self.source_col] > df["atf_upper"]) & (
                df[self.source_col].shift(1) <= df["atf_upper"].shift(1)
            )
            df["atf_crossing_lower"] = (df[self.source_col] < df["atf_lower"]) & (
                df[self.source_col].shift(1) >= df["atf_lower"].shift(1)
            )
            df["atf_returning_from_upper"] = (df[self.source_col] < df["atf_upper"]) & (
                df[self.source_col].shift(1) >= df["atf_upper"].shift(1)
            )
            df["atf_returning_from_lower"] = (df[self.source_col] > df["atf_lower"]) & (
                df[self.source_col].shift(1) <= df["atf_lower"].shift(1)
            )

            # Calculate position within the channel as a percentage (0% = lower band, 100% = upper band)
            df["atf_channel_position"] = (
                (df[self.source_col] - df["atf_lower"])
                / (df["atf_upper"] - df["atf_lower"])
                * 100
            )

            return df
        else:
            return self._add_empty_columns(data)

    def _add_empty_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add empty indicator columns to the dataframe."""
        df = data.copy()
        df["atf_period"] = np.nan
        df["atf_pearson_r"] = np.nan
        df["atf_confidence"] = None
        df["atf_slope"] = np.nan
        df["atf_midline"] = np.nan
        df["atf_upper"] = np.nan
        df["atf_lower"] = np.nan
        df["atf_above_upper"] = False
        df["atf_below_lower"] = False
        df["atf_crossing_upper"] = False
        df["atf_crossing_lower"] = False
        df["atf_returning_from_upper"] = False
        df["atf_returning_from_lower"] = False
        df["atf_channel_position"] = np.nan
        return df

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate a trading signal based on Adaptive Trend Finder values.

        Buy signals are generated when:
        - Price returns from the lower band (bouncing up)

        Sell signals are generated when:
        - Price returns from the upper band (bouncing down)

        Args:
            data: Price data with Adaptive Trend Finder values

        Returns:
            Signal object if conditions are met, None otherwise
        """
        required_cols = [
            "atf_midline",
            "atf_upper",
            "atf_lower",
            "atf_confidence",
            "atf_slope",
            "atf_channel_position",
        ]
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

        # Check the slope (trend direction)
        trend_is_up = (
            latest["atf_slope"] < 0
        )  # Note: In log regression with time reversed, negative slope = uptrend
        confidence_str = latest["atf_confidence"]
        confidence_value = self._confidence_to_value(confidence_str)

        # Check for buy signal - price returning from lower band in uptrend
        if latest.get("atf_returning_from_lower", False) and trend_is_up:
            channel_pos = latest["atf_channel_position"]

            return Signal(
                direction=SignalDirection.BUY,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence_value,
                params={
                    "price": latest[self.source_col],
                    "lower_band": latest["atf_lower"],
                    "midline": latest["atf_midline"],
                    "channel_position": channel_pos,
                    "period": latest["atf_period"],
                    "pearson_r": latest["atf_pearson_r"],
                    "confidence": confidence_str,
                    "trigger": "returning_from_lower",
                },
            )

        # Check for sell signal - price returning from upper band in downtrend
        elif latest.get("atf_returning_from_upper", False) and not trend_is_up:
            channel_pos = latest["atf_channel_position"]

            return Signal(
                direction=SignalDirection.SELL,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence_value,
                params={
                    "price": latest[self.source_col],
                    "upper_band": latest["atf_upper"],
                    "midline": latest["atf_midline"],
                    "channel_position": channel_pos,
                    "period": latest["atf_period"],
                    "pearson_r": latest["atf_pearson_r"],
                    "confidence": confidence_str,
                    "trigger": "returning_from_upper",
                },
            )

        # No signal
        return None

    def __str__(self) -> str:
        """String representation of the indicator."""
        mode = "Long-Term" if self.use_long_term else "Short-Term"
        return f"AdaptiveTrendFinder({mode}, dev_multiplier={self.dev_multiplier})"
