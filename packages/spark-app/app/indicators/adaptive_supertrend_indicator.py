import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator, Signal, SignalDirection

logger = logging.getLogger(__name__)


class AdaptiveSupertrendIndicator(BaseIndicator):
    """
    Machine Learning Adaptive SuperTrend indicator implementation.

    This indicator uses K-means clustering to identify different volatility regimes
    and adapts the SuperTrend factor accordingly. It clusters volatility into high,
    medium, and low regimes and applies the SuperTrend calculation with the appropriate
    ATR value for each regime.

    Based on the AlgoAlpha's Pine script implementation.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Adaptive SuperTrend indicator.

        Args:
            name: Indicator name
            params: Parameters for the indicator
                atr_length: Length for ATR calculation (default: 10)
                factor: SuperTrend multiplier (default: 3.0)
                training_length: Period for K-means training data (default: 100)
                high_vol_percentile: Initial high volatility percentile (default: 0.75)
                medium_vol_percentile: Initial medium volatility percentile (default: 0.50)
                low_vol_percentile: Initial low volatility percentile (default: 0.25)
                max_iterations: Maximum iterations for K-means (default: 10)
        """
        super().__init__(name, params)
        self.atr_length = self.params.get("atr_length", 10)
        self.factor = self.params.get("factor", 3.0)
        self.training_length = self.params.get("training_length", 100)
        self.high_vol_percentile = self.params.get("high_vol_percentile", 0.75)
        self.medium_vol_percentile = self.params.get("medium_vol_percentile", 0.50)
        self.low_vol_percentile = self.params.get("low_vol_percentile", 0.25)
        self.max_iterations = self.params.get("max_iterations", 10)

    def _calculate_atr(self, data: pd.DataFrame, length: int) -> pd.Series:
        """Calculate Average True Range."""
        high = data["high"]
        low = data["low"]
        close = data["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()

        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        atr = tr.rolling(window=length).mean()

        return atr

    def _kmeans_clustering(
        self, volatility: pd.Series, training_length: int
    ) -> Tuple[float, float, float, int]:
        """
        Perform K-means clustering on volatility data to identify regimes.

        Args:
            volatility: ATR series for volatility measurement
            training_length: Number of bars to use for training

        Returns:
            Tuple of (high_centroid, medium_centroid, low_centroid, current_cluster)
            where current_cluster is 0 for high, 1 for medium, 2 for low
        """
        # Get training data (most recent values)
        training_data = volatility.iloc[-training_length:].values

        # Remove NaN values
        training_data = training_data[~np.isnan(training_data)]

        if len(training_data) < training_length / 2:
            # Not enough data
            logger.warning(
                f"Not enough valid data for K-means clustering. Need at least {training_length/2}, got {len(training_data)}"
            )
            return np.nan, np.nan, np.nan, -1

        # Initial centroids based on percentiles
        vol_min = np.min(training_data)
        vol_max = np.max(training_data)
        vol_range = vol_max - vol_min

        # Initial centroid guesses based on percentiles
        high_centroid = vol_min + vol_range * self.high_vol_percentile
        medium_centroid = vol_min + vol_range * self.medium_vol_percentile
        low_centroid = vol_min + vol_range * self.low_vol_percentile

        # Perform K-means clustering iterations
        iterations = 0
        prev_high_centroid = None
        prev_medium_centroid = None
        prev_low_centroid = None

        while iterations < self.max_iterations and (
            prev_high_centroid != high_centroid
            or prev_medium_centroid != medium_centroid
            or prev_low_centroid != low_centroid
        ):
            prev_high_centroid = high_centroid
            prev_medium_centroid = medium_centroid
            prev_low_centroid = low_centroid

            # Cluster assignment
            high_cluster = []
            medium_cluster = []
            low_cluster = []

            for val in training_data:
                dist_high = abs(val - high_centroid)
                dist_medium = abs(val - medium_centroid)
                dist_low = abs(val - low_centroid)

                if dist_high < dist_medium and dist_high < dist_low:
                    high_cluster.append(val)
                elif dist_medium < dist_high and dist_medium < dist_low:
                    medium_cluster.append(val)
                else:
                    low_cluster.append(val)

            # Update centroids based on cluster means
            if high_cluster:
                high_centroid = np.mean(high_cluster)
            if medium_cluster:
                medium_centroid = np.mean(medium_cluster)
            if low_cluster:
                low_centroid = np.mean(low_cluster)

            iterations += 1

        # Determine which cluster the current volatility belongs to
        current_volatility = volatility.iloc[-1]
        if pd.isna(current_volatility):
            return high_centroid, medium_centroid, low_centroid, -1

        dist_high = abs(current_volatility - high_centroid)
        dist_medium = abs(current_volatility - medium_centroid)
        dist_low = abs(current_volatility - low_centroid)

        if dist_high < dist_medium and dist_high < dist_low:
            current_cluster = 0  # High volatility
        elif dist_medium < dist_high and dist_medium < dist_low:
            current_cluster = 1  # Medium volatility
        else:
            current_cluster = 2  # Low volatility

        return high_centroid, medium_centroid, low_centroid, current_cluster

    def _calculate_supertrend(
        self, data: pd.DataFrame, factor: float, atr: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate SuperTrend indicator values.

        Args:
            data: Price data as pandas DataFrame
            factor: Multiplier for ATR
            atr: ATR values

        Returns:
            Tuple of (supertrend_values, direction)
            where direction is 1 for uptrend (bearish) and -1 for downtrend (bullish)
        """
        hl2 = (data["high"] + data["low"]) / 2

        # Calculate upper and lower bands
        upper_band = hl2 + factor * atr
        lower_band = hl2 - factor * atr

        # SuperTrend calculation
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(index=data.index, dtype=int)

        # Initialize first value
        supertrend.iloc[0] = 0
        direction.iloc[0] = 1

        # Calculate SuperTrend
        for i in range(1, len(data)):
            # Skip if we have missing data
            if pd.isna(atr.iloc[i - 1]):
                supertrend.iloc[i] = 0
                direction.iloc[i] = 1
                continue

            # Previous values
            prev_upper = upper_band.iloc[i - 1]
            prev_lower = lower_band.iloc[i - 1]
            prev_supertrend = supertrend.iloc[i - 1]
            prev_close = data["close"].iloc[i - 1]

            # Current values
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]
            curr_close = data["close"].iloc[i]

            # Update upper and lower bands
            if (curr_upper < prev_upper) or (prev_close > prev_upper):
                upper_band.iloc[i] = curr_upper
            else:
                upper_band.iloc[i] = prev_upper

            if (curr_lower > prev_lower) or (prev_close < prev_lower):
                lower_band.iloc[i] = curr_lower
            else:
                lower_band.iloc[i] = prev_lower

            # Determine direction and supertrend value
            if prev_supertrend == prev_upper:
                # Previous trend was down (upper band)
                if curr_close > upper_band.iloc[i]:
                    # Price closed above upper band - trend changes to up
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = lower_band.iloc[i]
                else:
                    # Trend continues down
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = upper_band.iloc[i]
            else:
                # Previous trend was up (lower band)
                if curr_close < lower_band.iloc[i]:
                    # Price closed below lower band - trend changes to down
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = upper_band.iloc[i]
                else:
                    # Trend continues up
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = lower_band.iloc[i]

        return supertrend, direction

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Adaptive SuperTrend values for the provided price data.

        Args:
            data: Price data as pandas DataFrame with at least 'open', 'high', 'low', 'close' columns

        Returns:
            DataFrame with Adaptive SuperTrend values added as new columns
        """
        required_cols = ["open", "high", "low", "close"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"DataFrame must contain '{col}' price column")

        # Check if we have enough data for the calculation
        min_length = max(self.atr_length, self.training_length) + 10
        if len(data) < min_length:
            logger.warning(
                f"Not enough data points for Adaptive SuperTrend calculation. Need at least {min_length}, got {len(data)}"
            )
            return self._add_empty_columns(data)

        # Create a copy of the dataframe to avoid modifying the original
        df = data.copy()

        # Calculate ATR
        atr = self._calculate_atr(df, self.atr_length)
        df["ast_atr"] = atr

        # Apply K-means clustering to determine volatility regime
        (
            high_centroid,
            medium_centroid,
            low_centroid,
            current_cluster,
        ) = self._kmeans_clustering(atr, self.training_length)

        # Store centroids and cluster information
        df["ast_high_centroid"] = high_centroid
        df["ast_medium_centroid"] = medium_centroid
        df["ast_low_centroid"] = low_centroid
        df["ast_volatility_cluster"] = current_cluster

        # Get the centroid for the current cluster
        if current_cluster == 0:
            assigned_centroid = high_centroid
        elif current_cluster == 1:
            assigned_centroid = medium_centroid
        elif current_cluster == 2:
            assigned_centroid = low_centroid
        else:
            assigned_centroid = atr.iloc[-1]  # Use current ATR if clustering failed

        # Special case for NaN values
        if pd.isna(assigned_centroid):
            assigned_centroid = atr.iloc[-1]

        # Calculate the SuperTrend using the cluster-specific ATR
        supertrend_atr = pd.Series(index=df.index, dtype=float)
        supertrend_atr.iloc[:] = atr.iloc[:]
        supertrend_atr.iloc[-1] = assigned_centroid

        supertrend, direction = self._calculate_supertrend(
            df, self.factor, supertrend_atr
        )

        # Add SuperTrend values to the DataFrame
        df["ast_supertrend"] = supertrend
        df["ast_direction"] = direction

        # Calculate trend changes for signal generation
        df["ast_trend_change_up"] = (direction == -1) & (direction.shift(1) == 1)
        df["ast_trend_change_down"] = (direction == 1) & (direction.shift(1) == -1)

        # Add bullish/bearish indicators
        df["ast_is_bullish"] = direction == -1
        df["ast_is_bearish"] = direction == 1

        # Calculate price position relative to SuperTrend
        df["ast_above_supertrend"] = df["close"] > supertrend
        df["ast_below_supertrend"] = df["close"] < supertrend

        return df

    def _add_empty_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add empty indicator columns to the dataframe."""
        df = data.copy()
        df["ast_atr"] = np.nan
        df["ast_high_centroid"] = np.nan
        df["ast_medium_centroid"] = np.nan
        df["ast_low_centroid"] = np.nan
        df["ast_volatility_cluster"] = np.nan
        df["ast_supertrend"] = np.nan
        df["ast_direction"] = np.nan
        df["ast_trend_change_up"] = False
        df["ast_trend_change_down"] = False
        df["ast_is_bullish"] = False
        df["ast_is_bearish"] = False
        df["ast_above_supertrend"] = False
        df["ast_below_supertrend"] = False
        return df

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate a trading signal based on Adaptive SuperTrend values.

        Buy signals are generated when:
        - The trend changes from bearish to bullish (direction changes from 1 to -1)

        Sell signals are generated when:
        - The trend changes from bullish to bearish (direction changes from -1 to 1)

        Args:
            data: Price data with Adaptive SuperTrend values

        Returns:
            Signal object if conditions are met, None otherwise
        """
        required_cols = [
            "ast_supertrend",
            "ast_direction",
            "ast_trend_change_up",
            "ast_trend_change_down",
            "ast_volatility_cluster",
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

        # Determine volatility regime description
        volatility_cluster = latest["ast_volatility_cluster"]
        if volatility_cluster == 0:
            volatility_regime = "High"
        elif volatility_cluster == 1:
            volatility_regime = "Medium"
        elif volatility_cluster == 2:
            volatility_regime = "Low"
        else:
            volatility_regime = "Unknown"

        # Calculate confidence based on volatility regime and direction strength
        # Higher confidence in lower volatility regimes (more stable)
        base_confidence = 0.7
        vol_adjust = 0.0

        if volatility_cluster == 0:  # High volatility
            vol_adjust = -0.1  # Lower confidence in high volatility
        elif volatility_cluster == 2:  # Low volatility
            vol_adjust = 0.1  # Higher confidence in low volatility

        confidence = min(max(base_confidence + vol_adjust, 0.5), 0.95)

        # Check for buy signal - trend changed to bullish
        if latest.get("ast_trend_change_up", False):
            return Signal(
                direction=SignalDirection.BUY,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params={
                    "price": latest["close"],
                    "supertrend": latest["ast_supertrend"],
                    "volatility_regime": volatility_regime,
                    "atr": latest["ast_atr"],
                    "cluster": int(volatility_cluster),
                    "trigger": "trend_change_bullish",
                },
            )

        # Check for sell signal - trend changed to bearish
        elif latest.get("ast_trend_change_down", False):
            return Signal(
                direction=SignalDirection.SELL,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params={
                    "price": latest["close"],
                    "supertrend": latest["ast_supertrend"],
                    "volatility_regime": volatility_regime,
                    "atr": latest["ast_atr"],
                    "cluster": int(volatility_cluster),
                    "trigger": "trend_change_bearish",
                },
            )

        # No signal
        return None

    def __str__(self) -> str:
        """String representation of the indicator."""
        return f"AdaptiveSupertrend(atr_length={self.atr_length}, factor={self.factor}, training_length={self.training_length})"
