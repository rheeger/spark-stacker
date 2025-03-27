import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.indicators.base_indicator import (BaseIndicator, Signal,
                                           SignalDirection)

logger = logging.getLogger(__name__)


class UltimateMAIndicator(BaseIndicator):
    """
    Ultimate Moving Average (UMA) indicator implementation.

    This indicator provides multiple types of moving averages with flexible options:
    - Simple Moving Average (SMA)
    - Exponential Moving Average (EMA)
    - Weighted Moving Average (WMA)
    - Hull Moving Average (HullMA)
    - Volume Weighted Moving Average (VWMA)
    - Relative Moving Average (RMA)
    - Triple Exponential Moving Average (TEMA)
    - Tilson T3 Moving Average

    Based on CM_Ultimate_MA_MTF_V2 by ChrisMoody.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Ultimate MA indicator.

        Args:
            name: Indicator name
            params: Parameters for the UMA indicator
                source: Price source to use (default: "close")
                length: Moving Average period (default: 20)
                ma_type: Type of MA (1=SMA, 2=EMA, 3=WMA, 4=HullMA, 5=VWMA, 6=RMA, 7=TEMA, 8=Tilson T3) (default: 1)
                t3_factor: Tilson T3 factor * 0.1 (default: 7, which means 0.7)
                use_second_ma: Whether to use a second MA (default: False)
                length2: Second MA period (default: 50)
                ma_type2: Type of second MA (default: 1)
                t3_factor2: Second MA Tilson T3 factor * 0.1 (default: 7)
                color_based_on_direction: Whether to color based on direction (default: True)
                smooth_factor: Color smoothing factor (default: 2)
        """
        super().__init__(name, params)
        self.source_col = self.params.get("source", "close")
        self.length = self.params.get("length", 20)
        self.ma_type = self.params.get("ma_type", 1)
        self.t3_factor = self.params.get("t3_factor", 7) * 0.1

        self.use_second_ma = self.params.get("use_second_ma", False)
        self.length2 = self.params.get("length2", 50)
        self.ma_type2 = self.params.get("ma_type2", 1)
        self.t3_factor2 = self.params.get("t3_factor2", 7) * 0.1

        self.color_based_on_direction = self.params.get("color_based_on_direction", True)
        self.smooth_factor = self.params.get("smooth_factor", 2)

        # Validate parameters
        if self.ma_type < 1 or self.ma_type > 8:
            logger.warning(f"Invalid ma_type: {self.ma_type}. Using default (1=SMA).")
            self.ma_type = 1

        if self.ma_type2 < 1 or self.ma_type2 > 8:
            logger.warning(f"Invalid ma_type2: {self.ma_type2}. Using default (1=SMA).")
            self.ma_type2 = 1

    def _calculate_sma(self, data: pd.Series, length: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=length).mean()

    def _calculate_ema(self, data: pd.Series, length: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=length, adjust=False).mean()

    def _calculate_wma(self, data: pd.Series, length: int) -> pd.Series:
        """Calculate Weighted Moving Average."""
        weights = np.arange(1, length + 1)
        sum_weights = np.sum(weights)

        result = data.copy()
        result[:] = np.nan

        for i in range(length - 1, len(data)):
            window = data.iloc[i - length + 1:i + 1].values
            result.iloc[i] = np.sum(window * weights) / sum_weights

        return result

    def _calculate_hull_ma(self, data: pd.Series, length: int) -> pd.Series:
        """Calculate Hull Moving Average."""
        half_length = length // 2
        sqrt_length = int(round(math.sqrt(length)))

        wma1 = self._calculate_wma(data, half_length)
        wma2 = self._calculate_wma(data, length)

        # 2 * WMA(half period) - WMA(full period)
        raw = 2 * wma1 - wma2

        # WMA of the above by sqrt(period)
        hull = self._calculate_wma(raw, sqrt_length)

        return hull

    def _calculate_vwma(self, price: pd.Series, volume: pd.Series, length: int) -> pd.Series:
        """Calculate Volume Weighted Moving Average."""
        vol_price = price * volume
        return vol_price.rolling(window=length).sum() / volume.rolling(window=length).sum()

    def _calculate_rma(self, data: pd.Series, length: int) -> pd.Series:
        """Calculate Relative Moving Average (aka Wilder's Smoothing Method)."""
        alpha = 1.0 / length
        return data.ewm(alpha=alpha, adjust=False).mean()

    def _calculate_tema(self, data: pd.Series, length: int) -> pd.Series:
        """Calculate Triple Exponential Moving Average."""
        ema1 = self._calculate_ema(data, length)
        ema2 = self._calculate_ema(ema1, length)
        ema3 = self._calculate_ema(ema2, length)

        tema = 3 * ema1 - 3 * ema2 + ema3
        return tema

    def _calculate_t3(self, data: pd.Series, length: int, factor: float) -> pd.Series:
        """Calculate Tilson T3 Moving Average."""
        def gd(src: pd.Series, len_val: int, fctr: float) -> pd.Series:
            return (
                self._calculate_ema(src, len_val) * (1 + fctr) -
                self._calculate_ema(self._calculate_ema(src, len_val), len_val) * fctr
            )

        gd1 = gd(data, length, factor)
        gd2 = gd(gd1, length, factor)
        t3 = gd(gd2, length, factor)

        return t3

    def _calculate_ma(self, data: pd.Series, volume: Optional[pd.Series],
                      ma_type: int, length: int, t3_factor: float) -> pd.Series:
        """Calculate the specified type of moving average."""
        if ma_type == 1:
            return self._calculate_sma(data, length)
        elif ma_type == 2:
            return self._calculate_ema(data, length)
        elif ma_type == 3:
            return self._calculate_wma(data, length)
        elif ma_type == 4:
            return self._calculate_hull_ma(data, length)
        elif ma_type == 5:
            if volume is None:
                logger.warning("Volume data required for VWMA but not provided. Using SMA instead.")
                return self._calculate_sma(data, length)
            return self._calculate_vwma(data, volume, length)
        elif ma_type == 6:
            return self._calculate_rma(data, length)
        elif ma_type == 7:
            return self._calculate_tema(data, length)
        elif ma_type == 8:
            return self._calculate_t3(data, length, t3_factor)
        else:
            logger.warning(f"Unknown MA type: {ma_type}. Using SMA.")
            return self._calculate_sma(data, length)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ultimate MA values for the provided price data.

        Args:
            data: Price data as pandas DataFrame with at least 'close' column

        Returns:
            DataFrame with Ultimate MA values added as new columns
        """
        if self.source_col not in data.columns:
            raise ValueError(f"DataFrame must contain a '{self.source_col}' price column")

        # Check for minimum data length
        if len(data) < max(self.length, self.length2 if self.use_second_ma else 0):
            logger.warning(
                f"Not enough data points for Ultimate MA calculation. Need at least {self.length}, got {len(data)}"
            )
            return self._add_empty_columns(data)

        # Create a copy of the dataframe to avoid modifying the original
        df = data.copy()

        # Get volume if available (for VWMA)
        volume = df.get("volume", None)

        # Calculate the primary MA
        df["uma_line1"] = self._calculate_ma(
            df[self.source_col], volume, self.ma_type, self.length, self.t3_factor
        )

        # Calculate the secondary MA if enabled
        if self.use_second_ma:
            df["uma_line2"] = self._calculate_ma(
                df[self.source_col], volume, self.ma_type2, self.length2, self.t3_factor2
            )

        # Identify price crossing the moving averages
        # Crossing MA #1
        df["uma_price_crossing_up1"] = (df[self.source_col] > df["uma_line1"]) & (df[self.source_col].shift(1) <= df["uma_line1"].shift(1))
        df["uma_price_crossing_down1"] = (df[self.source_col] < df["uma_line1"]) & (df[self.source_col].shift(1) >= df["uma_line1"].shift(1))

        # Crossing MA #2 (if enabled)
        if self.use_second_ma:
            df["uma_price_crossing_up2"] = (df[self.source_col] > df["uma_line2"]) & (df[self.source_col].shift(1) <= df["uma_line2"].shift(1))
            df["uma_price_crossing_down2"] = (df[self.source_col] < df["uma_line2"]) & (df[self.source_col].shift(1) >= df["uma_line2"].shift(1))

            # MA crossovers
            df["uma_ma_crossing_up"] = (df["uma_line1"] > df["uma_line2"]) & (df["uma_line1"].shift(1) <= df["uma_line2"].shift(1))
            df["uma_ma_crossing_down"] = (df["uma_line1"] < df["uma_line2"]) & (df["uma_line1"].shift(1) >= df["uma_line2"].shift(1))

        # Determine direction trend
        if self.color_based_on_direction:
            smooth = self.smooth_factor
            df["uma_is_uptrend"] = df["uma_line1"] >= df["uma_line1"].shift(smooth)

            if self.use_second_ma:
                df["uma_is_uptrend2"] = df["uma_line2"] >= df["uma_line2"].shift(smooth)

        return df

    def _add_empty_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add empty indicator columns to the dataframe."""
        df = data.copy()
        df["uma_line1"] = np.nan
        df["uma_price_crossing_up1"] = False
        df["uma_price_crossing_down1"] = False
        df["uma_is_uptrend"] = False

        if self.use_second_ma:
            df["uma_line2"] = np.nan
            df["uma_price_crossing_up2"] = False
            df["uma_price_crossing_down2"] = False
            df["uma_ma_crossing_up"] = False
            df["uma_ma_crossing_down"] = False
            df["uma_is_uptrend2"] = False

        return df

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate a trading signal based on Ultimate MA values.

        Buy signals are generated when:
        - Price crosses above the MA (primary or secondary)
        - MAs cross (faster MA crosses above slower MA)

        Sell signals are generated when:
        - Price crosses below the MA (primary or secondary)
        - MAs cross (faster MA crosses below slower MA)

        Args:
            data: Price data with Ultimate MA values

        Returns:
            Signal object if conditions are met, None otherwise
        """
        if "uma_line1" not in data.columns:
            logger.warning("No uma_line1 column in data, cannot generate signal")
            return None

        if len(data) < 2:
            logger.warning("Not enough data points to generate a signal")
            return None

        # Get the latest data point
        latest = data.iloc[-1]
        symbol = latest.get("symbol", "UNKNOWN")

        # Define signal parameters
        confidence = 0.6  # Default confidence
        params = {
            "price": latest[self.source_col],
            "ma1": latest["uma_line1"],
            "ma1_type": self.ma_type,
            "length1": self.length,
        }

        # Add MA2 to params if used
        if self.use_second_ma:
            params.update({
                "ma2": latest["uma_line2"],
                "ma2_type": self.ma_type2,
                "length2": self.length2,
            })

        # Check for price crossing primary MA
        if latest.get("uma_price_crossing_up1", False):
            params["trigger"] = "price_crossing_up_ma1"
            if self.color_based_on_direction and latest.get("uma_is_uptrend", False):
                confidence = 0.75  # Higher confidence if MA is in uptrend

            return Signal(
                direction=SignalDirection.BUY,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params=params,
            )

        elif latest.get("uma_price_crossing_down1", False):
            params["trigger"] = "price_crossing_down_ma1"
            if self.color_based_on_direction and not latest.get("uma_is_uptrend", True):
                confidence = 0.75  # Higher confidence if MA is in downtrend

            return Signal(
                direction=SignalDirection.SELL,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                params=params,
            )

        # If using second MA, check for additional signals
        if self.use_second_ma:
            # Price crossing secondary MA
            if latest.get("uma_price_crossing_up2", False):
                params["trigger"] = "price_crossing_up_ma2"
                if self.color_based_on_direction and latest.get("uma_is_uptrend2", False):
                    confidence = 0.7  # Slightly higher confidence if MA is in uptrend

                return Signal(
                    direction=SignalDirection.BUY,
                    symbol=symbol,
                    indicator=self.name,
                    confidence=confidence,
                    params=params,
                )

            elif latest.get("uma_price_crossing_down2", False):
                params["trigger"] = "price_crossing_down_ma2"
                if self.color_based_on_direction and not latest.get("uma_is_uptrend2", True):
                    confidence = 0.7  # Slightly higher confidence if MA is in downtrend

                return Signal(
                    direction=SignalDirection.SELL,
                    symbol=symbol,
                    indicator=self.name,
                    confidence=confidence,
                    params=params,
                )

            # MA crossovers - usually stronger signals
            if latest.get("uma_ma_crossing_up", False):
                params["trigger"] = "ma_crossing_up"
                confidence = 0.8  # Higher confidence for MA crossover

                return Signal(
                    direction=SignalDirection.BUY,
                    symbol=symbol,
                    indicator=self.name,
                    confidence=confidence,
                    params=params,
                )

            elif latest.get("uma_ma_crossing_down", False):
                params["trigger"] = "ma_crossing_down"
                confidence = 0.8  # Higher confidence for MA crossover

                return Signal(
                    direction=SignalDirection.SELL,
                    symbol=symbol,
                    indicator=self.name,
                    confidence=confidence,
                    params=params,
                )

        # No signal
        return None

    def __str__(self) -> str:
        """String representation of the indicator."""
        ma_types = {
            1: "SMA", 2: "EMA", 3: "WMA", 4: "HullMA",
            5: "VWMA", 6: "RMA", 7: "TEMA", 8: "T3"
        }

        ma_desc = f"{ma_types.get(self.ma_type, 'Unknown')}({self.length})"

        if self.use_second_ma:
            ma_desc += f" + {ma_types.get(self.ma_type2, 'Unknown')}({self.length2})"

        return f"UltimateMA({ma_desc})"
