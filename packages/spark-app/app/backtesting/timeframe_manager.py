import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from app.backtesting.data_manager import DataManager

# Configure logging
logger = logging.getLogger(__name__)

class TimeframeManager:
    """
    Manages timeframe resolution switching for backtesting.

    This class handles loading and switching between different timeframe resolutions
    during backtesting to allow multi-timeframe analysis and strategy testing.
    """

    def __init__(self, data_dir: str = "data/market_datasets"):
        """
        Initialize the timeframe manager.

        Args:
            data_dir: Base directory containing market datasets
        """
        self.data_dir = data_dir
        self.data_manager = DataManager(data_dir)
        self.timeframe_data = {}  # Cache for loaded timeframe data

    def load_dataset(self, filepath: str, force_reload: bool = False) -> pd.DataFrame:
        """
        Load a dataset from file.

        Args:
            filepath: Path to the dataset CSV file
            force_reload: Whether to force reload if already cached

        Returns:
            DataFrame with the dataset
        """
        if not force_reload and filepath in self.timeframe_data:
            return self.timeframe_data[filepath]

        try:
            df = pd.read_csv(filepath)

            # Check if dataset contains required columns
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"Missing required columns in {filepath}: {missing_columns}")
                return pd.DataFrame()

            # Convert timestamp to datetime for easier manipulation
            if "timestamp" in df.columns:
                if pd.api.types.is_numeric_dtype(df["timestamp"]):
                    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                else:
                    df["datetime"] = pd.to_datetime(df["timestamp"])

            # Cache the dataset
            self.timeframe_data[filepath] = df
            return df

        except Exception as e:
            logger.error(f"Error loading dataset {filepath}: {e}")
            return pd.DataFrame()

    def get_available_timeframes(self, symbol: str, market_regime: str = None) -> List[str]:
        """
        Get list of available timeframes for a symbol.

        Args:
            symbol: Symbol to check (e.g., 'BTC', 'ETH')
            market_regime: Optional market regime to filter by (e.g., 'bull', 'bear', 'sideways')

        Returns:
            List of available timeframe intervals
        """
        available_timeframes = set()

        # Define paths to check based on market regime
        paths_to_check = []
        if market_regime:
            market_regime_dir = os.path.join(self.data_dir, market_regime)
            if os.path.exists(market_regime_dir):
                paths_to_check.append(market_regime_dir)
        else:
            # Check all regime directories and normalized directory
            for regime in ["bull", "bear", "sideways"]:
                regime_dir = os.path.join(self.data_dir, regime)
                if os.path.exists(regime_dir):
                    paths_to_check.append(regime_dir)

            normalized_dir = os.path.join(self.data_dir, "normalized")
            if os.path.exists(normalized_dir):
                paths_to_check.append(normalized_dir)

        # Define regex pattern to extract timeframe from filenames
        pattern = re.compile(f"{symbol}_([0-9]+[mhdw])_")

        # Check each directory for matching files
        for path in paths_to_check:
            for filename in os.listdir(path):
                if filename.startswith(f"{symbol}_") and filename.endswith(".csv"):
                    match = pattern.search(filename)
                    if match:
                        timeframe = match.group(1)
                        available_timeframes.add(timeframe)

        return sorted(list(available_timeframes))

    def load_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: List[str],
        market_regime: str = "bull",
        dataset_index: int = 1,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple timeframes.

        Args:
            symbol: Symbol to load data for (e.g., 'BTC', 'ETH')
            timeframes: List of timeframe intervals to load (e.g., ['1m', '5m', '1h'])
            market_regime: Market regime to load data for (e.g., 'bull', 'bear', 'sideways')
            dataset_index: Index of the dataset to load (for multiple datasets of same type)

        Returns:
            Dictionary mapping timeframes to DataFrames
        """
        result = {}

        for timeframe in timeframes:
            # Construct filepath
            filename = f"{symbol}_{timeframe}_{market_regime}_{dataset_index}.csv"
            filepath = os.path.join(self.data_dir, market_regime, filename)

            if not os.path.exists(filepath):
                logger.warning(f"Dataset not found: {filepath}")
                continue

            # Load dataset
            df = self.load_dataset(filepath)

            if not df.empty:
                result[timeframe] = df

        return result

    def align_timeframes(
        self,
        timeframe_data: Dict[str, pd.DataFrame],
        base_timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Align multiple timeframe data to a common timeline.

        Args:
            timeframe_data: Dictionary mapping timeframes to DataFrames
            base_timeframe: Base timeframe to align to

        Returns:
            Dictionary with aligned timeframe data
        """
        if base_timeframe not in timeframe_data:
            logger.error(f"Base timeframe {base_timeframe} not found in data")
            return timeframe_data

        base_df = timeframe_data[base_timeframe]

        # Get start and end times from base timeframe
        start_time = base_df["datetime"].min()
        end_time = base_df["datetime"].max()

        # Align other timeframes to this range
        result = {base_timeframe: base_df}

        for timeframe, df in timeframe_data.items():
            if timeframe == base_timeframe:
                continue

            # Filter to match base timeframe range
            filtered_df = df[(df["datetime"] >= start_time) & (df["datetime"] <= end_time)]

            if filtered_df.empty:
                logger.warning(f"No data in range for timeframe {timeframe}")
                continue

            result[timeframe] = filtered_df

        return result

    def get_current_candle(
        self,
        timeframe_data: Dict[str, pd.DataFrame],
        timeframe: str,
        current_time: pd.Timestamp
    ) -> Optional[pd.Series]:
        """
        Get the current candle for a specific timeframe at a given time.

        Args:
            timeframe_data: Dictionary mapping timeframes to DataFrames
            timeframe: Timeframe to get candle for
            current_time: Current timestamp

        Returns:
            Series with the current candle data, or None if not found
        """
        if timeframe not in timeframe_data:
            logger.error(f"Timeframe {timeframe} not found in data")
            return None

        df = timeframe_data[timeframe]

        # Find the last candle that closed before or at the current time
        candles = df[df["datetime"] <= current_time]

        if candles.empty:
            return None

        return candles.iloc[-1]

    def resample_on_the_fly(
        self,
        df: pd.DataFrame,
        source_timeframe: str,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample timeframe data on the fly during backtesting.

        Args:
            df: Source DataFrame with OHLCV data
            source_timeframe: Source timeframe interval
            target_timeframe: Target timeframe interval

        Returns:
            Resampled DataFrame
        """
        return self.data_manager.resample_timeframe(df, target_timeframe)
