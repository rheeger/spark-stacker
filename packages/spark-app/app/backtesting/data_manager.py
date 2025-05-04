import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Retrieve historical data for a specific symbol and timeframe.

        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            interval: Timeframe interval (e.g., '1m', '1h', '1d')
            start_time: Start time in milliseconds since epoch
            end_time: End time in milliseconds since epoch
            limit: Maximum number of candles to retrieve

        Returns:
            DataFrame with OHLCV data
        """
        pass


class ExchangeDataSource(DataSource):
    """Data source that retrieves data from exchange connectors."""

    def __init__(self, connector):
        """
        Initialize with an exchange connector.

        Args:
            connector: An instance of BaseConnector
        """
        self.connector = connector

    def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Retrieve historical data using the exchange connector.

        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            interval: Timeframe interval (e.g., '1m', '1h', '1d')
            start_time: Start time in milliseconds since epoch
            end_time: End time in milliseconds since epoch
            limit: Maximum number of candles to retrieve

        Returns:
            DataFrame with OHLCV data
        """
        # Ensure connector is connected
        if not self.connector.is_connected:
            self.connector.connect()

        # Get historical candles from connector
        candles = self.connector.get_historical_candles(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        # Convert to DataFrame
        if not candles:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(candles)

        # Ensure required columns exist
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' missing from data")
                raise ValueError(f"Required column '{col}' missing from data")

        # Ensure timestamp is in milliseconds
        if df["timestamp"].iloc[0] < 1e12:
            df["timestamp"] = df["timestamp"] * 1000

        # Sort by timestamp
        df = df.sort_values("timestamp")

        # Convert columns to numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        return df


class CSVDataSource(DataSource):
    """Data source that retrieves data from CSV files."""

    def __init__(self, data_dir: str):
        """
        Initialize with a directory containing CSV files.

        Args:
            data_dir: Path to directory containing CSV files
        """
        self.data_dir = data_dir
        self.cache = {}  # Cache for loaded data

    def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Retrieve historical data from CSV files.

        Args:
            symbol: The market symbol (e.g., 'ETH-USD')
            interval: Timeframe interval (e.g., '1m', '1h', '1d')
            start_time: Start time in milliseconds since epoch
            end_time: End time in milliseconds since epoch
            limit: Maximum number of candles to retrieve

        Returns:
            DataFrame with OHLCV data
        """
        # Create a cache key
        cache_key = f"{symbol}_{interval}"

        # Check if data is in cache
        if cache_key in self.cache:
            df = self.cache[cache_key]
        else:
            # Construct file path
            file_path = os.path.join(self.data_dir, f"{symbol}_{interval}.csv")

            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"CSV file not found: {file_path}")
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

            # Read CSV file
            df = pd.read_csv(file_path)

            # Ensure required columns exist
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in required_columns:
                if col not in df.columns:
                    logger.error(
                        f"Required column '{col}' missing from CSV: {file_path}"
                    )
                    raise ValueError(
                        f"Required column '{col}' missing from CSV: {file_path}"
                    )

            # Ensure timestamp is in milliseconds
            if df["timestamp"].iloc[0] < 1e12:
                df["timestamp"] = df["timestamp"] * 1000

            # Sort by timestamp
            df = df.sort_values("timestamp")

            # Convert columns to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])

            # Cache the data
            self.cache[cache_key] = df

        # Apply time filters if provided
        if start_time:
            df = df[df["timestamp"] >= start_time]

        if end_time:
            df = df[df["timestamp"] <= end_time]

        # Apply limit
        if limit and len(df) > limit:
            df = df.tail(limit)

        return df


class DataManager:
    """
    Manages historical data for backtesting and simulation.

    Handles data retrieval, storage, cleaning, and normalization.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data manager.

        Args:
            data_dir: Directory for storing and retrieving data
        """
        self.data_dir = data_dir
        self.data_sources = {}  # Registered data sources

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

    def register_data_source(self, name: str, data_source: DataSource):
        """
        Register a data source.

        Args:
            name: Name for the data source
            data_source: Instance of DataSource
        """
        self.data_sources[name] = data_source
        logger.info(f"Registered data source: {name}")

    def get_data(
        self,
        source_name: str,
        symbol: str,
        interval: str,
        start_time: Optional[Union[int, datetime]] = None,
        end_time: Optional[Union[int, datetime]] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Get historical data from a specified source.

        Args:
            source_name: Name of the registered data source
            symbol: The market symbol (e.g., 'ETH-USD')
            interval: Timeframe interval (e.g., '1m', '1h', '1d')
            start_time: Start time (timestamp in ms or datetime)
            end_time: End time (timestamp in ms or datetime)
            limit: Maximum number of candles to retrieve

        Returns:
            DataFrame with OHLCV data
        """
        # Check if data source exists
        if source_name not in self.data_sources:
            logger.error(f"Data source not found: {source_name}")
            raise ValueError(f"Data source not found: {source_name}")

        # Convert datetime to timestamp if needed
        if isinstance(start_time, datetime):
            start_time = int(start_time.timestamp() * 1000)

        if isinstance(end_time, datetime):
            end_time = int(end_time.timestamp() * 1000)

        # Get data from source
        data_source = self.data_sources[source_name]
        df = data_source.get_historical_data(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        return df

    def save_data(self, df: pd.DataFrame, symbol: str, interval: str) -> str:
        """
        Save data to CSV file.

        Args:
            df: DataFrame with OHLCV data
            symbol: The market symbol (e.g., 'ETH-USD')
            interval: Timeframe interval (e.g., '1m', '1h', '1d')

        Returns:
            Path to saved file
        """
        # Create file name
        file_name = f"{symbol}_{interval}.csv"
        file_path = os.path.join(self.data_dir, file_name)

        # Save to CSV
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} records to {file_path}")

        return file_path

    def download_data(
        self,
        source_name: str,
        symbol: str,
        interval: str,
        start_time: Union[int, datetime],
        end_time: Union[int, datetime],
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Download and optionally save historical data.

        Args:
            source_name: Name of the registered data source
            symbol: The market symbol (e.g., 'ETH-USD')
            interval: Timeframe interval (e.g., '1m', '1h', '1d')
            start_time: Start time (timestamp in ms or datetime)
            end_time: End time (timestamp in ms or datetime)
            save: Whether to save the data to CSV

        Returns:
            DataFrame with downloaded data
        """
        # Convert datetime to timestamp if needed
        if isinstance(start_time, datetime):
            start_time_ts = int(start_time.timestamp() * 1000)
        else:
            start_time_ts = start_time

        if isinstance(end_time, datetime):
            end_time_ts = int(end_time.timestamp() * 1000)
        else:
            end_time_ts = end_time

        # Calculate date ranges to handle API limitations
        # Most exchanges limit to 1000 candles per request
        interval_ms = self._interval_to_milliseconds(interval)
        max_candles_per_request = 1000

        # Initialize empty dataframe for results
        all_data = pd.DataFrame()

        # Download data in chunks
        current_start = start_time_ts

        while current_start < end_time_ts:
            # Calculate end time for this chunk
            chunk_end = min(
                current_start + (interval_ms * max_candles_per_request), end_time_ts
            )

            logger.info(
                f"Downloading {symbol} {interval} data from {self._ms_to_date_str(current_start)} to {self._ms_to_date_str(chunk_end)}"
            )

            # Get data for this chunk
            chunk_data = self.get_data(
                source_name=source_name,
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=chunk_end,
                limit=max_candles_per_request,
            )

            # Append to result dataframe
            all_data = pd.concat([all_data, chunk_data])

            # Update start time for next chunk
            if len(chunk_data) > 0:
                # Use last timestamp + interval as next start
                current_start = chunk_data["timestamp"].max() + interval_ms
            else:
                # If no data, move forward by the max window
                current_start = chunk_end

            # Add a small delay to avoid rate limiting
            time.sleep(0.5)

        # Remove duplicates
        all_data = all_data.drop_duplicates(subset="timestamp").sort_values("timestamp")

        # Save data if requested
        if save and not all_data.empty:
            self.save_data(all_data, symbol, interval)

        return all_data

    def _interval_to_milliseconds(self, interval: str) -> int:
        """Convert interval string to milliseconds."""
        unit = interval[-1]
        value = int(interval[:-1])

        if unit == "m":
            return value * 60 * 1000
        elif unit == "h":
            return value * 60 * 60 * 1000
        elif unit == "d":
            return value * 24 * 60 * 60 * 1000
        elif unit == "w":
            return value * 7 * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Unsupported interval unit: {unit}")

    def _ms_to_date_str(self, ms: int) -> str:
        """Convert milliseconds to date string."""
        dt = datetime.fromtimestamp(ms / 1000)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()

        # Remove rows with NaN values
        df = df.dropna()

        # Remove rows with zero volume (if volume column exists)
        if "volume" in df.columns:
            df = df[df["volume"] > 0]

        # Ensure price integrity (high >= low, etc.)
        df = df[df["high"] >= df["low"]]
        df = df[df["high"] >= df["close"]]
        df = df[df["high"] >= df["open"]]
        df = df[df["low"] <= df["close"]]
        df = df[df["low"] <= df["open"]]

        # Sort by timestamp
        df = df.sort_values("timestamp")

        return df

    def resample_timeframe(self, df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """
        Resample data to a different timeframe.

        Args:
            df: DataFrame with OHLCV data
            target_interval: Target timeframe interval (e.g., '1m', '5m', '1h', '4h', '1d')

        Returns:
            DataFrame resampled to the target interval
        """
        # Make a copy to avoid modifying the original
        df = df.copy()

        # Ensure df has a proper datetime index
        if 'timestamp' in df.columns:
            # Convert timestamp from ms to datetime if needed
            if pd.api.types.is_numeric_dtype(df['timestamp']):
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                df['datetime'] = pd.to_datetime(df['timestamp'])

            # Set index for resampling
            df = df.set_index('datetime')

        # Parse target interval for pandas resampling
        interval_map = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W'
        }

        if target_interval not in interval_map:
            raise ValueError(f"Unsupported target interval: {target_interval}")

        resample_rule = interval_map[target_interval]

        # Resample OHLCV data
        resampled = pd.DataFrame()
        resampled['open'] = df['open'].resample(resample_rule).first()
        resampled['high'] = df['high'].resample(resample_rule).max()
        resampled['low'] = df['low'].resample(resample_rule).min()
        resampled['close'] = df['close'].resample(resample_rule).last()
        resampled['volume'] = df['volume'].resample(resample_rule).sum()

        # Reset index and convert back to timestamp format
        resampled = resampled.reset_index()
        resampled['timestamp'] = resampled['datetime'].astype(int) // 10**6

        # Ensure we have the expected columns and order
        result = resampled[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        return result

    def get_multiple_timeframes(
        self,
        source_name: str,
        symbol: str,
        intervals: List[str],
        start_time: Union[int, datetime],
        end_time: Union[int, datetime],
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple timeframes.

        Args:
            source_name: Name of the registered data source
            symbol: The market symbol (e.g., 'ETH-USD')
            intervals: List of timeframe intervals (e.g., ['1m', '5m', '1h'])
            start_time: Start time (timestamp in ms or datetime)
            end_time: End time (timestamp in ms or datetime)

        Returns:
            Dictionary mapping intervals to DataFrames with OHLCV data
        """
        result = {}

        # Download the smallest timeframe data first
        base_intervals = ['1m', '5m', '15m', '1h', '4h', '1d']
        available_intervals = [i for i in base_intervals if i in intervals]

        if not available_intervals:
            logger.error(f"No valid intervals specified: {intervals}")
            return result

        # Sort intervals by granularity (smallest first)
        interval_values = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080
        }

        available_intervals.sort(key=lambda x: interval_values.get(x, 999999))
        base_interval = available_intervals[0]

        # Get base timeframe data
        logger.info(f"Downloading base timeframe data: {symbol} {base_interval}")
        base_df = self.download_data(
            source_name=source_name,
            symbol=symbol,
            interval=base_interval,
            start_time=start_time,
            end_time=end_time,
            save=True
        )

        if base_df.empty:
            logger.error(f"Failed to download base timeframe data: {symbol} {base_interval}")
            return result

        # Add base timeframe to result
        result[base_interval] = base_df

        # Generate other timeframes by resampling
        for interval in intervals:
            if interval == base_interval:
                continue

            logger.info(f"Generating {interval} timeframe by resampling from {base_interval}")
            try:
                resampled_df = self.resample_timeframe(base_df, interval)
                result[interval] = resampled_df

                # Save resampled data
                file_name = f"{symbol}_{interval}.csv"
                file_path = os.path.join(self.data_dir, file_name)
                resampled_df.to_csv(file_path, index=False)
                logger.info(f"Saved resampled {interval} data ({len(resampled_df)} records) to {file_path}")

            except Exception as e:
                logger.error(f"Error resampling to {interval}: {e}")

        return result
