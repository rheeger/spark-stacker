import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class DataNormalizer:
    """
    Preprocesses and normalizes market data for backtesting.
    Implements various normalization techniques appropriate for financial time series.
    """

    # Define mapping from filename method parts to full method names
    METHOD_MAPPING = {
        "z_score": "z_score",
        "min_max": "min_max",
        "percent_change": "percent_change",
        "rolling_z_score": "rolling_z_score",
        "log_return": "log_return",
        "unknown_method": "unknown_method"
    }

    def __init__(self, data_dir: str = "data/market_datasets"):
        """
        Initialize the data normalizer.

        Args:
            data_dir: Directory containing market datasets
        """
        self.data_dir = data_dir
        self.normalized_dir = os.path.join(data_dir, "normalized")

        # Create normalized data directory
        os.makedirs(self.normalized_dir, exist_ok=True)

    def normalize_dataset(
        self,
        filepath: str,
        normalization_method: str = "z_score",
        window_size: int = 30,
        save_normalized: bool = True
    ) -> pd.DataFrame:
        """
        Normalize a dataset using the specified method.

        Args:
            filepath: Path to the dataset CSV file
            normalization_method: Method to use for normalization
                (z_score, min_max, percent_change, rolling_z_score)
            window_size: Size of rolling window for methods that require it
            save_normalized: Whether to save the normalized dataset

        Returns:
            Normalized DataFrame
        """
        try:
            # Load the dataset
            df = pd.read_csv(filepath)

            # Check if dataset contains required columns
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"Missing required columns in {filepath}: {missing_columns}")
                return df

            # Create a copy of the original DataFrame
            normalized_df = df.copy()

            # Normalize price and volume data
            price_columns = ["open", "high", "low", "close"]

            if normalization_method == "z_score":
                # Z-score normalization (standardization)
                for col in price_columns + ["volume"]:
                    mean = normalized_df[col].mean()
                    std = normalized_df[col].std()
                    normalized_df[col + "_norm"] = (normalized_df[col] - mean) / std

            elif normalization_method == "min_max":
                # Min-max normalization (scaling to [0,1])
                for col in price_columns + ["volume"]:
                    min_val = normalized_df[col].min()
                    max_val = normalized_df[col].max()
                    normalized_df[col + "_norm"] = (normalized_df[col] - min_val) / (max_val - min_val)

            elif normalization_method == "percent_change":
                # Percent change from previous value
                for col in price_columns + ["volume"]:
                    normalized_df[col + "_norm"] = normalized_df[col].pct_change().fillna(0)

            elif normalization_method == "rolling_z_score":
                # Rolling Z-score normalization
                for col in price_columns + ["volume"]:
                    rolling_mean = normalized_df[col].rolling(window=window_size).mean()
                    rolling_std = normalized_df[col].rolling(window=window_size).std()
                    normalized_df[col + "_norm"] = ((normalized_df[col] - rolling_mean) / rolling_std).fillna(0)

            elif normalization_method == "log_return":
                # Log returns
                for col in price_columns:
                    normalized_df[col + "_norm"] = np.log(normalized_df[col] / normalized_df[col].shift(1)).fillna(0)

                # For volume, use percent change
                normalized_df["volume_norm"] = normalized_df["volume"].pct_change().fillna(0)

            else:
                logger.warning(f"Unknown normalization method: {normalization_method}, using original data")
                for col in price_columns + ["volume"]:
                    # Create a copy to ensure series has different name
                    normalized_df[col + "_norm"] = normalized_df[col].copy()

            # Add price movement features
            normalized_df["price_range"] = (normalized_df["high"] - normalized_df["low"]) / normalized_df["low"]
            normalized_df["close_to_high"] = (normalized_df["high"] - normalized_df["close"]) / normalized_df["high"]
            normalized_df["close_to_low"] = (normalized_df["close"] - normalized_df["low"]) / normalized_df["low"]

            # Save normalized dataset if requested
            if save_normalized:
                # Create normalized filename
                filename = os.path.basename(filepath)
                normalized_filename = f"{os.path.splitext(filename)[0]}_{normalization_method}.csv"
                normalized_filepath = os.path.join(self.normalized_dir, normalized_filename)

                # Save to CSV
                normalized_df.to_csv(normalized_filepath, index=False)
                logger.info(f"Saved normalized dataset to {normalized_filepath}")

            return normalized_df

        except Exception as e:
            logger.error(f"Error normalizing dataset {filepath}: {e}")
            return pd.DataFrame()

    def normalize_all_datasets(
        self,
        normalization_methods: List[str] = ["z_score", "min_max", "percent_change"],
        window_size: int = 30,
        save_normalized: bool = True
    ) -> Dict[str, int]:
        """
        Normalize all datasets in the data directory using the specified methods.

        Args:
            normalization_methods: List of normalization methods to apply
            window_size: Size of rolling window for methods that require it
            save_normalized: Whether to save the normalized datasets

        Returns:
            Dictionary with counts of normalized datasets per method
        """
        # Find all CSV files in the data directory (recursive)
        csv_files = []
        for root, _, files in os.walk(self.data_dir):
            # Skip the normalized directory
            if os.path.normpath(root) == os.path.normpath(self.normalized_dir):
                continue

            # Add CSV files
            for file in files:
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))

        # Initialize result counts
        results = {method: 0 for method in normalization_methods}

        # Process each file with each normalization method
        for method in normalization_methods:
            logger.info(f"Applying {method} normalization to {len(csv_files)} datasets...")

            for filepath in tqdm(csv_files, desc=f"{method} normalization"):
                normalized_df = self.normalize_dataset(
                    filepath=filepath,
                    normalization_method=method,
                    window_size=window_size,
                    save_normalized=save_normalized
                )

                if not normalized_df.empty:
                    results[method] += 1

        # Log summary
        logger.info("Normalization complete. Summary:")
        for method, count in results.items():
            logger.info(f"  - {method}: {count} datasets normalized")

        return results

    def list_normalized_datasets(self) -> Dict[str, List[str]]:
        """
        List all normalized datasets.

        Returns:
            Dictionary mapping normalization methods to lists of dataset files
        """
        result = {}

        if not os.path.exists(self.normalized_dir):
            return result

        normalized_files = [f for f in os.listdir(self.normalized_dir) if f.endswith('.csv')]

        # Define method matching regex
        # This will match the method name at the end of the filename before .csv
        method_pattern = re.compile(r'_([a-z_]+)\.csv$')

        # Group files by normalization method
        for file in normalized_files:
            match = method_pattern.search(file)
            if match:
                method_part = match.group(1)

                # Method may be a compound name like "z_score" or "min_max"
                # Try to find it in our known methods
                found = False
                for known_method, method_name in self.METHOD_MAPPING.items():
                    if file.endswith(f"_{known_method}.csv"):
                        method = method_name
                        found = True
                        break

                # If not found, use the matched part
                if not found:
                    method = method_part

                # Add to result
                if method not in result:
                    result[method] = []

                result[method].append(file)

        return result


def main():
    """Process all datasets with different normalization methods."""
    normalizer = DataNormalizer()
    normalizer.normalize_all_datasets()

    # Print summary of available normalized datasets
    datasets = normalizer.list_normalized_datasets()
    print("\nAvailable Normalized Datasets:")
    for method, files in datasets.items():
        print(f"\n{method.upper()} NORMALIZED DATASETS:")
        for file in files[:5]:  # Show just first 5 for each method
            print(f"  - {file}")

        if len(files) > 5:
            print(f"  - ... and {len(files) - 5} more")

if __name__ == "__main__":
    main()
