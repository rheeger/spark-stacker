#!/usr/bin/env python3
"""
Script to generate and preprocess market datasets for backtesting.
This script combines both the market dataset generation and normalization steps.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Define simple version of the classes to avoid import issues
class SimpleMarketDatasetGenerator:
    """
    Simplified generator for standard market datasets representing different market conditions.
    This is a standalone version for use in the script.
    """

    def __init__(self, data_dir: str = "data/market_datasets"):
        """
        Initialize the dataset generator.

        Args:
            data_dir: Directory for storing standard datasets
        """
        self.data_dir = data_dir

        # Create data directory structure
        os.makedirs(data_dir, exist_ok=True)
        for regime in ["bull", "bear", "sideways"]:
            os.makedirs(os.path.join(data_dir, regime), exist_ok=True)

    def list_available_datasets(self):
        """List all available market datasets."""
        result = {"bull": [], "bear": [], "sideways": []}

        for regime in result.keys():
            regime_dir = os.path.join(self.data_dir, regime)
            if os.path.exists(regime_dir):
                dataset_files = [f for f in os.listdir(regime_dir) if f.endswith('.csv')]
                result[regime] = dataset_files

        return result

    def generate_standard_datasets(self, symbols=None, exchange_type="kraken"):
        """
        Generate standard datasets for each market regime.

        This is a stub method that would normally connect to exchanges.
        In this script version, we'll just create the directory structure.
        """
        logger.warning(
            "Simplified version: Please run the full app.backtesting.market_dataset_generator "
            "module directly for actual data generation."
        )

        # Create directory structure only
        if symbols is None:
            symbols = ["BTC", "ETH"]

        for regime in ["bull", "bear", "sideways"]:
            regime_dir = os.path.join(self.data_dir, regime)
            os.makedirs(regime_dir, exist_ok=True)
            logger.info(f"Created {regime} market regime directory: {regime_dir}")

class SimpleDataNormalizer:
    """
    Simplified normalizer that lists existing normalized datasets.
    This is a standalone version for use in the script.
    """

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

    def list_normalized_datasets(self):
        """List all normalized datasets."""
        result = {}

        if not os.path.exists(self.normalized_dir):
            return result

        normalized_files = [f for f in os.listdir(self.normalized_dir) if f.endswith('.csv')]

        # Simple method extraction
        for file in normalized_files:
            parts = file.split("_")
            if len(parts) > 1:
                method = parts[-1].replace(".csv", "")

                if method not in result:
                    result[method] = []

                result[method].append(file)

        return result

    def normalize_all_datasets(self, normalization_methods=None, window_size=30):
        """
        Normalize all datasets in the data directory.

        This is a stub method in the simplified script version.
        """
        logger.warning(
            "Simplified version: Please run the full app.backtesting.data_normalizer "
            "module directly for actual data normalization."
        )

        # Just create the directory
        os.makedirs(self.normalized_dir, exist_ok=True)
        logger.info(f"Created normalized data directory: {self.normalized_dir}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and preprocess market datasets for backtesting."
    )

    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["BTC", "ETH"],
        help="Symbols to generate data for (default: BTC ETH)"
    )

    parser.add_argument(
        "--exchange",
        type=str,
        choices=["hyperliquid", "coinbase", "kraken"],
        default="kraken",
        help="Exchange to use for data (default: kraken)"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/market_datasets",
        help="Directory to store market datasets (default: data/market_datasets)"
    )

    parser.add_argument(
        "--normalization-methods",
        type=str,
        nargs="+",
        default=["z_score", "min_max", "percent_change"],
        choices=["z_score", "min_max", "percent_change", "rolling_z_score", "log_return"],
        help="Normalization methods to apply (default: z_score min_max percent_change)"
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Window size for rolling window normalization methods (default: 30)"
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download and only normalize existing data"
    )

    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Skip normalization and only download data"
    )

    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list existing datasets without downloading or normalizing"
    )

    parser.add_argument(
        "--use-full-modules",
        action="store_true",
        help="Use full modules instead of simplified versions (requires all dependencies)"
    )

    return parser.parse_args()

def main():
    """Run the market dataset generation and normalization process."""
    args = parse_arguments()

    try:
        # Initialize data directory
        logger.info(f"Using data directory: {args.data_dir}")

        # Default to simple implementations
        MarketDatasetGenerator = SimpleMarketDatasetGenerator
        DataNormalizer = SimpleDataNormalizer

        # Use full modules if requested
        if args.use_full_modules:
            logger.info("Attempting to use full modules (requires all dependencies)")
            try:
                from app.backtesting.data_normalizer import \
                    DataNormalizer as FullNormalizer
                from app.backtesting.market_dataset_generator import \
                    MarketDatasetGenerator as FullGenerator

                # Replace simple classes with full implementations
                MarketDatasetGenerator = FullGenerator
                DataNormalizer = FullNormalizer
                logger.info("Successfully loaded full modules")
            except ImportError as e:
                logger.error(f"Failed to import full modules: {e}")
                logger.info("Falling back to simplified implementations")

        # Step 1: Generate market datasets (if not skipped)
        if args.list_only:
            # Just list existing datasets
            generator = MarketDatasetGenerator(data_dir=args.data_dir)
            datasets = generator.list_available_datasets()

            print("\nAvailable Market Datasets:")
            for regime, files in datasets.items():
                if files:
                    print(f"\n{regime.upper()} MARKET DATASETS:")
                    for file in files:
                        print(f"  - {file}")
                else:
                    print(f"\n{regime.upper()} MARKET DATASETS: None found")

            # List normalized datasets too
            normalizer = DataNormalizer(data_dir=args.data_dir)
            norm_datasets = normalizer.list_normalized_datasets()

            print("\nAvailable Normalized Datasets:")
            if not norm_datasets:
                print("No normalized datasets found.")
            else:
                for method, files in norm_datasets.items():
                    if files:
                        print(f"\n{method.upper()} NORMALIZED DATASETS:")
                        for file in files[:5]:  # Show just first 5
                            print(f"  - {file}")

                        if len(files) > 5:
                            print(f"  - ... and {len(files) - 5} more")

            return

        # Step 1: Generate market datasets (if not skipped)
        if not args.skip_download:
            logger.info(f"Generating market datasets for symbols: {args.symbols} using {args.exchange}")
            generator = MarketDatasetGenerator(data_dir=args.data_dir)
            generator.generate_standard_datasets(
                symbols=args.symbols,
                exchange_type=args.exchange
            )
            logger.info("Market dataset generation complete.")
        else:
            logger.info("Skipping market dataset generation.")

        # Step 2: Normalize datasets (if not skipped)
        if not args.skip_normalize:
            logger.info(f"Normalizing datasets with methods: {args.normalization_methods}")
            normalizer = DataNormalizer(data_dir=args.data_dir)
            normalizer.normalize_all_datasets(
                normalization_methods=args.normalization_methods,
                window_size=args.window_size
            )
            logger.info("Dataset normalization complete.")
        else:
            logger.info("Skipping dataset normalization.")

        # Show summary of what we have
        logger.info("Listing available datasets:")

        # List raw datasets
        generator = MarketDatasetGenerator(data_dir=args.data_dir)
        datasets = generator.list_available_datasets()

        print("\nAvailable Market Datasets:")
        for regime, files in datasets.items():
            if files:
                print(f"\n{regime.upper()} MARKET DATASETS:")
                print(f"  Count: {len(files)} datasets")
                for file in files[:3]:  # Show just first 3 for brevity
                    print(f"  - {file}")
                if len(files) > 3:
                    print(f"  - ... and {len(files) - 3} more")

        # List normalized datasets
        normalizer = DataNormalizer(data_dir=args.data_dir)
        norm_datasets = normalizer.list_normalized_datasets()

        print("\nAvailable Normalized Datasets:")
        if not norm_datasets:
            print("No normalized datasets found.")
        else:
            for method, files in norm_datasets.items():
                if files:
                    print(f"\n{method.upper()} NORMALIZED DATASETS:")
                    print(f"  Count: {len(files)} datasets")
                    for file in files[:3]:  # Show just first 3 for brevity
                        print(f"  - {file}")
                    if len(files) > 3:
                        print(f"  - ... and {len(files) - 3} more")

    except Exception as e:
        logger.error(f"Error in market dataset processing: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
