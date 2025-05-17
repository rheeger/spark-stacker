#!/usr/bin/env python3
"""
Script to generate standard market datasets for backtesting.
Downloads and organizes historical data for different market regimes (bull, bear, sideways).
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path so we can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.backtesting.market_dataset_generator import MarketDatasetGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate standard market datasets for backtesting."
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
        "--list-only",
        action="store_true",
        help="Only list existing datasets without downloading new data"
    )

    return parser.parse_args()

def main():
    """Run the market dataset generator."""
    args = parse_arguments()

    try:
        # Initialize generator
        logger.info(f"Initializing market dataset generator with data directory: {args.data_dir}")
        generator = MarketDatasetGenerator(data_dir=args.data_dir)

        if args.list_only:
            # Just list existing datasets
            datasets = generator.list_available_datasets()
            print("\nAvailable Market Datasets:")

            if not any(datasets.values()):
                print("No datasets found. Run without --list-only to generate datasets.")
                return

            for regime, files in datasets.items():
                if files:
                    print(f"\n{regime.upper()} MARKET DATASETS:")
                    for file in files:
                        print(f"  - {file}")
        else:
            # Generate datasets
            logger.info(f"Generating datasets for symbols: {args.symbols} using {args.exchange} exchange")
            generator.generate_standard_datasets(
                symbols=args.symbols,
                exchange_type=args.exchange
            )

            # List generated datasets
            datasets = generator.list_available_datasets()
            print("\nAvailable Market Datasets:")
            for regime, files in datasets.items():
                if files:
                    print(f"\n{regime.upper()} MARKET DATASETS:")
                    for file in files:
                        print(f"  - {file}")

    except Exception as e:
        logger.error(f"Error generating market datasets: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
