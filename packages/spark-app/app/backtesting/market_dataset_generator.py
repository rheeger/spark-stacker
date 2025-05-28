import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from app.backtesting.data_manager import DataManager, ExchangeDataSource
from app.connectors.base_connector import MarketType
from app.connectors.connector_factory import ConnectorFactory
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Market regimes with predefined date ranges
MARKET_REGIMES = {
    "bull": {
        "BTC": [
            # 2020-2021 bull market
            (datetime(2020, 10, 1), datetime(2021, 4, 15)),
            # 2023 bull market
            (datetime(2023, 1, 1), datetime(2023, 4, 1))
        ],
        "ETH": [
            # 2020-2021 bull market
            (datetime(2020, 10, 1), datetime(2021, 4, 15)),
            # 2023 bull market
            (datetime(2023, 1, 1), datetime(2023, 4, 1))
        ]
    },
    "bear": {
        "BTC": [
            # 2022 crypto winter
            (datetime(2022, 1, 1), datetime(2022, 6, 30)),
            # 2018 bear market
            (datetime(2018, 1, 1), datetime(2018, 6, 30))
        ],
        "ETH": [
            # 2022 crypto winter
            (datetime(2022, 1, 1), datetime(2022, 6, 30)),
            # 2018 bear market
            (datetime(2018, 1, 1), datetime(2018, 6, 30))
        ]
    },
    "sideways": {
        "BTC": [
            # 2019 consolidation
            (datetime(2019, 3, 1), datetime(2019, 9, 1)),
            # Late 2023 consolidation
            (datetime(2023, 8, 1), datetime(2023, 12, 1))
        ],
        "ETH": [
            # 2019 consolidation
            (datetime(2019, 3, 1), datetime(2019, 9, 1)),
            # Late 2023 consolidation
            (datetime(2023, 8, 1), datetime(2023, 12, 1))
        ]
    }
}

# Standard intervals to collect
INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

class MarketDatasetGenerator:
    """
    Generator for standard market datasets representing different market conditions.
    Downloads and organizes historical market data for backtesting.
    """

    def __init__(self, data_dir: str = "tests/__test_data__/market_data"):
        """
        Initialize the dataset generator.

        Args:
            data_dir: Directory for storing standard datasets
        """
        self.data_dir = data_dir
        self.data_manager = DataManager(data_dir)

        # Create data directory structure
        os.makedirs(data_dir, exist_ok=True)
        for regime in MARKET_REGIMES.keys():
            os.makedirs(os.path.join(data_dir, regime), exist_ok=True)

    def load_environment(self):
        """Load environment variables from the shared .env file."""
        env_path = Path(__file__).parents[4] / "packages" / "shared" / ".env"
        logger.info(f"Loading environment from: {env_path}")
        load_dotenv(env_path)

        # Check if required environment variables are set
        required_vars = []

        # Check for Hyperliquid credentials
        if os.environ.get("WALLET_ADDRESS") and os.environ.get("PRIVATE_KEY"):
            required_vars.append("Hyperliquid")

        # Check for Coinbase credentials
        if os.environ.get("COINBASE_API_KEY") and os.environ.get("COINBASE_API_SECRET"):
            required_vars.append("Coinbase")

        # Check for Kraken credentials
        if os.environ.get("KRAKEN_API_KEY") and os.environ.get("KRAKEN_API_SECRET"):
            required_vars.append("Kraken")

        if not required_vars:
            logger.warning("No exchange credentials found in environment variables.")
            return False

        logger.info(f"Found credentials for: {', '.join(required_vars)}")
        return True

    def create_exchange_connector(self, exchange_type: str) -> bool:
        """
        Create and register an exchange connector for data retrieval.

        Args:
            exchange_type: Type of exchange ('hyperliquid', 'coinbase', 'kraken')

        Returns:
            True if connector was created successfully, False otherwise
        """
        try:
            if exchange_type.lower() == "hyperliquid":
                wallet_address = os.environ.get("WALLET_ADDRESS")
                private_key = os.environ.get("PRIVATE_KEY")
                testnet = os.environ.get("HYPERLIQUID_TESTNET", "false").lower() in ("true", "1", "t", "yes", "y")

                if not wallet_address or not private_key:
                    logger.error("Missing Hyperliquid credentials")
                    return False

                connector = ConnectorFactory.create_connector(
                    exchange_type="hyperliquid",
                    wallet_address=wallet_address,
                    private_key=private_key,
                    testnet=testnet,
                    market_types=[MarketType.PERPETUAL]
                )

            elif exchange_type.lower() == "coinbase":
                api_key = os.environ.get("COINBASE_API_KEY")
                api_secret = os.environ.get("COINBASE_API_SECRET")
                testnet = os.environ.get("COINBASE_USE_SANDBOX", "false").lower() in ("true", "1", "t", "yes", "y")

                if not api_key or not api_secret:
                    logger.error("Missing Coinbase credentials")
                    return False

                connector = ConnectorFactory.create_connector(
                    exchange_type="coinbase",
                    api_key=api_key,
                    api_secret=api_secret,
                    testnet=testnet,
                    market_types=[MarketType.SPOT]
                )

            elif exchange_type.lower() == "kraken":
                api_key = os.environ.get("KRAKEN_API_KEY")
                api_secret = os.environ.get("KRAKEN_API_SECRET")
                testnet = os.environ.get("KRAKEN_USE_SANDBOX", "false").lower() in ("true", "1", "t", "yes", "y")

                if not api_key or not api_secret:
                    logger.error("Missing Kraken credentials")
                    return False

                connector = ConnectorFactory.create_connector(
                    exchange_type="kraken",
                    api_key=api_key,
                    api_secret=api_secret,
                    testnet=testnet,
                    market_types=[MarketType.SPOT, MarketType.PERPETUAL]
                )

            else:
                logger.error(f"Unsupported exchange type: {exchange_type}")
                return False

            if not connector or not connector.connect():
                logger.error(f"Failed to connect to {exchange_type}")
                return False

            # Register data source with DataManager
            self.data_manager.register_data_source(exchange_type,
                                                  ExchangeDataSource(connector))
            logger.info(f"Successfully registered {exchange_type} data source")
            return True

        except Exception as e:
            logger.error(f"Error creating {exchange_type} connector: {e}")
            return False

    def generate_standard_datasets(
        self,
        symbols: List[str] = None,
        exchange_type: str = "kraken",
        intervals: List[str] = None,
        use_resampling: bool = True
    ):
        """
        Generate standard datasets for each market regime.

        Args:
            symbols: List of symbols to generate data for (default: ["BTC", "ETH"])
            exchange_type: Exchange to use for data (default: "kraken")
            intervals: List of timeframe intervals to collect (default: None, uses INTERVALS)
            use_resampling: Whether to use resampling for higher timeframes (default: True)
        """
        if symbols is None:
            symbols = ["BTC", "ETH"]

        if intervals is None:
            intervals = INTERVALS

        # Validate intervals
        valid_intervals = []
        for interval in intervals:
            if interval in INTERVALS:
                valid_intervals.append(interval)
            else:
                logger.warning(f"Ignoring unsupported interval: {interval}")

        if not valid_intervals:
            logger.error("No valid intervals specified")
            return

        intervals = valid_intervals

        # Load environment and create connector
        if not self.load_environment():
            logger.error("Failed to load environment variables")
            return

        if not self.create_exchange_connector(exchange_type):
            logger.error(f"Failed to create {exchange_type} connector")
            return

        # Process each market regime
        for regime, symbols_data in MARKET_REGIMES.items():
            regime_dir = os.path.join(self.data_dir, regime)
            os.makedirs(regime_dir, exist_ok=True)

            for symbol in symbols:
                if symbol not in symbols_data:
                    logger.warning(f"No {regime} market data defined for {symbol}, skipping")
                    continue

                # Process each date range for this symbol and regime
                for i, (start_date, end_date) in enumerate(symbols_data[symbol]):
                    # Determine if we can use resampling
                    if use_resampling and len(intervals) > 1:
                        logger.info(f"Using resampling to generate multiple timeframes for {symbol}")
                        try:
                            # Get multi-timeframe data
                            timeframe_data = self.data_manager.get_multiple_timeframes(
                                source_name=exchange_type,
                                symbol=symbol,
                                intervals=intervals,
                                start_time=start_date,
                                end_time=end_date
                            )

                            # Save each timeframe
                            for interval, df in timeframe_data.items():
                                if not df.empty:
                                    # Create descriptive filename
                                    filename = f"{symbol}_{interval}_{regime}_{i+1}.csv"
                                    file_path = os.path.join(regime_dir, filename)

                                    # Add market regime label
                                    df["market_regime"] = regime

                                    # Save to CSV
                                    df.to_csv(file_path, index=False)
                                    logger.info(f"Saved {len(df)} records to {file_path}")

                        except Exception as e:
                            logger.error(f"Error generating multiple timeframes for {symbol}: {e}")
                            # Fall back to individual downloads
                            use_resampling = False

                    # If not using resampling, download each interval individually
                    if not use_resampling:
                        for interval in intervals:
                            # Create descriptive filename
                            filename = f"{symbol}_{interval}_{regime}_{i+1}.csv"
                            file_path = os.path.join(regime_dir, filename)

                            # Skip if file already exists
                            if os.path.exists(file_path):
                                logger.info(f"Dataset already exists: {file_path}")
                                continue

                            # Download data
                            logger.info(f"Downloading {symbol} {interval} data for {regime} market "
                                    f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")

                            try:
                                df = self.data_manager.download_data(
                                    source_name=exchange_type,
                                    symbol=symbol,
                                    interval=interval,
                                    start_time=start_date,
                                    end_time=end_date,
                                    save=False
                                )

                                # Clean and save data
                                if not df.empty:
                                    df = self.data_manager.clean_data(df)

                                    # Add market regime label
                                    df["market_regime"] = regime

                                    # Save to CSV
                                    df.to_csv(file_path, index=False)
                                    logger.info(f"Saved {len(df)} records to {file_path}")
                                else:
                                    logger.warning(f"No data retrieved for {symbol} {interval} during "
                                                  f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

                            except Exception as e:
                                logger.error(f"Error downloading data for {symbol} {interval}: {e}")

    def list_available_datasets(self) -> Dict[str, List[str]]:
        """
        List all available market datasets.

        Returns:
            Dictionary mapping regimes to lists of available dataset files
        """
        result = {}

        for regime in MARKET_REGIMES.keys():
            regime_dir = os.path.join(self.data_dir, regime)
            if os.path.exists(regime_dir):
                dataset_files = [f for f in os.listdir(regime_dir) if f.endswith('.csv')]
                result[regime] = dataset_files

        return result

# Main function for command-line usage
def main():
    generator = MarketDatasetGenerator()
    generator.generate_standard_datasets()

    # Print summary of available datasets
    datasets = generator.list_available_datasets()
    print("\nAvailable Market Datasets:")
    for regime, files in datasets.items():
        print(f"\n{regime.upper()} MARKET DATASETS:")
        for file in files:
            print(f"  - {file}")

if __name__ == "__main__":
    main()
