# Market Data Implementation for Backtesting

## Overview

This document describes the implementation of the historical market data capabilities for the backtesting system. These enhancements address Section 2 of the Phase 3.5.1 checklist: "Enhance Backtesting with Real Market Data."

## Implemented Features

### 1. Standard Market Dataset Creation

- **MarketDatasetGenerator**: A class that downloads and organizes market data into standard datasets
- Support for different market regimes (bull, bear, sideways) with predefined date ranges
- Integration with existing exchange connectors (Hyperliquid, Coinbase, Kraken)
- Dataset organization in a consistent directory structure
- Command-line script for dataset generation

### 2. Data Normalization and Preprocessing

- **DataNormalizer**: A class that implements various normalization techniques for market data
- Supported normalization methods:
  - Z-score normalization (standardization)
  - Min-max normalization (scaling to [0,1])
  - Percent change normalization
  - Rolling z-score normalization
  - Log return calculation
- Additional feature engineering:
  - Price range calculation
  - Close-to-high ratio
  - Close-to-low ratio

## Usage

### Generating Market Datasets

```python
from app.backtesting.market_dataset_generator import MarketDatasetGenerator

# Initialize generator
generator = MarketDatasetGenerator(data_dir="data/market_datasets")

# Generate datasets for specific symbols
generator.generate_standard_datasets(
    symbols=["BTC", "ETH"],
    exchange_type="kraken"
)

# List available datasets
datasets = generator.list_available_datasets()
```

### Normalizing Market Data

```python
from app.backtesting.data_normalizer import DataNormalizer

# Initialize normalizer
normalizer = DataNormalizer(data_dir="data/market_datasets")

# Normalize a single dataset
normalized_df = normalizer.normalize_dataset(
    filepath="data/market_datasets/bull/BTC_1h_bull_1.csv",
    normalization_method="z_score"
)

# Normalize all datasets with multiple methods
normalizer.normalize_all_datasets(
    normalization_methods=["z_score", "min_max", "percent_change"],
    window_size=30
)
```

### Command-Line Interface

The `create_market_datasets.py` script provides a convenient interface for generating and normalizing datasets:

```bash
# List existing datasets
python scripts/create_market_datasets.py --list-only

# Generate datasets for specific symbols
python scripts/create_market_datasets.py --symbols BTC ETH --exchange kraken

# Normalize existing datasets
python scripts/create_market_datasets.py --skip-download --normalization-methods z_score min_max

# Generate and normalize in one go
python scripts/create_market_datasets.py
```

## Directory Structure

```
data/
└── market_datasets/
    ├── bull/              # Bull market datasets
    │   ├── BTC_1h_bull_1.csv
    │   ├── ETH_1h_bull_1.csv
    │   └── ...
    ├── bear/              # Bear market datasets
    │   ├── BTC_1h_bear_1.csv
    │   └── ...
    ├── sideways/          # Sideways market datasets
    │   ├── BTC_1h_sideways_1.csv
    │   └── ...
    └── normalized/        # Normalized datasets
        ├── BTC_1h_bull_1_z_score.csv
        ├── BTC_1h_bull_1_min_max.csv
        └── ...
```

## Next Steps

1. **Timeframe Support**: Implement support for different timeframes and resolution switching
2. **Market Regime Labeling**: Create market regime labeling for performance segmentation
3. **Realistic Trading Simulation**: Implement slippage modeling, fee structure, position sizing, and execution timing simulation

## Testing

The implementation includes comprehensive unit tests:

- `tests/unit/test_market_dataset_generator.py`: Tests for the MarketDatasetGenerator class
- `tests/unit/test_data_normalizer.py`: Tests for the DataNormalizer class

Run tests using:

```bash
python -m pytest tests/unit/test_market_dataset_generator.py
python -m pytest tests/unit/test_data_normalizer.py
```
