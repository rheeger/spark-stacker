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

### 3. Timeframe Support and Resolution Switching

- **Enhanced DataManager**: Supports resampling data to different timeframes
- **TimeframeManager**: A dedicated class for managing multiple timeframes during backtesting
- Multiple timeframe support:
  - Minute-level resolution (1m, 3m, 5m, 15m, 30m)
  - Hour-level resolution (1h, 2h, 4h, 6h, 12h)
  - Day-level resolution (1d, 3d, 1w)
- Intelligent resampling to generate higher timeframes from lower timeframes
- On-the-fly resolution switching during backtesting
- Multi-timeframe alignment capabilities for strategy development
- Timeframe-aware data loading and caching

## Usage

### Generating Market Datasets

```python
from app.backtesting.market_dataset_generator import MarketDatasetGenerator

# Initialize generator
generator = MarketDatasetGenerator(data_dir="data/market_datasets")

# Generate datasets for specific symbols with multiple timeframes
generator.generate_standard_datasets(
    symbols=["BTC", "ETH"],
    exchange_type="kraken",
    intervals=["1m", "5m", "15m", "1h", "4h", "1d"],
    use_resampling=True
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

### Working with Multiple Timeframes

```python
from app.backtesting.timeframe_manager import TimeframeManager

# Initialize timeframe manager
manager = TimeframeManager(data_dir="data/market_datasets")

# Get available timeframes for a symbol
timeframes = manager.get_available_timeframes("BTC", market_regime="bull")

# Load data for multiple timeframes
timeframe_data = manager.load_multi_timeframe_data(
    symbol="BTC",
    timeframes=["1m", "5m", "1h", "4h"],
    market_regime="bull"
)

# Align timeframes to a common timeline
aligned_data = manager.align_timeframes(timeframe_data, base_timeframe="1h")

# Get the current candle at a specific point in time for a specific timeframe
current_time = pd.Timestamp("2023-01-01 12:30:00")
current_candle = manager.get_current_candle(timeframe_data, "1h", current_time)

# Resample data on-the-fly to a different timeframe
resampled_data = manager.resample_on_the_fly(
    timeframe_data["1m"],
    source_timeframe="1m",
    target_timeframe="15m"
)
```

### Command-Line Interface

The `create_market_datasets.py` script provides a convenient interface for generating and normalizing datasets:

```bash
# List existing datasets
python scripts/create_market_datasets.py --list-only

# List available timeframes by symbol
python scripts/create_market_datasets.py --list-timeframes

# Generate datasets for specific symbols with custom timeframes
python scripts/create_market_datasets.py --symbols BTC ETH --exchange kraken --timeframes 1m 5m 1h 4h 1d

# Generate datasets with resampling disabled
python scripts/create_market_datasets.py --symbols BTC --no-resampling

# Normalize existing datasets
python scripts/create_market_datasets.py --skip-download --normalization-methods z_score min_max

# Generate and normalize in one go with all options
python scripts/create_market_datasets.py --symbols BTC ETH --timeframes 1m 5m 15m 1h 4h 1d --normalization-methods z_score min_max percent_change rolling_z_score
```

## Directory Structure

```
data/
└── market_datasets/
    ├── bull/              # Bull market datasets
    │   ├── BTC_1m_bull_1.csv
    │   ├── BTC_5m_bull_1.csv
    │   ├── BTC_1h_bull_1.csv
    │   ├── ETH_1m_bull_1.csv
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

1. **Market Regime Labeling**: Create market regime labeling for performance segmentation
2. **Realistic Trading Simulation**: Implement slippage modeling, fee structure, position sizing, and execution timing simulation
3. **Advanced Backtesting Capabilities**: Build comprehensive trade analytics and performance metrics
4. **Multi-Timeframe Strategy Optimization**: Create frameworks for developing and testing strategies across multiple timeframes

## Testing

The implementation includes comprehensive unit tests:

- `tests/unit/test_market_dataset_generator.py`: Tests for the MarketDatasetGenerator class
- `tests/unit/test_data_normalizer.py`: Tests for the DataNormalizer class
- `tests/unit/test_timeframe_manager.py`: Tests for the TimeframeManager class
- `tests/unit/test_data_manager.py`: Tests for the DataManager class with focus on timeframe conversion

Run tests using:

```bash
python -m pytest tests/unit/test_market_dataset_generator.py
python -m pytest tests/unit/test_data_normalizer.py
python -m pytest tests/unit/test_timeframe_manager.py
python -m pytest tests/unit/test_data_manager.py
```
