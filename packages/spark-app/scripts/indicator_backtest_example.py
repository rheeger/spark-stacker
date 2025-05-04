#!/usr/bin/env python3
"""
Example script demonstrating the use of the indicator backtesting framework.

This script shows how to:
1. Load indicators from a configuration file
2. Backtest indicators on historical data
3. Compare indicator performance
4. Generate performance reports
5. Optimize indicator parameters
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to allow imports
current_dir = Path(os.path.abspath(__file__)).parent
project_root = current_dir.parent
sys.path.append(str(project_root.parent.parent))  # Add the root directory to path

# Use relative imports instead of package imports
sys.path.append(str(project_root))
from app.backtesting.backtest_engine import BacktestEngine
from app.backtesting.data_manager import DataManager
from app.backtesting.indicator_backtest_manager import IndicatorBacktestManager
from app.backtesting.indicator_optimizer import IndicatorOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run the indicator backtesting example."""
    # Setup paths
    project_root = Path(os.path.abspath(__file__)).parents[1]
    config_file = project_root / "app" / "backtesting" / "indicator_configs" / "default_indicators.yaml"
    results_dir = project_root / "tests" / "test_results" / "backtesting_reports"

    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize backtesting components
    data_manager = DataManager()
    backtest_engine = BacktestEngine(data_manager=data_manager)
    backtest_manager = IndicatorBacktestManager(backtest_engine=backtest_engine)

    # Load indicators from configuration file
    logger.info(f"Loading indicators from {config_file}")
    backtest_manager.load_indicators_from_config(config_file)
    indicator_names = backtest_manager.list_indicators()
    logger.info(f"Loaded {len(indicator_names)} indicators: {indicator_names}")

    # Define backtest parameters
    symbol = "ETH"
    interval = "1h"
    data_source_name = "csv"  # Changed from hyperliquid to csv which is more likely to be available

    # Create sample data for testing
    from tests.indicator_testing.test_harness import IndicatorTestHarness
    test_harness = IndicatorTestHarness()
    sample_data_path = project_root / "tests" / "test_data" / "sample_eth_data.csv"

    # Check if sample data file exists, if not create one
    if not sample_data_path.exists():
        # Ensure directory exists
        sample_data_path.parent.mkdir(parents=True, exist_ok=True)

        # Create simple sample data for demonstration
        import numpy as np
        import pandas as pd

        # Create a date range
        dates = pd.date_range(start='2023-01-01', periods=500, freq='H')

        # Create price data with some trends
        np.random.seed(42)  # For reproducibility
        close_prices = np.random.randn(500).cumsum() + 1500  # Start around $1500

        # Add some volatility
        volatility = np.random.randn(500) * 10
        high_prices = close_prices + abs(volatility)
        low_prices = close_prices - abs(volatility)
        open_prices = close_prices.copy()
        np.random.shuffle(open_prices)

        # Generate some volume
        volume = np.random.randint(100, 1000, size=500)

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': [int(d.timestamp() * 1000) for d in dates],  # Convert to ms timestamp
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })

        # Save to CSV
        df.to_csv(sample_data_path, index=False)
        logger.info(f"Created sample data file: {sample_data_path}")

    # Register the sample data with the data manager
    data_manager.register_csv_data_source(
        file_path=str(sample_data_path),
        symbol=symbol,
        interval=interval
    )

    # Set date range for backtesting (use the date range from the sample data)
    # For demonstration, we'll use the last 90 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    # Run backtests for all indicators
    logger.info(f"Running backtests for period: {start_date.date()} to {end_date.date()}")

    try:
        results = backtest_manager.backtest_all_indicators(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_source_name=data_source_name,
            leverage=1.0
        )

        # Compare indicators
        comparison = backtest_manager.compare_indicators(metric_name="sharpe_ratio")
        logger.info("\nIndicator Performance Comparison (by Sharpe Ratio):")
        print(comparison)

        # Generate reports for each indicator
        for indicator_name in indicator_names:
            report = backtest_manager.generate_indicator_performance_report(
                indicator_name=indicator_name,
                output_dir=results_dir / indicator_name,
                include_plots=True
            )
            logger.info(f"Generated report for {indicator_name} at {report.get('summary_file')}")

        # Example of parameter optimization for MACD
        if "macd_standard" in indicator_names:
            logger.info("\nRunning parameter optimization for MACD...")

            # Initialize optimizer
            optimizer = IndicatorOptimizer(
                backtest_manager=backtest_manager,
                symbol=symbol,
                interval=interval,
                data_source_name=data_source_name,
                metric_to_optimize="sharpe_ratio",
                higher_is_better=True
            )

            # Define parameter grid
            param_grid = {
                "fast_period": [8, 12, 16],
                "slow_period": [21, 26, 30],
                "signal_period": [5, 9, 13]
            }

            # Run grid search
            best_params, best_indicator, best_result = optimizer.grid_search(
                indicator_type="macd",
                param_grid=param_grid,
                start_date=start_date,
                end_date=end_date,
                leverage=1.0
            )

            # Create optimization report
            optimizer.create_optimization_report(
                output_dir=results_dir / "macd_optimization",
                include_plots=True
            )

            # Run cross-validation on best parameters
            cv_results = optimizer.cross_validation(
                indicator_type="macd",
                params=best_params,
                start_date=start_date,
                end_date=end_date,
                n_folds=3,
                fold_size=30  # days
            )

            logger.info(f"MACD Cross-validation score: {cv_results['cv_score']:.4f} ± {cv_results['cv_std']:.4f}")

    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
