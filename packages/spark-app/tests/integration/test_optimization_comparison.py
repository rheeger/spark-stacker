import os
import tempfile
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.backtesting.backtest_engine import BacktestEngine
from app.backtesting.data_manager import CSVDataSource, DataManager
from app.backtesting.strategy import simple_moving_average_crossover_strategy
from app.indicators.bollinger_bands_indicator import BollingerBandsIndicator
from app.indicators.rsi_indicator import RSIIndicator


class TestOptimizationComparison:
    """Compare genetic algorithm optimization with grid search."""

    @pytest.fixture
    def sample_data_directory(self):
        """Create a temporary directory with sample data."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate sample data
            symbol = "BTC-USD"
            interval = "1d"

            # Generate 365 days of daily data
            start_date = datetime(2020, 1, 1)
            periods = 365

            dates = [start_date + timedelta(days=i) for i in range(periods)]
            timestamps = [int(date.timestamp() * 1000) for date in dates]

            # Generate price data with trends, cycles, and noise
            closes = []
            base_price = 10000.0

            for i in range(periods):
                # Add long-term trend
                trend = i * 10

                # Add market cycles (approximately 60-day cycles)
                cycle = 2000 * np.sin(2 * np.pi * i / 60)

                # Add shorter-term oscillations (approximately 10-day cycles)
                oscillation = 500 * np.sin(2 * np.pi * i / 10)

                # Add random noise
                noise = np.random.normal(0, 200)

                # Combine components
                price = base_price + trend + cycle + oscillation + noise
                closes.append(max(price, 1.0))  # Ensure price > 0

            # Create OHLCV data
            data = {
                "timestamp": timestamps,
                "open": [c * (1 - np.random.uniform(0, 0.01)) for c in closes],
                "high": [c * (1 + np.random.uniform(0, 0.02)) for c in closes],
                "low": [c * (1 - np.random.uniform(0, 0.02)) for c in closes],
                "close": closes,
                "volume": [np.random.uniform(1000, 10000) for _ in range(periods)],
            }

            # Create DataFrame
            df = pd.DataFrame(data)

            # Save to CSV
            file_path = os.path.join(temp_dir, f"{symbol}_{interval}.csv")
            df.to_csv(file_path, index=False)

            yield temp_dir

    @pytest.fixture
    def backtest_engine(self, sample_data_directory):
        """Create a BacktestEngine with sample data."""
        # Create a DataManager
        data_manager = DataManager(data_dir=sample_data_directory)

        # Register a CSV data source
        data_manager.register_data_source("csv", CSVDataSource(sample_data_directory))

        # Create a BacktestEngine
        engine = BacktestEngine(
            data_manager=data_manager,
            initial_balance={"USD": 10000.0},
            maker_fee=0.001,  # 0.1%
            taker_fee=0.002,  # 0.2%
            slippage_model="random",
        )

        return engine

    def test_optimization_methods_comparison(self, backtest_engine):
        """Compare genetic algorithm and grid search optimization methods."""
        # Common test parameters
        symbol = "BTC-USD"
        interval = "1d"
        start_date = "2020-01-01"
        end_date = "2020-06-30"  # First 6 months
        validation_date = "2020-12-31"  # Last 6 months

        print("\n=== Optimization Methods Comparison ===")

        # Set up parameter space
        # For grid search we need discrete options
        param_grid = {
            "fast_period": [5, 10, 15, 20],
            "slow_period": [30, 40, 50, 60],
            "position_size": [0.1, 0.3, 0.5],
        }

        # For genetic optimization we can include continuous ranges
        param_space = {
            "fast_period": list(range(5, 25, 5)),  # Same discrete options
            "slow_period": (25, 65, 5),  # Continuous range
            "position_size": [0.1, 0.3, 0.5],  # Same discrete options
        }

        # 1. Run grid search optimization
        print("\nRunning grid search optimization...")
        start_time = time.time()

        grid_best_params, grid_result = backtest_engine.optimize_parameters(
            strategy_func=simple_moving_average_crossover_strategy,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_source_name="csv",
            param_grid=param_grid,
            metric_to_optimize="sharpe_ratio",
        )

        grid_time = time.time() - start_time
        print(f"Grid search completed in {grid_time:.2f} seconds")
        print(f"Best parameters: {grid_best_params}")
        print(f"Best Sharpe ratio: {grid_result.metrics['sharpe_ratio']:.4f}")
        print(f"Total return: {grid_result.metrics['total_return']*100:.2f}%")

        # 2. Run genetic algorithm optimization
        print("\nRunning genetic algorithm optimization...")
        start_time = time.time()

        genetic_best_params, genetic_result = backtest_engine.genetic_optimize(
            strategy_func=simple_moving_average_crossover_strategy,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_source_name="csv",
            param_space=param_space,
            population_size=12,
            generations=4,
            crossover_rate=0.7,
            mutation_rate=0.2,
            tournament_size=3,
            metric_to_optimize="sharpe_ratio",
            random_seed=42,
        )

        genetic_time = time.time() - start_time
        print(f"Genetic optimization completed in {genetic_time:.2f} seconds")
        print(f"Best parameters: {genetic_best_params}")
        print(f"Best Sharpe ratio: {genetic_result.metrics['sharpe_ratio']:.4f}")
        print(f"Total return: {genetic_result.metrics['total_return']*100:.2f}%")

        # 3. Validate both results on out-of-sample data
        print("\nValidating on out-of-sample data...")

        # Validate grid search result
        grid_validation = backtest_engine.run_backtest(
            strategy_func=simple_moving_average_crossover_strategy,
            symbol=symbol,
            interval=interval,
            start_date=end_date,
            end_date=validation_date,
            data_source_name="csv",
            strategy_params=grid_best_params,
        )

        # Validate genetic optimization result
        genetic_validation = backtest_engine.run_backtest(
            strategy_func=simple_moving_average_crossover_strategy,
            symbol=symbol,
            interval=interval,
            start_date=end_date,
            end_date=validation_date,
            data_source_name="csv",
            strategy_params=genetic_best_params,
        )

        # 4. Compare validation results
        print("\nValidation Results:")
        print(
            f"Grid Search - Sharpe: {grid_validation.metrics['sharpe_ratio']:.4f}, "
            f"Return: {grid_validation.metrics['total_return']*100:.2f}%, "
            f"Max Drawdown: {grid_validation.metrics['max_drawdown']*100:.2f}%"
        )

        print(
            f"Genetic Algorithm - Sharpe: {genetic_validation.metrics['sharpe_ratio']:.4f}, "
            f"Return: {genetic_validation.metrics['total_return']*100:.2f}%, "
            f"Max Drawdown: {genetic_validation.metrics['max_drawdown']*100:.2f}%"
        )

        # 5. Compare performance
        print("\nPerformance Comparison:")
        print(
            f"Grid Search: {grid_time:.2f} seconds, "
            f"evaluating {len(param_grid['fast_period']) * len(param_grid['slow_period']) * len(param_grid['position_size'])} combinations"
        )

        print(
            f"Genetic Algorithm: {genetic_time:.2f} seconds, "
            f"evaluating ~{12 * 4} individuals (population_size * generations)"
        )

        # 6. Summarize findings
        grid_total_evals = (
            len(param_grid["fast_period"])
            * len(param_grid["slow_period"])
            * len(param_grid["position_size"])
        )
        genetic_total_evals = 12 * 4  # population_size * generations

        print("\nEfficiency Comparison:")
        print(f"Grid Search: {grid_time/grid_total_evals:.4f} seconds per evaluation")
        print(
            f"Genetic Algorithm: {genetic_time/genetic_total_evals:.4f} seconds per evaluation"
        )

        # Correct calculation of efficiency gain
        grid_time_per_eval = grid_time / grid_total_evals
        genetic_time_per_eval = genetic_time / genetic_total_evals

        if genetic_time_per_eval > 0:
            efficiency_gain = grid_time_per_eval / genetic_time_per_eval
            print(f"Efficiency ratio: {efficiency_gain:.2f}x")
        else:
            print("Cannot calculate efficiency ratio (division by zero)")

        # Assertions are kept minimal since performance can vary by environment
        # and parameter space can yield different optimal solutions

        # Verify basic functionality
        assert (
            "fast_period" in grid_best_params and "fast_period" in genetic_best_params
        )
        assert (
            "slow_period" in grid_best_params and "slow_period" in genetic_best_params
        )
        assert grid_result.metrics["sharpe_ratio"] > 0
        assert genetic_result.metrics["sharpe_ratio"] > 0

        # Verify continuous parameter handling in genetic algorithm
        if genetic_best_params["slow_period"] not in [30, 40, 50, 60]:
            print(
                "\nGenetic algorithm successfully used a continuous parameter value not available to grid search"
            )

        print("\nOptimization comparison test completed successfully!")
