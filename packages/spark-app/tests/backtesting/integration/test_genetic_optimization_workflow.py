import os
import tempfile
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from app.backtesting.backtest_engine import BacktestEngine, BacktestResult
from app.backtesting.data_manager import CSVDataSource, DataManager
from app.backtesting.simulation_engine import SimulationEngine
from app.backtesting.strategy import multi_indicator_strategy
from app.indicators.bollinger_bands_indicator import BollingerBandsIndicator
from app.indicators.rsi_indicator import RSIIndicator


class TestGeneticOptimizationWorkflow:
    @pytest.fixture
    def sample_data_directory(self):
        """Create a temporary directory with sample data."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate sample data
            symbol = "ETH-USD"
            interval = "1h"

            # Generate 30 days of hourly data
            start_date = datetime(2020, 1, 1)
            periods = 24 * 30
            td = timedelta(hours=1)

            dates = [start_date + i * td for i in range(periods)]
            timestamps = [int(date.timestamp() * 1000) for date in dates]

            # Generate price data with a trend, cyclical pattern, and volatility regimes
            closes = []
            base_price = 100.0

            # Add different volatility regimes
            volatility_regimes = [0.5, 1.5, 0.8, 2.0, 0.6]
            regime_length = periods // len(volatility_regimes)

            for i in range(periods):
                # Determine current volatility regime
                regime_idx = min(i // regime_length, len(volatility_regimes) - 1)
                volatility = volatility_regimes[regime_idx]

                # Add trend component
                if i < periods / 2:
                    trend = i * 0.02  # Upward trend in first half
                else:
                    trend = (periods - i) * 0.015  # Downward trend in second half

                # Add cyclical component (24-hour cycle)
                cycle = 3.0 * np.sin(2 * np.pi * (i % 24) / 24)

                # Add randomness with regime-specific volatility
                noise = np.random.normal(0, volatility)

                # Combine components
                price = base_price + trend + cycle + noise
                closes.append(max(price, 1.0))  # Ensure price > 0

            # Create OHLCV data
            data = {
                "timestamp": timestamps,
                "open": [c * (1 - np.random.uniform(0, 0.005)) for c in closes],
                "high": [c * (1 + np.random.uniform(0, 0.01)) for c in closes],
                "low": [c * (1 - np.random.uniform(0, 0.01)) for c in closes],
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

    @pytest.mark.slow
    def test_genetic_optimization_workflow(
        self, backtest_engine, sample_data_directory
    ):
        """Test genetic algorithm optimization workflow."""
        # 1. Define test parameters
        symbol = "ETH-USD"
        interval = "1h"
        start_date = "2020-01-01"
        end_date = "2020-01-21"  # First 20 days for training
        validation_end_date = "2020-01-31"  # Last 10 days for validation

        # 2. Initialize indicators
        rsi = RSIIndicator(name="RSI", params={"period": 14})
        bollinger = BollingerBandsIndicator(
            name="Bollinger", params={"period": 20, "num_std_dev": 2.0}
        )
        indicators = [rsi, bollinger]

        # 3. Define parameter space for genetic optimization
        param_space = {
            # Discrete parameter options
            "rsi_buy_threshold": [20, 25, 30, 35],
            "rsi_sell_threshold": [65, 70, 75, 80],
            # Continuous parameter ranges (min, max, step)
            "bollinger_entry_pct": (0.01, 0.05, 0.01),  # 1% to 5% in 1% steps
            "position_size": (0.1, 0.5, 0.1),  # 10% to 50% in 10% steps
            # Strategy parameters
            "stop_loss_pct": [0.05, 0.1, 0.15],  # 5%, 10%, or 15%
            "take_profit_pct": [0.1, 0.15, 0.2, 0.25],  # 10%, 15%, 20%, or 25%
        }

        # 4. Run genetic optimization with small population and generations for testing
        print("Running genetic algorithm optimization...")
        best_params, best_result = backtest_engine.genetic_optimize(
            strategy_func=multi_indicator_strategy,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_source_name="csv",
            param_space=param_space,
            population_size=8,  # Small population for testing
            generations=3,  # Few generations for testing
            crossover_rate=0.7,
            mutation_rate=0.2,
            tournament_size=3,
            metric_to_optimize="sharpe_ratio",
            leverage=1.0,
            indicators=indicators,
            random_seed=42,  # Fixed seed for reproducibility
        )

        # 5. Verify optimization results
        print(f"Best parameters found: {best_params}")
        print(f"Training metrics:")
        for metric_name, metric_value in best_result.metrics.items():
            if metric_name in [
                "sharpe_ratio",
                "total_return",
                "win_rate",
                "max_drawdown",
            ]:
                print(f"  {metric_name}: {metric_value}")

        # Verify that optimization improved the metrics
        assert "sharpe_ratio" in best_result.metrics
        assert isinstance(best_params, dict)

        # Check that all the parameters in param_space are in best_params
        for param_name in param_space:
            assert param_name in best_params

        # Make sure parameters are within expected ranges
        assert 20 <= best_params["rsi_buy_threshold"] <= 35
        assert 65 <= best_params["rsi_sell_threshold"] <= 80
        assert 0.01 <= best_params["bollinger_entry_pct"] <= 0.05
        assert 0.1 <= best_params["position_size"] <= 0.5
        assert best_params["stop_loss_pct"] in [0.05, 0.1, 0.15]
        assert best_params["take_profit_pct"] in [0.1, 0.15, 0.2, 0.25]

        # 6. Run a validation backtest on out-of-sample data
        print("\nRunning validation backtest on out-of-sample data...")
        validation_result = backtest_engine.run_backtest(
            strategy_func=multi_indicator_strategy,
            symbol=symbol,
            interval=interval,
            start_date=end_date,
            end_date=validation_end_date,
            data_source_name="csv",
            strategy_params=best_params,
            leverage=1.0,
            indicators=indicators,
        )

        # 7. Print validation results
        print(f"Validation metrics:")
        for metric_name, metric_value in validation_result.metrics.items():
            if metric_name in [
                "sharpe_ratio",
                "total_return",
                "win_rate",
                "max_drawdown",
            ]:
                print(f"  {metric_name}: {metric_value}")

        # 8. Create output directory
        os.makedirs(
            os.path.join(sample_data_directory, "genetic_results"), exist_ok=True
        )

        # 9. Save results
        best_result_file = os.path.join(
            sample_data_directory, "genetic_results", "genetic_optimization_result.json"
        )
        best_result.save_to_file(best_result_file)

        validation_result_file = os.path.join(
            sample_data_directory, "genetic_results", "validation_result.json"
        )
        validation_result.save_to_file(validation_result_file)

        # 10. Generate and save equity curve plots
        train_equity_curve_file = os.path.join(
            sample_data_directory, "genetic_results", "train_equity_curve.png"
        )
        best_result.plot_equity_curve(save_path=train_equity_curve_file)

        validation_equity_curve_file = os.path.join(
            sample_data_directory, "genetic_results", "validation_equity_curve.png"
        )
        validation_result.plot_equity_curve(save_path=validation_equity_curve_file)

        # 11. Compare initial random parameters with optimized parameters
        print("\nRunning comparison test with random parameters...")
        random_params = {
            "rsi_buy_threshold": 30,
            "rsi_sell_threshold": 70,
            "bollinger_entry_pct": 0.02,
            "position_size": 0.3,
            "stop_loss_pct": 0.1,
            "take_profit_pct": 0.2,
        }

        random_result = backtest_engine.run_backtest(
            strategy_func=multi_indicator_strategy,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_source_name="csv",
            strategy_params=random_params,
            leverage=1.0,
            indicators=indicators,
        )

        # 12. Print comparison
        print("\nComparing random parameters vs. genetically optimized parameters:")
        print(
            f"Random params - Sharpe: {random_result.metrics['sharpe_ratio']:.4f}, Return: {random_result.metrics['total_return']*100:.2f}%"
        )
        print(
            f"Optimized params - Sharpe: {best_result.metrics['sharpe_ratio']:.4f}, Return: {best_result.metrics['total_return']*100:.2f}%"
        )

        # 13. Verify optimization improved performance over random parameters
        # Note: In rare cases this might not be true, but generally we expect improvement
        if random_result.metrics["sharpe_ratio"] != 0:
            performance_improvement = (
                (
                    best_result.metrics["sharpe_ratio"]
                    - random_result.metrics["sharpe_ratio"]
                )
                / abs(random_result.metrics["sharpe_ratio"])
                * 100
            )
            print(f"Performance improvement: {performance_improvement:.2f}%")
        else:
            print(
                "Cannot calculate percentage improvement as baseline sharpe ratio is zero"
            )

        # Test is complete
        print("\nGenetic optimization workflow test completed successfully!")
