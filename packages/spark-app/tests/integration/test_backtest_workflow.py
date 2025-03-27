import json
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
from app.backtesting.strategy import (
    bollinger_bands_mean_reversion_strategy,
    macd_strategy,
    multi_indicator_strategy,
    rsi_strategy,
    simple_moving_average_crossover_strategy,
)


class TestBacktestWorkflow:
    @pytest.fixture
    def sample_data_directory(self):
        """Create a temporary directory with sample data."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate sample data for multiple symbols and intervals
            symbols = ["BTC-USD", "ETH-USD"]
            intervals = ["1h", "1d"]

            for symbol in symbols:
                for interval in intervals:
                    # Generate sample data
                    start_date = datetime(2020, 1, 1)
                    if interval == "1h":
                        periods = 24 * 90  # 90 days of hourly data
                        td = timedelta(hours=1)
                    else:
                        periods = 365  # 365 days of daily data
                        td = timedelta(days=1)

                    dates = [start_date + i * td for i in range(periods)]
                    timestamps = [int(date.timestamp() * 1000) for date in dates]

                    # Generate price data with a trend and cyclical pattern
                    closes = []
                    base_price = 100.0 if symbol == "ETH-USD" else 10000.0

                    for i in range(periods):
                        # Add trend
                        trend = i * 0.01

                        # Add cyclical component
                        cycle = 5.0 * np.sin(i / 20.0)

                        # Add randomness
                        noise = np.random.normal(0, 1.0)

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
                        "volume": [
                            np.random.uniform(1000, 10000) for _ in range(periods)
                        ],
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

    def test_complete_backtest_workflow(self, backtest_engine, sample_data_directory):
        """Test a complete backtesting workflow."""
        # 1. Define test parameters
        symbol = "ETH-USD"
        interval = "1d"
        start_date = "2020-01-01"
        end_date = "2020-12-01"

        # 2. Run backtests with different strategies
        strategies = {
            "Moving Average Crossover": simple_moving_average_crossover_strategy,
            "Bollinger Bands": bollinger_bands_mean_reversion_strategy,
            "RSI": rsi_strategy,
            "MACD": macd_strategy,
            "Multi-Indicator": multi_indicator_strategy,
        }

        results = {}

        for name, strategy in strategies.items():
            # Run the backtest
            print(f"Running backtest with {name} strategy...")
            result = backtest_engine.run_backtest(
                strategy_func=strategy,
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                data_source_name="csv",
            )

            # Store the result
            results[name] = result

            # Verify the result has expected attributes
            assert hasattr(result, "metrics")
            assert hasattr(result, "equity_curve")
            assert hasattr(result, "trades")
            assert hasattr(result, "symbol")

            # Verify the result has expected metrics
            assert "total_trades" in result.metrics
            assert "sharpe_ratio" in result.metrics
            assert "max_drawdown" in result.metrics

            # Save results to file
            os.makedirs(os.path.join(sample_data_directory, "results"), exist_ok=True)
            result_file = os.path.join(
                sample_data_directory,
                "results",
                f"{name.lower().replace(' ', '_')}_result.json",
            )
            result.save_to_file(result_file)

            # Verify the file was created
            assert os.path.exists(result_file)

            # Test loading the result from file
            loaded_result = BacktestResult.load_from_file(result_file)
            assert loaded_result.symbol == result.symbol
            assert (
                loaded_result.metrics["total_trades"] == result.metrics["total_trades"]
            )

        # 3. Compare strategy performance
        print("\nStrategy Performance Comparison:")
        for name, result in results.items():
            print(
                f"{name} - Sharpe Ratio: {result.metrics['sharpe_ratio']:.4f}, Return: {result.metrics['total_return']*100:.2f}%, Max Drawdown: {result.metrics['max_drawdown']*100:.2f}%"
            )

        # 4. Optimize parameters for the best performing strategy
        # For demonstration, we'll optimize the moving average strategy
        param_grid = {
            "fast_period": [5, 10, 15],
            "slow_period": [20, 30, 40],
            "position_size": [0.2, 0.5, 0.8],
        }

        print("\nOptimizing Moving Average Crossover strategy parameters...")
        best_params, best_result = backtest_engine.optimize_parameters(
            strategy_func=simple_moving_average_crossover_strategy,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_source_name="csv",
            param_grid=param_grid,
            metric_to_optimize="sharpe_ratio",
        )

        print(f"Best parameters found: {best_params}")
        print(f"Best Sharpe Ratio: {best_result.metrics['sharpe_ratio']:.4f}")

        # 5. Perform walk-forward analysis
        print("\nPerforming walk-forward analysis...")
        wfa_results = backtest_engine.walk_forward_analysis(
            strategy_func=simple_moving_average_crossover_strategy,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_source_name="csv",
            param_grid={
                "fast_period": [5, 10],
                "slow_period": [20, 30],
                "position_size": [0.5],
            },
            train_size=3,  # 3 months training
            test_size=1,  # 1 month testing
            metric_to_optimize="sharpe_ratio",
        )

        # Verify walk-forward analysis results
        assert isinstance(wfa_results, list)
        if (
            wfa_results
        ):  # There should be at least one window if the date range is sufficient
            for window in wfa_results:
                assert "window" in window
                assert "train_metrics" in window
                assert "test_metrics" in window
                assert "best_params" in window

        # 6. Save walk-forward analysis results
        wfa_file = os.path.join(
            sample_data_directory, "results", "walk_forward_analysis.json"
        )
        with open(wfa_file, "w") as f:
            json.dump(wfa_results, f, indent=2)

        # 7. Create equity curve plot for the best strategy
        best_strategy_name = max(
            results, key=lambda k: results[k].metrics["sharpe_ratio"]
        )
        best_strategy_result = results[best_strategy_name]

        plot_file = os.path.join(sample_data_directory, "results", "equity_curve.png")
        best_strategy_result.plot_equity_curve(save_path=plot_file)

        # Verify the plot was created
        assert os.path.exists(plot_file)

        # 8. Plot drawdown
        drawdown_file = os.path.join(sample_data_directory, "results", "drawdown.png")
        best_strategy_result.plot_drawdown(save_path=drawdown_file)

        # Verify the plot was created
        assert os.path.exists(drawdown_file)

        print("\nBacktest workflow completed successfully!")
