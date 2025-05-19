import os

import numpy as np
import pandas as pd
import pytest
from app.backtesting.backtest_engine import BacktestEngine
from app.backtesting.strategy import simple_moving_average_crossover_strategy
from app.connectors.base_connector import OrderSide


class TestBacktestEngine:
    def test_backtest_engine_initialization(self, backtest_env):
        """Test that BacktestEngine initializes correctly."""
        # Unpack the backtest environment fixture
        engine, data_manager, _, _ = backtest_env

        # Check attributes
        assert engine.data_manager == data_manager
        assert engine.initial_balance == {"USD": 10000.0}
        assert engine.maker_fee == 0.001
        assert engine.taker_fee == 0.002
        assert engine.slippage_model == "fixed"

    def test_run_backtest(self, backtest_env):
        """Test running a backtest with a simple strategy."""
        # Unpack the backtest environment fixture
        engine, _, symbol, interval = backtest_env

        # Run a backtest with moving average crossover strategy
        result = engine.run_backtest(
            strategy_func=simple_moving_average_crossover_strategy,
            symbol=symbol,
            interval=interval,
            start_date="2020-01-01",
            end_date="2020-04-10",
            data_source_name="csv",
            strategy_params={
                "fast_period": 10,
                "slow_period": 30,
                "position_size": 0.5,
            },
        )

        # Verify the result
        assert result is not None
        assert result.symbol == symbol
        assert isinstance(result.metrics, dict)
        assert isinstance(result.equity_curve, pd.DataFrame)
        assert isinstance(result.trades, list)

    def test_backtest_results_metrics(self, backtest_env):
        """Test that backtest results contain the expected metrics."""
        # Unpack the backtest environment fixture
        engine, _, symbol, interval = backtest_env

        # Run a backtest with moving average crossover strategy
        result = engine.run_backtest(
            strategy_func=simple_moving_average_crossover_strategy,
            symbol=symbol,
            interval=interval,
            start_date="2020-01-01",
            end_date="2020-04-10",
            data_source_name="csv",
            strategy_params={
                "fast_period": 10,
                "slow_period": 30,
                "position_size": 0.5,
            },
        )

        # Check for expected metrics
        expected_metrics = [
            "total_trades",
            "winning_trades",
            "losing_trades",
            "win_rate",
            "avg_profit",
            "avg_loss",
            "max_profit",
            "max_loss",
            "profit_factor",
            "total_return",
            "annualized_return",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
        ]

        for metric in expected_metrics:
            assert metric in result.metrics

    def test_drawdown_calculation(self, backtest_env):
        """Test drawdown calculation."""
        # Unpack the backtest environment fixture
        engine, _, symbol, interval = backtest_env

        # Run a backtest
        result = engine.run_backtest(
            strategy_func=simple_moving_average_crossover_strategy,
            symbol=symbol,
            interval=interval,
            start_date="2020-01-01",
            end_date="2020-04-10",
            data_source_name="csv",
        )

        # Check that drawdown columns exist in equity curve
        assert "drawdown" in result.equity_curve.columns
        assert "drawdown_abs" in result.equity_curve.columns

        # Verify drawdown calculations
        assert all(0 <= dd <= 1.0 for dd in result.equity_curve["drawdown"])
        assert all(dd >= 0 for dd in result.equity_curve["drawdown_abs"])

        # Verify max drawdown in metrics
        assert 0 <= result.metrics["max_drawdown"] <= 1.0

    def test_parameter_optimization(self, backtest_env):
        """Test parameter optimization functionality."""
        # Unpack the backtest environment fixture
        engine, _, symbol, interval = backtest_env

        # Define parameter grid
        param_grid = {
            "fast_period": [5, 10],
            "slow_period": [20, 30],
            "position_size": [0.3, 0.5],
        }

        # Run parameter optimization
        best_params, best_result = engine.optimize_parameters(
            strategy_func=simple_moving_average_crossover_strategy,
            symbol=symbol,
            interval=interval,
            start_date="2020-01-01",
            end_date="2020-04-10",
            data_source_name="csv",
            param_grid=param_grid,
            metric_to_optimize="sharpe_ratio",
        )

        # Verify results
        assert isinstance(best_params, dict)
        assert set(best_params.keys()) == {
            "fast_period",
            "slow_period",
            "position_size",
        }
        assert isinstance(
            best_result,
            type(
                engine.run_backtest(
                    strategy_func=simple_moving_average_crossover_strategy,
                    symbol=symbol,
                    interval=interval,
                    start_date="2020-01-01",
                    end_date="2020-04-10",
                    data_source_name="csv",
                )
            ),
        )

    def test_equity_calculation(self, backtest_env):
        """Test equity calculation during backtest."""
        # Unpack the backtest environment fixture
        engine, _, symbol, interval = backtest_env

        # Run a backtest
        result = engine.run_backtest(
            strategy_func=simple_moving_average_crossover_strategy,
            symbol=symbol,
            interval=interval,
            start_date="2020-01-01",
            end_date="2020-04-10",
            data_source_name="csv",
            strategy_params={
                "fast_period": 10,
                "slow_period": 30,
                "position_size": 0.5,
            },
        )

        # Verify equity curve
        assert "equity" in result.equity_curve.columns
        assert len(result.equity_curve) > 0

        # First equity value should equal initial balance
        assert abs(result.equity_curve["equity"].iloc[0] - 10000.0) < 1e-6

    def test_genetic_optimization(self, backtest_env):
        """Test genetic algorithm optimization functionality."""
        # Unpack the backtest environment fixture
        engine, _, symbol, interval = backtest_env

        # Define parameter space
        param_space = {
            "fast_period": [5, 10, 15],  # Discrete options
            "slow_period": (20, 40, 5),  # Range (min, max, step)
            "position_size": [0.3, 0.5, 0.7],  # Discrete options
        }

        # Run genetic optimization with small population and generations for testing
        best_params, best_result = engine.genetic_optimize(
            strategy_func=simple_moving_average_crossover_strategy,
            symbol=symbol,
            interval=interval,
            start_date="2020-01-01",
            end_date="2020-04-10",
            data_source_name="csv",
            param_space=param_space,
            population_size=5,
            generations=2,
            metric_to_optimize="sharpe_ratio",
            random_seed=42,  # For reproducible tests
        )

        # Verify results
        assert isinstance(best_params, dict)
        assert set(best_params.keys()) == {
            "fast_period",
            "slow_period",
            "position_size",
        }

        # Check parameter bounds
        assert best_params["fast_period"] in [5, 10, 15]
        assert 20 <= best_params["slow_period"] <= 40
        assert best_params["position_size"] in [0.3, 0.5, 0.7]

        # Verify best result is a valid BacktestResult
        assert isinstance(
            best_result,
            type(
                engine.run_backtest(
                    strategy_func=simple_moving_average_crossover_strategy,
                    symbol=symbol,
                    interval=interval,
                    start_date="2020-01-01",
                    end_date="2020-04-10",
                    data_source_name="csv",
                )
            ),
        )

        # Ensure sharpe ratio is calculated
        assert "sharpe_ratio" in best_result.metrics
