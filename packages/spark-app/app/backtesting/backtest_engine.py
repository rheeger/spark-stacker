import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..connectors.base_connector import OrderSide, OrderType
from ..indicators.base_indicator import Signal, SignalDirection
from .data_manager import DataManager
from .simulation_engine import SimulationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest results and performance metrics."""

    def __init__(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        initial_balance: Dict[str, float],
        final_balance: Dict[str, float],
        trades: List[Dict[str, Any]],
        equity_curve: pd.DataFrame,
        metrics: Dict[str, Any],
    ):
        """
        Initialize backtest results.

        Args:
            symbol: Tested symbol
            start_date: Backtest start date
            end_date: Backtest end date
            initial_balance: Starting account balance
            final_balance: Ending account balance
            trades: List of executed trades
            equity_curve: DataFrame with equity values over time
            metrics: Performance metrics
        """
        self.symbol = symbol

        # Convert dates to datetime if needed
        if isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            self.start_date = start_date

        if isinstance(end_date, str):
            self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            self.end_date = end_date

        self.initial_balance = initial_balance
        self.final_balance = final_balance
        self.trades = trades
        self.equity_curve = equity_curve
        self.metrics = metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        # Convert DataFrame to serializable format
        equity_curve_dict = []
        for _, row in self.equity_curve.iterrows():
            row_dict = {}
            for col, val in row.items():
                # Convert Timestamp to string or int
                if pd.api.types.is_datetime64_any_dtype(val) or isinstance(
                    val, pd.Timestamp
                ):
                    row_dict[col] = val.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    row_dict[col] = val
            equity_curve_dict.append(row_dict)

        return {
            "symbol": self.symbol,
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "trades": self.trades,
            "equity_curve": equity_curve_dict,
            "metrics": self.metrics,
        }

    def save_to_file(self, file_path: str) -> None:
        """
        Save results to JSON file.

        Args:
            file_path: Path to save results
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved backtest results to {file_path}")

    def plot_equity_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot equity curve.

        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.equity_curve["timestamp"], self.equity_curve["equity"], label="Equity"
        )

        # Format dates on x-axis
        plt.gcf().autofmt_xdate()

        plt.title(f"Equity Curve - {self.symbol}")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.grid(True)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Saved equity curve plot to {save_path}")
        else:
            plt.show()

    def plot_drawdown(self, save_path: Optional[str] = None) -> None:
        """
        Plot drawdown curve.

        Args:
            save_path: Optional path to save the plot
        """
        if "drawdown" not in self.equity_curve.columns:
            logger.warning("Drawdown data not available in equity curve")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(
            self.equity_curve["timestamp"],
            self.equity_curve["drawdown"] * 100,
            label="Drawdown %",
            color="red",
        )

        # Format dates on x-axis
        plt.gcf().autofmt_xdate()

        plt.title(f"Drawdown - {self.symbol}")
        plt.xlabel("Date")
        plt.ylabel("Drawdown %")
        plt.legend()
        plt.grid(True)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Saved drawdown plot to {save_path}")
        else:
            plt.show()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestResult":
        """Create a BacktestResult from a dictionary."""
        return cls(
            symbol=data["symbol"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            initial_balance=data["initial_balance"],
            final_balance=data["final_balance"],
            trades=data["trades"],
            equity_curve=pd.DataFrame(data["equity_curve"]),
            metrics=data["metrics"],
        )

    @classmethod
    def load_from_file(cls, file_path: str) -> "BacktestResult":
        """Load results from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class BacktestEngine:
    """
    Engine for running backtests and calculating performance metrics.
    """

    def __init__(
        self,
        data_manager: DataManager,
        initial_balance: Dict[str, float] = {"USD": 10000.0},
        maker_fee: float = 0.0001,  # 0.01%
        taker_fee: float = 0.0005,  # 0.05%
        slippage_model: str = "random",
        trading_days_per_year: int = 365,
    ):
        """
        Initialize the backtest engine.

        Args:
            data_manager: DataManager instance
            initial_balance: Starting balances for each asset
            maker_fee: Fee for maker orders
            taker_fee: Fee for taker orders
            slippage_model: Slippage model to use
            trading_days_per_year: Number of trading days per year
        """
        self.data_manager = data_manager
        self.initial_balance = initial_balance
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_model = slippage_model
        self.trading_days_per_year = trading_days_per_year

    def run_backtest(
        self,
        strategy_func: Callable[[pd.DataFrame, SimulationEngine, Dict[str, Any]], None],
        symbol: str,
        interval: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        data_source_name: str,
        strategy_params: Dict[str, Any] = None,
        leverage: float = 1.0,
        indicators: List[Any] = None,
    ) -> BacktestResult:
        """
        Run a backtest for a given strategy.

        Args:
            strategy_func: Function that implements the trading strategy
            symbol: Market symbol to backtest
            interval: Timeframe interval (e.g., '1m', '1h', '1d')
            start_date: Start date for backtest
            end_date: End date for backtest
            data_source_name: Name of the data source to use
            strategy_params: Parameters for the strategy
            leverage: Default leverage to use
            indicators: List of indicator instances to use

        Returns:
            BacktestResult with performance metrics
        """
        # Convert dates to timestamps if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Convert to milliseconds
        start_time_ms = int(start_date.timestamp() * 1000)
        end_time_ms = int(end_date.timestamp() * 1000)

        # Get historical data
        logger.info(
            f"Fetching historical data for {symbol} from {start_date} to {end_date}"
        )
        historical_data = self.data_manager.get_data(
            source_name=data_source_name,
            symbol=symbol,
            interval=interval,
            start_time=start_time_ms,
            end_time=end_time_ms,
        )

        # Clean data
        historical_data = self.data_manager.clean_data(historical_data)

        if historical_data.empty:
            logger.error(
                f"No historical data available for {symbol} in the specified date range"
            )
            raise ValueError(
                f"No historical data available for {symbol} in the specified date range"
            )

        logger.info(f"Running backtest with {len(historical_data)} candles")

        # Initialize simulation engine
        simulation_engine = SimulationEngine(
            initial_balance=self.initial_balance.copy(),
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            slippage_model=self.slippage_model,
        )

        # Prepare indicators if provided
        indicator_values = {}
        if indicators:
            for indicator in indicators:
                # Calculate indicator values for the entire dataset
                indicator_data = indicator.calculate(historical_data)
                # Store indicator columns in a dictionary
                non_ohlcv_columns = [
                    col
                    for col in indicator_data.columns
                    if col
                    not in ["timestamp", "open", "high", "low", "close", "volume"]
                ]
                indicator_values[indicator.name] = indicator_data[non_ohlcv_columns]

        # Run simulation
        equity_data = []

        for i in range(len(historical_data)):
            # Get current candle
            current_candle = historical_data.iloc[i]
            candle_dict = {symbol: current_candle}

            # Execute any open orders with current candle data
            for order in simulation_engine.get_orders():
                if order.status == "OPEN":
                    simulation_engine.execute_order(order, current_candle)

            # Update positions with current market data
            simulation_engine.update_positions(candle_dict)

            # Call strategy function
            if strategy_params is None:
                strategy_params = {}

            # Add indicator values for current candle to strategy params
            if indicators:
                for indicator_name, values in indicator_values.items():
                    if i < len(values):
                        for col in values.columns:
                            strategy_params[f"{indicator_name}_{col}"] = values.iloc[i][
                                col
                            ]

            # Call the strategy function
            strategy_func(
                historical_data.iloc[: i + 1].copy(),
                simulation_engine,
                {
                    "symbol": symbol,
                    "current_candle": current_candle,
                    "leverage": leverage,
                    **strategy_params,
                },
            )

            # Record equity for this timestamp
            current_prices = {symbol: current_candle["close"]}
            equity = simulation_engine.calculate_equity(current_prices)

            equity_data.append(
                {
                    "timestamp": current_candle["timestamp"],
                    "equity": equity,
                    "close_price": current_candle["close"],
                }
            )

            # Log progress every 10% of the simulation
            if i % max(1, len(historical_data) // 10) == 0:
                progress = (i / len(historical_data)) * 100
                logger.info(
                    f"Simulation progress: {progress:.1f}% - Current equity: {equity:.2f}"
                )

        # Calculate performance metrics
        equity_df = pd.DataFrame(equity_data)
        equity_df["datetime"] = pd.to_datetime(equity_df["timestamp"], unit="ms")
        equity_df = self._calculate_drawdown(equity_df)

        metrics = self._calculate_metrics(
            equity_df=equity_df,
            trades=simulation_engine.get_trade_history(),
            initial_balance=sum(self.initial_balance.values()),
            final_balance=sum(simulation_engine.get_balance().values()),
        )

        # Create and return backtest result
        result = BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=simulation_engine.get_balance(),
            trades=simulation_engine.get_trade_history(),
            equity_curve=equity_df,
            metrics=metrics,
        )

        return result

    def _calculate_drawdown(self, equity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate drawdown for equity curve.

        Args:
            equity_df: DataFrame with equity values

        Returns:
            DataFrame with added drawdown column
        """
        # Calculate rolling maximum equity
        equity_df["peak"] = equity_df["equity"].cummax()

        # Calculate drawdown in absolute and percentage terms
        equity_df["drawdown_abs"] = equity_df["peak"] - equity_df["equity"]
        equity_df["drawdown"] = equity_df["drawdown_abs"] / equity_df["peak"]

        return equity_df

    def _calculate_metrics(
        self,
        equity_df: pd.DataFrame,
        trades: List[Dict[str, Any]],
        initial_balance: float,
        final_balance: float,
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics.

        Args:
            equity_df: DataFrame with equity values
            trades: List of executed trades
            initial_balance: Starting account balance
            final_balance: Ending account balance

        Returns:
            Dictionary with performance metrics
        """
        # Initialize metrics dictionary
        metrics = {}

        # Basic metrics
        metrics["total_trades"] = len(trades)

        # Calculate winning and losing trades
        winning_trades = [
            t for t in trades if "realized_pnl" in t and t["realized_pnl"] > 0
        ]
        losing_trades = [
            t for t in trades if "realized_pnl" in t and t["realized_pnl"] < 0
        ]

        metrics["winning_trades"] = len(winning_trades)
        metrics["losing_trades"] = len(losing_trades)

        # Win rate
        if metrics["total_trades"] > 0:
            metrics["win_rate"] = metrics["winning_trades"] / metrics["total_trades"]
        else:
            metrics["win_rate"] = 0.0

        # Calculate profit and loss metrics
        if winning_trades:
            metrics["avg_profit"] = sum(
                t["realized_pnl"] for t in winning_trades
            ) / len(winning_trades)
            metrics["max_profit"] = max(t["realized_pnl"] for t in winning_trades)
        else:
            metrics["avg_profit"] = 0.0
            metrics["max_profit"] = 0.0

        if losing_trades:
            metrics["avg_loss"] = sum(t["realized_pnl"] for t in losing_trades) / len(
                losing_trades
            )
            metrics["max_loss"] = min(t["realized_pnl"] for t in losing_trades)
        else:
            metrics["avg_loss"] = 0.0
            metrics["max_loss"] = 0.0

        # Profit factor (ratio of gross profit to gross loss)
        gross_profit = sum(t["realized_pnl"] for t in winning_trades)
        gross_loss = abs(sum(t["realized_pnl"] for t in losing_trades))

        if gross_loss > 0:
            metrics["profit_factor"] = gross_profit / gross_loss
        else:
            metrics["profit_factor"] = float("inf") if gross_profit > 0 else 0.0

        # Calculate returns
        metrics["total_return"] = (
            (final_balance - initial_balance) / initial_balance
            if initial_balance > 0
            else 0.0
        )

        # Calculate annualized return
        days = (equity_df["datetime"].max() - equity_df["datetime"].min()).days
        if days > 0:
            years = days / 365
            metrics["annualized_return"] = (1 + metrics["total_return"]) ** (
                1 / years
            ) - 1
        else:
            metrics["annualized_return"] = 0.0

        # Calculate maximum drawdown
        metrics["max_drawdown"] = equity_df["drawdown"].max()
        metrics["max_drawdown_abs"] = equity_df["drawdown_abs"].max()

        # Calculate Sharpe ratio (annualized)
        if len(equity_df) > 1:
            # Calculate daily returns
            equity_df["daily_return"] = equity_df["equity"].pct_change()

            # Annualized Sharpe ratio
            daily_returns = equity_df["daily_return"].dropna()
            if len(daily_returns) > 0 and daily_returns.std() > 0:
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(
                    self.trading_days_per_year
                )
                metrics["sharpe_ratio"] = sharpe_ratio
            else:
                metrics["sharpe_ratio"] = 0.0
        else:
            metrics["sharpe_ratio"] = 0.0

        # Calculate Sortino ratio (annualized)
        if len(equity_df) > 1:
            # Calculate daily returns
            daily_returns = equity_df["daily_return"].dropna()

            # Calculate downside deviation (std dev of negative returns)
            negative_returns = daily_returns[daily_returns < 0]

            if len(negative_returns) > 0 and negative_returns.std() > 0:
                sortino_ratio = (
                    daily_returns.mean() / negative_returns.std()
                ) * np.sqrt(self.trading_days_per_year)
                metrics["sortino_ratio"] = sortino_ratio
            else:
                metrics["sortino_ratio"] = (
                    float("inf") if daily_returns.mean() > 0 else 0.0
                )
        else:
            metrics["sortino_ratio"] = 0.0

        # Calculate Calmar ratio (annualized return / maximum drawdown)
        if metrics["max_drawdown"] > 0:
            metrics["calmar_ratio"] = (
                metrics["annualized_return"] / metrics["max_drawdown"]
            )
        else:
            metrics["calmar_ratio"] = (
                float("inf") if metrics["annualized_return"] > 0 else 0.0
            )

        return metrics

    def optimize_parameters(
        self,
        strategy_func: Callable[[pd.DataFrame, SimulationEngine, Dict[str, Any]], None],
        symbol: str,
        interval: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        data_source_name: str,
        param_grid: Dict[str, List[Any]],
        metric_to_optimize: str = "sharpe_ratio",
        leverage: float = 1.0,
        indicators: List[Any] = None,
    ) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        Optimize strategy parameters using grid search.

        Args:
            strategy_func: Function that implements the trading strategy
            symbol: Market symbol to backtest
            interval: Timeframe interval (e.g., '1m', '1h', '1d')
            start_date: Start date for backtest
            end_date: End date for backtest
            data_source_name: Name of the data source to use
            param_grid: Dictionary of parameter names to lists of values
            metric_to_optimize: Metric to optimize for
            leverage: Default leverage to use
            indicators: List of indicator instances to use

        Returns:
            Tuple of (best_parameters, best_result)
        """
        # Generate all parameter combinations
        import itertools

        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        logger.info(
            f"Running parameter optimization with {len(param_combinations)} combinations"
        )

        best_value = float("-inf")
        best_params = None
        best_result = None

        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_keys, combination))

            logger.info(
                f"Testing combination {i+1}/{len(param_combinations)}: {params}"
            )

            try:
                # Run backtest with these parameters
                result = self.run_backtest(
                    strategy_func=strategy_func,
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    data_source_name=data_source_name,
                    strategy_params=params,
                    leverage=leverage,
                    indicators=indicators,
                )

                # Check if this result is better than current best
                current_value = result.metrics.get(metric_to_optimize, float("-inf"))

                if current_value > best_value:
                    best_value = current_value
                    best_params = params
                    best_result = result

                    logger.info(
                        f"New best parameters found: {best_params} with {metric_to_optimize} = {best_value}"
                    )

            except Exception as e:
                logger.error(f"Error testing parameters {params}: {e}")

        if best_params is None:
            raise ValueError("No valid parameter combination found")

        return best_params, best_result

    def walk_forward_analysis(
        self,
        strategy_func: Callable[[pd.DataFrame, SimulationEngine, Dict[str, Any]], None],
        symbol: str,
        interval: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        data_source_name: str,
        param_grid: Dict[str, List[Any]],
        train_size: int = 6,  # months
        test_size: int = 2,  # months
        metric_to_optimize: str = "sharpe_ratio",
        leverage: float = 1.0,
        indicators: List[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform walk-forward analysis.

        Args:
            strategy_func: Function that implements the trading strategy
            symbol: Market symbol to backtest
            interval: Timeframe interval (e.g., '1m', '1h', '1d')
            start_date: Start date for backtest
            end_date: End date for backtest
            data_source_name: Name of the data source to use
            param_grid: Dictionary of parameter names to lists of values
            train_size: Size of training window in months
            test_size: Size of testing window in months
            metric_to_optimize: Metric to optimize for
            leverage: Default leverage to use
            indicators: List of indicator instances to use

        Returns:
            List of dictionaries with walk-forward results
        """
        # Convert dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Generate time windows for walk-forward analysis
        from dateutil.relativedelta import relativedelta

        windows = []
        current_start = start_date

        while current_start < end_date:
            train_end = current_start + relativedelta(months=train_size)
            test_start = train_end
            test_end = test_start + relativedelta(months=test_size)

            # Adjust final window if it extends beyond end_date
            if test_end > end_date:
                test_end = end_date

            if test_start < end_date:  # Only add window if test period is valid
                windows.append(
                    {
                        "train_start": current_start,
                        "train_end": train_end,
                        "test_start": test_start,
                        "test_end": test_end,
                    }
                )

            current_start = test_end

        logger.info(f"Running walk-forward analysis with {len(windows)} windows")

        results = []

        for i, window in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")
            logger.info(
                f"Training period: {window['train_start']} to {window['train_end']}"
            )
            logger.info(
                f"Testing period: {window['test_start']} to {window['test_end']}"
            )

            try:
                # Optimize parameters on training data
                best_params, train_result = self.optimize_parameters(
                    strategy_func=strategy_func,
                    symbol=symbol,
                    interval=interval,
                    start_date=window["train_start"],
                    end_date=window["train_end"],
                    data_source_name=data_source_name,
                    param_grid=param_grid,
                    metric_to_optimize=metric_to_optimize,
                    leverage=leverage,
                    indicators=indicators,
                )

                logger.info(f"Best parameters found: {best_params}")

                # Test on out-of-sample data
                test_result = self.run_backtest(
                    strategy_func=strategy_func,
                    symbol=symbol,
                    interval=interval,
                    start_date=window["test_start"],
                    end_date=window["test_end"],
                    data_source_name=data_source_name,
                    strategy_params=best_params,
                    leverage=leverage,
                    indicators=indicators,
                )

                # Save results
                window_result = {
                    "window": i + 1,
                    "train_start": window["train_start"].strftime("%Y-%m-%d"),
                    "train_end": window["train_end"].strftime("%Y-%m-%d"),
                    "test_start": window["test_start"].strftime("%Y-%m-%d"),
                    "test_end": window["test_end"].strftime("%Y-%m-%d"),
                    "best_params": best_params,
                    "train_metrics": train_result.metrics,
                    "test_metrics": test_result.metrics,
                }

                results.append(window_result)

                logger.info(
                    f"Window {i+1} metrics - Train {metric_to_optimize}: {train_result.metrics[metric_to_optimize]:.4f}, Test {metric_to_optimize}: {test_result.metrics[metric_to_optimize]:.4f}"
                )

            except Exception as e:
                logger.error(f"Error processing window {i+1}: {e}")

        return results

    def genetic_optimize(
        self,
        strategy_func: Callable[[pd.DataFrame, SimulationEngine, Dict[str, Any]], None],
        symbol: str,
        interval: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        data_source_name: str,
        param_space: Dict[str, Union[List[Any], Tuple[float, float, float]]],
        population_size: int = 20,
        generations: int = 10,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.2,
        tournament_size: int = 3,
        metric_to_optimize: str = "sharpe_ratio",
        leverage: float = 1.0,
        indicators: List[Any] = None,
        random_seed: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        Optimize strategy parameters using a genetic algorithm.

        Args:
            strategy_func: Function that implements the trading strategy
            symbol: Market symbol to backtest
            interval: Timeframe interval (e.g., '1m', '1h', '1d')
            start_date: Start date for backtest
            end_date: End date for backtest
            data_source_name: Name of the data source to use
            param_space: Dictionary of parameter names to either:
                        - List of discrete values to choose from
                        - Tuple of (min, max, step) for continuous values
            population_size: Size of the population in each generation
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover (0.0 to 1.0)
            mutation_rate: Probability of mutation (0.0 to 1.0)
            tournament_size: Size of tournament for parent selection
            metric_to_optimize: Metric to optimize for
            leverage: Default leverage to use
            indicators: List of indicator instances to use
            random_seed: Optional seed for random number generation

        Returns:
            Tuple of (best_parameters, best_result)
        """
        import random
        from copy import deepcopy

        import numpy as np

        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        logger.info(
            f"Starting genetic algorithm optimization with population size {population_size} and {generations} generations"
        )

        # Helper functions for genetic algorithm
        def create_individual():
            """Create a random individual (set of parameters)"""
            individual = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, list):
                    # Select from discrete values
                    individual[param_name] = random.choice(param_range)
                elif isinstance(param_range, tuple) and len(param_range) == 3:
                    # Continuous range (min, max, step)
                    min_val, max_val, step = param_range
                    if (
                        isinstance(min_val, int)
                        and isinstance(max_val, int)
                        and isinstance(step, int)
                    ):
                        # Integer parameters
                        steps = int((max_val - min_val) / step) + 1
                        value = min_val + random.randint(0, steps - 1) * step
                        individual[param_name] = value
                    else:
                        # Float parameters
                        steps = int((max_val - min_val) / step) + 1
                        value = min_val + random.randint(0, steps - 1) * step
                        individual[param_name] = value
                else:
                    raise ValueError(
                        f"Invalid parameter range for {param_name}: {param_range}"
                    )
            return individual

        def evaluate_fitness(individual):
            """Evaluate fitness of an individual"""
            try:
                result = self.run_backtest(
                    strategy_func=strategy_func,
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    data_source_name=data_source_name,
                    strategy_params=individual,
                    leverage=leverage,
                    indicators=indicators,
                )

                # Return the metric to optimize
                return result.metrics.get(metric_to_optimize, float("-inf")), result
            except Exception as e:
                logger.error(f"Error evaluating individual {individual}: {e}")
                return float("-inf"), None

        def tournament_selection(population, fitnesses):
            """Select parent using tournament selection"""
            # Randomly select tournament_size individuals
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i][0] for i in tournament_indices]

            # Return the best individual from tournament
            best_idx = tournament_indices[
                tournament_fitnesses.index(max(tournament_fitnesses))
            ]
            return population[best_idx]

        def crossover(parent1, parent2):
            """Perform crossover between two parents"""
            if random.random() > crossover_rate:
                return deepcopy(parent1)

            child = {}
            for param_name in parent1.keys():
                # Randomly select from either parent
                if random.random() < 0.5:
                    child[param_name] = parent1[param_name]
                else:
                    child[param_name] = parent2[param_name]
            return child

        def mutate(individual):
            """Mutate an individual"""
            mutated = deepcopy(individual)

            for param_name, param_range in param_space.items():
                # Apply mutation with probability mutation_rate
                if random.random() < mutation_rate:
                    if isinstance(param_range, list):
                        # Select new random value from discrete options
                        mutated[param_name] = random.choice(param_range)
                    elif isinstance(param_range, tuple) and len(param_range) == 3:
                        # Continuous range (min, max, step)
                        min_val, max_val, step = param_range
                        if (
                            isinstance(min_val, int)
                            and isinstance(max_val, int)
                            and isinstance(step, int)
                        ):
                            # Integer parameters
                            steps = int((max_val - min_val) / step) + 1
                            value = min_val + random.randint(0, steps - 1) * step
                            mutated[param_name] = value
                        else:
                            # Float parameters
                            steps = int((max_val - min_val) / step) + 1
                            value = min_val + random.randint(0, steps - 1) * step
                            mutated[param_name] = value
            return mutated

        # Create initial population
        population = [create_individual() for _ in range(population_size)]

        # Initialize tracking for best individual
        best_individual = None
        best_fitness = float("-inf")
        best_result = None

        # Evolve over generations
        for generation in range(generations):
            logger.info(f"Generation {generation+1}/{generations}")

            # Evaluate fitness for current population
            fitness_results = []
            for i, individual in enumerate(population):
                logger.info(
                    f"Evaluating individual {i+1}/{population_size}: {individual}"
                )
                fitness, result = evaluate_fitness(individual)
                fitness_results.append((fitness, result))

                # Update best individual if needed
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual
                    best_result = result
                    logger.info(
                        f"New best individual found: {best_individual} with {metric_to_optimize} = {best_fitness}"
                    )

            # Early stopping check - if last generation, break the loop
            if generation == generations - 1:
                break

            # Create new population
            new_population = []

            # Elitism: Keep the best individual
            new_population.append(deepcopy(best_individual))

            # Fill the rest of the population with offspring
            while len(new_population) < population_size:
                # Select parents using tournament selection
                parent1 = tournament_selection(population, fitness_results)
                parent2 = tournament_selection(population, fitness_results)

                # Create offspring through crossover and mutation
                offspring = crossover(parent1, parent2)
                offspring = mutate(offspring)

                new_population.append(offspring)

            # Replace old population with new population
            population = new_population

        if best_individual is None:
            raise ValueError(
                "No valid parameter combination found during genetic optimization"
            )

        return best_individual, best_result
