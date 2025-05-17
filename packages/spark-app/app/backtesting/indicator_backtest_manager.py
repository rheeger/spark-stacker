import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..connectors.base_connector import OrderSide, OrderType
from ..indicators.base_indicator import BaseIndicator, Signal, SignalDirection
from .backtest_engine import BacktestEngine, BacktestResult
from .indicator_config_loader import IndicatorConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IndicatorBacktestManager:
    """
    Manages backtesting of indicators and generates performance reports.

    This class provides a standardized interface for integrating indicators
    with the backtesting engine, running tests, and analyzing results.
    """

    def __init__(self, backtest_engine: BacktestEngine):
        """
        Initialize the indicator backtest manager.

        Args:
            backtest_engine: Instance of the BacktestEngine
        """
        self.backtest_engine = backtest_engine
        self.indicators: Dict[str, BaseIndicator] = {}
        self.results: Dict[str, BacktestResult] = {}

    def load_indicators_from_config(self, config_path: Union[str, Path]) -> None:
        """
        Load indicators from a configuration file.

        Args:
            config_path: Path to the indicator configuration file
        """
        self.indicators = IndicatorConfigLoader.load_indicators(config_path)
        logger.info(f"Loaded {len(self.indicators)} indicators from configuration")

    def add_indicator(self, indicator: BaseIndicator) -> None:
        """
        Add an indicator to the manager.

        Args:
            indicator: Indicator instance to add
        """
        self.indicators[indicator.name] = indicator
        logger.info(f"Added indicator: {indicator.name}")

    def remove_indicator(self, indicator_name: str) -> None:
        """
        Remove an indicator from the manager.

        Args:
            indicator_name: Name of the indicator to remove
        """
        if indicator_name in self.indicators:
            del self.indicators[indicator_name]
            logger.info(f"Removed indicator: {indicator_name}")
        else:
            logger.warning(f"Indicator not found: {indicator_name}")

    def get_indicator(self, indicator_name: str) -> Optional[BaseIndicator]:
        """
        Get an indicator by name.

        Args:
            indicator_name: Name of the indicator to retrieve

        Returns:
            The indicator instance or None if not found
        """
        return self.indicators.get(indicator_name)

    def list_indicators(self) -> List[str]:
        """
        Get a list of all loaded indicator names.

        Returns:
            List of indicator names
        """
        return list(self.indicators.keys())

    def _create_indicator_strategy(self, indicator: BaseIndicator) -> callable:
        """
        Create a strategy function based on an indicator.

        Args:
            indicator: Indicator to create a strategy for

        Returns:
            Strategy function that can be used by the backtest engine
        """

        def indicator_strategy(
            data: pd.DataFrame,
            simulation_engine: Any,
            params: Dict[str, Any]
        ) -> None:
            """
            Strategy that trades based on indicator signals.

            Args:
                data: Historical price data
                simulation_engine: Trading simulation engine
                params: Additional parameters including the current candle
            """
            symbol = params.get("symbol")
            current_candle = params.get("current_candle")
            leverage = params.get("leverage", 1.0)

            if not (symbol and current_candle is not None):
                logger.warning("Missing required parameters in strategy call")
                return

            # Calculate indicator values on the full dataset first
            data_with_indicators = indicator.calculate(data)

            # Use only the current candle for signal generation but with indicators
            latest_data = data_with_indicators.iloc[-1:].copy()
            latest_data["symbol"] = symbol

            # Generate signal
            signal = indicator.generate_signal(data_with_indicators)

            # Execute trades based on signal
            if signal is not None:
                logger.debug(f"Generated signal: {signal}")

                # Check for existing position
                positions = simulation_engine.get_positions(symbol)
                position = positions[0] if positions else None
                position_side = None if position is None else position.side

                if signal.direction == SignalDirection.BUY and position_side != "LONG":
                    # Close any existing short position
                    if position_side == "SHORT":
                        # Place an order in the opposite direction to close the position
                        position_amount = position.amount if position else 0
                        if position_amount > 0:
                            simulation_engine.place_order(
                                symbol=symbol,
                                side=OrderSide.BUY,  # Opposite of SHORT
                                order_type=OrderType.MARKET,
                                amount=position_amount,
                                timestamp=current_candle["timestamp"],
                                current_candle=current_candle
                            )

                    # Calculate position size based on available balance
                    balance = simulation_engine.get_balance("USD")
                    position_size = (balance * 0.95) / current_candle["close"]  # Use 95% of available balance

                    # Open long position
                    simulation_engine.place_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        amount=position_size,
                        timestamp=current_candle["timestamp"],
                        current_candle=current_candle
                    )

                elif signal.direction == SignalDirection.SELL and position_side != "SHORT":
                    # Close any existing long position
                    if position_side == "LONG":
                        # Place an order in the opposite direction to close the position
                        position_amount = position.amount if position else 0
                        if position_amount > 0:
                            simulation_engine.place_order(
                                symbol=symbol,
                                side=OrderSide.SELL,  # Opposite of LONG
                                order_type=OrderType.MARKET,
                                amount=position_amount,
                                timestamp=current_candle["timestamp"],
                                current_candle=current_candle
                            )

                    # Calculate position size based on available balance
                    balance = simulation_engine.get_balance("USD")
                    position_size = (balance * 0.95) / current_candle["close"]  # Use 95% of available balance

                    # Open short position
                    simulation_engine.place_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        amount=position_size,
                        timestamp=current_candle["timestamp"],
                        current_candle=current_candle
                    )

        return indicator_strategy

    def backtest_indicator(
        self,
        indicator_name: str,
        symbol: str,
        interval: str,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        data_source_name: str,
        leverage: float = 1.0,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> Optional[BacktestResult]:
        """
        Run a backtest for a specific indicator.

        Args:
            indicator_name: Name of the indicator to backtest
            symbol: Market symbol to backtest
            interval: Timeframe interval (e.g., '1m', '1h', '1d')
            start_date: Start date for backtest
            end_date: End date for backtest
            data_source_name: Name of the data source to use
            leverage: Default leverage to use
            strategy_params: Additional parameters for the strategy

        Returns:
            BacktestResult with performance metrics or None if indicator not found
        """
        indicator = self.get_indicator(indicator_name)
        if not indicator:
            logger.error(f"Indicator not found: {indicator_name}")
            return None

        # Create strategy function for the indicator
        strategy_func = self._create_indicator_strategy(indicator)

        # Prepare strategy parameters
        params = strategy_params or {}

        # Run backtest
        logger.info(f"Running backtest for indicator: {indicator_name}")
        result = self.backtest_engine.run_backtest(
            strategy_func=strategy_func,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_source_name=data_source_name,
            strategy_params=params,
            leverage=leverage,
            indicators=[indicator]  # Pass the indicator for integrated calculations
        )

        # Store result
        self.results[indicator_name] = result

        return result

    def backtest_all_indicators(
        self,
        symbol: str,
        interval: str,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        data_source_name: str,
        leverage: float = 1.0,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, BacktestResult]:
        """
        Run backtests for all loaded indicators.

        Args:
            symbol: Market symbol to backtest
            interval: Timeframe interval (e.g., '1m', '1h', '1d')
            start_date: Start date for backtest
            end_date: End date for backtest
            data_source_name: Name of the data source to use
            leverage: Default leverage to use
            strategy_params: Additional parameters for the strategy

        Returns:
            Dictionary of backtest results keyed by indicator name
        """
        results = {}

        for indicator_name in self.indicators:
            result = self.backtest_indicator(
                indicator_name=indicator_name,
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                data_source_name=data_source_name,
                leverage=leverage,
                strategy_params=strategy_params
            )

            if result:
                results[indicator_name] = result

        self.results = results
        return results

    def compare_indicators(
        self,
        metric_name: str = "total_return",
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Compare indicator performance based on a specific metric.

        Args:
            metric_name: Name of the metric to compare
            ascending: Sort in ascending order if True

        Returns:
            DataFrame with indicator performance comparison
        """
        if not self.results:
            logger.warning("No backtest results available for comparison")
            return pd.DataFrame()

        comparison_data = []

        for indicator_name, result in self.results.items():
            metrics = result.metrics

            if metric_name not in metrics:
                logger.warning(f"Metric '{metric_name}' not found in results for {indicator_name}")
                continue

            data = {
                "indicator": indicator_name,
                metric_name: metrics[metric_name]
            }

            # Add other common metrics
            for common_metric in ["total_trades", "win_rate", "profit_factor", "max_drawdown", "sharpe_ratio"]:
                if common_metric in metrics:
                    data[common_metric] = metrics[common_metric]

            comparison_data.append(data)

        comparison_df = pd.DataFrame(comparison_data)

        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values(by=metric_name, ascending=ascending)

        return comparison_df

    def generate_indicator_performance_report(
        self,
        indicator_name: str,
        output_dir: Union[str, Path],
        include_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a performance report for a specific indicator.

        Args:
            indicator_name: Name of the indicator to generate a report for
            output_dir: Directory to save report files
            include_plots: Whether to include plots in the report

        Returns:
            Dictionary containing report data
        """
        result = self.results.get(indicator_name)
        if not result:
            logger.error(f"No backtest results available for indicator: {indicator_name}")
            return {}

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare report data
        report = {
            "indicator_name": indicator_name,
            "symbol": result.symbol,
            "period": f"{result.start_date} to {result.end_date}",
            "metrics": result.metrics,
            "trades": result.trades
        }

        # Save result to JSON file
        result_file = output_dir / f"{indicator_name}_result.json"
        result.save_to_file(str(result_file))
        report["result_file"] = str(result_file)

        # Generate and save plots
        if include_plots:
            plots = {}

            # Equity curve plot
            equity_plot_file = output_dir / f"{indicator_name}_equity.png"
            result.plot_equity_curve(save_path=str(equity_plot_file))
            plots["equity_curve"] = str(equity_plot_file)

            # Drawdown plot
            drawdown_plot_file = output_dir / f"{indicator_name}_drawdown.png"
            result.plot_drawdown(save_path=str(drawdown_plot_file))
            plots["drawdown"] = str(drawdown_plot_file)

            report["plots"] = plots

        # Create summary file
        summary_file = output_dir / f"{indicator_name}_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Performance Report for {indicator_name}\n")
            f.write(f"Symbol: {result.symbol}\n")
            f.write(f"Period: {result.start_date} to {result.end_date}\n\n")
            f.write("Performance Metrics:\n")

            for metric, value in result.metrics.items():
                if isinstance(value, float):
                    f.write(f"- {metric}: {value:.4f}\n")
                else:
                    f.write(f"- {metric}: {value}\n")

            f.write(f"\nTotal Trades: {len(result.trades)}\n")

            if result.trades:
                # Calculate trades statistics
                win_count = sum(1 for t in result.trades if t.get("realized_pnl", 0) > 0)
                lose_count = sum(1 for t in result.trades if t.get("realized_pnl", 0) < 0)

                f.write(f"Winning Trades: {win_count}\n")
                f.write(f"Losing Trades: {lose_count}\n")

                if len(result.trades) > 0:
                    win_rate = win_count / len(result.trades) * 100
                    f.write(f"Win Rate: {win_rate:.2f}%\n")

        report["summary_file"] = str(summary_file)

        logger.info(f"Generated performance report for {indicator_name} at {output_dir}")
        return report

    def plot_indicator_signals(
        self,
        indicator_name: str,
        data: pd.DataFrame,
        output_file: Optional[Union[str, Path]] = None
    ) -> Optional[Figure]:
        """
        Plot price data with indicator signals.

        Args:
            indicator_name: Name of the indicator to plot
            data: Price data as a pandas DataFrame
            output_file: Path to save the plot file

        Returns:
            Matplotlib Figure object if successful, None otherwise
        """
        indicator = self.get_indicator(indicator_name)
        if not indicator:
            logger.error(f"Indicator not found: {indicator_name}")
            return None

        # Calculate indicator values
        data_with_indicator = indicator.calculate(data.copy())

        # Generate signals for each candle
        if hasattr(indicator, "generate_signals"):
            data_with_signals = indicator.generate_signals(data_with_indicator.copy())
        else:
            # Fallback to manually processing each candle
            data_with_signals = data_with_indicator.copy()
            data_with_signals["signal"] = None

            for i in range(len(data_with_signals)):
                signal = indicator.generate_signal(data_with_signals.iloc[:i+1])
                if signal:
                    data_with_signals.loc[data_with_signals.index[i], "signal"] = signal.direction

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot price data
        ax.plot(data_with_signals.index, data_with_signals["close"], label="Close Price")

        # Plot buy signals
        buy_signals = data_with_signals[data_with_signals["signal"] == SignalDirection.BUY]
        if not buy_signals.empty:
            ax.scatter(
                buy_signals.index,
                buy_signals["close"],
                marker="^",
                color="green",
                s=100,
                label="Buy Signal"
            )

        # Plot sell signals
        sell_signals = data_with_signals[data_with_signals["signal"] == SignalDirection.SELL]
        if not sell_signals.empty:
            ax.scatter(
                sell_signals.index,
                sell_signals["close"],
                marker="v",
                color="red",
                s=100,
                label="Sell Signal"
            )

        # Customize plot
        ax.set_title(f"{indicator_name} Signals")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)

        # Format x-axis date labels
        fig.autofmt_xdate()

        # Save plot if output file specified
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file)
            logger.info(f"Saved indicator signals plot to {output_file}")

        return fig
