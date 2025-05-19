import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from app.backtesting.backtest_engine import BacktestEngine
from app.backtesting.data_manager import DataManager
from app.backtesting.indicator_backtest_manager import IndicatorBacktestManager
from app.indicators.indicator_factory import IndicatorFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IndicatorTestHarness:
    """
    Test harness for validating indicators with the backtesting system.

    This class provides methods to:
    - Test all indicators against standard market datasets
    - Generate performance reports
    - Compare indicator performance
    - Validate signal generation
    """

    def __init__(self, data_source_name: str = "csv"):
        """
        Initialize the indicator test harness.

        Args:
            data_source_name: Name of the data source to use for backtesting
        """
        self.data_source_name = data_source_name
        self.data_manager = DataManager()
        self.backtest_engine = BacktestEngine(data_manager=self.data_manager)
        self.backtest_manager = IndicatorBacktestManager(backtest_engine=self.backtest_engine)

        # Setup paths
        self.project_root = Path(os.path.abspath(__file__)).parents[3]
        self__test_data___dir = self.project_root / "tests" / "__test_data__" / "market_scenarios"
        self__test_results___dir = self.project_root / "tests" / "__test_results__" / "indicator_tests"

        # Ensure directories exist
        self__test_results___dir.mkdir(parents=True, exist_ok=True)

        # Default test parameters
        self.default_symbol = "ETH"
        self.default_interval = "1h"
        self.default_leverage = 1.0

        # Available market scenarios
        self.market_scenarios = {
            "bull": {
                "description": "Strong upward trend",
                "file": self__test_data___dir / "bull_market.csv"
            },
            "bear": {
                "description": "Strong downward trend",
                "file": self__test_data___dir / "bear_market.csv"
            },
            "sideways": {
                "description": "Sideways range-bound market",
                "file": self__test_data___dir / "sideways_market.csv"
            },
            "volatile": {
                "description": "Highly volatile market with quick reversals",
                "file": self__test_data___dir / "volatile_market.csv"
            }
        }

        # Test results
        self__test_results__: Dict[str, Dict[str, Any]] = {}

    def load_all_available_indicators(self) -> List[str]:
        """
        Load all available indicators from the indicator factory.

        Returns:
            List of loaded indicator names
        """
        indicator_types = IndicatorFactory.get_available_indicators()

        for indicator_type in indicator_types:
            # Create a default instance of each indicator type
            indicator_name = f"{indicator_type}_default"
            indicator = IndicatorFactory.create_indicator(
                name=indicator_name,
                indicator_type=indicator_type
            )

            if indicator:
                self.backtest_manager.add_indicator(indicator)

        logger.info(f"Loaded {len(self.backtest_manager.list_indicators())} indicators")
        return self.backtest_manager.list_indicators()

    def load_custom_market_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load custom market data from a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with market data
        """
        try:
            # Register the CSV file with the data manager
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"CSV file not found: {file_path}")
                return pd.DataFrame()

            symbol = "CUSTOM"
            interval = "1h"

            # Register with data manager if needed
            self.data_manager.register_csv_data_source(
                file_path=str(file_path),
                symbol=symbol,
                interval=interval
            )

            # Get the data
            start_time = 0  # Unix timestamp in ms (0 = get all data)
            end_time = int(datetime.now().timestamp() * 1000)  # Current time in ms

            data = self.data_manager.get_data(
                source_name="csv",
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )

            logger.info(f"Loaded {len(data)} candles from {file_path}")
            return data

        except Exception as e:
            logger.error(f"Error loading market data from {file_path}: {e}")
            return pd.DataFrame()

    def prepare_test_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare test datasets for different market scenarios.

        If the scenario files don't exist, this method will return empty DataFrames.

        Returns:
            Dictionary of market scenario names and their corresponding DataFrames
        """
        datasets = {}

        for scenario_name, scenario_info in self.market_scenarios.items():
            file_path = scenario_info["file"]

            if file_path.exists():
                # Register the CSV file with the data manager
                symbol = f"{scenario_name.upper()}_{self.default_symbol}"

                self.data_manager.register_csv_data_source(
                    file_path=str(file_path),
                    symbol=symbol,
                    interval=self.default_interval
                )

                # Get the data
                start_time = 0  # Unix timestamp in ms (0 = get all data)
                end_time = int(datetime.now().timestamp() * 1000)  # Current time in ms

                data = self.data_manager.get_data(
                    source_name="csv",
                    symbol=symbol,
                    interval=self.default_interval,
                    start_time=start_time,
                    end_time=end_time
                )

                datasets[scenario_name] = data
                logger.info(f"Loaded {len(data)} candles for {scenario_name} scenario")
            else:
                logger.warning(f"Scenario file not found: {file_path}")
                datasets[scenario_name] = pd.DataFrame()

        return datasets

    def test_indicator_signal_generation(
        self,
        indicator_name: str,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Test an indicator's signal generation capabilities.

        Args:
            indicator_name: Name of the indicator to test
            data: Market data for testing

        Returns:
            Dictionary with test results
        """
        if data.empty:
            logger.error("Empty data provided for testing")
            return {"valid": False, "error": "Empty data"}

        indicator = self.backtest_manager.get_indicator(indicator_name)
        if not indicator:
            logger.error(f"Indicator not found: {indicator_name}")
            return {"valid": False, "error": f"Indicator not found: {indicator_name}"}

        try:
            # Calculate indicator values
            data_with_indicator = indicator.calculate(data.copy())

            # Check if calculation was successful (no empty indicator columns)
            indicator_columns = [
                col for col in data_with_indicator.columns
                if col not in ["timestamp", "open", "high", "low", "close", "volume"]
            ]

            if not indicator_columns:
                logger.warning(f"No indicator columns added to data by {indicator_name}")
                return {"valid": False, "error": "No indicator columns added"}

            # Check for NaN values
            nan_percentage = data_with_indicator[indicator_columns].isna().mean().mean() * 100

            # Add symbol column if not present
            if "symbol" not in data_with_indicator.columns:
                data_with_indicator["symbol"] = self.default_symbol

            # Test generating signals
            signals = []
            signal_indices = []

            # Try to use generate_signals method if available (batch processing)
            if hasattr(indicator, "generate_signals"):
                try:
                    data_with_signals = indicator.generate_signals(data_with_indicator.copy())
                    if "signal" in data_with_signals.columns:
                        signal_rows = data_with_signals[~data_with_signals["signal"].isna()]
                        signal_count = len(signal_rows)
                        signals = signal_rows["signal"].tolist()
                        signal_indices = signal_rows.index.tolist()
                except Exception as e:
                    logger.warning(f"Error using generate_signals method: {e}")
                    # Fall back to individual signal generation
                    signals = []
                    signal_indices = []

            # If no signals yet, try individual signal generation
            if not signals:
                for i in range(len(data_with_indicator)):
                    window = data_with_indicator.iloc[:i+1]
                    signal = indicator.generate_signal(window)
                    if signal:
                        signals.append(signal.direction)
                        signal_indices.append(i)

            signal_count = len(signals)

            # Plot data with indicator signals if signals were generated
            plot_path = None
            if signal_count > 0:
                plot_path = self__test_results___dir / "plots" / f"{indicator_name}_signals.png"
                plot_path.parent.mkdir(parents=True, exist_ok=True)

                fig, ax = plt.subplots(figsize=(12, 6))

                # Plot price
                ax.plot(data["close"], label="Close Price")

                # Plot buy and sell signals
                buy_indices = [idx for i, idx in enumerate(signal_indices) if signals[i] == "BUY"]
                sell_indices = [idx for i, idx in enumerate(signal_indices) if signals[i] == "SELL"]

                if buy_indices:
                    ax.scatter(
                        buy_indices,
                        data.iloc[buy_indices]["close"],
                        marker="^",
                        color="green",
                        s=100,
                        label="Buy Signal"
                    )

                if sell_indices:
                    ax.scatter(
                        sell_indices,
                        data.iloc[sell_indices]["close"],
                        marker="v",
                        color="red",
                        s=100,
                        label="Sell Signal"
                    )

                ax.set_title(f"{indicator_name} Signals")
                ax.set_xlabel("Candle")
                ax.set_ylabel("Price")
                ax.legend()
                ax.grid(True)

                plt.savefig(plot_path)
                plt.close()

            # Prepare signal stats by direction
            buy_count = sum(1 for s in signals if s == "BUY")
            sell_count = sum(1 for s in signals if s == "SELL")
            neutral_count = sum(1 for s in signals if s == "NEUTRAL")

            return {
                "valid": True,
                "indicator_columns": indicator_columns,
                "nan_percentage": nan_percentage,
                "signal_count": signal_count,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "neutral_count": neutral_count,
                "plot_path": str(plot_path) if plot_path else None
            }

        except Exception as e:
            logger.error(f"Error testing signal generation for {indicator_name}: {e}")
            return {"valid": False, "error": str(e)}

    def test_indicator_with_backtest(
        self,
        indicator_name: str,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        market_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Test an indicator with the backtesting engine.

        Args:
            indicator_name: Name of the indicator to test
            symbol: Market symbol to use
            start_date: Start date for backtest
            end_date: End date for backtest
            market_type: Type of market scenario for documentation

        Returns:
            Dictionary with test results
        """
        try:
            result = self.backtest_manager.backtest_indicator(
                indicator_name=indicator_name,
                symbol=symbol,
                interval=self.default_interval,
                start_date=start_date,
                end_date=end_date,
                data_source_name=self.data_source_name,
                leverage=self.default_leverage,
                strategy_params={}
            )

            if not result:
                logger.error(f"No backtest result returned for {indicator_name}")
                return {"success": False, "error": "No backtest result returned"}

            # Extract key metrics
            metrics = result.metrics
            trades = result.trades

            # Generate plots
            plots = {}
            plots_dir = self__test_results___dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Equity curve plot
            equity_plot_file = plots_dir / f"{indicator_name}_{market_type}_equity.png"
            result.plot_equity_curve(save_path=str(equity_plot_file))
            plots["equity_curve"] = str(equity_plot_file)

            # Drawdown plot
            drawdown_plot_file = plots_dir / f"{indicator_name}_{market_type}_drawdown.png"
            result.plot_drawdown(save_path=str(drawdown_plot_file))
            plots["drawdown"] = str(drawdown_plot_file)

            return {
                "success": True,
                "market_type": market_type,
                "metrics": metrics,
                "trade_count": len(trades),
                "initial_balance": result.initial_balance,
                "final_balance": result.final_balance,
                "plots": plots
            }

        except Exception as e:
            logger.error(f"Error in backtest for {indicator_name}: {e}")
            return {"success": False, "error": str(e)}

    def test_all_indicators(self, scenarios: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Test all loaded indicators with backtesting in different market scenarios.

        Args:
            scenarios: List of scenario names to test, or None for all scenarios

        Returns:
            Dictionary of test results by indicator name
        """
        # Prepare datasets
        datasets = self.prepare_test_datasets()

        # If scenarios not specified, test all available scenarios
        if scenarios is None:
            scenarios = list(datasets.keys())

        # Filter to only include available scenarios
        scenarios = [s for s in scenarios if s in datasets and not datasets[s].empty]

        if not scenarios:
            logger.error("No valid scenarios available for testing")
            return {}

        logger.info(f"Testing all indicators in scenarios: {scenarios}")

        # Get all indicator names
        indicator_names = self.backtest_manager.list_indicators()

        # Test results by indicator
        all_results = {}

        for indicator_name in indicator_names:
            logger.info(f"Testing indicator: {indicator_name}")

            indicator_results = {
                "name": indicator_name,
                "scenarios": {},
                "signal_tests": {}
            }

            # Test signal generation in each scenario
            for scenario in scenarios:
                data = datasets[scenario]
                if not data.empty:
                    signal_test = self.test_indicator_signal_generation(
                        indicator_name=indicator_name,
                        data=data
                    )
                    indicator_results["signal_tests"][scenario] = signal_test

            # Run backtests for each scenario
            for scenario in scenarios:
                data = datasets[scenario]
                if not data.empty:
                    # Extract date range from the data
                    start_date = pd.to_datetime(data["timestamp"].iloc[0], unit="ms")
                    end_date = pd.to_datetime(data["timestamp"].iloc[-1], unit="ms")

                    symbol = f"{scenario.upper()}_{self.default_symbol}"

                    backtest_result = self.test_indicator_with_backtest(
                        indicator_name=indicator_name,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        market_type=scenario
                    )

                    indicator_results["scenarios"][scenario] = backtest_result

            all_results[indicator_name] = indicator_results

        # Store test results
        self__test_results__ = all_results

        return all_results

    def save_test_results(self) -> Tuple[str, str]:
        """
        Save test results to JSON and Markdown files.

        Returns:
            Tuple of (json_file_path, markdown_file_path)
        """
        if not self__test_results__:
            logger.warning("No test results to save")
            return None, None

        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save to JSON
        json_file = self__test_results___dir / f"indicator_tests_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(self__test_results__, f, indent=2, default=str)

        # Create Markdown report
        md_file = self__test_results___dir / f"indicator_tests_{timestamp}.md"
        with open(md_file, "w") as f:
            f.write("# Indicator Backtesting Test Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overview table
            f.write("## Summary\n\n")
            f.write("| Indicator | Valid | Scenarios Tested | Signal Generation | Profitability |\n")
            f.write("|-----------|-------|------------------|-------------------|---------------|\n")

            for indicator_name, results in self__test_results__.items():
                # Check validity
                signal_tests = results.get("signal_tests", {})
                valid = all(test.get("valid", False) for test in signal_tests.values())

                # Count tested scenarios
                scenarios = results.get("scenarios", {})
                scenario_count = len(scenarios)

                # Signal generation stats
                total_signals = sum(
                    test.get("signal_count", 0)
                    for test in signal_tests.values()
                )

                # Profitability
                profitable_scenarios = sum(
                    1 for s in scenarios.values()
                    if s.get("success", False) and s.get("metrics", {}).get("total_return", 0) > 0
                )

                # Add row to table
                f.write(f"| {indicator_name} | {'✓' if valid else '✗'} | {scenario_count} | ")
                f.write(f"{total_signals} signals | {profitable_scenarios}/{scenario_count} scenarios |\n")

            # Detailed results by indicator
            f.write("\n## Detailed Results\n\n")

            for indicator_name, results in self__test_results__.items():
                f.write(f"### {indicator_name}\n\n")

                # Signal generation tests
                f.write("#### Signal Generation\n\n")
                f.write("| Scenario | Valid | Signals | Buy | Sell | NaN % |\n")
                f.write("|----------|-------|---------|-----|------|------|\n")

                signal_tests = results.get("signal_tests", {})
                for scenario, test in signal_tests.items():
                    valid = test.get("valid", False)
                    signal_count = test.get("signal_count", 0)
                    buy_count = test.get("buy_count", 0)
                    sell_count = test.get("sell_count", 0)
                    nan_percentage = test.get("nan_percentage", 0)

                    f.write(f"| {scenario} | {'✓' if valid else '✗'} | {signal_count} | ")
                    f.write(f"{buy_count} | {sell_count} | {nan_percentage:.2f}% |\n")

                # Backtest results
                f.write("\n#### Backtest Performance\n\n")
                f.write("| Scenario | Success | Total Return | Sharpe | Max DD | Win Rate | Trades |\n")
                f.write("|----------|---------|--------------|--------|--------|----------|--------|\n")

                scenarios = results.get("scenarios", {})
                for scenario, backtest in scenarios.items():
                    success = backtest.get("success", False)
                    metrics = backtest.get("metrics", {})

                    total_return = metrics.get("total_return", 0) * 100  # Convert to percentage
                    sharpe = metrics.get("sharpe_ratio", 0)
                    max_dd = metrics.get("max_drawdown", 0) * 100  # Convert to percentage
                    win_rate = metrics.get("win_rate", 0) * 100  # Convert to percentage
                    trade_count = backtest.get("trade_count", 0)

                    f.write(f"| {scenario} | {'✓' if success else '✗'} | {total_return:.2f}% | ")
                    f.write(f"{sharpe:.2f} | {max_dd:.2f}% | {win_rate:.2f}% | {trade_count} |\n")

                f.write("\n")

        logger.info(f"Saved test results to {json_file} and {md_file}")
        return str(json_file), str(md_file)


def run_indicator_tests():
    """Run tests for all available indicators and save the results."""
    # Initialize test harness
    test_harness = IndicatorTestHarness()

    # Load all available indicators
    indicator_names = test_harness.load_all_available_indicators()
    logger.info(f"Loaded {len(indicator_names)} indicators for testing")

    # Run tests
    results = test_harness.test_all_indicators()

    # Save results
    json_file, md_file = test_harness.save_test_results()

    logger.info(f"Testing completed. Results saved to:")
    logger.info(f"- JSON: {json_file}")
    logger.info(f"- Markdown: {md_file}")

    return results


if __name__ == "__main__":
    run_indicator_tests()
