import logging
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..indicators.base_indicator import BaseIndicator
from ..indicators.indicator_factory import IndicatorFactory
from .backtest_engine import BacktestEngine, BacktestResult
from .indicator_backtest_manager import IndicatorBacktestManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IndicatorOptimizer:
    """
    Framework for optimizing indicator parameters through backtesting.

    This class provides methods for grid search, cross-validation,
    and parameter sensitivity analysis to find optimal indicator settings.
    """

    def __init__(
        self,
        backtest_manager: IndicatorBacktestManager,
        symbol: str,
        interval: str,
        data_source_name: str,
        metric_to_optimize: str = "sharpe_ratio",
        higher_is_better: bool = True
    ):
        """
        Initialize the indicator optimizer.

        Args:
            backtest_manager: IndicatorBacktestManager instance
            symbol: Market symbol to use for optimization
            interval: Timeframe interval to use
            data_source_name: Name of the data source to use
            metric_to_optimize: Performance metric to optimize
            higher_is_better: True if higher metric values are better, False otherwise
        """
        self.backtest_manager = backtest_manager
        self.symbol = symbol
        self.interval = interval
        self.data_source_name = data_source_name
        self.metric_to_optimize = metric_to_optimize
        self.higher_is_better = higher_is_better

        # Track optimization results
        self.optimization_results: Dict[str, Any] = {}

    def grid_search(
        self,
        indicator_type: str,
        param_grid: Dict[str, List[Any]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        leverage: float = 1.0,
        strategy_params: Optional[Dict[str, Any]] = None,
        max_parallel: int = 1
    ) -> Tuple[Dict[str, Any], BaseIndicator, BacktestResult]:
        """
        Perform grid search to find optimal indicator parameters.

        Args:
            indicator_type: Type of indicator to optimize
            param_grid: Dictionary of parameter names and possible values
            start_date: Start date for backtests
            end_date: End date for backtests
            leverage: Default leverage to use
            strategy_params: Additional parameters for the strategy
            max_parallel: Maximum number of parallel backtests (1 = sequential)

        Returns:
            Tuple of (best parameters, optimized indicator, best result)
        """
        if not param_grid:
            logger.warning("Empty parameter grid provided")
            return {}, None, None

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        logger.info(f"Running grid search with {len(param_combinations)} parameter combinations")

        # Prepare results storage
        results = []
        best_score = float("-inf") if self.higher_is_better else float("inf")
        best_params = None
        best_result = None
        best_indicator = None

        # Track start time for progress reporting
        start_time = time.time()

        # Run backtests for all parameter combinations
        for i, combination in enumerate(tqdm(param_combinations, desc="Grid search progress")):
            # Create parameter dictionary for this combination
            params = {param_names[j]: combination[j] for j in range(len(param_names))}

            # Create indicator with these parameters
            indicator_name = f"{indicator_type}_grid_{i}"
            indicator = IndicatorFactory.create_indicator(
                name=indicator_name,
                indicator_type=indicator_type,
                params=params
            )

            if not indicator:
                logger.warning(f"Failed to create indicator with parameters: {params}")
                continue

            # Add indicator to backtest manager
            self.backtest_manager.add_indicator(indicator)

            # Run backtest
            try:
                result = self.backtest_manager.backtest_indicator(
                    indicator_name=indicator_name,
                    symbol=self.symbol,
                    interval=self.interval,
                    start_date=start_date,
                    end_date=end_date,
                    data_source_name=self.data_source_name,
                    leverage=leverage,
                    strategy_params=strategy_params
                )

                if not result:
                    logger.warning(f"No result returned for parameters: {params}")
                    continue

                # Extract optimization metric
                if self.metric_to_optimize not in result.metrics:
                    logger.warning(f"Metric '{self.metric_to_optimize}' not found in results for parameters: {params}")
                    continue

                score = result.metrics[self.metric_to_optimize]

                # Store result
                results.append({
                    "parameters": params,
                    "score": score,
                    "result": result,
                    "indicator": indicator
                })

                # Check if this is the best score so far
                is_better = (score > best_score) if self.higher_is_better else (score < best_score)
                if is_better:
                    best_score = score
                    best_params = params
                    best_result = result
                    best_indicator = indicator

                # Progress report every 5% of combinations
                if i % max(1, len(param_combinations) // 20) == 0:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / (i + 1)) * (len(param_combinations) - i - 1)
                    logger.info(
                        f"Progress: {i + 1}/{len(param_combinations)} combinations "
                        f"({(i + 1) / len(param_combinations) * 100:.1f}%) - "
                        f"Best {self.metric_to_optimize}: {best_score:.4f} - "
                        f"ETA: {remaining:.1f}s"
                    )

            except Exception as e:
                logger.error(f"Error during backtest for parameters {params}: {e}")
                continue
            finally:
                # Clean up indicator
                self.backtest_manager.remove_indicator(indicator_name)

        # Sort results by score
        results.sort(key=lambda x: x["score"], reverse=self.higher_is_better)

        # Store optimization results
        self.optimization_results = {
            "indicator_type": indicator_type,
            "param_grid": param_grid,
            "results": results,
            "best_params": best_params,
            "best_score": best_score,
            "metric": self.metric_to_optimize,
            "higher_is_better": self.higher_is_better,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        logger.info(f"Grid search complete - Best {self.metric_to_optimize}: {best_score:.4f} with parameters: {best_params}")

        # Return best parameters and result
        return best_params, best_indicator, best_result

    def cross_validation(
        self,
        indicator_type: str,
        params: Dict[str, Any],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        n_folds: int = 5,
        fold_size: int = 30,  # days
        leverage: float = 1.0,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform time-series cross-validation to evaluate parameter robustness.

        Args:
            indicator_type: Type of indicator to validate
            params: Parameters for the indicator
            start_date: Start date for full period
            end_date: End date for full period
            n_folds: Number of validation folds
            fold_size: Size of each fold in days
            leverage: Default leverage to use
            strategy_params: Additional parameters for the strategy

        Returns:
            Dictionary with cross-validation results
        """
        # Convert dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Create time windows for cross-validation
        total_days = (end_date - start_date).days
        if total_days < n_folds * fold_size:
            logger.warning(
                f"Time period too short for {n_folds} folds of {fold_size} days each. "
                f"Reducing to {total_days // fold_size} folds."
            )
            n_folds = max(2, total_days // fold_size)

        # Create expanding window folds
        fold_end_days = [fold_size * (i + 1) for i in range(n_folds)]
        folds = []

        for i, end_days in enumerate(fold_end_days):
            train_end = start_date + pd.Timedelta(days=end_days - fold_size)
            test_start = train_end
            test_end = min(start_date + pd.Timedelta(days=end_days), end_date)

            folds.append({
                "fold": i + 1,
                "train_start": start_date,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end
            })

        logger.info(f"Running {n_folds}-fold cross-validation for parameters: {params}")

        # Run backtest for each fold
        fold_results = []

        for fold in tqdm(folds, desc="Cross-validation progress"):
            # Create indicator with provided parameters
            indicator_name = f"{indicator_type}_cv_fold{fold['fold']}"
            indicator = IndicatorFactory.create_indicator(
                name=indicator_name,
                indicator_type=indicator_type,
                params=params
            )

            if not indicator:
                logger.warning(f"Failed to create indicator for fold {fold['fold']}")
                continue

            # Add indicator to backtest manager
            self.backtest_manager.add_indicator(indicator)

            try:
                # Train phase (just to initialize indicator with training data)
                train_result = self.backtest_manager.backtest_indicator(
                    indicator_name=indicator_name,
                    symbol=self.symbol,
                    interval=self.interval,
                    start_date=fold["train_start"],
                    end_date=fold["train_end"],
                    data_source_name=self.data_source_name,
                    leverage=leverage,
                    strategy_params=strategy_params
                )

                # Test phase (to evaluate performance on unseen data)
                test_result = self.backtest_manager.backtest_indicator(
                    indicator_name=indicator_name,
                    symbol=self.symbol,
                    interval=self.interval,
                    start_date=fold["test_start"],
                    end_date=fold["test_end"],
                    data_source_name=self.data_source_name,
                    leverage=leverage,
                    strategy_params=strategy_params
                )

                if not test_result:
                    logger.warning(f"No test result for fold {fold['fold']}")
                    continue

                # Extract optimization metric
                if self.metric_to_optimize not in test_result.metrics:
                    logger.warning(f"Metric '{self.metric_to_optimize}' not found in results for fold {fold['fold']}")
                    continue

                score = test_result.metrics[self.metric_to_optimize]

                # Store fold result
                fold_results.append({
                    "fold": fold["fold"],
                    "train_period": f"{fold['train_start']} to {fold['train_end']}",
                    "test_period": f"{fold['test_start']} to {fold['test_end']}",
                    "score": score,
                    "metrics": test_result.metrics
                })

            except Exception as e:
                logger.error(f"Error during cross-validation for fold {fold['fold']}: {e}")
                continue
            finally:
                # Clean up indicator
                self.backtest_manager.remove_indicator(indicator_name)

        # Calculate cross-validation metrics
        if not fold_results:
            logger.error("No valid fold results")
            return {"cv_score": None, "cv_std": None, "fold_results": []}

        scores = [fold["score"] for fold in fold_results]
        cv_score = np.mean(scores)
        cv_std = np.std(scores)

        logger.info(f"Cross-validation complete - Mean {self.metric_to_optimize}: {cv_score:.4f} ± {cv_std:.4f}")

        # Return cross-validation results
        return {
            "cv_score": cv_score,
            "cv_std": cv_std,
            "fold_results": fold_results,
            "params": params,
            "n_folds": n_folds,
            "metric": self.metric_to_optimize,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def parameter_sensitivity_analysis(
        self,
        indicator_type: str,
        base_params: Dict[str, Any],
        param_ranges: Dict[str, List[Any]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        leverage: float = 1.0,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform parameter sensitivity analysis to understand how changes in each parameter affect performance.

        Args:
            indicator_type: Type of indicator to analyze
            base_params: Base parameters for the indicator
            param_ranges: Dictionary of parameter names and ranges to test
            start_date: Start date for backtests
            end_date: End date for backtests
            leverage: Default leverage to use
            strategy_params: Additional parameters for the strategy

        Returns:
            Dictionary with sensitivity analysis results
        """
        if not param_ranges:
            logger.warning("Empty parameter ranges provided")
            return {}

        # Prepare results storage
        sensitivity_results = {}

        logger.info(f"Running parameter sensitivity analysis for {indicator_type}")

        # First, run backtest with base parameters
        base_indicator_name = f"{indicator_type}_base"
        base_indicator = IndicatorFactory.create_indicator(
            name=base_indicator_name,
            indicator_type=indicator_type,
            params=base_params
        )

        if not base_indicator:
            logger.error(f"Failed to create indicator with base parameters: {base_params}")
            return {}

        # Add base indicator to backtest manager
        self.backtest_manager.add_indicator(base_indicator)

        try:
            base_result = self.backtest_manager.backtest_indicator(
                indicator_name=base_indicator_name,
                symbol=self.symbol,
                interval=self.interval,
                start_date=start_date,
                end_date=end_date,
                data_source_name=self.data_source_name,
                leverage=leverage,
                strategy_params=strategy_params
            )

            if not base_result:
                logger.error("No result returned for base parameters")
                return {}

            # Extract optimization metric for base parameters
            if self.metric_to_optimize not in base_result.metrics:
                logger.error(f"Metric '{self.metric_to_optimize}' not found in results for base parameters")
                return {}

            base_score = base_result.metrics[self.metric_to_optimize]

            logger.info(f"Base {self.metric_to_optimize}: {base_score:.4f} with parameters: {base_params}")

            # For each parameter, vary it and see how the metric changes
            for param_name, param_values in tqdm(param_ranges.items(), desc="Testing parameters"):
                param_results = []

                for value in tqdm(param_values, desc=f"Testing {param_name}", leave=False):
                    # Skip if this is the base value
                    if param_name in base_params and base_params[param_name] == value:
                        # Still add the base score for this value
                        param_results.append({
                            "value": value,
                            "score": base_score,
                            "change": 0.0  # No change for base value
                        })
                        continue

                    # Create modified parameters
                    modified_params = base_params.copy()
                    modified_params[param_name] = value

                    # Create indicator with modified parameters
                    indicator_name = f"{indicator_type}_{param_name}_{value}"
                    indicator = IndicatorFactory.create_indicator(
                        name=indicator_name,
                        indicator_type=indicator_type,
                        params=modified_params
                    )

                    if not indicator:
                        logger.warning(f"Failed to create indicator with {param_name}={value}")
                        continue

                    # Add indicator to backtest manager
                    self.backtest_manager.add_indicator(indicator)

                    try:
                        # Run backtest
                        result = self.backtest_manager.backtest_indicator(
                            indicator_name=indicator_name,
                            symbol=self.symbol,
                            interval=self.interval,
                            start_date=start_date,
                            end_date=end_date,
                            data_source_name=self.data_source_name,
                            leverage=leverage,
                            strategy_params=strategy_params
                        )

                        if not result:
                            logger.warning(f"No result returned for {param_name}={value}")
                            continue

                        # Extract optimization metric
                        if self.metric_to_optimize not in result.metrics:
                            logger.warning(f"Metric '{self.metric_to_optimize}' not found in results for {param_name}={value}")
                            continue

                        score = result.metrics[self.metric_to_optimize]

                        # Calculate change from base score
                        change = score - base_score

                        # Store result
                        param_results.append({
                            "value": value,
                            "score": score,
                            "change": change
                        })

                    except Exception as e:
                        logger.error(f"Error during backtest for {param_name}={value}: {e}")
                        continue
                    finally:
                        # Clean up indicator
                        self.backtest_manager.remove_indicator(indicator_name)

                # Store parameter results
                sensitivity_results[param_name] = param_results

        except Exception as e:
            logger.error(f"Error during sensitivity analysis: {e}")
            return {}
        finally:
            # Clean up base indicator
            self.backtest_manager.remove_indicator(base_indicator_name)

        # Return sensitivity analysis results
        return {
            "base_params": base_params,
            "base_score": base_score,
            "sensitivity": sensitivity_results,
            "metric": self.metric_to_optimize,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def save_optimization_results(self, output_path: Union[str, Path]) -> None:
        """
        Save optimization results to a file.

        Args:
            output_path: Path to save the optimization results
        """
        if not self.optimization_results:
            logger.warning("No optimization results to save")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a simplified version of results for serialization
        serializable_results = {
            "indicator_type": self.optimization_results["indicator_type"],
            "param_grid": self.optimization_results["param_grid"],
            "best_params": self.optimization_results["best_params"],
            "best_score": self.optimization_results["best_score"],
            "metric": self.optimization_results["metric"],
            "higher_is_better": self.optimization_results["higher_is_better"],
            "timestamp": self.optimization_results["timestamp"],
            "results": []
        }

        # Add top N results
        for i, result in enumerate(self.optimization_results["results"][:10]):  # Save top 10 results
            serializable_results["results"].append({
                "rank": i + 1,
                "parameters": result["parameters"],
                "score": result["score"],
                # Add key metrics from the backtest
                "metrics": {k: v for k, v in result["result"].metrics.items() if k in [
                    "total_return", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
                    "max_drawdown", "win_rate", "profit_factor", "total_trades"
                ]}
            })

        # Save to file
        import json
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Saved optimization results to {output_path}")

    def create_optimization_report(
        self,
        output_dir: Union[str, Path],
        include_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Create a detailed optimization report.

        Args:
            output_dir: Directory to save report files
            include_plots: Whether to include plots in the report

        Returns:
            Dictionary with report file paths
        """
        if not self.optimization_results:
            logger.warning("No optimization results to report")
            return {}

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full optimization results
        results_file = output_dir / "optimization_results.json"
        self.save_optimization_results(results_file)

        # Create report file
        report_file = output_dir / "optimization_report.md"

        with open(report_file, "w") as f:
            # Write report header
            f.write(f"# Indicator Optimization Report\n\n")
            f.write(f"Generated: {self.optimization_results['timestamp']}\n\n")
            f.write(f"## Optimization Parameters\n\n")
            f.write(f"- Indicator Type: {self.optimization_results['indicator_type']}\n")
            f.write(f"- Metric: {self.optimization_results['metric']}\n")
            f.write(f"- Higher is Better: {self.optimization_results['higher_is_better']}\n\n")

            f.write(f"## Parameter Grid\n\n")
            for param, values in self.optimization_results['param_grid'].items():
                f.write(f"- {param}: {values}\n")
            f.write("\n")

            f.write(f"## Best Parameters\n\n")
            f.write(f"- Score ({self.optimization_results['metric']}): {self.optimization_results['best_score']:.4f}\n")
            f.write(f"- Parameters:\n")
            for param, value in self.optimization_results['best_params'].items():
                f.write(f"  - {param}: {value}\n")
            f.write("\n")

            f.write(f"## Top Results\n\n")
            f.write(f"| Rank | {self.optimization_results['metric']} | ")

            # Add metric columns
            metric_columns = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
            for metric in metric_columns:
                f.write(f"{metric} | ")

            # Add parameter columns
            param_names = list(self.optimization_results['param_grid'].keys())
            for param in param_names:
                f.write(f"{param} | ")
            f.write("\n")

            # Add separator row
            f.write(f"|------|------|")
            for _ in metric_columns:
                f.write(f"------|")
            for _ in param_names:
                f.write(f"------|")
            f.write("\n")

            # Add result rows
            for i, result in enumerate(self.optimization_results["results"][:10]):  # Top 10 results
                f.write(f"| {i+1} | {result['score']:.4f} | ")

                # Add metric values
                for metric in metric_columns:
                    value = result["result"].metrics.get(metric, "N/A")
                    if isinstance(value, float):
                        f.write(f"{value:.4f} | ")
                    else:
                        f.write(f"{value} | ")

                # Add parameter values
                for param in param_names:
                    f.write(f"{result['parameters'].get(param, 'N/A')} | ")
                f.write("\n")

            f.write("\n")

        logger.info(f"Generated optimization report at {report_file}")

        # Generate plots if requested
        plots = {}
        if include_plots and len(self.optimization_results["results"]) > 0:
            import matplotlib.pyplot as plt

            # Plot parameter distributions for top results
            for param_name in self.optimization_results['param_grid'].keys():
                param_values = [r["parameters"][param_name] for r in self.optimization_results["results"][:20] if param_name in r["parameters"]]
                scores = [r["score"] for r in self.optimization_results["results"][:20] if param_name in r["parameters"]]

                if len(param_values) > 1:
                    plt.figure(figsize=(10, 6))

                    # Check if parameter values are numeric
                    if all(isinstance(v, (int, float)) for v in param_values):
                        # Create scatter plot for numeric parameters
                        plt.scatter(param_values, scores)
                        plt.plot(param_values, scores, 'r--', alpha=0.5)
                    else:
                        # Create bar chart for categorical parameters
                        plt.bar(range(len(param_values)), scores, tick_label=[str(v) for v in param_values])

                    plt.title(f"Impact of {param_name} on {self.optimization_results['metric']}")
                    plt.xlabel(param_name)
                    plt.ylabel(self.optimization_results['metric'])
                    plt.grid(True, alpha=0.3)

                    plot_file = output_dir / f"param_impact_{param_name}.png"
                    plt.savefig(plot_file)
                    plt.close()

                    plots[f"param_impact_{param_name}"] = str(plot_file)

            # Plot distribution of scores
            plt.figure(figsize=(10, 6))
            scores = [r["score"] for r in self.optimization_results["results"]]
            plt.hist(scores, bins=20)
            plt.title(f"Distribution of {self.optimization_results['metric']} Scores")
            plt.xlabel(self.optimization_results['metric'])
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)

            plot_file = output_dir / f"score_distribution.png"
            plt.savefig(plot_file)
            plt.close()

            plots["score_distribution"] = str(plot_file)

        return {
            "report_file": str(report_file),
            "results_file": str(results_file),
            "plots": plots
        }
