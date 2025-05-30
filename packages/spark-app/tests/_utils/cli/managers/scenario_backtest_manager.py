"""
Scenario Backtest Manager

This module provides scenario-specific backtesting capabilities:
- Move from `tests/_utils/scenario_backtest_manager.py`
- Add integration with core scenario manager
- Add parallel scenario execution
- Add scenario result aggregation
- Add scenario-specific performance metrics
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from app.backtesting.backtest_engine import BacktestResult

from ..core.config_manager import ConfigManager
from ..core.data_manager import DataManager as CLIDataManager
from ..core.scenario_manager import (ScenarioConfig, ScenarioManager,
                                     ScenarioResult, ScenarioType)
from .indicator_backtest_manager import IndicatorBacktestManager
from .strategy_backtest_manager import StrategyBacktestManager

logger = logging.getLogger(__name__)


@dataclass
class ScenarioBacktestResult:
    """Result of a scenario-based backtest."""
    scenario_name: str
    scenario_type: ScenarioType
    backtest_result: BacktestResult
    scenario_metadata: Dict[str, Any]
    execution_time: float


class ScenarioBacktestManager:
    """
    Manages multi-scenario backtesting with parallel execution
    and comprehensive result aggregation.

    This class coordinates with ScenarioManager to run backtests
    across multiple market scenarios and provides detailed analysis.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        data_manager: CLIDataManager,
        scenario_manager: ScenarioManager,
        strategy_manager: Optional[StrategyBacktestManager] = None,
        indicator_manager: Optional[IndicatorBacktestManager] = None,
        output_dir: Optional[Path] = None,
        max_workers: int = 4
    ):
        """
        Initialize the scenario backtest manager.

        Args:
            config_manager: ConfigManager for configuration access
            data_manager: CLI DataManager for data operations
            scenario_manager: ScenarioManager for scenario coordination
            strategy_manager: Optional StrategyBacktestManager
            indicator_manager: Optional IndicatorBacktestManager
            output_dir: Optional output directory for results
            max_workers: Maximum number of parallel workers
        """
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.scenario_manager = scenario_manager
        self.strategy_manager = strategy_manager
        self.indicator_manager = indicator_manager
        self.output_dir = output_dir or Path("./scenario_backtest_results")
        self.max_workers = max_workers

        # Results storage
        self.scenario_results: Dict[str, List[ScenarioBacktestResult]] = {}
        self.aggregated_results: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.execution_stats: Dict[str, Any] = {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ScenarioBacktestManager with {max_workers} workers")
        logger.info(f"Output directory: {self.output_dir}")

    def run_strategy_scenarios(
        self,
        strategy_name: str,
        days: int = 30,
        scenario_filter: Optional[List[str]] = None,
        parallel_execution: bool = True,
        include_real_data: bool = True
    ) -> Dict[str, ScenarioBacktestResult]:
        """
        Run a strategy across multiple scenarios.

        Args:
            strategy_name: Name of strategy to test
            days: Number of days for each scenario
            scenario_filter: Optional list of scenario names to run
            parallel_execution: Whether to run scenarios in parallel
            include_real_data: Whether to include real data scenario

        Returns:
            Dictionary mapping scenario names to results
        """
        if not self.strategy_manager:
            raise ValueError("StrategyBacktestManager not provided")

        logger.info(f"Running strategy '{strategy_name}' across scenarios")

        # Load strategy configuration
        self.strategy_manager.load_strategy_from_config(strategy_name)
        self.strategy_manager.initialize_strategy_components()

        # Get scenario configurations
        scenarios = self._get_scenarios_to_run(scenario_filter, include_real_data, days)

        # Create backtest function for strategy
        def backtest_function(scenario_type: ScenarioType, scenario_config: ScenarioConfig, target_name: str):
            return self._run_strategy_scenario_backtest(
                strategy_name=target_name,
                scenario_config=scenario_config,
                days=days
            )

        # Execute scenarios
        results = self._execute_scenarios(
            target_name=strategy_name,
            backtest_function=backtest_function,
            scenarios=scenarios,
            parallel_execution=parallel_execution
        )

        # Store results
        self.scenario_results[strategy_name] = list(results.values())

        # Generate aggregated analysis
        self._generate_strategy_scenario_analysis(strategy_name)

        logger.info(f"Completed {len(results)} scenario backtests for strategy: {strategy_name}")
        return results

    def run_indicator_scenarios(
        self,
        indicator_name: str,
        symbol: str,
        timeframe: str = "1h",
        days: int = 30,
        scenario_filter: Optional[List[str]] = None,
        parallel_execution: bool = True,
        include_real_data: bool = True
    ) -> Dict[str, ScenarioBacktestResult]:
        """
        Run an indicator across multiple scenarios.

        Args:
            indicator_name: Name of indicator to test
            symbol: Market symbol
            timeframe: Timeframe for testing
            days: Number of days for each scenario
            scenario_filter: Optional list of scenario names to run
            parallel_execution: Whether to run scenarios in parallel
            include_real_data: Whether to include real data scenario

        Returns:
            Dictionary mapping scenario names to results
        """
        if not self.indicator_manager:
            raise ValueError("IndicatorBacktestManager not provided")

        logger.info(f"Running indicator '{indicator_name}' across scenarios")

        # Get scenario configurations
        scenarios = self._get_scenarios_to_run(scenario_filter, include_real_data, days)

        # Create backtest function for indicator
        def backtest_function(scenario_type: ScenarioType, scenario_config: ScenarioConfig, target_name: str):
            return self._run_indicator_scenario_backtest(
                indicator_name=target_name,
                symbol=symbol,
                timeframe=timeframe,
                scenario_config=scenario_config,
                days=days
            )

        # Execute scenarios
        results = self._execute_scenarios(
            target_name=indicator_name,
            backtest_function=backtest_function,
            scenarios=scenarios,
            parallel_execution=parallel_execution
        )

        # Store results
        result_key = f"{indicator_name}_{symbol}_{timeframe}"
        self.scenario_results[result_key] = list(results.values())

        # Generate aggregated analysis
        self._generate_indicator_scenario_analysis(result_key)

        logger.info(f"Completed {len(results)} scenario backtests for indicator: {indicator_name}")
        return results

    def compare_scenarios_performance(
        self,
        target_names: List[str],
        target_type: str = "strategy",
        metric: str = "total_return_pct"
    ) -> Dict[str, Any]:
        """
        Compare performance across different scenarios for multiple targets.

        Args:
            target_names: List of strategy or indicator names to compare
            target_type: Type of target ("strategy" or "indicator")
            metric: Metric to compare by

        Returns:
            Dictionary with comparison analysis
        """
        comparison_data = {}

        for target_name in target_names:
            if target_type == "strategy":
                results = self.scenario_results.get(target_name, [])
            else:
                # For indicators, we might have multiple keys with different symbols/timeframes
                results = []
                for key, scenario_results in self.scenario_results.items():
                    if key.startswith(target_name):
                        results.extend(scenario_results)

            if results:
                comparison_data[target_name] = self._extract_scenario_metrics(results, metric)

        # Generate comparison analysis
        analysis = {
            'targets': target_names,
            'target_type': target_type,
            'metric': metric,
            'data': comparison_data,
            'ranking': self._rank_targets_by_scenario_performance(comparison_data, metric),
            'robustness_analysis': self._analyze_robustness(comparison_data),
            'scenario_correlation': self._analyze_scenario_correlation(comparison_data)
        }

        logger.info(f"Compared {len(target_names)} {target_type}s across scenarios by {metric}")
        return analysis

    def _get_scenarios_to_run(
        self,
        scenario_filter: Optional[List[str]],
        include_real_data: bool,
        days: int = 30
    ) -> List[ScenarioConfig]:
        """Get list of scenarios to run."""
        all_scenarios = list(self.scenario_manager.scenario_configs.values())

        if scenario_filter:
            scenarios = [s for s in all_scenarios if s.name in scenario_filter]
        else:
            scenarios = [s for s in all_scenarios if s.enabled]

        # Add real data scenario if requested
        if include_real_data:
            real_data_scenario = ScenarioConfig(
                scenario_type=ScenarioType.REAL_DATA,
                name="real_data",
                description="Real market data scenario",
                duration_days=days,
                parameters={},
                enabled=True
            )
            scenarios.append(real_data_scenario)

        return scenarios

    def _execute_scenarios(
        self,
        target_name: str,
        backtest_function: Callable,
        scenarios: List[ScenarioConfig],
        parallel_execution: bool
    ) -> Dict[str, ScenarioBacktestResult]:
        """Execute scenarios with optional parallel processing."""
        results = {}

        if parallel_execution and len(scenarios) > 1:
            logger.info(f"Running {len(scenarios)} scenarios in parallel")
            results = self._execute_scenarios_parallel(target_name, backtest_function, scenarios)
        else:
            logger.info(f"Running {len(scenarios)} scenarios sequentially")
            results = self._execute_scenarios_sequential(target_name, backtest_function, scenarios)

        return results

    def _execute_scenarios_parallel(
        self,
        target_name: str,
        backtest_function: Callable,
        scenarios: List[ScenarioConfig]
    ) -> Dict[str, ScenarioBacktestResult]:
        """Execute scenarios in parallel."""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all scenarios
            future_to_scenario = {
                executor.submit(
                    self._execute_single_scenario,
                    target_name,
                    backtest_function,
                    scenario
                ): scenario for scenario in scenarios
            }

            # Collect results as they complete
            for future in as_completed(future_to_scenario):
                scenario = future_to_scenario[future]
                try:
                    result = future.result()
                    results[scenario.name] = result
                    logger.info(f"Completed scenario: {scenario.name}")
                except Exception as e:
                    logger.error(f"Scenario {scenario.name} failed: {e}")

        return results

    def _execute_scenarios_sequential(
        self,
        target_name: str,
        backtest_function: Callable,
        scenarios: List[ScenarioConfig]
    ) -> Dict[str, ScenarioBacktestResult]:
        """Execute scenarios sequentially."""
        results = {}

        for scenario in scenarios:
            try:
                result = self._execute_single_scenario(target_name, backtest_function, scenario)
                results[scenario.name] = result
                logger.info(f"Completed scenario: {scenario.name}")
            except Exception as e:
                logger.error(f"Scenario {scenario.name} failed: {e}")

        return results

    def _execute_single_scenario(
        self,
        target_name: str,
        backtest_function: Callable,
        scenario: ScenarioConfig
    ) -> ScenarioBacktestResult:
        """Execute a single scenario backtest."""
        start_time = time.time()

        try:
            # Execute the backtest
            backtest_result = backtest_function(
                scenario_type=scenario.scenario_type,
                scenario_config=scenario,
                target_name=target_name
            )

            execution_time = time.time() - start_time

            # Create scenario backtest result
            scenario_result = ScenarioBacktestResult(
                scenario_name=scenario.name,
                scenario_type=scenario.scenario_type,
                backtest_result=backtest_result,
                scenario_metadata={
                    'scenario_description': scenario.description,
                    'scenario_parameters': scenario.parameters,
                    'target_name': target_name
                },
                execution_time=execution_time
            )

            return scenario_result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Scenario {scenario.name} failed after {execution_time:.2f}s: {e}")
            raise

    def _run_strategy_scenario_backtest(
        self,
        strategy_name: str,
        scenario_config: ScenarioConfig,
        days: int
    ) -> BacktestResult:
        """Run a strategy backtest for a specific scenario."""
        use_real_data = scenario_config.scenario_type == ScenarioType.REAL_DATA

        # For synthetic scenarios, we would generate appropriate data
        # For now, we'll use the basic backtest functionality
        result = self.strategy_manager.backtest_strategy(
            days=days,
            use_real_data=use_real_data,
            additional_params={
                'scenario_type': scenario_config.scenario_type.value,
                'scenario_name': scenario_config.name
            }
        )

        return result

    def _run_indicator_scenario_backtest(
        self,
        indicator_name: str,
        symbol: str,
        timeframe: str,
        scenario_config: ScenarioConfig,
        days: int
    ) -> BacktestResult:
        """Run an indicator backtest for a specific scenario."""
        use_real_data = scenario_config.scenario_type == ScenarioType.REAL_DATA

        result = self.indicator_manager.backtest_indicator(
            indicator_name=indicator_name,
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            use_real_data=use_real_data,
            strategy_params={
                'scenario_type': scenario_config.scenario_type.value,
                'scenario_name': scenario_config.name
            }
        )

        return result

    def _generate_strategy_scenario_analysis(self, strategy_name: str) -> None:
        """Generate aggregated analysis for strategy scenario results."""
        results = self.scenario_results.get(strategy_name, [])
        if not results:
            return

        analysis = {
            'strategy_name': strategy_name,
            'total_scenarios': len(results),
            'performance_by_scenario': {},
            'robustness_metrics': {},
            'best_scenario': None,
            'worst_scenario': None,
            'consistency_score': 0.0
        }

        # Analyze performance by scenario
        returns = []
        for result in results:
            scenario_name = result.scenario_name
            backtest_result = result.backtest_result

            performance = {
                'total_return_pct': backtest_result.metrics.get('total_return_pct', 0),
                'win_rate': backtest_result.metrics.get('win_rate', 0),
                'sharpe_ratio': backtest_result.metrics.get('sharpe_ratio', 0),
                'max_drawdown': backtest_result.metrics.get('max_drawdown', 0),
                'total_trades': backtest_result.metrics.get('total_trades', 0),
                'execution_time': result.execution_time
            }

            analysis['performance_by_scenario'][scenario_name] = performance
            returns.append(performance['total_return_pct'])

        # Calculate robustness metrics
        if returns:
            analysis['robustness_metrics'] = {
                'avg_return': sum(returns) / len(returns),
                'return_std': pd.Series(returns).std(),
                'min_return': min(returns),
                'max_return': max(returns),
                'positive_scenarios': sum(1 for r in returns if r > 0),
                'negative_scenarios': sum(1 for r in returns if r < 0)
            }

            # Find best and worst scenarios
            performance_data = analysis['performance_by_scenario']
            analysis['best_scenario'] = max(performance_data.keys(),
                                          key=lambda k: performance_data[k]['total_return_pct'])
            analysis['worst_scenario'] = min(performance_data.keys(),
                                           key=lambda k: performance_data[k]['total_return_pct'])

            # Calculate consistency score (inverse of coefficient of variation)
            cv = abs(analysis['robustness_metrics']['return_std'] /
                    analysis['robustness_metrics']['avg_return']) if analysis['robustness_metrics']['avg_return'] != 0 else float('inf')
            analysis['consistency_score'] = 1 / (1 + cv)

        self.aggregated_results[strategy_name] = analysis
        logger.debug(f"Generated scenario analysis for strategy: {strategy_name}")

    def _generate_indicator_scenario_analysis(self, result_key: str) -> None:
        """Generate aggregated analysis for indicator scenario results."""
        results = self.scenario_results.get(result_key, [])
        if not results:
            return

        # Similar analysis as strategy but adapted for indicators
        analysis = {
            'indicator_key': result_key,
            'total_scenarios': len(results),
            'performance_by_scenario': {},
            'robustness_metrics': {},
            'best_scenario': None,
            'worst_scenario': None,
            'consistency_score': 0.0
        }

        returns = []
        for result in results:
            scenario_name = result.scenario_name
            backtest_result = result.backtest_result

            performance = {
                'total_return_pct': backtest_result.metrics.get('total_return_pct', 0),
                'win_rate': backtest_result.metrics.get('win_rate', 0),
                'sharpe_ratio': backtest_result.metrics.get('sharpe_ratio', 0),
                'max_drawdown': backtest_result.metrics.get('max_drawdown', 0),
                'total_trades': backtest_result.metrics.get('total_trades', 0),
                'execution_time': result.execution_time
            }

            analysis['performance_by_scenario'][scenario_name] = performance
            returns.append(performance['total_return_pct'])

        # Calculate robustness metrics (same as strategy)
        if returns:
            analysis['robustness_metrics'] = {
                'avg_return': sum(returns) / len(returns),
                'return_std': pd.Series(returns).std(),
                'min_return': min(returns),
                'max_return': max(returns),
                'positive_scenarios': sum(1 for r in returns if r > 0),
                'negative_scenarios': sum(1 for r in returns if r < 0)
            }

            performance_data = analysis['performance_by_scenario']
            analysis['best_scenario'] = max(performance_data.keys(),
                                          key=lambda k: performance_data[k]['total_return_pct'])
            analysis['worst_scenario'] = min(performance_data.keys(),
                                           key=lambda k: performance_data[k]['total_return_pct'])

            cv = abs(analysis['robustness_metrics']['return_std'] /
                    analysis['robustness_metrics']['avg_return']) if analysis['robustness_metrics']['avg_return'] != 0 else float('inf')
            analysis['consistency_score'] = 1 / (1 + cv)

        self.aggregated_results[result_key] = analysis
        logger.debug(f"Generated scenario analysis for indicator: {result_key}")

    def _extract_scenario_metrics(
        self,
        results: List[ScenarioBacktestResult],
        metric: str
    ) -> Dict[str, float]:
        """Extract a specific metric from scenario results."""
        metrics = {}
        for result in results:
            metrics[result.scenario_name] = result.backtest_result.metrics.get(metric, 0)
        return metrics

    def _rank_targets_by_scenario_performance(
        self,
        comparison_data: Dict[str, Dict[str, float]],
        metric: str
    ) -> List[Tuple[str, float]]:
        """Rank targets by average scenario performance."""
        rankings = []

        for target_name, scenario_metrics in comparison_data.items():
            avg_performance = sum(scenario_metrics.values()) / len(scenario_metrics) if scenario_metrics else 0
            rankings.append((target_name, avg_performance))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def _analyze_robustness(self, comparison_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze robustness across targets and scenarios."""
        robustness_analysis = {}

        for target_name, scenario_metrics in comparison_data.items():
            if not scenario_metrics:
                continue

            values = list(scenario_metrics.values())
            robustness_analysis[target_name] = {
                'avg_performance': sum(values) / len(values),
                'std_performance': pd.Series(values).std(),
                'min_performance': min(values),
                'max_performance': max(values),
                'positive_scenarios': sum(1 for v in values if v > 0),
                'coefficient_of_variation': pd.Series(values).std() / abs(pd.Series(values).mean()) if pd.Series(values).mean() != 0 else float('inf')
            }

        return robustness_analysis

    def _analyze_scenario_correlation(self, comparison_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze correlation between different scenarios."""
        if len(comparison_data) < 2:
            return {}

        # Create a DataFrame for correlation analysis
        df_data = {}
        for target_name, scenario_metrics in comparison_data.items():
            df_data[target_name] = scenario_metrics

        df = pd.DataFrame(df_data)

        if df.empty:
            return {}

        # Calculate correlation matrix between scenarios
        scenario_correlation = df.T.corr()

        return {
            'correlation_matrix': scenario_correlation.to_dict(),
            'highly_correlated_scenarios': self._find_highly_correlated_scenarios(scenario_correlation),
            'uncorrelated_scenarios': self._find_uncorrelated_scenarios(scenario_correlation)
        }

    def _find_highly_correlated_scenarios(self, correlation_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """Find highly correlated scenario pairs."""
        high_correlations = []

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    high_correlations.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr
                    ))

        return high_correlations

    def _find_uncorrelated_scenarios(self, correlation_matrix: pd.DataFrame, threshold: float = 0.2) -> List[Tuple[str, str, float]]:
        """Find uncorrelated scenario pairs."""
        low_correlations = []

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) <= threshold:
                    low_correlations.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr
                    ))

        return low_correlations

    def get_scenario_summary(self, target_name: str) -> Dict[str, Any]:
        """Get summary of scenario results for a target."""
        return self.aggregated_results.get(target_name, {})

    def export_scenario_results(self, target_name: str, output_path: Optional[Path] = None) -> Path:
        """Export scenario results to file."""
        if target_name not in self.scenario_results:
            raise ValueError(f"No scenario results found for: {target_name}")

        if not output_path:
            output_path = self.output_dir / f"{target_name}_scenario_results.json"

        # Prepare data for export
        export_data = {
            'target_name': target_name,
            'scenario_results': [],
            'aggregated_analysis': self.aggregated_results.get(target_name, {})
        }

        for result in self.scenario_results[target_name]:
            export_data['scenario_results'].append({
                'scenario_name': result.scenario_name,
                'scenario_type': result.scenario_type.value,
                'execution_time': result.execution_time,
                'metrics': result.backtest_result.metrics,
                'metadata': result.scenario_metadata
            })

        # Save to file
        import json
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported scenario results to: {output_path}")
        return output_path

    def clear_results(self) -> None:
        """Clear all scenario results and analysis."""
        self.scenario_results.clear()
        self.aggregated_results.clear()
        self.execution_stats.clear()
        logger.info("Cleared all scenario backtest results")
