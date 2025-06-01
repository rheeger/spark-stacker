"""
Comparison Manager

This module provides centralized comparison logic:
- Centralize all comparison logic
- Handle strategy-to-strategy comparisons
- Handle indicator-to-indicator comparisons
- Handle cross-type comparisons
- Add statistical comparison features
- Add comparison result export functionality
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from app.backtesting.backtest_engine import BacktestResult
from core.config_manager import ConfigManager
from core.data_manager import DataManager as CLIDataManager
from managers.indicator_backtest_manager import IndicatorBacktestManager
from managers.scenario_backtest_manager import ScenarioBacktestManager
from managers.strategy_backtest_manager import StrategyBacktestManager
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ComparisonTarget:
    """Represents a target for comparison."""
    name: str
    type: str  # 'strategy' or 'indicator'
    metadata: Dict[str, Any]


@dataclass
class ComparisonResult:
    """Result of a comparison operation."""
    targets: List[ComparisonTarget]
    comparison_type: str
    metrics_comparison: Dict[str, Dict[str, float]]
    rankings: Dict[str, List[Tuple[str, float]]]
    statistical_analysis: Dict[str, Any]
    robustness_analysis: Dict[str, Any]
    execution_time: float
    timestamp: datetime


class ComparisonManager:
    """
    Centralizes all comparison logic for strategies, indicators,
    and cross-type comparisons with statistical analysis.
    """

    # Standard metrics for comparison
    STANDARD_METRICS = [
        'total_return_pct',
        'win_rate',
        'sharpe_ratio',
        'max_drawdown',
        'total_trades',
        'profit_factor',
        'avg_trade_duration'
    ]

    def __init__(
        self,
        config_manager: ConfigManager,
        data_manager: CLIDataManager,
        strategy_manager: Optional[StrategyBacktestManager] = None,
        indicator_manager: Optional[IndicatorBacktestManager] = None,
        scenario_manager: Optional[ScenarioBacktestManager] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the comparison manager.

        Args:
            config_manager: ConfigManager for configuration access
            data_manager: CLI DataManager for data operations
            strategy_manager: Optional StrategyBacktestManager
            indicator_manager: Optional IndicatorBacktestManager
            scenario_manager: Optional ScenarioBacktestManager
            output_dir: Optional output directory for results
        """
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.strategy_manager = strategy_manager
        self.indicator_manager = indicator_manager
        self.scenario_manager = scenario_manager
        self.output_dir = output_dir or Path("./comparison_results")

        # Results storage
        self.comparison_results: Dict[str, ComparisonResult] = {}
        self.cached_backtests: Dict[str, BacktestResult] = {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ComparisonManager with output dir: {self.output_dir}")

    def compare_strategies(
        self,
        strategy_names: List[str],
        test_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        include_scenarios: bool = False
    ) -> ComparisonResult:
        """
        Compare multiple strategies.

        Args:
            strategy_names: List of strategy names to compare
            test_params: Optional parameters for backtesting (days, symbol, etc.)
            metrics: Optional list of metrics to compare
            include_scenarios: Whether to include scenario-based comparison

        Returns:
            ComparisonResult with detailed analysis
        """
        if not self.strategy_manager:
            raise ValueError("StrategyBacktestManager not provided")

        logger.info(f"Comparing {len(strategy_names)} strategies")
        start_time = time.time()

        # Set default parameters
        test_params = test_params or {'days': 30}
        metrics = metrics or self.STANDARD_METRICS

        # Create comparison targets
        targets = []
        for strategy_name in strategy_names:
            strategy_config = self.config_manager.get_strategy_config(strategy_name)
            targets.append(ComparisonTarget(
                name=strategy_name,
                type='strategy',
                metadata={
                    'market': strategy_config.get('market', 'Unknown'),
                    'exchange': strategy_config.get('exchange', 'Unknown'),
                    'timeframe': strategy_config.get('timeframe', 'Unknown'),
                    'indicators': list(strategy_config.get('indicators', {}).keys())
                }
            ))

        # Run backtests for all strategies
        backtest_results = {}
        for strategy_name in strategy_names:
            result = self._run_strategy_backtest(strategy_name, test_params)
            if result:
                backtest_results[strategy_name] = result

        # Include scenario-based comparison if requested
        scenario_results = {}
        if include_scenarios and self.scenario_manager:
            include_real_data = test_params.get('use_real_data', True)
            scenario_filter = test_params.get('scenario_filter', None)
            for strategy_name in strategy_names:
                scenario_results[strategy_name] = self.scenario_manager.run_strategy_scenarios(
                    strategy_name=strategy_name,
                    days=test_params.get('days', 30),
                    parallel_execution=True,
                    scenario_filter=scenario_filter,
                    include_real_data=include_real_data
                )

        # Generate comparison analysis
        comparison_result = self._generate_comparison_analysis(
            targets=targets,
            backtest_results=backtest_results,
            scenario_results=scenario_results,
            metrics=metrics,
            comparison_type='strategy-to-strategy',
            execution_time=time.time() - start_time
        )

        # Store result
        result_key = f"strategies_{'_'.join(strategy_names)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.comparison_results[result_key] = comparison_result

        logger.info(f"Strategy comparison completed in {comparison_result.execution_time:.2f}s")
        return comparison_result

    def compare_indicators(
        self,
        indicator_names: List[str],
        symbol: str,
        timeframe: str = "1h",
        test_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        include_scenarios: bool = False
    ) -> ComparisonResult:
        """
        Compare multiple indicators.

        Args:
            indicator_names: List of indicator names to compare
            symbol: Market symbol for testing
            timeframe: Timeframe for testing
            test_params: Optional parameters for backtesting
            metrics: Optional list of metrics to compare
            include_scenarios: Whether to include scenario-based comparison

        Returns:
            ComparisonResult with detailed analysis
        """
        if not self.indicator_manager:
            raise ValueError("IndicatorBacktestManager not provided")

        logger.info(f"Comparing {len(indicator_names)} indicators")
        start_time = time.time()

        # Set default parameters
        test_params = test_params or {'days': 30}
        metrics = metrics or self.STANDARD_METRICS

        # Create comparison targets
        targets = []
        for indicator_name in indicator_names:
            indicator = self.indicator_manager.get_indicator(indicator_name)
            targets.append(ComparisonTarget(
                name=indicator_name,
                type='indicator',
                metadata={
                    'indicator_type': getattr(indicator, 'indicator_type', 'Unknown'),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'parameters': getattr(indicator, 'parameters', {})
                }
            ))

        # Run backtests for all indicators
        backtest_results = {}
        for indicator_name in indicator_names:
            result = self._run_indicator_backtest(
                indicator_name=indicator_name,
                symbol=symbol,
                timeframe=timeframe,
                test_params=test_params
            )
            if result:
                backtest_results[indicator_name] = result

        # Include scenario-based comparison if requested
        scenario_results = {}
        if include_scenarios and self.scenario_manager:
            for indicator_name in indicator_names:
                scenario_results[indicator_name] = self.scenario_manager.run_indicator_scenarios(
                    indicator_name=indicator_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    days=test_params.get('days', 30),
                    parallel_execution=True
                )

        # Generate comparison analysis
        comparison_result = self._generate_comparison_analysis(
            targets=targets,
            backtest_results=backtest_results,
            scenario_results=scenario_results,
            metrics=metrics,
            comparison_type='indicator-to-indicator',
            execution_time=time.time() - start_time
        )

        # Store result
        result_key = f"indicators_{'_'.join(indicator_names)}_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.comparison_results[result_key] = comparison_result

        logger.info(f"Indicator comparison completed in {comparison_result.execution_time:.2f}s")
        return comparison_result

    def compare_cross_type(
        self,
        strategies: List[str],
        indicators: List[str],
        symbol: str,
        timeframe: str = "1h",
        test_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None
    ) -> ComparisonResult:
        """
        Compare strategies against indicators (cross-type comparison).

        Args:
            strategies: List of strategy names
            indicators: List of indicator names
            symbol: Market symbol for testing
            timeframe: Timeframe for testing
            test_params: Optional parameters for backtesting
            metrics: Optional list of metrics to compare

        Returns:
            ComparisonResult with cross-type analysis
        """
        if not self.strategy_manager or not self.indicator_manager:
            raise ValueError("Both StrategyBacktestManager and IndicatorBacktestManager required for cross-type comparison")

        logger.info(f"Cross-type comparison: {len(strategies)} strategies vs {len(indicators)} indicators")
        start_time = time.time()

        # Set default parameters
        test_params = test_params or {'days': 30}
        metrics = metrics or self.STANDARD_METRICS

        # Create comparison targets
        targets = []

        # Add strategies
        for strategy_name in strategies:
            strategy_config = self.config_manager.get_strategy_config(strategy_name)
            targets.append(ComparisonTarget(
                name=strategy_name,
                type='strategy',
                metadata={
                    'market': strategy_config.get('market', 'Unknown'),
                    'exchange': strategy_config.get('exchange', 'Unknown'),
                    'timeframe': strategy_config.get('timeframe', 'Unknown'),
                    'indicators': list(strategy_config.get('indicators', {}).keys())
                }
            ))

        # Add indicators
        for indicator_name in indicators:
            indicator = self.indicator_manager.get_indicator(indicator_name)
            targets.append(ComparisonTarget(
                name=indicator_name,
                type='indicator',
                metadata={
                    'indicator_type': getattr(indicator, 'indicator_type', 'Unknown'),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'parameters': getattr(indicator, 'parameters', {})
                }
            ))

        # Run backtests
        backtest_results = {}

        # Strategy backtests
        for strategy_name in strategies:
            result = self._run_strategy_backtest(strategy_name, test_params)
            if result:
                backtest_results[strategy_name] = result

        # Indicator backtests
        for indicator_name in indicators:
            result = self._run_indicator_backtest(
                indicator_name=indicator_name,
                symbol=symbol,
                timeframe=timeframe,
                test_params=test_params
            )
            if result:
                backtest_results[indicator_name] = result

        # Generate comparison analysis
        comparison_result = self._generate_comparison_analysis(
            targets=targets,
            backtest_results=backtest_results,
            scenario_results={},
            metrics=metrics,
            comparison_type='cross-type',
            execution_time=time.time() - start_time
        )

        # Store result
        result_key = f"crosstype_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.comparison_results[result_key] = comparison_result

        logger.info(f"Cross-type comparison completed in {comparison_result.execution_time:.2f}s")
        return comparison_result

    def _run_strategy_backtest(
        self,
        strategy_name: str,
        test_params: Dict[str, Any]
    ) -> Optional[BacktestResult]:
        """Run a backtest for a strategy."""
        try:
            # Check cache first
            cache_key = f"strategy_{strategy_name}_{hash(str(test_params))}"
            if cache_key in self.cached_backtests:
                return self.cached_backtests[cache_key]

            # Load and run strategy
            self.strategy_manager.load_strategy_from_config(strategy_name)
            self.strategy_manager.initialize_strategy_components()

            result = self.strategy_manager.backtest_strategy(
                days=test_params.get('days', 30),
                use_real_data=test_params.get('use_real_data', True),
                leverage=test_params.get('leverage', 1.0)
            )

            # Cache result
            if result:
                self.cached_backtests[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Failed to run strategy backtest for {strategy_name}: {e}")
            return None

    def _run_indicator_backtest(
        self,
        indicator_name: str,
        symbol: str,
        timeframe: str,
        test_params: Dict[str, Any]
    ) -> Optional[BacktestResult]:
        """Run a backtest for an indicator."""
        try:
            # Check cache first
            cache_key = f"indicator_{indicator_name}_{symbol}_{timeframe}_{hash(str(test_params))}"
            if cache_key in self.cached_backtests:
                return self.cached_backtests[cache_key]

            result = self.indicator_manager.backtest_indicator(
                indicator_name=indicator_name,
                symbol=symbol,
                timeframe=timeframe,
                days=test_params.get('days', 30),
                use_real_data=test_params.get('use_real_data', True),
                leverage=test_params.get('leverage', 1.0)
            )

            # Cache result
            if result:
                self.cached_backtests[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Failed to run indicator backtest for {indicator_name}: {e}")
            return None

    def _generate_comparison_analysis(
        self,
        targets: List[ComparisonTarget],
        backtest_results: Dict[str, BacktestResult],
        scenario_results: Dict[str, Any],
        metrics: List[str],
        comparison_type: str,
        execution_time: float
    ) -> ComparisonResult:
        """Generate comprehensive comparison analysis."""

        # Extract metrics for comparison
        metrics_comparison = {}
        for target in targets:
            target_name = target.name
            if target_name in backtest_results:
                result = backtest_results[target_name]
                metrics_comparison[target_name] = {}
                for metric in metrics:
                    metrics_comparison[target_name][metric] = result.metrics.get(metric, 0)

        # Generate rankings for each metric
        rankings = {}
        for metric in metrics:
            rankings[metric] = self._rank_targets_by_metric(metrics_comparison, metric)

        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(metrics_comparison, metrics)

        # Robustness analysis
        robustness_analysis = self._analyze_robustness(
            metrics_comparison, scenario_results
        )

        return ComparisonResult(
            targets=targets,
            comparison_type=comparison_type,
            metrics_comparison=metrics_comparison,
            rankings=rankings,
            statistical_analysis=statistical_analysis,
            robustness_analysis=robustness_analysis,
            execution_time=execution_time,
            timestamp=datetime.now()
        )

    def _rank_targets_by_metric(
        self,
        metrics_comparison: Dict[str, Dict[str, float]],
        metric: str
    ) -> List[Tuple[str, float]]:
        """Rank targets by a specific metric."""
        rankings = []
        for target_name, target_metrics in metrics_comparison.items():
            metric_value = target_metrics.get(metric, 0)
            rankings.append((target_name, metric_value))

        # Sort by metric value (descending for most metrics, ascending for drawdown)
        reverse_sort = metric != 'max_drawdown'
        rankings.sort(key=lambda x: x[1], reverse=reverse_sort)
        return rankings

    def _perform_statistical_analysis(
        self,
        metrics_comparison: Dict[str, Dict[str, float]],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Perform statistical analysis on the comparison results."""
        analysis = {}

        for metric in metrics:
            values = [target_metrics.get(metric, 0) for target_metrics in metrics_comparison.values()]

            if len(values) < 2:
                continue

            analysis[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'coefficient_of_variation': np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else float('inf')
            }

            # Perform normality test
            if len(values) >= 3:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(values)
                    analysis[metric]['normality_test'] = {
                        'shapiro_statistic': shapiro_stat,
                        'shapiro_p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    }
                except:
                    analysis[metric]['normality_test'] = None

        # Correlation analysis between metrics
        if len(metrics) > 1 and len(metrics_comparison) > 2:
            correlation_matrix = self._calculate_metric_correlations(metrics_comparison, metrics)
            analysis['metric_correlations'] = correlation_matrix

        return analysis

    def _calculate_metric_correlations(
        self,
        metrics_comparison: Dict[str, Dict[str, float]],
        metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between different metrics."""
        # Create DataFrame for correlation analysis
        data = {}
        for metric in metrics:
            data[metric] = [target_metrics.get(metric, 0) for target_metrics in metrics_comparison.values()]

        df = pd.DataFrame(data)
        correlation_matrix = df.corr()

        return correlation_matrix.to_dict()

    def _analyze_robustness(
        self,
        metrics_comparison: Dict[str, Dict[str, float]],
        scenario_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze robustness of targets across different scenarios."""
        robustness_analysis = {}

        # Basic robustness based on standard metrics
        for target_name, target_metrics in metrics_comparison.items():
            returns = target_metrics.get('total_return_pct', 0)
            sharpe = target_metrics.get('sharpe_ratio', 0)
            drawdown = target_metrics.get('max_drawdown', 0)
            win_rate = target_metrics.get('win_rate', 0)

            # Calculate a composite robustness score
            robustness_score = self._calculate_robustness_score(returns, sharpe, drawdown, win_rate)

            robustness_analysis[target_name] = {
                'robustness_score': robustness_score,
                'risk_adjusted_return': returns / max(abs(drawdown), 1),  # Avoid division by zero
                'consistency_indicator': win_rate * (1 - abs(drawdown) / 100)
            }

        # Enhanced robustness analysis if scenario results are available
        if scenario_results:
            for target_name, scenarios in scenario_results.items():
                if target_name in robustness_analysis:
                    scenario_returns = []
                    for scenario_result in scenarios.values():
                        if hasattr(scenario_result, 'backtest_result'):
                            scenario_returns.append(
                                scenario_result.backtest_result.metrics.get('total_return_pct', 0)
                            )

                    if scenario_returns:
                        robustness_analysis[target_name]['scenario_consistency'] = {
                            'avg_scenario_return': np.mean(scenario_returns),
                            'scenario_std': np.std(scenario_returns),
                            'positive_scenarios': sum(1 for r in scenario_returns if r > 0),
                            'negative_scenarios': sum(1 for r in scenario_returns if r <= 0),
                            'scenario_robustness_score': 1 / (1 + np.std(scenario_returns) / max(abs(np.mean(scenario_returns)), 1))
                        }

        return robustness_analysis

    def _calculate_robustness_score(
        self,
        returns: float,
        sharpe: float,
        drawdown: float,
        win_rate: float
    ) -> float:
        """Calculate a composite robustness score."""
        # Normalize components to 0-1 scale
        return_score = max(0, min(1, (returns + 50) / 100))  # Assumes -50% to +50% range
        sharpe_score = max(0, min(1, (sharpe + 2) / 4))      # Assumes -2 to +2 Sharpe range
        drawdown_score = max(0, 1 - abs(drawdown) / 50)      # Assumes 0% to 50% drawdown range
        win_rate_score = win_rate / 100                       # Already in 0-1 range

        # Weighted combination
        robustness_score = (
            0.3 * return_score +
            0.3 * sharpe_score +
            0.2 * drawdown_score +
            0.2 * win_rate_score
        )

        return robustness_score

    def get_comparison_summary(self, result_key: str) -> Dict[str, Any]:
        """Get summary of a comparison result."""
        if result_key not in self.comparison_results:
            raise ValueError(f"Comparison result not found: {result_key}")

        result = self.comparison_results[result_key]

        # Generate summary
        summary = {
            'comparison_type': result.comparison_type,
            'targets_count': len(result.targets),
            'execution_time': result.execution_time,
            'timestamp': result.timestamp.isoformat(),
            'best_performers': {},
            'worst_performers': {},
            'statistical_summary': {}
        }

        # Find best and worst performers for key metrics
        key_metrics = ['total_return_pct', 'sharpe_ratio', 'win_rate']
        for metric in key_metrics:
            if metric in result.rankings:
                rankings = result.rankings[metric]
                if rankings:
                    summary['best_performers'][metric] = rankings[0]
                    summary['worst_performers'][metric] = rankings[-1]

        # Statistical summary
        for metric, stats in result.statistical_analysis.items():
            if isinstance(stats, dict) and 'mean' in stats:
                summary['statistical_summary'][metric] = {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'range': stats['max'] - stats['min']
                }

        return summary

    def export_comparison_result(
        self,
        result_key: str,
        output_path: Optional[Path] = None,
        include_detailed_data: bool = True
    ) -> Path:
        """Export comparison result to file."""
        if result_key not in self.comparison_results:
            raise ValueError(f"Comparison result not found: {result_key}")

        result = self.comparison_results[result_key]

        if not output_path:
            output_path = self.output_dir / f"{result_key}_comparison.json"

        # Prepare export data
        export_data = {
            'comparison_info': {
                'type': result.comparison_type,
                'timestamp': result.timestamp.isoformat(),
                'execution_time': result.execution_time,
                'targets_count': len(result.targets)
            },
            'targets': [
                {
                    'name': target.name,
                    'type': target.type,
                    'metadata': target.metadata
                } for target in result.targets
            ],
            'metrics_comparison': result.metrics_comparison,
            'rankings': result.rankings,
            'statistical_analysis': result.statistical_analysis,
            'robustness_analysis': result.robustness_analysis
        }

        # Save to file
        import json
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported comparison result to: {output_path}")
        return output_path

    def clear_cache(self) -> None:
        """Clear cached backtests and comparison results."""
        self.cached_backtests.clear()
        self.comparison_results.clear()
        logger.info("Cleared comparison cache")

    def get_available_comparisons(self) -> List[str]:
        """Get list of available comparison result keys."""
        return list(self.comparison_results.keys())

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_backtests': len(self.cached_backtests),
            'comparison_results': len(self.comparison_results),
            'total_memory_usage': f"{len(str(self.cached_backtests)) + len(str(self.comparison_results))} chars"
        }

    def run_strategy_comparison(
        self,
        strategy_names: List[str],
        scenarios: List[str],
        days: int = 30
    ) -> ComparisonResult:
        """
        Run strategy comparison with specific scenarios.

        This method provides a convenient interface for running strategy comparisons
        with scenario-based testing, matching the expected interface from strategy commands.

        Args:
            strategy_names: List of strategy names to compare
            scenarios: List of scenario names to test (e.g., ["bull", "bear", "real"])
            days: Number of days to test (default: 30)

        Returns:
            ComparisonResult with strategy comparison data including scenario results
        """
        # Determine if real data scenario is included
        include_real_data = "real" in scenarios

        # Set up test parameters
        test_params = {
            'days': days,
            'use_real_data': include_real_data,
            'scenario_filter': scenarios
        }

        # Initialize scenario manager if not already done
        if not self.scenario_manager and scenarios:
            from cli.core.scenario_manager import ScenarioManager
            self.scenario_manager = ScenarioBacktestManager(
                config_manager=self.config_manager,
                data_manager=self.data_manager,
                scenario_manager=ScenarioManager(
                    data_manager=self.data_manager,
                    config_manager=self.config_manager,
                    default_duration_days=days
                ),
                strategy_manager=self.strategy_manager,
                output_dir=self.output_dir
            )

        # Run the comparison with scenarios
        return self.compare_strategies(
            strategy_names=strategy_names,
            test_params=test_params,
            metrics=self.STANDARD_METRICS,
            include_scenarios=True if scenarios else False
        )
