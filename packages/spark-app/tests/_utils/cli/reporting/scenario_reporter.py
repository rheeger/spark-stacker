"""
Multi-Scenario Report Generation Module

This module handles multi-scenario reporting and analysis including:
- Scenario performance comparison tables
- Scenario robustness analysis
- Scenario-specific visualizations
- Scenario correlation analysis
- Scenario optimization recommendations
- Cross-scenario statistical analysis
"""

import json
import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.config_manager import ConfigManager
from core.scenario_manager import ScenarioManager

logger = logging.getLogger(__name__)


class ScenarioReporter:
    """
    Handles multi-scenario report generation and analysis.

    Centralizes multi-scenario reporting logic including performance comparison,
    robustness analysis, and scenario-specific insights.
    """

    def __init__(self, config_manager: ConfigManager, scenario_manager: ScenarioManager):
        """
        Initialize the ScenarioReporter.

        Args:
            config_manager: ConfigManager instance for configuration access
            scenario_manager: ScenarioManager for scenario coordination
        """
        self.config_manager = config_manager
        self.scenario_manager = scenario_manager

        # Define standard scenario types
        self.scenario_types = [
            "bull_market",
            "bear_market",
            "sideways_market",
            "high_volatility",
            "low_volatility",
            "choppy_market",
            "gap_heavy",
            "real_data"
        ]

    def generate_multi_scenario_report(
        self,
        strategy_name: str,
        scenario_results: Dict[str, Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive multi-scenario report.

        Args:
            strategy_name: Name of the strategy
            scenario_results: Dictionary of results keyed by scenario name
            output_path: Optional path to save the report

        Returns:
            Dictionary containing the complete multi-scenario report
        """
        logger.info(f"Generating multi-scenario report for strategy: {strategy_name}")

        try:
            # Convert ScenarioBacktestResult objects to expected dictionary format
            converted_results = self._convert_scenario_results(scenario_results)

            report = {
                "strategy_name": strategy_name,
                "report_type": "multi_scenario_analysis",
                "generated_at": datetime.now().isoformat(),
                "scenarios_analyzed": list(converted_results.keys()),
                "strategy_configuration": self._get_strategy_configuration(strategy_name),
                "scenario_performance_comparison": self._compare_scenario_performance(converted_results),
                "robustness_analysis": self._analyze_strategy_robustness(converted_results),
                "scenario_correlation_analysis": self._analyze_scenario_correlations(converted_results),
                "worst_case_analysis": self._analyze_worst_case_scenarios(converted_results),
                "adaptability_analysis": self._analyze_strategy_adaptability(converted_results),
                "scenario_rankings": self._rank_scenarios_by_performance(converted_results),
                "optimization_recommendations": self._generate_scenario_optimizations(converted_results),
                "scenario_insights": self._generate_scenario_insights(converted_results),
                "visualization_data": self._prepare_visualization_data(converted_results)
            }

            if output_path:
                self._save_report(report, output_path)

            return report

        except Exception as e:
            logger.error(f"Error generating multi-scenario report for {strategy_name}: {e}")
            raise

    def _convert_scenario_results(self, scenario_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Convert ScenarioBacktestResult objects to expected dictionary format."""
        converted = {}

        for scenario_name, result in scenario_results.items():
            # Check if result is a ScenarioBacktestResult object
            if hasattr(result, 'backtest_result'):
                backtest_result = result.backtest_result
                # Skip scenarios that failed to produce results
                if not backtest_result or not hasattr(backtest_result, 'metrics'):
                    logger.warning(f"Skipping scenario {scenario_name} - no valid backtest results")
                    continue

                converted[scenario_name] = {
                    "performance_metrics": {
                        "total_return": backtest_result.metrics.get('total_return_pct', 0),
                        "win_rate": backtest_result.metrics.get('win_rate', 0),
                        "profit_factor": backtest_result.metrics.get('profit_factor', 0),
                        "max_drawdown": backtest_result.metrics.get('max_drawdown', 0),
                        "sharpe_ratio": backtest_result.metrics.get('sharpe_ratio', 0),
                        "total_trades": backtest_result.metrics.get('total_trades', 0),
                        "total_return_pct": backtest_result.metrics.get('total_return_pct', 0)
                    },
                    "trades": backtest_result.trades,
                    "equity_curve": getattr(backtest_result, 'equity_curve', []),
                    "scenario_metadata": result.scenario_metadata,
                    "execution_time": result.execution_time
                }
            else:
                # If it's already a dictionary, use it as is
                converted[scenario_name] = result

        # If no valid results, raise an error with helpful message
        if not converted:
            raise ValueError("No valid scenario results to generate report. All scenarios failed to produce backtest results.")

        return converted

    def generate_cross_scenario_comparison_report(
        self,
        strategies: Dict[str, Dict[str, Dict[str, Any]]],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate comparison report across multiple strategies and scenarios.

        Args:
            strategies: Dictionary of {strategy_name: {scenario_name: results}}
            output_path: Optional path to save the report

        Returns:
            Dictionary containing the cross-scenario comparison report
        """
        logger.info(f"Generating cross-scenario comparison report for {len(strategies)} strategies")

        try:
            report = {
                "report_type": "cross_scenario_comparison",
                "generated_at": datetime.now().isoformat(),
                "strategies_analyzed": list(strategies.keys()),
                "scenarios_analyzed": self._get_common_scenarios(strategies),
                "strategy_robustness_comparison": self._compare_strategy_robustness(strategies),
                "scenario_difficulty_analysis": self._analyze_scenario_difficulty(strategies),
                "best_performers_by_scenario": self._identify_best_performers_by_scenario(strategies),
                "consistency_rankings": self._rank_strategies_by_consistency(strategies),
                "diversification_analysis": self._analyze_cross_scenario_diversification(strategies),
                "portfolio_recommendations": self._recommend_scenario_aware_portfolio(strategies)
            }

            if output_path:
                self._save_report(report, output_path)

            return report

        except Exception as e:
            logger.error(f"Error generating cross-scenario comparison report: {e}")
            raise

    def _get_strategy_configuration(self, strategy_name: str) -> Dict[str, Any]:
        """Get strategy configuration for context."""
        try:
            config = self.config_manager.get_strategy_config(strategy_name)
            return {
                "market": config.get("market"),
                "exchange": config.get("exchange"),
                "timeframe": config.get("timeframe"),
                "indicators": config.get("indicators", []),  # Fix: indicators is a list, not a dict
                "position_sizing": config.get("position_sizing", {}),
                "risk_management": {
                    "stop_loss": config.get("stop_loss"),
                    "take_profit": config.get("take_profit")
                }
            }
        except Exception as e:
            logger.warning(f"Could not get configuration for strategy {strategy_name}: {e}")
            return {"error": str(e)}

    def _compare_scenario_performance(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance metrics across all scenarios."""
        metrics = ["total_return", "win_rate", "profit_factor", "max_drawdown", "sharpe_ratio", "total_trades"]

        comparison = {
            "performance_matrix": {},
            "scenario_rankings": {},
            "performance_statistics": {},
            "unified_metrics_table": {}
        }

        # Create performance matrix
        for metric in metrics:
            comparison["performance_matrix"][metric] = {}
            values = []

            for scenario_name, results in scenario_results.items():
                performance = results.get("performance_metrics", {})
                value = performance.get(metric, 0)
                comparison["performance_matrix"][metric][scenario_name] = value
                values.append(value)

            # Calculate statistics for each metric
            if values:
                comparison["performance_statistics"][metric] = {
                    "best_scenario": max(scenario_results.items(),
                                       key=lambda x: x[1].get("performance_metrics", {}).get(metric, 0))[0],
                    "worst_scenario": min(scenario_results.items(),
                                        key=lambda x: x[1].get("performance_metrics", {}).get(metric, 0))[0],
                    "average": round(statistics.mean(values), 4),
                    "std_dev": round(statistics.stdev(values) if len(values) > 1 else 0, 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "range": round(max(values) - min(values), 4)
                }

        # Create unified metrics table for easy comparison
        for scenario_name, results in scenario_results.items():
            performance = results.get("performance_metrics", {})
            comparison["unified_metrics_table"][scenario_name] = {
                metric: performance.get(metric, 0) for metric in metrics
            }

        return comparison

    def _analyze_strategy_robustness(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze strategy robustness across scenarios."""
        # Calculate consistency metrics
        returns = []
        win_rates = []
        drawdowns = []
        sharpe_ratios = []

        for scenario_name, results in scenario_results.items():
            performance = results.get("performance_metrics", {})
            returns.append(performance.get("total_return", 0))
            win_rates.append(performance.get("win_rate", 0))
            drawdowns.append(performance.get("max_drawdown", 0))
            sharpe_ratios.append(performance.get("sharpe_ratio", 0))

        # Calculate robustness scores
        consistency_score = self._calculate_consistency_score(returns)
        adaptability_score = self._calculate_adaptability_score(scenario_results)
        risk_consistency = self._calculate_risk_consistency(drawdowns)

        # Overall robustness score (0-100 scale)
        robustness_score = (consistency_score * 0.4 + adaptability_score * 0.4 + risk_consistency * 0.2)

        robustness_analysis = {
            "overall_robustness_score": round(robustness_score, 2),
            "consistency_score": round(consistency_score, 2),
            "adaptability_score": round(adaptability_score, 2),
            "risk_consistency_score": round(risk_consistency, 2),
            "performance_variance": {
                "return_std_dev": round(statistics.stdev(returns) if len(returns) > 1 else 0, 4),
                "win_rate_std_dev": round(statistics.stdev(win_rates) if len(win_rates) > 1 else 0, 4),
                "drawdown_std_dev": round(statistics.stdev(drawdowns) if len(drawdowns) > 1 else 0, 4),
                "sharpe_std_dev": round(statistics.stdev(sharpe_ratios) if len(sharpe_ratios) > 1 else 0, 4)
            },
            "positive_scenario_count": sum(1 for r in returns if r > 0),
            "negative_scenario_count": sum(1 for r in returns if r < 0),
            "robustness_interpretation": self._interpret_robustness_score(robustness_score)
        }

        return robustness_analysis

    def _analyze_scenario_correlations(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between scenario performance."""
        scenario_names = list(scenario_results.keys())

        if len(scenario_names) < 2:
            return {"error": "Need at least 2 scenarios for correlation analysis"}

        correlations = {}
        scenario_returns = {}

        # Extract returns for each scenario
        for scenario_name, results in scenario_results.items():
            trades = results.get("trades", [])
            returns = [trade.get("pnl", 0) for trade in trades]
            scenario_returns[scenario_name] = returns

        # Calculate pairwise correlations
        for i, scenario1 in enumerate(scenario_names):
            for j, scenario2 in enumerate(scenario_names[i+1:], i+1):
                returns1 = scenario_returns[scenario1]
                returns2 = scenario_returns[scenario2]

                correlation = self._calculate_correlation(returns1, returns2)
                correlations[f"{scenario1}_vs_{scenario2}"] = {
                    "correlation": round(correlation, 3),
                    "interpretation": self._interpret_correlation(correlation)
                }

        return {
            "pairwise_correlations": correlations,
            "average_correlation": round(statistics.mean([
                corr["correlation"] for corr in correlations.values()
            ]), 3) if correlations else 0,
            "correlation_insights": self._generate_correlation_insights(correlations)
        }

    def _analyze_worst_case_scenarios(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze worst-case scenario performance."""
        scenario_performance = []

        for scenario_name, results in scenario_results.items():
            performance = results.get("performance_metrics", {})

            # Calculate composite performance score
            total_return = performance.get("total_return", 0)
            max_drawdown = performance.get("max_drawdown", 0)
            win_rate = performance.get("win_rate", 0)

            # Simple composite score (higher is better)
            composite_score = total_return - (max_drawdown * 2) + (win_rate * 0.5)

            scenario_performance.append({
                "scenario": scenario_name,
                "composite_score": round(composite_score, 2),
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate
            })

        # Sort by composite score (ascending for worst-case analysis)
        scenario_performance.sort(key=lambda x: x["composite_score"])

        worst_scenarios = scenario_performance[:3]  # Bottom 3 scenarios

        return {
            "worst_performing_scenarios": worst_scenarios,
            "worst_case_performance": worst_scenarios[0] if worst_scenarios else None,
            "scenario_performance_ranking": scenario_performance,
            "worst_case_insights": self._generate_worst_case_insights(worst_scenarios),
            "improvement_recommendations": self._recommend_worst_case_improvements(worst_scenarios)
        }

    def _analyze_strategy_adaptability(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how well the strategy adapts to different market conditions."""
        market_condition_performance = {
            "trending": [],  # bull + bear
            "sideways": [],  # sideways + choppy
            "volatile": [],  # high_vol + gap_heavy
            "stable": []     # low_vol + real_data
        }

        # Categorize scenarios by market condition type
        condition_mapping = {
            "bull_market": "trending",
            "bear_market": "trending",
            "sideways_market": "sideways",
            "choppy_market": "sideways",
            "high_volatility": "volatile",
            "gap_heavy": "volatile",
            "low_volatility": "stable",
            "real_data": "stable"
        }

        for scenario_name, results in scenario_results.items():
            condition = condition_mapping.get(scenario_name)
            if condition:
                performance = results.get("performance_metrics", {})
                total_return = performance.get("total_return", 0)
                market_condition_performance[condition].append(total_return)

        # Calculate average performance by condition
        condition_averages = {}
        for condition, returns in market_condition_performance.items():
            if returns:
                condition_averages[condition] = {
                    "average_return": round(statistics.mean(returns), 4),
                    "std_dev": round(statistics.stdev(returns) if len(returns) > 1 else 0, 4),
                    "scenario_count": len(returns),
                    "best_return": round(max(returns), 4),
                    "worst_return": round(min(returns), 4)
                }
            else:
                condition_averages[condition] = {
                    "average_return": 0,
                    "std_dev": 0,
                    "scenario_count": 0,
                    "best_return": 0,
                    "worst_return": 0
                }

        # Calculate adaptability score
        adaptability_score = self._calculate_adaptability_score(scenario_results)

        return {
            "adaptability_score": round(adaptability_score, 2),
            "market_condition_performance": condition_averages,
            "best_conditions": max(condition_averages.items(),
                                 key=lambda x: x[1]["average_return"])[0] if condition_averages else None,
            "worst_conditions": min(condition_averages.items(),
                                  key=lambda x: x[1]["average_return"])[0] if condition_averages else None,
            "adaptability_insights": self._generate_adaptability_insights(condition_averages)
        }

    def _rank_scenarios_by_performance(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Rank scenarios by strategy performance."""
        scenario_scores = []

        for scenario_name, results in scenario_results.items():
            performance = results.get("performance_metrics", {})

            # Calculate weighted performance score
            total_return = performance.get("total_return", 0)
            win_rate = performance.get("win_rate", 0)
            profit_factor = performance.get("profit_factor", 0)
            max_drawdown = performance.get("max_drawdown", 0)
            sharpe_ratio = performance.get("sharpe_ratio", 0)

            # Weighted score calculation
            score = (
                total_return * 0.3 +
                win_rate * 0.2 +
                profit_factor * 0.2 +
                (-max_drawdown) * 0.15 +  # Negative because lower is better
                sharpe_ratio * 0.15
            )

            scenario_scores.append({
                "scenario": scenario_name,
                "performance_score": round(score, 4),
                "metrics": {
                    "total_return": total_return,
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": sharpe_ratio
                }
            })

        # Sort by performance score (descending)
        scenario_scores.sort(key=lambda x: x["performance_score"], reverse=True)

        return {
            "scenario_rankings": scenario_scores,
            "best_scenario": scenario_scores[0] if scenario_scores else None,
            "worst_scenario": scenario_scores[-1] if scenario_scores else None,
            "performance_tiers": self._categorize_scenario_tiers(scenario_scores)
        }

    def _generate_scenario_optimizations(self, scenario_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on scenario analysis."""
        optimizations = []

        # Find consistently poor performing scenarios
        poor_scenarios = []
        for scenario_name, results in scenario_results.items():
            performance = results.get("performance_metrics", {})
            total_return = performance.get("total_return", 0)
            win_rate = performance.get("win_rate", 0)

            if total_return < 0 or win_rate < 0.4:
                poor_scenarios.append({
                    "scenario": scenario_name,
                    "return": total_return,
                    "win_rate": win_rate
                })

        if poor_scenarios:
            optimizations.append({
                "type": "scenario_specific_optimization",
                "priority": "high",
                "weak_scenarios": [s["scenario"] for s in poor_scenarios],
                "suggestion": "Consider adding scenario-specific filters or adjusting parameters for these market conditions",
                "affected_scenarios": len(poor_scenarios)
            })

        # Check for high variance across scenarios
        returns = [results.get("performance_metrics", {}).get("total_return", 0)
                  for results in scenario_results.values()]

        if len(returns) > 1:
            return_std = statistics.stdev(returns)
            mean_return = statistics.mean(returns)
            cv = return_std / abs(mean_return) if mean_return != 0 else float('inf')

            if cv > 1.0:  # High coefficient of variation
                optimizations.append({
                    "type": "consistency_improvement",
                    "priority": "medium",
                    "current_cv": round(cv, 2),
                    "suggestion": "Strategy shows high variability across scenarios. Consider adding robustness measures.",
                    "target_cv": "<0.5"
                })

        return optimizations

    def _generate_scenario_insights(self, scenario_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate insights and observations from scenario analysis."""
        insights = []

        if not scenario_results:
            return ["No scenario results available for insight generation"]

        total_scenarios = len(scenario_results)
        positive_scenarios = sum(
            1 for results in scenario_results.values()
            if results.get("performance_metrics", {}).get("total_return", 0) > 0
        )

        # Overall performance insights
        if positive_scenarios == total_scenarios:
            insights.append(f"Strategy shows excellent robustness with positive returns across all {total_scenarios} scenarios")
        elif positive_scenarios / total_scenarios >= 0.75:
            insights.append(f"Strategy shows strong robustness with positive returns in {positive_scenarios}/{total_scenarios} scenarios")
        elif positive_scenarios / total_scenarios >= 0.5:
            insights.append(f"Strategy shows moderate robustness with positive returns in {positive_scenarios}/{total_scenarios} scenarios")
        else:
            insights.append(f"Strategy shows limited robustness with positive returns in only {positive_scenarios}/{total_scenarios} scenarios")

        # Volatility insights
        real_data_result = scenario_results.get("real_data")

        # Check for valid real_data_result to compare with synthetic scenarios
        if real_data_result is not None and self._is_valid_result_data(real_data_result):
            real_return = real_data_result.get("performance_metrics", {}).get("total_return", 0)
            synthetic_returns = [
                results.get("performance_metrics", {}).get("total_return", 0)
                for name, results in scenario_results.items()
                if name != "real_data"
            ]

            if synthetic_returns:
                avg_synthetic = statistics.mean(synthetic_returns)
                if abs(real_return - avg_synthetic) / abs(avg_synthetic) < 0.2 if avg_synthetic != 0 else False:
                    insights.append("Real data performance closely matches synthetic scenario averages - good model validation")
                elif real_return > avg_synthetic:
                    insights.append("Real data performance exceeds synthetic scenario averages - strategy may perform better in live conditions")
                else:
                    insights.append("Real data performance below synthetic scenario averages - consider additional validation")
        else:
            return insights

    def _prepare_visualization_data(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data for visualization components."""
        viz_data = {
            "radar_chart_data": self._prepare_radar_chart_data(scenario_results),
            "heatmap_data": self._prepare_heatmap_data(scenario_results),
            "equity_curves": self._prepare_equity_curves_data(scenario_results),
            "performance_bars": self._prepare_performance_bars_data(scenario_results)
        }

        return viz_data

    # Helper methods for complex calculations
    def _calculate_consistency_score(self, returns: List[float]) -> float:
        """Calculate consistency score based on return variance."""
        if not returns or len(returns) < 2:
            return 0.0

        mean_return = statistics.mean(returns)
        std_dev = statistics.stdev(returns)

        # Coefficient of variation (lower is more consistent)
        cv = std_dev / abs(mean_return) if mean_return != 0 else float('inf')

        # Convert to 0-100 score (higher is better)
        consistency_score = max(0, 100 - (cv * 50))
        return min(100, consistency_score)

    def _calculate_adaptability_score(self, scenario_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate adaptability score based on performance across diverse conditions."""
        positive_count = sum(
            1 for results in scenario_results.values()
            if results.get("performance_metrics", {}).get("total_return", 0) > 0
        )

        total_scenarios = len(scenario_results)
        if total_scenarios == 0:
            return 0.0

        # Base score from positive scenario ratio
        base_score = (positive_count / total_scenarios) * 100

        # Bonus for performing well in difficult scenarios
        difficult_scenarios = ["bear_market", "choppy_market", "high_volatility"]
        difficult_performance = [
            scenario_results[scenario].get("performance_metrics", {}).get("total_return", 0)
            for scenario in difficult_scenarios
            if scenario in scenario_results
        ]

        if difficult_performance:
            difficult_positive = sum(1 for perf in difficult_performance if perf > 0)
            difficult_bonus = (difficult_positive / len(difficult_performance)) * 20
            base_score += difficult_bonus

        return min(100, base_score)

    def _calculate_risk_consistency(self, drawdowns: List[float]) -> float:
        """Calculate risk consistency score."""
        if not drawdowns or len(drawdowns) < 2:
            return 50.0  # Neutral score

        mean_dd = statistics.mean(drawdowns)
        std_dd = statistics.stdev(drawdowns)

        # Lower variance in drawdowns is better
        cv = std_dd / mean_dd if mean_dd != 0 else 0

        # Convert to 0-100 score
        risk_consistency = max(0, 100 - (cv * 100))
        return min(100, risk_consistency)

    def _calculate_correlation(self, returns1: List[float], returns2: List[float]) -> float:
        """Calculate correlation between two return series."""
        if len(returns1) != len(returns2) or len(returns1) < 2:
            return 0.0

        mean1 = statistics.mean(returns1)
        mean2 = statistics.mean(returns2)

        numerator = sum((returns1[i] - mean1) * (returns2[i] - mean2) for i in range(len(returns1)))

        sum_sq1 = sum((r - mean1) ** 2 for r in returns1)
        sum_sq2 = sum((r - mean2) ** 2 for r in returns2)

        denominator = (sum_sq1 * sum_sq2) ** 0.5

        return numerator / denominator if denominator != 0 else 0.0

    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation value."""
        abs_corr = abs(correlation)

        if abs_corr < 0.3:
            return "Low correlation - scenarios are independent"
        elif abs_corr < 0.7:
            return "Moderate correlation - some similarity in performance"
        else:
            return "High correlation - similar performance patterns"

    def _interpret_robustness_score(self, score: float) -> str:
        """Interpret robustness score."""
        if score >= 80:
            return "Excellent robustness - consistent performance across scenarios"
        elif score >= 60:
            return "Good robustness - generally consistent with some variation"
        elif score >= 40:
            return "Moderate robustness - significant variation across scenarios"
        else:
            return "Poor robustness - inconsistent performance across scenarios"

    def _categorize_scenario_tiers(self, scenario_scores: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Categorize scenarios into performance tiers."""
        if not scenario_scores:
            return {"tier1": [], "tier2": [], "tier3": []}

        total_scenarios = len(scenario_scores)
        tier1_count = max(1, total_scenarios // 3)
        tier2_count = max(1, total_scenarios // 3)

        return {
            "tier1": [s["scenario"] for s in scenario_scores[:tier1_count]],
            "tier2": [s["scenario"] for s in scenario_scores[tier1_count:tier1_count + tier2_count]],
            "tier3": [s["scenario"] for s in scenario_scores[tier1_count + tier2_count:]]
        }

    # Additional analysis methods for cross-scenario reporting
    def _get_common_scenarios(self, strategies: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
        """Get scenarios common to all strategies."""
        if not strategies:
            return []

        all_scenarios = set()
        first_strategy = True

        for strategy_results in strategies.values():
            scenario_set = set(strategy_results.keys())
            if first_strategy:
                all_scenarios = scenario_set
                first_strategy = False
            else:
                all_scenarios &= scenario_set

        return sorted(list(all_scenarios))

    def _compare_strategy_robustness(self, strategies: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Compare robustness across multiple strategies."""
        robustness_comparison = {}

        for strategy_name, scenario_results in strategies.items():
            robustness_analysis = self._analyze_strategy_robustness(scenario_results)
            robustness_comparison[strategy_name] = {
                "robustness_score": robustness_analysis["overall_robustness_score"],
                "consistency_score": robustness_analysis["consistency_score"],
                "adaptability_score": robustness_analysis["adaptability_score"],
                "positive_scenarios": robustness_analysis["positive_scenario_count"]
            }

        # Rank strategies by robustness
        ranked_strategies = sorted(
            robustness_comparison.items(),
            key=lambda x: x[1]["robustness_score"],
            reverse=True
        )

        return {
            "strategy_robustness": robustness_comparison,
            "robustness_rankings": ranked_strategies,
            "most_robust_strategy": ranked_strategies[0] if ranked_strategies else None,
            "least_robust_strategy": ranked_strategies[-1] if ranked_strategies else None
        }

    # Visualization data preparation methods
    def _prepare_radar_chart_data(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data for radar/spider chart visualization."""
        metrics = ["total_return", "win_rate", "profit_factor", "sharpe_ratio"]
        radar_data = {}

        for scenario_name, results in scenario_results.items():
            performance = results.get("performance_metrics", {})
            radar_data[scenario_name] = {
                metric: performance.get(metric, 0) for metric in metrics
            }

        return {
            "metrics": metrics,
            "scenario_data": radar_data,
            "chart_config": {
                "type": "radar",
                "scale_min": 0,
                "scale_max": 100
            }
        }

    def _prepare_heatmap_data(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data for scenario performance heatmap."""
        heatmap_data = []

        for scenario_name, results in scenario_results.items():
            performance = results.get("performance_metrics", {})
            total_return = performance.get("total_return", 0)

            # Color coding: green for positive, red for negative
            color_intensity = min(100, max(0, abs(total_return) * 10))
            color = "green" if total_return >= 0 else "red"

            heatmap_data.append({
                "scenario": scenario_name,
                "value": total_return,
                "color": color,
                "intensity": color_intensity
            })

        return {
            "data": heatmap_data,
            "config": {
                "color_scale": ["#ff0000", "#ffffff", "#00ff00"],
                "value_range": [-50, 50]
            }
        }

    def _prepare_equity_curves_data(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data for equity curve overlays."""
        equity_curves = {}

        for scenario_name, results in scenario_results.items():
            equity_curve = results.get("equity_curve", [])
            if equity_curve is not None and len(equity_curve) > 0:
                equity_curves[scenario_name] = equity_curve

        return {
            "curves": equity_curves,
            "chart_config": {
                "type": "line",
                "overlay": True,
                "colors": self._generate_scenario_colors()
            }
        }

    def _prepare_performance_bars_data(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data for performance bar charts."""
        bar_data = []

        for scenario_name, results in scenario_results.items():
            performance = results.get("performance_metrics", {})
            bar_data.append({
                "scenario": scenario_name,
                "total_return": performance.get("total_return", 0),
                "win_rate": performance.get("win_rate", 0),
                "max_drawdown": performance.get("max_drawdown", 0)
            })

        return {
            "data": bar_data,
            "metrics": ["total_return", "win_rate", "max_drawdown"],
            "chart_config": {
                "type": "grouped_bar",
                "colors": ["#1f77b4", "#ff7f0e", "#d62728"]
            }
        }

    def _generate_scenario_colors(self) -> Dict[str, str]:
        """Generate consistent colors for each scenario type."""
        return {
            "bull_market": "#2ca02c",
            "bear_market": "#d62728",
            "sideways_market": "#ff7f0e",
            "high_volatility": "#9467bd",
            "low_volatility": "#8c564b",
            "choppy_market": "#e377c2",
            "gap_heavy": "#7f7f7f",
            "real_data": "#1f77b4"
        }

    # Placeholder methods for complex analysis
    def _analyze_scenario_difficulty(self, strategies: Dict) -> Dict[str, Any]:
        """
        Analyze which scenarios are most difficult across all strategies.

        Args:
            strategies: Dict of {strategy_name: {scenario_name: results}}

        Returns:
            Dict containing scenario difficulty analysis
        """
        try:
            scenario_difficulty = {}
            scenario_stats = {}

            # Get common scenarios across all strategies
            common_scenarios = self._get_common_scenarios(strategies)

            for scenario in common_scenarios:
                returns = []
                sharpe_ratios = []
                max_drawdowns = []
                win_rates = []

                # Collect metrics from all strategies for this scenario
                for strategy_name, strategy_results in strategies.items():
                    if scenario in strategy_results:
                        results = strategy_results[scenario]

                        # Extract key performance metrics
                        if 'total_return_pct' in results:
                            returns.append(results['total_return_pct'])
                        if 'sharpe_ratio' in results:
                            sharpe_ratios.append(results['sharpe_ratio'])
                        if 'max_drawdown_pct' in results:
                            max_drawdowns.append(abs(results['max_drawdown_pct']))
                        if 'win_rate' in results:
                            win_rates.append(results['win_rate'] * 100)

                # Calculate difficulty metrics
                if returns:
                    avg_return = statistics.mean(returns)
                    return_volatility = statistics.stdev(returns) if len(returns) > 1 else 0
                    avg_sharpe = statistics.mean(sharpe_ratios) if sharpe_ratios else 0
                    avg_drawdown = statistics.mean(max_drawdowns) if max_drawdowns else 0
                    avg_win_rate = statistics.mean(win_rates) if win_rates else 0

                    # Calculate difficulty score (lower return, higher volatility, higher drawdown = more difficult)
                    difficulty_score = (
                        (50 - avg_return) * 0.3 +  # Return component (lower is harder)
                        return_volatility * 0.2 +   # Volatility component (higher is harder)
                        avg_drawdown * 0.3 +        # Drawdown component (higher is harder)
                        (100 - avg_win_rate) * 0.2  # Win rate component (lower is harder)
                    )

                    scenario_stats[scenario] = {
                        "average_return": round(avg_return, 2),
                        "return_volatility": round(return_volatility, 2),
                        "average_sharpe": round(avg_sharpe, 3),
                        "average_drawdown": round(avg_drawdown, 2),
                        "average_win_rate": round(avg_win_rate, 1),
                        "difficulty_score": round(difficulty_score, 2),
                        "strategies_tested": len(returns)
                    }

            # Rank scenarios by difficulty
            ranked_scenarios = sorted(
                scenario_stats.items(),
                key=lambda x: x[1]["difficulty_score"],
                reverse=True
            )

            scenario_difficulty = {
                "difficulty_ranking": [
                    {
                        "scenario": scenario,
                        "rank": idx + 1,
                        "difficulty_score": stats["difficulty_score"],
                        "difficulty_tier": self._categorize_difficulty(stats["difficulty_score"])
                    }
                    for idx, (scenario, stats) in enumerate(ranked_scenarios)
                ],
                "scenario_statistics": scenario_stats,
                "most_difficult": ranked_scenarios[0][0] if ranked_scenarios else None,
                "easiest": ranked_scenarios[-1][0] if ranked_scenarios else None,
                "analysis_insights": self._generate_difficulty_insights(ranked_scenarios)
            }

            return scenario_difficulty

        except Exception as e:
            logger.error(f"Error analyzing scenario difficulty: {e}")
            return {"error": f"Scenario difficulty analysis failed: {str(e)}"}

    def _identify_best_performers_by_scenario(self, strategies: Dict) -> Dict[str, Any]:
        """
        Identify best performing strategy for each scenario.

        Args:
            strategies: Dict of {strategy_name: {scenario_name: results}}

        Returns:
            Dict containing best performers by scenario
        """
        try:
            best_performers = {}
            scenario_leaderboards = {}

            # Get common scenarios
            common_scenarios = self._get_common_scenarios(strategies)

            for scenario in common_scenarios:
                scenario_performances = []

                # Collect performance data for all strategies in this scenario
                for strategy_name, strategy_results in strategies.items():
                    if scenario in strategy_results:
                        results = strategy_results[scenario]

                        # Calculate composite performance score
                        total_return = results.get('total_return_pct', 0)
                        sharpe_ratio = results.get('sharpe_ratio', 0)
                        max_drawdown = abs(results.get('max_drawdown_pct', 0))
                        win_rate = results.get('win_rate', 0) * 100

                        # Composite score weighing multiple factors
                        performance_score = (
                            total_return * 0.4 +           # Return weight: 40%
                            sharpe_ratio * 20 * 0.3 +      # Sharpe weight: 30% (scaled)
                            (100 - max_drawdown) * 0.2 +   # Drawdown weight: 20% (inverted)
                            win_rate * 0.1                 # Win rate weight: 10%
                        )

                        scenario_performances.append({
                            "strategy": strategy_name,
                            "total_return": total_return,
                            "sharpe_ratio": sharpe_ratio,
                            "max_drawdown": max_drawdown,
                            "win_rate": win_rate,
                            "performance_score": round(performance_score, 2)
                        })

                # Sort by performance score
                scenario_performances.sort(key=lambda x: x["performance_score"], reverse=True)

                # Store results
                if scenario_performances:
                    best_performers[scenario] = {
                        "winner": scenario_performances[0],
                        "runner_up": scenario_performances[1] if len(scenario_performances) > 1 else None,
                        "performance_gap": (
                            scenario_performances[0]["performance_score"] -
                            scenario_performances[1]["performance_score"]
                            if len(scenario_performances) > 1 else 0
                        )
                    }

                    scenario_leaderboards[scenario] = scenario_performances

            # Find overall patterns
            strategy_wins = {}
            for scenario, data in best_performers.items():
                winner = data["winner"]["strategy"]
                strategy_wins[winner] = strategy_wins.get(winner, 0) + 1

            # Most versatile strategy (wins across scenarios)
            most_versatile = max(strategy_wins.items(), key=lambda x: x[1]) if strategy_wins else None

            return {
                "best_performers_by_scenario": best_performers,
                "scenario_leaderboards": scenario_leaderboards,
                "strategy_win_counts": strategy_wins,
                "most_versatile_strategy": {
                    "strategy": most_versatile[0] if most_versatile else None,
                    "scenario_wins": most_versatile[1] if most_versatile else 0,
                    "win_percentage": round((most_versatile[1] / len(common_scenarios)) * 100, 1) if most_versatile else 0
                },
                "scenario_competitiveness": self._analyze_scenario_competitiveness(scenario_leaderboards)
            }

        except Exception as e:
            logger.error(f"Error identifying best performers by scenario: {e}")
            return {"error": f"Best performers analysis failed: {str(e)}"}

    def _rank_strategies_by_consistency(self, strategies: Dict) -> Dict[str, Any]:
        """
        Rank strategies by consistency across scenarios.

        Args:
            strategies: Dict of {strategy_name: {scenario_name: results}}

        Returns:
            Dict containing consistency rankings
        """
        try:
            strategy_consistency = {}

            for strategy_name, scenario_results in strategies.items():
                if not scenario_results:
                    continue

                # Collect performance metrics across scenarios
                returns = []
                sharpe_ratios = []
                drawdowns = []
                win_rates = []

                for scenario, results in scenario_results.items():
                    if 'total_return_pct' in results:
                        returns.append(results['total_return_pct'])
                    if 'sharpe_ratio' in results:
                        sharpe_ratios.append(results['sharpe_ratio'])
                    if 'max_drawdown_pct' in results:
                        drawdowns.append(abs(results['max_drawdown_pct']))
                    if 'win_rate' in results:
                        win_rates.append(results['win_rate'] * 100)

                if not returns:
                    continue

                # Calculate consistency metrics
                return_consistency = self._calculate_consistency_score(returns)
                sharpe_consistency = self._calculate_consistency_score(sharpe_ratios) if sharpe_ratios else 0
                drawdown_consistency = 100 - self._calculate_consistency_score(drawdowns) if drawdowns else 0  # Lower variance in drawdowns is better
                win_rate_consistency = self._calculate_consistency_score(win_rates) if win_rates else 0

                # Calculate overall consistency score
                overall_consistency = (
                    return_consistency * 0.35 +
                    sharpe_consistency * 0.25 +
                    drawdown_consistency * 0.25 +
                    win_rate_consistency * 0.15
                )

                # Calculate additional metrics
                avg_return = statistics.mean(returns)
                min_return = min(returns)
                max_return = max(returns)
                return_range = max_return - min_return

                negative_scenarios = sum(1 for r in returns if r < 0)
                scenario_count = len(returns)

                strategy_consistency[strategy_name] = {
                    "overall_consistency_score": round(overall_consistency, 2),
                    "return_consistency": round(return_consistency, 2),
                    "sharpe_consistency": round(sharpe_consistency, 2),
                    "drawdown_consistency": round(drawdown_consistency, 2),
                    "win_rate_consistency": round(win_rate_consistency, 2),
                    "average_return": round(avg_return, 2),
                    "return_range": round(return_range, 2),
                    "negative_scenarios": negative_scenarios,
                    "positive_scenario_rate": round(((scenario_count - negative_scenarios) / scenario_count) * 100, 1),
                    "scenarios_tested": scenario_count,
                    "consistency_tier": self._categorize_consistency(overall_consistency)
                }

            # Rank strategies by consistency
            ranked_strategies = sorted(
                strategy_consistency.items(),
                key=lambda x: x[1]["overall_consistency_score"],
                reverse=True
            )

            return {
                "consistency_rankings": [
                    {
                        "rank": idx + 1,
                        "strategy": strategy,
                        "consistency_score": metrics["overall_consistency_score"],
                        "consistency_tier": metrics["consistency_tier"],
                        "positive_scenario_rate": metrics["positive_scenario_rate"]
                    }
                    for idx, (strategy, metrics) in enumerate(ranked_strategies)
                ],
                "strategy_consistency_details": strategy_consistency,
                "most_consistent_strategy": ranked_strategies[0][0] if ranked_strategies else None,
                "least_consistent_strategy": ranked_strategies[-1][0] if ranked_strategies else None,
                "consistency_insights": self._generate_consistency_insights(ranked_strategies)
            }

        except Exception as e:
            logger.error(f"Error ranking strategies by consistency: {e}")
            return {"error": f"Consistency ranking failed: {str(e)}"}

    def _analyze_cross_scenario_diversification(self, strategies: Dict) -> Dict[str, Any]:
        """
        Analyze diversification benefits across scenarios.

        Args:
            strategies: Dict of {strategy_name: {scenario_name: results}}

        Returns:
            Dict containing diversification analysis
        """
        try:
            # Calculate correlation matrix between strategies across scenarios
            strategy_names = list(strategies.keys())
            common_scenarios = self._get_common_scenarios(strategies)

            correlation_matrix = {}
            strategy_returns = {}

            # Collect returns for each strategy across scenarios
            for strategy in strategy_names:
                returns = []
                for scenario in common_scenarios:
                    if scenario in strategies[strategy]:
                        returns.append(strategies[strategy][scenario].get('total_return_pct', 0))
                    else:
                        returns.append(0)  # Use 0 for missing scenarios
                strategy_returns[strategy] = returns

            # Calculate pairwise correlations
            for i, strategy1 in enumerate(strategy_names):
                correlation_matrix[strategy1] = {}
                for j, strategy2 in enumerate(strategy_names):
                    if i == j:
                        correlation_matrix[strategy1][strategy2] = 1.0
                    else:
                        corr = self._calculate_correlation(
                            strategy_returns[strategy1],
                            strategy_returns[strategy2]
                        )
                        correlation_matrix[strategy1][strategy2] = round(corr, 3)

            # Identify diversification opportunities
            low_correlation_pairs = []
            high_correlation_pairs = []

            for strategy1 in strategy_names:
                for strategy2 in strategy_names:
                    if strategy1 < strategy2:  # Avoid duplicates
                        corr = correlation_matrix[strategy1][strategy2]
                        pair_info = {
                            "strategy1": strategy1,
                            "strategy2": strategy2,
                            "correlation": corr,
                            "diversification_benefit": round((1 - abs(corr)) * 100, 1)
                        }

                        if abs(corr) < 0.3:
                            low_correlation_pairs.append(pair_info)
                        elif abs(corr) > 0.7:
                            high_correlation_pairs.append(pair_info)

            # Sort by diversification benefit
            low_correlation_pairs.sort(key=lambda x: x["diversification_benefit"], reverse=True)
            high_correlation_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

            # Calculate portfolio effects for top combinations
            portfolio_analysis = self._analyze_portfolio_combinations(strategies, low_correlation_pairs[:5])

            return {
                "correlation_matrix": correlation_matrix,
                "diversification_opportunities": {
                    "low_correlation_pairs": low_correlation_pairs[:10],  # Top 10
                    "high_correlation_pairs": high_correlation_pairs[:5],   # Top 5 redundant pairs
                },
                "portfolio_analysis": portfolio_analysis,
                "diversification_insights": self._generate_diversification_insights(
                    correlation_matrix, low_correlation_pairs, high_correlation_pairs
                ),
                "average_correlation": round(
                    statistics.mean([
                        abs(correlation_matrix[s1][s2])
                        for s1 in strategy_names
                        for s2 in strategy_names
                        if s1 != s2
                    ]) if len(strategy_names) > 1 else 0, 3
                )
            }

        except Exception as e:
            logger.error(f"Error analyzing cross-scenario diversification: {e}")
            return {"error": f"Diversification analysis failed: {str(e)}"}

    def _recommend_scenario_aware_portfolio(self, strategies: Dict) -> Dict[str, Any]:
        """
        Recommend portfolio allocation considering scenario performance.

        Args:
            strategies: Dict of {strategy_name: {scenario_name: results}}

        Returns:
            Dict containing portfolio recommendations
        """
        try:
            strategy_names = list(strategies.keys())
            common_scenarios = self._get_common_scenarios(strategies)

            if len(strategy_names) < 2:
                return {
                    "recommendations": [],
                    "message": "Portfolio analysis requires at least 2 strategies"
                }

            # Calculate strategy performance profiles
            strategy_profiles = {}
            for strategy in strategy_names:
                returns = []
                sharpe_ratios = []
                max_drawdowns = []

                for scenario in common_scenarios:
                    if scenario in strategies[strategy]:
                        results = strategies[strategy][scenario]
                        returns.append(results.get('total_return_pct', 0))
                        sharpe_ratios.append(results.get('sharpe_ratio', 0))
                        max_drawdowns.append(abs(results.get('max_drawdown_pct', 0)))

                if returns:
                    strategy_profiles[strategy] = {
                        "avg_return": statistics.mean(returns),
                        "return_volatility": statistics.stdev(returns) if len(returns) > 1 else 0,
                        "avg_sharpe": statistics.mean(sharpe_ratios) if sharpe_ratios else 0,
                        "avg_drawdown": statistics.mean(max_drawdowns) if max_drawdowns else 0,
                        "consistency_score": self._calculate_consistency_score(returns),
                        "scenario_wins": sum(1 for r in returns if r > 0)
                    }

            # Generate portfolio recommendations
            recommendations = []

            # 1. Conservative Portfolio (focus on consistency and low drawdown)
            conservative_weights = self._calculate_conservative_allocation(strategy_profiles)
            recommendations.append({
                "name": "Conservative Portfolio",
                "description": "Emphasizes consistency and drawdown protection",
                "allocation": conservative_weights,
                "target_investor": "Risk-averse, steady returns preferred",
                "expected_characteristics": "Lower volatility, consistent performance"
            })

            # 2. Aggressive Portfolio (focus on returns)
            aggressive_weights = self._calculate_aggressive_allocation(strategy_profiles)
            recommendations.append({
                "name": "Aggressive Portfolio",
                "description": "Maximizes return potential",
                "allocation": aggressive_weights,
                "target_investor": "High risk tolerance, growth focused",
                "expected_characteristics": "Higher returns, higher volatility"
            })

            # 3. Balanced Portfolio (optimize risk-adjusted returns)
            balanced_weights = self._calculate_balanced_allocation(strategy_profiles)
            recommendations.append({
                "name": "Balanced Portfolio",
                "description": "Optimizes risk-adjusted returns (Sharpe ratio)",
                "allocation": balanced_weights,
                "target_investor": "Moderate risk tolerance, balanced approach",
                "expected_characteristics": "Good risk-adjusted returns"
            })

            # 4. Scenario-Adaptive Portfolio (best performer in each scenario gets higher weight)
            adaptive_weights = self._calculate_adaptive_allocation(strategies, common_scenarios)
            recommendations.append({
                "name": "Scenario-Adaptive Portfolio",
                "description": "Weights strategies based on scenario performance leadership",
                "allocation": adaptive_weights,
                "target_investor": "Sophisticated, scenario-aware investing",
                "expected_characteristics": "Adapts well to different market conditions"
            })

            return {
                "portfolio_recommendations": recommendations,
                "strategy_profiles": strategy_profiles,
                "allocation_insights": self._generate_allocation_insights(strategy_profiles, recommendations),
                "risk_considerations": self._generate_risk_considerations(strategies),
                "implementation_notes": [
                    "Allocations are suggestions based on historical scenario performance",
                    "Consider rebalancing frequency and transaction costs",
                    "Monitor performance and adjust allocations as needed",
                    "Diversification does not guarantee profits or prevent losses"
                ]
            }

        except Exception as e:
            logger.error(f"Error generating portfolio recommendations: {e}")
            return {"error": f"Portfolio recommendation failed: {str(e)}"}

    # Helper methods for the new implementations
    def _categorize_difficulty(self, difficulty_score: float) -> str:
        """Categorize scenario difficulty."""
        if difficulty_score >= 70:
            return "Very Difficult"
        elif difficulty_score >= 50:
            return "Difficult"
        elif difficulty_score >= 30:
            return "Moderate"
        else:
            return "Easy"

    def _generate_difficulty_insights(self, ranked_scenarios: List[Tuple]) -> List[str]:
        """Generate insights from difficulty analysis."""
        insights = []
        if ranked_scenarios:
            hardest = ranked_scenarios[0]
            easiest = ranked_scenarios[-1]

            insights.append(f"Most challenging scenario: {hardest[0]} (difficulty score: {hardest[1]['difficulty_score']})")
            insights.append(f"Easiest scenario: {easiest[0]} (difficulty score: {easiest[1]['difficulty_score']})")

            if hardest[1]["difficulty_score"] - easiest[1]["difficulty_score"] > 30:
                insights.append("High variation in scenario difficulty - strategies need strong adaptability")

        return insights

    def _categorize_consistency(self, consistency_score: float) -> str:
        """Categorize strategy consistency."""
        if consistency_score >= 80:
            return "Highly Consistent"
        elif consistency_score >= 60:
            return "Consistent"
        elif consistency_score >= 40:
            return "Moderately Consistent"
        else:
            return "Inconsistent"

    def _generate_consistency_insights(self, ranked_strategies: List[Tuple]) -> List[str]:
        """Generate insights from consistency analysis."""
        insights = []
        if ranked_strategies:
            most_consistent = ranked_strategies[0]
            least_consistent = ranked_strategies[-1]

            insights.append(f"Most consistent strategy: {most_consistent[0]} (score: {most_consistent[1]['overall_consistency_score']})")
            insights.append(f"Least consistent strategy: {least_consistent[0]} (score: {least_consistent[1]['overall_consistency_score']})")

            avg_consistency = statistics.mean([s[1]['overall_consistency_score'] for s in ranked_strategies])
            insights.append(f"Average consistency score: {round(avg_consistency, 1)}")

        return insights

    def _save_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """Save scenario report to file."""
        # Check if output_path is a directory and construct filename if needed
        if output_path.is_dir():
            # Generate filename with timestamp for the report
            strategy_name = report.get("strategy_name", "strategy")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_scenario_report_{strategy_name}_{timestamp}.json"
            output_path = output_path / filename

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Multi-scenario report saved to: {output_path}")

    def export_scenario_results(
        self,
        report_data: Dict[str, Any],
        output_format: str = "json",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export scenario results in specified format.

        Args:
            report_data: Complete scenario report data
            output_format: Export format (json, csv, xlsx)
            output_path: Optional custom output path

        Returns:
            Path to the exported file
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = report_data.get("strategy_name", "strategy")
            output_path = Path(f"scenario_report_{strategy_name}_{timestamp}.{output_format}")

        if output_format.lower() == "json":
            self._save_report(report_data, output_path)
        elif output_format.lower() == "csv":
            logger.warning("CSV export for scenario reports not yet implemented")
        elif output_format.lower() == "xlsx":
            logger.warning("Excel export for scenario reports not yet implemented")
        else:
            raise ValueError(f"Unsupported export format: {output_format}")

        logger.info(f"Scenario results exported to: {output_path}")
        return output_path

    def _generate_correlation_insights(self, correlations: Dict) -> List[str]:
        """Generate insights from correlation analysis."""
        return ["Correlation insights not yet implemented"]

    def _generate_worst_case_insights(self, worst_scenarios: List[Dict]) -> List[str]:
        """Generate insights from worst-case analysis."""
        return ["Worst-case insights not yet implemented"]

    def _recommend_worst_case_improvements(self, worst_scenarios: List[Dict]) -> List[Dict[str, Any]]:
        """Recommend improvements for worst-case scenarios."""
        return [{"placeholder": "Worst-case improvements not yet implemented"}]

    def _generate_adaptability_insights(self, condition_averages: Dict) -> List[str]:
        """Generate insights from adaptability analysis."""
        return ["Adaptability insights not yet implemented"]

    # Additional helper methods for new implementations
    def _analyze_scenario_competitiveness(self, scenario_leaderboards: Dict) -> Dict[str, Any]:
        """Analyze how competitive each scenario is (performance gaps between strategies)."""
        competitiveness = {}

        for scenario, performances in scenario_leaderboards.items():
            if len(performances) < 2:
                competitiveness[scenario] = {
                    "competitive_level": "Not Competitive",
                    "performance_spread": 0,
                    "description": "Only one strategy tested"
                }
                continue

            scores = [p["performance_score"] for p in performances]
            max_score = max(scores)
            min_score = min(scores)
            spread = max_score - min_score

            # Determine competitiveness level
            if spread < 10:
                level = "Highly Competitive"
                description = "Very close performance between strategies"
            elif spread < 25:
                level = "Competitive"
                description = "Moderate performance differences"
            elif spread < 50:
                level = "Somewhat Competitive"
                description = "Clear performance leaders emerging"
            else:
                level = "Not Competitive"
                description = "Large performance gaps between strategies"

            competitiveness[scenario] = {
                "competitive_level": level,
                "performance_spread": round(spread, 2),
                "description": description,
                "winner_advantage": round(performances[0]["performance_score"] - performances[1]["performance_score"], 2) if len(performances) > 1 else 0
            }

        return competitiveness

    def _analyze_portfolio_combinations(self, strategies: Dict, low_correlation_pairs: List[Dict]) -> Dict[str, Any]:
        """Analyze portfolio effects for strategy combinations."""
        portfolio_effects = []

        for pair in low_correlation_pairs:
            strategy1 = pair["strategy1"]
            strategy2 = pair["strategy2"]

            # Get common scenarios for both strategies
            common_scenarios = []
            for scenario in self.scenario_types:
                if (scenario in strategies.get(strategy1, {}) and
                    scenario in strategies.get(strategy2, {})):
                    common_scenarios.append(scenario)

            if not common_scenarios:
                continue

            # Calculate individual and combined performance
            s1_returns = []
            s2_returns = []
            combined_returns = []

            for scenario in common_scenarios:
                s1_return = strategies[strategy1][scenario].get('total_return_pct', 0)
                s2_return = strategies[strategy2][scenario].get('total_return_pct', 0)
                # Simple 50/50 allocation
                combined_return = (s1_return + s2_return) / 2

                s1_returns.append(s1_return)
                s2_returns.append(s2_return)
                combined_returns.append(combined_return)

            if s1_returns and s2_returns and combined_returns:
                portfolio_effects.append({
                    "strategy1": strategy1,
                    "strategy2": strategy2,
                    "correlation": pair["correlation"],
                    "individual_avg_returns": {
                        strategy1: round(statistics.mean(s1_returns), 2),
                        strategy2: round(statistics.mean(s2_returns), 2)
                    },
                    "combined_avg_return": round(statistics.mean(combined_returns), 2),
                    "volatility_reduction": round(
                        (statistics.stdev(s1_returns) + statistics.stdev(s2_returns)) / 2 -
                        statistics.stdev(combined_returns), 2
                    ) if len(combined_returns) > 1 else 0,
                    "diversification_benefit": pair["diversification_benefit"]
                })

        return {
            "portfolio_combinations": portfolio_effects[:5],  # Top 5 combinations
            "average_volatility_reduction": round(
                statistics.mean([p["volatility_reduction"] for p in portfolio_effects]), 2
            ) if portfolio_effects else 0
        }

    def _generate_diversification_insights(self, correlation_matrix: Dict, low_corr_pairs: List, high_corr_pairs: List) -> List[str]:
        """Generate insights from diversification analysis."""
        insights = []

        if low_corr_pairs:
            best_pair = low_corr_pairs[0]
            insights.append(f"Best diversification opportunity: {best_pair['strategy1']} + {best_pair['strategy2']} "
                          f"(correlation: {best_pair['correlation']}, benefit: {best_pair['diversification_benefit']}%)")

        if high_corr_pairs:
            redundant_pair = high_corr_pairs[0]
            insights.append(f"Most redundant pair: {redundant_pair['strategy1']} + {redundant_pair['strategy2']} "
                          f"(correlation: {redundant_pair['correlation']})")

        # Overall diversification potential
        avg_abs_corr = statistics.mean([
            abs(correlation_matrix[s1][s2])
            for s1 in correlation_matrix.keys()
            for s2 in correlation_matrix[s1].keys()
            if s1 != s2
        ]) if len(correlation_matrix) > 1 else 0

        if avg_abs_corr < 0.3:
            insights.append("Strong diversification potential across strategies")
        elif avg_abs_corr > 0.7:
            insights.append("Limited diversification potential - strategies are highly correlated")
        else:
            insights.append("Moderate diversification potential available")

        return insights

    def _calculate_conservative_allocation(self, strategy_profiles: Dict) -> Dict[str, float]:
        """Calculate allocation for conservative portfolio."""
        if not strategy_profiles:
            return {}

        # Weight by inverse drawdown and consistency
        weights = {}
        total_score = 0

        for strategy, profile in strategy_profiles.items():
            # Conservative score: high consistency, low drawdown
            drawdown_score = 100 - profile["avg_drawdown"]  # Lower drawdown is better
            consistency_score = profile["consistency_score"]
            conservative_score = (drawdown_score * 0.6 + consistency_score * 0.4)

            weights[strategy] = max(conservative_score, 1)  # Ensure minimum weight
            total_score += weights[strategy]

        # Normalize to percentages
        return {strategy: round((weight / total_score) * 100, 1) for strategy, weight in weights.items()}

    def _calculate_aggressive_allocation(self, strategy_profiles: Dict) -> Dict[str, float]:
        """Calculate allocation for aggressive portfolio."""
        if not strategy_profiles:
            return {}

        # Weight by returns and scenario wins
        weights = {}
        total_score = 0

        for strategy, profile in strategy_profiles.items():
            # Aggressive score: high returns, high win rate
            return_score = max(profile["avg_return"], 0)  # Only positive returns count
            win_score = profile["scenario_wins"] * 10  # Scale up win count
            aggressive_score = return_score * 0.7 + win_score * 0.3

            weights[strategy] = max(aggressive_score, 1)  # Ensure minimum weight
            total_score += weights[strategy]

        # Normalize to percentages
        return {strategy: round((weight / total_score) * 100, 1) for strategy, weight in weights.items()}

    def _calculate_balanced_allocation(self, strategy_profiles: Dict) -> Dict[str, float]:
        """Calculate allocation for balanced portfolio."""
        if not strategy_profiles:
            return {}

        # Weight by Sharpe ratio and overall balance
        weights = {}
        total_score = 0

        for strategy, profile in strategy_profiles.items():
            # Balanced score: good Sharpe ratio with moderate consistency
            sharpe_score = max(profile["avg_sharpe"], 0) * 20  # Scale Sharpe ratio
            consistency_bonus = profile["consistency_score"] * 0.5
            return_component = max(profile["avg_return"], 0) * 0.3

            balanced_score = sharpe_score + consistency_bonus + return_component

            weights[strategy] = max(balanced_score, 1)  # Ensure minimum weight
            total_score += weights[strategy]

        # Normalize to percentages
        return {strategy: round((weight / total_score) * 100, 1) for strategy, weight in weights.items()}

    def _calculate_adaptive_allocation(self, strategies: Dict, common_scenarios: List[str]) -> Dict[str, float]:
        """Calculate allocation for scenario-adaptive portfolio."""
        if not strategies or not common_scenarios:
            return {}

        # Weight by scenario leadership
        weights = {}
        total_score = 0

        for strategy in strategies.keys():
            scenario_leadership_score = 0

            for scenario in common_scenarios:
                if scenario in strategies[strategy]:
                    # Get performance in this scenario
                    performance = strategies[strategy][scenario].get('total_return_pct', 0)

                    # Compare to other strategies in same scenario
                    other_performances = []
                    for other_strategy, other_results in strategies.items():
                        if other_strategy != strategy and scenario in other_results:
                            other_performances.append(other_results[scenario].get('total_return_pct', 0))

                    if other_performances:
                        avg_others = statistics.mean(other_performances)
                        if performance > avg_others:
                            scenario_leadership_score += (performance - avg_others)

            weights[strategy] = max(scenario_leadership_score, 1)  # Ensure minimum weight
            total_score += weights[strategy]

        # Normalize to percentages
        if total_score > 0:
            return {strategy: round((weight / total_score) * 100, 1) for strategy, weight in weights.items()}
        else:
            # Equal weights if no clear leaders
            equal_weight = round(100 / len(strategies), 1)
            return {strategy: equal_weight for strategy in strategies.keys()}

    def _generate_allocation_insights(self, strategy_profiles: Dict, recommendations: List[Dict]) -> List[str]:
        """Generate insights about portfolio allocations."""
        insights = []

        # Find most recommended strategy across portfolios
        strategy_total_weights = {}
        for rec in recommendations:
            for strategy, weight in rec["allocation"].items():
                strategy_total_weights[strategy] = strategy_total_weights.get(strategy, 0) + weight

        if strategy_total_weights:
            most_recommended = max(strategy_total_weights.items(), key=lambda x: x[1])
            insights.append(f"Most recommended strategy across portfolios: {most_recommended[0]} "
                          f"(total weight: {round(most_recommended[1], 1)}%)")

        # Identify strategies with consistent high allocations
        high_allocation_strategies = []
        for rec in recommendations:
            for strategy, weight in rec["allocation"].items():
                if weight >= 25:  # 25% or higher allocation
                    high_allocation_strategies.append(strategy)

        if high_allocation_strategies:
            consistent_high = [s for s in set(high_allocation_strategies)
                             if high_allocation_strategies.count(s) >= 2]
            if consistent_high:
                insights.append(f"Consistently high-weighted strategies: {', '.join(consistent_high)}")

        return insights

    def _generate_risk_considerations(self, strategies: Dict) -> List[str]:
        """Generate risk considerations for portfolio allocation."""
        considerations = [
            "Past performance does not guarantee future results",
            "Scenario analysis is based on synthetic and limited historical data",
            "Market conditions may differ significantly from tested scenarios",
            "Consider transaction costs and rebalancing frequency in implementation"
        ]

        # Add specific considerations based on strategy count
        strategy_count = len(strategies)
        if strategy_count < 3:
            considerations.append("Limited strategy diversification with fewer than 3 strategies")
        elif strategy_count > 10:
            considerations.append("High number of strategies may increase complexity and costs")

        return considerations

    def generate_visual_scenario_comparison(
        self,
        strategy_name: str,
        scenario_results: Dict[str, Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive visual scenario performance comparison.

        Args:
            strategy_name: Name of the strategy
            scenario_results: Dictionary of results keyed by scenario name
            output_path: Optional path to save visualizations

        Returns:
            Dictionary containing all visualization data and configurations
        """
        logger.info(f"Generating visual scenario comparison for strategy: {strategy_name}")

        try:
            visualizations = {
                "strategy_name": strategy_name,
                "visualization_type": "multi_scenario_visual_comparison",
                "generated_at": datetime.now().isoformat(),

                # Core visualizations
                "radar_chart": self._generate_scenario_radar_chart(scenario_results),
                "performance_heatmap": self._generate_scenario_heatmap(scenario_results),
                "equity_curves_overlay": self._generate_equity_curves_overlay(scenario_results),
                "trade_distribution_charts": self._generate_trade_distribution_charts(scenario_results),

                # Interactive features
                "scenario_highlighting_config": self._generate_scenario_highlighting_config(scenario_results),
                "market_condition_timeline": self._generate_market_condition_timeline(scenario_results),

                # Chart configurations
                "chart_themes": self._generate_chart_themes(),
                "responsive_configs": self._generate_responsive_chart_configs(),
                "accessibility_features": self._generate_accessibility_features(),

                # Export configurations
                "export_formats": ["png", "svg", "pdf", "html"],
                "interactive_elements": self._generate_interactive_elements_config(scenario_results)
            }

            if output_path:
                self._save_visual_report(visualizations, output_path)

            return visualizations

        except Exception as e:
            logger.error(f"Error generating visual scenario comparison for {strategy_name}: {e}")
            raise

    def _generate_scenario_radar_chart(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate radar/spider chart configuration for scenario performance."""
        try:
            # Define performance dimensions for radar chart
            dimensions = [
                "total_return_pct",
                "win_rate",
                "sharpe_ratio",
                "profit_factor",
                "max_drawdown_pct"  # Will be inverted for display
            ]

            dimension_labels = {
                "total_return_pct": "Total Return %",
                "win_rate": "Win Rate %",
                "sharpe_ratio": "Sharpe Ratio",
                "profit_factor": "Profit Factor",
                "max_drawdown_pct": "Drawdown Resistance %"
            }

            # Prepare data for each scenario
            radar_data = []
            scenario_colors = self._generate_scenario_colors()

            for scenario, results in scenario_results.items():
                if not results:
                    continue

                # Extract and normalize values for radar chart
                values = []
                raw_values = {}

                for dimension in dimensions:
                    if dimension == "win_rate":
                        # Convert to percentage
                        value = results.get(dimension, 0) * 100
                        raw_values[dimension] = value
                        values.append(min(value, 100))  # Cap at 100%

                    elif dimension == "max_drawdown_pct":
                        # Invert drawdown - higher is better for display
                        drawdown = abs(results.get(dimension, 0))
                        resistance = 100 - min(drawdown, 100)
                        raw_values[dimension] = resistance
                        values.append(resistance)

                    elif dimension == "sharpe_ratio":
                        # Scale Sharpe ratio for visibility (multiply by 20, cap at 100)
                        sharpe = results.get(dimension, 0)
                        scaled_sharpe = min(max(sharpe * 20, 0), 100)
                        raw_values[dimension] = scaled_sharpe
                        values.append(scaled_sharpe)

                    elif dimension == "profit_factor":
                        # Scale profit factor (multiply by 25, cap at 100)
                        pf = results.get(dimension, 1)
                        scaled_pf = min(max((pf - 1) * 25, 0), 100)
                        raw_values[dimension] = scaled_pf
                        values.append(scaled_pf)

                    else:
                        # Total return percentage (cap at 100% for chart)
                        value = results.get(dimension, 0)
                        raw_values[dimension] = value
                        values.append(min(max(value, -100), 100))

                radar_data.append({
                    "scenario": scenario,
                    "scenario_label": scenario.replace("_", " ").title(),
                    "values": values,
                    "raw_values": raw_values,
                    "color": scenario_colors.get(scenario, "#888888"),
                    "fill_opacity": 0.1,
                    "stroke_width": 2
                })

            return {
                "chart_type": "radar",
                "title": "Strategy Performance Across Market Scenarios",
                "dimensions": dimensions,
                "dimension_labels": dimension_labels,
                "data": radar_data,
                "config": {
                    "max_value": 100,
                    "grid_levels": 5,
                    "show_grid": True,
                    "show_legend": True,
                    "animate": True,
                    "responsive": True
                },
                "insights": self._generate_radar_chart_insights(radar_data)
            }

        except Exception as e:
            logger.error(f"Error generating radar chart: {e}")
            return {"error": f"Radar chart generation failed: {str(e)}"}

    def _generate_scenario_heatmap(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate scenario performance heatmap with color coding."""
        try:
            # Define metrics for heatmap rows
            metrics = [
                "total_return_pct",
                "win_rate",
                "sharpe_ratio",
                "profit_factor",
                "max_drawdown_pct",
                "total_trades",
                "avg_trade_duration_hours"
            ]

            metric_labels = {
                "total_return_pct": "Total Return %",
                "win_rate": "Win Rate %",
                "sharpe_ratio": "Sharpe Ratio",
                "profit_factor": "Profit Factor",
                "max_drawdown_pct": "Max Drawdown %",
                "total_trades": "Total Trades",
                "avg_trade_duration_hours": "Avg Trade Duration (hrs)"
            }

            # Prepare heatmap data matrix
            heatmap_data = []
            scenarios = list(scenario_results.keys())

            for metric in metrics:
                row_data = {
                    "metric": metric,
                    "metric_label": metric_labels.get(metric, metric),
                    "values": [],
                    "raw_values": [],
                    "color_scale": "RdYlGn" if metric != "max_drawdown_pct" else "RdYlGn_r"  # Reverse for drawdown
                }

                metric_values = []
                for scenario in scenarios:
                    results = scenario_results.get(scenario, {})

                    if metric == "win_rate":
                        value = results.get(metric, 0) * 100
                    elif metric == "max_drawdown_pct":
                        value = abs(results.get(metric, 0))  # Show as positive
                    else:
                        value = results.get(metric, 0)

                    metric_values.append(value)
                    row_data["raw_values"].append(value)

                # Normalize values for color mapping (0-100 scale)
                if metric_values:
                    if metric == "max_drawdown_pct":
                        # For drawdown, lower is better - invert the scale
                        max_val = max(metric_values) if metric_values else 1
                        normalized = [100 - (val / max_val * 100) if max_val > 0 else 50 for val in metric_values]
                    else:
                        # For other metrics, higher is better
                        max_val = max(metric_values) if metric_values else 1
                        min_val = min(metric_values) if metric_values else 0
                        range_val = max_val - min_val if max_val != min_val else 1
                        normalized = [((val - min_val) / range_val * 100) if range_val > 0 else 50 for val in metric_values]
                else:
                    normalized = [50] * len(scenarios)  # Neutral color

                row_data["values"] = normalized
                heatmap_data.append(row_data)

            return {
                "chart_type": "heatmap",
                "title": "Scenario Performance Heatmap",
                "subtitle": "Green = Good Performance, Red = Poor Performance",
                "scenarios": scenarios,
                "scenario_labels": [s.replace("_", " ").title() for s in scenarios],
                "data": heatmap_data,
                "config": {
                    "color_scale": "RdYlGn",
                    "show_values": True,
                    "cell_border": True,
                    "responsive": True,
                    "tooltip_enabled": True
                },
                "summary": self._generate_heatmap_summary(heatmap_data, scenarios)
            }

        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            return {"error": f"Heatmap generation failed: {str(e)}"}

    def _generate_equity_curves_overlay(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overlay charts showing equity curves for all scenarios."""
        try:
            equity_curves = []
            scenario_colors = self._generate_scenario_colors()

            for scenario, results in scenario_results.items():
                if "equity_curve" in results and results["equity_curve"] is not None and len(results.get("equity_curve", [])) > 0:
                    equity_data = results["equity_curve"]

                    # Normalize equity curve to percentage gains from starting point
                    if equity_data is not None and len(equity_data) > 0:
                        starting_value = equity_data[0] if isinstance(equity_data[0], (int, float)) else equity_data[0].get("equity", 10000)

                        curve_points = []
                        for i, point in enumerate(equity_data):
                            if isinstance(point, dict):
                                equity = point.get("equity", starting_value)
                                timestamp = point.get("timestamp", i)
                            else:
                                equity = point
                                timestamp = i

                            percentage_gain = ((equity - starting_value) / starting_value) * 100
                            curve_points.append({
                                "timestamp": timestamp,
                                "equity": equity,
                                "percentage_gain": round(percentage_gain, 2),
                                "scenario": scenario
                            })

                        equity_curves.append({
                            "scenario": scenario,
                            "scenario_label": scenario.replace("_", " ").title(),
                            "data": curve_points,
                            "color": scenario_colors.get(scenario, "#888888"),
                            "line_width": 2,
                            "line_style": "solid"
                        })

            # Calculate summary statistics for overlay
            overlay_stats = self._calculate_overlay_statistics(equity_curves)

            return {
                "chart_type": "line_overlay",
                "title": "Equity Curves Comparison Across Scenarios",
                "x_axis": {
                    "label": "Time Period",
                    "type": "sequential"
                },
                "y_axis": {
                    "label": "Percentage Gain/Loss (%)",
                    "type": "percentage"
                },
                "curves": equity_curves,
                "statistics": overlay_stats,
                "config": {
                    "show_grid": True,
                    "show_legend": True,
                    "animate": False,  # Too many lines for smooth animation
                    "zoom_enabled": True,
                    "crosshair": True,
                    "responsive": True
                },
                "insights": self._generate_equity_overlay_insights(equity_curves, overlay_stats)
            }

        except Exception as e:
            logger.error(f"Error generating equity curves overlay: {e}")
            return {"error": f"Equity curves overlay generation failed: {str(e)}"}

    def _generate_trade_distribution_charts(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate trade distribution charts by scenario type."""
        try:
            distribution_analysis = {
                "trade_count_distribution": self._analyze_trade_count_distribution(scenario_results),
                "win_loss_distribution": self._analyze_win_loss_distribution(scenario_results),
                "trade_duration_distribution": self._analyze_trade_duration_distribution(scenario_results),
                "profit_loss_distribution": self._analyze_profit_loss_distribution(scenario_results)
            }

            return {
                "chart_type": "distribution_analysis",
                "title": "Trade Distribution Analysis by Scenario",
                "distributions": distribution_analysis,
                "config": {
                    "chart_style": "modern",
                    "show_statistics": True,
                    "responsive": True
                },
                "insights": self._generate_distribution_insights(distribution_analysis)
            }

        except Exception as e:
            logger.error(f"Error generating trade distribution charts: {e}")
            return {"error": f"Trade distribution generation failed: {str(e)}"}

    def _generate_scenario_highlighting_config(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate configuration for scenario-specific trade highlighting."""
        try:
            highlighting_config = {
                "scenario_colors": self._generate_scenario_colors(),
                "highlight_rules": {},
                "interactive_features": {
                    "click_to_highlight": True,
                    "hover_effects": True,
                    "toggle_scenarios": True,
                    "scenario_filters": True
                }
            }

            # Generate highlighting rules for each scenario
            for scenario, results in scenario_results.items():
                if "trades" in results and results["trades"]:
                    trades = results["trades"]

                    # Create highlighting rules based on trade characteristics
                    scenario_rules = {
                        "scenario_name": scenario,
                        "trade_markers": {
                            "entry_color": highlighting_config["scenario_colors"].get(scenario, "#888888"),
                            "exit_color": highlighting_config["scenario_colors"].get(scenario, "#888888"),
                            "marker_size": 8,
                            "marker_opacity": 0.8
                        },
                        "trade_lines": {
                            "line_color": highlighting_config["scenario_colors"].get(scenario, "#888888"),
                            "line_width": 2,
                            "line_opacity": 0.6,
                            "line_style": "solid"
                        },
                        "selection_effects": {
                            "highlight_color": "#FFD700",  # Gold for selection
                            "highlight_width": 4,
                            "fade_others": True,
                            "fade_opacity": 0.3
                        }
                    }

                    highlighting_config["highlight_rules"][scenario] = scenario_rules

            return highlighting_config

        except Exception as e:
            logger.error(f"Error generating scenario highlighting config: {e}")
            return {"error": f"Highlighting config generation failed: {str(e)}"}

    def _generate_market_condition_timeline(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate market condition timeline showing when each scenario type occurred."""
        try:
            timeline_data = []
            scenario_colors = self._generate_scenario_colors()

            # Create timeline segments for each scenario
            for i, (scenario, results) in enumerate(scenario_results.items()):
                # Simulate timeline positioning (in real implementation, this would use actual dates)
                start_time = i * 100  # Relative positioning
                duration = 90  # Standard scenario duration

                timeline_segment = {
                    "scenario": scenario,
                    "scenario_label": scenario.replace("_", " ").title(),
                    "start_time": start_time,
                    "end_time": start_time + duration,
                    "duration": duration,
                    "color": scenario_colors.get(scenario, "#888888"),
                    "performance_summary": {
                        "total_return": results.get("total_return_pct", 0),
                        "win_rate": results.get("win_rate", 0) * 100,
                        "total_trades": results.get("total_trades", 0)
                    },
                    "market_characteristics": self._get_scenario_characteristics(scenario)
                }

                timeline_data.append(timeline_segment)

            return {
                "chart_type": "timeline",
                "title": "Market Scenario Timeline",
                "subtitle": "Sequence of market conditions tested",
                "timeline_segments": timeline_data,
                "config": {
                    "show_performance_overlay": True,
                    "interactive_segments": True,
                    "zoom_enabled": True,
                    "responsive": True
                },
                "summary": {
                    "total_scenarios": len(timeline_data),
                    "total_duration": sum(segment["duration"] for segment in timeline_data),
                    "best_performing_scenario": max(timeline_data, key=lambda x: x["performance_summary"]["total_return"])["scenario"] if timeline_data else None
                }
            }

        except Exception as e:
            logger.error(f"Error generating market condition timeline: {e}")
            return {"error": f"Timeline generation failed: {str(e)}"}

    # Helper methods for visual generation
    def _generate_chart_themes(self) -> Dict[str, Any]:
        """Generate chart theme configurations."""
        return {
            "default": {
                "background_color": "#ffffff",
                "grid_color": "#e0e0e0",
                "text_color": "#333333",
                "accent_color": "#007bff"
            },
            "dark": {
                "background_color": "#2b2b2b",
                "grid_color": "#404040",
                "text_color": "#ffffff",
                "accent_color": "#4dabf7"
            },
            "high_contrast": {
                "background_color": "#ffffff",
                "grid_color": "#000000",
                "text_color": "#000000",
                "accent_color": "#ff0000"
            }
        }

    def _generate_responsive_chart_configs(self) -> Dict[str, Any]:
        """Generate responsive chart configurations for different screen sizes."""
        return {
            "mobile": {
                "max_width": 768,
                "chart_height": 300,
                "font_size": 12,
                "legend_position": "bottom"
            },
            "tablet": {
                "max_width": 1024,
                "chart_height": 400,
                "font_size": 14,
                "legend_position": "right"
            },
            "desktop": {
                "min_width": 1025,
                "chart_height": 500,
                "font_size": 16,
                "legend_position": "right"
            }
        }

    def _generate_accessibility_features(self) -> Dict[str, Any]:
        """Generate accessibility features for charts."""
        return {
            "screen_reader": {
                "alt_text_enabled": True,
                "data_tables": True,
                "keyboard_navigation": True
            },
            "color_blind_friendly": {
                "color_palette": "viridis",  # Color-blind friendly palette
                "pattern_fills": True,
                "high_contrast_mode": True
            },
            "motor_accessibility": {
                "large_click_targets": True,
                "touch_friendly": True,
                "keyboard_shortcuts": True
            }
        }

    def _generate_interactive_elements_config(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate configuration for interactive chart elements."""
        return {
            "tooltips": {
                "enabled": True,
                "show_on_hover": True,
                "custom_content": True,
                "animation": "fade"
            },
            "zoom_pan": {
                "zoom_enabled": True,
                "pan_enabled": True,
                "reset_button": True,
                "wheel_zoom": True
            },
            "selection": {
                "multi_select": True,
                "brush_selection": True,
                "click_selection": True,
                "keyboard_selection": True
            },
            "filters": {
                "scenario_filter": True,
                "performance_filter": True,
                "time_range_filter": True,
                "custom_filters": True
            }
        }

    def _save_visual_report(self, visualizations: Dict[str, Any], output_path: Path) -> None:
        """Save visual report data to file."""
        with open(output_path, 'w') as f:
            json.dump(visualizations, f, indent=2, default=str)

    # Additional helper methods for visual scenario comparison
    def _generate_radar_chart_insights(self, radar_data: List[Dict]) -> List[str]:
        """Generate insights from radar chart analysis."""
        insights = []

        if not radar_data:
            return ["No data available for radar chart analysis"]

        # Find the scenario with the most balanced performance (least variance across dimensions)
        most_balanced = None
        lowest_variance = float('inf')

        # Find the scenario with highest average performance
        highest_performer = None
        highest_avg = -float('inf')

        for scenario_data in radar_data:
            values = scenario_data.get("values", [])
            if values:
                variance = statistics.variance(values) if len(values) > 1 else 0
                avg_performance = statistics.mean(values)

                if variance < lowest_variance:
                    lowest_variance = variance
                    most_balanced = scenario_data["scenario_label"]

                if avg_performance > highest_avg:
                    highest_avg = avg_performance
                    highest_performer = scenario_data["scenario_label"]

        if most_balanced:
            insights.append(f"Most balanced performance across all metrics: {most_balanced}")
        if highest_performer:
            insights.append(f"Highest overall performance: {highest_performer}")

        # Check for scenarios that excel in specific dimensions
        dimensions = ["total_return_pct", "win_rate", "sharpe_ratio", "profit_factor", "max_drawdown_pct"]
        for i, dimension in enumerate(dimensions):
            best_scenario = max(radar_data, key=lambda x: x.get("values", [0] * 5)[i] if len(x.get("values", [])) > i else 0)
            if best_scenario and best_scenario.get("values"):
                dimension_name = dimension.replace("_", " ").title()
                insights.append(f"Best {dimension_name}: {best_scenario['scenario_label']}")

        return insights

    def _generate_heatmap_summary(self, heatmap_data: List[Dict], scenarios: List[str]) -> Dict[str, Any]:
        """Generate summary statistics for heatmap."""
        summary = {
            "total_scenarios": len(scenarios),
            "total_metrics": len(heatmap_data),
            "best_overall_scenario": None,
            "worst_overall_scenario": None,
            "most_consistent_metric": None,
            "most_variable_metric": None
        }

        # Calculate overall scenario scores (average across all metrics)
        scenario_scores = {}
        for scenario_idx, scenario in enumerate(scenarios):
            total_score = 0
            metric_count = 0

            for metric_data in heatmap_data:
                values = metric_data.get("values", [])
                if len(values) > scenario_idx:
                    total_score += values[scenario_idx]
                    metric_count += 1

            if metric_count > 0:
                scenario_scores[scenario] = total_score / metric_count

        if scenario_scores:
            best_scenario = max(scenario_scores.items(), key=lambda x: x[1])
            worst_scenario = min(scenario_scores.items(), key=lambda x: x[1])

            summary["best_overall_scenario"] = {
                "scenario": best_scenario[0],
                "score": round(best_scenario[1], 2)
            }
            summary["worst_overall_scenario"] = {
                "scenario": worst_scenario[0],
                "score": round(worst_scenario[1], 2)
            }

        # Find most and least consistent metrics (lowest/highest variance across scenarios)
        if heatmap_data:
            metric_variances = []
            for metric_data in heatmap_data:
                values = metric_data.get("values", [])
                if len(values) > 1:
                    variance = statistics.variance(values)
                    metric_variances.append((metric_data["metric_label"], variance))

            if metric_variances:
                most_consistent = min(metric_variances, key=lambda x: x[1])
                most_variable = max(metric_variances, key=lambda x: x[1])

                summary["most_consistent_metric"] = most_consistent[0]
                summary["most_variable_metric"] = most_variable[0]

        return summary

    def _calculate_overlay_statistics(self, equity_curves: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for equity curve overlay."""
        stats = {
            "total_curves": len(equity_curves),
            "best_performing_scenario": None,
            "worst_performing_scenario": None,
            "most_volatile_scenario": None,
            "least_volatile_scenario": None,
            "correlation_summary": None
        }

        if not equity_curves:
            return stats

        scenario_stats = {}

        for curve in equity_curves:
            data = curve.get("data", [])
            scenario = curve.get("scenario")

            if data and scenario:
                percentage_gains = [point.get("percentage_gain", 0) for point in data]

                if percentage_gains:
                    final_return = percentage_gains[-1]
                    volatility = statistics.stdev(percentage_gains) if len(percentage_gains) > 1 else 0

                    scenario_stats[scenario] = {
                        "final_return": final_return,
                        "volatility": volatility,
                        "scenario_label": curve.get("scenario_label", scenario)
                    }

        if scenario_stats:
            # Best/worst performers
            best_performer = max(scenario_stats.items(), key=lambda x: x[1]["final_return"])
            worst_performer = min(scenario_stats.items(), key=lambda x: x[1]["final_return"])

            stats["best_performing_scenario"] = {
                "scenario": best_performer[1]["scenario_label"],
                "final_return": round(best_performer[1]["final_return"], 2)
            }
            stats["worst_performing_scenario"] = {
                "scenario": worst_performer[1]["scenario_label"],
                "final_return": round(worst_performer[1]["final_return"], 2)
            }

            # Most/least volatile
            most_volatile = max(scenario_stats.items(), key=lambda x: x[1]["volatility"])
            least_volatile = min(scenario_stats.items(), key=lambda x: x[1]["volatility"])

            stats["most_volatile_scenario"] = {
                "scenario": most_volatile[1]["scenario_label"],
                "volatility": round(most_volatile[1]["volatility"], 2)
            }
            stats["least_volatile_scenario"] = {
                "scenario": least_volatile[1]["scenario_label"],
                "volatility": round(least_volatile[1]["volatility"], 2)
            }

        return stats

    def _generate_equity_overlay_insights(self, equity_curves: List[Dict], overlay_stats: Dict[str, Any]) -> List[str]:
        """Generate insights from equity curve overlay analysis."""
        insights = []

        if not equity_curves:
            return ["No equity curve data available for analysis"]

        # Add insights from overlay statistics
        if overlay_stats.get("best_performing_scenario"):
            best = overlay_stats["best_performing_scenario"]
            insights.append(f"Best performing scenario: {best['scenario']} ({best['final_return']}% final return)")

        if overlay_stats.get("worst_performing_scenario"):
            worst = overlay_stats["worst_performing_scenario"]
            insights.append(f"Worst performing scenario: {worst['scenario']} ({worst['final_return']}% final return)")

        if overlay_stats.get("most_volatile_scenario"):
            volatile = overlay_stats["most_volatile_scenario"]
            insights.append(f"Most volatile scenario: {volatile['scenario']} ({volatile['volatility']}% volatility)")

        # Analyze curve patterns
        positive_scenarios = sum(1 for curve in equity_curves
                               if curve.get("data") and curve["data"][-1].get("percentage_gain", 0) > 0)
        total_scenarios = len(equity_curves)

    def _is_valid_result_data(self, result_data: Any) -> bool:
        """
        Safely validate result data to prevent DataFrame comparison errors.

        Args:
            result_data: The result data to validate

        Returns:
            True if the result data is valid for processing, False otherwise
        """
        try:
            # Check if result_data is None
            if result_data is None:
                return False

            # Check if it's a dictionary with expected structure
            if not isinstance(result_data, dict):
                return False

            # Check if it has performance_metrics
            performance_metrics = result_data.get("performance_metrics")
            if not performance_metrics or not isinstance(performance_metrics, dict):
                return False

            # Check if total_return exists and is a valid number
            total_return = performance_metrics.get("total_return")
            if total_return is None:
                return False

            # Ensure total_return is a scalar number, not a DataFrame or Series
            if hasattr(total_return, 'empty'):  # pandas DataFrame/Series check
                return False

            # Check if it's a valid numeric type
            if not isinstance(total_return, (int, float)):
                return False

            # Additional validation for other expected fields
            expected_fields = ["win_rate", "profit_factor", "max_drawdown", "sharpe_ratio"]
            for field in expected_fields:
                value = performance_metrics.get(field)
                if value is not None and hasattr(value, 'empty'):  # DataFrame/Series check
                    return False

            return True

        except Exception as e:
            logger.warning(f"Error validating result data: {e}")
            return False
