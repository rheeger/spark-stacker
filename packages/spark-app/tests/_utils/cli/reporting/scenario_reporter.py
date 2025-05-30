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
            report = {
                "strategy_name": strategy_name,
                "report_type": "multi_scenario_analysis",
                "generated_at": datetime.now().isoformat(),
                "scenarios_analyzed": list(scenario_results.keys()),
                "strategy_configuration": self._get_strategy_configuration(strategy_name),
                "scenario_performance_comparison": self._compare_scenario_performance(scenario_results),
                "robustness_analysis": self._analyze_strategy_robustness(scenario_results),
                "scenario_correlation_analysis": self._analyze_scenario_correlations(scenario_results),
                "worst_case_analysis": self._analyze_worst_case_scenarios(scenario_results),
                "adaptability_analysis": self._analyze_strategy_adaptability(scenario_results),
                "scenario_rankings": self._rank_scenarios_by_performance(scenario_results),
                "optimization_recommendations": self._generate_scenario_optimizations(scenario_results),
                "scenario_insights": self._generate_scenario_insights(scenario_results),
                "visualization_data": self._prepare_visualization_data(scenario_results)
            }

            if output_path:
                self._save_report(report, output_path)

            return report

        except Exception as e:
            logger.error(f"Error generating multi-scenario report for {strategy_name}: {e}")
            raise

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
                "indicators": list(config.get("indicators", {}).keys()),
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
        """Generate actionable insights from scenario analysis."""
        insights = []

        # Performance insights
        positive_scenarios = sum(1 for results in scenario_results.values()
                               if results.get("performance_metrics", {}).get("total_return", 0) > 0)
        total_scenarios = len(scenario_results)

        if positive_scenarios / total_scenarios >= 0.75:
            insights.append(f"Strategy shows strong robustness with positive returns in {positive_scenarios}/{total_scenarios} scenarios")
        elif positive_scenarios / total_scenarios >= 0.5:
            insights.append(f"Strategy shows moderate robustness with positive returns in {positive_scenarios}/{total_scenarios} scenarios")
        else:
            insights.append(f"Strategy shows limited robustness with positive returns in only {positive_scenarios}/{total_scenarios} scenarios")

        # Volatility insights
        real_data_result = scenario_results.get("real_data")
        if real_data_result:
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
            if equity_curve:
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
        """Analyze which scenarios are most difficult across all strategies."""
        return {"placeholder": "Scenario difficulty analysis not yet implemented"}

    def _identify_best_performers_by_scenario(self, strategies: Dict) -> Dict[str, Any]:
        """Identify best performing strategy for each scenario."""
        return {"placeholder": "Best performers by scenario not yet implemented"}

    def _rank_strategies_by_consistency(self, strategies: Dict) -> Dict[str, Any]:
        """Rank strategies by consistency across scenarios."""
        return {"placeholder": "Consistency rankings not yet implemented"}

    def _analyze_cross_scenario_diversification(self, strategies: Dict) -> Dict[str, Any]:
        """Analyze diversification benefits across scenarios."""
        return {"placeholder": "Cross-scenario diversification not yet implemented"}

    def _recommend_scenario_aware_portfolio(self, strategies: Dict) -> Dict[str, Any]:
        """Recommend portfolio allocation considering scenario performance."""
        return {"placeholder": "Scenario-aware portfolio recommendations not yet implemented"}

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

    def _save_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """Save scenario report to file."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

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
