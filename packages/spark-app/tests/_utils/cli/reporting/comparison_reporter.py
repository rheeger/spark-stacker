"""
Comparison Report Generation Module

This module handles all types of comparison reporting including:
- Strategy-to-strategy comparisons
- Indicator-to-indicator comparisons
- Cross-type comparisons (strategy vs indicator)
- Side-by-side comparison displays
- Ranking and scoring displays
- Statistical significance testing
- Comparison visualization generation
- Comparison result export
"""

import json
import logging
import statistics
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from core.config_manager import ConfigManager
from managers.comparison_manager import ComparisonManager
from reporting.scenario_reporter import ScenarioReporter

logger = logging.getLogger(__name__)


class ComparisonReporter:
    """
    Handles comparison report generation and analysis.

    Centralizes all comparison reporting logic including side-by-side comparisons,
    statistical analysis, ranking, and visualization generation.
    """

    def __init__(self, config_manager: ConfigManager, comparison_manager: ComparisonManager):
        """
        Initialize the ComparisonReporter.

        Args:
            config_manager: ConfigManager instance for configuration access
            comparison_manager: ComparisonManager for comparison logic coordination
        """
        self.config_manager = config_manager
        self.comparison_manager = comparison_manager
        # Initialize ScenarioReporter for cross-scenario analysis
        self.scenario_reporter = ScenarioReporter(config_manager)

    def generate_strategy_comparison_report(
        self,
        strategy_names: List[str],
        comparison_results: Dict[str, Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive strategy comparison report.

        Args:
            strategy_names: List of strategy names being compared
            comparison_results: Dictionary of strategy results keyed by strategy name
            output_path: Optional path to save the report

        Returns:
            Dictionary containing the complete comparison report
        """
        logger.info(f"Generating strategy comparison report for: {strategy_names}")

        try:
            report = {
                "comparison_type": "strategy_comparison",
                "generated_at": datetime.now().isoformat(),
                "strategies_compared": strategy_names,
                "strategy_configurations": self._get_strategy_configurations(strategy_names),
                "performance_comparison": self._compare_strategy_performance(comparison_results),
                "risk_comparison": self._compare_strategy_risk_metrics(comparison_results),
                "ranking_analysis": self._rank_strategies(comparison_results),
                "statistical_analysis": self._perform_statistical_analysis(comparison_results),
                "correlation_analysis": self._analyze_strategy_correlations(comparison_results),
                "diversification_analysis": self._analyze_diversification_benefits(comparison_results),
                "market_condition_analysis": self._analyze_performance_by_market_condition(comparison_results),
                "sensitivity_analysis": self._perform_sensitivity_analysis(strategy_names, comparison_results),
                "optimization_recommendations": self._generate_comparison_optimizations(comparison_results),
                "allocation_suggestions": self._suggest_portfolio_allocation(comparison_results)
            }

            if output_path:
                self._save_report(report, output_path)

            return report

        except Exception as e:
            logger.error(f"Error generating strategy comparison report: {e}")
            raise

    def generate_cross_type_comparison_report(
        self,
        strategies: Dict[str, Dict[str, Any]],
        indicators: Dict[str, Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate comparison report between strategies and indicators.

        Args:
            strategies: Dictionary of strategy results
            indicators: Dictionary of indicator results
            output_path: Optional path to save the report

        Returns:
            Dictionary containing the cross-type comparison report
        """
        logger.info("Generating cross-type comparison report (strategies vs indicators)")

        try:
            report = {
                "comparison_type": "cross_type_comparison",
                "generated_at": datetime.now().isoformat(),
                "strategies_analyzed": list(strategies.keys()),
                "indicators_analyzed": list(indicators.keys()),
                "performance_comparison": self._compare_cross_type_performance(strategies, indicators),
                "complexity_analysis": self._analyze_complexity_vs_performance(strategies, indicators),
                "strategy_vs_indicator_insights": self._analyze_strategy_vs_indicator_insights(strategies, indicators),
                "migration_suggestions": self._suggest_indicator_to_strategy_migration(strategies, indicators)
            }

            if output_path:
                self._save_report(report, output_path)

            return report

        except Exception as e:
            logger.error(f"Error generating cross-type comparison report: {e}")
            raise

    def _get_strategy_configurations(self, strategy_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get configuration details for all strategies."""
        configurations = {}

        for strategy_name in strategy_names:
            try:
                config = self.config_manager.get_strategy_config(strategy_name)
                configurations[strategy_name] = {
                    "market": config.get("market"),
                    "exchange": config.get("exchange"),
                    "timeframe": config.get("timeframe"),
                    "indicators": list(config.get("indicators", {}).keys()),
                    "position_sizing": config.get("position_sizing", {}),
                    "risk_settings": {
                        "stop_loss": config.get("stop_loss"),
                        "take_profit": config.get("take_profit")
                    }
                }
            except Exception as e:
                logger.warning(f"Could not get configuration for strategy {strategy_name}: {e}")
                configurations[strategy_name] = {"error": str(e)}

        return configurations

    def _compare_strategy_performance(self, comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance metrics across strategies."""
        metrics = ["total_return", "win_rate", "profit_factor", "max_drawdown", "sharpe_ratio"]

        comparison = {
            "side_by_side": {},
            "best_performers": {},
            "performance_ranges": {},
            "normalized_scores": {}
        }

        # Side-by-side comparison
        for metric in metrics:
            comparison["side_by_side"][metric] = {}
            values = []

            for strategy_name, results in comparison_results.items():
                performance = results.get("performance_metrics", {})
                value = performance.get(metric, 0)
                comparison["side_by_side"][metric][strategy_name] = value
                values.append(value)

            # Best performers for each metric
            if values:
                if metric in ["total_return", "win_rate", "profit_factor", "sharpe_ratio"]:
                    best_value = max(values)
                    best_strategy = [name for name, val in comparison["side_by_side"][metric].items() if val == best_value][0]
                else:  # max_drawdown - lower is better
                    best_value = min(values)
                    best_strategy = [name for name, val in comparison["side_by_side"][metric].items() if val == best_value][0]

                comparison["best_performers"][metric] = {
                    "strategy": best_strategy,
                    "value": best_value
                }

                # Performance ranges
                comparison["performance_ranges"][metric] = {
                    "min": min(values),
                    "max": max(values),
                    "average": statistics.mean(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0
                }

        # Normalized scores (0-100 scale)
        for strategy_name in comparison_results.keys():
            comparison["normalized_scores"][strategy_name] = self._calculate_normalized_score(
                strategy_name, comparison["side_by_side"]
            )

        return comparison

    def _compare_strategy_risk_metrics(self, comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare risk metrics across strategies."""
        risk_metrics = ["max_drawdown", "var_95", "consecutive_losses", "average_loss"]

        risk_comparison = {
            "risk_profiles": {},
            "risk_rankings": {},
            "risk_adjusted_returns": {}
        }

        for strategy_name, results in comparison_results.items():
            performance = results.get("performance_metrics", {})
            risk_analysis = results.get("risk_analysis", {})

            # Risk profile
            risk_comparison["risk_profiles"][strategy_name] = {
                "max_drawdown": performance.get("max_drawdown", 0),
                "volatility": self._calculate_volatility(results),
                "var_95": risk_analysis.get("var_95", 0),
                "risk_score": self._calculate_risk_score(performance, risk_analysis)
            }

            # Risk-adjusted returns
            total_return = performance.get("total_return", 0)
            max_drawdown = performance.get("max_drawdown", 1)
            sharpe_ratio = performance.get("sharpe_ratio", 0)

            risk_comparison["risk_adjusted_returns"][strategy_name] = {
                "return_drawdown_ratio": total_return / max_drawdown if max_drawdown > 0 else 0,
                "sharpe_ratio": sharpe_ratio,
                "calmar_ratio": total_return / max_drawdown if max_drawdown > 0 else 0
            }

        # Risk rankings (lower risk score is better)
        risk_scores = [(name, data["risk_score"]) for name, data in risk_comparison["risk_profiles"].items()]
        risk_scores.sort(key=lambda x: x[1])

        risk_comparison["risk_rankings"] = {
            "lowest_risk": risk_scores[0] if risk_scores else None,
            "highest_risk": risk_scores[-1] if risk_scores else None,
            "full_ranking": risk_scores
        }

        return risk_comparison

    def _rank_strategies(self, comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Rank strategies by multiple criteria."""
        ranking_criteria = {
            "total_return": {"weight": 0.3, "higher_better": True},
            "sharpe_ratio": {"weight": 0.25, "higher_better": True},
            "max_drawdown": {"weight": 0.2, "higher_better": False},
            "win_rate": {"weight": 0.15, "higher_better": True},
            "profit_factor": {"weight": 0.1, "higher_better": True}
        }

        strategy_scores = {}

        # Calculate weighted scores for each strategy
        for strategy_name, results in comparison_results.items():
            performance = results.get("performance_metrics", {})
            total_score = 0

            for criterion, config in ranking_criteria.items():
                value = performance.get(criterion, 0)

                # Normalize value (0-100 scale) - simplified normalization
                if config["higher_better"]:
                    normalized = min(100, max(0, value * 10))
                else:
                    normalized = min(100, max(0, 100 - value))

                weighted_score = normalized * config["weight"]
                total_score += weighted_score

            strategy_scores[strategy_name] = {
                "total_score": round(total_score, 2),
                "breakdown": {
                    criterion: performance.get(criterion, 0)
                    for criterion in ranking_criteria.keys()
                }
            }

        # Sort by total score
        ranked_strategies = sorted(
            strategy_scores.items(),
            key=lambda x: x[1]["total_score"],
            reverse=True
        )

        return {
            "ranking_methodology": ranking_criteria,
            "strategy_scores": strategy_scores,
            "final_ranking": ranked_strategies,
            "top_strategy": ranked_strategies[0] if ranked_strategies else None,
            "performance_tiers": self._categorize_performance_tiers(ranked_strategies)
        }

    def _perform_statistical_analysis(self, comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis on strategy comparison."""
        returns_data = {}

        # Extract returns data for each strategy
        for strategy_name, results in comparison_results.items():
            trades = results.get("trades", [])
            returns = [trade.get("pnl", 0) for trade in trades]
            if returns:
                returns_data[strategy_name] = returns

        if len(returns_data) < 2:
            return {"error": "Need at least 2 strategies with trades for statistical analysis"}

        statistical_tests = {}

        # Pairwise t-tests for significance
        strategy_pairs = list(combinations(returns_data.keys(), 2))

        for strategy1, strategy2 in strategy_pairs:
            returns1 = returns_data[strategy1]
            returns2 = returns_data[strategy2]

            # Simplified t-test (would use scipy.stats in real implementation)
            mean1 = statistics.mean(returns1)
            mean2 = statistics.mean(returns2)

            statistical_tests[f"{strategy1}_vs_{strategy2}"] = {
                "mean_difference": round(mean1 - mean2, 4),
                "strategy1_mean": round(mean1, 4),
                "strategy2_mean": round(mean2, 4),
                "sample_sizes": {"strategy1": len(returns1), "strategy2": len(returns2)},
                "significant_difference": abs(mean1 - mean2) > 0.1  # Simplified significance test
            }

        return {
            "pairwise_comparisons": statistical_tests,
            "summary": {
                "total_comparisons": len(statistical_tests),
                "significant_differences": sum(1 for test in statistical_tests.values() if test["significant_difference"])
            }
        }

    def _analyze_strategy_correlations(self, comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between strategy returns."""
        returns_data = {}

        # Extract synchronized returns data
        for strategy_name, results in comparison_results.items():
            trades = results.get("trades", [])
            returns = [trade.get("pnl", 0) for trade in trades]
            if returns:
                returns_data[strategy_name] = returns

        if len(returns_data) < 2:
            return {"error": "Need at least 2 strategies for correlation analysis"}

        correlations = {}
        strategy_pairs = list(combinations(returns_data.keys(), 2))

        for strategy1, strategy2 in strategy_pairs:
            returns1 = returns_data[strategy1]
            returns2 = returns_data[strategy2]

            # Simplified correlation calculation (would use numpy in real implementation)
            correlation = self._calculate_correlation(returns1, returns2)

            correlations[f"{strategy1}_vs_{strategy2}"] = {
                "correlation": round(correlation, 3),
                "interpretation": self._interpret_correlation(correlation)
            }

        return {
            "pairwise_correlations": correlations,
            "diversification_potential": self._assess_diversification_potential(correlations)
        }

    def _analyze_diversification_benefits(self, comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze diversification benefits of combining strategies."""
        strategy_returns = {}

        for strategy_name, results in comparison_results.items():
            performance = results.get("performance_metrics", {})
            strategy_returns[strategy_name] = {
                "total_return": performance.get("total_return", 0),
                "max_drawdown": performance.get("max_drawdown", 0),
                "sharpe_ratio": performance.get("sharpe_ratio", 0),
                "volatility": performance.get("volatility", 0)
            }

        # Portfolio combinations analysis
        portfolio_combinations = []

        for r in range(2, min(len(strategy_returns) + 1, 4)):  # Combinations of 2-3 strategies
            for combo in combinations(strategy_returns.keys(), r):
                portfolio_performance = self._calculate_portfolio_performance(combo, strategy_returns)
                portfolio_combinations.append({
                    "strategies": combo,
                    "portfolio_return": portfolio_performance["return"],
                    "portfolio_risk": portfolio_performance["risk"],
                    "diversification_ratio": portfolio_performance["diversification_ratio"]
                })

        # Sort by risk-adjusted return
        portfolio_combinations.sort(
            key=lambda x: x["portfolio_return"] / x["portfolio_risk"] if x["portfolio_risk"] > 0 else 0,
            reverse=True
        )

        return {
            "individual_strategies": strategy_returns,
            "portfolio_combinations": portfolio_combinations[:5],  # Top 5 combinations
            "best_portfolio": portfolio_combinations[0] if portfolio_combinations else None,
            "diversification_insights": self._generate_diversification_insights(portfolio_combinations)
        }

    def _generate_comparison_optimizations(self, comparison_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on comparison analysis."""
        optimizations = []

        # Find best performing strategy for benchmarking
        best_performer = None
        best_score = 0

        for strategy_name, results in comparison_results.items():
            performance = results.get("performance_metrics", {})
            score = performance.get("sharpe_ratio", 0)
            if score > best_score:
                best_score = score
                best_performer = strategy_name

        if not best_performer:
            return optimizations

        best_performance = comparison_results[best_performer]["performance_metrics"]

        # Compare each strategy against the best performer
        for strategy_name, results in comparison_results.items():
            if strategy_name == best_performer:
                continue

            performance = results.get("performance_metrics", {})

            # Win rate optimization
            if performance.get("win_rate", 0) < best_performance.get("win_rate", 0) * 0.8:
                optimizations.append({
                    "strategy": strategy_name,
                    "type": "win_rate_improvement",
                    "current_value": performance.get("win_rate", 0),
                    "target_value": best_performance.get("win_rate", 0),
                    "suggestion": f"Consider adopting entry criteria from {best_performer}"
                })

            # Risk optimization
            if performance.get("max_drawdown", 0) > best_performance.get("max_drawdown", 0) * 1.5:
                optimizations.append({
                    "strategy": strategy_name,
                    "type": "risk_reduction",
                    "current_value": performance.get("max_drawdown", 0),
                    "target_value": best_performance.get("max_drawdown", 0),
                    "suggestion": f"Consider adopting risk management from {best_performer}"
                })

        return optimizations

    def _suggest_portfolio_allocation(self, comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Suggest optimal portfolio allocation based on strategy performance."""
        strategy_metrics = {}

        for strategy_name, results in comparison_results.items():
            performance = results.get("performance_metrics", {})
            strategy_metrics[strategy_name] = {
                "return": performance.get("total_return", 0),
                "risk": performance.get("max_drawdown", 1),
                "sharpe": performance.get("sharpe_ratio", 0)
            }

        # Simple allocation based on risk-adjusted returns
        total_sharpe = sum(metrics["sharpe"] for metrics in strategy_metrics.values() if metrics["sharpe"] > 0)

        allocations = {}
        if total_sharpe > 0:
            for strategy_name, metrics in strategy_metrics.items():
                if metrics["sharpe"] > 0:
                    allocation = (metrics["sharpe"] / total_sharpe) * 100
                    allocations[strategy_name] = round(allocation, 1)
                else:
                    allocations[strategy_name] = 0
        else:
            # Equal allocation if no positive Sharpe ratios
            equal_allocation = 100 / len(strategy_metrics)
            for strategy_name in strategy_metrics.keys():
                allocations[strategy_name] = round(equal_allocation, 1)

        return {
            "recommended_allocations": allocations,
            "allocation_method": "sharpe_ratio_weighted",
            "portfolio_expected_return": self._calculate_portfolio_expected_return(allocations, strategy_metrics),
            "portfolio_expected_risk": self._calculate_portfolio_expected_risk(allocations, strategy_metrics)
        }

    def _analyze_performance_by_market_condition(self, comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze strategy performance by different market conditions using scenario data.
        Coordinates with ScenarioReporter for cross-scenario analysis.
        """
        logger.info("Analyzing performance by market condition")

        market_condition_analysis = {
            "scenario_performance": {},
            "best_performers_by_condition": {},
            "worst_performers_by_condition": {},
            "condition_rankings": {},
            "adaptability_scores": {}
        }

        # Define market conditions (scenarios)
        market_conditions = ["bull", "bear", "sideways", "volatile", "low-vol", "choppy", "gaps"]

        # Extract scenario-specific performance data
        for condition in market_conditions:
            market_condition_analysis["scenario_performance"][condition] = {}
            condition_returns = {}

            for strategy_name, results in comparison_results.items():
                scenario_results = results.get("scenario_results", {})
                condition_result = scenario_results.get(condition, {})

                if condition_result:
                    performance = condition_result.get("performance_metrics", {})
                    total_return = performance.get("total_return", 0)
                    condition_returns[strategy_name] = total_return

                    market_condition_analysis["scenario_performance"][condition][strategy_name] = {
                        "total_return": total_return,
                        "win_rate": performance.get("win_rate", 0),
                        "max_drawdown": performance.get("max_drawdown", 0),
                        "sharpe_ratio": performance.get("sharpe_ratio", 0)
                    }

            # Identify best and worst performers for this condition
            if condition_returns:
                best_strategy = max(condition_returns.keys(), key=lambda x: condition_returns[x])
                worst_strategy = min(condition_returns.keys(), key=lambda x: condition_returns[x])

                market_condition_analysis["best_performers_by_condition"][condition] = {
                    "strategy": best_strategy,
                    "return": condition_returns[best_strategy]
                }
                market_condition_analysis["worst_performers_by_condition"][condition] = {
                    "strategy": worst_strategy,
                    "return": condition_returns[worst_strategy]
                }

                # Rank strategies for this condition
                ranked_strategies = sorted(condition_returns.items(), key=lambda x: x[1], reverse=True)
                market_condition_analysis["condition_rankings"][condition] = ranked_strategies

        # Calculate adaptability scores (consistency across conditions)
        for strategy_name in comparison_results.keys():
            returns_across_conditions = []

            for condition in market_conditions:
                condition_data = market_condition_analysis["scenario_performance"].get(condition, {})
                strategy_data = condition_data.get(strategy_name, {})
                if strategy_data:
                    returns_across_conditions.append(strategy_data.get("total_return", 0))

            if returns_across_conditions:
                # Calculate coefficient of variation (lower is more consistent)
                mean_return = statistics.mean(returns_across_conditions)
                if mean_return != 0:
                    std_dev = statistics.stdev(returns_across_conditions) if len(returns_across_conditions) > 1 else 0
                    cv = std_dev / abs(mean_return)
                    adaptability_score = max(0, 100 - (cv * 50))  # Convert to 0-100 scale
                else:
                    adaptability_score = 0

                market_condition_analysis["adaptability_scores"][strategy_name] = {
                    "score": round(adaptability_score, 2),
                    "mean_return": round(mean_return, 2),
                    "std_dev": round(std_dev, 2) if len(returns_across_conditions) > 1 else 0,
                    "coefficient_of_variation": round(cv, 3) if mean_return != 0 else 0
                }

        return market_condition_analysis

    def _perform_sensitivity_analysis(self, strategy_names: List[str], comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform sensitivity analysis for position sizing and timeframe changes.
        """
        logger.info("Performing sensitivity analysis for strategy parameters")

        sensitivity_analysis = {
            "position_sizing_sensitivity": {},
            "timeframe_sensitivity": {},
            "parameter_impact_rankings": {},
            "optimization_potential": {}
        }

        for strategy_name in strategy_names:
            try:
                strategy_config = self.config_manager.get_strategy_config(strategy_name)
                if not strategy_config:
                    continue

                current_results = comparison_results.get(strategy_name, {})
                baseline_return = current_results.get("performance_metrics", {}).get("total_return", 0)

                # Position sizing sensitivity analysis
                current_position_sizing = strategy_config.get("position_sizing", {})
                position_sizing_variants = self._generate_position_sizing_variants(current_position_sizing)

                sensitivity_analysis["position_sizing_sensitivity"][strategy_name] = {
                    "baseline_return": baseline_return,
                    "current_method": current_position_sizing.get("method", "unknown"),
                    "sensitivity_estimates": self._estimate_position_sizing_impact(
                        strategy_name, position_sizing_variants, baseline_return
                    )
                }

                # Timeframe sensitivity analysis
                current_timeframe = strategy_config.get("timeframe", "1h")
                timeframe_variants = self._generate_timeframe_variants(current_timeframe)

                sensitivity_analysis["timeframe_sensitivity"][strategy_name] = {
                    "baseline_return": baseline_return,
                    "current_timeframe": current_timeframe,
                    "sensitivity_estimates": self._estimate_timeframe_impact(
                        strategy_name, timeframe_variants, baseline_return
                    )
                }

                # Calculate optimization potential
                max_estimated_improvement = 0
                best_optimization_type = None

                for variant_data in sensitivity_analysis["position_sizing_sensitivity"][strategy_name]["sensitivity_estimates"]:
                    improvement = variant_data.get("estimated_improvement", 0)
                    if improvement > max_estimated_improvement:
                        max_estimated_improvement = improvement
                        best_optimization_type = f"Position sizing: {variant_data['method']}"

                for variant_data in sensitivity_analysis["timeframe_sensitivity"][strategy_name]["sensitivity_estimates"]:
                    improvement = variant_data.get("estimated_improvement", 0)
                    if improvement > max_estimated_improvement:
                        max_estimated_improvement = improvement
                        best_optimization_type = f"Timeframe: {variant_data['timeframe']}"

                sensitivity_analysis["optimization_potential"][strategy_name] = {
                    "max_estimated_improvement": round(max_estimated_improvement, 2),
                    "best_optimization": best_optimization_type,
                    "optimization_priority": "high" if max_estimated_improvement > 20 else
                                          "medium" if max_estimated_improvement > 10 else "low"
                }

            except Exception as e:
                logger.warning(f"Sensitivity analysis failed for strategy {strategy_name}: {e}")
                sensitivity_analysis["optimization_potential"][strategy_name] = {
                    "error": str(e)
                }

        # Rank strategies by optimization potential
        optimization_rankings = sorted(
            [(name, data.get("max_estimated_improvement", 0))
             for name, data in sensitivity_analysis["optimization_potential"].items()
             if "max_estimated_improvement" in data],
            key=lambda x: x[1],
            reverse=True
        )

        sensitivity_analysis["parameter_impact_rankings"] = {
            "by_optimization_potential": optimization_rankings,
            "high_priority_strategies": [
                name for name, data in sensitivity_analysis["optimization_potential"].items()
                if data.get("optimization_priority") == "high"
            ]
        }

        return sensitivity_analysis

    def _generate_position_sizing_variants(self, current_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate position sizing variants for sensitivity analysis."""
        current_method = current_config.get("method", "fixed_usd")
        current_value = current_config.get("value", 100)

        variants = []

        if current_method == "fixed_usd":
            # Test different USD amounts
            test_amounts = [50, 100, 200, 500, 1000]
            for amount in test_amounts:
                if amount != current_value:
                    variants.append({"method": "fixed_usd", "value": amount})

            # Test percentage-based sizing
            variants.extend([
                {"method": "percent_balance", "value": 1.0},
                {"method": "percent_balance", "value": 2.0},
                {"method": "percent_balance", "value": 5.0}
            ])

        elif current_method == "percent_balance":
            # Test different percentages
            test_percentages = [0.5, 1.0, 2.0, 5.0, 10.0]
            for percentage in test_percentages:
                if percentage != current_value:
                    variants.append({"method": "percent_balance", "value": percentage})

            # Test fixed USD sizing
            variants.extend([
                {"method": "fixed_usd", "value": 100},
                {"method": "fixed_usd", "value": 500},
                {"method": "fixed_usd", "value": 1000}
            ])

        # Add volatility-based sizing
        variants.append({"method": "volatility_adjusted", "value": 2.0})

        return variants[:5]  # Limit to 5 variants for efficiency

    def _generate_timeframe_variants(self, current_timeframe: str) -> List[str]:
        """Generate timeframe variants for sensitivity analysis."""
        all_timeframes = ["15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d"]

        try:
            current_index = all_timeframes.index(current_timeframe)
        except ValueError:
            current_index = 2  # Default to 1h if current timeframe not found

        # Generate variants around current timeframe
        variants = []
        for i in range(max(0, current_index - 2), min(len(all_timeframes), current_index + 3)):
            if i != current_index:
                variants.append(all_timeframes[i])

        return variants

    def _estimate_position_sizing_impact(
        self, strategy_name: str, variants: List[Dict[str, Any]], baseline_return: float
    ) -> List[Dict[str, Any]]:
        """Estimate the impact of different position sizing methods."""
        estimates = []

        for variant in variants:
            method = variant.get("method", "unknown")
            value = variant.get("value", 0)

            # Simplified impact estimation based on method characteristics
            estimated_impact = 0
            risk_factor = 1.0

            if method == "fixed_usd":
                # Fixed USD sizing impact based on amount
                if value > 500:
                    estimated_impact = 5  # Higher amounts may reduce relative transaction costs
                    risk_factor = 1.2  # But increase risk
                elif value < 100:
                    estimated_impact = -10  # Very small positions may hurt performance
                    risk_factor = 0.8

            elif method == "percent_balance":
                # Percentage-based sizing impact
                if 1.0 <= value <= 3.0:
                    estimated_impact = 10  # Optimal range for many strategies
                    risk_factor = 1.0
                elif value > 5.0:
                    estimated_impact = 15  # Higher returns but much higher risk
                    risk_factor = 2.0
                elif value < 1.0:
                    estimated_impact = -5  # Too conservative
                    risk_factor = 0.5

            elif method == "volatility_adjusted":
                estimated_impact = 12  # Generally good for adapting to market conditions
                risk_factor = 0.9

            # Estimate new return (simplified)
            estimated_return = baseline_return * (1 + estimated_impact / 100) * risk_factor
            estimated_improvement = ((estimated_return - baseline_return) / baseline_return * 100) if baseline_return != 0 else 0

            estimates.append({
                "method": method,
                "value": value,
                "estimated_return": round(estimated_return, 2),
                "estimated_improvement": round(estimated_improvement, 2),
                "risk_factor": risk_factor,
                "confidence": "low"  # These are rough estimates
            })

        return estimates

    def _estimate_timeframe_impact(
        self, strategy_name: str, variants: List[str], baseline_return: float
    ) -> List[Dict[str, Any]]:
        """Estimate the impact of different timeframes."""
        estimates = []

        # Get strategy configuration to understand indicator characteristics
        try:
            strategy_config = self.config_manager.get_strategy_config(strategy_name)
            indicators = strategy_config.get("indicators", {})
        except:
            indicators = {}

        for timeframe in variants:
            # Simplified impact estimation based on timeframe characteristics
            estimated_impact = 0

            # Shorter timeframes
            if timeframe in ["15m", "30m"]:
                if any("momentum" in str(indicator).lower() for indicator in indicators.values()):
                    estimated_impact = 8  # Momentum strategies may benefit from shorter timeframes
                else:
                    estimated_impact = -5  # But may add noise for other strategies

            # Medium timeframes
            elif timeframe in ["2h", "4h"]:
                estimated_impact = 3  # Generally good balance

            # Longer timeframes
            elif timeframe in ["8h", "12h", "1d"]:
                if any("trend" in str(indicator).lower() for indicator in indicators.values()):
                    estimated_impact = 10  # Trend strategies may benefit from longer timeframes
                else:
                    estimated_impact = 0  # Neutral for others

            # Estimate new return
            estimated_return = baseline_return * (1 + estimated_impact / 100)
            estimated_improvement = ((estimated_return - baseline_return) / baseline_return * 100) if baseline_return != 0 else 0

            estimates.append({
                "timeframe": timeframe,
                "estimated_return": round(estimated_return, 2),
                "estimated_improvement": round(estimated_improvement, 2),
                "rationale": self._get_timeframe_rationale(timeframe, indicators),
                "confidence": "low"  # These are rough estimates
            })

        return estimates

    def _get_timeframe_rationale(self, timeframe: str, indicators: Dict[str, Any]) -> str:
        """Get rationale for timeframe impact estimation."""
        if timeframe in ["15m", "30m"]:
            return "Shorter timeframes may capture quick momentum but add noise"
        elif timeframe in ["2h", "4h"]:
            return "Medium timeframes provide good balance of signal and noise"
        elif timeframe in ["8h", "12h", "1d"]:
            return "Longer timeframes may capture stronger trends but miss quick opportunities"
        else:
            return "Impact depends on strategy characteristics"

    # Helper methods
    def _calculate_normalized_score(self, strategy_name: str, side_by_side: Dict[str, Dict[str, Any]]) -> float:
        """Calculate normalized performance score (0-100)."""
        # Simplified scoring - would be more sophisticated in real implementation
        metrics = ["total_return", "win_rate", "profit_factor", "sharpe_ratio"]
        scores = []

        for metric in metrics:
            if metric in side_by_side and strategy_name in side_by_side[metric]:
                value = side_by_side[metric][strategy_name]
                # Simple normalization
                normalized = min(100, max(0, value * 10))
                scores.append(normalized)

        return round(statistics.mean(scores) if scores else 0, 2)

    def _calculate_volatility(self, results: Dict[str, Any]) -> float:
        """Calculate volatility from trade results."""
        trades = results.get("trades", [])
        returns = [trade.get("pnl", 0) for trade in trades]

        if len(returns) < 2:
            return 0.0

        return statistics.stdev(returns)

    def _calculate_risk_score(self, performance: Dict[str, Any], risk_analysis: Dict[str, Any]) -> float:
        """Calculate composite risk score."""
        max_drawdown = performance.get("max_drawdown", 0)
        var_95 = risk_analysis.get("var_95", 0)

        # Simple risk score calculation
        return (max_drawdown * 0.6) + (abs(var_95) * 0.4)

    def _categorize_performance_tiers(self, ranked_strategies: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, List[str]]:
        """Categorize strategies into performance tiers."""
        if not ranked_strategies:
            return {"tier1": [], "tier2": [], "tier3": []}

        total_strategies = len(ranked_strategies)
        tier1_count = max(1, total_strategies // 3)
        tier2_count = max(1, total_strategies // 3)

        return {
            "tier1": [strategy[0] for strategy in ranked_strategies[:tier1_count]],
            "tier2": [strategy[0] for strategy in ranked_strategies[tier1_count:tier1_count + tier2_count]],
            "tier3": [strategy[0] for strategy in ranked_strategies[tier1_count + tier2_count:]]
        }

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
            return "Low correlation - good diversification potential"
        elif abs_corr < 0.7:
            return "Moderate correlation - some diversification benefit"
        else:
            return "High correlation - limited diversification benefit"

    def _assess_diversification_potential(self, correlations: Dict[str, Dict[str, Any]]) -> str:
        """Assess overall diversification potential."""
        avg_correlation = statistics.mean([
            abs(corr_data["correlation"])
            for corr_data in correlations.values()
        ])

        if avg_correlation < 0.4:
            return "Excellent diversification potential"
        elif avg_correlation < 0.6:
            return "Good diversification potential"
        else:
            return "Limited diversification potential"

    def _calculate_portfolio_performance(self, strategies: Tuple[str], strategy_returns: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate portfolio performance for strategy combination."""
        # Simplified portfolio calculation (equal weights)
        weight = 1.0 / len(strategies)

        portfolio_return = sum(strategy_returns[strategy]["total_return"] * weight for strategy in strategies)
        portfolio_risk = sum(strategy_returns[strategy]["max_drawdown"] * weight for strategy in strategies)

        individual_avg_risk = statistics.mean([strategy_returns[strategy]["max_drawdown"] for strategy in strategies])
        diversification_ratio = individual_avg_risk / portfolio_risk if portfolio_risk > 0 else 1.0

        return {
            "return": round(portfolio_return, 2),
            "risk": round(portfolio_risk, 2),
            "diversification_ratio": round(diversification_ratio, 2)
        }

    def _generate_diversification_insights(self, portfolio_combinations: List[Dict[str, Any]]) -> List[str]:
        """Generate insights about diversification benefits."""
        insights = []

        if not portfolio_combinations:
            return ["No portfolio combinations available for analysis"]

        best_combo = portfolio_combinations[0]

        insights.append(f"Best portfolio combination: {', '.join(best_combo['strategies'])}")
        insights.append(f"Diversification ratio: {best_combo['diversification_ratio']}")

        if best_combo['diversification_ratio'] > 1.2:
            insights.append("Strong diversification benefits observed")
        elif best_combo['diversification_ratio'] > 1.0:
            insights.append("Moderate diversification benefits observed")
        else:
            insights.append("Limited diversification benefits - strategies may be too similar")

        return insights

    def _calculate_portfolio_expected_return(self, allocations: Dict[str, float], strategy_metrics: Dict[str, Dict[str, Any]]) -> float:
        """Calculate expected portfolio return."""
        return sum(
            (allocation / 100) * strategy_metrics[strategy]["return"]
            for strategy, allocation in allocations.items()
        )

    def _calculate_portfolio_expected_risk(self, allocations: Dict[str, float], strategy_metrics: Dict[str, Dict[str, Any]]) -> float:
        """Calculate expected portfolio risk."""
        return sum(
            (allocation / 100) * strategy_metrics[strategy]["risk"]
            for strategy, allocation in allocations.items()
        )

    def _compare_cross_type_performance(self, strategies: Dict, indicators: Dict) -> Dict[str, Any]:
        """Compare performance between strategies and indicators."""
        # This would implement strategy vs indicator comparison logic
        return {"placeholder": "Cross-type comparison not yet implemented"}

    def _analyze_complexity_vs_performance(self, strategies: Dict, indicators: Dict) -> Dict[str, Any]:
        """Analyze complexity vs performance trade-offs."""
        return {"placeholder": "Complexity analysis not yet implemented"}

    def _analyze_strategy_vs_indicator_insights(self, strategies: Dict, indicators: Dict) -> Dict[str, Any]:
        """Analyze insights from strategy vs indicator comparison."""
        return {"placeholder": "Strategy vs indicator insights not yet implemented"}

    def _suggest_indicator_to_strategy_migration(self, strategies: Dict, indicators: Dict) -> List[Dict[str, Any]]:
        """Suggest migration paths from indicators to strategies."""
        return [{"placeholder": "Migration suggestions not yet implemented"}]

    def _save_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """Save comparison report to file."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def export_comparison_results(
        self,
        report_data: Dict[str, Any],
        output_format: str = "json",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export comparison results in specified format.

        Args:
            report_data: Complete comparison report data
            output_format: Export format (json, csv, xlsx)
            output_path: Optional custom output path

        Returns:
            Path to the exported file
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_type = report_data.get("comparison_type", "comparison")
            output_path = Path(f"{comparison_type}_report_{timestamp}.{output_format}")

        if output_format.lower() == "json":
            self._save_report(report_data, output_path)
        elif output_format.lower() == "csv":
            logger.warning("CSV export for comparison reports not yet implemented")
        elif output_format.lower() == "xlsx":
            logger.warning("Excel export for comparison reports not yet implemented")
        else:
            raise ValueError(f"Unsupported export format: {output_format}")

        logger.info(f"Comparison results exported to: {output_path}")
        return output_path
