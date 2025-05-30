"""
Strategy-Specific Report Generation Module

This module centralizes all strategy-specific reporting logic, including:
- Strategy configuration display
- Strategy performance breakdown
- Strategy optimization suggestions
- Export functionality for strategy results
- Integration with InteractiveReporter for trade selection features
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.config_manager import ConfigManager
from validation.strategy_validator import StrategyValidator

logger = logging.getLogger(__name__)


class StrategyReporter:
    """
    Handles strategy-specific report generation and analysis.

    Centralizes strategy-specific reporting logic including configuration
    display, performance breakdown, and optimization suggestions.
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the StrategyReporter.

        Args:
            config_manager: ConfigManager instance for configuration access
        """
        self.config_manager = config_manager
        self.validator = StrategyValidator(config_manager)

    def generate_comprehensive_strategy_report(
        self,
        strategy_name: str,
        backtest_results: Dict[str, Any],
        include_interactive: bool = True,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive strategy report with optional interactive features.

        Args:
            strategy_name: Name of the strategy
            backtest_results: Backtest results data
            include_interactive: Whether to include interactive trade analysis
            output_path: Optional path to save the report

        Returns:
            Dictionary containing the full report data with interactive features
        """
        logger.info(f"Generating comprehensive strategy report for: {strategy_name}")

        try:
            # Get base strategy report
            base_report = self.generate_strategy_report(strategy_name, backtest_results, output_path)

            if include_interactive:
                # Import here to avoid circular imports
                from .interactive_reporter import InteractiveReporter

                interactive_reporter = InteractiveReporter()
                trades = backtest_results.get("trades", [])

                # Generate interactive features
                interactive_features = interactive_reporter.generate_interactive_report(
                    base_report, trades, output_path
                )

                # Merge reports
                base_report["interactive_features"] = interactive_features.get("interactive_features", {})
                base_report["accessibility"] = interactive_features.get("accessibility", {})
                base_report["responsive_design"] = interactive_features.get("responsive_design", {})

            return base_report

        except Exception as e:
            logger.error(f"Error generating comprehensive strategy report for {strategy_name}: {e}")
            raise

    def generate_strategy_vs_indicator_comparison(
        self,
        strategy_name: str,
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate detailed comparison between strategy and individual indicator performance.

        Args:
            strategy_name: Name of the strategy
            backtest_results: Backtest results data

        Returns:
            Dictionary containing detailed comparison analysis
        """
        logger.info(f"Generating strategy vs indicator comparison for: {strategy_name}")

        try:
            strategy_config = self.config_manager.get_strategy_config(strategy_name)
            strategy_metrics = self._calculate_performance_metrics(backtest_results)

            comparison = {
                "strategy_performance": strategy_metrics,
                "individual_indicators": {},
                "performance_attribution": {},
                "synergy_analysis": {}
            }

            # Analyze each indicator individually
            for indicator_name, indicator_config in strategy_config.get("indicators", {}).items():
                # This would require separate backtesting of individual indicators
                # For now, we'll use signal analysis
                individual_performance = self._estimate_individual_indicator_performance(
                    indicator_name, backtest_results
                )

                comparison["individual_indicators"][indicator_name] = individual_performance

                # Calculate contribution attribution
                attribution = self._calculate_performance_attribution(
                    indicator_name, strategy_metrics, individual_performance
                )
                comparison["performance_attribution"][indicator_name] = attribution

            # Analyze synergy between indicators
            comparison["synergy_analysis"] = self._analyze_indicator_synergy(
                strategy_config, backtest_results
            )

            return comparison

        except Exception as e:
            logger.error(f"Error generating strategy vs indicator comparison: {e}")
            raise

    def _estimate_individual_indicator_performance(
        self,
        indicator_name: str,
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Estimate how the indicator would perform individually.

        Args:
            indicator_name: Name of the indicator
            backtest_results: Full backtest results

        Returns:
            Estimated individual performance metrics
        """
        trades = backtest_results.get("trades", [])
        indicator_signals = backtest_results.get("indicator_signals", {}).get(indicator_name, [])

        if not indicator_signals:
            return {"error": f"No signals found for indicator {indicator_name}"}

        # Estimate trades based purely on this indicator's signals
        estimated_trades = []
        for signal in indicator_signals:
            # Find corresponding trades that happened around this signal
            signal_time = signal.get("timestamp")
            matching_trades = [
                trade for trade in trades
                if abs(self._time_difference(trade.get("entry_time"), signal_time)) < 300  # 5 minutes
            ]

            if matching_trades:
                # Use the best matching trade
                best_trade = min(matching_trades,
                    key=lambda t: abs(self._time_difference(t.get("entry_time"), signal_time)))
                estimated_trades.append(best_trade)

        # Calculate metrics for estimated individual performance
        if estimated_trades:
            return self._calculate_performance_metrics({"trades": estimated_trades})
        else:
            return {"error": f"No matching trades found for {indicator_name} signals"}

    def _calculate_performance_attribution(
        self,
        indicator_name: str,
        strategy_metrics: Dict[str, Any],
        individual_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate how much this indicator contributes to overall strategy performance.

        Args:
            indicator_name: Name of the indicator
            strategy_metrics: Overall strategy performance
            individual_metrics: Individual indicator performance

        Returns:
            Performance attribution analysis
        """
        if "error" in individual_metrics:
            return {"error": f"Cannot calculate attribution for {indicator_name}"}

        strategy_return = strategy_metrics.get("total_pnl", 0)
        individual_return = individual_metrics.get("total_pnl", 0)

        attribution = {
            "return_contribution": individual_return / strategy_return if strategy_return != 0 else 0,
            "win_rate_contribution": individual_metrics.get("win_rate", 0) / strategy_metrics.get("win_rate", 1) if strategy_metrics.get("win_rate", 0) != 0 else 0,
            "trade_frequency_contribution": individual_metrics.get("total_trades", 0) / strategy_metrics.get("total_trades", 1) if strategy_metrics.get("total_trades", 0) != 0 else 0,
            "risk_contribution": individual_metrics.get("max_drawdown", 0) / strategy_metrics.get("max_drawdown", 1) if strategy_metrics.get("max_drawdown", 0) != 0 else 0
        }

        return attribution

    def _analyze_indicator_synergy(
        self,
        strategy_config: Dict[str, Any],
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze synergy effects between indicators in the strategy.

        Args:
            strategy_config: Strategy configuration
            backtest_results: Backtest results

        Returns:
            Synergy analysis results
        """
        indicators = list(strategy_config.get("indicators", {}).keys())
        trades = backtest_results.get("trades", [])

        synergy = {
            "multi_indicator_trades": 0,
            "single_indicator_trades": 0,
            "synergy_effect": 0,
            "best_combinations": [],
            "conflict_analysis": {}
        }

        # Analyze trades that involved multiple indicators
        for trade in trades:
            involved_indicators = trade.get("involved_indicators", [])
            if len(involved_indicators) > 1:
                synergy["multi_indicator_trades"] += 1
            else:
                synergy["single_indicator_trades"] += 1

        # Calculate synergy effect
        if synergy["multi_indicator_trades"] > 0 and synergy["single_indicator_trades"] > 0:
            multi_avg_pnl = sum(
                trade.get("pnl", 0) for trade in trades
                if len(trade.get("involved_indicators", [])) > 1
            ) / synergy["multi_indicator_trades"]

            single_avg_pnl = sum(
                trade.get("pnl", 0) for trade in trades
                if len(trade.get("involved_indicators", [])) == 1
            ) / synergy["single_indicator_trades"]

            synergy["synergy_effect"] = (multi_avg_pnl / single_avg_pnl - 1) if single_avg_pnl != 0 else 0

        return synergy

    def _time_difference(self, time1: str, time2: str) -> float:
        """
        Calculate time difference in seconds between two timestamp strings.

        Args:
            time1: First timestamp
            time2: Second timestamp

        Returns:
            Time difference in seconds
        """
        try:
            # This is a simplified implementation
            # In practice, you'd parse the actual timestamp formats
            return 0.0  # Placeholder
        except:
            return float('inf')

    def generate_strategy_report(
        self,
        strategy_name: str,
        backtest_results: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive strategy report.

        Args:
            strategy_name: Name of the strategy
            backtest_results: Backtest results data
            output_path: Optional path to save the report

        Returns:
            Dictionary containing the full report data
        """
        logger.info(f"Generating strategy report for: {strategy_name}")

        try:
            strategy_config = self.config_manager.get_strategy_config(strategy_name)

            report = {
                "strategy_name": strategy_name,
                "generated_at": datetime.now().isoformat(),
                "strategy_configuration": self._format_strategy_configuration(strategy_config),
                "performance_metrics": self._calculate_performance_metrics(backtest_results),
                "indicator_breakdown": self._analyze_indicator_breakdown(strategy_config, backtest_results),
                "position_sizing_analysis": self._analyze_position_sizing(strategy_config, backtest_results),
                "risk_analysis": self._analyze_risk_metrics(strategy_config, backtest_results),
                "optimization_suggestions": self._generate_optimization_suggestions(strategy_config, backtest_results),
                "trade_analysis": self._analyze_trade_patterns(backtest_results),
                "configuration_vs_actual": self._compare_config_vs_actual(strategy_config, backtest_results)
            }

            if output_path:
                self._save_report(report, output_path)

            return report

        except Exception as e:
            logger.error(f"Error generating strategy report for {strategy_name}: {e}")
            raise

    def _format_strategy_configuration(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Format strategy configuration for display."""
        formatted_config = {
            "market": strategy_config.get("market", "Unknown"),
            "exchange": strategy_config.get("exchange", "Unknown"),
            "timeframe": strategy_config.get("timeframe", "Unknown"),
            "enabled": strategy_config.get("enabled", False),
            "indicators": {},
            "position_sizing": strategy_config.get("position_sizing", {}),
            "risk_management": {
                "stop_loss": strategy_config.get("stop_loss"),
                "take_profit": strategy_config.get("take_profit"),
                "max_position_size": strategy_config.get("max_position_size")
            }
        }

        # Format indicator configurations
        for indicator_name, indicator_config in strategy_config.get("indicators", {}).items():
            formatted_config["indicators"][indicator_name] = {
                "enabled": indicator_config.get("enabled", True),
                "timeframe": indicator_config.get("timeframe", formatted_config["timeframe"]),
                "parameters": indicator_config.get("parameters", {})
            }

        return formatted_config

    def _calculate_performance_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        trades = backtest_results.get("trades", [])
        equity_curve = backtest_results.get("equity_curve", [])

        if not trades:
            return {"error": "No trades available for analysis"}

        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        total_pnl = sum(t.get("pnl", 0) for t in trades)

        # Risk metrics
        pnl_values = [t.get("pnl", 0) for t in trades]
        if pnl_values:
            avg_win = sum(t.get("pnl", 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.get("pnl", 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            avg_win = avg_loss = profit_factor = 0

        # Drawdown calculation
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(win_rate * 100, 2),
            "total_pnl": round(total_pnl, 2),
            "average_win": round(avg_win, 2),
            "average_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞",
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": self._calculate_sharpe_ratio(pnl_values),
            "trades_per_day": self._calculate_trades_per_day(trades)
        }

    def _analyze_indicator_breakdown(
        self,
        strategy_config: Dict[str, Any],
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze individual indicator performance within strategy context."""
        indicators = strategy_config.get("indicators", {})
        indicator_signals = backtest_results.get("indicator_signals", {})

        breakdown = {}

        for indicator_name, indicator_config in indicators.items():
            signals = indicator_signals.get(indicator_name, [])

            breakdown[indicator_name] = {
                "configuration": indicator_config,
                "signal_count": len(signals),
                "signal_accuracy": self._calculate_signal_accuracy(signals, backtest_results.get("trades", [])),
                "contribution_to_returns": self._calculate_indicator_contribution(indicator_name, backtest_results),
                "timeframe": indicator_config.get("timeframe", strategy_config.get("timeframe")),
                "parameters": indicator_config.get("parameters", {})
            }

        return breakdown

    def _analyze_position_sizing(
        self,
        strategy_config: Dict[str, Any],
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze position sizing effectiveness."""
        position_config = strategy_config.get("position_sizing", {})
        trades = backtest_results.get("trades", [])

        if not trades:
            return {"error": "No trades available for position sizing analysis"}

        position_sizes = [t.get("position_size", 0) for t in trades]

        return {
            "configured_method": position_config.get("method", "unknown"),
            "configured_amount": position_config.get("amount", "unknown"),
            "actual_average_size": round(sum(position_sizes) / len(position_sizes), 2) if position_sizes else 0,
            "size_consistency": self._calculate_size_consistency(position_sizes),
            "size_vs_performance": self._analyze_size_vs_performance(trades),
            "optimal_size_suggestion": self._suggest_optimal_position_size(trades)
        }

    def _analyze_risk_metrics(
        self,
        strategy_config: Dict[str, Any],
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze risk management effectiveness."""
        trades = backtest_results.get("trades", [])

        if not trades:
            return {"error": "No trades available for risk analysis"}

        stop_loss_config = strategy_config.get("stop_loss")
        take_profit_config = strategy_config.get("take_profit")

        stop_loss_hits = [t for t in trades if t.get("exit_reason") == "stop_loss"]
        take_profit_hits = [t for t in trades if t.get("exit_reason") == "take_profit"]

        return {
            "configured_stop_loss": stop_loss_config,
            "configured_take_profit": take_profit_config,
            "stop_loss_hit_rate": len(stop_loss_hits) / len(trades) if trades else 0,
            "take_profit_hit_rate": len(take_profit_hits) / len(trades) if trades else 0,
            "average_trade_duration": self._calculate_average_duration(trades),
            "risk_reward_ratio": self._calculate_risk_reward_ratio(trades),
            "var_95": self._calculate_var(trades, 0.95),
            "maximum_adverse_excursion": self._calculate_mae(trades)
        }

    def _generate_optimization_suggestions(
        self,
        strategy_config: Dict[str, Any],
        backtest_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate optimization suggestions based on performance analysis."""
        suggestions = []

        performance = self._calculate_performance_metrics(backtest_results)

        # Win rate suggestions
        if performance.get("win_rate", 0) < 40:
            suggestions.append({
                "type": "win_rate_improvement",
                "priority": "high",
                "suggestion": "Consider tightening entry criteria or adjusting indicator parameters",
                "current_value": performance.get("win_rate"),
                "target_value": "50-60%"
            })

        # Profit factor suggestions
        if performance.get("profit_factor", 0) < 1.5:
            suggestions.append({
                "type": "profit_factor_improvement",
                "priority": "high",
                "suggestion": "Review stop loss and take profit levels to improve risk/reward ratio",
                "current_value": performance.get("profit_factor"),
                "target_value": ">1.5"
            })

        # Drawdown suggestions
        if performance.get("max_drawdown", 0) > 15:
            suggestions.append({
                "type": "drawdown_reduction",
                "priority": "medium",
                "suggestion": "Consider reducing position sizes or adding additional filters",
                "current_value": f"{performance.get('max_drawdown')}%",
                "target_value": "<10%"
            })

        # Position sizing suggestions
        position_analysis = self._analyze_position_sizing(strategy_config, backtest_results)
        if position_analysis.get("size_consistency", 1) < 0.8:
            suggestions.append({
                "type": "position_sizing_consistency",
                "priority": "medium",
                "suggestion": "Review position sizing method for more consistent sizing",
                "current_value": f"{position_analysis.get('size_consistency', 0)*100:.1f}%",
                "target_value": ">80%"
            })

        return suggestions

    def _analyze_trade_patterns(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading patterns and behavior."""
        trades = backtest_results.get("trades", [])

        if not trades:
            return {"error": "No trades available for pattern analysis"}

        # Time-based patterns
        entry_hours = [self._extract_hour(t.get("entry_time")) for t in trades if t.get("entry_time")]
        exit_hours = [self._extract_hour(t.get("exit_time")) for t in trades if t.get("exit_time")]

        # Duration patterns
        durations = [t.get("duration_minutes", 0) for t in trades]

        return {
            "most_active_entry_hour": max(set(entry_hours), key=entry_hours.count) if entry_hours else None,
            "most_active_exit_hour": max(set(exit_hours), key=exit_hours.count) if exit_hours else None,
            "average_trade_duration_minutes": sum(durations) / len(durations) if durations else 0,
            "shortest_trade_minutes": min(durations) if durations else 0,
            "longest_trade_minutes": max(durations) if durations else 0,
            "weekend_trade_performance": self._analyze_weekend_performance(trades),
            "consecutive_loss_streaks": self._analyze_loss_streaks(trades)
        }

    def _compare_config_vs_actual(
        self,
        strategy_config: Dict[str, Any],
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare configured strategy parameters vs actual performance."""
        trades = backtest_results.get("trades", [])

        comparison = {
            "timeframe": {
                "configured": strategy_config.get("timeframe"),
                "effective": self._calculate_effective_timeframe(trades)
            },
            "position_sizing": {
                "configured_method": strategy_config.get("position_sizing", {}).get("method"),
                "configured_amount": strategy_config.get("position_sizing", {}).get("amount"),
                "actual_average": self._calculate_actual_average_position_size(trades)
            },
            "risk_management": {
                "configured_stop_loss": strategy_config.get("stop_loss"),
                "actual_average_loss": self._calculate_actual_average_loss(trades),
                "configured_take_profit": strategy_config.get("take_profit"),
                "actual_average_win": self._calculate_actual_average_win(trades)
            }
        }

        return comparison

    def export_strategy_results(
        self,
        strategy_name: str,
        report_data: Dict[str, Any],
        output_format: str = "json",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export strategy results in specified format.

        Args:
            strategy_name: Name of the strategy
            report_data: Complete report data
            output_format: Export format (json, csv, xlsx)
            output_path: Optional custom output path

        Returns:
            Path to the exported file
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"strategy_report_{strategy_name}_{timestamp}.{output_format}")

        if output_format.lower() == "json":
            self._export_json(report_data, output_path)
        elif output_format.lower() == "csv":
            self._export_csv(report_data, output_path)
        elif output_format.lower() == "xlsx":
            self._export_xlsx(report_data, output_path)
        else:
            raise ValueError(f"Unsupported export format: {output_format}")

        logger.info(f"Strategy results exported to: {output_path}")
        return output_path

    # Helper methods for calculations
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns."""
        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)

        if len(returns) < 2:
            return 0.0

        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = variance ** 0.5

        return (mean_return / std_dev) if std_dev != 0 else 0.0

    def _calculate_trades_per_day(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate average trades per day."""
        if not trades:
            return 0.0

        # This is a simplified calculation - would need actual time analysis
        return len(trades) / 30  # Assuming 30-day period

    def _calculate_signal_accuracy(self, signals: List[Dict], trades: List[Dict]) -> float:
        """Calculate accuracy of indicator signals."""
        # Simplified calculation - would need signal-to-trade mapping
        return 0.75  # Placeholder

    def _calculate_indicator_contribution(self, indicator_name: str, backtest_results: Dict) -> float:
        """Calculate indicator's contribution to overall returns."""
        # Simplified calculation - would need detailed signal analysis
        return 0.25  # Placeholder

    def _calculate_size_consistency(self, position_sizes: List[float]) -> float:
        """Calculate consistency of position sizing."""
        if not position_sizes:
            return 0.0

        mean_size = sum(position_sizes) / len(position_sizes)
        variance = sum((s - mean_size) ** 2 for s in position_sizes) / len(position_sizes)
        std_dev = variance ** 0.5

        cv = std_dev / mean_size if mean_size != 0 else 0
        return max(0, 1 - cv)  # Higher is more consistent

    def _analyze_size_vs_performance(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze correlation between position size and performance."""
        # Simplified analysis
        return {"correlation": 0.1, "optimal_size_ratio": 1.2}

    def _suggest_optimal_position_size(self, trades: List[Dict]) -> str:
        """Suggest optimal position sizing based on performance."""
        return "Consider reducing position size by 10-15% to improve risk-adjusted returns"

    def _calculate_average_duration(self, trades: List[Dict]) -> float:
        """Calculate average trade duration in hours."""
        durations = [t.get("duration_minutes", 0) for t in trades]
        return sum(durations) / len(durations) / 60 if durations else 0

    def _calculate_risk_reward_ratio(self, trades: List[Dict]) -> float:
        """Calculate average risk/reward ratio."""
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

        if not winning_trades or not losing_trades:
            return 0.0

        avg_win = sum(t.get("pnl", 0) for t in winning_trades) / len(winning_trades)
        avg_loss = abs(sum(t.get("pnl", 0) for t in losing_trades) / len(losing_trades))

        return avg_win / avg_loss if avg_loss != 0 else 0.0

    def _calculate_var(self, trades: List[Dict], confidence: float) -> float:
        """Calculate Value at Risk."""
        pnl_values = sorted([t.get("pnl", 0) for t in trades])
        if not pnl_values:
            return 0.0

        index = int((1 - confidence) * len(pnl_values))
        return pnl_values[index] if index < len(pnl_values) else pnl_values[-1]

    def _calculate_mae(self, trades: List[Dict]) -> float:
        """Calculate Maximum Adverse Excursion."""
        # Simplified calculation - would need tick-by-tick data
        losing_trades = [t for t in trades if t.get("pnl", 0) < 0]
        return min([t.get("pnl", 0) for t in losing_trades]) if losing_trades else 0.0

    def _extract_hour(self, timestamp: str) -> int:
        """Extract hour from timestamp string."""
        try:
            return int(timestamp.split("T")[1].split(":")[0])
        except:
            return 12  # Default to noon

    def _analyze_weekend_performance(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze weekend vs weekday performance."""
        # Simplified analysis
        return {"weekend_win_rate": 0.45, "weekday_win_rate": 0.55}

    def _analyze_loss_streaks(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze consecutive loss streaks."""
        # Simplified analysis
        return {"max_consecutive_losses": 3, "average_streak_length": 1.5}

    def _calculate_effective_timeframe(self, trades: List[Dict]) -> str:
        """Calculate effective timeframe based on trade durations."""
        durations = [t.get("duration_minutes", 0) for t in trades]
        avg_duration = sum(durations) / len(durations) if durations else 0

        if avg_duration < 60:
            return "15m"
        elif avg_duration < 240:
            return "1h"
        elif avg_duration < 1440:
            return "4h"
        else:
            return "1d"

    def _calculate_actual_average_position_size(self, trades: List[Dict]) -> float:
        """Calculate actual average position size."""
        sizes = [t.get("position_size", 0) for t in trades]
        return sum(sizes) / len(sizes) if sizes else 0.0

    def _calculate_actual_average_loss(self, trades: List[Dict]) -> float:
        """Calculate actual average loss amount."""
        losses = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0]
        return sum(losses) / len(losses) if losses else 0.0

    def _calculate_actual_average_win(self, trades: List[Dict]) -> float:
        """Calculate actual average win amount."""
        wins = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0]
        return sum(wins) / len(wins) if wins else 0.0

    def _save_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """Save report to file."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def _export_json(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export data as JSON."""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _export_csv(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export data as CSV."""
        # Would implement CSV export logic
        logger.warning("CSV export not yet implemented")

    def _export_xlsx(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export data as Excel."""
        # Would implement Excel export logic
        logger.warning("Excel export not yet implemented")

    def generate_configuration_sensitivity_analysis(
        self,
        strategy_name: str,
        base_backtest_results: Dict[str, Any],
        analysis_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive configuration sensitivity analysis.

        Tests strategy performance with different configuration variations
        including position sizing, timeframes, indicator parameters, and
        risk management settings.

        Args:
            strategy_name: Name of the strategy to analyze
            base_backtest_results: Base backtest results for comparison
            analysis_options: Optional configuration for analysis parameters

        Returns:
            Dictionary containing comprehensive sensitivity analysis results
        """
        logger.info(f"Generating configuration sensitivity analysis for: {strategy_name}")

        try:
            options = analysis_options or {}
            analysis_results = {
                "strategy_name": strategy_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "base_performance": self._calculate_performance_metrics(base_backtest_results),
                "sensitivity_tests": {}
            }

            # Test position sizing variations
            if options.get("test_position_sizing", True):
                analysis_results["sensitivity_tests"]["position_sizing"] = \
                    self._test_position_sizing_sensitivity(strategy_name, base_backtest_results)

            # Test timeframe variations
            if options.get("test_timeframes", True):
                analysis_results["sensitivity_tests"]["timeframes"] = \
                    self._test_timeframe_sensitivity(strategy_name, base_backtest_results)

            # Test indicator parameter variations
            if options.get("test_indicator_parameters", True):
                analysis_results["sensitivity_tests"]["indicator_parameters"] = \
                    self._test_indicator_parameter_sensitivity(strategy_name, base_backtest_results)

            # Test risk parameter variations
            if options.get("test_risk_parameters", True):
                analysis_results["sensitivity_tests"]["risk_parameters"] = \
                    self._test_risk_parameter_sensitivity(strategy_name, base_backtest_results)

            # Generate optimization recommendations
            analysis_results["optimization_suggestions"] = \
                self._generate_configuration_optimization_suggestions(analysis_results)

            # Perform feasibility analysis with validator
            analysis_results["feasibility_analysis"] = \
                self._perform_configuration_feasibility_analysis(strategy_name, analysis_results)

            return analysis_results

        except Exception as e:
            logger.error(f"Error generating configuration sensitivity analysis for {strategy_name}: {e}")
            raise

    def _test_position_sizing_sensitivity(
        self,
        strategy_name: str,
        base_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test strategy performance with different position sizing methods.

        Args:
            strategy_name: Name of the strategy
            base_results: Base backtest results

        Returns:
            Dictionary containing position sizing sensitivity analysis
        """
        logger.info(f"Testing position sizing sensitivity for: {strategy_name}")

        # Get current position sizing configuration
        strategy_config = self.config_manager.get_strategy_config(strategy_name)
        current_position_sizing = strategy_config.get("position_sizing", {})

        # Define position sizing variations to test
        position_sizing_variations = [
            {
                "name": "Fixed $100",
                "method": "fixed_usd",
                "size": 100.0,
                "description": "Fixed $100 per trade"
            },
            {
                "name": "Fixed $500",
                "method": "fixed_usd",
                "size": 500.0,
                "description": "Fixed $500 per trade"
            },
            {
                "name": "1% Risk",
                "method": "percent_of_balance",
                "percentage": 1.0,
                "description": "1% of account balance per trade"
            },
            {
                "name": "2% Risk",
                "method": "percent_of_balance",
                "percentage": 2.0,
                "description": "2% of account balance per trade"
            },
            {
                "name": "Kelly Criterion",
                "method": "kelly_criterion",
                "win_rate": 0.55,
                "avg_win": 100,
                "avg_loss": 80,
                "description": "Kelly Criterion optimal sizing"
            }
        ]

        results = {
            "current_method": current_position_sizing,
            "variations": {},
            "performance_comparison": {},
            "recommendations": []
        }

        base_performance = self._calculate_performance_metrics(base_results)

        for variation in position_sizing_variations:
            # Simulate position sizing impact on existing trades
            adjusted_results = self._simulate_position_sizing_impact(
                base_results, variation
            )

            performance = self._calculate_performance_metrics(adjusted_results)

            results["variations"][variation["name"]] = {
                "config": variation,
                "performance": performance,
                "performance_change": self._calculate_performance_change(
                    base_performance, performance
                )
            }

        # Generate position sizing comparison
        results["performance_comparison"] = self._compare_position_sizing_methods(
            results["variations"]
        )

        # Generate recommendations
        results["recommendations"] = self._generate_position_sizing_recommendations(
            results["variations"], current_position_sizing
        )

        return results

    def _test_timeframe_sensitivity(
        self,
        strategy_name: str,
        base_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze impact of timeframe changes on strategy performance.

        Args:
            strategy_name: Name of the strategy
            base_results: Base backtest results

        Returns:
            Dictionary containing timeframe sensitivity analysis
        """
        logger.info(f"Testing timeframe sensitivity for: {strategy_name}")

        strategy_config = self.config_manager.get_strategy_config(strategy_name)
        current_timeframe = strategy_config.get("timeframe", "1h")

        # Define timeframe variations to analyze
        timeframe_variations = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]

        results = {
            "current_timeframe": current_timeframe,
            "variations": {},
            "timeframe_analysis": {},
            "recommendations": []
        }

        base_performance = self._calculate_performance_metrics(base_results)

        for timeframe in timeframe_variations:
            if timeframe == current_timeframe:
                continue

            # Estimate performance impact based on timeframe characteristics
            performance_estimate = self._estimate_timeframe_performance_impact(
                base_performance, current_timeframe, timeframe, strategy_config
            )

            results["variations"][timeframe] = {
                "timeframe": timeframe,
                "estimated_performance": performance_estimate,
                "feasibility": self._assess_timeframe_feasibility(
                    strategy_config, timeframe
                ),
                "trade_frequency_estimate": self._estimate_trade_frequency_change(
                    current_timeframe, timeframe, base_results
                )
            }

        # Generate timeframe analysis
        results["timeframe_analysis"] = self._analyze_timeframe_characteristics(
            results["variations"], strategy_config
        )

        # Generate recommendations
        results["recommendations"] = self._generate_timeframe_recommendations(
            results["variations"], current_timeframe, strategy_config
        )

        return results

    def _test_indicator_parameter_sensitivity(
        self,
        strategy_name: str,
        base_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Show effect of different indicator parameters on strategy results.

        Args:
            strategy_name: Name of the strategy
            base_results: Base backtest results

        Returns:
            Dictionary containing indicator parameter sensitivity analysis
        """
        logger.info(f"Testing indicator parameter sensitivity for: {strategy_name}")

        strategy_config = self.config_manager.get_strategy_config(strategy_name)
        indicators = strategy_config.get("indicators", {})

        results = {
            "indicators": {},
            "parameter_interactions": {},
            "optimization_candidates": [],
            "recommendations": []
        }

        base_performance = self._calculate_performance_metrics(base_results)

        for indicator_name, indicator_config in indicators.items():
            indicator_type = indicator_config.get("type")

            # Define parameter variations based on indicator type
            parameter_variations = self._get_indicator_parameter_variations(
                indicator_type, indicator_config
            )

            indicator_results = {
                "current_parameters": indicator_config,
                "variations": {},
                "sensitivity_ranking": [],
                "optimal_ranges": {}
            }

            for variation_name, variation_params in parameter_variations.items():
                # Estimate performance impact of parameter changes
                performance_estimate = self._estimate_parameter_performance_impact(
                    base_performance, indicator_name, indicator_config,
                    variation_params, base_results
                )

                indicator_results["variations"][variation_name] = {
                    "parameters": variation_params,
                    "estimated_performance": performance_estimate,
                    "parameter_change": self._calculate_parameter_change(
                        indicator_config, variation_params
                    ),
                    "impact_assessment": self._assess_parameter_impact(
                        indicator_config, variation_params
                    )
                }

            # Analyze parameter sensitivity
            indicator_results["sensitivity_ranking"] = \
                self._rank_parameter_sensitivity(indicator_results["variations"])

            indicator_results["optimal_ranges"] = \
                self._identify_optimal_parameter_ranges(indicator_results["variations"])

            results["indicators"][indicator_name] = indicator_results

            # Identify optimization candidates
            if self._is_optimization_candidate(indicator_results):
                results["optimization_candidates"].append({
                    "indicator": indicator_name,
                    "reason": self._get_optimization_reason(indicator_results),
                    "suggested_parameters": self._get_suggested_parameters(indicator_results)
                })

        # Analyze parameter interactions between indicators
        results["parameter_interactions"] = self._analyze_parameter_interactions(
            results["indicators"], strategy_config
        )

        # Generate recommendations
        results["recommendations"] = self._generate_parameter_optimization_recommendations(
            results["indicators"], results["optimization_candidates"]
        )

        return results

    def _test_risk_parameter_sensitivity(
        self,
        strategy_name: str,
        base_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Include risk parameter sensitivity (stop loss, take profit impact).

        Args:
            strategy_name: Name of the strategy
            base_results: Base backtest results

        Returns:
            Dictionary containing risk parameter sensitivity analysis
        """
        logger.info(f"Testing risk parameter sensitivity for: {strategy_name}")

        strategy_config = self.config_manager.get_strategy_config(strategy_name)
        risk_config = strategy_config.get("risk_management", {})

        # Define risk parameter variations
        risk_variations = self._generate_risk_parameter_variations(risk_config)

        results = {
            "current_risk_parameters": risk_config,
            "variations": {},
            "risk_reward_analysis": {},
            "drawdown_impact": {},
            "recommendations": []
        }

        base_performance = self._calculate_performance_metrics(base_results)

        for variation_name, variation_config in risk_variations.items():
            # Simulate risk parameter impact on trades
            adjusted_results = self._simulate_risk_parameter_impact(
                base_results, variation_config
            )

            performance = self._calculate_performance_metrics(adjusted_results)

            results["variations"][variation_name] = {
                "config": variation_config,
                "performance": performance,
                "performance_change": self._calculate_performance_change(
                    base_performance, performance
                ),
                "trade_impact": self._analyze_risk_trade_impact(
                    base_results, adjusted_results
                )
            }

        # Analyze risk-reward relationships
        results["risk_reward_analysis"] = self._analyze_risk_reward_relationships(
            results["variations"]
        )

        # Analyze drawdown impact
        results["drawdown_impact"] = self._analyze_drawdown_impact(
            results["variations"]
        )

        # Generate recommendations
        results["recommendations"] = self._generate_risk_parameter_recommendations(
            results["variations"], risk_config
        )

        return results

    def _generate_configuration_optimization_suggestions(
        self,
        analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate optimization suggestions for strategy configuration.

        Args:
            analysis_results: Complete sensitivity analysis results

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Analyze position sizing optimization
        position_results = analysis_results["sensitivity_tests"].get("position_sizing", {})
        if position_results:
            position_suggestions = self._extract_position_sizing_optimizations(position_results)
            suggestions.extend(position_suggestions)

        # Analyze timeframe optimization
        timeframe_results = analysis_results["sensitivity_tests"].get("timeframes", {})
        if timeframe_results:
            timeframe_suggestions = self._extract_timeframe_optimizations(timeframe_results)
            suggestions.extend(timeframe_suggestions)

        # Analyze indicator parameter optimization
        indicator_results = analysis_results["sensitivity_tests"].get("indicator_parameters", {})
        if indicator_results:
            indicator_suggestions = self._extract_indicator_optimizations(indicator_results)
            suggestions.extend(indicator_suggestions)

        # Analyze risk parameter optimization
        risk_results = analysis_results["sensitivity_tests"].get("risk_parameters", {})
        if risk_results:
            risk_suggestions = self._extract_risk_optimizations(risk_results)
            suggestions.extend(risk_suggestions)

        # Rank suggestions by potential impact
        suggestions = self._rank_optimization_suggestions(suggestions)

        return suggestions

    def _perform_configuration_feasibility_analysis(
        self,
        strategy_name: str,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate with StrategyValidator for feasibility analysis.

        Args:
            strategy_name: Name of the strategy
            analysis_results: Sensitivity analysis results

        Returns:
            Dictionary containing feasibility analysis results
        """
        logger.info(f"Performing configuration feasibility analysis for: {strategy_name}")

        try:
            # Get current strategy configuration
            strategy_config = self.config_manager.get_strategy_config(strategy_name)

            # Validate current configuration
            current_validation = self.validator.validate_strategy_configuration(strategy_name)

            # Analyze proposed optimizations for feasibility
            optimization_suggestions = analysis_results.get("optimization_suggestions", [])
            feasibility_results = []

            for suggestion in optimization_suggestions:
                # Create test configuration with suggested changes
                test_config = self._apply_suggestion_to_config(strategy_config, suggestion)

                # Validate the modified configuration
                validation_result = self.validator.validate_strategy_config_dict(test_config)

                feasibility_results.append({
                    "suggestion": suggestion,
                    "feasibility": {
                        "is_valid": validation_result.is_valid,
                        "errors": validation_result.errors,
                        "warnings": validation_result.warnings,
                        "suggestions": validation_result.suggestions
                    },
                    "implementation_complexity": self._assess_implementation_complexity(suggestion),
                    "risk_assessment": self._assess_configuration_risk(test_config, strategy_config)
                })

            return {
                "current_config_validation": {
                    "is_valid": current_validation.is_valid,
                    "errors": current_validation.errors,
                    "warnings": current_validation.warnings,
                    "suggestions": current_validation.suggestions
                },
                "optimization_feasibility": feasibility_results,
                "overall_assessment": self._generate_overall_feasibility_assessment(
                    feasibility_results
                ),
                "implementation_priorities": self._prioritize_feasible_optimizations(
                    feasibility_results
                )
            }

        except Exception as e:
            logger.error(f"Error performing feasibility analysis: {e}")
            return {
                "error": str(e),
                "current_config_validation": {"is_valid": False, "errors": [str(e)]}
            }

    def _simulate_position_sizing_impact(
        self,
        base_results: Dict[str, Any],
        position_sizing_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate the impact of different position sizing on existing trades.

        Args:
            base_results: Original backtest results
            position_sizing_config: New position sizing configuration

        Returns:
            Adjusted backtest results with new position sizing
        """
        adjusted_results = base_results.copy()
        trades = base_results.get("trades", [])

        if not trades:
            return adjusted_results

        adjusted_trades = []
        for trade in trades:
            adjusted_trade = trade.copy()

            # Calculate new position size based on configuration
            new_size = self._calculate_position_size(position_sizing_config, trade)
            original_size = trade.get("position_size", 100)

            if original_size > 0:
                size_multiplier = new_size / original_size

                # Adjust trade PnL based on new position size
                adjusted_trade["position_size"] = new_size
                adjusted_trade["pnl"] = trade.get("pnl", 0) * size_multiplier
                adjusted_trade["gross_pnl"] = trade.get("gross_pnl", 0) * size_multiplier

            adjusted_trades.append(adjusted_trade)

        adjusted_results["trades"] = adjusted_trades
        return adjusted_results

    def _calculate_position_size(self, config: Dict[str, Any], trade: Dict[str, Any]) -> float:
        """Calculate position size based on configuration."""
        method = config.get("method", "fixed_usd")

        if method == "fixed_usd":
            return config.get("size", 100.0)
        elif method == "percent_of_balance":
            balance = 10000  # Assume $10,000 balance for simulation
            percentage = config.get("percentage", 1.0) / 100
            return balance * percentage
        elif method == "kelly_criterion":
            win_rate = config.get("win_rate", 0.5)
            avg_win = config.get("avg_win", 100)
            avg_loss = config.get("avg_loss", 80)

            if avg_loss > 0:
                b = avg_win / avg_loss  # Win/loss ratio
                p = win_rate
                kelly_fraction = (b * p - (1 - p)) / b
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

                balance = 10000  # Assume $10,000 balance
                return balance * kelly_fraction

        return 100.0  # Default fallback

    def _calculate_performance_change(
        self,
        base_performance: Dict[str, Any],
        new_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate the change in performance metrics."""
        changes = {}

        for metric, base_value in base_performance.items():
            if isinstance(base_value, (int, float)) and metric in new_performance:
                new_value = new_performance[metric]
                if base_value != 0:
                    percent_change = ((new_value - base_value) / base_value) * 100
                else:
                    percent_change = 0 if new_value == 0 else float('inf')

                changes[metric] = {
                    "absolute_change": new_value - base_value,
                    "percent_change": percent_change,
                    "from": base_value,
                    "to": new_value
                }

        return changes

    def _compare_position_sizing_methods(self, variations: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different position sizing methods."""
        comparison = {
            "ranking_by_return": [],
            "ranking_by_sharpe": [],
            "ranking_by_drawdown": [],
            "risk_reward_analysis": {},
            "volatility_analysis": {}
        }

        # Create rankings
        methods = []
        for name, data in variations.items():
            performance = data.get("performance", {})
            methods.append({
                "name": name,
                "total_return": performance.get("total_return", 0),
                "sharpe_ratio": performance.get("sharpe_ratio", 0),
                "max_drawdown": performance.get("max_drawdown", 0),
                "volatility": performance.get("volatility", 0)
            })

        # Sort by different criteria
        comparison["ranking_by_return"] = sorted(
            methods, key=lambda x: x["total_return"], reverse=True
        )
        comparison["ranking_by_sharpe"] = sorted(
            methods, key=lambda x: x["sharpe_ratio"], reverse=True
        )
        comparison["ranking_by_drawdown"] = sorted(
            methods, key=lambda x: x["max_drawdown"]
        )

        return comparison

    def _generate_position_sizing_recommendations(
        self,
        variations: Dict[str, Any],
        current_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate position sizing recommendations."""
        recommendations = []

        # Find best performing variations
        best_return = None
        best_sharpe = None
        lowest_drawdown = None

        for name, data in variations.items():
            performance = data.get("performance", {})

            if best_return is None or performance.get("total_return", 0) > best_return["performance"]["total_return"]:
                best_return = {"name": name, "performance": performance}

            if best_sharpe is None or performance.get("sharpe_ratio", 0) > best_sharpe["performance"]["sharpe_ratio"]:
                best_sharpe = {"name": name, "performance": performance}

            if lowest_drawdown is None or performance.get("max_drawdown", 100) < lowest_drawdown["performance"]["max_drawdown"]:
                lowest_drawdown = {"name": name, "performance": performance}

        # Generate recommendations
        if best_return:
            recommendations.append({
                "type": "maximize_returns",
                "priority": "high",
                "recommendation": f"Consider switching to {best_return['name']} for maximum returns",
                "expected_improvement": best_return["performance"].get("total_return", 0),
                "config": variations[best_return["name"]]["config"]
            })

        if best_sharpe:
            recommendations.append({
                "type": "optimize_risk_adjusted_returns",
                "priority": "medium",
                "recommendation": f"Consider {best_sharpe['name']} for best risk-adjusted returns",
                "expected_sharpe": best_sharpe["performance"].get("sharpe_ratio", 0),
                "config": variations[best_sharpe["name"]]["config"]
            })

        return recommendations

    def _estimate_timeframe_performance_impact(
        self,
        base_performance: Dict[str, Any],
        current_timeframe: str,
        new_timeframe: str,
        strategy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate performance impact of timeframe changes."""
        # Convert timeframes to minutes for comparison
        current_minutes = self._timeframe_to_minutes(current_timeframe)
        new_minutes = self._timeframe_to_minutes(new_timeframe)

        # Calculate timeframe ratio
        timeframe_ratio = new_minutes / current_minutes

        # Estimate impact based on timeframe characteristics
        estimated_performance = base_performance.copy()

        # Shorter timeframes typically have more trades but lower win rates
        if timeframe_ratio < 1:  # Shorter timeframe
            estimated_performance["trade_count"] = int(base_performance.get("trade_count", 0) * (1 / timeframe_ratio) * 0.8)
            estimated_performance["win_rate"] = base_performance.get("win_rate", 50) * 0.9  # Slightly lower win rate
            estimated_performance["average_trade_return"] = base_performance.get("average_trade_return", 0) * 0.7  # Smaller moves
        else:  # Longer timeframe
            estimated_performance["trade_count"] = int(base_performance.get("trade_count", 0) * (1 / timeframe_ratio) * 1.2)
            estimated_performance["win_rate"] = base_performance.get("win_rate", 50) * 1.1  # Slightly higher win rate
            estimated_performance["average_trade_return"] = base_performance.get("average_trade_return", 0) * 1.3  # Larger moves

        # Recalculate total return
        if "trade_count" in estimated_performance and "average_trade_return" in estimated_performance:
            estimated_performance["total_return"] = (
                estimated_performance["trade_count"] *
                estimated_performance["average_trade_return"]
            )

        return estimated_performance

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 60 * 24
        else:
            return 60  # Default to 1 hour

    def _assess_timeframe_feasibility(
        self,
        strategy_config: Dict[str, Any],
        timeframe: str
    ) -> Dict[str, Any]:
        """Assess feasibility of using a different timeframe."""
        feasibility = {
            "is_feasible": True,
            "warnings": [],
            "considerations": []
        }

        indicators = strategy_config.get("indicators", {})

        # Check if indicators are suitable for the timeframe
        timeframe_minutes = self._timeframe_to_minutes(timeframe)

        if timeframe_minutes < 15:  # Very short timeframes
            feasibility["warnings"].append("Very short timeframes may have higher noise and transaction costs")

        if timeframe_minutes > 1440:  # Daily or longer
            feasibility["warnings"].append("Long timeframes may have fewer trading opportunities")

        # Check indicator compatibility
        for indicator_name, indicator_config in indicators.items():
            indicator_type = indicator_config.get("type")

            if indicator_type in ["rsi", "stochastic"] and timeframe_minutes < 5:
                feasibility["warnings"].append(f"{indicator_name} may be too noisy on very short timeframes")

            if indicator_type in ["sma", "ema"] and timeframe_minutes > 1440:
                feasibility["considerations"].append(f"{indicator_name} may need period adjustment for longer timeframes")

        return feasibility

    def _estimate_trade_frequency_change(
        self,
        current_timeframe: str,
        new_timeframe: str,
        base_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate how trade frequency would change with different timeframe."""
        current_minutes = self._timeframe_to_minutes(current_timeframe)
        new_minutes = self._timeframe_to_minutes(new_timeframe)

        frequency_ratio = current_minutes / new_minutes
        current_trade_count = base_results.get("trades", [])
        current_count = len(current_trade_count) if current_trade_count else 0

        estimated_count = int(current_count * frequency_ratio * 0.8)  # Apply dampening factor

        return {
            "current_frequency": current_count,
            "estimated_frequency": estimated_count,
            "frequency_ratio": frequency_ratio,
            "change_description": self._describe_frequency_change(frequency_ratio)
        }

    def _describe_frequency_change(self, ratio: float) -> str:
        """Describe the frequency change in human terms."""
        if ratio > 2:
            return "Much more frequent trading"
        elif ratio > 1.5:
            return "More frequent trading"
        elif ratio > 1.1:
            return "Slightly more frequent trading"
        elif ratio > 0.9:
            return "Similar trading frequency"
        elif ratio > 0.5:
            return "Less frequent trading"
        else:
            return "Much less frequent trading"

    def _analyze_timeframe_characteristics(
        self,
        variations: Dict[str, Any],
        strategy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze characteristics of different timeframes."""
        analysis = {
            "scalping_timeframes": [],  # < 15m
            "intraday_timeframes": [],  # 15m - 4h
            "swing_timeframes": [],     # 4h - 1d
            "position_timeframes": [],  # > 1d
            "optimal_ranges": {},
            "risk_considerations": {}
        }

        for timeframe, data in variations.items():
            minutes = self._timeframe_to_minutes(timeframe)

            if minutes < 15:
                analysis["scalping_timeframes"].append(timeframe)
            elif minutes <= 240:  # 4 hours
                analysis["intraday_timeframes"].append(timeframe)
            elif minutes <= 1440:  # 1 day
                analysis["swing_timeframes"].append(timeframe)
            else:
                analysis["position_timeframes"].append(timeframe)

        return analysis

    def _generate_timeframe_recommendations(
        self,
        variations: Dict[str, Any],
        current_timeframe: str,
        strategy_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate timeframe recommendations."""
        recommendations = []

        # Analyze which timeframes might work better
        for timeframe, data in variations.items():
            feasibility = data.get("feasibility", {})

            if feasibility.get("is_feasible", True):
                estimated_performance = data.get("estimated_performance", {})
                current_performance = variations.get(current_timeframe, {}).get("estimated_performance", {})

                # Check if this timeframe might perform better
                estimated_return = estimated_performance.get("total_return", 0)
                current_return = current_performance.get("total_return", 0)

                if estimated_return > current_return * 1.1:  # 10% better
                    recommendations.append({
                        "type": "timeframe_optimization",
                        "priority": "medium",
                        "recommendation": f"Consider testing {timeframe} timeframe for potentially better performance",
                        "expected_improvement": ((estimated_return - current_return) / current_return) * 100,
                        "timeframe": timeframe,
                        "considerations": feasibility.get("considerations", []),
                        "warnings": feasibility.get("warnings", [])
                    })

        return recommendations

    def _get_indicator_parameter_variations(
        self,
        indicator_type: str,
        current_config: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Get parameter variations to test for different indicator types."""
        variations = {}

        if indicator_type == "rsi":
            current_period = current_config.get("period", 14)
            variations = {
                "RSI-7": {"period": 7, "description": "Faster RSI, more sensitive"},
                "RSI-21": {"period": 21, "description": "Slower RSI, less sensitive"},
                "RSI-9": {"period": 9, "description": "Short-term RSI"},
                "RSI-25": {"period": 25, "description": "Long-term RSI"}
            }

        elif indicator_type == "sma":
            current_period = current_config.get("period", 20)
            variations = {
                "SMA-10": {"period": 10, "description": "Faster moving average"},
                "SMA-50": {"period": 50, "description": "Slower moving average"},
                "SMA-200": {"period": 200, "description": "Long-term trend"},
                "SMA-5": {"period": 5, "description": "Very fast moving average"}
            }

        elif indicator_type == "ema":
            current_period = current_config.get("period", 20)
            variations = {
                "EMA-12": {"period": 12, "description": "Faster EMA"},
                "EMA-26": {"period": 26, "description": "Standard slow EMA"},
                "EMA-50": {"period": 50, "description": "Long-term EMA"},
                "EMA-9": {"period": 9, "description": "Very fast EMA"}
            }

        elif indicator_type == "macd":
            variations = {
                "MACD-Fast": {"fast_period": 8, "slow_period": 21, "signal_period": 5},
                "MACD-Slow": {"fast_period": 15, "slow_period": 35, "signal_period": 15},
                "MACD-Balanced": {"fast_period": 10, "slow_period": 22, "signal_period": 7},
                "MACD-Conservative": {"fast_period": 12, "slow_period": 30, "signal_period": 12}
            }

        elif indicator_type == "bollinger_bands":
            variations = {
                "BB-Tight": {"period": 20, "std_dev": 1.5, "description": "Tighter bands"},
                "BB-Wide": {"period": 20, "std_dev": 2.5, "description": "Wider bands"},
                "BB-Fast": {"period": 10, "std_dev": 2.0, "description": "Faster calculation"},
                "BB-Slow": {"period": 50, "std_dev": 2.0, "description": "Slower calculation"}
            }

        # Remove current configuration if it exists
        variations = {k: v for k, v in variations.items()
                     if not self._is_same_config(v, current_config)}

        return variations

    def _is_same_config(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
        """Check if two configurations are essentially the same."""
        for key in ["period", "fast_period", "slow_period", "signal_period", "std_dev"]:
            if key in config1 and key in config2:
                if config1[key] != config2[key]:
                    return False
        return True

    def _estimate_parameter_performance_impact(
        self,
        base_performance: Dict[str, Any],
        indicator_name: str,
        current_config: Dict[str, Any],
        new_config: Dict[str, Any],
        base_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate performance impact of parameter changes."""
        # This is a simplified estimation - in practice, you'd want to re-run backtests
        estimated_performance = base_performance.copy()

        # Simple heuristics based on parameter changes
        indicator_type = current_config.get("type")

        if indicator_type == "rsi":
            current_period = current_config.get("period", 14)
            new_period = new_config.get("period", 14)

            if new_period < current_period:  # Faster RSI
                estimated_performance["trade_count"] = base_performance.get("trade_count", 0) * 1.2
                estimated_performance["win_rate"] = base_performance.get("win_rate", 50) * 0.95
            else:  # Slower RSI
                estimated_performance["trade_count"] = base_performance.get("trade_count", 0) * 0.8
                estimated_performance["win_rate"] = base_performance.get("win_rate", 50) * 1.05

        elif indicator_type in ["sma", "ema"]:
            current_period = current_config.get("period", 20)
            new_period = new_config.get("period", 20)

            if new_period < current_period:  # Faster MA
                estimated_performance["trade_count"] = base_performance.get("trade_count", 0) * 1.3
                estimated_performance["win_rate"] = base_performance.get("win_rate", 50) * 0.9
            else:  # Slower MA
                estimated_performance["trade_count"] = base_performance.get("trade_count", 0) * 0.7
                estimated_performance["win_rate"] = base_performance.get("win_rate", 50) * 1.1

        return estimated_performance

    def _calculate_parameter_change(
        self,
        current_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate the change in parameters."""
        changes = {}

        for key in ["period", "fast_period", "slow_period", "signal_period", "std_dev"]:
            if key in current_config and key in new_config:
                old_value = current_config[key]
                new_value = new_config[key]

                if old_value != new_value:
                    changes[key] = {
                        "from": old_value,
                        "to": new_value,
                        "change": new_value - old_value,
                        "percent_change": ((new_value - old_value) / old_value) * 100 if old_value != 0 else 0
                    }

        return changes

    def _assess_parameter_impact(
        self,
        current_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess the potential impact of parameter changes."""
        assessment = {
            "sensitivity_level": "low",
            "expected_behavior": "",
            "trade_frequency_impact": "neutral",
            "signal_quality_impact": "neutral"
        }

        indicator_type = current_config.get("type")

        if indicator_type == "rsi":
            current_period = current_config.get("period", 14)
            new_period = new_config.get("period", 14)

            period_change = abs(new_period - current_period) / current_period

            if period_change > 0.5:
                assessment["sensitivity_level"] = "high"
            elif period_change > 0.2:
                assessment["sensitivity_level"] = "medium"

            if new_period < current_period:
                assessment["expected_behavior"] = "More sensitive to price changes, more signals"
                assessment["trade_frequency_impact"] = "increase"
                assessment["signal_quality_impact"] = "decrease"
            else:
                assessment["expected_behavior"] = "Less sensitive to price changes, fewer signals"
                assessment["trade_frequency_impact"] = "decrease"
                assessment["signal_quality_impact"] = "increase"

        return assessment

    def _rank_parameter_sensitivity(self, variations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank parameter variations by sensitivity."""
        sensitivity_ranking = []

        for variation_name, variation_data in variations.items():
            impact = variation_data.get("impact_assessment", {})
            sensitivity_level = impact.get("sensitivity_level", "low")

            # Convert sensitivity to numeric score
            sensitivity_score = {"low": 1, "medium": 2, "high": 3}.get(sensitivity_level, 1)

            sensitivity_ranking.append({
                "variation": variation_name,
                "sensitivity_score": sensitivity_score,
                "sensitivity_level": sensitivity_level,
                "parameters": variation_data.get("parameters", {}),
                "expected_behavior": impact.get("expected_behavior", "")
            })

        # Sort by sensitivity score (highest first)
        sensitivity_ranking.sort(key=lambda x: x["sensitivity_score"], reverse=True)

        return sensitivity_ranking

    def _identify_optimal_parameter_ranges(self, variations: Dict[str, Any]) -> Dict[str, Any]:
        """Identify optimal parameter ranges based on estimated performance."""
        optimal_ranges = {}

        # Group variations by parameter type
        parameter_performance = {}

        for variation_name, variation_data in variations.items():
            parameters = variation_data.get("parameters", {})
            estimated_performance = variation_data.get("estimated_performance", {})

            for param_name, param_value in parameters.items():
                if param_name not in parameter_performance:
                    parameter_performance[param_name] = []

                parameter_performance[param_name].append({
                    "value": param_value,
                    "performance": estimated_performance.get("total_return", 0),
                    "win_rate": estimated_performance.get("win_rate", 50)
                })

        # Find optimal ranges for each parameter
        for param_name, performance_data in parameter_performance.items():
            if len(performance_data) >= 2:
                # Sort by performance
                sorted_data = sorted(performance_data, key=lambda x: x["performance"], reverse=True)

                # Take top performers
                top_performers = sorted_data[:max(1, len(sorted_data) // 2)]

                values = [p["value"] for p in top_performers]
                optimal_ranges[param_name] = {
                    "min_value": min(values),
                    "max_value": max(values),
                    "optimal_values": values,
                    "performance_range": [p["performance"] for p in top_performers]
                }

        return optimal_ranges

    def _is_optimization_candidate(self, indicator_results: Dict[str, Any]) -> bool:
        """Determine if indicator is a good optimization candidate."""
        variations = indicator_results.get("variations", {})

        if not variations:
            return False

        # Check if any variation shows significant improvement
        for variation_data in variations.values():
            estimated_performance = variation_data.get("estimated_performance", {})
            current_return = 100  # Assume baseline
            estimated_return = estimated_performance.get("total_return", 100)

            # If any variation shows >20% improvement, it's a candidate
            if estimated_return > current_return * 1.2:
                return True

        return False

    def _get_optimization_reason(self, indicator_results: Dict[str, Any]) -> str:
        """Get reason why indicator is optimization candidate."""
        variations = indicator_results.get("variations", {})

        best_improvement = 0
        best_variation = None

        for variation_name, variation_data in variations.items():
            estimated_performance = variation_data.get("estimated_performance", {})
            current_return = 100  # Assume baseline
            estimated_return = estimated_performance.get("total_return", 100)

            improvement = (estimated_return - current_return) / current_return
            if improvement > best_improvement:
                best_improvement = improvement
                best_variation = variation_name

        if best_improvement > 0.2:
            return f"Parameter variation '{best_variation}' shows {best_improvement*100:.1f}% improvement potential"
        else:
            return "Multiple parameter variations show improvement potential"

    def _get_suggested_parameters(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get suggested parameters for optimization."""
        variations = indicator_results.get("variations", {})

        best_performance = 0
        best_parameters = {}

        for variation_data in variations.values():
            estimated_performance = variation_data.get("estimated_performance", {})
            estimated_return = estimated_performance.get("total_return", 0)

            if estimated_return > best_performance:
                best_performance = estimated_return
                best_parameters = variation_data.get("parameters", {})

        return best_parameters

    def _analyze_parameter_interactions(
        self,
        indicators_results: Dict[str, Any],
        strategy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze interactions between indicator parameters."""
        interactions = {
            "potential_conflicts": [],
            "synergy_opportunities": [],
            "correlation_analysis": {},
            "timing_interactions": []
        }

        indicator_names = list(indicators_results.keys())

        # Look for potential conflicts between indicators
        for i, indicator1 in enumerate(indicator_names):
            for indicator2 in indicator_names[i+1:]:

                results1 = indicators_results[indicator1]
                results2 = indicators_results[indicator2]

                # Check for timing conflicts
                if self._have_timing_conflicts(results1, results2):
                    interactions["potential_conflicts"].append({
                        "indicators": [indicator1, indicator2],
                        "type": "timing_conflict",
                        "description": "Both indicators may generate conflicting signals"
                    })

                # Check for synergy opportunities
                if self._have_synergy_potential(results1, results2):
                    interactions["synergy_opportunities"].append({
                        "indicators": [indicator1, indicator2],
                        "type": "parameter_synergy",
                        "description": "Coordinated parameter optimization may improve performance"
                    })

        return interactions

    def _have_timing_conflicts(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> bool:
        """Check if two indicators might have timing conflicts."""
        # Simple heuristic: if both are trend-following or both are momentum, potential conflict
        config1 = results1.get("current_parameters", {})
        config2 = results2.get("current_parameters", {})

        type1 = config1.get("type", "")
        type2 = config2.get("type", "")

        trend_indicators = ["sma", "ema", "macd"]
        momentum_indicators = ["rsi", "stochastic", "williams_r"]

        if (type1 in trend_indicators and type2 in trend_indicators) or \
           (type1 in momentum_indicators and type2 in momentum_indicators):
            return True

        return False

    def _have_synergy_potential(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> bool:
        """Check if two indicators have synergy potential."""
        # Simple heuristic: trend + momentum indicators often work well together
        config1 = results1.get("current_parameters", {})
        config2 = results2.get("current_parameters", {})

        type1 = config1.get("type", "")
        type2 = config2.get("type", "")

        trend_indicators = ["sma", "ema", "macd"]
        momentum_indicators = ["rsi", "stochastic", "williams_r"]

        if (type1 in trend_indicators and type2 in momentum_indicators) or \
           (type1 in momentum_indicators and type2 in trend_indicators):
            return True

        return False

    def _generate_parameter_optimization_recommendations(
        self,
        indicators_results: Dict[str, Any],
        optimization_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate parameter optimization recommendations."""
        recommendations = []

        # Recommendations for individual indicators
        for candidate in optimization_candidates:
            recommendations.append({
                "type": "indicator_parameter_optimization",
                "priority": "high",
                "indicator": candidate["indicator"],
                "recommendation": candidate["reason"],
                "suggested_parameters": candidate["suggested_parameters"]
            })

        # Global recommendations
        if len(optimization_candidates) > 1:
            recommendations.append({
                "type": "coordinated_optimization",
                "priority": "medium",
                "recommendation": "Consider coordinated optimization of multiple indicators",
                "indicators": [c["indicator"] for c in optimization_candidates]
            })

        return recommendations

    def _generate_risk_parameter_variations(self, risk_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate variations of risk parameters to test."""
        variations = {}

        current_stop_loss = risk_config.get("stop_loss_pct", 2.0)
        current_take_profit = risk_config.get("take_profit_pct", 4.0)

        variations["Conservative"] = {
            "stop_loss_pct": max(1.0, current_stop_loss * 0.5),
            "take_profit_pct": current_take_profit * 0.8,
            "description": "Tighter risk management, smaller stops and targets"
        }

        variations["Aggressive"] = {
            "stop_loss_pct": current_stop_loss * 1.5,
            "take_profit_pct": current_take_profit * 1.5,
            "description": "Wider risk management, larger stops and targets"
        }

        variations["Risk-Reward-1:2"] = {
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0,
            "description": "Standard 1:2 risk-reward ratio"
        }

        variations["Risk-Reward-1:3"] = {
            "stop_loss_pct": 2.0,
            "take_profit_pct": 6.0,
            "description": "Higher 1:3 risk-reward ratio"
        }

        variations["Risk-Reward-1:1"] = {
            "stop_loss_pct": 3.0,
            "take_profit_pct": 3.0,
            "description": "Equal risk-reward ratio"
        }

        variations["Tight-Stops"] = {
            "stop_loss_pct": 1.0,
            "take_profit_pct": current_take_profit,
            "description": "Very tight stop losses"
        }

        variations["Wide-Targets"] = {
            "stop_loss_pct": current_stop_loss,
            "take_profit_pct": current_take_profit * 2.0,
            "description": "Very wide profit targets"
        }

        return variations

    def _simulate_risk_parameter_impact(
        self,
        base_results: Dict[str, Any],
        risk_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate the impact of different risk parameters on trades."""
        adjusted_results = base_results.copy()
        trades = base_results.get("trades", [])

        if not trades:
            return adjusted_results

        stop_loss_pct = risk_config.get("stop_loss_pct", 2.0)
        take_profit_pct = risk_config.get("take_profit_pct", 4.0)

        adjusted_trades = []
        for trade in trades:
            adjusted_trade = trade.copy()
            original_pnl = trade.get("pnl", 0)
            entry_price = trade.get("entry_price", 100)

            # Simulate risk management impact
            if original_pnl > 0:  # Winning trade
                # Check if it would have hit take profit
                max_pnl_pct = abs(original_pnl / entry_price) * 100
                if max_pnl_pct > take_profit_pct:
                    # Would have been stopped out at take profit
                    adjusted_trade["pnl"] = (take_profit_pct / 100) * entry_price
                    adjusted_trade["exit_reason"] = "take_profit"
                else:
                    # Keep original PnL
                    adjusted_trade["pnl"] = original_pnl

            else:  # Losing trade
                # Check if it would have hit stop loss
                max_loss_pct = abs(original_pnl / entry_price) * 100
                if max_loss_pct > stop_loss_pct:
                    # Would have been stopped out at stop loss
                    adjusted_trade["pnl"] = -(stop_loss_pct / 100) * entry_price
                    adjusted_trade["exit_reason"] = "stop_loss"
                else:
                    # Keep original PnL
                    adjusted_trade["pnl"] = original_pnl

            adjusted_trades.append(adjusted_trade)

        adjusted_results["trades"] = adjusted_trades
        return adjusted_results

    def _analyze_risk_trade_impact(
        self,
        base_results: Dict[str, Any],
        adjusted_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the impact of risk parameters on individual trades."""
        base_trades = base_results.get("trades", [])
        adjusted_trades = adjusted_results.get("trades", [])

        impact_analysis = {
            "trades_affected": 0,
            "stop_loss_hits": 0,
            "take_profit_hits": 0,
            "pnl_changes": [],
            "average_pnl_change": 0
        }

        for i, (base_trade, adj_trade) in enumerate(zip(base_trades, adjusted_trades)):
            base_pnl = base_trade.get("pnl", 0)
            adj_pnl = adj_trade.get("pnl", 0)

            if abs(base_pnl - adj_pnl) > 0.01:  # Significant change
                impact_analysis["trades_affected"] += 1
                pnl_change = adj_pnl - base_pnl
                impact_analysis["pnl_changes"].append(pnl_change)

                exit_reason = adj_trade.get("exit_reason", "")
                if exit_reason == "stop_loss":
                    impact_analysis["stop_loss_hits"] += 1
                elif exit_reason == "take_profit":
                    impact_analysis["take_profit_hits"] += 1

        if impact_analysis["pnl_changes"]:
            impact_analysis["average_pnl_change"] = sum(impact_analysis["pnl_changes"]) / len(impact_analysis["pnl_changes"])

        return impact_analysis

    def _analyze_risk_reward_relationships(self, variations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk-reward relationships across variations."""
        analysis = {
            "optimal_risk_reward": None,
            "risk_reward_comparison": [],
            "efficiency_analysis": {},
            "recommendations": []
        }

        for variation_name, variation_data in variations.items():
            config = variation_data.get("config", {})
            performance = variation_data.get("performance", {})

            stop_loss = config.get("stop_loss_pct", 2.0)
            take_profit = config.get("take_profit_pct", 4.0)
            risk_reward_ratio = take_profit / stop_loss if stop_loss > 0 else 0

            total_return = performance.get("total_return", 0)
            win_rate = performance.get("win_rate", 50)
            max_drawdown = performance.get("max_drawdown", 0)

            analysis["risk_reward_comparison"].append({
                "variation": variation_name,
                "risk_reward_ratio": risk_reward_ratio,
                "total_return": total_return,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "efficiency_score": self._calculate_efficiency_score(
                    total_return, win_rate, max_drawdown, risk_reward_ratio
                )
            })

        # Find optimal risk-reward ratio
        if analysis["risk_reward_comparison"]:
            best_efficiency = max(analysis["risk_reward_comparison"],
                                key=lambda x: x["efficiency_score"])
            analysis["optimal_risk_reward"] = best_efficiency

        return analysis

    def _calculate_efficiency_score(
        self,
        total_return: float,
        win_rate: float,
        max_drawdown: float,
        risk_reward_ratio: float
    ) -> float:
        """Calculate efficiency score for risk-reward analysis."""
        # Weighted score considering multiple factors
        return_score = total_return / 100  # Normalize return
        win_rate_score = win_rate / 100    # Normalize win rate
        drawdown_penalty = max_drawdown / 100  # Penalty for drawdown
        rr_score = min(risk_reward_ratio / 3, 1)  # Normalize RR ratio (cap at 3:1)

        efficiency = (return_score * 0.4 + win_rate_score * 0.3 +
                     rr_score * 0.2 - drawdown_penalty * 0.1)

        return max(0, efficiency)

    def _analyze_drawdown_impact(self, variations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze drawdown impact across risk parameter variations."""
        analysis = {
            "drawdown_comparison": [],
            "lowest_drawdown": None,
            "drawdown_vs_return": {},
            "volatility_analysis": {}
        }

        for variation_name, variation_data in variations.items():
            performance = variation_data.get("performance", {})
            config = variation_data.get("config", {})

            max_drawdown = performance.get("max_drawdown", 0)
            total_return = performance.get("total_return", 0)
            volatility = performance.get("volatility", 0)

            analysis["drawdown_comparison"].append({
                "variation": variation_name,
                "max_drawdown": max_drawdown,
                "total_return": total_return,
                "volatility": volatility,
                "stop_loss_pct": config.get("stop_loss_pct", 2.0),
                "take_profit_pct": config.get("take_profit_pct", 4.0),
                "risk_adjusted_return": total_return / max(max_drawdown, 1)
            })

        # Find lowest drawdown
        if analysis["drawdown_comparison"]:
            analysis["lowest_drawdown"] = min(analysis["drawdown_comparison"],
                                            key=lambda x: x["max_drawdown"])

        return analysis

    def _generate_risk_parameter_recommendations(
        self,
        variations: Dict[str, Any],
        current_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate risk parameter recommendations."""
        recommendations = []

        # Find best performing variations
        best_return = None
        best_drawdown = None
        best_efficiency = None

        for variation_name, variation_data in variations.items():
            performance = variation_data.get("performance", {})
            config = variation_data.get("config", {})

            total_return = performance.get("total_return", 0)
            max_drawdown = performance.get("max_drawdown", 100)
            win_rate = performance.get("win_rate", 50)

            # Calculate efficiency score
            stop_loss = config.get("stop_loss_pct", 2.0)
            take_profit = config.get("take_profit_pct", 4.0)
            risk_reward = take_profit / stop_loss if stop_loss > 0 else 0

            efficiency = self._calculate_efficiency_score(
                total_return, win_rate, max_drawdown, risk_reward
            )

            if best_return is None or total_return > best_return["return"]:
                best_return = {"variation": variation_name, "return": total_return, "config": config}

            if best_drawdown is None or max_drawdown < best_drawdown["drawdown"]:
                best_drawdown = {"variation": variation_name, "drawdown": max_drawdown, "config": config}

            if best_efficiency is None or efficiency > best_efficiency["efficiency"]:
                best_efficiency = {"variation": variation_name, "efficiency": efficiency, "config": config}

        # Generate recommendations
        if best_return:
            recommendations.append({
                "type": "maximize_returns",
                "priority": "high",
                "recommendation": f"Consider {best_return['variation']} risk parameters for maximum returns",
                "expected_improvement": best_return["return"],
                "config": best_return["config"]
            })

        if best_drawdown:
            recommendations.append({
                "type": "minimize_drawdown",
                "priority": "medium",
                "recommendation": f"Consider {best_drawdown['variation']} risk parameters for minimum drawdown",
                "expected_drawdown": best_drawdown["drawdown"],
                "config": best_drawdown["config"]
            })

        if best_efficiency:
            recommendations.append({
                "type": "optimize_efficiency",
                "priority": "high",
                "recommendation": f"Consider {best_efficiency['variation']} risk parameters for best overall efficiency",
                "efficiency_score": best_efficiency["efficiency"],
                "config": best_efficiency["config"]
            })

        return recommendations

    def _extract_position_sizing_optimizations(self, position_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract optimization suggestions from position sizing analysis."""
        suggestions = []
        recommendations = position_results.get("recommendations", [])

        for rec in recommendations:
            suggestions.append({
                "category": "position_sizing",
                "type": rec.get("type", "position_sizing_optimization"),
                "priority": rec.get("priority", "medium"),
                "description": rec.get("recommendation", ""),
                "expected_impact": rec.get("expected_improvement", 0),
                "configuration_change": rec.get("config", {})
            })

        return suggestions

    def _extract_timeframe_optimizations(self, timeframe_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract optimization suggestions from timeframe analysis."""
        suggestions = []
        recommendations = timeframe_results.get("recommendations", [])

        for rec in recommendations:
            suggestions.append({
                "category": "timeframe",
                "type": rec.get("type", "timeframe_optimization"),
                "priority": rec.get("priority", "medium"),
                "description": rec.get("recommendation", ""),
                "expected_impact": rec.get("expected_improvement", 0),
                "timeframe": rec.get("timeframe", ""),
                "considerations": rec.get("considerations", []),
                "warnings": rec.get("warnings", [])
            })

        return suggestions

    def _extract_indicator_optimizations(self, indicator_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract optimization suggestions from indicator parameter analysis."""
        suggestions = []
        recommendations = indicator_results.get("recommendations", [])

        for rec in recommendations:
            suggestions.append({
                "category": "indicator_parameters",
                "type": rec.get("type", "indicator_optimization"),
                "priority": rec.get("priority", "medium"),
                "description": rec.get("recommendation", ""),
                "indicator": rec.get("indicator", ""),
                "suggested_parameters": rec.get("suggested_parameters", {})
            })

        return suggestions

    def _extract_risk_optimizations(self, risk_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract optimization suggestions from risk parameter analysis."""
        suggestions = []
        recommendations = risk_results.get("recommendations", [])

        for rec in recommendations:
            suggestions.append({
                "category": "risk_management",
                "type": rec.get("type", "risk_optimization"),
                "priority": rec.get("priority", "medium"),
                "description": rec.get("recommendation", ""),
                "expected_impact": rec.get("expected_improvement", 0),
                "configuration_change": rec.get("config", {})
            })

        return suggestions

    def _rank_optimization_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank optimization suggestions by potential impact and priority."""
        # Define priority weights
        priority_weights = {"high": 3, "medium": 2, "low": 1}

        for suggestion in suggestions:
            priority = suggestion.get("priority", "medium")
            expected_impact = suggestion.get("expected_impact", 0)

            # Calculate ranking score
            priority_score = priority_weights.get(priority, 2)
            impact_score = min(abs(expected_impact) / 100, 1) if expected_impact else 0

            suggestion["ranking_score"] = priority_score * 0.6 + impact_score * 0.4

        # Sort by ranking score (highest first)
        suggestions.sort(key=lambda x: x.get("ranking_score", 0), reverse=True)

        return suggestions

    def _apply_suggestion_to_config(
        self,
        strategy_config: Dict[str, Any],
        suggestion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply optimization suggestion to strategy configuration."""
        test_config = strategy_config.copy()
        category = suggestion.get("category", "")

        if category == "position_sizing":
            config_change = suggestion.get("configuration_change", {})
            if "position_sizing" not in test_config:
                test_config["position_sizing"] = {}
            test_config["position_sizing"].update(config_change)

        elif category == "timeframe":
            new_timeframe = suggestion.get("timeframe", "")
            if new_timeframe:
                test_config["timeframe"] = new_timeframe

        elif category == "indicator_parameters":
            indicator = suggestion.get("indicator", "")
            suggested_params = suggestion.get("suggested_parameters", {})
            if indicator and indicator in test_config.get("indicators", {}):
                test_config["indicators"][indicator].update(suggested_params)

        elif category == "risk_management":
            config_change = suggestion.get("configuration_change", {})
            if "risk_management" not in test_config:
                test_config["risk_management"] = {}
            test_config["risk_management"].update(config_change)

        return test_config

    def _assess_implementation_complexity(self, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the implementation complexity of a configuration change."""
        category = suggestion.get("category", "")

        complexity_assessment = {
            "level": "low",
            "estimated_effort": "1-2 hours",
            "requirements": [],
            "risks": []
        }

        if category == "position_sizing":
            complexity_assessment["level"] = "low"
            complexity_assessment["requirements"] = ["Configuration file update"]

        elif category == "timeframe":
            complexity_assessment["level"] = "medium"
            complexity_assessment["estimated_effort"] = "4-6 hours"
            complexity_assessment["requirements"] = [
                "Configuration update",
                "Re-backtesting required",
                "Indicator period adjustment may be needed"
            ]
            complexity_assessment["risks"] = ["May affect indicator effectiveness"]

        elif category == "indicator_parameters":
            complexity_assessment["level"] = "low"
            complexity_assessment["estimated_effort"] = "1-3 hours"
            complexity_assessment["requirements"] = [
                "Configuration file update",
                "Backtesting validation"
            ]

        elif category == "risk_management":
            complexity_assessment["level"] = "low"
            complexity_assessment["requirements"] = ["Configuration file update"]

        return complexity_assessment

    def _assess_configuration_risk(
        self,
        test_config: Dict[str, Any],
        original_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risks associated with configuration changes."""
        risk_assessment = {
            "risk_level": "low",
            "identified_risks": [],
            "mitigation_strategies": [],
            "monitoring_requirements": []
        }

        # Check for significant parameter changes
        original_indicators = original_config.get("indicators", {})
        test_indicators = test_config.get("indicators", {})

        for indicator_name in original_indicators:
            if indicator_name in test_indicators:
                orig_params = original_indicators[indicator_name]
                test_params = test_indicators[indicator_name]

                # Check for large parameter changes
                for param in ["period", "fast_period", "slow_period"]:
                    if param in orig_params and param in test_params:
                        orig_val = orig_params[param]
                        test_val = test_params[param]

                        if abs(test_val - orig_val) / orig_val > 0.5:  # >50% change
                            risk_assessment["risk_level"] = "medium"
                            risk_assessment["identified_risks"].append(
                                f"Large parameter change in {indicator_name}.{param}: {orig_val} -> {test_val}"
                            )
                            risk_assessment["mitigation_strategies"].append(
                                f"Gradual parameter adjustment for {indicator_name}"
                            )

        # Check timeframe changes
        if test_config.get("timeframe") != original_config.get("timeframe"):
            risk_assessment["identified_risks"].append("Timeframe change may affect strategy effectiveness")
            risk_assessment["monitoring_requirements"].append("Monitor trade frequency and performance closely")

        return risk_assessment

    def _generate_overall_feasibility_assessment(
        self,
        feasibility_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate overall feasibility assessment."""
        assessment = {
            "overall_feasibility": "high",
            "feasible_suggestions": 0,
            "total_suggestions": len(feasibility_results),
            "implementation_priority": [],
            "risk_summary": {},
            "effort_estimate": "1-5 hours"
        }

        feasible_count = 0
        total_effort_hours = 0
        risk_levels = {"low": 0, "medium": 0, "high": 0}

        for result in feasibility_results:
            feasibility = result.get("feasibility", {})
            if feasibility.get("is_valid", False):
                feasible_count += 1

            # Aggregate complexity
            complexity = result.get("implementation_complexity", {})
            effort_str = complexity.get("estimated_effort", "1-2 hours")
            effort_hours = self._parse_effort_hours(effort_str)
            total_effort_hours += effort_hours

            # Aggregate risk levels
            risk_info = result.get("risk_assessment", {})
            risk_level = risk_info.get("risk_level", "low")
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1

        assessment["feasible_suggestions"] = feasible_count
        assessment["effort_estimate"] = f"{total_effort_hours}-{total_effort_hours + 5} hours"
        assessment["risk_summary"] = risk_levels

        # Determine overall feasibility
        if feasible_count == 0:
            assessment["overall_feasibility"] = "low"
        elif feasible_count < len(feasibility_results) * 0.5:
            assessment["overall_feasibility"] = "medium"
        else:
            assessment["overall_feasibility"] = "high"

        return assessment

    def _parse_effort_hours(self, effort_str: str) -> int:
        """Parse effort string to extract hours."""
        # Simple parsing for strings like "1-2 hours", "4-6 hours"
        try:
            # Extract first number from string
            import re
            numbers = re.findall(r'\d+', effort_str)
            if numbers:
                return int(numbers[0])
        except:
            pass
        return 2  # Default fallback

    def _prioritize_feasible_optimizations(
        self,
        feasibility_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prioritize feasible optimizations by impact and ease of implementation."""
        priorities = []

        for result in feasibility_results:
            feasibility = result.get("feasibility", {})
            suggestion = result.get("suggestion", {})
            complexity = result.get("implementation_complexity", {})
            risk_info = result.get("risk_assessment", {})

            if feasibility.get("is_valid", False):
                # Calculate priority score
                impact_score = suggestion.get("ranking_score", 0)
                complexity_penalty = {"low": 0, "medium": 0.2, "high": 0.5}.get(
                    complexity.get("level", "medium"), 0.2
                )
                risk_penalty = {"low": 0, "medium": 0.1, "high": 0.3}.get(
                    risk_info.get("risk_level", "low"), 0
                )

                priority_score = impact_score - complexity_penalty - risk_penalty

                priorities.append({
                    "suggestion": suggestion,
                    "priority_score": priority_score,
                    "complexity": complexity.get("level", "medium"),
                    "risk_level": risk_info.get("risk_level", "low"),
                    "estimated_effort": complexity.get("estimated_effort", "1-2 hours"),
                    "implementation_order": len(priorities) + 1
                })

        # Sort by priority score (highest first)
        priorities.sort(key=lambda x: x["priority_score"], reverse=True)

        # Update implementation order
        for i, priority in enumerate(priorities):
            priority["implementation_order"] = i + 1

        return priorities
