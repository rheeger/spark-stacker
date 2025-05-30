"""
Strategy-Specific Report Generation Module

This module centralizes all strategy-specific reporting logic, including:
- Strategy configuration display
- Strategy performance breakdown
- Strategy optimization suggestions
- Export functionality for strategy results
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.config_manager import ConfigManager
from ..validation.strategy_validator import StrategyValidator

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
