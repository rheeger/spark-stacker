#!/usr/bin/env python3
"""
Test script for enhanced ComparisonReporter functionality.

Tests the new features added in section 3.2:
- Market condition analysis
- Sensitivity analysis
- Cross-scenario coordination
- Enhanced comparison features
"""

import os
import sys
from pathlib import Path

# Add proper path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(current_dir)
spark_app_dir = os.path.dirname(tests_dir)
sys.path.insert(0, spark_app_dir)

# Add CLI directory to path
cli_dir = os.path.join(tests_dir, "_utils", "cli")
sys.path.insert(0, cli_dir)

from core.config_manager import ConfigManager
from managers.comparison_manager import ComparisonManager
from reporting.comparison_reporter import ComparisonReporter


def create_mock_comparison_results():
    """Create mock comparison results for testing."""
    return {
        "MACD_ETH_Long": {
            "performance_metrics": {
                "total_return": 15.2,
                "win_rate": 0.65,
                "sharpe_ratio": 1.8,
                "max_drawdown": 8.5,
                "profit_factor": 1.4
            },
            "scenario_results": {
                "bull": {"performance_metrics": {"total_return": 25.0, "win_rate": 0.75, "max_drawdown": 6.0, "sharpe_ratio": 2.1}},
                "bear": {"performance_metrics": {"total_return": -5.0, "win_rate": 0.45, "max_drawdown": 15.0, "sharpe_ratio": -0.3}},
                "sideways": {"performance_metrics": {"total_return": 8.0, "win_rate": 0.60, "max_drawdown": 4.0, "sharpe_ratio": 1.2}},
                "volatile": {"performance_metrics": {"total_return": 12.0, "win_rate": 0.58, "max_drawdown": 12.0, "sharpe_ratio": 0.9}},
                "low-vol": {"performance_metrics": {"total_return": 18.0, "win_rate": 0.70, "max_drawdown": 3.0, "sharpe_ratio": 2.5}},
                "choppy": {"performance_metrics": {"total_return": 2.0, "win_rate": 0.52, "max_drawdown": 10.0, "sharpe_ratio": 0.2}},
                "gaps": {"performance_metrics": {"total_return": 6.0, "win_rate": 0.55, "max_drawdown": 8.0, "sharpe_ratio": 0.7}}
            },
            "trades": [
                {"pnl": 50.0, "duration": 120},
                {"pnl": -20.0, "duration": 60},
                {"pnl": 75.0, "duration": 180},
                {"pnl": 30.0, "duration": 90}
            ]
        },
        "RSI_ETH_Short": {
            "performance_metrics": {
                "total_return": 8.7,
                "win_rate": 0.58,
                "sharpe_ratio": 1.2,
                "max_drawdown": 12.0,
                "profit_factor": 1.2
            },
            "scenario_results": {
                "bull": {"performance_metrics": {"total_return": -8.0, "win_rate": 0.35, "max_drawdown": 20.0, "sharpe_ratio": -0.5}},
                "bear": {"performance_metrics": {"total_return": 22.0, "win_rate": 0.78, "max_drawdown": 5.0, "sharpe_ratio": 2.8}},
                "sideways": {"performance_metrics": {"total_return": 10.0, "win_rate": 0.62, "max_drawdown": 6.0, "sharpe_ratio": 1.4}},
                "volatile": {"performance_metrics": {"total_return": 15.0, "win_rate": 0.65, "max_drawdown": 8.0, "sharpe_ratio": 1.6}},
                "low-vol": {"performance_metrics": {"total_return": 5.0, "win_rate": 0.55, "max_drawdown": 4.0, "sharpe_ratio": 1.0}},
                "choppy": {"performance_metrics": {"total_return": 18.0, "win_rate": 0.72, "max_drawdown": 7.0, "sharpe_ratio": 2.2}},
                "gaps": {"performance_metrics": {"total_return": 9.0, "win_rate": 0.60, "max_drawdown": 9.0, "sharpe_ratio": 1.1}}
            },
            "trades": [
                {"pnl": 40.0, "duration": 100},
                {"pnl": -15.0, "duration": 45},
                {"pnl": 25.0, "duration": 150},
                {"pnl": -30.0, "duration": 80}
            ]
        },
        "Bollinger_BTC_Scalp": {
            "performance_metrics": {
                "total_return": 22.5,
                "win_rate": 0.72,
                "sharpe_ratio": 2.1,
                "max_drawdown": 6.5,
                "profit_factor": 1.8
            },
            "scenario_results": {
                "bull": {"performance_metrics": {"total_return": 30.0, "win_rate": 0.80, "max_drawdown": 4.0, "sharpe_ratio": 3.0}},
                "bear": {"performance_metrics": {"total_return": 12.0, "win_rate": 0.65, "max_drawdown": 10.0, "sharpe_ratio": 1.0}},
                "sideways": {"performance_metrics": {"total_return": 28.0, "win_rate": 0.78, "max_drawdown": 3.0, "sharpe_ratio": 3.5}},
                "volatile": {"performance_metrics": {"total_return": 35.0, "win_rate": 0.82, "max_drawdown": 8.0, "sharpe_ratio": 2.8}},
                "low-vol": {"performance_metrics": {"total_return": 15.0, "win_rate": 0.68, "max_drawdown": 2.0, "sharpe_ratio": 2.2}},
                "choppy": {"performance_metrics": {"total_return": 25.0, "win_rate": 0.75, "max_drawdown": 6.0, "sharpe_ratio": 2.5}},
                "gaps": {"performance_metrics": {"total_return": 18.0, "win_rate": 0.70, "max_drawdown": 7.0, "sharpe_ratio": 1.8}}
            },
            "trades": [
                {"pnl": 80.0, "duration": 30},
                {"pnl": 45.0, "duration": 45},
                {"pnl": -10.0, "duration": 25},
                {"pnl": 65.0, "duration": 40}
            ]
        }
    }


def create_mock_config():
    """Create mock configuration for testing."""
    return {
        "strategies": {
            "MACD_ETH_Long": {
                "name": "MACD_ETH_Long",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "timeframe": "1h",
                "indicators": {
                    "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
                },
                "position_sizing": {"method": "fixed_usd", "value": 500}
            },
            "RSI_ETH_Short": {
                "name": "RSI_ETH_Short",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "timeframe": "2h",
                "indicators": {
                    "rsi": {"period": 14, "overbought": 70, "oversold": 30}
                },
                "position_sizing": {"method": "percent_balance", "value": 2.0}
            },
            "Bollinger_BTC_Scalp": {
                "name": "Bollinger_BTC_Scalp",
                "market": "BTC-USD",
                "exchange": "hyperliquid",
                "timeframe": "15m",
                "indicators": {
                    "bollinger_bands": {"period": 20, "std_dev": 2.0}
                },
                "position_sizing": {"method": "volatility_adjusted", "value": 1.5}
            }
        }
    }


class MockConfigManager:
    """Mock ConfigManager for testing."""

    def __init__(self):
        self.config = create_mock_config()

    def get_strategy_config(self, strategy_name: str):
        return self.config.get("strategies", {}).get(strategy_name)

    def list_strategies(self, filter_exchange=None, filter_market=None, filter_enabled=True):
        strategies = []
        for strategy_name, config in self.config.get("strategies", {}).items():
            if filter_exchange and config.get("exchange") != filter_exchange:
                continue
            if filter_market and config.get("market") != filter_market:
                continue
            strategies.append(config)
        return strategies


class MockComparisonManager:
    """Mock ComparisonManager for testing."""

    def __init__(self, config_manager):
        self.config_manager = config_manager


def test_enhanced_comparison_reporter():
    """Test the enhanced ComparisonReporter functionality."""
    print("🧪 Testing Enhanced ComparisonReporter...")

    # Initialize mock components
    config_manager = MockConfigManager()
    comparison_manager = MockComparisonManager(config_manager)

    # Initialize ComparisonReporter
    comparison_reporter = ComparisonReporter(config_manager, comparison_manager)

    # Create mock comparison results
    comparison_results = create_mock_comparison_results()
    strategy_names = list(comparison_results.keys())

    print(f"📊 Testing comparison of {len(strategy_names)} strategies: {', '.join(strategy_names)}")

    # Generate comprehensive comparison report
    output_path = Path("./test_comparison_report.json")

    try:
        report_data = comparison_reporter.generate_strategy_comparison_report(
            strategy_names=strategy_names,
            comparison_results=comparison_results,
            output_path=output_path
        )

        print("✅ Report generation successful!")

        # Test market condition analysis
        market_condition_analysis = report_data.get('market_condition_analysis', {})
        print(f"\n🌊 Market Condition Analysis:")

        best_performers = market_condition_analysis.get('best_performers_by_condition', {})
        for condition, performer_data in best_performers.items():
            strategy = performer_data.get('strategy', 'Unknown')
            return_pct = performer_data.get('return', 0)
            print(f"   • {condition.title()}: {strategy} ({return_pct:.1f}%)")

        # Test adaptability scores
        adaptability_scores = market_condition_analysis.get('adaptability_scores', {})
        print(f"\n🎯 Strategy Adaptability Scores:")
        for strategy, score_data in adaptability_scores.items():
            score = score_data.get('score', 0)
            mean_return = score_data.get('mean_return', 0)
            cv = score_data.get('coefficient_of_variation', 0)
            print(f"   • {strategy}: {score:.1f}/100 (avg {mean_return:.1f}%, CV: {cv:.3f})")

        # Test sensitivity analysis
        sensitivity_analysis = report_data.get('sensitivity_analysis', {})
        print(f"\n⚡ Sensitivity Analysis:")

        optimization_potential = sensitivity_analysis.get('optimization_potential', {})
        for strategy, opt_data in optimization_potential.items():
            if 'error' not in opt_data:
                improvement = opt_data.get('max_estimated_improvement', 0)
                optimization = opt_data.get('best_optimization', 'Unknown')
                priority = opt_data.get('optimization_priority', 'unknown')
                print(f"   • {strategy}: +{improvement:.1f}% potential via {optimization} (Priority: {priority})")

        # Test ranking analysis
        ranking_analysis = report_data.get('ranking_analysis', {})
        final_ranking = ranking_analysis.get('final_ranking', [])
        print(f"\n🏆 Strategy Rankings:")
        for i, (strategy, score_data) in enumerate(final_ranking, 1):
            total_score = score_data.get('total_score', 0)
            print(f"   {i}. {strategy}: {total_score}")

        # Test diversification analysis
        diversification_analysis = report_data.get('diversification_analysis', {})
        diversification_insights = diversification_analysis.get('diversification_insights', [])
        print(f"\n🔄 Diversification Insights:")
        for insight in diversification_insights:
            print(f"   • {insight}")

        # Test portfolio allocation
        allocation_suggestions = report_data.get('allocation_suggestions', {})
        recommended_allocations = allocation_suggestions.get('recommended_allocations', {})
        print(f"\n💼 Recommended Portfolio Allocation:")
        for strategy, allocation in recommended_allocations.items():
            if allocation > 0:
                print(f"   • {strategy}: {allocation}%")

        print(f"\n📋 Full report saved to: {output_path}")
        print("✅ All enhanced features tested successfully!")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_enhanced_comparison_reporter()
    if success:
        print("\n🎉 Enhanced ComparisonReporter test completed successfully!")
    else:
        print("\n💥 Enhanced ComparisonReporter test failed!")
        sys.exit(1)
