"""
Strategy Command Handlers

This module contains CLI commands for strategy backtesting and management:
- strategy: Run backtest for a specific strategy from config
- compare-strategies: Compare multiple strategies with multi-scenario testing
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click

# Add proper path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
cli_dir = os.path.dirname(current_dir)
utils_dir = os.path.dirname(cli_dir)
tests_dir = os.path.dirname(utils_dir)
spark_app_dir = os.path.dirname(tests_dir)
sys.path.insert(0, spark_app_dir)

# Import app components
from app.backtesting.backtest_engine import BacktestEngine

# Add CLI directory to path for CLI module imports
sys.path.insert(0, cli_dir)

# Import required managers and core modules (now using absolute paths)
from core.config_manager import ConfigManager
from core.data_manager import DataManager
from core.scenario_manager import ScenarioManager
from managers.comparison_manager import ComparisonManager
from managers.scenario_backtest_manager import ScenarioBacktestManager
from managers.strategy_backtest_manager import StrategyBacktestManager
from reporting.comparison_reporter import ComparisonReporter
from reporting.scenario_reporter import ScenarioReporter
from reporting.strategy_reporter import StrategyReporter
from validation.strategy_validator import StrategyValidator

logger = logging.getLogger(__name__)


def register_strategy_commands(cli_group):
    """Register strategy-related commands with the CLI group."""
    cli_group.add_command(strategy)
    cli_group.add_command(compare_strategies)


@click.command("strategy")
@click.option("--strategy-name", required=True, help="Name of strategy from config.json")
@click.option("--days", default=30, help="Number of days to test (default: 30)")
@click.option("--scenarios", default="all", help="Scenarios to run: all, bull, bear, sideways, volatile, low-vol, choppy, gaps, real, or comma-separated list")
@click.option("--scenario-only", is_flag=True, help="Run single scenario instead of full suite")
@click.option("--override-timeframe", help="Override strategy timeframe for testing")
@click.option("--override-market", help="Override strategy market for testing")
@click.option("--override-position-size", type=float, help="Override position sizing for testing")
@click.option("--use-real-data", is_flag=True, help="Use real data instead of synthetic (legacy compatibility)")
@click.option("--export-data", is_flag=True, help="Export scenario data for external analysis")
@click.option("--output-dir", help="Output directory for reports (default: tests/results)")
@click.pass_context
def strategy(ctx, strategy_name: str, days: int, scenarios: str, scenario_only: bool,
             override_timeframe: Optional[str], override_market: Optional[str],
             override_position_size: Optional[float], use_real_data: bool,
             export_data: bool, output_dir: Optional[str]):
    """
    Run backtest for a specific strategy from config.json.

    By default, runs multi-scenario testing (all 7 synthetic scenarios + real data)
    to evaluate strategy robustness across different market conditions.

    Examples:
        # Run full multi-scenario test for a strategy
        python cli/main.py strategy --strategy-name "MACD_ETH_Long"

        # Test specific scenarios only
        python cli/main.py strategy --strategy-name "MACD_ETH_Long" --scenarios "bull,bear,real"

        # Run single scenario with overrides
        python cli/main.py strategy --strategy-name "MACD_ETH_Long" --scenario-only --scenarios "bull" --override-timeframe "4h"
    """
    try:
        # Initialize managers and components
        config_manager = ConfigManager(ctx.obj.get('config_path'))
        data_manager = DataManager()
        strategy_validator = StrategyValidator(config_manager)
        backtest_engine = BacktestEngine()

        # Determine output directory
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize strategy backtest manager
        strategy_manager = StrategyBacktestManager(
            backtest_engine=backtest_engine,
            config_manager=config_manager,
            data_manager=data_manager,
            strategy_validator=strategy_validator,
            output_dir=output_path
        )

        # Load strategy configuration
        click.echo(f"🔍 Loading strategy: {strategy_name}")
        strategy_config = strategy_manager.load_strategy_from_config(strategy_name)

        # Build configuration overrides
        overrides = {}
        position_sizing_overrides = {}

        if override_timeframe:
            overrides['timeframe'] = override_timeframe
            click.echo(f"⚠️  Overriding timeframe: {strategy_config.timeframe} → {override_timeframe}")

        if override_market:
            overrides['market'] = override_market
            click.echo(f"⚠️  Overriding market: {strategy_config.market} → {override_market}")

        if override_position_size:
            # Handle position sizing override more intelligently
            current_position_info = strategy_manager.get_current_position_sizing_info()
            if 'error' in current_position_info:
                click.echo(f"⚠️  Warning: Could not get current position sizing info: {current_position_info['error']}")

            # Determine how to apply the override based on current method
            method = current_position_info.get('method', 'fixed_usd')

            if method == 'fixed_usd':
                position_sizing_overrides['fixed_usd_amount'] = override_position_size
                click.echo(f"⚠️  Overriding fixed USD amount: ${current_position_info.get('fixed_usd_amount', 'unknown')} → ${override_position_size}")
            elif method == 'percentage_equity':
                position_sizing_overrides['equity_percentage'] = override_position_size / 100.0
                click.echo(f"⚠️  Overriding equity percentage: {current_position_info.get('equity_percentage', 'unknown')*100:.1f}% → {override_position_size}%")
            else:
                # For other methods, override max position size
                position_sizing_overrides['max_position_size_usd'] = override_position_size
                click.echo(f"⚠️  Overriding max position size: ${current_position_info.get('max_position_size_usd', 'unknown')} → ${override_position_size}")

        # Initialize strategy components with overrides
        click.echo("🔧 Initializing strategy components...")
        strategy_manager.initialize_strategy_components(overrides)

        # Apply position sizing overrides after initialization
        if position_sizing_overrides:
            click.echo("💰 Applying position sizing overrides...")
            strategy_manager.apply_position_sizing_overrides(position_sizing_overrides)

            # Display final position sizing configuration
            final_position_info = strategy_manager.get_current_position_sizing_info()
            if 'error' not in final_position_info:
                click.echo(f"💰 Position sizing method: {final_position_info['method']}")
                if final_position_info['method'] == 'fixed_usd':
                    click.echo(f"💰 Fixed USD amount: ${final_position_info['fixed_usd_amount']}")
                elif final_position_info['method'] == 'percentage_equity':
                    click.echo(f"💰 Equity percentage: {final_position_info['equity_percentage']*100:.1f}%")
                click.echo(f"💰 Max position size: ${final_position_info['max_position_size_usd']}")

        click.echo(f"🎯 Running backtest for strategy: {strategy_name}")
        click.echo(f"📊 Market: {strategy_config.market}")
        click.echo(f"⏱️  Timeframe: {strategy_config.timeframe}")
        click.echo(f"📅 Duration: {days} days")

        # Parse scenarios
        if scenarios.lower() == "all":
            scenario_list = ["bull", "bear", "sideways", "volatile", "low-vol", "choppy", "gaps", "real"]
        else:
            scenario_list = [s.strip() for s in scenarios.split(",")]

        # Validate scenario names
        valid_scenarios = ["bull", "bear", "sideways", "volatile", "low-vol", "choppy", "gaps", "real"]
        invalid_scenarios = [s for s in scenario_list if s not in valid_scenarios]
        if invalid_scenarios:
            raise click.ClickException(
                f"Invalid scenarios: {', '.join(invalid_scenarios)}\n"
                f"Valid scenarios: {', '.join(valid_scenarios)}"
            )

        click.echo(f"🎲 Scenarios: {', '.join(scenario_list)}")

        # Handle legacy use-real-data flag
        if use_real_data and "real" not in scenario_list:
            scenario_list = ["real"]
            click.echo("⚠️  Legacy --use-real-data flag detected, running real data scenario only")

        # Check for multi-scenario vs single scenario
        if scenario_only and len(scenario_list) > 1:
            click.echo("⚠️  --scenario-only flag specified but multiple scenarios selected. Running all specified scenarios.")

        # Run backtesting based on scenario configuration
        if len(scenario_list) == 1 and scenario_list[0] == "real":
            # Single real data backtest (legacy compatibility)
            click.echo("\n🚀 Running single real data backtest...")
            result = strategy_manager.backtest_strategy(
                days=days,
                use_real_data=True,
                leverage=1.0
            )

            # Generate strategy report
            strategy_reporter = StrategyReporter(config_manager, output_path)
            report_path = strategy_reporter.generate_strategy_report(
                strategy_name=strategy_name,
                backtest_result=result,
                output_format='html'
            )

            click.echo(f"✅ Backtest completed successfully!")
            click.echo(f"📊 Total trades: {result.metrics.get('total_trades', 0)}")
            click.echo(f"💰 Total return: {result.metrics.get('total_return_pct', 0):.2f}%")
            click.echo(f"📈 Win rate: {result.metrics.get('win_rate', 0):.1f}%")
            click.echo(f"📋 Report saved to: {report_path}")

        else:
            # Multi-scenario testing
            click.echo(f"\n🚀 Running multi-scenario backtest across {len(scenario_list)} scenarios...")

            # Initialize scenario manager and scenario backtest manager
            scenario_manager = ScenarioManager(
                data_manager=data_manager,
                config_manager=config_manager,
                default_duration_days=days
            )

            scenario_backtest_manager = ScenarioBacktestManager(
                config_manager=config_manager,
                data_manager=data_manager,
                scenario_manager=scenario_manager,
                strategy_manager=strategy_manager,
                output_dir=output_path
            )

            # Run scenario testing
            scenario_results = scenario_backtest_manager.run_strategy_scenarios(
                strategy_name=strategy_name,
                days=days,
                scenario_filter=scenario_list
            )

            # Generate scenario report
            scenario_reporter = ScenarioReporter(config_manager, data_manager, output_path)
            report_path = scenario_reporter.generate_scenario_report(
                strategy_name=strategy_name,
                scenario_results=scenario_results,
                output_format='html'
            )

            # Display summary
            click.echo(f"✅ Multi-scenario backtest completed!")
            click.echo(f"📊 Scenarios tested: {len(scenario_results)}")

            # Show key metrics across scenarios
            for scenario_name, result in scenario_results.items():
                if hasattr(result, 'backtest_result'):
                    metrics = result.backtest_result.metrics
                    click.echo(f"  {scenario_name}: {metrics.get('total_return_pct', 0):.1f}% return, {metrics.get('total_trades', 0)} trades")
                elif hasattr(result, 'total_return'):
                    click.echo(f"  {scenario_name}: {result.total_return:.1f}% return, {result.total_trades} trades")

            click.echo(f"📋 Comprehensive report saved to: {report_path}")

            # Calculate and display robustness score
            total_returns = []
            for result in scenario_results.values():
                if hasattr(result, 'backtest_result'):
                    total_returns.append(result.backtest_result.metrics.get('total_return_pct', 0))
                elif hasattr(result, 'total_return'):
                    total_returns.append(result.total_return)

            if total_returns:
                avg_return = sum(total_returns) / len(total_returns)
                return_variance = sum((r - avg_return) ** 2 for r in total_returns) / len(total_returns) if len(total_returns) > 1 else 0
                consistency_score = max(0, 100 - (return_variance / (avg_return ** 2) * 100)) if avg_return != 0 else 0
                click.echo(f"🎯 Strategy robustness score: {consistency_score:.1f}% (higher is better)")

        if export_data:
            click.echo(f"📁 Scenario data exported to: {output_path}/scenario_data/")

    except Exception as e:
        logger.error(f"Strategy backtesting failed: {e}")
        raise click.ClickException(f"Strategy backtesting failed: {str(e)}")


@click.command("compare-strategies")
@click.option("--strategy-names", help="Comma-separated list of strategy names to compare")
@click.option("--all-strategies", is_flag=True, help="Compare all enabled strategies")
@click.option("--same-market", help="Filter strategies to specific market (e.g., ETH-USD)")
@click.option("--same-exchange", help="Filter strategies to specific exchange")
@click.option("--days", default=30, help="Number of days to test (default: 30)")
@click.option("--scenarios", default="all", help="Scenarios for comparison: all, bull, bear, etc.")
@click.option("--output-dir", help="Output directory for reports (default: tests/results)")
@click.pass_context
def compare_strategies(ctx, strategy_names: Optional[str], all_strategies: bool,
                      same_market: Optional[str], same_exchange: Optional[str],
                      days: int, scenarios: str, output_dir: Optional[str]):
    """
    Compare multiple strategies with multi-scenario testing.

    Runs all strategies through the same set of market scenarios for fair comparison,
    including cross-scenario robustness scoring.

    Examples:
        # Compare specific strategies
        python cli/main.py compare-strategies --strategy-names "MACD_ETH_Long,RSI_ETH_Short"

        # Compare all enabled strategies
        python cli/main.py compare-strategies --all-strategies

        # Compare strategies on same market
        python cli/main.py compare-strategies --all-strategies --same-market "ETH-USD"

        # Compare with specific scenarios
        python cli/main.py compare-strategies --all-strategies --scenarios "bull,bear,real"
    """
    try:
        # Initialize managers and components
        config_manager = ConfigManager(ctx.obj.get('config_path'))
        data_manager = DataManager()

        # Determine output directory
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine strategies to compare
        if all_strategies:
            strategies_to_compare = config_manager.list_strategies(
                filter_exchange=same_exchange,
                filter_market=same_market,
                filter_enabled=True
            )
        elif strategy_names:
            strategy_name_list = [name.strip() for name in strategy_names.split(",")]
            strategies_to_compare = []

            for name in strategy_name_list:
                strategy_config = config_manager.get_strategy_config(name)
                if not strategy_config:
                    available_strategies = [s.get('name', 'unnamed') for s in config_manager.list_strategies()]
                    raise click.ClickException(
                        f"Strategy '{name}' not found in configuration.\n"
                        f"Available strategies: {', '.join(available_strategies)}"
                    )

                # Apply filters if specified
                if same_market and strategy_config.get('market', '').upper() != same_market.upper():
                    continue
                if same_exchange and strategy_config.get('exchange', '').lower() != same_exchange.lower():
                    continue

                strategies_to_compare.append(strategy_config)
        else:
            raise click.ClickException("Must specify either --strategy-names or --all-strategies")

        if not strategies_to_compare:
            raise click.ClickException("No strategies found matching the specified criteria")

        # Parse scenarios
        if scenarios.lower() == "all":
            scenario_list = ["bull", "bear", "sideways", "volatile", "low-vol", "choppy", "gaps", "real"]
        else:
            scenario_list = [s.strip() for s in scenarios.split(",")]

        # Validate scenario names
        valid_scenarios = ["bull", "bear", "sideways", "volatile", "low-vol", "choppy", "gaps", "real"]
        invalid_scenarios = [s for s in scenario_list if s not in valid_scenarios]
        if invalid_scenarios:
            raise click.ClickException(
                f"Invalid scenarios: {', '.join(invalid_scenarios)}\n"
                f"Valid scenarios: {', '.join(valid_scenarios)}"
            )

        # Display comparison setup
        click.echo(f"🔍 Comparing {len(strategies_to_compare)} strategies:")
        for i, strategy in enumerate(strategies_to_compare, 1):
            click.echo(f"   {i}. {strategy.get('name', 'unnamed')} ({strategy.get('market', 'unknown')}, {strategy.get('exchange', 'unknown')})")

        click.echo(f"\n📅 Duration: {days} days")
        click.echo(f"🎲 Scenarios: {', '.join(scenario_list)}")

        if same_market:
            click.echo(f"🎯 Market filter: {same_market}")
        if same_exchange:
            click.echo(f"🏢 Exchange filter: {same_exchange}")

        # Initialize comparison manager
        comparison_manager = ComparisonManager(
            config_manager=config_manager,
            data_manager=data_manager,
            output_dir=output_path
        )

        # Run strategy comparison
        click.echo(f"\n🚀 Running strategy comparison across {len(scenario_list)} scenarios...")
        strategy_names_list = [s['name'] for s in strategies_to_compare]

        comparison_results = comparison_manager.run_strategy_comparison(
            strategy_names=strategy_names_list,
            scenarios=scenario_list,
            days=days
        )

        # Generate comparison report
        comparison_reporter = ComparisonReporter(config_manager, data_manager, output_path)
        report_path = comparison_reporter.generate_comparison_report(
            comparison_results=comparison_results,
            output_format='html'
        )

        # Display summary results
        click.echo(f"✅ Strategy comparison completed!")
        click.echo(f"📊 Strategies compared: {len(comparison_results.strategy_results)}")
        click.echo(f"🎲 Scenarios tested: {len(scenario_list)}")

        # Show ranking summary
        click.echo(f"\n🏆 Strategy Rankings (by average return):")
        sorted_strategies = sorted(
            comparison_results.strategy_results.items(),
            key=lambda x: x[1].overall_metrics.get('average_return_pct', 0),
            reverse=True
        )

        for i, (strategy_name, results) in enumerate(sorted_strategies[:5], 1):  # Top 5
            avg_return = results.overall_metrics.get('average_return_pct', 0)
            consistency = results.overall_metrics.get('consistency_score', 0)
            click.echo(f"   {i}. {strategy_name}: {avg_return:.1f}% avg return, {consistency:.1f}% consistency")

        click.echo(f"\n📋 Detailed comparison report saved to: {report_path}")

    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        raise click.ClickException(f"Strategy comparison failed: {str(e)}")
