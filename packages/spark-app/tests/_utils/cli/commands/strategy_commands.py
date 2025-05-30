"""
Strategy Command Handlers

This module contains CLI commands for strategy backtesting and management:
- strategy: Run backtest for a specific strategy from config
- compare-strategies: Compare multiple strategies with multi-scenario testing
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

import click

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
    # Import configuration loading from main module
    from ..main import (get_strategy_config, load_config,
                        validate_strategy_config)

    # Load configuration
    config = load_config(ctx.obj.get('config_path'))
    if not config:
        raise click.ClickException("Could not load configuration. Use --config to specify config file path.")

    # Validate strategy exists
    strategy_config = get_strategy_config(config, strategy_name)
    if not strategy_config:
        available_strategies = [s.get('name', 'unnamed') for s in config.get('strategies', [])]
        raise click.ClickException(
            f"Strategy '{strategy_name}' not found in configuration.\n"
            f"Available strategies: {', '.join(available_strategies)}"
        )

    # Validate strategy configuration
    validation_errors = validate_strategy_config(config, strategy_name)
    if validation_errors:
        raise click.ClickException(
            f"Strategy configuration validation failed:\n" +
            "\n".join([f"  • {error}" for error in validation_errors])
        )

    # Apply overrides if specified
    effective_config = strategy_config.copy()
    if override_timeframe:
        effective_config['timeframe'] = override_timeframe
        click.echo(f"⚠️  Overriding timeframe: {strategy_config.get('timeframe')} → {override_timeframe}")

    if override_market:
        effective_config['market'] = override_market
        click.echo(f"⚠️  Overriding market: {strategy_config.get('market')} → {override_market}")

    if override_position_size:
        effective_config['max_position_size'] = override_position_size
        click.echo(f"⚠️  Overriding position size: {strategy_config.get('max_position_size')} → {override_position_size}")

    # Determine output directory
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")

    os.makedirs(output_dir, exist_ok=True)

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

    click.echo(f"🎯 Running backtest for strategy: {strategy_name}")
    click.echo(f"📊 Market: {effective_config.get('market', 'Unknown')}")
    click.echo(f"⏱️  Timeframe: {effective_config.get('timeframe', 'Unknown')}")
    click.echo(f"📅 Duration: {days} days")
    click.echo(f"🎲 Scenarios: {', '.join(scenario_list)}")

    if scenario_only and len(scenario_list) > 1:
        click.echo("⚠️  --scenario-only flag specified but multiple scenarios selected. Running all specified scenarios.")

    # TODO: Implement actual strategy backtesting logic
    # This will be integrated with the StrategyBacktestManager in subsequent tasks
    click.echo("\n📝 Strategy backtesting implementation pending...")
    click.echo("   This will be implemented in subsequent tasks with:")
    click.echo("   • StrategyBacktestManager integration")
    click.echo("   • ScenarioManager for multi-scenario testing")
    click.echo("   • StrategyReporter for comprehensive reporting")
    click.echo(f"   • Results will be saved to: {output_dir}")

    if export_data:
        click.echo(f"📁 Scenario data export enabled - will save to: {output_dir}/scenario_data/")

    # For now, show what would be implemented
    click.echo(f"\n✅ Strategy command handler created successfully!")
    click.echo(f"   Strategy: {strategy_name}")
    click.echo(f"   Configuration validated: ✅")
    click.echo(f"   Scenarios configured: {len(scenario_list)}")


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
    # Import configuration loading from main module
    from ..main import list_strategies, load_config

    # Load configuration
    config = load_config(ctx.obj.get('config_path'))
    if not config:
        raise click.ClickException("Could not load configuration. Use --config to specify config file path.")

    # Determine strategies to compare
    if all_strategies:
        strategies_to_compare = list_strategies(
            config,
            filter_exchange=same_exchange,
            filter_market=same_market,
            filter_enabled=True
        )
    elif strategy_names:
        strategy_name_list = [name.strip() for name in strategy_names.split(",")]
        strategies_to_compare = []

        for name in strategy_name_list:
            strategy_config = None
            for s in config.get('strategies', []):
                if s.get('name') == name:
                    strategy_config = s
                    break

            if not strategy_config:
                available_strategies = [s.get('name', 'unnamed') for s in config.get('strategies', [])]
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

    # Determine output directory
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")

    os.makedirs(output_dir, exist_ok=True)

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

    # TODO: Implement actual strategy comparison logic
    # This will be integrated with the ComparisonManager in subsequent tasks
    click.echo("\n📝 Strategy comparison implementation pending...")
    click.echo("   This will be implemented in subsequent tasks with:")
    click.echo("   • ComparisonManager for unified comparison logic")
    click.echo("   • ScenarioManager for consistent scenario testing")
    click.echo("   • ComparisonReporter for side-by-side analysis")
    click.echo("   • Statistical significance testing")
    click.echo("   • Cross-scenario robustness scoring")
    click.echo(f"   • Results will be saved to: {output_dir}")

    # For now, show what would be implemented
    click.echo(f"\n✅ Strategy comparison handler created successfully!")
    click.echo(f"   Strategies to compare: {len(strategies_to_compare)}")
    click.echo(f"   Scenarios configured: {len(scenario_list)}")
    click.echo(f"   Fair comparison across all scenarios: ✅")
