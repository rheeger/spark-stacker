"""
Comparison Command Handlers

This module contains CLI commands for unified comparison functionality:
- unified-compare: Compare strategies, indicators, or mixed types
- cross-type-compare: Compare strategies against indicators
- benchmark-compare: Compare against benchmark strategies
"""

import logging
import os
from typing import List, Optional

import click

logger = logging.getLogger(__name__)


def register_comparison_commands(cli_group):
    """Register comparison-related commands with the CLI group."""
    cli_group.add_command(unified_compare)
    cli_group.add_command(cross_type_compare)
    cli_group.add_command(benchmark_compare)


@click.command("unified-compare")
@click.option("--items", required=True, help="Comma-separated list of items to compare (strategies or indicators)")
@click.option("--type", "comparison_type", type=click.Choice(['auto', 'strategy', 'indicator']),
              default='auto', help="Type of comparison (default: auto-detect)")
@click.option("--days", default=30, help="Number of days to test (default: 30)")
@click.option("--scenarios", default="all", help="Scenarios for comparison: all, bull, bear, etc.")
@click.option("--output-dir", help="Output directory for reports (default: tests/results)")
@click.option("--parallel", is_flag=True, help="Run comparisons in parallel for faster execution")
@click.option("--cache-results", is_flag=True, help="Cache comparison results for future use")
@click.pass_context
def unified_compare(ctx, items: str, comparison_type: str, days: int, scenarios: str,
                   output_dir: Optional[str], parallel: bool, cache_results: bool):
    """
    Unified comparison interface for strategies, indicators, or mixed types.

    Automatically detects the type of items being compared and applies
    appropriate comparison logic with consistent metrics and reporting.

    Examples:
        # Compare strategies
        python cli/main.py unified-compare --items "MACD_ETH_Long,RSI_ETH_Short"

        # Compare indicators (with deprecation warning)
        python cli/main.py unified-compare --items "MACD_ETH_1h,RSI_ETH_1h" --type indicator

        # Mixed comparison (advanced)
        python cli/main.py unified-compare --items "MACD_Strategy,MACD_Indicator" --type auto

        # Parallel execution with caching
        python cli/main.py unified-compare --items "Strategy1,Strategy2,Strategy3" --parallel --cache-results
    """
    # Import configuration utilities from main module
    from ..main import load_config

    # Load configuration
    config = load_config(ctx.obj.get('config_path'))
    if not config:
        raise click.ClickException("Could not load configuration. Use --config to specify config file path.")

    # Parse items to compare
    item_list = [item.strip() for item in items.split(',')]

    if len(item_list) < 2:
        raise click.ClickException("At least 2 items are required for comparison")

    # Auto-detect comparison type if needed
    if comparison_type == 'auto':
        detected_type = detect_comparison_type(config, item_list)
        comparison_type = detected_type
        click.echo(f"🔍 Auto-detected comparison type: {comparison_type}")

    # Validate items exist in configuration
    valid_items, item_types = validate_comparison_items(config, item_list, comparison_type)

    if not valid_items:
        raise click.ClickException("No valid items found for comparison")

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
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "__test_results__")

    os.makedirs(output_dir, exist_ok=True)

    # Display comparison setup
    click.echo("🔍 Unified Comparison Setup")
    click.echo("=" * 40)
    click.echo(f"📊 Comparison type: {comparison_type}")
    click.echo(f"📋 Items to compare: {len(valid_items)}")

    for i, (item_name, item_type) in enumerate(zip(valid_items, item_types), 1):
        click.echo(f"   {i}. {item_name} ({item_type})")

    click.echo(f"\n📅 Duration: {days} days")
    click.echo(f"🎲 Scenarios: {', '.join(scenario_list)}")
    click.echo(f"⚡ Parallel execution: {'Enabled' if parallel else 'Disabled'}")
    click.echo(f"💾 Result caching: {'Enabled' if cache_results else 'Disabled'}")

    # Show deprecation warning for indicator comparisons
    if comparison_type == 'indicator':
        click.echo("\n⚠️  " + "="*50, err=True)
        click.echo("⚠️  DEPRECATION WARNING: Indicator-only comparison", err=True)
        click.echo("⚠️  " + "="*50, err=True)
        click.echo("⚠️  Comparing indicators without strategy context provides", err=True)
        click.echo("⚠️  limited trading insights. Consider using strategy", err=True)
        click.echo("⚠️  comparison for more meaningful analysis.", err=True)
        click.echo("⚠️  " + "="*50, err=True)

    # TODO: Implement actual unified comparison logic
    click.echo("\n📝 Unified comparison implementation pending...")
    click.echo("   This will be integrated with:")
    click.echo("   • ComparisonManager for unified comparison logic")
    click.echo("   • Automatic type detection and routing")
    click.echo("   • Parallel execution coordination")
    click.echo("   • Result caching and reuse")
    click.echo("   • Cross-type comparison capabilities")
    click.echo(f"   • Results will be saved to: {output_dir}")

    # For now, show what would be implemented
    click.echo(f"\n✅ Unified comparison handler created successfully!")
    click.echo(f"   Items to compare: {len(valid_items)}")
    click.echo(f"   Scenarios configured: {len(scenario_list)}")
    click.echo(f"   Comparison type: {comparison_type}")


@click.command("cross-type-compare")
@click.option("--strategies", help="Comma-separated list of strategy names")
@click.option("--indicators", help="Comma-separated list of indicator names")
@click.option("--days", default=30, help="Number of days to test (default: 30)")
@click.option("--scenarios", default="all", help="Scenarios for comparison")
@click.option("--output-dir", help="Output directory for reports (default: tests/results)")
@click.option("--normalize-metrics", is_flag=True, help="Normalize metrics for fair cross-type comparison")
@click.pass_context
def cross_type_compare(ctx, strategies: Optional[str], indicators: Optional[str], days: int,
                      scenarios: str, output_dir: Optional[str], normalize_metrics: bool):
    """
    Compare strategies against indicators (advanced comparison).

    This advanced comparison normalizes different types of performance metrics
    to enable meaningful comparison between full strategies and individual indicators.

    Examples:
        # Compare strategies vs indicators
        python cli/main.py cross-type-compare --strategies "MACD_ETH_Long" --indicators "MACD_ETH_1h"

        # Multiple items comparison
        python cli/main.py cross-type-compare --strategies "Strategy1,Strategy2" --indicators "Indicator1,Indicator2"

        # Normalized metrics for fair comparison
        python cli/main.py cross-type-compare --strategies "MACD_Strategy" --indicators "MACD_Indicator" --normalize-metrics
    """
    if not strategies and not indicators:
        raise click.ClickException("Must specify either --strategies or --indicators (or both)")

    # Import configuration utilities from main module
    from ..main import load_config

    # Load configuration
    config = load_config(ctx.obj.get('config_path'))
    if not config:
        raise click.ClickException("Could not load configuration. Use --config to specify config file path.")

    # Parse and validate items
    strategy_list = []
    indicator_list = []

    if strategies:
        strategy_names = [name.strip() for name in strategies.split(',')]
        for name in strategy_names:
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

            strategy_list.append(name)

    if indicators:
        indicator_names = [name.strip() for name in indicators.split(',')]
        for name in indicator_names:
            indicator_config = None
            for i in config.get('indicators', []):
                if i.get('name') == name:
                    indicator_config = i
                    break

            if not indicator_config:
                available_indicators = [i.get('name', 'unnamed') for i in config.get('indicators', [])]
                raise click.ClickException(
                    f"Indicator '{name}' not found in configuration.\n"
                    f"Available indicators: {', '.join(available_indicators)}"
                )

            indicator_list.append(name)

    # Parse scenarios
    if scenarios.lower() == "all":
        scenario_list = ["bull", "bear", "sideways", "volatile", "low-vol", "choppy", "gaps", "real"]
    else:
        scenario_list = [s.strip() for s in scenarios.split(",")]

    # Determine output directory
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "__test_results__")

    os.makedirs(output_dir, exist_ok=True)

    # Display comparison setup
    click.echo("🔄 Cross-Type Comparison Setup")
    click.echo("=" * 40)

    if strategy_list:
        click.echo(f"📊 Strategies ({len(strategy_list)}):")
        for i, strategy in enumerate(strategy_list, 1):
            click.echo(f"   {i}. {strategy}")

    if indicator_list:
        click.echo(f"📈 Indicators ({len(indicator_list)}):")
        for i, indicator in enumerate(indicator_list, 1):
            click.echo(f"   {i}. {indicator}")

    click.echo(f"\n📅 Duration: {days} days")
    click.echo(f"🎲 Scenarios: {', '.join(scenario_list)}")
    click.echo(f"⚖️  Normalize metrics: {'Enabled' if normalize_metrics else 'Disabled'}")

    # Show methodology explanation
    click.echo("\n📋 Cross-Type Comparison Methodology:")
    click.echo("   • Strategies: Full backtesting with position sizing and risk management")
    click.echo("   • Indicators: Signal-based analysis with normalized scoring")
    if normalize_metrics:
        click.echo("   • Metrics normalization: Risk-adjusted returns, Sharpe ratio equivalents")
        click.echo("   • Fair comparison: Account for different risk profiles")

    # Show warning about methodology
    click.echo("\n⚠️  Important Notes:")
    click.echo("   • Cross-type comparison has inherent limitations")
    click.echo("   • Strategy performance includes position sizing and risk management")
    click.echo("   • Indicator performance is signal-based without trading context")
    click.echo("   • Results should be interpreted with caution")

    # TODO: Implement actual cross-type comparison logic
    click.echo("\n📝 Cross-type comparison implementation pending...")
    click.echo("   This will include:")
    click.echo("   • Separate execution pipelines for strategies vs indicators")
    click.echo("   • Metric normalization and risk adjustment")
    click.echo("   • Statistical significance testing")
    click.echo("   • Fair comparison methodologies")
    click.echo(f"   • Results will be saved to: {output_dir}")

    # For now, show what would be implemented
    click.echo(f"\n✅ Cross-type comparison handler created successfully!")
    click.echo(f"   Strategies: {len(strategy_list)}")
    click.echo(f"   Indicators: {len(indicator_list)}")
    click.echo(f"   Scenarios: {len(scenario_list)}")


@click.command("benchmark-compare")
@click.option("--items", required=True, help="Comma-separated list of items to compare against benchmarks")
@click.option("--benchmark-type", type=click.Choice(['market', 'popular', 'custom', 'all']),
              default='market', help="Type of benchmark comparison (default: market)")
@click.option("--custom-benchmarks", help="Comma-separated list of custom benchmark names")
@click.option("--days", default=30, help="Number of days to test (default: 30)")
@click.option("--scenarios", default="all", help="Scenarios for comparison")
@click.option("--output-dir", help="Output directory for reports (default: tests/results)")
@click.pass_context
def benchmark_compare(ctx, items: str, benchmark_type: str, custom_benchmarks: Optional[str],
                     days: int, scenarios: str, output_dir: Optional[str]):
    """
    Compare strategies/indicators against benchmark strategies.

    Provides comparison against market benchmarks, popular strategies,
    or custom benchmark configurations for performance evaluation.

    Examples:
        # Compare against market benchmarks
        python cli/main.py benchmark-compare --items "MyStrategy" --benchmark-type market

        # Compare against popular strategies
        python cli/main.py benchmark-compare --items "MyStrategy" --benchmark-type popular

        # Compare against custom benchmarks
        python cli/main.py benchmark-compare --items "MyStrategy" --benchmark-type custom --custom-benchmarks "Benchmark1,Benchmark2"

        # Compare against all benchmark types
        python cli/main.py benchmark-compare --items "MyStrategy" --benchmark-type all
    """
    # Import configuration utilities from main module
    from ..main import load_config

    # Load configuration
    config = load_config(ctx.obj.get('config_path'))
    if not config:
        raise click.ClickException("Could not load configuration. Use --config to specify config file path.")

    # Parse items to compare
    item_list = [item.strip() for item in items.split(',')]

    # Validate items exist in configuration
    valid_items = []
    for item in item_list:
        # Check if it's a strategy
        strategy_found = any(s.get('name') == item for s in config.get('strategies', []))
        # Check if it's an indicator
        indicator_found = any(i.get('name') == item for i in config.get('indicators', []))

        if strategy_found or indicator_found:
            valid_items.append(item)
        else:
            click.echo(f"⚠️  Item '{item}' not found in configuration (skipping)")

    if not valid_items:
        raise click.ClickException("No valid items found for benchmark comparison")

    # Determine benchmarks based on type
    benchmarks = []

    if benchmark_type in ['market', 'all']:
        # Market benchmarks (buy-and-hold, DCA, etc.)
        market_benchmarks = [
            'Buy_and_Hold_ETH',
            'DCA_Daily_ETH',
            'Random_Strategy_ETH'
        ]
        benchmarks.extend(market_benchmarks)
        click.echo(f"📊 Market benchmarks: {', '.join(market_benchmarks)}")

    if benchmark_type in ['popular', 'all']:
        # Popular strategy benchmarks from config
        strategies = config.get('strategies', [])
        popular_strategies = []

        # Define criteria for "popular" (you can adjust this)
        for strategy in strategies:
            strategy_name = strategy.get('name', '')
            # Consider strategies with common indicators as "popular"
            if any(indicator in strategy_name.upper() for indicator in ['MACD', 'RSI', 'EMA', 'SMA']):
                popular_strategies.append(strategy_name)

        # Limit to top 3 popular strategies
        popular_strategies = popular_strategies[:3]
        benchmarks.extend(popular_strategies)

        if popular_strategies:
            click.echo(f"🌟 Popular benchmarks: {', '.join(popular_strategies)}")
        else:
            click.echo("🌟 No popular benchmarks found in configuration")

    if benchmark_type in ['custom', 'all']:
        if custom_benchmarks:
            custom_list = [name.strip() for name in custom_benchmarks.split(',')]

            # Validate custom benchmarks exist
            valid_custom = []
            for custom in custom_list:
                if any(s.get('name') == custom for s in config.get('strategies', [])):
                    valid_custom.append(custom)
                else:
                    click.echo(f"⚠️  Custom benchmark '{custom}' not found in configuration")

            benchmarks.extend(valid_custom)
            if valid_custom:
                click.echo(f"🎯 Custom benchmarks: {', '.join(valid_custom)}")
        else:
            click.echo("🎯 No custom benchmarks specified")

    if not benchmarks:
        raise click.ClickException("No valid benchmarks found for comparison")

    # Parse scenarios
    if scenarios.lower() == "all":
        scenario_list = ["bull", "bear", "sideways", "volatile", "low-vol", "choppy", "gaps", "real"]
    else:
        scenario_list = [s.strip() for s in scenarios.split(",")]

    # Determine output directory
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "__test_results__")

    os.makedirs(output_dir, exist_ok=True)

    # Display comparison setup
    click.echo("\n📊 Benchmark Comparison Setup")
    click.echo("=" * 40)
    click.echo(f"🎯 Items to test ({len(valid_items)}):")
    for i, item in enumerate(valid_items, 1):
        click.echo(f"   {i}. {item}")

    click.echo(f"\n📊 Benchmarks ({len(benchmarks)}):")
    for i, benchmark in enumerate(benchmarks, 1):
        click.echo(f"   {i}. {benchmark}")

    click.echo(f"\n📅 Duration: {days} days")
    click.echo(f"🎲 Scenarios: {', '.join(scenario_list)}")
    click.echo(f"📈 Benchmark type: {benchmark_type}")

    # Show comparison methodology
    click.echo("\n📋 Benchmark Comparison Methodology:")
    click.echo("   • All items tested under identical conditions")
    click.echo("   • Same market data and time periods")
    click.echo("   • Consistent risk management parameters")
    click.echo("   • Statistical significance testing")
    click.echo("   • Risk-adjusted performance metrics")

    # TODO: Implement actual benchmark comparison logic
    click.echo("\n📝 Benchmark comparison implementation pending...")
    click.echo("   This will include:")
    click.echo("   • Automatic benchmark strategy generation")
    click.echo("   • Parallel execution of items and benchmarks")
    click.echo("   • Statistical significance testing")
    click.echo("   • Risk-adjusted performance comparison")
    click.echo("   • Ranking and scoring against benchmarks")
    click.echo(f"   • Results will be saved to: {output_dir}")

    # For now, show what would be implemented
    click.echo(f"\n✅ Benchmark comparison handler created successfully!")
    click.echo(f"   Items to test: {len(valid_items)}")
    click.echo(f"   Benchmarks: {len(benchmarks)}")
    click.echo(f"   Total comparisons: {len(valid_items) * len(benchmarks)}")


def detect_comparison_type(config: dict, item_list: List[str]) -> str:
    """
    Auto-detect the type of comparison based on the items provided.

    Args:
        config: Configuration dictionary
        item_list: List of item names to compare

    Returns:
        Detected comparison type: 'strategy', 'indicator', or 'mixed'
    """
    strategies = {s.get('name') for s in config.get('strategies', [])}
    indicators = {i.get('name') for i in config.get('indicators', [])}

    strategy_count = sum(1 for item in item_list if item in strategies)
    indicator_count = sum(1 for item in item_list if item in indicators)

    if strategy_count > 0 and indicator_count == 0:
        return 'strategy'
    elif indicator_count > 0 and strategy_count == 0:
        return 'indicator'
    elif strategy_count > 0 and indicator_count > 0:
        return 'mixed'
    else:
        return 'unknown'


def validate_comparison_items(config: dict, item_list: List[str], comparison_type: str) -> tuple:
    """
    Validate that comparison items exist in configuration.

    Args:
        config: Configuration dictionary
        item_list: List of item names to validate
        comparison_type: Type of comparison ('strategy', 'indicator', 'mixed')

    Returns:
        Tuple of (valid_items, item_types)
    """
    strategies = {s.get('name'): s for s in config.get('strategies', [])}
    indicators = {i.get('name'): i for i in config.get('indicators', [])}

    valid_items = []
    item_types = []

    for item in item_list:
        if item in strategies:
            valid_items.append(item)
            item_types.append('strategy')
        elif item in indicators:
            valid_items.append(item)
            item_types.append('indicator')
        else:
            logger.warning(f"Item '{item}' not found in configuration")

    return valid_items, item_types
