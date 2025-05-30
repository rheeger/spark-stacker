"""
Indicator Command Handlers (Legacy)

This module contains CLI commands for legacy indicator backtesting:
- demo: Demo indicator backtesting with synthetic data
- real-data: Indicator backtesting with real market data
- compare: Compare multiple indicators
- compare-popular: Compare popular indicators

⚠️  DEPRECATION WARNING: These commands are maintained for backward compatibility.
New functionality should use strategy-based commands instead.
"""

import logging
import os
from typing import Optional

import click

logger = logging.getLogger(__name__)


def register_indicator_commands(cli_group):
    """Register legacy indicator-related commands with the CLI group."""
    cli_group.add_command(demo)
    cli_group.add_command(real_data)
    cli_group.add_command(compare)
    cli_group.add_command(compare_popular)


def show_deprecation_warning(command_name: str, suggested_strategy: Optional[str] = None):
    """Show deprecation warning for legacy indicator commands."""
    click.echo("⚠️  " + "="*70, err=True)
    click.echo("⚠️  DEPRECATION WARNING", err=True)
    click.echo("⚠️  " + "="*70, err=True)
    click.echo(f"⚠️  The '{command_name}' command is deprecated.", err=True)
    click.echo("⚠️  ", err=True)
    click.echo("⚠️  Indicator-only testing has limited value compared to full", err=True)
    click.echo("⚠️  strategy backtesting which includes position sizing, risk", err=True)
    click.echo("⚠️  management, and multi-scenario testing.", err=True)
    click.echo("⚠️  ", err=True)
    click.echo("⚠️  Recommended alternatives:", err=True)
    click.echo("⚠️    • Use 'strategy --strategy-name <name>' for full strategy tests", err=True)
    click.echo("⚠️    • Use 'compare-strategies' for strategy comparisons", err=True)
    click.echo("⚠️    • Use 'list-strategies' to see available strategies", err=True)

    if suggested_strategy:
        click.echo("⚠️  ", err=True)
        click.echo(f"⚠️  Suggested strategy: {suggested_strategy}", err=True)

    click.echo("⚠️  " + "="*70, err=True)
    click.echo()


@click.command("demo")
@click.option("--indicator", default="MACD", help="Indicator type to test (default: MACD)")
@click.option("--symbol", default="ETHUSDT", help="Trading symbol (default: ETHUSDT)")
@click.option("--timeframe", default="1h", help="Timeframe (default: 1h)")
@click.option("--days", default=30, type=int, help="Number of days of synthetic data (default: 30)")
@click.option("--output-dir", help="Output directory for reports (default: tests/results)")
@click.option("--suggest-strategy", is_flag=True, help="Show strategy suggestions for this indicator")
@click.pass_context
def demo(ctx, indicator: str, symbol: str, timeframe: str, days: int,
         output_dir: Optional[str], suggest_strategy: bool):
    """
    Demo indicator backtesting with synthetic data.

    ⚠️  DEPRECATED: Use strategy-based backtesting instead for comprehensive testing.

    This command demonstrates indicator behavior with synthetic market data,
    but does not include position sizing, risk management, or realistic trading scenarios.

    Examples:
        # Basic demo
        python cli/main.py demo

        # Demo with specific indicator
        python cli/main.py demo --indicator RSI --timeframe 4h

        # Show strategy suggestions
        python cli/main.py demo --suggest-strategy
    """
    show_deprecation_warning("demo")

    # Load configuration to suggest strategies
    config = None
    try:
        from ..main import load_config
        config = load_config(ctx.obj.get('config_path'))
    except Exception as e:
        logger.warning(f"Could not load configuration: {e}")

    # Show strategy suggestions if requested or if config is available
    if suggest_strategy or config:
        click.echo("💡 Strategy Suggestions:")

        if config:
            # Find strategies that use this indicator
            strategies = config.get('strategies', [])
            matching_strategies = []

            for strategy in strategies:
                strategy_indicators = strategy.get('indicators', [])
                strategy_symbol = strategy.get('market', '').replace('-', '').upper()

                # Check if strategy uses this indicator type or symbol
                uses_indicator = any(
                    indicator.lower() in ind_name.lower()
                    for ind_name in strategy_indicators
                )
                matches_symbol = symbol.upper() in strategy_symbol or strategy_symbol in symbol.upper()

                if uses_indicator or matches_symbol:
                    matching_strategies.append(strategy)

            if matching_strategies:
                click.echo("   📊 Found matching strategies in config:")
                for strategy in matching_strategies[:3]:  # Show top 3
                    strategy_name = strategy.get('name', 'unnamed')
                    strategy_market = strategy.get('market', 'unknown')
                    strategy_timeframe = strategy.get('timeframe', 'unknown')
                    click.echo(f"      • {strategy_name} ({strategy_market}, {strategy_timeframe})")

                if len(matching_strategies) > 3:
                    click.echo(f"      ... and {len(matching_strategies) - 3} more")

                click.echo(f"\n   ✅ Try: python cli/main.py strategy --strategy-name \"{matching_strategies[0].get('name')}\"")
            else:
                click.echo("   ⚠️  No matching strategies found in configuration")

        # Generic suggestions based on indicator and symbol
        click.echo(f"\n   🎯 For {indicator} on {symbol}:")
        click.echo("      • Create a strategy configuration in config.json")
        click.echo("      • Include position sizing and risk management")
        click.echo("      • Test with multiple market scenarios")
        click.echo("      • Compare with other strategies")
        click.echo()

    # Determine output directory
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")

    os.makedirs(output_dir, exist_ok=True)

    # Show what the demo would do
    click.echo(f"🎯 Demo backtesting for {indicator}")
    click.echo(f"📊 Symbol: {symbol}")
    click.echo(f"⏱️  Timeframe: {timeframe}")
    click.echo(f"📅 Synthetic data days: {days}")
    click.echo(f"📁 Output: {output_dir}")

    # TODO: Implement actual demo logic using IndicatorBacktestManager
    click.echo("\n📝 Demo implementation pending...")
    click.echo("   This will be integrated with IndicatorBacktestManager")
    click.echo("   For now, use strategy-based backtesting instead")
    click.echo("   Example: python cli/main.py list-strategies")


@click.command("real-data")
@click.option("--indicator", default="MACD", help="Indicator type to test (default: MACD)")
@click.option("--symbol", default="ETHUSDT", help="Trading symbol (default: ETHUSDT)")
@click.option("--timeframe", default="1h", help="Timeframe (default: 1h)")
@click.option("--days", default=30, type=int, help="Number of days of historical data (default: 30)")
@click.option("--output-dir", help="Output directory for reports (default: tests/results)")
@click.pass_context
def real_data(ctx, indicator: str, symbol: str, timeframe: str, days: int, output_dir: Optional[str]):
    """
    Indicator backtesting with real market data.

    ⚠️  DEPRECATED: Use strategy-based backtesting instead for comprehensive testing.

    This command tests indicators with real market data but lacks the context
    of position sizing, risk management, and trading strategy.

    Examples:
        # Basic real data test
        python cli/main.py real-data

        # Test RSI with longer history
        python cli/main.py real-data --indicator RSI --days 90
    """
    show_deprecation_warning("real-data", f"Strategy with {indicator} on {symbol}")

    # Determine output directory
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")

    os.makedirs(output_dir, exist_ok=True)

    # Show what the command would do
    click.echo(f"🎯 Real data backtesting for {indicator}")
    click.echo(f"📊 Symbol: {symbol}")
    click.echo(f"⏱️  Timeframe: {timeframe}")
    click.echo(f"📅 Historical data days: {days}")
    click.echo(f"📁 Output: {output_dir}")

    # TODO: Implement actual real data logic
    click.echo("\n📝 Real data implementation pending...")
    click.echo("   This will fetch real market data and run indicator tests")
    click.echo("   For now, use strategy-based backtesting instead")
    click.echo("   Example: python cli/main.py strategy --strategy-name <name> --scenarios real")


@click.command("compare")
@click.option("--indicators", default="MACD,RSI,EMA", help="Comma-separated list of indicators (default: MACD,RSI,EMA)")
@click.option("--symbol", default="ETHUSDT", help="Trading symbol (default: ETHUSDT)")
@click.option("--timeframe", default="1h", help="Timeframe (default: 1h)")
@click.option("--days", default=30, type=int, help="Number of days to test (default: 30)")
@click.option("--use-real-data", is_flag=True, help="Use real data instead of synthetic")
@click.option("--output-dir", help="Output directory for reports (default: tests/results)")
@click.pass_context
def compare(ctx, indicators: str, symbol: str, timeframe: str, days: int,
           use_real_data: bool, output_dir: Optional[str]):
    """
    Compare multiple indicators.

    ⚠️  DEPRECATED: Use strategy comparison instead for meaningful comparisons.

    Comparing indicators in isolation doesn't provide meaningful trading insights
    since real trading involves position sizing, risk management, and strategy logic.

    Examples:
        # Compare default indicators
        python cli/main.py compare

        # Compare custom indicators
        python cli/main.py compare --indicators "MACD,Bollinger,Stochastic"

        # Compare with real data
        python cli/main.py compare --use-real-data
    """
    show_deprecation_warning("compare", "Compare strategies instead")

    indicator_list = [ind.strip() for ind in indicators.split(',')]

    # Determine output directory
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")

    os.makedirs(output_dir, exist_ok=True)

    # Show what the comparison would do
    click.echo(f"🔍 Comparing {len(indicator_list)} indicators:")
    for i, indicator in enumerate(indicator_list, 1):
        click.echo(f"   {i}. {indicator}")

    click.echo(f"\n📊 Symbol: {symbol}")
    click.echo(f"⏱️  Timeframe: {timeframe}")
    click.echo(f"📅 Duration: {days} days")
    click.echo(f"📈 Data source: {'Real market data' if use_real_data else 'Synthetic data'}")
    click.echo(f"📁 Output: {output_dir}")

    # Suggest strategy comparison instead
    click.echo("\n💡 Better Alternative:")
    click.echo("   Use 'compare-strategies' to compare complete trading strategies")
    click.echo("   that include these indicators with proper position sizing and risk management")
    click.echo("   Example: python cli/main.py compare-strategies --all-strategies")

    # TODO: Implement actual comparison logic
    click.echo("\n📝 Indicator comparison implementation pending...")
    click.echo("   This will compare indicators across multiple metrics")
    click.echo("   For now, use strategy comparison instead")


@click.command("compare-popular")
@click.option("--symbol", default="ETHUSDT", help="Trading symbol (default: ETHUSDT)")
@click.option("--timeframe", default="1h", help="Timeframe (default: 1h)")
@click.option("--days", default=30, type=int, help="Number of days to test (default: 30)")
@click.option("--use-real-data", is_flag=True, help="Use real data instead of synthetic")
@click.option("--output-dir", help="Output directory for reports (default: tests/results)")
@click.pass_context
def compare_popular(ctx, symbol: str, timeframe: str, days: int,
                   use_real_data: bool, output_dir: Optional[str]):
    """
    Compare popular indicators (MACD, RSI, EMA, Bollinger Bands, Stochastic).

    ⚠️  DEPRECATED: Use strategy comparison instead for meaningful analysis.

    This command compares popular indicators but without trading context.
    Strategy-based comparison provides more actionable insights.

    Examples:
        # Compare popular indicators
        python cli/main.py compare-popular

        # Compare with real data on different symbol
        python cli/main.py compare-popular --symbol BTCUSDT --use-real-data
    """
    show_deprecation_warning("compare-popular", "Popular strategy comparison")

    popular_indicators = ["MACD", "RSI", "EMA", "Bollinger", "Stochastic"]

    # Determine output directory
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")

    os.makedirs(output_dir, exist_ok=True)

    # Show what the comparison would do
    click.echo(f"🌟 Comparing {len(popular_indicators)} popular indicators:")
    for i, indicator in enumerate(popular_indicators, 1):
        click.echo(f"   {i}. {indicator}")

    click.echo(f"\n📊 Symbol: {symbol}")
    click.echo(f"⏱️  Timeframe: {timeframe}")
    click.echo(f"📅 Duration: {days} days")
    click.echo(f"📈 Data source: {'Real market data' if use_real_data else 'Synthetic data'}")
    click.echo(f"📁 Output: {output_dir}")

    # Load configuration to suggest popular strategies
    try:
        from ..main import load_config
        config = load_config(ctx.obj.get('config_path'))

        if config:
            strategies = config.get('strategies', [])
            popular_strategies = [s for s in strategies[:5] if s.get('enabled', True)]  # Top 5 enabled

            if popular_strategies:
                click.echo("\n💡 Popular Strategies Available:")
                for strategy in popular_strategies:
                    strategy_name = strategy.get('name', 'unnamed')
                    strategy_market = strategy.get('market', 'unknown')
                    click.echo(f"   • {strategy_name} ({strategy_market})")

                click.echo(f"\n   ✅ Try: python cli/main.py compare-strategies --all-strategies")
    except Exception as e:
        logger.warning(f"Could not load configuration: {e}")

    # TODO: Implement actual popular comparison logic
    click.echo("\n📝 Popular indicator comparison implementation pending...")
    click.echo("   This will compare the most commonly used indicators")
    click.echo("   For now, use strategy comparison for better insights")
