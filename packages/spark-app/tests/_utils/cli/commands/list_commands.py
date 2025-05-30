"""
List Command Handlers

This module contains CLI commands for listing and displaying information:
- list-strategies: Show all strategies from config with details
- list-indicators: Show indicators with strategy context
"""

import logging
from typing import Any, Dict, List, Optional

import click

logger = logging.getLogger(__name__)


def register_list_commands(cli_group):
    """Register list-related commands with the CLI group."""
    cli_group.add_command(list_strategies)
    cli_group.add_command(list_indicators)


@click.command("list-strategies")
@click.option("--filter-exchange", help="Filter by exchange name")
@click.option("--filter-market", help="Filter by market (e.g., ETH-USD)")
@click.option("--filter-enabled", type=bool, help="Filter by enabled status (true/false)")
@click.option("--show-details", is_flag=True, help="Show detailed information for each strategy")
@click.option("--sort-by", default="name", type=click.Choice(["name", "market", "exchange", "timeframe"]),
              help="Sort strategies by field (default: name)")
@click.pass_context
def list_strategies(ctx, filter_exchange: Optional[str], filter_market: Optional[str],
                   filter_enabled: Optional[bool], show_details: bool, sort_by: str):
    """
    List all strategies from config with status and details.

    Shows strategy information including market, exchange, timeframe, indicators,
    and position sizing configuration.

    Examples:
        # List all strategies
        python cli/main.py list-strategies

        # List only enabled strategies with details
        python cli/main.py list-strategies --filter-enabled true --show-details

        # List strategies for specific market
        python cli/main.py list-strategies --filter-market "ETH-USD"

        # List and sort by timeframe
        python cli/main.py list-strategies --sort-by timeframe
    """
    # Import configuration and utility functions from main module
    from ..main import display_strategy_info
    from ..main import list_strategies as list_strategies_func
    from ..main import load_config

    # Load configuration
    config = load_config(ctx.obj.get('config_path'))
    if not config:
        raise click.ClickException("Could not load configuration. Use --config to specify config file path.")

    # Get filtered strategies
    strategies = list_strategies_func(
        config,
        filter_exchange=filter_exchange,
        filter_market=filter_market,
        filter_enabled=filter_enabled
    )

    if not strategies:
        click.echo("❌ No strategies found matching the specified criteria.")

        # Show available filters
        all_strategies = config.get('strategies', [])
        if all_strategies:
            exchanges = set(s.get('exchange') for s in all_strategies if s.get('exchange'))
            markets = set(s.get('market') for s in all_strategies if s.get('market'))

            click.echo("\n💡 Available filters:")
            if exchanges:
                click.echo(f"   Exchanges: {', '.join(sorted(exchanges))}")
            if markets:
                click.echo(f"   Markets: {', '.join(sorted(markets))}")

            enabled_count = len([s for s in all_strategies if s.get('enabled', True)])
            disabled_count = len(all_strategies) - enabled_count
            click.echo(f"   Enabled: {enabled_count}, Disabled: {disabled_count}")

        return

    # Sort strategies
    sort_key_map = {
        "name": lambda s: s.get('name', '').lower(),
        "market": lambda s: s.get('market', '').lower(),
        "exchange": lambda s: s.get('exchange', '').lower(),
        "timeframe": lambda s: s.get('timeframe', '').lower()
    }

    if sort_by in sort_key_map:
        strategies.sort(key=sort_key_map[sort_by])

    # Display header
    click.echo(f"📊 Found {len(strategies)} strategies")

    if any([filter_exchange, filter_market, filter_enabled is not None]):
        filters = []
        if filter_exchange:
            filters.append(f"exchange={filter_exchange}")
        if filter_market:
            filters.append(f"market={filter_market}")
        if filter_enabled is not None:
            filters.append(f"enabled={filter_enabled}")
        click.echo(f"🔍 Filters: {', '.join(filters)}")

    click.echo(f"📋 Sorted by: {sort_by}")
    click.echo()

    # Display strategies
    if show_details:
        # Detailed view
        for i, strategy in enumerate(strategies, 1):
            click.echo(f"{'='*60}")
            click.echo(f"Strategy {i} of {len(strategies)}")
            click.echo(f"{'='*60}")
            click.echo(display_strategy_info(strategy, config, show_indicators=True))
            click.echo()
    else:
        # Summary view
        click.echo("📋 Strategy Summary:")
        click.echo()

        # Table header
        click.echo(f"{'Name':<25} {'Market':<12} {'Exchange':<12} {'Timeframe':<10} {'Status':<8} {'Indicators':<12}")
        click.echo("-" * 90)

        for strategy in strategies:
            name = strategy.get('name', 'Unnamed')[:24]
            market = strategy.get('market', 'Unknown')[:11]
            exchange = strategy.get('exchange', 'Unknown')[:11]
            timeframe = strategy.get('timeframe', 'Unknown')[:9]
            status = '✅ Enabled' if strategy.get('enabled', True) else '❌ Disabled'
            indicator_count = len(strategy.get('indicators', []))
            indicators_display = f"{indicator_count} indicators"

            click.echo(f"{name:<25} {market:<12} {exchange:<12} {timeframe:<10} {status:<8} {indicators_display:<12}")

        # Summary statistics
        click.echo()
        click.echo("📈 Summary Statistics:")

        enabled_strategies = [s for s in strategies if s.get('enabled', True)]
        disabled_strategies = [s for s in strategies if not s.get('enabled', True)]

        click.echo(f"   Total strategies: {len(strategies)}")
        click.echo(f"   Enabled: {len(enabled_strategies)}")
        click.echo(f"   Disabled: {len(disabled_strategies)}")

        # Group by exchange and market
        exchanges = {}
        markets = {}

        for strategy in strategies:
            exchange = strategy.get('exchange', 'Unknown')
            market = strategy.get('market', 'Unknown')

            exchanges[exchange] = exchanges.get(exchange, 0) + 1
            markets[market] = markets.get(market, 0) + 1

        if len(exchanges) > 1:
            click.echo(f"   Exchanges: {', '.join([f'{ex}({count})' for ex, count in sorted(exchanges.items())])}")

        if len(markets) > 1:
            click.echo(f"   Markets: {', '.join([f'{market}({count})' for market, count in sorted(markets.items())])}")

    # Provide next steps
    click.echo("\n💡 Next steps:")
    click.echo("   • Use --show-details for full strategy information")
    click.echo("   • Use 'strategy --strategy-name <name>' to backtest a specific strategy")
    click.echo("   • Use 'compare-strategies' to compare multiple strategies")


@click.command("list-indicators")
@click.option("--filter-strategy", help="Filter by strategy name")
@click.option("--filter-market", help="Filter by market")
@click.option("--filter-type", help="Filter by indicator type")
@click.option("--show-config", is_flag=True, help="Show indicator configuration parameters")
@click.option("--show-strategies", is_flag=True, help="Show which strategies use each indicator")
@click.pass_context
def list_indicators(ctx, filter_strategy: Optional[str], filter_market: Optional[str],
                   filter_type: Optional[str], show_config: bool, show_strategies: bool):
    """
    List indicators with strategy context and configuration.

    Shows indicators from config with information about which strategies use them,
    their parameters, and timeframe/market context.

    Examples:
        # List all indicators
        python cli/main.py list-indicators

        # List indicators used by specific strategy
        python cli/main.py list-indicators --filter-strategy "MACD_ETH_Long"

        # List indicators with their configuration
        python cli/main.py list-indicators --show-config

        # List indicators and which strategies use them
        python cli/main.py list-indicators --show-strategies
    """
    # Import configuration loading from main module
    from ..main import load_config

    # Load configuration
    config = load_config(ctx.obj.get('config_path'))
    if not config:
        raise click.ClickException("Could not load configuration. Use --config to specify config file path.")

    indicators = config.get('indicators', [])
    strategies = config.get('strategies', [])

    if not indicators:
        click.echo("❌ No indicators found in configuration.")
        return

    # Create strategy-indicator mapping
    strategy_indicators_map = {}
    for strategy in strategies:
        strategy_name = strategy.get('name', 'unnamed')
        strategy_indicators = strategy.get('indicators', [])
        for indicator_name in strategy_indicators:
            if indicator_name not in strategy_indicators_map:
                strategy_indicators_map[indicator_name] = []
            strategy_indicators_map[indicator_name].append({
                'name': strategy_name,
                'market': strategy.get('market', 'Unknown'),
                'timeframe': strategy.get('timeframe', 'Unknown')
            })

    # Apply filters
    filtered_indicators = []
    for indicator in indicators:
        indicator_name = indicator.get('name', 'unnamed')

        # Filter by strategy
        if filter_strategy:
            strategy_uses_indicator = any(
                filter_strategy.lower() in [s['name'].lower() for s in strategy_indicators_map.get(indicator_name, [])]
            )
            if not strategy_uses_indicator:
                continue

        # Filter by market
        if filter_market:
            indicator_markets = set()
            for strategy_info in strategy_indicators_map.get(indicator_name, []):
                indicator_markets.add(strategy_info['market'])

            if filter_market.upper() not in [market.upper() for market in indicator_markets]:
                continue

        # Filter by type
        if filter_type:
            if indicator.get('type', '').lower() != filter_type.lower():
                continue

        filtered_indicators.append(indicator)

    if not filtered_indicators:
        click.echo("❌ No indicators found matching the specified criteria.")

        # Show available filters
        if indicators:
            types = set(ind.get('type') for ind in indicators if ind.get('type'))
            all_markets = set()
            all_strategy_names = set()

            for indicator_name, strategy_list in strategy_indicators_map.items():
                for strategy_info in strategy_list:
                    all_markets.add(strategy_info['market'])
                    all_strategy_names.add(strategy_info['name'])

            click.echo("\n💡 Available filters:")
            if types:
                click.echo(f"   Types: {', '.join(sorted(types))}")
            if all_markets:
                click.echo(f"   Markets: {', '.join(sorted(all_markets))}")
            if all_strategy_names:
                click.echo(f"   Strategies: {', '.join(sorted(all_strategy_names))}")

        return

    # Display header
    click.echo(f"📈 Found {len(filtered_indicators)} indicators")

    if any([filter_strategy, filter_market, filter_type]):
        filters = []
        if filter_strategy:
            filters.append(f"strategy={filter_strategy}")
        if filter_market:
            filters.append(f"market={filter_market}")
        if filter_type:
            filters.append(f"type={filter_type}")
        click.echo(f"🔍 Filters: {', '.join(filters)}")

    click.echo()

    # Display indicators
    for i, indicator in enumerate(filtered_indicators, 1):
        indicator_name = indicator.get('name', 'Unnamed')
        indicator_type = indicator.get('type', 'Unknown')
        indicator_timeframe = indicator.get('timeframe', 'Unknown')
        indicator_symbol = indicator.get('symbol', 'Unknown')
        enabled = '✅' if indicator.get('enabled', True) else '❌'

        click.echo(f"{'='*50}")
        click.echo(f"📊 Indicator {i}: {indicator_name}")
        click.echo(f"{'='*50}")
        click.echo(f"   Type: {indicator_type}")
        click.echo(f"   Timeframe: {indicator_timeframe}")
        click.echo(f"   Symbol: {indicator_symbol}")
        click.echo(f"   Status: {enabled}")

        # Show configuration if requested
        if show_config:
            click.echo("\n   📋 Configuration:")
            config_keys = [k for k in indicator.keys() if k not in ['name', 'type', 'timeframe', 'symbol', 'enabled']]
            if config_keys:
                for key in sorted(config_keys):
                    value = indicator[key]
                    click.echo(f"      {key}: {value}")
            else:
                click.echo("      No additional configuration parameters")

        # Show strategies that use this indicator
        if show_strategies or not show_config:
            using_strategies = strategy_indicators_map.get(indicator_name, [])
            if using_strategies:
                click.echo(f"\n   🎯 Used by {len(using_strategies)} strategies:")
                for strategy_info in using_strategies:
                    click.echo(f"      • {strategy_info['name']} ({strategy_info['market']}, {strategy_info['timeframe']})")
            else:
                click.echo("\n   ⚠️  Not used by any strategies")

        click.echo()

    # Summary statistics
    click.echo("📈 Summary Statistics:")

    enabled_indicators = [ind for ind in filtered_indicators if ind.get('enabled', True)]
    disabled_indicators = [ind for ind in filtered_indicators if not ind.get('enabled', True)]

    click.echo(f"   Total indicators: {len(filtered_indicators)}")
    click.echo(f"   Enabled: {len(enabled_indicators)}")
    click.echo(f"   Disabled: {len(disabled_indicators)}")

    # Group by type
    types = {}
    for indicator in filtered_indicators:
        indicator_type = indicator.get('type', 'Unknown')
        types[indicator_type] = types.get(indicator_type, 0) + 1

    if len(types) > 1:
        click.echo(f"   Types: {', '.join([f'{type_name}({count})' for type_name, count in sorted(types.items())])}")

    # Usage statistics
    unused_indicators = [ind for ind in filtered_indicators if ind.get('name') not in strategy_indicators_map]
    if unused_indicators:
        click.echo(f"   Unused indicators: {len(unused_indicators)}")

    # Provide next steps
    click.echo("\n💡 Next steps:")
    click.echo("   • Use --show-config to see indicator parameters")
    click.echo("   • Use --show-strategies to see strategy usage")
    click.echo("   • Use --filter-strategy to see indicators for specific strategy")
    if unused_indicators:
        click.echo("   • Consider removing unused indicators from configuration")
