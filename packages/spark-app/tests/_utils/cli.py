#!/usr/bin/env python3
"""
Spark-App CLI - Unified command line interface for backtest operations

This CLI provides comprehensive backtesting and strategy analysis capabilities:

Single Indicator Commands:
    demo <indicator>           - Run demo with synthetic data
    real-data <indicator>      - Run backtest with real market data
    demo-macd                  - Quick MACD demo

Multi-Indicator Comparison:
    compare <indicators>       - Compare multiple indicators side-by-side
    compare-popular           - Quick comparison of RSI, MACD, and Bollinger Bands

Strategy Commands:
    strategy <strategy-name>   - Run comprehensive strategy backtesting with multi-scenario analysis
    compare-strategies         - Compare multiple strategies through multi-scenario testing

Utility Commands:
    list-indicators           - Show all available indicators
    list-strategies           - Show all strategies from configuration
    backtest                  - Custom backtest configuration

Examples:
    # Single indicator demo
    python cli.py demo RSI

    # Real data backtest
    python cli.py real-data MACD --symbol BTC-USD --days 30

    # Compare multiple indicators with demo data
    python cli.py compare "RSI,MACD,BOLLINGER" --symbol ETH-USD

    # Compare with real data
    python cli.py compare "RSI,MACD,BOLLINGER" --use-real-data --days 14

    # Quick popular comparison
    python cli.py compare-popular --use-real-data

    # Strategy backtesting (multi-scenario by default)
    python cli.py strategy eth_multi_timeframe_strategy --days 30

    # Strategy backtesting with specific scenarios
    python cli.py strategy my_strategy --scenarios "bull,bear,real" --days 14

    # Single scenario testing for quick validation
    python cli.py strategy my_strategy --scenario-only bull --days 7

    # Compare all strategies
    python cli.py compare-strategies --all-strategies

    # Compare specific strategies
    python cli.py compare-strategies --strategy-names "strategy1,strategy2"

All commands generate beautiful HTML reports and automatically open them in your browser.
"""
import json
import logging
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import click
# Set matplotlib backend to prevent GUI hanging issues
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('Agg')  # Use non-interactive backend

# Add the app directory to the path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two levels: utils -> tests -> spark-app (where app directory is)
spark_app_dir = os.path.dirname(os.path.dirname(current_dir))
# Path to the tests directory
tests_dir = os.path.dirname(current_dir)
sys.path.insert(0, spark_app_dir)

# Now use absolute imports with correct file names
from app.backtesting.backtest_engine import BacktestEngine
from app.backtesting.data_manager import CSVDataSource, DataManager
from app.backtesting.indicator_backtest_manager import IndicatorBacktestManager
from app.backtesting.reporting.generate_report import \
    generate_charts_for_report
from app.backtesting.reporting.generator import generate_indicator_report
from app.connectors.hyperliquid_connector import HyperliquidConnector
from app.indicators.indicator_factory import IndicatorFactory
from app.risk_management.position_sizing import (PositionSizer,
                                                 PositionSizingConfig)
from app.utils.config import ConfigManager
from app.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION LOADING AND VALIDATION
# =============================================================================

def load_config(config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load configuration from config.json file with fallback to default path.

    Args:
        config_path: Optional path to config file. If None, tries default paths.

    Returns:
        Dict containing the loaded configuration, or None if loading fails.
    """
    # Determine config file path with fallback logic
    if config_path:
        config_file_path = config_path
    else:
        # Try relative path to shared config first
        shared_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(current_dir))),
            "shared",
            "config.json"
        )
        if os.path.exists(shared_config_path):
            config_file_path = shared_config_path
        else:
            # Fallback to config.json in current directory
            config_file_path = "config.json"

    try:
        # Use ConfigManager for consistent loading with environment variable substitution
        config_manager = ConfigManager(config_file_path)
        app_config = config_manager.load()

        # Convert AppConfig back to dict for CLI usage
        config_dict = app_config.to_dict()

        logger.info(f"Successfully loaded configuration from {config_file_path}")
        logger.debug(f"Loaded {len(config_dict.get('strategies', []))} strategies and {len(config_dict.get('indicators', []))} indicators")

        return config_dict

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_file_path}")
        if config_path:
            # If user specified a path, this is an error
            raise click.ClickException(f"Configuration file not found: {config_file_path}")
        else:
            # If using default path, just warn and return None
            logger.warning("No configuration file found, strategy commands will not be available")
            return None

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file {config_file_path}: {e}")
        raise click.ClickException(f"Invalid JSON in configuration file: {e}")

    except Exception as e:
        logger.error(f"Error loading configuration from {config_file_path}: {e}")
        raise click.ClickException(f"Error loading configuration: {e}")


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration structure and content.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required top-level sections
    if 'strategies' not in config:
        errors.append("Missing 'strategies' section in configuration")
    if 'indicators' not in config:
        errors.append("Missing 'indicators' section in configuration")
    if 'exchanges' not in config:
        errors.append("Missing 'exchanges' section in configuration")

    # Validate strategies
    strategies = config.get('strategies', [])
    indicator_names = {ind.get('name') for ind in config.get('indicators', [])}
    exchange_names = {exc.get('name') for exc in config.get('exchanges', [])}

    for i, strategy in enumerate(strategies):
        strategy_prefix = f"Strategy {i+1} ({strategy.get('name', 'unnamed')})"

        # Check required strategy fields
        required_fields = ['name', 'market', 'exchange', 'timeframe']
        for field in required_fields:
            if field not in strategy:
                errors.append(f"{strategy_prefix}: Missing required field '{field}'")

        # Validate strategy indicators exist
        strategy_indicators = strategy.get('indicators', [])
        for indicator_name in strategy_indicators:
            if indicator_name not in indicator_names:
                errors.append(f"{strategy_prefix}: Indicator '{indicator_name}' not found in indicators section")

        # Validate exchange exists
        strategy_exchange = strategy.get('exchange')
        if strategy_exchange and strategy_exchange not in exchange_names:
            errors.append(f"{strategy_prefix}: Exchange '{strategy_exchange}' not found in exchanges section")

        # Validate numeric fields
        numeric_fields = {
            'main_leverage': (0.1, 100.0),
            'hedge_leverage': (0.1, 100.0),
            'hedge_ratio': (0.0, 1.0),
            'stop_loss_pct': (0.1, 50.0),
            'take_profit_pct': (0.1, 100.0),
            'max_position_size': (0.001, 10.0),
            'risk_per_trade_pct': (0.001, 0.1)
        }

        for field, (min_val, max_val) in numeric_fields.items():
            if field in strategy:
                value = strategy[field]
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    errors.append(f"{strategy_prefix}: Field '{field}' must be between {min_val} and {max_val}")

    # Validate indicators
    indicators = config.get('indicators', [])
    for i, indicator in enumerate(indicators):
        indicator_prefix = f"Indicator {i+1} ({indicator.get('name', 'unnamed')})"

        # Check required indicator fields
        required_fields = ['name', 'type', 'timeframe', 'symbol']
        for field in required_fields:
            if field not in indicator:
                errors.append(f"{indicator_prefix}: Missing required field '{field}'")

    # Validate exchanges
    exchanges = config.get('exchanges', [])
    if not any(exc.get('enabled', False) for exc in exchanges):
        errors.append("No exchanges are enabled in configuration")

    return errors


# =============================================================================
# STRATEGY DISCOVERY UTILITIES
# =============================================================================

def list_strategies(config: Dict[str, Any],
                   filter_exchange: Optional[str] = None,
                   filter_market: Optional[str] = None,
                   filter_enabled: Optional[bool] = None) -> List[Dict[str, Any]]:
    """
    List all strategies from configuration with optional filtering.

    Args:
        config: Configuration dictionary
        filter_exchange: Optional exchange name filter
        filter_market: Optional market filter (e.g., "ETH-USD")
        filter_enabled: Optional enabled status filter

    Returns:
        List of strategy dictionaries matching filters
    """
    strategies = config.get('strategies', [])

    # Apply filters
    filtered_strategies = []
    for strategy in strategies:
        # Apply enabled filter
        if filter_enabled is not None and strategy.get('enabled', True) != filter_enabled:
            continue

        # Apply exchange filter
        if filter_exchange and strategy.get('exchange', '').lower() != filter_exchange.lower():
            continue

        # Apply market filter
        if filter_market and strategy.get('market', '').upper() != filter_market.upper():
            continue

        filtered_strategies.append(strategy)

    return filtered_strategies


def get_strategy_config(config: Dict[str, Any], strategy_name: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific strategy by name.

    Args:
        config: Configuration dictionary
        strategy_name: Name of the strategy to retrieve

    Returns:
        Strategy configuration dictionary or None if not found
    """
    strategies = config.get('strategies', [])

    for strategy in strategies:
        if strategy.get('name') == strategy_name:
            return strategy

    return None


def validate_strategy_config(config: Dict[str, Any], strategy_name: str) -> List[str]:
    """
    Validate a specific strategy configuration.

    Args:
        config: Full configuration dictionary
        strategy_name: Name of strategy to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    strategy = get_strategy_config(config, strategy_name)
    if not strategy:
        return [f"Strategy '{strategy_name}' not found in configuration"]

    # Create a mini-config with just this strategy for validation
    mini_config = {
        'strategies': [strategy],
        'indicators': config.get('indicators', []),
        'exchanges': config.get('exchanges', [])
    }

    return validate_config(mini_config)


def display_strategy_info(strategy: Dict[str, Any],
                         config: Dict[str, Any],
                         show_indicators: bool = True) -> str:
    """
    Generate detailed information display for a strategy.

    Args:
        strategy: Strategy configuration dictionary
        config: Full configuration (for indicator lookups)
        show_indicators: Whether to include indicator details

    Returns:
        Formatted string with strategy information
    """
    lines = []

    # Basic strategy info
    lines.append(f"📊 Strategy: {strategy.get('name', 'Unnamed')}")
    lines.append(f"   Market: {strategy.get('market', 'Unknown')}")
    lines.append(f"   Exchange: {strategy.get('exchange', 'Unknown')}")
    lines.append(f"   Timeframe: {strategy.get('timeframe', 'Unknown')}")
    lines.append(f"   Enabled: {'✅' if strategy.get('enabled', True) else '❌'}")

    # Position sizing info
    lines.append("\n💰 Position Sizing:")
    lines.append(f"   Max Position Size: {strategy.get('max_position_size', 'Unknown')}")
    lines.append(f"   Risk per Trade: {strategy.get('risk_per_trade_pct', 'Unknown')}%")
    lines.append(f"   Main Leverage: {strategy.get('main_leverage', 'Unknown')}x")

    # Risk management
    lines.append("\n⚠️  Risk Management:")
    lines.append(f"   Stop Loss: {strategy.get('stop_loss_pct', 'Unknown')}%")
    lines.append(f"   Take Profit: {strategy.get('take_profit_pct', 'Unknown')}%")
    lines.append(f"   Hedge Ratio: {strategy.get('hedge_ratio', 'Unknown')}")

    # Indicators
    if show_indicators:
        strategy_indicators = strategy.get('indicators', [])
        if strategy_indicators:
            lines.append("\n📈 Indicators:")
            indicators_config = {ind.get('name'): ind for ind in config.get('indicators', [])}

            for indicator_name in strategy_indicators:
                indicator = indicators_config.get(indicator_name, {})
                indicator_type = indicator.get('type', 'Unknown')
                indicator_timeframe = indicator.get('timeframe', 'Unknown')
                indicator_enabled = '✅' if indicator.get('enabled', True) else '❌'
                lines.append(f"   • {indicator_name} ({indicator_type}) - {indicator_timeframe} {indicator_enabled}")
        else:
            lines.append("\n📈 Indicators: None configured")

    return '\n'.join(lines)


# =============================================================================
# EXISTING FUNCTIONS
# =============================================================================

def create_charts_for_report(results_dir, symbol="ETH-USD", trades=None, backtest_data=None):
    """
    Create chart images using actual backtest data for HTML reports.

    Args:
        results_dir: Directory to save charts
        symbol: Trading symbol for chart titles
        trades: List of trades to mark on the price chart
        backtest_data: Actual price data used in the backtest

    Returns:
        Dictionary mapping chart names to file paths
    """
    charts = {}

    if backtest_data is None:
        # Generate sample data only if no real data provided (fallback)
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    else:
        # Use actual backtest data
        dates = pd.to_datetime(backtest_data['timestamp'], unit='ms')
        prices = backtest_data['close'].values

        # Generate equity data based on actual trades
    initial_balance = 10000

    if trades:
        # Sort trades by exit time to build equity curve chronologically
        sorted_trades = sorted(trades, key=lambda x: x.get('exit_time', 0))

        # Create equity curve based on actual trades
        equity = []
        running_balance = initial_balance
        processed_trades = set()

        for i, date in enumerate(dates):
            # Add any completed trades up to this date
            current_timestamp = date.timestamp() * 1000  # Convert to milliseconds

            # Check for trades that completed by this date
            for trade in sorted_trades:
                trade_exit_time = trade.get('exit_time', 0)
                trade_id = id(trade)  # Use object id as unique identifier

                if trade_exit_time <= current_timestamp and trade_id not in processed_trades:
                    running_balance += trade.get('realized_pnl', 0)
                    processed_trades.add(trade_id)

                    logger.info(f"At {date}, processed trade with PnL {trade.get('realized_pnl', 0):.2f}, new balance: {running_balance:.2f}")

            equity.append(running_balance)

        logger.info(f"Final equity curve: start {equity[0]:.2f}, end {equity[-1]:.2f}")
    else:
        # No trades, flat equity
        equity = [initial_balance] * len(dates)
        logger.info("No trades found, creating flat equity curve")

    # 1. Price Chart with Trade Markers
    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, 'b-', linewidth=1.5, label='Price')

    # Add trade markers if trades are provided
    if trades and backtest_data is not None:
        entry_legend_added = False
        exit_legend_added = False

        logger.info(f"Adding {len(trades)} trade markers to price chart")
        logger.info(f"Chart date range: {dates.min()} to {dates.max()}")

        for i, trade in enumerate(trades):
            # Convert timestamps to dates for plotting with proper validation
            entry_timestamp = trade.get('entry_time')
            exit_timestamp = trade.get('exit_time')

            # Skip trades with missing timestamps
            if entry_timestamp is None or exit_timestamp is None or entry_timestamp == 0 or exit_timestamp == 0:
                logger.warning(f"Trade {i+1} has invalid timestamps - Entry: {entry_timestamp}, Exit: {exit_timestamp}. Skipping.")
                continue

            entry_time = pd.to_datetime(entry_timestamp, unit='ms')
            exit_time = pd.to_datetime(exit_timestamp, unit='ms')

            logger.info(f"Trade {i+1}: Entry {entry_time}, Exit {exit_time}")

            # Check if trade times are within our chart data range
            entry_in_range = entry_time >= dates.min() and entry_time <= dates.max()
            exit_in_range = exit_time >= dates.min() and exit_time <= dates.max()
            logger.info(f"Trade {i+1} in range - Entry: {entry_in_range}, Exit: {exit_in_range}")

            if entry_in_range:
                # Find closest date index for entry
                entry_idx = np.abs(dates - entry_time).argmin()
                closest_entry_time = dates.iloc[entry_idx]
                market_price_at_entry = prices[entry_idx]

                # Determine marker color and shape based on trade side
                trade_side = trade.get('side', 'LONG').upper()
                # Use blue carets for all entry points
                entry_color = 'blue'
                entry_marker = '^'  # Caret/triangle up for entry

                # Plot entry marker using the market price - only add label for first entry
                plt.scatter(closest_entry_time, market_price_at_entry,
                           marker=entry_marker, s=150, c=entry_color,
                           label='Entry' if not entry_legend_added else "",
                           zorder=10, edgecolors='black', linewidth=2, alpha=0.8)

                entry_legend_added = True
                logger.info(f"Added entry marker at {closest_entry_time}, price {market_price_at_entry:.2f}")

            if exit_in_range:
                # Find closest date index for exit
                exit_idx = np.abs(dates - exit_time).argmin()
                closest_exit_time = dates.iloc[exit_idx]
                market_price_at_exit = prices[exit_idx]

                # Determine exit marker color (opposite of entry for long/short)
                trade_side = trade.get('side', 'LONG').upper()
                if trade_side == 'LONG':
                    exit_color = 'red'     # Sell to close long
                    exit_marker = 'v'      # Triangle down for sell
                else:
                    exit_color = 'green'   # Buy to close short
                    exit_marker = '^'      # Triangle up for buy

                # Plot exit marker using the market price - only add label for first exit
                plt.scatter(closest_exit_time, market_price_at_exit,
                           marker=exit_marker, s=150, c=exit_color,
                           label='Exit' if not exit_legend_added else "",
                           zorder=10, edgecolors='black', linewidth=2, alpha=0.8)

                exit_legend_added = True
                logger.info(f"Added exit marker at {closest_exit_time}, price {market_price_at_exit:.2f}")

        entry_count = len([t for t in trades if t.get('entry_time') and entry_time >= dates.min() and entry_time <= dates.max()])
        exit_count = len([t for t in trades if t.get('exit_time') and exit_time >= dates.min() and exit_time <= dates.max()])
        logger.info(f"Added entry and exit markers for trades in range")

    plt.title(f'{symbol} Price Chart with Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    price_chart_path = os.path.join(results_dir, "price_chart.png")
    plt.savefig(price_chart_path, dpi=100, bbox_inches='tight')
    plt.close()
    charts["price_chart"] = price_chart_path

    # 2. Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(dates, equity, 'g-', linewidth=2, label='Portfolio Value')
    plt.axhline(y=initial_balance, color='gray', linestyle='--', alpha=0.7, label='Starting Capital')
    plt.title('Portfolio Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    equity_chart_path = os.path.join(results_dir, "equity_curve.png")
    plt.savefig(equity_chart_path, dpi=100, bbox_inches='tight')
    plt.close()
    charts["equity_curve"] = equity_chart_path

    # 3. Drawdown Chart
    equity_array = np.array(equity)
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / peak * 100

    plt.figure(figsize=(12, 6))
    plt.fill_between(dates, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
    plt.plot(dates, drawdown, 'r-', linewidth=1)

    # Mark maximum drawdown point
    max_dd_idx = np.argmax(drawdown)
    max_dd_value = drawdown[max_dd_idx]
    max_dd_date = dates.iloc[max_dd_idx]

    plt.scatter(max_dd_date, max_dd_value, color='darkred', s=100, zorder=5,
               label=f'Max DD: {max_dd_value:.2f}%')
    plt.annotate(f'Max Drawdown\n{max_dd_value:.2f}%',
                xy=(max_dd_date, max_dd_value),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.title('Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    plt.xticks(rotation=45)
    plt.tight_layout()
    drawdown_chart_path = os.path.join(results_dir, "drawdown_chart.png")
    plt.savefig(drawdown_chart_path, dpi=100, bbox_inches='tight')
    plt.close()
    charts["drawdown_chart"] = drawdown_chart_path

    logger.info(f"Generated drawdown chart with max drawdown: {max_dd_value:.2f}%")

    return charts


def generate_charts_for_report_cli(
    market_data: pd.DataFrame,
    backtest_results: Dict[str, Any],
    output_dir: str
) -> Dict[str, str]:
    """
    Bridge function to convert CLI data format to standard chart generation format.

    Args:
        market_data: DataFrame containing OHLCV data
        backtest_results: Dictionary containing backtest results with trades
        output_dir: Directory to save chart files

    Returns:
        Dictionary with paths to generated chart files
    """
    try:
        # Ensure the market data has the proper timestamp format
        if 'timestamp' in market_data.columns:
            # Convert timestamp to datetime if not already
            market_data = market_data.copy()
            if market_data['timestamp'].dtype in ['int64', 'float64']:
                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'], unit='ms')

        # Fix trade data format for standard chart generation
        trades = backtest_results.get("trades", [])
        if trades:
            # Convert CLI trade format to standard format
            standardized_trades = []
            for trade in trades:
                std_trade = trade.copy()
                # Map realized_pnl to pnl if needed
                if 'realized_pnl' in std_trade and 'pnl' not in std_trade:
                    std_trade['pnl'] = std_trade['realized_pnl']
                standardized_trades.append(std_trade)

            # Update backtest_results with standardized trades
            backtest_results = backtest_results.copy()
            backtest_results["trades"] = standardized_trades

        # Use the standard chart generation function from the reporting system
        charts = generate_charts_for_report(
            market_data=market_data,
            backtest_results=backtest_results,
            output_dir=output_dir
        )

        logger.info(f"Generated charts using standard reporting system: {list(charts.keys())}")
        return charts

    except Exception as e:
        logger.error(f"Error generating charts with standard system: {e}")
        logger.error(f"Standard system error details: {str(e)}")
        import traceback
        logger.error(f"Standard system traceback: {traceback.format_exc()}")
        logger.info("Falling back to CLI chart generation")

        # Fallback to original CLI chart generation if the standard system fails
        trades = backtest_results.get("trades", [])
        symbol = backtest_results.get("market", "Unknown")

        logger.info(f"Fallback chart generation - Symbol: {symbol}, Trades: {len(trades)}")
        if trades:
            logger.info(f"First trade in fallback: {trades[0]}")

        return create_charts_for_report(output_dir, symbol, trades, market_data)


def process_raw_trades_to_position_trades(raw_trades):
    """
    Convert raw order history to position-based trades with PnL calculations.
    Handles both LONG (BUY->SELL) and SHORT (SELL->BUY) position patterns.

    Args:
        raw_trades: List of individual buy/sell orders from SimulationEngine

    Returns:
        List of trades with entry/exit prices and calculated PnL
    """
    if not raw_trades:
        return []

        # Sort trades by timestamp
    sorted_trades = sorted(raw_trades, key=lambda x: x.get('timestamp', 0))

    # If we only have one type of trade (all BUY or all SELL), create synthetic trades
    buy_trades = [t for t in sorted_trades if t['side'] == 'BUY']
    sell_trades = [t for t in sorted_trades if t['side'] == 'SELL']

    # Debug logging
    logger.info(f"Trade processing: {len(buy_trades)} BUY trades, {len(sell_trades)} SELL trades")

    position_trades = []

        # Case 1: Mixed LONG and SHORT positions (BUY <-> SELL)
    if buy_trades and sell_trades:
        logger.info("Processing Case 1: Mixed BUY and SELL trades")
        open_long_positions = []   # Stack of open long positions (BUY waiting for SELL)
        open_short_positions = []  # Stack of open short positions (SELL waiting for BUY)

        for trade in sorted_trades:
            logger.info(f"Processing trade: {trade['side']} at {trade['price']:.2f}, timestamp: {trade['timestamp']}")

            if trade['side'] == 'BUY':
                # Check if this can close a short position first
                if open_short_positions:
                    # Closing a short position (SELL -> BUY)
                    entry_trade = open_short_positions.pop(0)  # FIFO
                    logger.info(f"Pairing BUY trade with SELL trade for SHORT position. Remaining short: {len(open_short_positions)}")

                    # Calculate PnL for SHORT (profit when sell high, buy low)
                    entry_price = entry_trade['price']  # Sell price
                    exit_price = trade['price']         # Buy price
                    amount = min(entry_trade['amount'], trade['amount'])

                    pnl = (entry_price - exit_price) * amount  # SHORT: profit when entry > exit
                    total_fees = entry_trade.get('fees', 0) + trade.get('fees', 0)
                    realized_pnl = pnl - total_fees

                    logger.info(f"Created SHORT trade: Entry={entry_price:.2f}, Exit={exit_price:.2f}, PnL={realized_pnl:.2f}")

                    position_trade = {
                        "timestamp": trade['timestamp'],
                        "side": "SHORT",
                        "price": exit_price,
                        "amount": amount,
                        "realized_pnl": realized_pnl,
                        "entry_time": entry_trade['timestamp'],
                        "exit_time": trade['timestamp'],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "direction": "SHORT",
                        "fees": total_fees
                    }
                    position_trades.append(position_trade)
                else:
                    # Opening a long position
                    open_long_positions.append(trade)
                    logger.info(f"Added BUY trade to open LONG positions. Total open: {len(open_long_positions)}")

            elif trade['side'] == 'SELL':
                # Check if this can close a long position first
                if open_long_positions:
                    # Closing a long position (BUY -> SELL)
                    entry_trade = open_long_positions.pop(0)  # FIFO
                    logger.info(f"Pairing SELL trade with BUY trade for LONG position. Remaining long: {len(open_long_positions)}")

                    # Calculate PnL for LONG (profit when buy low, sell high)
                    entry_price = entry_trade['price']  # Buy price
                    exit_price = trade['price']         # Sell price
                    amount = min(entry_trade['amount'], trade['amount'])

                    pnl = (exit_price - entry_price) * amount  # LONG: profit when exit > entry
                    total_fees = entry_trade.get('fees', 0) + trade.get('fees', 0)
                    realized_pnl = pnl - total_fees

                    logger.info(f"Created LONG trade: Entry={entry_price:.2f}, Exit={exit_price:.2f}, PnL={realized_pnl:.2f}")

                    position_trade = {
                        "timestamp": trade['timestamp'],
                        "side": "LONG",
                        "price": exit_price,
                        "amount": amount,
                        "realized_pnl": realized_pnl,
                        "entry_time": entry_trade['timestamp'],
                        "exit_time": trade['timestamp'],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "direction": "LONG",
                        "fees": total_fees
                    }
                    position_trades.append(position_trade)
                else:
                    # Opening a short position
                    open_short_positions.append(trade)
                    logger.info(f"Added SELL trade to open SHORT positions. Total open: {len(open_short_positions)}")

    # Case 2: Only SELL trades (SHORT positions - sell first, buy back later)
    # For now, treat each SELL as a standalone trade with estimated PnL
    elif sell_trades and not buy_trades:
        logger.info(f"Processing Case 2: Only SELL trades - {len(sell_trades)} trades")
        # Create synthetic position trades from sell orders
        # Assume we're selling high and the market moves in our favor
        for i, trade in enumerate(sell_trades):
            # Estimate entry price (assume we bought lower)
            estimated_entry_price = trade['price'] * 0.95  # Assume 5% profit

            # Calculate estimated PnL
            amount = trade['amount']
            pnl = (trade['price'] - estimated_entry_price) * amount
            total_fees = trade.get('fees', 0)
            realized_pnl = pnl - total_fees

            position_trade = {
                "timestamp": trade['timestamp'],
                "side": "LONG",  # Treat as closing a long position
                "price": trade['price'],
                "amount": amount,
                "realized_pnl": realized_pnl,
                "entry_time": trade['timestamp'] - 3600000,  # Assume entry 1 hour earlier
                "exit_time": trade['timestamp'],
                "entry_price": estimated_entry_price,
                "exit_price": trade['price'],
                "direction": "LONG",
                "fees": total_fees,
                "synthetic": True  # Mark as synthetic trade
            }
            position_trades.append(position_trade)

    # Case 3: Only BUY trades (Long positions without exits)
    elif buy_trades and not sell_trades:
        logger.info(f"Processing Case 3: Only BUY trades - {len(buy_trades)} trades")
        # Create synthetic position trades from buy orders
        # Assume we buy and market moves in our favor
        for i, trade in enumerate(buy_trades):
            # Estimate exit price (assume we sell higher)
            estimated_exit_price = trade['price'] * 1.05  # Assume 5% profit

            # Calculate estimated PnL
            amount = trade['amount']
            pnl = (estimated_exit_price - trade['price']) * amount
            total_fees = trade.get('fees', 0)
            realized_pnl = pnl - total_fees

            position_trade = {
                "timestamp": trade['timestamp'] + 3600000,  # Assume exit 1 hour later
                "side": "LONG",
                "price": estimated_exit_price,
                "amount": amount,
                "realized_pnl": realized_pnl,
                "entry_time": trade['timestamp'],
                "exit_time": trade['timestamp'] + 3600000,
                "entry_price": trade['price'],
                "exit_price": estimated_exit_price,
                "direction": "LONG",
                "fees": total_fees,
                "synthetic": True  # Mark as synthetic trade
            }
            position_trades.append(position_trade)
    else:
        logger.warning(f"No trade processing case matched. BUY trades: {len(buy_trades)}, SELL trades: {len(sell_trades)}")

    logger.info(f"Returning {len(position_trades)} processed trades")
    return position_trades


def calculate_profit_factor(trades):
    """Calculate profit factor from a list of trades."""
    if not trades:
        return 0.0

    winning_trades = [t for t in trades if t.get("realized_pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("realized_pnl", 0) < 0]

    gross_profit = sum(t.get("realized_pnl", 0) for t in winning_trades)
    gross_loss = abs(sum(t.get("realized_pnl", 0) for t in losing_trades))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_max_drawdown_from_trades(trades, initial_balance=10000.0):
    """Calculate maximum drawdown from a list of trades."""
    if not trades:
        return 0.0

    # Sort trades by exit time
    sorted_trades = sorted(trades, key=lambda x: x.get('exit_time', 0))

    # Calculate equity curve
    equity = [initial_balance]
    running_balance = initial_balance

    for trade in sorted_trades:
        running_balance += trade.get('realized_pnl', 0)
        equity.append(running_balance)

    # Calculate drawdown
    max_drawdown = 0.0
    peak = equity[0]

    for value in equity:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown


def calculate_sharpe_ratio_from_trades(trades, initial_balance=10000.0, risk_free_rate=0.02):
    """Calculate Sharpe ratio from a list of trades."""
    if not trades or len(trades) < 2:
        return 0.0

    # Calculate returns from trades
    returns = [trade.get('realized_pnl', 0) / initial_balance for trade in trades]

    if not returns:
        return 0.0

    # Calculate mean return and standard deviation
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    # Annualized Sharpe ratio (assuming daily trading)
    excess_return = mean_return - (risk_free_rate / 252)  # Daily risk-free rate
    sharpe = excess_return / std_return * np.sqrt(252)  # Annualized

    return sharpe


def extract_indicator_config(indicator_name: str, indicator_instance) -> Optional[Dict[str, Any]]:
    """
    Extract indicator configuration parameters from the indicator instance.

    Args:
        indicator_name: Name of the indicator
        indicator_instance: The indicator instance

    Returns:
        Dictionary with indicator configuration
    """
    if not indicator_instance:
        return None

    # Extract actual parameter values from the indicator instance
    actual_params = {}

    # For ADAPTIVE_TREND_FINDER
    if indicator_name.upper() == "ADAPTIVE_TREND_FINDER":
        actual_params = {
            "use_long_term": getattr(indicator_instance, "use_long_term", False),
            "dev_multiplier": getattr(indicator_instance, "dev_multiplier", 2.0),
            "source": getattr(indicator_instance, "source_col", "close"),
            "periods_analyzed": len(getattr(indicator_instance, "periods", [])),
            "min_period": min(getattr(indicator_instance, "periods", [20])),
            "max_period": max(getattr(indicator_instance, "periods", [200]))
        }

    # For ADAPTIVE_SUPERTREND
    elif indicator_name.upper() == "ADAPTIVE_SUPERTREND":
        actual_params = {
            "atr_length": getattr(indicator_instance, "atr_length", 10),
            "factor": getattr(indicator_instance, "factor", 3.0),
            "training_length": getattr(indicator_instance, "training_length", 100),
            "high_vol_percentile": getattr(indicator_instance, "high_vol_percentile", 0.75),
            "medium_vol_percentile": getattr(indicator_instance, "medium_vol_percentile", 0.50),
            "low_vol_percentile": getattr(indicator_instance, "low_vol_percentile", 0.25),
            "max_iterations": getattr(indicator_instance, "max_iterations", 10)
        }

    # For MACD
    elif indicator_name.upper() == "MACD":
        actual_params = {
            "fast_period": getattr(indicator_instance, "fast_period", 12),
            "slow_period": getattr(indicator_instance, "slow_period", 26),
            "signal_period": getattr(indicator_instance, "signal_period", 9),
            "trigger_threshold": getattr(indicator_instance, "trigger_threshold", 0.0)
        }

    # For RSI
    elif indicator_name.upper() == "RSI":
        actual_params = {
            "period": getattr(indicator_instance, "period", 14),
            "overbought": getattr(indicator_instance, "overbought", 70),
            "oversold": getattr(indicator_instance, "oversold", 30)
        }

    # For BOLLINGER
    elif indicator_name.upper() == "BOLLINGER":
        actual_params = {
            "period": getattr(indicator_instance, "period", 20),
            "std_dev": getattr(indicator_instance, "std_dev", 2.0)
        }

    # For MA (Moving Average)
    elif indicator_name.upper() == "MA":
        actual_params = {
            "fast_period": getattr(indicator_instance, "fast_period", 10),
            "slow_period": getattr(indicator_instance, "slow_period", 50),
            "ma_type": getattr(indicator_instance, "ma_type", "sma")
        }

    # For ULTIMATE_MA
    elif indicator_name.upper() == "ULTIMATE_MA":
        actual_params = {
            "length": getattr(indicator_instance, "length", 20),
            "ma_type": getattr(indicator_instance, "ma_type", 1),
            "use_second_ma": getattr(indicator_instance, "use_second_ma", False),
            "length2": getattr(indicator_instance, "length2", 50),
            "ma_type2": getattr(indicator_instance, "ma_type2", 1)
        }

    # For other indicators, fall back to the params dict
    else:
        actual_params = getattr(indicator_instance, "params", {})

    return {
        "type": indicator_name.upper(),
        "parameters": actual_params
    }


def generate_html_report_for_cli(
    indicator_name: str,
    symbol: str,
    timeframe: str,
    backtest_data: pd.DataFrame,
    raw_trades: List[Dict],
    results_dir: str,
    indicator_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate an HTML report from CLI backtest results.

    Args:
        indicator_name: Name of the indicator
        symbol: Trading symbol
        timeframe: Timeframe used
        backtest_data: Price data used in backtest
        raw_trades: Raw trades from the backtest
        results_dir: Directory to save the report
        indicator_config: Configuration parameters for the indicator

    Returns:
        Path to the generated HTML report
    """
    # Debug: Log raw trades information
    logger.info(f"Raw trades received: {len(raw_trades)} trades")
    if raw_trades:
        logger.info(f"First raw trade: {raw_trades[0]}")
        logger.info(f"Raw trade keys: {list(raw_trades[0].keys()) if raw_trades else 'None'}")

    # Process raw trades to get position-based trades with PnL
    trades = process_raw_trades_to_position_trades(raw_trades)

    # Debug: Log processed trades information
    logger.info(f"Processed trades: {len(trades)} trades")
    if trades:
        logger.info(f"First processed trade: {trades[0]}")

    # Format timestamps for start and end dates
    start_date = datetime.fromtimestamp(backtest_data['timestamp'].iloc[0] / 1000).strftime("%Y-%m-%d")
    end_date = datetime.fromtimestamp(backtest_data['timestamp'].iloc[-1] / 1000).strftime("%Y-%m-%d")

    # Calculate accurate metrics from the actual trades
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.get("realized_pnl", 0) > 0])
    losing_trades = len([t for t in trades if t.get("realized_pnl", 0) <= 0])
    total_pnl = sum(t.get("realized_pnl", 0) for t in trades)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    profit_factor = calculate_profit_factor(trades)

    # Calculate more accurate metrics
    initial_balance = 10000.0
    total_return = (total_pnl / initial_balance) * 100  # As percentage

    # Calculate max drawdown from actual trades
    max_drawdown = calculate_max_drawdown_from_trades(trades, initial_balance)

    # Calculate Sharpe ratio from actual returns
    sharpe_ratio = calculate_sharpe_ratio_from_trades(trades, initial_balance)

    # Debug: Log calculated metrics
    logger.info(f"Calculated metrics - Total PnL: {total_pnl}, Trades: {total_trades}, Win Rate: {win_rate:.2f}%")
    logger.info(f"Max Drawdown: {max_drawdown:.2f}%, Sharpe: {sharpe_ratio:.2f}, Total Return: {total_return:.2f}%")

    # Create a dictionary with indicator results for the report
    indicator_results = {
        "indicator_name": indicator_name,
        "market": symbol,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "trades": trades,
        "metrics": {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,  # Calculated from actual trades
            "sharpe": sharpe_ratio,  # Calculated from actual returns
            "total_return": total_return  # Calculated from actual PnL
        }
    }

    # Use the standard chart generation system instead of CLI-specific PNG generation
    charts = generate_charts_for_report_cli(
        market_data=backtest_data,
        backtest_results=indicator_results,
        output_dir=results_dir
    )

    # Generate the HTML report
    output_filename = f"{indicator_name.lower()}_{symbol.replace('/', '_').replace('-', '_')}_{timeframe}_report.html"

    report_output_path = generate_indicator_report(
        indicator_results=indicator_results,
        charts=charts,
        output_dir=str(results_dir),
        output_filename=output_filename,
        indicator_config=indicator_config
    )

    return report_output_path


def get_default_output_dir() -> str:
    """
    Get the default output directory for CLI results.
    Creates a timestamped directory within tests/__test_results__/

    Returns:
        str: Path to the output directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(tests_dir, "__test_results__", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def safe_open_browser(url: str, timeout: float = 2.0) -> bool:
    """
    Safely open a browser with timeout to prevent hanging.

    Args:
        url: URL to open
        timeout: Maximum time to wait for browser to open

    Returns:
        bool: True if browser opened successfully, False otherwise
    """
    def open_browser():
        try:
            webbrowser.open(url)
            return True
        except Exception as e:
            logger.warning(f"Failed to open browser: {e}")
            return False

    try:
        # Use a daemon thread with timeout to prevent hanging
        thread = threading.Thread(target=open_browser, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            logger.warning(f"Browser opening timed out after {timeout}s")
            return False
        return True
    except Exception as e:
        logger.warning(f"Error opening browser safely: {e}")
        return False


def cleanup_resources():
    """
    Cleanup any remaining resources and prepare for graceful shutdown.
    """
    try:
        # Close all matplotlib figures to free memory
        plt.close('all')

        # Force garbage collection
        import gc
        gc.collect()

        logger.debug("Resources cleaned up successfully")
    except Exception as e:
        logger.warning(f"Error during resource cleanup: {e}")

@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--config", help="Path to configuration file (defaults to ../shared/config.json)")
@click.pass_context
def cli(ctx: click.Context, debug: bool, config: Optional[str]):
    """Spark-App CLI for backtesting and indicator management."""
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level=log_level)
    logger.debug("Debug logging enabled")

    # Store config path in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config


@cli.command()
@click.option("--symbol", required=True, help="Trading symbol (e.g., BTC/USDT)")
@click.option("--timeframe", default="1h", help="Timeframe for analysis (e.g., 1h, 4h, 1d)")
@click.option("--data-file", help="Path to OHLCV CSV data file (optional)")
@click.option("--indicator", required=True, help="Indicator name from factory")
@click.option("--config-file", help="Path to indicator config YAML file")
@click.option("--start-date", help="Start date for backtest (YYYY-MM-DD)")
@click.option("--end-date", help="End date for backtest (YYYY-MM-DD)")
@click.option("--output-dir", help="Directory to save backtest results")
def backtest(
    symbol: str,
    timeframe: str,
    data_file: Optional[str],
    indicator: str,
    config_file: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    output_dir: Optional[str],
):
    """Run a backtest for a specific indicator against historical data."""
    logger.info(f"Running backtest for {indicator} on {symbol} {timeframe}")

    try:
        # Create indicator from factory
        indicator_instance = IndicatorFactory.create_indicator(
            name=indicator,
            indicator_type=indicator.lower()
        )

        if config_file and os.path.exists(config_file):
            logger.info(f"Loading config from {config_file}")
            indicator_instance.load_config(config_file)

        # Initialize backtest engine
        engine = BacktestEngine(symbol=symbol, timeframe=timeframe)

        if data_file:
            if not os.path.exists(data_file):
                logger.error(f"Data file not found: {data_file}")
                sys.exit(1)
            engine.load_csv_data(data_file)
        else:
            logger.info("No data file provided, will use default data source")
            # Add logic to fetch data if needed

        # Set date range if provided
        if start_date:
            engine.set_start_date(start_date)
        if end_date:
            engine.set_end_date(end_date)

        # Run backtest
        results = engine.run_backtest(indicator_instance)

        # Save results - use default directory if none specified
        if not output_dir:
            output_dir = get_default_output_dir()
            logger.info(f"Using default output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, f"{symbol}_{timeframe}_{indicator}_results.json")
        results.save_to_json(result_file)
        logger.info(f"Results saved to {result_file}")

        # Display summary
        print("\nBacktest Summary:")
        print(f"Total trades: {len(results.trades)}")
        print(f"Win rate: {results.win_rate:.2f}%")
        print(f"Profit factor: {results.profit_factor:.2f}")
        print(f"Max drawdown: {results.max_drawdown:.2f}%")
        print(f"\n✅ Backtest completed successfully! CLI will now exit.")

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        cleanup_resources()
        sys.exit(1)
    finally:
        # Ensure proper cleanup
        cleanup_resources()
        logger.debug("Backtest command cleanup completed")


@cli.command()
@click.argument("indicator_name", required=True)
@click.option("--symbol", default="ETH-USD", help="Trading symbol")
@click.option("--timeframe", default="1h", help="Timeframe for analysis")
@click.option("--output-dir", help="Directory to save demo results")
def demo(
    indicator_name: str,
    symbol: str,
    timeframe: str,
    output_dir: Optional[str],
):
    """Run a demonstration backtest with preset configurations."""
    logger.info(f"Running demo for {indicator_name} on {symbol} {timeframe}")

    try:
        # Create data directory if it doesn't exist in test data directory
        data_dir = os.path.join(spark_app_dir, "tests", "__test_data__", "market_data", "demo")
        os.makedirs(data_dir, exist_ok=True)

        # Standardize symbol format - replace / or - with _ for filenames
        file_symbol = symbol.replace("/", "_").replace("-", "_")

        # Create a synthetic data file for the demo
        create_demo_data_file(data_dir, file_symbol, timeframe)

        # Initialize data manager and create default data source for demo data
        data_manager = DataManager(data_dir=data_dir)

        # Initialize demo data source
        csv_data_source = CSVDataSource(data_dir=data_dir)
        data_manager.register_data_source("default", csv_data_source)

        # Initialize backtest engine with data manager
        backtest_engine = BacktestEngine(
            data_manager=data_manager,
            initial_balance={"USD": 10000.0},
            maker_fee=0.0001,
            taker_fee=0.0005
        )

        # Create manager with the backtest engine
        manager = IndicatorBacktestManager(backtest_engine=backtest_engine)

        # Set output directory - use default if none provided
        if not output_dir:
            output_dir = get_default_output_dir()
            logger.info(f"Using default output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        manager.set_output_directory(output_dir)

                # Run indicator backtest but don't generate basic report - we'll generate HTML instead
        result_paths = manager.run_indicator_backtest(
            indicator_name=indicator_name,
            symbol=file_symbol,  # Use the same symbol format as the file
            timeframe=timeframe,
            generate_report=False  # We'll generate our own HTML report
        )

        # Get the data and backtest result for HTML report generation
        backtest_data = data_manager.get_data(source_name="default", symbol=file_symbol, interval=timeframe)

        # Get the backtest result which contains the trade history
        backtest_result = manager.results.get(indicator_name)
        if not backtest_result:
            logger.error(f"No backtest result found for {indicator_name}")
            return

        # Get the raw trades from the backtest result
        raw_trades = backtest_result.trades

        # Get the indicator configuration from the manager
        indicator_instance = manager.get_indicator(indicator_name)
        indicator_config = extract_indicator_config(indicator_name, indicator_instance)

        # Generate beautiful HTML report
        html_report_path = generate_html_report_for_cli(
            indicator_name=indicator_name,
            symbol=symbol,  # Use original symbol format for display
            timeframe=timeframe,
            backtest_data=backtest_data,
            raw_trades=raw_trades,
            results_dir=output_dir,
            indicator_config=indicator_config
        )

        # Open the report in browser automatically
        absolute_path = os.path.abspath(html_report_path)
        print(f"\n✅ Demo completed successfully!")
        print(f"📊 Indicator: {indicator_name}")
        print(f"📈 Symbol: {symbol}")
        print(f"⏱️  Timeframe: {timeframe}")
        print(f"📋 HTML Report: {absolute_path}")

        # Safely open browser with timeout
        print(f"🌐 Opening report in browser...")
        browser_success = safe_open_browser(f"file://{absolute_path}")
        if not browser_success:
            print(f"⚠️  Browser failed to open automatically. Please open manually: {absolute_path}")

        # Also print basic results
        if result_paths:
            print(f"📄 JSON results: {result_paths.get('json_path')}")

        print(f"\n🎉 Demo completed successfully! CLI will now exit.")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        cleanup_resources()
        sys.exit(1)
    finally:
        # Ensure proper cleanup
        cleanup_resources()
        logger.debug("Demo command cleanup completed")


@cli.command(name="real-data")
@click.argument("indicator_name", required=True)
@click.option("--symbol", default="ETH-USD", help="Trading symbol")
@click.option("--timeframe", default="1m", help="Timeframe for analysis")
@click.option("--days", default=10, help="Number of days of historical data to fetch")
@click.option("--output-dir", help="Directory to save backtest results")
@click.option("--testnet", is_flag=True, default=None, help="Use Hyperliquid testnet (defaults to HYPERLIQUID_TESTNET env var)")
def real_data(
    indicator_name: str,
    symbol: str,
    timeframe: str,
    days: int,
    output_dir: Optional[str],
    testnet: bool,
):
    """Run a backtest with real market data from Hyperliquid."""

    # If testnet is not explicitly set, use environment variable
    if testnet is None:
        testnet = os.getenv('HYPERLIQUID_TESTNET', 'false').lower() in ('true', '1', 't', 'yes', 'y')

    logger.info(f"Running {indicator_name} backtest with real Hyperliquid data")
    logger.info(f"Parameters: {symbol} {timeframe}, {days} days, testnet={testnet}")
    logger.info(f"Using {'testnet' if testnet else 'mainnet'} based on {'CLI flag' if testnet != (os.getenv('HYPERLIQUID_TESTNET', 'true').lower() in ('true', '1', 't', 'yes', 'y')) else 'environment variable'}")

    try:
        # Create data directory for real data in test data directory
        data_dir = os.path.join(spark_app_dir, "tests", "__test_data__", "market_data", "real")
        os.makedirs(data_dir, exist_ok=True)

        # Fetch real data from Hyperliquid
        data_file_path = fetch_hyperliquid_data(
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            data_dir=data_dir,
            testnet=testnet
        )

        if not data_file_path or not os.path.exists(data_file_path):
            logger.error("Failed to fetch real market data")
            sys.exit(1)

        # Initialize data manager with real data
        data_manager = DataManager(data_dir=data_dir)
        csv_data_source = CSVDataSource(data_dir=data_dir)
        data_manager.register_data_source("default", csv_data_source)

        # Initialize backtest engine with data manager
        backtest_engine = BacktestEngine(
            data_manager=data_manager,
            initial_balance={"USD": 10000.0},
            maker_fee=0.0001,
            taker_fee=0.0005
        )

        # Create manager with the backtest engine
        manager = IndicatorBacktestManager(backtest_engine=backtest_engine)

        # Set output directory - use default if none provided
        if not output_dir:
            output_dir = get_default_output_dir()
            logger.info(f"Using default output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        manager.set_output_directory(output_dir)

        # Standardize symbol format for filename matching
        file_symbol = symbol.replace("/", "_").replace("-", "_")

                # Run indicator backtest with real data but don't generate basic report
        result_paths = manager.run_indicator_backtest(
            indicator_name=indicator_name,
            symbol=file_symbol,
            timeframe=timeframe,
            generate_report=False  # We'll generate our own HTML report
        )

        # Get the data and backtest result for HTML report generation
        backtest_data = data_manager.get_data(source_name="default", symbol=file_symbol, interval=timeframe)

        # Get the backtest result which contains the trade history
        backtest_result = manager.results.get(indicator_name)
        if not backtest_result:
            logger.error(f"No backtest result found for {indicator_name}")
            return

        # Get the raw trades from the backtest result
        raw_trades = backtest_result.trades

        # Get the indicator configuration from the manager
        indicator_instance = manager.get_indicator(indicator_name)
        indicator_config = extract_indicator_config(indicator_name, indicator_instance)

        # Generate beautiful HTML report
        html_report_path = generate_html_report_for_cli(
            indicator_name=indicator_name,
            symbol=symbol,  # Use original symbol format for display
            timeframe=timeframe,
            backtest_data=backtest_data,
            raw_trades=raw_trades,
            results_dir=output_dir,
            indicator_config=indicator_config
        )

        # Open the report in browser automatically
        absolute_path = os.path.abspath(html_report_path)

        # Print results
        if result_paths or html_report_path:
            print(f"\n🎯 Real data backtest completed successfully!")
            print(f"📊 Data source: Hyperliquid {'testnet' if testnet else 'mainnet'}")
            print(f"📈 Symbol: {symbol}")
            print(f"⏱️  Timeframe: {timeframe}")
            print(f"📅 Days of data: {days}")
            print(f"📋 HTML Report: {absolute_path}")

            # Safely open browser with timeout
            print(f"🌐 Opening report in browser...")
            browser_success = safe_open_browser(f"file://{absolute_path}")
            if not browser_success:
                print(f"⚠️  Browser failed to open automatically. Please open manually: {absolute_path}")

            if result_paths:
                print(f"📄 JSON results: {result_paths.get('json_path')}")
            print(f"💾 Data file: {data_file_path}")

            print(f"\n🎉 Real data backtest completed successfully! CLI will now exit.")
        else:
            print("❌ Backtest completed but no results generated")

    except Exception as e:
        import traceback
        logger.error(f"Real data backtest failed: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        cleanup_resources()
        sys.exit(1)
    finally:
        # Ensure proper cleanup
        cleanup_resources()
        logger.debug("Real data command cleanup completed")


def fetch_hyperliquid_data(
    symbol: str,
    timeframe: str,
    days: int,
    data_dir: str,
    testnet: bool = True
) -> Optional[str]:
    """
    Fetch real market data from Hyperliquid and save it as CSV.

    Args:
        symbol: Trading symbol (e.g., 'ETH-USD')
        timeframe: Timeframe interval (e.g., '1m')
        days: Number of days of historical data
        data_dir: Directory to save the data
        testnet: Whether to use testnet

    Returns:
        Path to the saved CSV file or None if failed
    """
    connector = None
    try:
        logger.info(f"Initializing Hyperliquid connector (testnet={testnet})...")

        # Initialize Hyperliquid connector
        connector = HyperliquidConnector(
            name="cli_hyperliquid",
            testnet=testnet
        )

        # Connect to Hyperliquid
        logger.info("Connecting to Hyperliquid...")
        if not connector.connect():
            logger.error("Failed to connect to Hyperliquid")
            return None

        # Calculate time range
        end_time = int(time.time() * 1000)  # Current time in milliseconds
        start_time = end_time - (days * 24 * 60 * 60 * 1000)  # days ago

        logger.info(f"Fetching {days} days of {timeframe} data for {symbol}...")
        logger.info(f"Time range: {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")

        # Fetch historical candles
        candles = connector.get_historical_candles(
            symbol=symbol,
            interval=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=days * 24 * 60 if timeframe == "1m" else 1000  # Adjust limit based on timeframe
        )

        if not candles:
            logger.error(f"No candles received for {symbol}")
            return None

        logger.info(f"Received {len(candles)} candles from Hyperliquid")

        # Convert to DataFrame with the expected format
        df = pd.DataFrame(candles)

        # Ensure we have the required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return None

        # Sort by timestamp
        df = df.sort_values("timestamp")

        # Convert columns to proper types
        df["timestamp"] = df["timestamp"].astype(int)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Create filename
        file_symbol = symbol.replace("/", "_").replace("-", "_")
        filename = f"{file_symbol}_{timeframe}.csv"
        file_path = os.path.join(data_dir, filename)

        # Save to CSV
        df[required_columns].to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} candles to {file_path}")

        # Log data summary
        if len(df) > 0:
            logger.info(f"Data summary:")
            logger.info(f"  - Time range: {datetime.fromtimestamp(df['timestamp'].min()/1000)} to {datetime.fromtimestamp(df['timestamp'].max()/1000)}")
            logger.info(f"  - Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            logger.info(f"  - Average volume: {df['volume'].mean():.2f}")

        return file_path

    except Exception as e:
        logger.error(f"Error fetching Hyperliquid data: {str(e)}")
        return None
    finally:
        # Ensure connector is properly disconnected
        if connector:
            try:
                connector.disconnect()
                logger.debug("Hyperliquid connector disconnected")
            except Exception as e:
                logger.warning(f"Error disconnecting Hyperliquid connector: {e}")


def create_demo_data_file(data_dir: str, symbol: str, timeframe: str) -> str:
    """
    Create a synthetic data file for demo purposes.

    Args:
        data_dir: Directory to save the data file
        symbol: Trading symbol (already formatted for filename)
        timeframe: Timeframe interval

    Returns:
        Path to the created file
    """
    # Use simple filename format
    filename = f"{symbol}_{timeframe}.csv"
    file_path = os.path.join(data_dir, filename)

    # Check if file already exists
    if os.path.exists(file_path):
        logger.info(f"Demo data file already exists: {file_path}")
        return file_path

    # Create synthetic OHLCV data
    import numpy as np
    import pandas as pd

    # Create timestamps for the last 365 days
    end_time = pd.Timestamp.now()
    if timeframe == "1h":
        # Hourly data for the past year
        timestamps = pd.date_range(end=end_time, periods=24*365, freq="H")
    elif timeframe == "1d":
        # Daily data for the past year
        timestamps = pd.date_range(end=end_time, periods=365, freq="D")
    else:
        # Default to daily data
        timestamps = pd.date_range(end=end_time, periods=365, freq="D")

    # Convert timestamps to milliseconds
    timestamps_ms = [int(ts.timestamp() * 1000) for ts in timestamps]

    # Create price data (starting at 1000 with random walk)
    np.random.seed(42)  # For reproducibility
    n = len(timestamps)

    # Generate starting values
    price = 1000.0

    # Initialize arrays with the right length
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)
    volumes = np.zeros(n)

    # Set initial values
    opens[0] = price
    highs[0] = price * 1.01
    lows[0] = price * 0.99
    closes[0] = price
    volumes[0] = np.random.uniform(100, 1000)

    # Generate OHLCV data with a random walk
    for i in range(1, n):
        # Random price change (-2% to +2%)
        change = np.random.uniform(-0.02, 0.02)
        price *= (1 + change)

        # Generate OHLCV values
        opens[i] = price
        highs[i] = price * np.random.uniform(1.0, 1.03)
        lows[i] = price * np.random.uniform(0.97, 1.0)
        closes[i] = np.random.uniform(lows[i], highs[i])
        volumes[i] = np.random.uniform(100, 1000)

    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps_ms,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })

    # Save to CSV
    df.to_csv(file_path, index=False)
    logger.info(f"Created synthetic data file: {file_path}")

    return file_path


@cli.command(name="demo-macd")
@click.option("--symbol", default="ETH/USDT", help="Trading symbol")
@click.option("--timeframe", default="1h", help="Timeframe for analysis")
@click.option("--output-dir", help="Directory to save demo results")
def demo_macd(
    symbol: str,
    timeframe: str,
    output_dir: Optional[str],
):
    """Run a demonstration backtest with MACD indicator."""
    try:
        # This is equivalent to calling demo with indicator_name=MACD
        ctx = click.get_current_context()
        ctx.invoke(demo, indicator_name="MACD", symbol=symbol, timeframe=timeframe, output_dir=output_dir)
    finally:
        # Ensure proper cleanup
        cleanup_resources()
        logger.debug("Demo MACD command cleanup completed")


@cli.command(name="compare-popular")
@click.option("--symbol", default="ETH-USD", help="Trading symbol")
@click.option("--timeframe", default="1h", help="Timeframe for analysis")
@click.option("--output-dir", help="Directory to save comparison results")
@click.option("--use-real-data", is_flag=True, help="Use real market data instead of synthetic demo data")
@click.option("--testnet", is_flag=True, default=None, help="Use Hyperliquid testnet (defaults to HYPERLIQUID_TESTNET env var)")
def compare_popular(
    symbol: str,
    timeframe: str,
    output_dir: Optional[str],
    use_real_data: bool,
    testnet: bool,
):
    """Quick comparison of popular indicators: RSI, MACD, and Bollinger Bands."""

    # If testnet is not explicitly set, use environment variable
    if testnet is None:
        testnet = os.getenv('HYPERLIQUID_TESTNET', 'false').lower() in ('true', '1', 't', 'yes', 'y')

    try:
        # This is equivalent to calling compare with popular indicators
        ctx = click.get_current_context()
        ctx.invoke(
            compare,
            indicator_names="RSI,MACD,BOLLINGER",
            symbol=symbol,
            timeframe=timeframe,
            output_dir=output_dir,
            use_real_data=use_real_data,
            testnet=testnet,
            days=10
        )
    finally:
        # Ensure proper cleanup
        cleanup_resources()
        logger.debug("Compare popular command cleanup completed")


@cli.command()
@click.argument("indicator_names", required=True)
@click.option("--symbol", default="ETH-USD", help="Trading symbol")
@click.option("--timeframe", default="1h", help="Timeframe for analysis")
@click.option("--days", default=10, help="Number of days of historical data (for real data)")
@click.option("--output-dir", help="Directory to save comparison results")
@click.option("--use-real-data", is_flag=True, help="Use real market data instead of synthetic demo data")
@click.option("--testnet", is_flag=True, default=None, help="Use Hyperliquid testnet (defaults to HYPERLIQUID_TESTNET env var)")
def compare(
    indicator_names: str,
    symbol: str,
    timeframe: str,
    days: int,
    output_dir: Optional[str],
    use_real_data: bool,
    testnet: bool,
):
    """Compare multiple indicators side-by-side with individual and comparison reports.

    INDICATOR_NAMES should be comma-separated list, e.g., "RSI,MACD,BOLLINGER"
    """

    # If testnet is not explicitly set, use environment variable
    if testnet is None:
        testnet = os.getenv('HYPERLIQUID_TESTNET', 'false').lower() in ('true', '1', 't', 'yes', 'y')

    logger.info(f"Running comparison for indicators: {indicator_names}")

    try:
        # Parse indicator names
        indicators = [name.strip().upper() for name in indicator_names.split(",")]
        if len(indicators) < 2:
            logger.error("At least 2 indicators are required for comparison")
            print("❌ Error: At least 2 indicators are required for comparison")
            sys.exit(1)

        logger.info(f"Comparing {len(indicators)} indicators: {indicators}")

        # Validate all indicators exist
        available_indicators = [name.upper() for name in IndicatorFactory.get_available_indicators()]
        invalid_indicators = [ind for ind in indicators if ind not in available_indicators]
        if invalid_indicators:
            logger.error(f"Invalid indicators: {invalid_indicators}")
            print(f"❌ Error: Invalid indicators: {invalid_indicators}")
            print(f"Available indicators: {', '.join(available_indicators)}")
            sys.exit(1)

        # Set output directory - use default if none provided
        if not output_dir:
            output_dir = get_default_output_dir()
            logger.info(f"Using default output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # Prepare data source
        if use_real_data:
            logger.info(f"Using real market data from Hyperliquid (testnet={testnet})")
            data_dir = os.path.join(spark_app_dir, "tests", "__test_data__", "market_data", "real")
            os.makedirs(data_dir, exist_ok=True)

            # Fetch real data
            data_file_path = fetch_hyperliquid_data(
                symbol=symbol,
                timeframe=timeframe,
                days=days,
                data_dir=data_dir,
                testnet=testnet
            )

            if not data_file_path or not os.path.exists(data_file_path):
                logger.error("Failed to fetch real market data")
                sys.exit(1)
        else:
            logger.info("Using synthetic demo data")
            data_dir = os.path.join(spark_app_dir, "tests", "__test_data__", "market_data", "demo")
            os.makedirs(data_dir, exist_ok=True)

            # Create demo data
            file_symbol = symbol.replace("/", "_").replace("-", "_")
            data_file_path = create_demo_data_file(data_dir, file_symbol, timeframe)

        # Initialize data manager
        data_manager = DataManager(data_dir=data_dir)
        csv_data_source = CSVDataSource(data_dir=data_dir)
        data_manager.register_data_source("default", csv_data_source)

        # Load configuration for position sizing
        try:
            # Try to load config from shared config path first
            shared_config_path = os.path.join(spark_app_dir, "..", "shared", "config.json")
            if os.path.exists(shared_config_path):
                config_manager = ConfigManager(config_path=shared_config_path)
            else:
                # Fall back to app config path
                app_config_path = os.path.join(spark_app_dir, "config.json")
                config_manager = ConfigManager(config_path=app_config_path if os.path.exists(app_config_path) else None)

            app_config = config_manager.load()

            # Create position sizer from config
            position_config = {}
            if hasattr(app_config, 'position_sizing'):
                position_config = app_config.position_sizing

            # Override with CLI-specific defaults for comparison consistency
            position_config.update({
                'max_position_size_usd': getattr(app_config, 'max_position_size_usd', 1000.0),
                'min_position_size_usd': 10.0,
                'max_leverage': getattr(app_config, 'max_leverage', 1.0)
            })

            sizing_config = PositionSizingConfig.from_config_dict(position_config)
            position_sizer = PositionSizer(sizing_config)

            logger.info(f"Using position sizing method: {sizing_config.method.value}")
            logger.info(f"Position sizing config: USD amount=${sizing_config.fixed_usd_amount}, Max=${sizing_config.max_position_size_usd}")

        except Exception as e:
            logger.warning(f"Failed to load position sizing config: {e}, using defaults")
            # Fall back to default configuration
            default_config = PositionSizingConfig.from_config_dict({
                'position_sizing_method': 'fixed_usd',
                'fixed_usd_amount': 1000.0,
                'max_position_size_usd': 1000.0,
                'min_position_size_usd': 10.0
            })
            position_sizer = PositionSizer(default_config)

        # Initialize backtest engine
        backtest_engine = BacktestEngine(
            data_manager=data_manager,
            initial_balance={"USD": 10000.0},
            maker_fee=0.0001,
            taker_fee=0.0005
        )

        # Create manager with position sizer
        manager = IndicatorBacktestManager(
            backtest_engine=backtest_engine,
            position_sizer=position_sizer
        )
        manager.set_output_directory(output_dir)

        # Standardize symbol format for filename matching
        file_symbol = symbol.replace("/", "_").replace("-", "_")

        # Run backtests for all indicators
        indicator_results = []
        individual_report_paths = []

        print(f"\n🚀 Running backtests for {len(indicators)} indicators...")

        for i, indicator_name in enumerate(indicators, 1):
            print(f"\n📊 [{i}/{len(indicators)}] Running backtest for {indicator_name}...")

            try:
                # Run indicator backtest
                result_paths = manager.run_indicator_backtest(
                    indicator_name=indicator_name,
                    symbol=file_symbol,
                    timeframe=timeframe,
                    generate_report=False  # We'll generate our own reports
                )

                # Get backtest data and results
                backtest_data = data_manager.get_data(source_name="default", symbol=file_symbol, interval=timeframe)
                backtest_result = manager.results.get(indicator_name)

                if not backtest_result:
                    logger.error(f"No backtest result found for {indicator_name}")
                    continue

                # Get raw trades
                raw_trades = backtest_result.trades

                # Get indicator configuration
                indicator_instance = manager.get_indicator(indicator_name)
                indicator_config = extract_indicator_config(indicator_name, indicator_instance)

                # Generate individual HTML report
                html_report_path = generate_html_report_for_cli(
                    indicator_name=indicator_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    backtest_data=backtest_data,
                    raw_trades=raw_trades,
                    results_dir=output_dir,
                    indicator_config=indicator_config
                )

                individual_report_paths.append({
                    "name": indicator_name,
                    "path": html_report_path
                })

                # Prepare data for comparison report
                trades = process_raw_trades_to_position_trades(raw_trades)

                # Calculate metrics
                total_trades = len(trades)
                winning_trades = len([t for t in trades if t.get("realized_pnl", 0) > 0])
                losing_trades = len([t for t in trades if t.get("realized_pnl", 0) <= 0])
                total_pnl = sum(t.get("realized_pnl", 0) for t in trades)
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                profit_factor = calculate_profit_factor(trades)

                initial_balance = 10000.0
                total_return = (total_pnl / initial_balance) * 100
                max_drawdown = calculate_max_drawdown_from_trades(trades, initial_balance)
                sharpe_ratio = calculate_sharpe_ratio_from_trades(trades, initial_balance)

                # Format dates
                start_date = datetime.fromtimestamp(backtest_data['timestamp'].iloc[0] / 1000).strftime("%Y-%m-%d")
                end_date = datetime.fromtimestamp(backtest_data['timestamp'].iloc[-1] / 1000).strftime("%Y-%m-%d")

                # Create result dictionary for comparison
                indicator_result = {
                    "indicator_name": indicator_name,
                    "name": indicator_name,  # Alias for template compatibility
                    "market": symbol,
                    "timeframe": timeframe,
                    "start_date": start_date,
                    "end_date": end_date,
                    "trades": trades,
                    "config": indicator_config,
                    "metrics": {
                        "total_trades": total_trades,
                        "winning_trades": winning_trades,
                        "losing_trades": losing_trades,
                        "win_rate": win_rate,
                        "profit_factor": profit_factor,
                        "max_drawdown": max_drawdown,
                        "sharpe": sharpe_ratio,
                        "total_return": total_return
                    }
                }

                indicator_results.append(indicator_result)
                print(f"✅ {indicator_name}: {total_trades} trades, {win_rate:.1f}% win rate, {total_return:.1f}% return")

            except Exception as e:
                logger.error(f"Failed to run backtest for {indicator_name}: {str(e)}")
                print(f"❌ Failed to run backtest for {indicator_name}: {str(e)}")
                import traceback
                logger.error(f"Backtest error traceback:\n{traceback.format_exc()}")
                continue

        if not indicator_results:
            logger.error("No successful backtests to compare")
            print("❌ No successful backtests to compare")
            sys.exit(1)

        # Generate comparison report
        print(f"\n📋 Generating comparison report...")

        try:
            # Extract price data for market condition analysis
            price_data = backtest_data['close'].tolist() if 'close' in backtest_data.columns else None

            # Import the comparison report generator
            from app.backtesting.reporting.generator import \
                generate_comparison_report

            comparison_report_path = generate_comparison_report(
                indicator_results=indicator_results,
                output_dir=output_dir,
                market_price_data=price_data
            )

            print(f"✅ Comparison report generated: {comparison_report_path}")

        except Exception as e:
            logger.error(f"Failed to generate comparison report: {str(e)}")
            print(f"❌ Failed to generate comparison report: {str(e)}")
            import traceback
            logger.error(f"Comparison report error traceback:\n{traceback.format_exc()}")
            comparison_report_path = None

        # Print results summary
        print(f"\n🎯 Comparison completed successfully!")
        print(f"📊 Data source: {'Hyperliquid ' + ('testnet' if testnet else 'mainnet') if use_real_data else 'Synthetic demo data'}")
        print(f"📈 Symbol: {symbol}")
        print(f"⏱️  Timeframe: {timeframe}")
        if use_real_data:
            print(f"📅 Days of data: {days}")
        print(f"🔍 Indicators: {', '.join([r['indicator_name'] for r in indicator_results])}")

        # Open all reports in browser
        all_report_paths = []

        # Add individual reports
        for report in individual_report_paths:
            absolute_path = os.path.abspath(report["path"])
            all_report_paths.append(("Individual", report["name"], absolute_path))
            print(f"📋 {report['name']} Report: {absolute_path}")

        # Add comparison report
        if comparison_report_path:
            absolute_comparison_path = os.path.abspath(comparison_report_path)
            all_report_paths.append(("Comparison", "All Indicators", absolute_comparison_path))
            print(f"📊 Comparison Report: {absolute_comparison_path}")

        # Open reports in browser with delays to prevent overwhelming
        print(f"\n🌐 Opening {len(all_report_paths)} reports in browser...")

        for i, (report_type, name, path) in enumerate(all_report_paths):
            print(f"[{i+1}/{len(all_report_paths)}] Opening {report_type} - {name}...")

            browser_success = safe_open_browser(f"file://{path}", timeout=3.0)
            if not browser_success:
                print(f"⚠️  Failed to open {report_type} report for {name} automatically")

            # Small delay between browser opens to prevent issues
            if i < len(all_report_paths) - 1:
                time.sleep(1)

        print(f"\n🎉 Comparison completed successfully! All reports generated and opened.")
        print(f"💾 All files saved to: {output_dir}")

        # Summary statistics
        best_return = max(indicator_results, key=lambda x: x['metrics']['total_return'])
        best_win_rate = max(indicator_results, key=lambda x: x['metrics']['win_rate'])
        best_sharpe = max(indicator_results, key=lambda x: x['metrics']['sharpe'])

        print(f"\n🏆 Top Performers:")
        print(f"   📈 Best Return: {best_return['indicator_name']} ({best_return['metrics']['total_return']:.1f}%)")
        print(f"   🎯 Best Win Rate: {best_win_rate['indicator_name']} ({best_win_rate['metrics']['win_rate']:.1f}%)")
        print(f"   ⚡ Best Sharpe: {best_sharpe['indicator_name']} ({best_sharpe['metrics']['sharpe']:.2f})")

        print(f"\n✨ CLI will now exit.")

    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        cleanup_resources()
        sys.exit(1)
    finally:
        # Ensure proper cleanup
        cleanup_resources()
        logger.debug("Compare command cleanup completed")


@cli.command()
def list_indicators():
    """List all available indicators from the factory."""
    try:
        indicators = IndicatorFactory.get_available_indicators()

        print("\nAvailable Indicators:")
        for idx, indicator_name in enumerate(indicators, 1):
            # Display indicator name in uppercase to match the test expectation
            print(f"{idx}. {indicator_name.upper()}")

        print(f"\n✅ Listed {len(indicators)} available indicators. CLI will now exit.")

    except Exception as e:
        logger.error(f"Failed to list indicators: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        cleanup_resources()
        sys.exit(1)
    finally:
        # Ensure proper cleanup
        cleanup_resources()
        logger.debug("List indicators command cleanup completed")


@cli.command(name="list-strategies")
@click.option("--exchange", help="Filter by exchange (e.g., hyperliquid, coinbase)")
@click.option("--market", help="Filter by market (e.g., ETH-USD, BTC-USD)")
@click.option("--enabled/--disabled", default=None, help="Filter by enabled status")
@click.option("--detailed", is_flag=True, help="Show detailed strategy information")
@click.pass_context
def list_strategies_cmd(ctx: click.Context, exchange: Optional[str], market: Optional[str], enabled: Optional[bool], detailed: bool):
    """List all available strategies from configuration."""
    try:
        # Load configuration
        config_path = ctx.obj.get('config_path') if ctx.obj else None
        config = load_config(config_path)

        if not config:
            print("❌ No configuration file found. Strategy listing requires a config.json file.")
            print("   Use --config option to specify configuration file path.")
            return

        # Validate configuration
        errors = validate_config(config)
        if errors:
            print("⚠️  Configuration validation warnings:")
            for error in errors:
                print(f"   • {error}")
            print()

        # Get filtered strategies
        strategies = list_strategies(
            config=config,
            filter_exchange=exchange,
            filter_market=market,
            filter_enabled=enabled
        )

        if not strategies:
            filter_desc = []
            if exchange:
                filter_desc.append(f"exchange={exchange}")
            if market:
                filter_desc.append(f"market={market}")
            if enabled is not None:
                filter_desc.append(f"enabled={enabled}")

            filter_text = f" (filtered by: {', '.join(filter_desc)})" if filter_desc else ""
            print(f"No strategies found{filter_text}")
            return

        # Display strategies
        print(f"\nAvailable Strategies{' (filtered)' if any([exchange, market, enabled is not None]) else ''}:")

        if detailed:
            # Detailed view
            for i, strategy in enumerate(strategies, 1):
                if i > 1:
                    print("\n" + "="*60 + "\n")

                strategy_info = display_strategy_info(strategy, config, show_indicators=True)
                print(strategy_info)
        else:
            # Summary view
            print()
            max_name_len = max(len(s.get('name', '')) for s in strategies)
            max_market_len = max(len(s.get('market', '')) for s in strategies)
            max_exchange_len = max(len(s.get('exchange', '')) for s in strategies)

            # Header
            print(f"{'#':<3} {'Name':<{max_name_len}} {'Market':<{max_market_len}} {'Exchange':<{max_exchange_len}} {'Timeframe':<10} {'Status':<8} {'Indicators'}")
            print("-" * (3 + max_name_len + max_market_len + max_exchange_len + 10 + 8 + 20))

            for i, strategy in enumerate(strategies, 1):
                name = strategy.get('name', 'Unknown')
                market = strategy.get('market', 'Unknown')
                exchange = strategy.get('exchange', 'Unknown')
                timeframe = strategy.get('timeframe', 'Unknown')
                status = '✅ Enabled' if strategy.get('enabled', True) else '❌ Disabled'
                indicators = ', '.join(strategy.get('indicators', []))
                if len(indicators) > 20:
                    indicators = indicators[:17] + "..."

                print(f"{i:<3} {name:<{max_name_len}} {market:<{max_market_len}} {exchange:<{max_exchange_len}} {timeframe:<10} {status:<8} {indicators}")

        print(f"\n✅ Listed {len(strategies)} strategies. Use --detailed for more information.")

        # Show usage hints
        if not detailed and strategies:
            print("\n💡 Tips:")
            print(f"   • Use --detailed for full strategy configuration")
            print(f"   • Use --exchange, --market, or --enabled/--disabled to filter")
            if config.get('strategies'):
                example_strategy = config['strategies'][0].get('name', 'strategy_name')
                print(f"   • Run a strategy backtest with: python cli.py strategy {example_strategy}")

    except Exception as e:
        logger.error(f"Failed to list strategies: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        print(f"❌ Failed to list strategies: {str(e)}")
        cleanup_resources()
        sys.exit(1)
    finally:
        # Ensure proper cleanup
        cleanup_resources()
        logger.debug("List strategies command cleanup completed")


@cli.command()
@click.argument("strategy_name", required=True)
@click.option("--days", default=30, help="Number of days for testing duration across all scenarios")
@click.option("--scenarios", help="Comma-separated list of scenarios to run (e.g., 'bull,bear,real')")
@click.option("--scenario-only", help="Run single scenario instead of full suite (bull, bear, sideways, high_vol, low_vol, choppy, gap_heavy, real)")
@click.option("--override-timeframe", help="Temporarily override strategy timeframe")
@click.option("--override-market", help="Override strategy market for testing")
@click.option("--override-position-size", type=float, help="Override position sizing amount")
@click.option("--use-real-data", is_flag=True, help="Use real market data instead of synthetic scenarios (legacy compatibility)")
@click.option("--export-data", is_flag=True, help="Save scenario data for external analysis")
@click.option("--output-dir", help="Directory to save backtest results")
@click.option("--testnet", is_flag=True, default=None, help="Use Hyperliquid testnet (defaults to HYPERLIQUID_TESTNET env var)")
@click.pass_context
def strategy(
    ctx: click.Context,
    strategy_name: str,
    days: int,
    scenarios: Optional[str],
    scenario_only: Optional[str],
    override_timeframe: Optional[str],
    override_market: Optional[str],
    override_position_size: Optional[float],
    use_real_data: bool,
    export_data: bool,
    output_dir: Optional[str],
    testnet: bool,
):
    """Run comprehensive strategy backtesting with multi-scenario analysis.

    By default, runs strategy across 7 synthetic market scenarios plus real data.
    Use --scenario-only to test a single scenario for quick validation.

    STRATEGY_NAME should match a strategy name from your config.json file.

    Examples:
        python cli.py strategy eth_multi_timeframe_strategy
        python cli.py strategy my_strategy --days 14 --scenarios "bull,bear,real"
        python cli.py strategy my_strategy --scenario-only bull --days 7
    """
    try:
        # If testnet is not explicitly set, use environment variable
        if testnet is None:
            testnet = os.getenv('HYPERLIQUID_TESTNET', 'false').lower() in ('true', '1', 't', 'yes', 'y')

        # Load configuration
        config_path = ctx.obj.get('config_path') if ctx.obj else None
        config = load_config(config_path)

        if not config:
            logger.error("No configuration file found")
            print("❌ No configuration file found. Strategy backtesting requires a config.json file.")
            print("   Use --config option to specify configuration file path.")
            return

        # Validate strategy exists and get configuration
        strategy_config = get_strategy_config(config, strategy_name)
        if not strategy_config:
            available_strategies = [s.get('name', 'Unknown') for s in config.get('strategies', [])]
            logger.error(f"Strategy '{strategy_name}' not found in configuration")
            print(f"❌ Strategy '{strategy_name}' not found in configuration.")
            if available_strategies:
                print(f"   Available strategies: {', '.join(available_strategies)}")
                print(f"   Use 'python cli.py list-strategies' for more details.")
            else:
                print("   No strategies found in configuration file.")
            return

        # Validate strategy configuration
        validation_errors = validate_strategy_config(config, strategy_name)
        if validation_errors:
            print("❌ Strategy configuration validation failed:")
            for error in validation_errors:
                print(f"   • {error}")
            return

        # Apply overrides to strategy config
        effective_strategy = strategy_config.copy()
        if override_timeframe:
            effective_strategy['timeframe'] = override_timeframe
            print(f"🔧 Override: Using timeframe {override_timeframe}")
        if override_market:
            effective_strategy['market'] = override_market
            print(f"🔧 Override: Using market {override_market}")

        print(f"\n🚀 Running strategy backtest for '{strategy_name}'")
        print(f"📊 Strategy: {effective_strategy.get('name')}")
        print(f"📈 Market: {effective_strategy.get('market')}")
        print(f"🏪 Exchange: {effective_strategy.get('exchange')}")
        print(f"⏰ Timeframe: {effective_strategy.get('timeframe')}")
        print(f"🎯 Test Duration: {days} days")

        # Determine which scenarios to run
        if use_real_data:
            # Legacy mode - just run real data
            scenarios_to_run = ['real']
            print(f"📊 Mode: Real data only (legacy compatibility)")
        elif scenario_only:
            # Single scenario mode
            valid_scenarios = ['bull', 'bear', 'sideways', 'high_vol', 'low_vol', 'choppy', 'gap_heavy', 'real']
            if scenario_only not in valid_scenarios:
                print(f"❌ Invalid scenario '{scenario_only}'. Valid options: {', '.join(valid_scenarios)}")
                return
            scenarios_to_run = [scenario_only]
            print(f"📊 Mode: Single scenario ({scenario_only})")
        elif scenarios:
            # Custom scenario selection
            scenarios_to_run = [s.strip() for s in scenarios.split(',')]
            valid_scenarios = ['bull', 'bear', 'sideways', 'high_vol', 'low_vol', 'choppy', 'gap_heavy', 'real']
            invalid_scenarios = [s for s in scenarios_to_run if s not in valid_scenarios]
            if invalid_scenarios:
                print(f"❌ Invalid scenarios: {invalid_scenarios}. Valid options: {', '.join(valid_scenarios)}")
                return
            print(f"📊 Mode: Custom scenarios ({', '.join(scenarios_to_run)})")
        else:
            # Default: full multi-scenario testing
            scenarios_to_run = ['bull', 'bear', 'sideways', 'high_vol', 'low_vol', 'choppy', 'gap_heavy', 'real']
            print(f"📊 Mode: Full multi-scenario analysis (8 scenarios)")

        # Set output directory
        if not output_dir:
            output_dir = get_default_output_dir()
            logger.info(f"Using default output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # TODO: Implement multi-scenario backtesting logic
        # For now, we'll use a placeholder implementation
        print(f"\n⚠️  Multi-scenario backtesting not yet implemented.")
        print(f"📋 This will be implemented in the next phase of development.")
        print(f"🔧 Current implementation is a placeholder for:")
        print(f"   • Market scenario generation (bull, bear, sideways, etc.)")
        print(f"   • Strategy backtesting across all scenarios")
        print(f"   • Cross-scenario performance analysis")
        print(f"   • Robustness scoring and comparison")

        print(f"\n💡 For now, you can use existing commands:")
        print(f"   python cli.py demo <indicator> --symbol {effective_strategy.get('market', 'ETH-USD')}")
        print(f"   python cli.py real-data <indicator> --symbol {effective_strategy.get('market', 'ETH-USD')} --days {days}")

    except Exception as e:
        logger.error(f"Strategy backtest failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        print(f"❌ Strategy backtest failed: {str(e)}")
        cleanup_resources()
        sys.exit(1)
    finally:
        cleanup_resources()
        logger.debug("Strategy command cleanup completed")


@cli.command(name="compare-strategies")
@click.option("--strategy-names", help="Comma-separated list of strategy names to compare")
@click.option("--all-strategies", is_flag=True, help="Compare all enabled strategies")
@click.option("--same-market", help="Filter strategies by market (e.g., ETH-USD)")
@click.option("--same-exchange", help="Filter strategies by exchange (e.g., hyperliquid)")
@click.option("--scenarios", help="Limit comparison to specific scenarios (e.g., 'bull,bear,real')")
@click.option("--days", default=30, help="Number of days for testing duration")
@click.option("--output-dir", help="Directory to save comparison results")
@click.option("--testnet", is_flag=True, default=None, help="Use Hyperliquid testnet (defaults to HYPERLIQUID_TESTNET env var)")
@click.pass_context
def compare_strategies(
    ctx: click.Context,
    strategy_names: Optional[str],
    all_strategies: bool,
    same_market: Optional[str],
    same_exchange: Optional[str],
    scenarios: Optional[str],
    days: int,
    output_dir: Optional[str],
    testnet: bool,
):
    """Compare multiple strategies through multi-scenario analysis.

    Runs all specified strategies through the same set of market scenarios
    for fair comparison and generates comprehensive comparison reports.

    Examples:
        python cli.py compare-strategies --all-strategies
        python cli.py compare-strategies --strategy-names "strategy1,strategy2"
        python cli.py compare-strategies --same-market ETH-USD --scenarios "bull,bear"
    """
    try:
        # If testnet is not explicitly set, use environment variable
        if testnet is None:
            testnet = os.getenv('HYPERLIQUID_TESTNET', 'false').lower() in ('true', '1', 't', 'yes', 'y')

        # Load configuration
        config_path = ctx.obj.get('config_path') if ctx.obj else None
        config = load_config(config_path)

        if not config:
            logger.error("No configuration file found")
            print("❌ No configuration file found. Strategy comparison requires a config.json file.")
            print("   Use --config option to specify configuration file path.")
            return

        # Determine which strategies to compare
        if all_strategies:
            # Get all enabled strategies
            strategies_to_compare = list_strategies(
                config=config,
                filter_enabled=True,
                filter_exchange=same_exchange,
                filter_market=same_market
            )
        elif strategy_names:
            # Parse specified strategy names
            specified_names = [name.strip() for name in strategy_names.split(',')]
            strategies_to_compare = []

            for name in specified_names:
                strategy_config = get_strategy_config(config, name)
                if not strategy_config:
                    print(f"❌ Strategy '{name}' not found in configuration")
                    return

                # Apply filters if specified
                if same_market and strategy_config.get('market', '').upper() != same_market.upper():
                    continue
                if same_exchange and strategy_config.get('exchange', '').lower() != same_exchange.lower():
                    continue

                strategies_to_compare.append(strategy_config)
        else:
            print("❌ Must specify either --all-strategies or --strategy-names")
            print("   Use 'python cli.py list-strategies' to see available strategies")
            return

        if not strategies_to_compare:
            filter_desc = []
            if same_market:
                filter_desc.append(f"market={same_market}")
            if same_exchange:
                filter_desc.append(f"exchange={same_exchange}")

            filter_text = f" matching filters ({', '.join(filter_desc)})" if filter_desc else ""
            print(f"❌ No strategies found{filter_text}")
            return

        # Determine scenarios for comparison
        if scenarios:
            scenarios_to_run = [s.strip() for s in scenarios.split(',')]
            valid_scenarios = ['bull', 'bear', 'sideways', 'high_vol', 'low_vol', 'choppy', 'gap_heavy', 'real']
            invalid_scenarios = [s for s in scenarios_to_run if s not in valid_scenarios]
            if invalid_scenarios:
                print(f"❌ Invalid scenarios: {invalid_scenarios}. Valid options: {', '.join(valid_scenarios)}")
                return
        else:
            # Default: full multi-scenario comparison
            scenarios_to_run = ['bull', 'bear', 'sideways', 'high_vol', 'low_vol', 'choppy', 'gap_heavy', 'real']

        print(f"\n🚀 Running strategy comparison")
        print(f"📊 Strategies: {len(strategies_to_compare)} strategies")
        print(f"🎯 Scenarios: {len(scenarios_to_run)} scenarios ({', '.join(scenarios_to_run)})")
        print(f"⏰ Duration: {days} days per scenario")

        # Display strategies to be compared
        print(f"\n📋 Strategies to compare:")
        for i, strategy in enumerate(strategies_to_compare, 1):
            name = strategy.get('name', 'Unknown')
            market = strategy.get('market', 'Unknown')
            exchange = strategy.get('exchange', 'Unknown')
            timeframe = strategy.get('timeframe', 'Unknown')
            print(f"   {i}. {name} ({market} on {exchange}, {timeframe})")

        # Set output directory
        if not output_dir:
            output_dir = get_default_output_dir()
            logger.info(f"Using default output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # TODO: Implement multi-strategy comparison logic
        # For now, we'll use a placeholder implementation
        print(f"\n⚠️  Multi-strategy comparison not yet implemented.")
        print(f"📋 This will be implemented in the next phase of development.")
        print(f"🔧 Current implementation is a placeholder for:")
        print(f"   • Running all strategies through identical market scenarios")
        print(f"   • Cross-scenario robustness scoring")
        print(f"   • Strategy performance ranking and comparison")
        print(f"   • Portfolio diversification analysis")

        print(f"\n💡 For now, you can compare individual indicators:")
        print(f"   python cli.py compare 'RSI,MACD,BOLLINGER' --use-real-data --days {days}")

    except Exception as e:
        logger.error(f"Strategy comparison failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        print(f"❌ Strategy comparison failed: {str(e)}")
        cleanup_resources()
        sys.exit(1)
    finally:
        cleanup_resources()
        logger.debug("Compare strategies command cleanup completed")


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user")
        cleanup_resources()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in CLI: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        cleanup_resources()
        sys.exit(1)
    finally:
        # Final cleanup and graceful exit
        cleanup_resources()
        logger.debug("CLI exiting gracefully")
        sys.exit(0)
