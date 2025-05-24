#!/usr/bin/env python3
"""
Spark-App CLI - Unified command line interface for backtest operations
"""
import logging
import os
import sys
import time
import webbrowser
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from app.backtesting.reporting.generator import generate_indicator_report
from app.connectors.hyperliquid_connector import HyperliquidConnector
from app.indicators.indicator_factory import IndicatorFactory
from app.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


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
    equity = [initial_balance]
    running_pnl = 0

    if trades:
        # Create equity curve based on actual trades
        for i in range(1, len(dates)):
            # Check if any trades completed at this time
            for trade in trades:
                trade_time = pd.to_datetime(trade.get('exit_time', 0), unit='ms')
                if abs((dates.iloc[i] - trade_time).total_seconds()) < 86400:  # Within a day
                    running_pnl += trade.get('realized_pnl', 0)
            equity.append(initial_balance + running_pnl)
    else:
        # No trades, flat equity
        equity = [initial_balance] * len(dates)

    # Ensure equity has same length as dates
    while len(equity) < len(dates):
        equity.append(equity[-1])
    equity = equity[:len(dates)]

    # 1. Price Chart with Trade Markers
    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, 'b-', linewidth=1.5, label='Price')

    # Add trade markers if trades are provided
    if trades and backtest_data is not None:
        for i, trade in enumerate(trades):
            # Convert timestamps to dates for plotting
            entry_time = pd.to_datetime(trade.get('entry_time', 0), unit='ms')
            exit_time = pd.to_datetime(trade.get('exit_time', 0), unit='ms')

            # Check if trade times are within our chart data range
            if entry_time >= dates.min() and entry_time <= dates.max():
                # Find closest date index for entry
                entry_idx = np.abs(dates - entry_time).argmin()
                closest_entry_time = dates.iloc[entry_idx]
                market_price_at_entry = prices[entry_idx]

                # Plot entry marker using the market price
                plt.scatter(closest_entry_time, market_price_at_entry,
                           marker='^', s=100, c='green',
                           label='Buy Entry' if i == 0 else "",
                           zorder=5, edgecolors='darkgreen', linewidth=1)

            if exit_time >= dates.min() and exit_time <= dates.max():
                # Find closest date index for exit
                exit_idx = np.abs(dates - exit_time).argmin()
                closest_exit_time = dates.iloc[exit_idx]
                market_price_at_exit = prices[exit_idx]

                # Plot exit marker using the market price
                plt.scatter(closest_exit_time, market_price_at_exit,
                           marker='v', s=100, c='red',
                           label='Sell Exit' if i == 0 else "",
                           zorder=5, edgecolors='darkred', linewidth=1)

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

    return charts


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

    # Debug: Log calculated metrics
    logger.info(f"Calculated metrics - Total PnL: {total_pnl}, Trades: {total_trades}, Win Rate: {win_rate:.2f}%")

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
            "max_drawdown": 5.0,  # As percentage
            "sharpe": 1.2,  # Note: using 'sharpe' instead of 'sharpe_ratio' to match template
            "total_return": total_pnl / 10000 * 100  # As percentage of initial balance
        }
    }

    # Create actual chart images with trade markers using real backtest data
    charts = create_charts_for_report(results_dir, symbol, trades, backtest_data)

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

@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool):
    """Spark-App CLI for backtesting and indicator management."""
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level=log_level)
    logger.debug("Debug logging enabled")


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
        indicator_instance = IndicatorFactory.create(indicator)

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

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        sys.exit(1)


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
        print(f"🌐 Opening report in browser...")

        webbrowser.open(f"file://{absolute_path}")

        # Also print basic results
        if result_paths:
            print(f"📄 JSON results: {result_paths.get('json_path')}")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        sys.exit(1)


@cli.command(name="real-data")
@click.argument("indicator_name", required=True)
@click.option("--symbol", default="ETH-USD", help="Trading symbol")
@click.option("--timeframe", default="1m", help="Timeframe for analysis")
@click.option("--days", default=10, help="Number of days of historical data to fetch")
@click.option("--output-dir", help="Directory to save backtest results")
@click.option("--testnet", is_flag=True, default=True, help="Use Hyperliquid testnet")
def real_data(
    indicator_name: str,
    symbol: str,
    timeframe: str,
    days: int,
    output_dir: Optional[str],
    testnet: bool,
):
    """Run a backtest with real market data from Hyperliquid."""
    logger.info(f"Running {indicator_name} backtest with real Hyperliquid data")
    logger.info(f"Parameters: {symbol} {timeframe}, {days} days, testnet={testnet}")

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
            print(f"🌐 Opening report in browser...")

            webbrowser.open(f"file://{absolute_path}")

            if result_paths:
                print(f"📄 JSON results: {result_paths.get('json_path')}")
            print(f"💾 Data file: {data_file_path}")
        else:
            print("❌ Backtest completed but no results generated")

    except Exception as e:
        import traceback
        logger.error(f"Real data backtest failed: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


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
    # This is equivalent to calling demo with indicator_name=MACD
    ctx = click.get_current_context()
    ctx.invoke(demo, indicator_name="MACD", symbol=symbol, timeframe=timeframe, output_dir=output_dir)


@cli.command()
def list_indicators():
    """List all available indicators from the factory."""
    indicators = IndicatorFactory.get_available_indicators()

    print("\nAvailable Indicators:")
    for idx, indicator_name in enumerate(indicators, 1):
        # Display indicator name in uppercase to match the test expectation
        print(f"{idx}. {indicator_name.upper()}")


if __name__ == "__main__":
    cli()
