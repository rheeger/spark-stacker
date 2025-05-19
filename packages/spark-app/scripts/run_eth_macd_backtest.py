#!/usr/bin/env python3
"""
Script to run a backtest for ETH-USD with MACD indicator on 1-minute timeframe
and generate an HTML report.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# Add parent directory to path to allow imports
current_dir = Path(os.path.abspath(__file__)).parent
project_root = current_dir.parent
sys.path.append(str(project_root.parent.parent))  # Add the root directory to path

# Use relative imports instead of package imports
sys.path.append(str(project_root))
from app.backtesting.backtest_engine import BacktestEngine
from app.backtesting.data_manager import DataManager
from app.backtesting.indicator_backtest_manager import IndicatorBacktestManager
# Import reporting modules
from app.backtesting.reporting.generator import generate_indicator_report
from app.backtesting.reporting.metrics import calculate_performance_metrics
from app.backtesting.reporting.transformer import transform_backtest_results
from app.backtesting.reporting.visualizations import (generate_drawdown_chart,
                                                      generate_equity_curve,
                                                      generate_price_chart)
from app.indicators.macd_indicator import MACDIndicator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_data(output_path, interval="1m", days=3):
    """
    Create sample ETH-USD data for demonstration purposes.

    Args:
        output_path: Path to save the data
        interval: Time interval between data points
        days: Number of days of data to generate
    """
    # Determine the number of intervals based on the interval string
    if interval.endswith('m'):
        minutes = int(interval[:-1])
        periods = int(days * 24 * 60 / minutes)
    elif interval.endswith('h'):
        hours = int(interval[:-1])
        periods = int(days * 24 / hours)
    else:
        periods = days

    # Create a date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    if interval.endswith('m'):
        minutes = int(interval[:-1])
        dates = pd.date_range(start=start_date, end=end_date, freq=f'{minutes}T')
    elif interval.endswith('h'):
        hours = int(interval[:-1])
        dates = pd.date_range(start=start_date, end=end_date, freq=f'{hours}H')
    else:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create price data with some trends and noise
    import numpy as np
    np.random.seed(42)  # For reproducibility

    # Generate a price series with trends, cycles, and noise
    t = np.linspace(0, days, len(dates))

    # Base price around $3000
    base_price = 3000

    # Add trend component
    trend = 100 * np.sin(t/days * np.pi * 2)

    # Add cycles
    cycles = 50 * np.sin(t * np.pi) + 30 * np.sin(t * 5 * np.pi)

    # Add some random noise
    noise = np.random.randn(len(dates)) * 5

    # Combine components
    close_prices = base_price + trend + cycles + np.cumsum(noise)

    # Generate open, high, low prices
    volatility = np.random.randn(len(dates)) * 10
    high_prices = close_prices + np.abs(volatility)
    low_prices = close_prices - np.abs(volatility)

    # Open prices based on previous close with some noise
    open_prices = np.roll(close_prices, 1) + np.random.randn(len(dates)) * 3
    open_prices[0] = close_prices[0] - np.random.rand() * 5

    # Generate some volume
    volume = np.random.randint(1, 10, size=len(dates)) * 10 + np.abs(np.random.randn(len(dates)) * 50)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': [int(d.timestamp() * 1000) for d in dates],  # Convert to ms timestamp
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Created sample data file: {output_path}")
    return output_path


def main():
    """Run the ETH-USD MACD backtest and generate a report."""
    # Setup paths
    project_root = Path(os.path.abspath(__file__)).parents[1]

    # Define paths for data and results
    data_dir = project_root / "tests" / "__test_data__" / "market_data"
    results_dir = project_root / "tests" / "__test_results__" / "backtesting_reports"
    # Note: CSV file name must match pattern {symbol}_{interval}.csv to work with CSVDataSource
    sample_data_path = data_dir / "ETH-USD_1m.csv"

    # Ensure directories exist
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Symbol and interval settings
    symbol = "ETH-USD"
    interval = "1m"

    # Create sample data if it doesn't exist
    if not sample_data_path.exists():
        create_sample_data(sample_data_path, interval=interval, days=3)

    # Initialize backtesting components
    data_manager = DataManager()
    backtest_engine = BacktestEngine(data_manager=data_manager)
    backtest_manager = IndicatorBacktestManager(backtest_engine=backtest_engine)

    # Register the data source
    from app.backtesting.data_manager import CSVDataSource
    csv_data_source = CSVDataSource(os.path.dirname(str(sample_data_path)))
    data_manager.register_data_source("csv", csv_data_source)

    # Create and add MACD indicator
    macd_indicator = MACDIndicator(
        name="MACD",
        fast_period=12,
        slow_period=26,
        signal_period=9
    )
    backtest_manager.add_indicator(macd_indicator)

    # Define backtest period (use the whole data range)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)

    # Run the backtest
    logger.info(f"Running MACD backtest for {symbol} with {interval} data")
    logger.info(f"Time period: {start_date.date()} to {end_date.date()}")

    try:
        # Run backtest
        backtest_result = backtest_manager.backtest_indicator(
            indicator_name="MACD",
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_source_name="csv",
            leverage=1.0
        )

        if backtest_result:
            # Save backtest results to JSON
            results_json_path = results_dir / "macd_eth_usd_1m_results.json"
            backtest_result.save_to_file(str(results_json_path))
            logger.info(f"Saved backtest results to {results_json_path}")

            # Generate indicator performance report
            report_dir = results_dir / "MACD"
            report_dir.mkdir(exist_ok=True)

            report = backtest_manager.generate_indicator_performance_report(
                indicator_name="MACD",
                output_dir=report_dir,
                include_plots=True
            )

            # Now generate an HTML report using the reporting module
            logger.info("Generating HTML report from backtest results")

            # Load the JSON results
            with open(results_json_path, 'r') as f:
                results_data = json.load(f)

            # Load the market data
            market_data = pd.read_csv(sample_data_path)
            if 'timestamp' in market_data.columns:
                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'], unit='ms')

            # Process the backtest results for the HTML template
            # Add default metrics if they're missing
            default_metrics = {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "total_trades": 0
            }

            # Get metrics from results or use defaults
            metrics = results_data.get("metrics", {})
            # Update with any missing default metrics
            for key, value in default_metrics.items():
                if key not in metrics:
                    metrics[key] = value

            transformed_results = {
                "indicator_name": "MACD",
                "market": symbol,
                "timeframe": interval,
                "start_date": results_data.get("start_date"),
                "end_date": results_data.get("end_date"),
                "metrics": metrics,
                "trades": results_data.get("trades", [])
            }

            # Generate charts for the report
            charts_dir = report_dir / "charts"
            charts_dir.mkdir(exist_ok=True)

            # Generate price chart
            price_chart_file = charts_dir / "price_chart.html"
            price_chart_path = generate_price_chart(
                df=market_data,
                trades=results_data.get("trades", []),
                filename=str(price_chart_file)
            )

            # Generate equity curve
            equity_curve_file = charts_dir / "equity_curve.html"
            equity_curve_path, equity_df = generate_equity_curve(
                trades=results_data.get("trades", []),
                filename=str(equity_curve_file)
            )

            # Fix: If equity_df is empty, create a basic DataFrame with required columns
            if equity_df.empty:
                # Create a minimal equity DataFrame with required columns for the drawdown chart
                dates = pd.date_range(start=pd.to_datetime(results_data.get("start_date")),
                                     end=pd.to_datetime(results_data.get("end_date")),
                                     periods=10)
                equity_df = pd.DataFrame({
                    'date': dates,
                    'equity': [10000.0] * len(dates),  # Flat equity line (no trades)
                    'drawdown': [0.0] * len(dates)     # No drawdown
                })

                # If no equity curve was generated, create one now
                if equity_curve_path is None:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=[10000.0] * len(dates),
                            mode='lines',
                            name='Equity'
                        )
                    )
                    fig.update_layout(
                        title="Equity Curve (No Trades)",
                        xaxis_title="Date",
                        yaxis_title="Equity ($)"
                    )
                    fig.write_html(str(equity_curve_file))
                    equity_curve_path = str(equity_curve_file)
                    logger.info(f"Created default equity curve at: {equity_curve_path}")

            # Generate drawdown chart
            drawdown_file = charts_dir / "drawdown.html"
            drawdown_path = generate_drawdown_chart(
                equity_curve=equity_df,
                filename=str(drawdown_file)
            )

            # Create charts dictionary for the report
            charts = {
                "price_chart": os.path.relpath(price_chart_path, str(report_dir)),
                "equity_curve": os.path.relpath(equity_curve_path, str(report_dir)) if equity_curve_path else "",
                "drawdown_chart": os.path.relpath(drawdown_path, str(report_dir))
            }

            # Generate the HTML report
            report_filename = f"macd_eth_usd_1m_report.html"
            html_report_path = generate_indicator_report(
                indicator_results=transformed_results,
                charts=charts,
                output_dir=str(report_dir),
                output_filename=report_filename
            )

            logger.info(f"Generated HTML report: {html_report_path}")
            logger.info(f"Report directory: {report_dir}")

            # Print summary metrics
            print("\nMACD Backtest Results Summary:")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")

            print(f"\nTotal trades: {len(results_data.get('trades', []))}")
            print(f"Report generated at: {html_report_path}")

        else:
            logger.error("Backtest failed to produce results")

    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
