"""
Command-line interface for generating indicator backtest reports.

This script provides a convenient interface for generating static HTML reports
from backtest results stored in JSON files.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from app.backtesting.reporting.generator import (generate_comparison_report,
                                                 generate_indicator_report)
from app.backtesting.reporting.metrics import calculate_performance_metrics
from app.backtesting.reporting.transformer import transform_backtest_results
from app.backtesting.reporting.visualizations import (generate_drawdown_chart,
                                                      generate_equity_curve,
                                                      generate_price_chart)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_backtest_results(file_path: str) -> Dict[str, Any]:
    """
    Load backtest results from a JSON file.

    Args:
        file_path: Path to the backtest results JSON file

    Returns:
        Dictionary containing backtest results data
    """
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded backtest results from {file_path}")
        return results
    except Exception as e:
        logger.error(f"Failed to load backtest results from {file_path}: {e}")
        raise


def load_market_data(file_path: str) -> pd.DataFrame:
    """
    Load market data from a CSV file.

    Args:
        file_path: Path to the market data CSV file

    Returns:
        DataFrame containing market OHLCV data
    """
    try:
        df = pd.read_csv(file_path)

        # Ensure timestamps are datetime objects
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        logger.info(f"Loaded market data from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load market data from {file_path}: {e}")
        raise


def generate_charts_for_report(
    market_data: pd.DataFrame,
    backtest_results: Dict[str, Any],
    output_dir: str
) -> Dict[str, str]:
    """
    Generate charts for the backtest report.

    Args:
        market_data: DataFrame containing market OHLCV data
        backtest_results: Dictionary containing backtest results
        output_dir: Directory to save chart files

    Returns:
        Dictionary with paths to generated chart files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Extract trades from results
    trades = backtest_results.get("trades", [])

    # Generate file paths
    indicator_name = backtest_results.get("indicator_name", "unknown").lower().replace(" ", "_")
    market = backtest_results.get("market", "unknown").lower().replace("/", "_")

    price_chart_file = os.path.join(charts_dir, f"{indicator_name}_{market}_price_chart.html")
    equity_curve_file = os.path.join(charts_dir, f"{indicator_name}_{market}_equity_curve.html")
    drawdown_file = os.path.join(charts_dir, f"{indicator_name}_{market}_drawdown.html")

    # Generate price chart
    price_chart_path = generate_price_chart(
        df=market_data,
        trades=trades,
        filename=price_chart_file
    )

    # Generate equity curve
    equity_curve_path, equity_df = generate_equity_curve(
        trades=trades,
        filename=equity_curve_file
    )

    # Generate drawdown chart
    drawdown_path = generate_drawdown_chart(
        equity_curve=equity_df,
        filename=drawdown_file
    )

    # Return chart paths dictionary
    return {
        "price_chart": os.path.relpath(price_chart_path, output_dir),
        "equity_curve": os.path.relpath(equity_curve_path, output_dir),
        "drawdown_chart": os.path.relpath(drawdown_path, output_dir)
    }


def process_backtest_results(
    results_file: str,
    market_data_file: str,
    output_dir: str
) -> str:
    """
    Process backtest results and generate a report.

    Args:
        results_file: Path to the backtest results JSON file
        market_data_file: Path to the market data CSV file
        output_dir: Directory to save the report

    Returns:
        Path to the generated report file
    """
    # Load data
    results = load_backtest_results(results_file)
    market_data = load_market_data(market_data_file)

    # Calculate metrics if not present
    if "metrics" not in results or not results["metrics"]:
        results["metrics"] = calculate_performance_metrics(results.get("trades", []))

    # Generate charts
    charts = generate_charts_for_report(market_data, results, output_dir)

    # Transform results for the template
    transformed_results = transform_backtest_results(results)

    # Add charts to the results
    transformed_results["charts"] = charts

    # Generate the report
    report_path = generate_indicator_report(
        indicator_results=transformed_results,
        charts=charts,
        output_dir=output_dir
    )

    return report_path


def process_multiple_results(
    results_files: List[str],
    output_dir: str
) -> str:
    """
    Process multiple backtest results and generate a comparison report.

    Args:
        results_files: List of paths to backtest results JSON files
        output_dir: Directory to save the report

    Returns:
        Path to the generated comparison report file
    """
    # Load all results
    results_list = []
    for file_path in results_files:
        try:
            results = load_backtest_results(file_path)

            # Calculate metrics if not present
            if "metrics" not in results or not results["metrics"]:
                results["metrics"] = calculate_performance_metrics(results.get("trades", []))

            # Transform results
            transformed_results = transform_backtest_results(results)
            results_list.append(transformed_results)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")

    if not results_list:
        raise ValueError("No valid results files were processed")

    # Generate comparison report
    report_path = generate_comparison_report(
        indicator_results=results_list,
        output_dir=output_dir
    )

    return report_path


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate HTML reports from backtest results")

    # Required arguments
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to the backtest results JSON file (or comma-separated list for comparison)"
    )

    # Optional arguments
    parser.add_argument(
        "--market-data",
        type=str,
        help="Path to the market data CSV file (required for single indicator reports)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the report (defaults to tests/test_results/backtesting_reports)"
    )

    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Generate a comparison report from multiple result files"
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        # Set default output directory if not specified
        if args.output_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            args.output_dir = str(project_root / "tests" / "__test_results__" / "backtesting_reports")

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Process based on mode
        if args.comparison:
            # Split the comma-separated list of result files
            results_files = [file.strip() for file in args.results.split(",")]

            # Generate comparison report
            report_path = process_multiple_results(results_files, args.output_dir)
            logger.info(f"Generated comparison report: {report_path}")

        else:
            # Single indicator report
            if not args.market_data:
                parser.error("--market-data is required for single indicator reports")

            # Generate single indicator report
            report_path = process_backtest_results(
                args.results,
                args.market_data,
                args.output_dir
            )
            logger.info(f"Generated indicator report: {report_path}")

        print(f"Report generated successfully: {report_path}")
        return 0

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
