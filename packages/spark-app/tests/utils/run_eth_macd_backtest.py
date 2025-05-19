#!/usr/bin/env python3
"""
Script to run a backtest for ETH-USD with MACD indicator on 1-minute timeframe
and generate an HTML report.

This script has been refactored to use the app CLI interface as part of the
backtesting suite refactor (Phase 3.5.1, task 4.5-C).
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def run_macd_backtest(
    symbol="ETH-USD", timeframe="1h", output_dir=None, verbose=False
):
    """
    Run the ETH-USD MACD backtest via the CLI.

    Args:
        symbol: Trading symbol to use
        timeframe: Timeframe to use
        output_dir: Directory to save results to
        verbose: Whether to enable verbose output

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Running MACD backtest for {symbol} with {timeframe} timeframe")

    # Find Python interpreter
    python_path = sys.executable
    if os.path.exists(os.path.join(ROOT_DIR, ".venv", "bin", "python")):
        python_path = os.path.join(ROOT_DIR, ".venv", "bin", "python")
    elif os.path.exists(os.path.join(ROOT_DIR, ".venv", "Scripts", "python.exe")):
        python_path = os.path.join(ROOT_DIR, ".venv", "Scripts", "python.exe")

    # Build CLI command
    cmd = [python_path, "-m", "app.cli", "demo-macd"]

    # Add command line options
    cmd.extend(["--symbol", symbol, "--timeframe", timeframe])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cmd.extend(["--output-dir", output_dir])

    if verbose:
        cmd.append("--debug")

    try:
        logger.info(f"Executing command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=ROOT_DIR,
        )
        stdout, stderr = process.communicate()

        print(stdout)
        if stderr:
            print(stderr)

        if process.returncode == 0:
            logger.info("Backtest completed successfully")
            return True
        else:
            logger.error(f"Backtest failed with exit code {process.returncode}")
            return False

    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        return False


def main():
    """Run the ETH-USD MACD backtest via CLI command."""
    parser = argparse.ArgumentParser(description="Run MACD backtest for ETH-USD")
    parser.add_argument(
        "--symbol", help="Trading symbol (default: ETH-USD)", default="ETH-USD"
    )
    parser.add_argument(
        "--timeframe", help="Timeframe (default: 1h)", default="1h"
    )
    parser.add_argument(
        "--output-dir", help="Directory to save results to (default: tests/__test_results__/backtesting_reports)"
    )
    parser.add_argument(
        "--verbose", "-v", help="Enable verbose output", action="store_true"
    )

    args = parser.parse_args()

    # Default output directory if not specified
    if not args.output_dir:
        args.output_dir = os.path.join(ROOT_DIR, "tests", "__test_results__", "backtesting_reports")

    # Run the backtest
    success = run_macd_backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
