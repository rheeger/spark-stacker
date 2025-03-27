#!/usr/bin/env python3
"""
Test runner script for Spark Stacker.

This script runs all the tests and generates a coverage report.
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestRunner")

# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Market data cache directory
MARKET_DATA_CACHE_DIR = os.path.join(ROOT_DIR, "tests", "test_data", "market_data")
# Required exchanges and timeframes for tests
REQUIRED_DATA_FILES = {
    "hyperliquid_ETH_USD_1h.csv",  # Default test data
    "hyperliquid_BTC_USD_1h.csv",  # Common test data
}
CACHE_MAX_AGE_HOURS = 24  # Refresh data if older than this many hours


def check_market_data_cache():
    """
    Check if market data cache exists and is up to date.

    Returns:
        bool: True if cache needs refresh, False otherwise
    """
    # Create cache directory if it doesn't exist
    cache_dir = Path(MARKET_DATA_CACHE_DIR)
    if not cache_dir.exists():
        logger.info(f"Market data cache directory not found: {cache_dir}")
        return True

    # Check if required files exist
    missing_files = []
    oldest_file_time = None

    # Check each required file
    for filename in REQUIRED_DATA_FILES:
        file_path = cache_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
            continue

        # Check file modification time
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        if oldest_file_time is None or mtime < oldest_file_time:
            oldest_file_time = mtime

    # Report missing files
    if missing_files:
        logger.info(f"Missing required market data files: {', '.join(missing_files)}")
        return True

    # Check if oldest file is too old
    if oldest_file_time:
        age = datetime.now() - oldest_file_time
        if age > timedelta(hours=CACHE_MAX_AGE_HOURS):
            logger.info(
                f"Market data cache is {age.total_seconds() / 3600:.1f} hours old (older than {CACHE_MAX_AGE_HOURS} hours)"
            )
            return True

    logger.info("Market data cache is up to date")
    return False


def refresh_market_data():
    """
    Refresh the market data cache by running the refresh script.

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Refreshing market data cache...")
    refresh_script = os.path.join(ROOT_DIR, "scripts", "refresh_test_market_data.py")

    try:
        result = subprocess.run(
            [sys.executable, refresh_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )

        if result.returncode == 0:
            logger.info("Market data cache refreshed successfully")
            return True
        else:
            logger.warning(
                f"Market data refresh completed with warnings/errors (code {result.returncode})"
            )
            return result.returncode == 0

    except Exception as e:
        logger.error(f"Error refreshing market data: {str(e)}")
        return False


def run_tests(
    test_path=None,
    coverage=True,
    verbose=False,
    failfast=False,
    allow_synthetic_data=False,
):
    """Run tests with optional coverage report."""
    logger.info("Running tests...")

    # Construct the command
    cmd = ["python", "-m"]

    if coverage:
        cmd.extend(["pytest", "--cov=app", "--cov-report=term", "--cov-report=html"])
    else:
        cmd.append("pytest")

    # Add additional arguments
    if verbose:
        cmd.append("-v")

    if failfast:
        cmd.append("-x")

    # Add flag to allow synthetic data if requested
    if allow_synthetic_data:
        cmd.append("--allow-synthetic-data")

    # Add the test path if specified
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")

    # Run the command
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
            logger.info("Tests passed!")
            if coverage:
                logger.info("Coverage report generated in htmlcov/ directory")
            return True
        else:
            logger.error("Tests failed!")
            return False

    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return False


def lint_code(path="app/"):
    """Run linting checks on the code."""
    logger.info("Running linting checks...")

    # Construct the command
    cmd = ["pylint", path]

    # Run the command
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
            logger.info("Linting checks passed!")
            return True
        else:
            logger.warning("Linting checks found issues.")
            return False

    except Exception as e:
        logger.error(f"Error running linting: {str(e)}")
        return False


def type_check(path="app/"):
    """Run type checking on the code."""
    logger.info("Running type checking...")

    # Construct the command
    cmd = ["mypy", path]

    # Run the command
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
            logger.info("Type checking passed!")
            return True
        else:
            logger.warning("Type checking found issues.")
            return False

    except Exception as e:
        logger.error(f"Error running type checking: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for Spark Stacker")
    parser.add_argument(
        "--path", "-p", help="Path to test file or directory", default=None
    )
    parser.add_argument(
        "--no-coverage", "-nc", help="Disable coverage report", action="store_true"
    )
    parser.add_argument(
        "--verbose", "-v", help="Show verbose output", action="store_true"
    )
    parser.add_argument(
        "--fail-fast", "-x", help="Stop at first failure", action="store_true"
    )
    parser.add_argument("--lint", "-l", help="Run linting checks", action="store_true")
    parser.add_argument(
        "--type-check", "-t", help="Run type checking", action="store_true"
    )
    parser.add_argument(
        "--all-checks",
        "-a",
        help="Run all checks (tests, linting, type checking)",
        action="store_true",
    )
    parser.add_argument(
        "--refresh-data",
        "-r",
        help="Force refresh market data cache",
        action="store_true",
    )
    parser.add_argument(
        "--skip-data-refresh",
        "-s",
        help="Skip market data refresh even if cache is old",
        action="store_true",
    )
    parser.add_argument(
        "--allow-synthetic-data",
        help="Allow tests to use synthetic data if cache is missing",
        action="store_true",
    )

    args = parser.parse_args()

    # Check and refresh market data cache if needed
    if args.refresh_data or (not args.skip_data_refresh and check_market_data_cache()):
        refresh_result = refresh_market_data()
        if not refresh_result and not args.allow_synthetic_data:
            logger.warning(
                "⚠️ Market data refresh failed. Tests may fail unless --allow-synthetic-data is used."
            )

    # Run type checking if requested
    if args.type_check or args.all_checks:
        type_check()
        print("\n" + "-" * 80 + "\n")

    # Run linting if requested
    if args.lint or args.all_checks:
        lint_code()
        print("\n" + "-" * 80 + "\n")

    # Always run tests
    run_tests(
        test_path=args.path,
        coverage=not args.no_coverage,
        verbose=args.verbose,
        failfast=args.fail_fast,
        allow_synthetic_data=args.allow_synthetic_data,
    )
