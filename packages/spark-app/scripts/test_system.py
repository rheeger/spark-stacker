#!/usr/bin/env python3
"""
Comprehensive test script that handles data preparation and test execution.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SystemTest")

# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Market data cache directory
MARKET_DATA_CACHE_DIR = os.path.join(ROOT_DIR, "tests", "test_data", "market_data")

# Path to virtual environment Python interpreter
VENV_PYTHON = os.path.join(ROOT_DIR, ".venv", "bin", "python")
if not os.path.exists(VENV_PYTHON):
    # Windows uses Scripts/python.exe instead of bin/python
    VENV_PYTHON = os.path.join(ROOT_DIR, ".venv", "Scripts", "python.exe")

if not os.path.exists(VENV_PYTHON):
    logger.error(f"Virtual environment Python not found at {VENV_PYTHON}")
    logger.error("Please run 'scripts/setup_test_env.sh' first to create the virtual environment")
    sys.exit(1)
else:
    logger.info(f"Using Python interpreter: {VENV_PYTHON}")


def ensure_data_cache_exists():
    """
    Ensure the market data cache exists by running the data refresh script.
    Only generates essential data to save time.

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Ensuring market data cache exists...")
    refresh_script = os.path.join(ROOT_DIR, "scripts", "refresh_test_market_data.py")

    try:
        # Run with --essential-only flag to only generate the minimal required data
        result = subprocess.run(
            [VENV_PYTHON, refresh_script, "--essential-only"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            logger.error(f"Data cache preparation failed with code {result.returncode}")
            logger.warning("Tests will still run but may use synthetic data or fail")
            return False
        else:
            logger.info("Data cache preparation successful")
            return True

    except Exception as e:
        logger.error(f"Error during data cache preparation: {str(e)}")
        return False


def run_tests(test_path=None, verbose=False, capture_output=False):
    """
    Run the specified tests.

    Args:
        test_path: Path to test file or directory
        verbose: Whether to run tests in verbose mode
        capture_output: Whether to capture test output in logs

    Returns:
        bool: True if tests passed, False otherwise
    """
    logger.info(f"Running tests: {test_path if test_path else 'all'}")

    # Build pytest command
    cmd = [VENV_PYTHON, "-m", "pytest"]
    if verbose:
        cmd.append("-v")

    # Add path if specified
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")

    # Add flag to allow synthetic data as fallback
    cmd.append("--allow-synthetic-data")

    try:
        logger.info(f"Executing: {' '.join(cmd)}")

        if capture_output:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=False,
            )

            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

            success = result.returncode == 0
        else:
            # Run with direct output to terminal
            success = subprocess.call(cmd) == 0

        if success:
            logger.info("Tests passed successfully!")
        else:
            logger.error("Tests failed!")

        return success

    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests with data preparation")
    parser.add_argument("--path", "-p", help="Path to test file or directory", default=None)
    parser.add_argument("--verbose", "-v", help="Show verbose output", action="store_true")
    parser.add_argument("--skip-data-prep", "-s", help="Skip data preparation", action="store_true")
    parser.add_argument("--capture-output", "-c", help="Capture test output in logs", action="store_true")

    args = parser.parse_args()

    start_time = time.time()

    # Prepare data cache first if not skipped
    data_prep_ok = True
    if not args.skip_data_prep:
        data_prep_ok = ensure_data_cache_exists()

    # Run tests
    test_ok = run_tests(args.path, args.verbose, args.capture_output)

    elapsed_time = time.time() - start_time

    # Show summary
    logger.info(f"Testing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Data preparation: {'SKIPPED' if args.skip_data_prep else 'OK' if data_prep_ok else 'FAILED'}")
    logger.info(f"Test execution: {'PASSED' if test_ok else 'FAILED'}")

    # Exit with status code
    sys.exit(0 if test_ok else 1)


if __name__ == "__main__":
    main()
