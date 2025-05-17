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
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SystemTest")

# Get the root directory of the project
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Add the project root to Python path so tests can import from app
sys.path.insert(0, ROOT_DIR)
# Also add the app directory for direct imports
sys.path.insert(0, os.path.join(ROOT_DIR, "app"))

# Market data cache directory
MARKET_DATA_CACHE_DIR = os.path.join(ROOT_DIR, "tests", "test_data", "market_data")

# Check if we're running in CI
IN_CI = os.environ.get("CI", "").lower() == "true"

def find_venv_python():
    """Find the Python interpreter in the virtual environment."""
    if IN_CI:
        # In CI, just use the system Python
        logger.info("Running in CI environment, using system Python")
        return sys.executable

    venv_dir = os.path.join(ROOT_DIR, ".venv")
    if not os.path.exists(venv_dir):
        logger.error(f"Virtual environment not found at {venv_dir}")
        if IN_CI:
            # In CI, this is not an error
            return sys.executable
        else:
            return None

    # Try common locations
    possible_paths = [
        os.path.join(venv_dir, "bin", "python"),  # Unix
        os.path.join(venv_dir, "bin", "python3"),  # Unix alternative
        os.path.join(venv_dir, "Scripts", "python.exe"),  # Windows
    ]

    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found Python interpreter at: {path}")
            return path

    logger.error("Could not find Python interpreter in virtual environment")
    if IN_CI:
        # In CI, fall back to system Python
        return sys.executable
    else:
        return None


# Find the virtual environment Python
VENV_PYTHON = find_venv_python()
if not VENV_PYTHON:
    logger.error(
        "Please run 'scripts/setup_test_env.sh' first to create the virtual environment"
    )
    if not IN_CI:
        # Only exit if not in CI
        sys.exit(1)
    else:
        # In CI, use system Python
        VENV_PYTHON = sys.executable
        logger.info(f"Using system Python in CI: {VENV_PYTHON}")


def ensure_test_env():
    """Ensure the test environment is properly set up."""
    # In CI, we assume the environment is already set up
    if IN_CI:
        return True

    # Check if pytest is installed
    try:
        subprocess.run(
            [VENV_PYTHON, "-c", "import pytest"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError:
        logger.error("pytest not found in virtual environment")
        logger.error("Running setup_test_env.sh to install dependencies...")
        setup_script = os.path.join(SCRIPT_DIR, "setup_test_env.sh")
        try:
            subprocess.run([setup_script], check=True)
        except subprocess.CalledProcessError:
            logger.error("Failed to set up test environment")
            return False
    return True


def ensure_data_cache_exists():
    """
    Ensure the market data cache exists by running the data refresh script.
    Only generates essential data to save time.

    Returns:
        bool: True if successful, False otherwise
    """
    # In CI, we'll use synthetic data
    if IN_CI:
        logger.info("Running in CI, using synthetic data")
        return True

    logger.info("Ensuring market data cache exists...")
    refresh_script = os.path.join(SCRIPT_DIR, "refresh_test_market_data.py")

    if not os.path.exists(refresh_script):
        logger.warning(f"Data refresh script not found at {refresh_script}")
        logger.warning("Tests will use synthetic data")
        return True

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
            logger.warning("Tests will use synthetic data")
            return True
        else:
            logger.info("Data cache preparation successful")
            return True

    except Exception as e:
        logger.error(f"Error during data cache preparation: {str(e)}")
        logger.warning("Tests will use synthetic data")
        return True


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
    if not ensure_test_env():
        return False

    logger.info(f"Running tests: {test_path if test_path else 'all'}")

    # Build pytest command
    cmd = [VENV_PYTHON, "-m", "pytest"]
    if verbose:
        cmd.append("-v")

    # Add coverage reporting
    cmd.extend(["--cov=app", "--cov-report=term-missing"])

    # Add path if specified
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append(os.path.join(ROOT_DIR, "tests"))

    # Add flag to allow synthetic data as fallback
    cmd.append("--allow-synthetic-data")

    # Set PYTHONPATH to include app directory
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT_DIR}:{os.path.join(ROOT_DIR, 'app')}:{env.get('PYTHONPATH', '')}"

    try:
        logger.info(f"Executing: {' '.join(cmd)}")
        logger.info(f"Working directory: {ROOT_DIR}")
        logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")

        # Always run from the project root
        os.chdir(ROOT_DIR)

        if capture_output:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=False,
                env=env
            )

            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

            success = result.returncode == 0
        else:
            # Run with direct output to terminal
            success = subprocess.call(cmd, env=env) == 0

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
    parser.add_argument(
        "--path", "-p", help="Path to test file or directory", default=None
    )
    parser.add_argument(
        "--verbose", "-v", help="Show verbose output", action="store_true"
    )
    parser.add_argument(
        "--skip-data-prep", "-s", help="Skip data preparation", action="store_true"
    )
    parser.add_argument(
        "--capture-output",
        "-c",
        help="Capture test output in logs",
        action="store_true",
    )

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
    logger.info(
        f"Data preparation: {'SKIPPED' if args.skip_data_prep else 'OK' if data_prep_ok else 'FAILED'}"
    )
    logger.info(f"Test execution: {'PASSED' if test_ok else 'FAILED'}")

    # Exit with status code
    sys.exit(0 if test_ok else 1)


if __name__ == "__main__":
    main()
