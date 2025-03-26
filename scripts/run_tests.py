#!/usr/bin/env python3
"""
Test runner script for Spark Stacker.

This script runs all the tests and generates a coverage report.
"""

import os
import sys
import subprocess
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestRunner")

# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_tests(test_path=None, coverage=True, verbose=False, failfast=False):
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

    args = parser.parse_args()

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
    )
