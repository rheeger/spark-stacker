#!/usr/bin/env python3
"""
Project-wide linting script that checks both Python code and Markdown documentation.
Run from the project root with: python scripts/lint_project.py
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Directories to exclude from linting
EXCLUDE_DIRS = [
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "_htmlcov",
]


def run_command(command, description):
    """Run a shell command and log its output."""
    logger.info(f"Running {description}...")
    try:
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        )
        logger.info(f"{description} completed successfully")
        if result.stdout:
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed with exit code {e.returncode}")
        if e.stdout:
            logger.info(e.stdout)
        if e.stderr:
            logger.error(e.stderr)
        return False


def find_files(extension, exclude_dirs=None):
    """Find all files with the given extension in the project."""
    if exclude_dirs is None:
        exclude_dirs = EXCLUDE_DIRS

    files = []
    for root, dirs, filenames in os.walk(PROJECT_ROOT):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))

    return files


def lint_python_files():
    """Lint all Python files in the project."""
    python_files = find_files(".py")

    if not python_files:
        logger.info("No Python files found to lint")
        return True

    success = True

    # Run black
    black_command = ["black", "--check", "."]
    black_success = run_command(black_command, "Black code formatting check")

    # If black check failed, apply formatting
    if not black_success:
        logger.info("Applying black formatting...")
        black_format_command = ["black", "."]
        run_command(black_format_command, "Black code formatting")

    # Run pylint
    pylint_command = ["pylint"] + python_files
    pylint_success = run_command(pylint_command, "Pylint code linting")
    success = success and pylint_success

    # Run mypy
    mypy_command = ["mypy"] + python_files
    mypy_success = run_command(mypy_command, "MyPy type checking")
    success = success and mypy_success

    return success


def lint_markdown_files():
    """Lint all Markdown files in the project."""
    markdown_files = find_files(".md")

    if not markdown_files:
        logger.info("No Markdown files found to lint")
        return True

    # Check if markdownlint-cli is installed
    try:
        subprocess.run(
            ["markdownlint", "--version"], check=True, capture_output=True, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning(
            "markdownlint-cli not found. Please install it with: npm install -g markdownlint-cli"
        )
        logger.warning("Skipping Markdown linting")
        return True

    # Run markdownlint
    markdown_command = ["markdownlint"] + markdown_files + ["--fix"]
    markdown_success = run_command(markdown_command, "Markdown linting")

    return markdown_success


def main():
    """Run all linters and formatters."""
    logger.info("Starting project-wide linting...")

    python_success = lint_python_files()
    markdown_success = lint_markdown_files()

    if python_success and markdown_success:
        logger.info("✅ All linting checks passed!")
        return 0
    else:
        logger.error("❌ Linting failed. Please fix the issues and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
