#!/usr/bin/env python3
"""
Spark-App CLI - Entry point that forwards to tests/utils/cli
This maintains backward compatibility with existing usage
"""
import logging
import os
import sys
from typing import Any

# Import the CLI from tests/utils/cli.py
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
tests_utils_dir = os.path.join(app_dir, "tests", "utils")
sys.path.insert(0, app_dir)

try:
    # Import the CLI from tests/utils/cli.py
    from tests.utils.cli import cli
except ImportError:
    # If that fails, try a direct import (might be needed in some environments)
    sys.path.insert(0, tests_utils_dir)
    from cli import cli

if __name__ == "__main__":
    cli()
