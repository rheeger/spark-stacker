#!/usr/bin/env python3
"""
CLI entry point for the Spark App.

This module provides a command-line interface for backtesting,
indicator management, and other app functionality.
"""

import os
import sys
from pathlib import Path

# Add the project root to path to enable imports
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
tests_utils_dir = project_root / "tests" / "_utils"

# Add paths for imports
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(tests_utils_dir))

# Import and run the CLI from tests utils
try:
    from tests._utils.cli import cli

    # Always run CLI when this module is executed
    cli()
except ImportError as e:
    print(f"Failed to import CLI: {e}")
    sys.exit(1)
