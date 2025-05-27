"""
Integration tests for the CLI functionality.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("command", [
    ["--help"],
    ["list-indicators", "--help"],
    ["backtest", "--help"],
    ["demo", "--help"],
])
def test_cli_help_commands(command):
    """Test that CLI help commands run successfully."""
    # Get root spark-app directory
    app_dir = Path(__file__).parent.parent.parent.parent.resolve()

    # Path to the CLI utility in tests/_utils/cli.py
    cli_path = app_dir / "tests" / "_utils" / "cli.py"

    # Construct the full command using the actual CLI location
    full_command = [sys.executable, str(cli_path)] + command

    # Run the command with a timeout to ensure it completes quickly
    result = subprocess.run(
        full_command,
        cwd=app_dir,
        capture_output=True,
        text=True,
        timeout=10  # 10-second timeout as specified in the audit
    )

    # Check the command succeeded
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "Error" not in result.stdout


@pytest.mark.parametrize("indicator", ["MACD"])
def test_cli_list_indicators_output(indicator):
    """Test that the list-indicators command shows expected indicators."""
    # Get root spark-app directory
    app_dir = Path(__file__).parent.parent.parent.parent.resolve()

    # Path to the CLI utility in tests/_utils/cli.py
    cli_path = app_dir / "tests" / "_utils" / "cli.py"

    # Construct the command using the actual CLI location
    full_command = [sys.executable, str(cli_path), "list-indicators"]

    # Run the command
    result = subprocess.run(
        full_command,
        cwd=app_dir,
        capture_output=True,
        text=True,
        timeout=10
    )

    # Check for success and expected indicator
    assert result.returncode == 0
    assert indicator in result.stdout
