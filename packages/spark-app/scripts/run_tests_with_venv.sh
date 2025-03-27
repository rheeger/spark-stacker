#!/bin/bash
# Run tests with virtual environment
# This script ensures the virtual environment is active and then runs the tests

set -e # Exit immediately if a command exits with non-zero status

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the package root directory
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PACKAGE_DIR"

# Check if virtual environment exists, create if it doesn't
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found, creating it..."
    "$SCRIPT_DIR/setup_test_env.sh"
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    echo "Error: Virtual environment activation script not found."
    exit 1
fi

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Check if numpy is available
if ! python -c "import numpy" &>/dev/null; then
    echo "Error: numpy package not available in the virtual environment."
    echo "Please run ./scripts/setup_test_env.sh to install required packages."
    exit 1
fi

# Run tests from the package directory
python "$SCRIPT_DIR/test_system.py" "$@"
