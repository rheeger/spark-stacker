#!/bin/bash
# Setup testing environment with required packages

set -e # Exit immediately if a command exits with non-zero status

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the package root directory
PACKAGE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PACKAGE_DIR"

echo "Setting up test environment..."

# Check if python3 is available
if command -v python3 &>/dev/null; then
    echo "Using $(python3 --version)"

    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi

    # Activate virtual environment
    echo "Activating virtual environment..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
    else
        echo "Error: Virtual environment activation script not found."
        exit 1
    fi
else
    echo "Python 3 not found. Please install Python 3 and try again."
    exit 1
fi

echo "Using virtual environment: $VIRTUAL_ENV"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install all requirements
echo "Installing all requirements..."
python -m pip install -r requirements.txt

# Install additional test dependencies
echo "Installing test dependencies..."
python -m pip install pytest pytest-cov pytest-mock coverage

# Verify installation
echo "Verifying installation..."
python -c "
import sys
import pytest
import pandas
import numpy
import matplotlib
import coinbase
print('Python version:', sys.version)
print('All required packages installed successfully!')
"

echo "Test environment setup complete!"
