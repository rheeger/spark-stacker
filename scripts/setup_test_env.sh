#!/bin/bash
# Setup testing environment with required packages

set -e # Exit immediately if a command exits with non-zero status

echo "Setting up test environment..."

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "No virtual environment detected. Creating one..."

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
        source .venv/bin/activate
    else
        echo "Python 3 not found. Please install Python 3 and try again."
        exit 1
    fi
else
    echo "Using virtual environment: $VIRTUAL_ENV"
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install pytest pytest-cov pytest-mock numpy pandas matplotlib

# Verify installation
echo "Verifying installation..."
python -c "import pytest, pandas, numpy, matplotlib; print('All required packages installed successfully!')"

echo "Test environment setup complete!"
