import os
import sys

import pytest

# Add the app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))


def test_imports():
    """Test that we can import the required modules."""
    from app.utils import config
    assert hasattr(config, 'AppConfig'), "AppConfig class not found in config module"

    from app.core import trading_engine
    assert hasattr(trading_engine, 'TradingEngine'), "TradingEngine class not found in trading_engine module"

if __name__ == "__main__":
    # Run the test directly
    test_imports()
    print("All imports successful!")
