import os
import sys

import pytest

# Set environment variables to match conftest.py
os.environ['PYTEST_RUNNING'] = 'True'

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print(f"Project root: {project_root}")

# Add the project root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to sys.path")

# Also add the app directory to the Python path to enable direct imports
app_path = os.path.join(project_root, "app")
if app_path not in sys.path:
    sys.path.insert(0, app_path)
    print(f"Added {app_path} to sys.path")

# For troubleshooting, print PYTHONPATH and sys.path
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"sys.path[0:3]: {sys.path[0:3]}")

def test_imports():
    """Test that we can import the required modules."""
    try:
        # Print environment and directory info
        print(f"Current directory: {os.getcwd()}")

        # Check if app/utils/config.py exists
        config_path = os.path.join(app_path, "utils", "config.py")
        print(f"config.py exists: {os.path.exists(config_path)}")
        if os.path.exists(config_path):
            print(f"First few lines of config.py:")
            with open(config_path, 'r') as f:
                for i, line in enumerate(f):
                    if i < 5:  # Print first 5 lines
                        print(f"  {line.strip()}")

        # Try imports
        from app.utils import config
        print("Successfully imported app.utils.config")
        print(f"Config module location: {config.__file__}")
        assert hasattr(config, 'AppConfig'), "AppConfig class not found in config module"

        from app.core import trading_engine
        print("Successfully imported app.core.trading_engine")
        assert hasattr(trading_engine, 'TradingEngine'), "TradingEngine class not found"

    except ImportError as e:
        print(f"Import error: {e}")
        # Try alternative paths
        try:
            # Try importing directly
            sys.path.insert(0, os.path.dirname(project_root))
            print(f"Trying alternative path: {os.path.dirname(project_root)}")

            import app.utils.config as config
            print("Successfully imported app.utils.config using absolute path")
            assert hasattr(config, 'AppConfig'), "AppConfig class not found in config module"

            import app.core.trading_engine as trading_engine
            print("Successfully imported app.core.trading_engine using absolute path")
            assert hasattr(trading_engine, 'TradingEngine'), "TradingEngine class not found"
        except Exception as e2:
            print(f"Second import attempt failed: {e2}")
            pytest.fail(f"Import failed: {e} and then {e2}")

if __name__ == "__main__":
    # Run the test directly
    test_imports()
    print("All imports successful!")
