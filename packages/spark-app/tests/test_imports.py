import pytest


def test_imports():
    """Test that we can import the required modules."""
    try:
        from app.core.trading_engine import TradingEngine
        from app.utils.config import AppConfig
        assert True, "Imports successful"
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
