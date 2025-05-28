"""
Test Fixtures Package

This package contains test fixtures for strategies, indicators, signals, and other
test data used throughout the Spark Stacker test suite.

Usage:
    from tests._fixtures import FixtureLoader, load_fixture_data

    loader = FixtureLoader()
    strategy_config = loader.get_single_strategy_config()

    # Or use convenience functions
    indicator_data = load_fixture_data('indicators.json', 'all_test_indicators')
"""

from .fixture_loader import FixtureLoader, load_fixture_data

__all__ = ['FixtureLoader', 'load_fixture_data']
