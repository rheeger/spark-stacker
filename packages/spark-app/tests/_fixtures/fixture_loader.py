"""
Fixture Loader Utility

This module provides utilities for loading test fixtures for strategies, indicators,
signals, and other test data used throughout the test suite.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class FixtureLoader:
    """Utility class for loading test fixtures."""

    def __init__(self, fixtures_dir: Optional[str] = None):
        """
        Initialize the fixture loader.

        Args:
            fixtures_dir: Path to fixtures directory. If None, uses default location.
        """
        if fixtures_dir is None:
            self.fixtures_dir = Path(__file__).parent
        else:
            self.fixtures_dir = Path(fixtures_dir)

    def load_json_fixture(self, filename: str) -> Dict[str, Any]:
        """
        Load a JSON fixture file.

        Args:
            filename: Name of the fixture file (with or without .json extension)

        Returns:
            Dictionary containing fixture data

        Raises:
            FileNotFoundError: If fixture file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        if not filename.endswith('.json'):
            filename += '.json'

        filepath = self.fixtures_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Fixture file not found: {filepath}")

        with open(filepath, 'r') as f:
            return json.load(f)

    def load_strategies_fixture(self, key: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load strategy configuration fixtures.

        Args:
            key: Specific strategy key to load. If None, returns all strategies.

        Returns:
            Strategy configuration(s)
        """
        data = self.load_json_fixture('strategies.json')
        if key is None:
            return data

        if key not in data:
            raise KeyError(f"Strategy fixture key '{key}' not found")

        return data[key]

    def get_single_strategy_config(self) -> Dict[str, Any]:
        """Get a minimal single strategy configuration for testing."""
        return self.load_strategies_fixture('single_strategy_single_indicator')


# Convenience functions
def load_fixture_data(filename: str, key: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Convenience function to load fixture data.

    Args:
        filename: Name of the fixture file
        key: Optional key to extract specific data

    Returns:
        Fixture data
    """
    loader = FixtureLoader()
    data = loader.load_json_fixture(filename)

    if key is None:
        return data

    if key not in data:
        raise KeyError(f"Key '{key}' not found in fixture '{filename}'")

    return data[key]
