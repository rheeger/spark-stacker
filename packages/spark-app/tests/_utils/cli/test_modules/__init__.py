"""
Module Tests for CLI Architecture

This package contains comprehensive tests for the modular CLI architecture:
- Unit tests for individual modules
- Integration tests for module interactions
- Performance tests for the new architecture
- Backward compatibility tests
- Error handling and edge case tests

Test Structure:
- test_core_modules/      - Tests for core modules (config_manager, data_manager, etc.)
- test_command_modules/   - Tests for command handler modules
- test_manager_modules/   - Tests for specialized manager classes
- test_reporting_modules/ - Tests for reporting modules
- test_validation_modules/- Tests for validation modules
- test_utils_modules/     - Tests for utility modules
- test_integration/       - Integration tests for module interactions
- test_performance/       - Performance tests for new architecture
- test_compatibility/     - Backward compatibility tests
"""

import os
import sys
from pathlib import Path

import pytest

# Add the CLI modules to the path for testing
cli_dir = Path(__file__).parent.parent
sys.path.insert(0, str(cli_dir))

# Common test utilities and fixtures
__all__ = [
    'pytest',
    'cli_dir'
]
