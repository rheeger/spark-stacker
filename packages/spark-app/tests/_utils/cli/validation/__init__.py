"""
Validation and Error Handling Modules

This package contains validation and error handling modules:
- config_validator: Configuration validation and repair utilities
- strategy_validator: Strategy-specific validation logic
- data_validator: Data quality validation and consistency checks
"""

from .config_validator import (ConfigIssue, ConfigValidationResult,
                               ConfigValidator)
from .data_validator import DataIssue, DataValidationResult, DataValidator
from .strategy_validator import StrategyValidator, ValidationResult

__all__ = [
    # Validator classes
    "ConfigValidator",
    "StrategyValidator",
    "DataValidator",

    # Result classes
    "ConfigValidationResult",
    "ValidationResult",  # Strategy validation result
    "DataValidationResult",

    # Issue classes
    "ConfigIssue",
    "DataIssue",
]
