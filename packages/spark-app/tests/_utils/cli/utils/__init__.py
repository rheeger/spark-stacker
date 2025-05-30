"""
Utility Functions and Helpers

This package contains utility functions and helper modules:
- cli_helpers: CLI utility functions and common operations
- output_formatters: Console output formatting and styling
- progress_trackers: Progress tracking and user feedback utilities
- error_handlers: Centralized error handling patterns and utilities
"""

# Import error handling utilities
from .error_handlers import (CLIError, ConfigurationError, DataFetchingError,
                             ErrorCategory, IndicatorError, StrategyError,
                             ValidationError, config_error_handler,
                             data_error_handler, data_fetch_retry,
                             graceful_degradation, handle_cli_errors,
                             is_network_error, is_retriable_error,
                             network_retry, retry_with_backoff,
                             strategy_error_handler, validate_required_params)

# TODO: Import utility functions when modules are implemented
# from .cli_helpers import *
# from .output_formatters import *
# from .progress_trackers import *

__all__ = [
    # Error handling classes
    "CLIError",
    "ConfigurationError",
    "DataFetchingError",
    "StrategyError",
    "IndicatorError",
    "ValidationError",
    "ErrorCategory",

    # Error handling decorators and functions
    "handle_cli_errors",
    "retry_with_backoff",
    "graceful_degradation",
    "validate_required_params",
    "data_fetch_retry",
    "network_retry",
    "config_error_handler",
    "strategy_error_handler",
    "data_error_handler",
    "is_network_error",
    "is_retriable_error",

    # Functions will be added as modules are implemented
]
