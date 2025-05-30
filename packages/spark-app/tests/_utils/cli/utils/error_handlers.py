"""
Error Handling Utilities

This module provides centralized error handling patterns and utilities for the CLI:
- Specific error handling for strategy configuration issues
- Helpful error messages with fix suggestions
- Graceful degradation for partial configuration issues
- Error logging with sufficient context for debugging
- Retry mechanisms for data fetching failures
"""

import logging
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

import click

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors for specialized handling."""
    CONFIGURATION = "configuration"
    DATA_FETCHING = "data_fetching"
    VALIDATION = "validation"
    STRATEGY = "strategy"
    INDICATOR = "indicator"
    POSITION_SIZING = "position_sizing"
    EXCHANGE_CONNECTIVITY = "exchange_connectivity"
    RESOURCE = "resource"
    FILE_IO = "file_io"
    NETWORK = "network"


class CLIError(Exception):
    """Base exception for CLI-specific errors."""

    def __init__(self, message: str, category: ErrorCategory,
                 fix_suggestions: Optional[List[str]] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.category = category
        self.fix_suggestions = fix_suggestions or []
        self.context = context or {}


class ConfigurationError(CLIError):
    """Errors related to configuration issues."""

    def __init__(self, message: str, fix_suggestions: Optional[List[str]] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.CONFIGURATION, fix_suggestions, context)


class DataFetchingError(CLIError):
    """Errors related to data fetching operations."""

    def __init__(self, message: str, fix_suggestions: Optional[List[str]] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.DATA_FETCHING, fix_suggestions, context)


class StrategyError(CLIError):
    """Errors related to strategy operations."""

    def __init__(self, message: str, fix_suggestions: Optional[List[str]] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.STRATEGY, fix_suggestions, context)


class IndicatorError(CLIError):
    """Errors related to indicator operations."""

    def __init__(self, message: str, fix_suggestions: Optional[List[str]] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.INDICATOR, fix_suggestions, context)


class ValidationError(CLIError):
    """Errors related to validation operations."""

    def __init__(self, message: str, fix_suggestions: Optional[List[str]] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.VALIDATION, fix_suggestions, context)


def handle_cli_errors(error_category: Optional[ErrorCategory] = None):
    """
    Decorator for handling CLI errors with appropriate error messages and suggestions.

    Args:
        error_category: Optional category to filter error handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CLIError as e:
                # Handle CLI-specific errors with context
                logger.error(f"CLI Error in {func.__name__}: {e}", extra={
                    'category': e.category.value,
                    'context': e.context,
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                })

                # Build detailed error message
                error_msg = f"❌ {e}"

                if e.fix_suggestions:
                    error_msg += "\n\n💡 Suggestions to fix this issue:"
                    for i, suggestion in enumerate(e.fix_suggestions, 1):
                        error_msg += f"\n   {i}. {suggestion}"

                if e.context:
                    error_msg += f"\n\n🔍 Context: {_format_context(e.context)}"

                raise click.ClickException(error_msg)

            except FileNotFoundError as e:
                logger.error(f"File not found in {func.__name__}: {e}")
                suggestions = [
                    "Check if the file path is correct",
                    "Ensure the file exists and is readable",
                    "Run with --help to see expected file locations"
                ]
                error_msg = f"❌ File not found: {e}\n\n💡 Suggestions:\n"
                error_msg += "\n".join(f"   {i}. {s}" for i, s in enumerate(suggestions, 1))
                raise click.ClickException(error_msg)

            except PermissionError as e:
                logger.error(f"Permission error in {func.__name__}: {e}")
                suggestions = [
                    "Check file/directory permissions",
                    "Run with appropriate user privileges",
                    "Ensure output directory is writable"
                ]
                error_msg = f"❌ Permission denied: {e}\n\n💡 Suggestions:\n"
                error_msg += "\n".join(f"   {i}. {s}" for i, s in enumerate(suggestions, 1))
                raise click.ClickException(error_msg)

            except KeyError as e:
                logger.error(f"Missing required key in {func.__name__}: {e}")
                if error_category == ErrorCategory.CONFIGURATION:
                    suggestions = [
                        "Check your config.json file format",
                        "Ensure all required configuration keys are present",
                        "Run 'validate-config' command to check configuration"
                    ]
                else:
                    suggestions = [
                        "Check if all required parameters are provided",
                        "Verify data structure contains expected keys"
                    ]
                error_msg = f"❌ Missing required configuration: {e}\n\n💡 Suggestions:\n"
                error_msg += "\n".join(f"   {i}. {s}" for i, s in enumerate(suggestions, 1))
                raise click.ClickException(error_msg)

            except ValueError as e:
                logger.error(f"Invalid value in {func.__name__}: {e}")
                suggestions = [
                    "Check parameter values and formats",
                    "Ensure numeric values are within valid ranges",
                    "Verify string parameters match expected options"
                ]
                error_msg = f"❌ Invalid value: {e}\n\n💡 Suggestions:\n"
                error_msg += "\n".join(f"   {i}. {s}" for i, s in enumerate(suggestions, 1))
                raise click.ClickException(error_msg)

            except Exception as e:
                # Log unexpected errors with full context
                logger.error(f"Unexpected error in {func.__name__}: {e}", extra={
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'error_type': type(e).__name__
                }, exc_info=True)

                suggestions = [
                    "Check the logs for more detailed error information",
                    "Verify your configuration and input parameters",
                    "Try running with simpler parameters to isolate the issue",
                    "Report this issue if it persists"
                ]
                error_msg = f"❌ Unexpected error: {e}\n\n💡 Suggestions:\n"
                error_msg += "\n".join(f"   {i}. {s}" for i, s in enumerate(suggestions, 1))
                raise click.ClickException(error_msg)

        return wrapper
    return decorator


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    should_retry: Optional[Callable[[Exception], bool]] = None
):
    """
    Decorator for retrying operations with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        exceptions: Tuple of exception types to retry on
        should_retry: Optional function to determine if exception should trigger retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # Check if we should retry this exception
                    if should_retry and not should_retry(e):
                        logger.warning(f"Exception not retryable: {e}")
                        break

                    if attempt < max_retries:
                        delay = (backoff_factor ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")

            # If we get here, all retries failed
            raise last_exception

        return wrapper
    return decorator


def graceful_degradation(
    fallback_value: Any = None,
    log_level: int = logging.WARNING,
    fallback_func: Optional[Callable] = None
):
    """
    Decorator for graceful degradation when operations fail.

    Args:
        fallback_value: Value to return if operation fails
        log_level: Log level for recording the degradation
        fallback_func: Optional function to call for fallback behavior
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(log_level, f"Graceful degradation in {func.__name__}: {e}")

                if fallback_func:
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback function also failed: {fallback_error}")
                        return fallback_value

                return fallback_value

        return wrapper
    return decorator


def validate_required_params(**required_params):
    """
    Decorator to validate required parameters for CLI commands.

    Args:
        **required_params: Dict of parameter names and their validators
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            missing_params = []
            invalid_params = []

            for param_name, validator in required_params.items():
                if param_name not in kwargs:
                    missing_params.append(param_name)
                elif validator and not validator(kwargs[param_name]):
                    invalid_params.append(param_name)

            if missing_params:
                raise ConfigurationError(
                    f"Missing required parameters: {', '.join(missing_params)}",
                    fix_suggestions=[
                        f"Provide the missing parameter(s): {', '.join(missing_params)}",
                        "Check command help with --help for parameter details"
                    ]
                )

            if invalid_params:
                raise ValidationError(
                    f"Invalid values for parameters: {', '.join(invalid_params)}",
                    fix_suggestions=[
                        f"Check the format/values for: {', '.join(invalid_params)}",
                        "Refer to documentation for valid parameter ranges"
                    ]
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def _format_context(context: Dict[str, Any]) -> str:
    """Format context dictionary for display in error messages."""
    if not context:
        return "No additional context available"

    formatted_items = []
    for key, value in context.items():
        # Truncate long values
        if isinstance(value, str) and len(value) > 100:
            value = value[:97] + "..."
        formatted_items.append(f"{key}: {value}")

    return ", ".join(formatted_items)


def is_network_error(exception: Exception) -> bool:
    """Check if an exception is network-related."""
    network_error_types = [
        "ConnectionError",
        "TimeoutError",
        "HTTPError",
        "URLError",
        "RequestException"
    ]
    return any(error_type in str(type(exception)) for error_type in network_error_types)


def is_retriable_error(exception: Exception) -> bool:
    """Check if an exception should trigger a retry."""
    # Network errors are typically retriable
    if is_network_error(exception):
        return True

    # Some specific error messages indicate temporary issues
    retriable_messages = [
        "rate limit",
        "temporarily unavailable",
        "service unavailable",
        "timeout",
        "connection reset"
    ]
    error_message = str(exception).lower()
    return any(msg in error_message for msg in retriable_messages)


# Pre-built decorators for common scenarios
data_fetch_retry = retry_with_backoff(
    max_retries=3,
    backoff_factor=2.0,
    should_retry=is_retriable_error
)

network_retry = retry_with_backoff(
    max_retries=5,
    backoff_factor=1.5,
    should_retry=is_network_error
)

config_error_handler = handle_cli_errors(ErrorCategory.CONFIGURATION)
strategy_error_handler = handle_cli_errors(ErrorCategory.STRATEGY)
data_error_handler = handle_cli_errors(ErrorCategory.DATA_FETCHING)
