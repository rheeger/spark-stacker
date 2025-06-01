"""
Spark-App CLI - Modular Command Line Interface

This package provides a modular, extensible CLI for backtesting and strategy analysis.

Architecture:
- commands/: Command handler modules for different CLI commands
- core/: Core business logic modules (config, data, orchestration)
- managers/: Specialized manager classes for different backtest types
- reporting/: Report generation modules for different output types
- validation/: Validation and error handling modules
- utils/: Utility functions and helpers

Usage:
    from cli.main import cli
    cli()
"""

import warnings

__version__ = "1.0.0"
__author__ = "Spark Stacker Team"

# Issue deprecation warning for backward compatibility imports
warnings.warn(
    "Importing from 'cli' package is deprecated. "
    "Please use 'from cli.main import cli' or run 'python cli/main.py' directly.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export main CLI functions for backward compatibility
from .main import (cleanup_resources, cli, display_strategy_info,
                   get_default_output_dir, get_strategy_config,
                   list_strategies, load_config, validate_config,
                   validate_strategy_config)

__all__ = [
    "cli",
    "load_config",
    "validate_config",
    "list_strategies",
    "get_strategy_config",
    "validate_strategy_config",
    "display_strategy_info",
    "cleanup_resources",
    "get_default_output_dir"
]
