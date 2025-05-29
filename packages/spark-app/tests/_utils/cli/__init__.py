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

__version__ = "1.0.0"
__author__ = "Spark Stacker Team"

# Re-export main CLI function for backward compatibility
from .main import cli

__all__ = ["cli"]
