#!/usr/bin/env python3
"""
Main CLI Entry Point - Modular Architecture

This file serves as the main entry point for the Spark-App CLI, coordinating
command handlers from the modular architecture.

This replaces the monolithic cli.py with a clean, organized structure.
"""

import logging
import os
import sys

import click
# Set matplotlib backend to prevent GUI hanging issues
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

# Add the app directory to the path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up levels: cli -> _utils -> tests -> spark-app (where app directory is)
spark_app_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, spark_app_dir)

# Set up logging
from app.utils.logging_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Global CLI options
@click.group()
@click.option('--config', '-c',
              help='Path to configuration file (default: ../shared/config.json)')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
@click.option('--debug', '-d', is_flag=True,
              help='Enable debug logging')
@click.pass_context
def cli(ctx, config, verbose, debug):
    """
    Spark-App CLI - Unified command line interface for backtest operations

    This modular CLI provides comprehensive backtesting and strategy analysis capabilities.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Store global options in context
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose
    ctx.obj['debug'] = debug

    # Set up logging level based on flags
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Verbose logging enabled")

    logger.debug(f"CLI initialized with config: {config}")


# TODO: Import and register command handlers from modules
# This will be done in subsequent tasks (1.4.2+)
# from .commands.strategy_commands import register_strategy_commands
# from .commands.indicator_commands import register_indicator_commands
# from .commands.comparison_commands import register_comparison_commands
# from .commands.list_commands import register_list_commands
# from .commands.utility_commands import register_utility_commands

# Temporary placeholder commands to verify structure
@cli.command()
def test_structure():
    """Test command to verify modular CLI structure is working."""
    click.echo("✅ Modular CLI structure created successfully!")
    click.echo("\n📁 Architecture Overview:")
    click.echo("   • commands/     - Command handler modules")
    click.echo("   • core/         - Core business logic modules")
    click.echo("   • managers/     - Specialized manager classes")
    click.echo("   • reporting/    - Report generation modules")
    click.echo("   • validation/   - Validation and error handling")
    click.echo("   • utils/        - Utility functions and helpers")

    # List the created files
    import os
    cli_dir = os.path.dirname(__file__)
    click.echo(f"\n📂 CLI Directory: {cli_dir}")

    for root, dirs, files in os.walk(cli_dir):
        level = root.replace(cli_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        click.echo(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.py'):
                click.echo(f"{subindent}{file}")


if __name__ == '__main__':
    cli()
