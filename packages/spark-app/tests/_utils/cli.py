#!/usr/bin/env python3
"""
Spark-App CLI - Backward Compatibility Shim

⚠️  DEPRECATION NOTICE:
This file (cli.py) is deprecated and maintained only for backward compatibility.
The CLI has been migrated to a modular architecture located at:
    packages/spark-app/tests/_utils/cli/

Migration Path:
    OLD: python cli.py <command>
    NEW: python -m cli.main <command>
    OR:  python cli/main.py <command>

This shim will be removed in a future version. Please update your scripts
and workflows to use the new location.

For more information about the new modular architecture, see:
    packages/spark-app/tests/_utils/cli/README.md
"""

import os
import sys
import warnings

# Add the app directory to the path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
spark_app_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, spark_app_dir)

# Issue deprecation warning
warnings.warn(
    "\n" + "="*80 + "\n"
    "⚠️  DEPRECATION WARNING: cli.py has moved to a modular architecture\n"
    "\n"
    "The CLI functionality has been migrated to: tests/_utils/cli/\n"
    "\n"
    "Please update your usage:\n"
    "  OLD: python cli.py <command>\n"
    "  NEW: python -m cli.main <command>\n"
    "   OR: python cli/main.py <command>\n"
    "\n"
    "This backward compatibility shim will be removed in a future version.\n"
    "Update your scripts and workflows to use the new location.\n"
    + "="*80,
    DeprecationWarning,
    stacklevel=2
)

try:
    # Add the CLI directory to the Python path
    cli_dir = os.path.join(current_dir, "cli")
    sys.path.insert(0, cli_dir)

    # Import the main CLI function from the new modular location
    # Re-export all the utility functions for backward compatibility
    from main import (cleanup_resources, cli, display_strategy_info,
                      get_default_output_dir, get_strategy_config,
                      list_strategies, load_config, validate_config,
                      validate_strategy_config)

except ImportError as e:
    print(f"❌ Error importing from new CLI location: {e}")
    print("\n🔧 The CLI migration may be incomplete. Please check:")
    print("   1. The cli/ directory exists in tests/_utils/")
    print("   2. The cli/main.py file contains the CLI functionality")
    print("   3. All required dependencies are installed")
    print("\n📚 For troubleshooting, see: packages/shared/docs/checklists/phase3.5.3-backtesting-improvements.md")
    sys.exit(1)

# Maintain the original entry point behavior
if __name__ == "__main__":
    try:
        # Print deprecation notice to stderr so it doesn't interfere with CLI output
        print("\n⚠️  Using deprecated cli.py - please migrate to: python cli/main.py", file=sys.stderr)
        print("   See deprecation notice at the top of this file for details\n", file=sys.stderr)

        # Execute the CLI with the same behavior as before
        cli()

    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user", file=sys.stderr)
        cleanup_resources()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error in CLI: {str(e)}", file=sys.stderr)
        cleanup_resources()
        sys.exit(1)
    finally:
        # Final cleanup and graceful exit
        cleanup_resources()
        # Force exit to prevent hanging - this is the key fix
        os._exit(0)

# Export the CLI function for programmatic usage
__all__ = [
    'cli',
    'load_config',
    'validate_config',
    'list_strategies',
    'get_strategy_config',
    'validate_strategy_config',
    'display_strategy_info',
    'cleanup_resources',
    'get_default_output_dir'
]
