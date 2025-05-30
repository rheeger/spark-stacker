#!/usr/bin/env python3
"""
Main CLI Entry Point - Migrated from monolithic cli.py

This file serves as the main entry point for the Spark-App CLI, currently
delegating to the original cli.py while the modular architecture is being developed.

This maintains full backward compatibility during the transition phase.
"""

import os
import subprocess
import sys

# Add the app directory to the path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up levels: cli -> _utils -> tests -> spark-app (where app directory is)
spark_app_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, spark_app_dir)


def main():
    """
    Main entry point that delegates to the original CLI during migration.

    This ensures full backward compatibility while the modular architecture
    is being developed in subsequent tasks.
    """
    # Path to the original CLI
    original_cli_path = os.path.join(os.path.dirname(current_dir), "cli.py")

    # Get the python executable from current environment (venv if active)
    python_executable = sys.executable

    # Prepare command arguments
    cmd = [python_executable, original_cli_path] + sys.argv[1:]

    try:
        # Execute the original CLI with the same arguments
        result = subprocess.run(cmd,
                              cwd=os.path.dirname(original_cli_path),
                              env=os.environ.copy())

        # Exit with the same code as the original CLI
        sys.exit(result.returncode)

    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error executing CLI: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
