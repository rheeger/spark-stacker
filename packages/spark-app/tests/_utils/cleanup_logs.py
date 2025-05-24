#!/usr/bin/env python3
"""
Cleanup script to remove old logs and unify logging paths.

This script removes legacy log directories to ensure a fresh start
with the new unified logging system.
"""

import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def cleanup_logs():
    """Remove all old log directories to create a clean slate."""
    # Get project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent  # Navigate to repo root

    # Define log paths to clean up
    log_paths = [
        project_root / "logs",  # Root logs directory
        project_root / "packages" / "spark-app" / "app" / "logs",  # Old app/logs directory
    ]

    # Create the new logs directory structure
    new_logs_dir = project_root / "packages" / "spark-app" / "logs"
    new_logs_dir.mkdir(exist_ok=True, parents=True)

    # Move app/logs README to logs directory if it exists
    app_logs_readme = project_root / "packages" / "spark-app" / "app" / "logs" / "README.md"
    if app_logs_readme.exists():
        try:
            new_readme_path = new_logs_dir / "README.md"
            if not new_readme_path.exists():
                logger.info(f"Moving README from {app_logs_readme} to {new_readme_path}")
                shutil.copy2(app_logs_readme, new_readme_path)
        except Exception as e:
            logger.error(f"Error copying README: {e}")

    # Clean up old log directories
    for log_path in log_paths:
        if log_path.exists():
            logger.info(f"Cleaning up log directory: {log_path}")
            try:
                # Remove all build-specific directories
                for item in log_path.iterdir():
                    if item.is_dir() and "_" in item.name:  # Match build directories with timestamps
                        logger.info(f"Removing old log directory: {item}")
                        shutil.rmtree(item)

                # If this is app/logs directory, create a redirecting README
                if "app/logs" in str(log_path):
                    with open(log_path / "README.md", "w") as f:
                        f.write("# Logs Moved\n\n")
                        f.write("Logs have been moved to `packages/spark-app/logs` directory.\n")
                        f.write("Please look there for all application logs.\n")

                logger.info(f"Successfully cleaned up: {log_path}")
            except Exception as e:
                logger.error(f"Error cleaning up {log_path}: {e}")

    logger.info(f"Log cleanup complete. New logs will be created in: {new_logs_dir}")
    logger.info("Symlinks to these logs will be created in the root /logs directory for compatibility")

if __name__ == "__main__":
    logger.info("Starting log cleanup process...")
    cleanup_logs()
    logger.info("Log cleanup completed successfully.")
