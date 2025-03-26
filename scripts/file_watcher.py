#!/usr/bin/env python3
"""
File watcher script for Spark Stacker.

This script watches for changes in the project files and automatically runs the
appropriate tests when files are modified.
"""

import os
import sys
import time
import subprocess
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FileWatcher")

# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class SparkStackerTestHandler(FileSystemEventHandler):
    """File system event handler for running tests on file changes."""

    def __init__(self, test_command="pytest"):
        self.test_command = test_command
        self.last_modified_time = time.time()
        self.cooldown_seconds = 2  # Cooldown period to avoid multiple runs
        self.file_mappings = self._build_test_mappings()

    def _build_test_mappings(self):
        """Build mappings from source files to their corresponding test files."""
        mappings = {
            "app/indicators/rsi_indicator.py": "tests/unit/test_rsi_indicator.py",
            "app/indicators/indicator_factory.py": "tests/unit/test_indicator_factory.py",
            "app/indicators/base_indicator.py": "tests/unit/test_rsi_indicator.py",
            "app/risk_management/risk_manager.py": "tests/unit/test_risk_manager.py",
            "app/webhook/webhook_server.py": "tests/unit/test_webhook_server.py",
            "app/connectors/base_connector.py": "tests/unit/test_base_connector.py",
            "app/connectors/connector_factory.py": "tests/unit/test_connector_factory.py",
            "app/core/trading_engine.py": "tests/unit/test_trading_engine.py",
        }
        return mappings

    def on_modified(self, event):
        """Called when a file or directory is modified."""
        if event.is_directory:
            return

        current_time = time.time()
        if current_time - self.last_modified_time < self.cooldown_seconds:
            return

        self.last_modified_time = current_time
        file_path = event.src_path
        rel_path = os.path.relpath(file_path, ROOT_DIR)

        # Skip hidden files, cache files, and non-Python files
        if (
            os.path.basename(file_path).startswith(".")
            or "__pycache__" in file_path
            or not file_path.endswith(".py")
        ):
            return

        logger.info(f"File modified: {rel_path}")

        if rel_path.startswith("tests/"):
            # If a test file was modified, run that specific test
            self._run_test(rel_path)
        elif rel_path in self.file_mappings:
            # If a source file with a mapping was modified, run its corresponding test
            self._run_test(self.file_mappings[rel_path])
        elif rel_path.startswith("app/"):
            # For any other source file, try to find a matching test
            test_file = self._find_matching_test(rel_path)
            if test_file:
                self._run_test(test_file)
            else:
                logger.info(f"No matching test found for {rel_path}")

    def _find_matching_test(self, source_file):
        """Find a matching test file for a source file."""
        # Extract the base filename without extension
        filename = os.path.basename(source_file)
        base_name = os.path.splitext(filename)[0]

        # First, try to find a test file with the same name
        test_file = f"tests/unit/test_{base_name}.py"
        if os.path.exists(os.path.join(ROOT_DIR, test_file)):
            return test_file

        # If not found, check if there's a test file that contains the name
        test_dir = os.path.join(ROOT_DIR, "tests/unit")
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if (
                    file.startswith("test_")
                    and base_name in file
                    and file.endswith(".py")
                ):
                    return f"tests/unit/{file}"

        return None

    def _run_test(self, test_file):
        """Run a specific test file."""
        logger.info(f"Running test: {test_file}")
        try:
            cmd = [self.test_command, os.path.join(ROOT_DIR, test_file), "-v"]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=ROOT_DIR,
            )
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                logger.info("Tests passed!")
            else:
                logger.error("Tests failed!")

            # Print test output
            for line in stdout.splitlines():
                print(line)

            if stderr:
                for line in stderr.splitlines():
                    print(line)

        except Exception as e:
            logger.error(f"Error running test: {str(e)}")


def start_file_watcher(directories=None):
    """Start the file watcher."""
    if directories is None:
        directories = ["app", "tests"]

    logger.info("Starting file watcher...")
    logger.info(f"Watching directories: {', '.join(directories)}")

    event_handler = SparkStackerTestHandler()
    observer = Observer()

    for directory in directories:
        path = os.path.join(ROOT_DIR, directory)
        if os.path.exists(path):
            observer.schedule(event_handler, path, recursive=True)
        else:
            logger.warning(f"Directory {directory} does not exist!")

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping file watcher...")
        observer.stop()

    observer.join()


if __name__ == "__main__":
    # Parse command line arguments
    watch_dirs = ["app", "tests"]
    if len(sys.argv) > 1:
        watch_dirs = sys.argv[1:]

    start_file_watcher(watch_dirs)
