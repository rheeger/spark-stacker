"""
Integration tests for the CLI functionality.

Tests both the new modular CLI structure and backward compatibility
with the legacy CLI interface.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest


class TestCLIBackwardCompatibility:
    """Test backward compatibility with the legacy CLI interface."""

    @pytest.mark.parametrize("command", [
        ["--help"],
        ["list-indicators", "--help"],
        ["demo", "--help"],
    ])
    def test_legacy_cli_help_commands(self, command):
        """Test that legacy CLI help commands still work through the compatibility shim."""
        # Get root spark-app directory
        app_dir = Path(__file__).parent.parent.parent.resolve()

        # Path to the legacy CLI shim in tests/_utils/cli.py
        legacy_cli_path = app_dir / "tests" / "_utils" / "cli.py"

        # Construct the full command using the legacy CLI location
        full_command = [sys.executable, str(legacy_cli_path)] + command

        # Run the command with a timeout to ensure it completes quickly
        result = subprocess.run(
            full_command,
            cwd=app_dir,
            capture_output=True,
            text=True,
            timeout=10  # 10-second timeout as specified in the audit
        )

        # Check the command succeeded
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "Error" not in result.stdout

        # Verify deprecation warning appears in stderr
        assert "DEPRECATION WARNING" in result.stderr or "deprecated" in result.stderr.lower()

    @pytest.mark.parametrize("indicator", ["MACD"])
    def test_legacy_cli_list_indicators_output(self, indicator):
        """Test that the legacy list-indicators command shows expected indicators."""
        # Get root spark-app directory
        app_dir = Path(__file__).parent.parent.parent.resolve()

        # Path to the legacy CLI shim in tests/_utils/cli.py
        legacy_cli_path = app_dir / "tests" / "_utils" / "cli.py"

        # Construct the command using the legacy CLI location
        full_command = [sys.executable, str(legacy_cli_path), "list-indicators"]

        # Run the command
        result = subprocess.run(
            full_command,
            cwd=app_dir,
            capture_output=True,
            text=True,
            timeout=10
        )

        # Check for success and expected indicator
        assert result.returncode == 0
        assert indicator in result.stdout


class TestModularCLI:
    """Test the new modular CLI structure."""

    @pytest.mark.parametrize("command", [
        ["--help"],
        ["list-strategies", "--help"],
        ["strategy", "--help"],
        ["compare-strategies", "--help"],
    ])
    def test_modular_cli_help_commands(self, command):
        """Test that new modular CLI help commands work correctly."""
        # Get root spark-app directory
        app_dir = Path(__file__).parent.parent.parent.resolve()

        # Path to the new modular CLI main.py
        modular_cli_path = app_dir / "tests" / "_utils" / "cli" / "main.py"

        # Construct the full command using the new CLI location
        full_command = [sys.executable, str(modular_cli_path)] + command

        # Run the command with a timeout
        result = subprocess.run(
            full_command,
            cwd=app_dir,
            capture_output=True,
            text=True,
            timeout=15  # Slightly longer timeout for new CLI
        )

        # Check the command succeeded
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "Error" not in result.stdout

    def test_modular_cli_list_strategies(self):
        """Test the new list-strategies command."""
        # Get root spark-app directory
        app_dir = Path(__file__).parent.parent.parent.resolve()

        # Path to the new modular CLI main.py
        modular_cli_path = app_dir / "tests" / "_utils" / "cli" / "main.py"

        # Construct the command
        full_command = [sys.executable, str(modular_cli_path), "list-strategies"]

        # Run the command
        result = subprocess.run(
            full_command,
            cwd=app_dir,
            capture_output=True,
            text=True,
            timeout=15
        )

        # Check for success (even if no strategies are configured)
        assert result.returncode == 0
        # Command should either show strategies or indicate none are configured

    def test_modular_cli_config_validation(self):
        """Test the new validate-config command."""
        # Get root spark-app directory
        app_dir = Path(__file__).parent.parent.parent.resolve()

        # Path to the new modular CLI main.py
        modular_cli_path = app_dir / "tests" / "_utils" / "cli" / "main.py"

        # Construct the command
        full_command = [sys.executable, str(modular_cli_path), "validate-config"]

        # Run the command
        result = subprocess.run(
            full_command,
            cwd=app_dir,
            capture_output=True,
            text=True,
            timeout=15
        )

        # Command should complete (might succeed or fail depending on config)
        # but should not crash or hang
        assert result.returncode in [0, 1]  # Either success or validation failure


class TestCLIModuleImports:
    """Test that CLI modules can be imported correctly."""

    def test_import_cli_main(self):
        """Test that the main CLI module can be imported."""
        # Get the CLI directory
        cli_dir = Path(__file__).parent.parent.parent / "tests" / "_utils" / "cli"

        # Add to path
        import sys
        sys.path.insert(0, str(cli_dir))

        try:
            # Test importing main CLI
            from main import cli
            assert callable(cli)

            # Test importing core modules
            from core.config_manager import ConfigManager
            from core.data_manager import DataManager

            assert ConfigManager is not None
            assert DataManager is not None

        except ImportError as e:
            pytest.fail(f"Failed to import CLI modules: {e}")
        finally:
            # Clean up path
            if str(cli_dir) in sys.path:
                sys.path.remove(str(cli_dir))

    def test_import_command_modules(self):
        """Test that command modules can be imported."""
        # Get the CLI directory
        cli_dir = Path(__file__).parent.parent.parent / "tests" / "_utils" / "cli"

        # Add to path
        import sys
        sys.path.insert(0, str(cli_dir))

        try:
            # Test importing command modules
            from commands.indicator_commands import setup_indicator_commands
            from commands.list_commands import setup_list_commands
            from commands.strategy_commands import setup_strategy_commands

            assert callable(setup_strategy_commands)
            assert callable(setup_indicator_commands)
            assert callable(setup_list_commands)

        except ImportError as e:
            pytest.fail(f"Failed to import command modules: {e}")
        finally:
            # Clean up path
            if str(cli_dir) in sys.path:
                sys.path.remove(str(cli_dir))

    def test_import_manager_modules(self):
        """Test that manager modules can be imported."""
        # Get the CLI directory
        cli_dir = Path(__file__).parent.parent.parent / "tests" / "_utils" / "cli"

        # Add to path
        import sys
        sys.path.insert(0, str(cli_dir))

        try:
            # Test importing manager modules
            from managers.comparison_manager import ComparisonManager
            from managers.indicator_backtest_manager import \
                IndicatorBacktestManager
            from managers.strategy_backtest_manager import \
                StrategyBacktestManager

            assert StrategyBacktestManager is not None
            assert IndicatorBacktestManager is not None
            assert ComparisonManager is not None

        except ImportError as e:
            pytest.fail(f"Failed to import manager modules: {e}")
        finally:
            # Clean up path
            if str(cli_dir) in sys.path:
                sys.path.remove(str(cli_dir))


class TestCLICompatibilityShim:
    """Test the compatibility shim functionality."""

    def test_shim_imports_work(self):
        """Test that the compatibility shim imports work correctly."""
        # Get root spark-app directory
        app_dir = Path(__file__).parent.parent.parent.resolve()

        # Add to path
        import sys
        sys.path.insert(0, str(app_dir / "tests" / "_utils"))

        try:
            # Import from the shim (should trigger deprecation warning)
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                from cli import cli, list_strategies, load_config

                # Verify imports work
                assert callable(cli)
                assert callable(load_config)
                assert callable(list_strategies)

                # Verify deprecation warning was issued
                assert len(w) > 0
                assert any("deprecation" in str(warning.message).lower() for warning in w)

        except ImportError as e:
            pytest.fail(f"Failed to import from compatibility shim: {e}")
        finally:
            # Clean up path
            if str(app_dir / "tests" / "_utils") in sys.path:
                sys.path.remove(str(app_dir / "tests" / "_utils"))

    def test_shim_error_handling(self):
        """Test that the compatibility shim handles import errors gracefully."""
        # This test verifies that if the modular CLI is broken,
        # the shim provides helpful error messages

        # Get the shim file content to verify error handling exists
        app_dir = Path(__file__).parent.parent.parent.resolve()
        shim_path = app_dir / "tests" / "_utils" / "cli.py"

        assert shim_path.exists()

        with open(shim_path, 'r') as f:
            content = f.read()

        # Verify error handling code exists
        assert "ImportError" in content
        assert "except ImportError" in content
        assert "troubleshooting" in content.lower()


class TestEndToEndWorkflows:
    """Test end-to-end CLI workflows with both legacy and modular interfaces."""

    def test_legacy_to_modular_migration_suggestion(self):
        """Test that legacy CLI suggests migration to modular CLI."""
        # Get root spark-app directory
        app_dir = Path(__file__).parent.parent.parent.resolve()

        # Path to the legacy CLI shim
        legacy_cli_path = app_dir / "tests" / "_utils" / "cli.py"

        # Run a legacy command
        result = subprocess.run(
            [sys.executable, str(legacy_cli_path), "--help"],
            cwd=app_dir,
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should succeed but show migration guidance
        assert result.returncode == 0

        # Check for migration guidance in stderr (where deprecation warnings go)
        stderr_content = result.stderr.lower()
        assert any(keyword in stderr_content for keyword in [
            "deprecation", "deprecated", "migrate", "new location"
        ])

    def test_modular_cli_performance(self):
        """Test that modular CLI has reasonable startup performance."""
        # Get root spark-app directory
        app_dir = Path(__file__).parent.parent.parent.resolve()

        # Path to the new modular CLI
        modular_cli_path = app_dir / "tests" / "_utils" / "cli" / "main.py"

        import time
        start_time = time.time()

        # Run a simple command
        result = subprocess.run(
            [sys.executable, str(modular_cli_path), "--help"],
            cwd=app_dir,
            capture_output=True,
            text=True,
            timeout=15
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete successfully
        assert result.returncode == 0

        # Should complete in reasonable time (less than 5 seconds for help)
        assert execution_time < 5.0

    def test_error_propagation(self):
        """Test that errors are properly propagated through the modular structure."""
        # Get root spark-app directory
        app_dir = Path(__file__).parent.parent.parent.resolve()

        # Path to the new modular CLI
        modular_cli_path = app_dir / "tests" / "_utils" / "cli" / "main.py"

        # Run an invalid command
        result = subprocess.run(
            [sys.executable, str(modular_cli_path), "invalid-command"],
            cwd=app_dir,
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should fail gracefully with helpful error message
        assert result.returncode != 0

        # Should provide helpful error message
        output = result.stdout + result.stderr
        assert any(keyword in output.lower() for keyword in [
            "invalid", "unknown", "command", "usage", "help"
        ])
