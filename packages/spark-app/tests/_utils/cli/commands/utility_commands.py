"""
Utility Command Handlers

This module contains CLI utility commands:
- validate-config: Configuration validation and checking
- migrate-config: Config file migration and updates
- diagnose: System troubleshooting and diagnostics
- clean-cache: Cache cleaning and maintenance
- export-examples: Generate example configurations
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

logger = logging.getLogger(__name__)


def register_utility_commands(cli_group):
    """Register utility commands with the CLI group."""
    cli_group.add_command(validate_config)
    cli_group.add_command(migrate_config)
    cli_group.add_command(diagnose)
    cli_group.add_command(clean_cache)
    cli_group.add_command(export_examples)
    cli_group.add_command(test_real_data)
    cli_group.add_command(diagnose_data_issues)


@click.command("validate-config")
@click.option("--config", help="Path to configuration file to validate")
@click.option("--fix", is_flag=True, help="Attempt to fix common configuration issues")
@click.option("--strict", is_flag=True, help="Use strict validation rules")
@click.option("--output-report", help="Save validation report to file")
@click.pass_context
def validate_config(ctx, config: Optional[str], fix: bool, strict: bool, output_report: Optional[str]):
    """
    Validate configuration file structure and content.

    Checks configuration for common issues, validates strategy and indicator
    definitions, and provides suggestions for improvements.

    Examples:
        # Validate default config
        python cli/main.py validate-config

        # Validate specific config with strict rules
        python cli/main.py validate-config --config my-config.json --strict

        # Validate and attempt fixes
        python cli/main.py validate-config --fix

        # Generate validation report
        python cli/main.py validate-config --output-report validation-report.txt
    """
    # Import configuration utilities from main module
    from ..main import load_config
    from ..main import validate_config as validate_config_func

    # Determine config file path
    config_path = config or ctx.obj.get('config_path')

    click.echo("🔍 Configuration Validation")
    click.echo("=" * 50)

    if config_path:
        click.echo(f"📄 Config file: {config_path}")
    else:
        click.echo("📄 Config file: Using default path discovery")

    click.echo(f"⚙️  Validation mode: {'Strict' if strict else 'Standard'}")
    click.echo()

    # Load configuration
    try:
        config_data = load_config(config_path)
        if not config_data:
            raise click.ClickException("Could not load configuration file")

        click.echo("✅ Configuration file loaded successfully")

    except Exception as e:
        click.echo(f"❌ Failed to load configuration: {e}")
        if fix:
            click.echo("\n🔧 Attempting to diagnose and fix...")
            # TODO: Implement configuration repair logic
            click.echo("   Configuration repair logic will be implemented")
        return

    # Validate configuration
    validation_errors = validate_config_func(config_data)

    # Additional strict validation checks
    if strict:
        additional_errors = []

        # Check for unused indicators
        strategies = config_data.get('strategies', [])
        indicators = config_data.get('indicators', [])

        used_indicators = set()
        for strategy in strategies:
            used_indicators.update(strategy.get('indicators', []))

        indicator_names = set(ind.get('name') for ind in indicators)
        unused_indicators = indicator_names - used_indicators

        if unused_indicators:
            additional_errors.append(f"Unused indicators found: {', '.join(unused_indicators)}")

        # Check for missing exchange configurations
        required_exchanges = set(s.get('exchange') for s in strategies if s.get('exchange'))
        available_exchanges = set(e.get('name') for e in config_data.get('exchanges', []))
        missing_exchanges = required_exchanges - available_exchanges

        if missing_exchanges:
            additional_errors.append(f"Missing exchange configurations: {', '.join(missing_exchanges)}")

        # Check for inconsistent timeframes
        timeframe_issues = []
        for strategy in strategies:
            strategy_timeframe = strategy.get('timeframe')
            strategy_indicators = strategy.get('indicators', [])

            for indicator_name in strategy_indicators:
                indicator = next((ind for ind in indicators if ind.get('name') == indicator_name), None)
                if indicator and indicator.get('timeframe') != strategy_timeframe:
                    timeframe_issues.append(
                        f"Strategy '{strategy.get('name')}' timeframe ({strategy_timeframe}) "
                        f"differs from indicator '{indicator_name}' timeframe ({indicator.get('timeframe')})"
                    )

        additional_errors.extend(timeframe_issues)
        validation_errors.extend(additional_errors)

    # Display validation results
    click.echo("\n📋 Validation Results:")
    click.echo("-" * 30)

    if not validation_errors:
        click.echo("✅ Configuration is valid!")

        # Show summary statistics
        strategies = config_data.get('strategies', [])
        indicators = config_data.get('indicators', [])
        exchanges = config_data.get('exchanges', [])

        click.echo(f"\n📊 Configuration Summary:")
        click.echo(f"   Strategies: {len(strategies)} ({len([s for s in strategies if s.get('enabled', True)])} enabled)")
        click.echo(f"   Indicators: {len(indicators)} ({len([i for i in indicators if i.get('enabled', True)])} enabled)")
        click.echo(f"   Exchanges: {len(exchanges)} ({len([e for e in exchanges if e.get('enabled', True)])} enabled)")

    else:
        click.echo(f"❌ Found {len(validation_errors)} validation errors:")
        click.echo()

        for i, error in enumerate(validation_errors, 1):
            click.echo(f"   {i}. {error}")

        if fix:
            click.echo("\n🔧 Attempting to fix configuration...")
            # TODO: Implement configuration fixing logic
            click.echo("   Configuration fixing logic will be implemented")
            click.echo("   This will include:")
            click.echo("   • Removing unused indicators")
            click.echo("   • Adding missing exchange configurations")
            click.echo("   • Fixing timeframe inconsistencies")
            click.echo("   • Correcting validation errors")

    # Generate report if requested
    if output_report:
        report_data = {
            'validation_timestamp': click.DateTime().convert(None, None, ctx),
            'config_file': config_path or 'default',
            'validation_mode': 'strict' if strict else 'standard',
            'errors': validation_errors,
            'summary': {
                'total_strategies': len(config_data.get('strategies', [])),
                'total_indicators': len(config_data.get('indicators', [])),
                'total_exchanges': len(config_data.get('exchanges', [])),
                'validation_passed': len(validation_errors) == 0
            }
        }

        try:
            with open(output_report, 'w') as f:
                json.dump(report_data, f, indent=2)
            click.echo(f"\n📄 Validation report saved to: {output_report}")
        except Exception as e:
            click.echo(f"\n❌ Failed to save report: {e}")

    # Provide next steps
    click.echo("\n💡 Next steps:")
    if validation_errors:
        click.echo("   • Fix validation errors before running strategies")
        click.echo("   • Use --fix flag to attempt automatic repairs")
        if not strict:
            click.echo("   • Use --strict for additional validation checks")
    else:
        click.echo("   • Configuration is ready for strategy backtesting")
        click.echo("   • Use 'list-strategies' to see available strategies")
        click.echo("   • Use 'strategy --strategy-name <name>' to run backtests")


@click.command("migrate-config")
@click.option("--config", help="Path to configuration file to migrate")
@click.option("--output", help="Output path for migrated config (default: <original>-migrated.json)")
@click.option("--backup", is_flag=True, default=True, help="Create backup of original config")
@click.option("--dry-run", is_flag=True, help="Show what would be migrated without making changes")
@click.pass_context
def migrate_config(ctx, config: Optional[str], output: Optional[str], backup: bool, dry_run: bool):
    """
    Migrate configuration file to latest format.

    Updates configuration files to match the latest schema and adds
    missing fields with default values.

    Examples:
        # Migrate default config
        python cli/main.py migrate-config

        # Migrate specific config with custom output
        python cli/main.py migrate-config --config old-config.json --output new-config.json

        # Preview migration without making changes
        python cli/main.py migrate-config --dry-run
    """
    # Determine config file path
    config_path = config or ctx.obj.get('config_path')

    if not config_path or not os.path.exists(config_path):
        raise click.ClickException(f"Configuration file not found: {config_path}")

    # Determine output path
    if not output:
        base_path = os.path.splitext(config_path)[0]
        output = f"{base_path}-migrated.json"

    click.echo("🔄 Configuration Migration")
    click.echo("=" * 40)
    click.echo(f"📄 Source: {config_path}")
    click.echo(f"📄 Target: {output}")
    click.echo(f"🔧 Mode: {'Dry run' if dry_run else 'Live migration'}")
    click.echo()

    # Load current configuration
    try:
        with open(config_path, 'r') as f:
            current_config = json.load(f)
        click.echo("✅ Configuration loaded successfully")
    except Exception as e:
        raise click.ClickException(f"Failed to load configuration: {e}")

    # Analyze what needs to be migrated
    migration_changes = []

    # Check for missing version
    if 'version' not in current_config:
        migration_changes.append("Add version field")

    # Check for missing required sections
    required_sections = ['strategies', 'indicators', 'exchanges', 'global_settings']
    for section in required_sections:
        if section not in current_config:
            migration_changes.append(f"Add missing '{section}' section")

    # Check for deprecated field names
    deprecated_fields = ['old_field_name', 'legacy_setting', 'position_size']
    for field in deprecated_fields:
        if field in current_config:
            migration_changes.append(f"Migrate deprecated field '{field}'")

    # Check for missing schema version
    if 'schema_version' not in current_config:
        migration_changes.append("Add schema version tracking")

    # Display migration plan
    if migration_changes:
        click.echo("\n📋 Migration plan:")
        for i, change in enumerate(migration_changes, 1):
            click.echo(f"   {i}. {change}")
    else:
        click.echo("✅ Configuration is already up to date")
        return

    if dry_run:
        click.echo(f"\n🔍 Dry run complete - no changes made")
        click.echo(f"   Run without --dry-run to apply {len(migration_changes)} changes")
        return

    # Create backup if requested
    if backup:
        backup_path = f"{config_path}.backup"
        try:
            shutil.copy2(config_path, backup_path)
            click.echo(f"📄 Backup created: {backup_path}")
        except Exception as e:
            click.echo(f"⚠️  Failed to create backup: {e}")

    # Apply migration changes
    click.echo("\n🔄 Applying migration...")
    migrated_config = current_config.copy()

    # Add version if missing
    if 'version' not in migrated_config:
        migrated_config['version'] = '1.0.0'
        click.echo("   ✅ Added version field")

    # Add missing required sections
    required_sections = {
        'strategies': [],
        'indicators': {},
        'exchanges': {},
        'global_settings': {
            'default_timeframe': '1h',
            'default_position_size': 0.1,
            'risk_management': {
                'max_portfolio_risk': 0.02,
                'max_position_risk': 0.01
            }
        }
    }

    for section, default_value in required_sections.items():
        if section not in migrated_config:
            migrated_config[section] = default_value
            click.echo(f"   ✅ Added '{section}' section")

    # Migrate deprecated field names
    deprecated_migrations = {
        'old_field_name': 'new_field_name',
        'legacy_setting': 'modern_setting',
        'position_size': 'position_sizing'  # Example migration
    }

    for old_field, new_field in deprecated_migrations.items():
        if old_field in migrated_config and new_field not in migrated_config:
            migrated_config[new_field] = migrated_config.pop(old_field)
            click.echo(f"   ✅ Migrated '{old_field}' to '{new_field}'")

    # Update schema version to indicate migration
    migrated_config['schema_version'] = '2.0.0'
    migrated_config['migrated_at'] = datetime.now().isoformat()

    # Save migrated configuration
    try:
        with open(output, 'w') as f:
            json.dump(migrated_config, f, indent=2)
        click.echo(f"✅ Migration completed successfully")
        click.echo(f"📄 Migrated config saved to: {output}")

        # Validate the migrated config
        try:
            from ..validation.config_validator import ConfigValidator
            validator = ConfigValidator()
            validation_result = validator.validate_config(migrated_config)

            if validation_result.is_valid:
                click.echo("✅ Migrated configuration is valid")
            else:
                click.echo("⚠️  Migrated configuration has validation issues:")
                for error in validation_result.errors[:3]:  # Show first 3 errors
                    click.echo(f"   • {error}")
                if len(validation_result.errors) > 3:
                    click.echo(f"   ... and {len(validation_result.errors) - 3} more")

        except ImportError:
            click.echo("⚠️  Configuration validation skipped (validator not available)")

    except Exception as e:
        raise click.ClickException(f"Failed to save migrated configuration: {e}")


@click.command("diagnose")
@click.option("--component", type=click.Choice(['all', 'config', 'data', 'connections', 'dependencies']),
              default='all', help="Component to diagnose (default: all)")
@click.option("--output-report", help="Save diagnostic report to file")
@click.pass_context
def diagnose(ctx, component: str, output_report: Optional[str]):
    """
    Run system diagnostics and troubleshooting.

    Checks system configuration, dependencies, data sources, and
    connection health to identify potential issues.

    Examples:
        # Full system diagnosis
        python cli/main.py diagnose

        # Diagnose specific component
        python cli/main.py diagnose --component config

        # Generate diagnostic report
        python cli/main.py diagnose --output-report system-diagnosis.txt
    """
    click.echo("🩺 System Diagnostics")
    click.echo("=" * 30)
    click.echo(f"🔍 Checking: {component}")
    click.echo()

    diagnostic_results = {}

    # Configuration diagnostics
    if component in ['all', 'config']:
        click.echo("📋 Configuration Health:")
        config_issues = []

        try:
            from ..main import load_config
            config = load_config(ctx.obj.get('config_path'))
            if config:
                click.echo("   ✅ Configuration loads successfully")
                strategies = config.get('strategies', [])
                indicators = config.get('indicators', [])
                click.echo(f"   📊 Found {len(strategies)} strategies, {len(indicators)} indicators")
            else:
                config_issues.append("Configuration file not found or invalid")
                click.echo("   ❌ Configuration loading failed")
        except Exception as e:
            config_issues.append(f"Configuration error: {e}")
            click.echo(f"   ❌ Configuration error: {e}")

        diagnostic_results['config'] = config_issues
        click.echo()

    # Dependencies diagnostics
    if component in ['all', 'dependencies']:
        click.echo("📦 Dependencies Health:")
        dependency_issues = []

        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 11):
            click.echo(f"   ✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            dependency_issues.append(f"Python version {python_version.major}.{python_version.minor} < 3.11 (recommended)")
            click.echo(f"   ⚠️  Python version: {python_version.major}.{python_version.minor}.{python_version.micro} (recommend 3.11+)")

        # Check critical imports
        critical_modules = ['click', 'pandas', 'numpy', 'matplotlib']
        for module in critical_modules:
            try:
                __import__(module)
                click.echo(f"   ✅ {module} available")
            except ImportError:
                dependency_issues.append(f"Missing module: {module}")
                click.echo(f"   ❌ {module} not available")

        diagnostic_results['dependencies'] = dependency_issues
        click.echo()

    # Data sources diagnostics
    if component in ['all', 'data']:
        click.echo("📊 Data Sources Health:")
        data_issues = []

        # Check data directories
        tests_dir = os.path.dirname(os.path.dirname(__file__))
        results_dir = os.path.join(tests_dir, "__test_results__")

        if os.path.exists(results_dir):
            click.echo(f"   ✅ Results directory exists: {results_dir}")
        else:
            click.echo(f"   📁 Results directory will be created: {results_dir}")

        # Check for cached data
        cache_patterns = ["*.csv", "*.json", "*.pkl"]
        cache_files = []
        for pattern in cache_patterns:
            cache_files.extend(Path(tests_dir).rglob(pattern))

        if cache_files:
            click.echo(f"   📄 Found {len(cache_files)} cached data files")
        else:
            click.echo("   📄 No cached data files found")

        diagnostic_results['data'] = data_issues
        click.echo()

    # Connections diagnostics
    if component in ['all', 'connections']:
        click.echo("🌐 Connections Health:")
        connection_issues = []

        # Test internet connectivity (basic)
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            click.echo("   ✅ Internet connectivity available")
        except OSError:
            connection_issues.append("No internet connectivity")
            click.echo("   ❌ Internet connectivity issues")

        # TODO: Test exchange API connections
        click.echo("   📡 Exchange API testing pending implementation")

        diagnostic_results['connections'] = connection_issues
        click.echo()

    # Summary
    total_issues = sum(len(issues) for issues in diagnostic_results.values())

    if total_issues == 0:
        click.echo("✅ All systems healthy!")
    else:
        click.echo(f"⚠️  Found {total_issues} potential issues:")
        for component_name, issues in diagnostic_results.items():
            if issues:
                click.echo(f"\n   {component_name.title()}:")
                for issue in issues:
                    click.echo(f"     • {issue}")

    # Generate report if requested
    if output_report:
        report_data = {
            'diagnostic_timestamp': str(click.DateTime().convert(None, None, ctx)),
            'component_checked': component,
            'results': diagnostic_results,
            'total_issues': total_issues,
            'system_info': {
                'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'platform': sys.platform
            }
        }

        try:
            with open(output_report, 'w') as f:
                json.dump(report_data, f, indent=2)
            click.echo(f"\n📄 Diagnostic report saved to: {output_report}")
        except Exception as e:
            click.echo(f"\n❌ Failed to save report: {e}")

    # Provide next steps
    click.echo("\n💡 Next steps:")
    if total_issues > 0:
        click.echo("   • Address identified issues before running strategies")
        click.echo("   • Check documentation for troubleshooting guides")
    else:
        click.echo("   • System is ready for backtesting operations")
        click.echo("   • Use 'validate-config' to check configuration")


@click.command("clean-cache")
@click.option("--type", "cache_type", type=click.Choice(['all', 'data', 'reports', 'logs']),
              default='all', help="Type of cache to clean (default: all)")
@click.option("--dry-run", is_flag=True, help="Show what would be cleaned without removing files")
@click.option("--force", is_flag=True, help="Force removal without confirmation")
@click.pass_context
def clean_cache(ctx, cache_type: str, dry_run: bool, force: bool):
    """
    Clean cached data and temporary files.

    Removes cached data files, old reports, and log files to free up space
    and ensure fresh data for new backtests.

    Examples:
        # Clean all caches
        python cli/main.py clean-cache

        # Clean only data cache
        python cli/main.py clean-cache --type data

        # Preview what would be cleaned
        python cli/main.py clean-cache --dry-run

        # Force clean without confirmation
        python cli/main.py clean-cache --force
    """
    click.echo("🧹 Cache Cleaning")
    click.echo("=" * 25)
    click.echo(f"🗂️  Cache type: {cache_type}")
    click.echo(f"🔍 Mode: {'Dry run' if dry_run else 'Live cleaning'}")
    click.echo()

    tests_dir = os.path.dirname(os.path.dirname(__file__))

    files_to_remove = []
    total_size = 0

    # Data cache files
    if cache_type in ['all', 'data']:
        data_patterns = ['**/*.csv', '**/*.pkl', '**/*cache*', '**/*.parquet']
        click.echo("📊 Scanning data cache files...")

        for pattern in data_patterns:
            for file_path in Path(tests_dir).glob(pattern):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    files_to_remove.append(('data', file_path, size))
                    total_size += size

    # Report files
    if cache_type in ['all', 'reports']:
        report_patterns = ['**/results/**/*.html', '**/results/**/*.json', '**/results/**/*.png']
        click.echo("📈 Scanning report files...")

        for pattern in report_patterns:
            for file_path in Path(tests_dir).glob(pattern):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    files_to_remove.append(('reports', file_path, size))
                    total_size += size

    # Log files
    if cache_type in ['all', 'logs']:
        log_patterns = ['**/*.log', '**/*.log.*', '**/logs/**/*']
        click.echo("📝 Scanning log files...")

        for pattern in log_patterns:
            for file_path in Path(tests_dir).glob(pattern):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    files_to_remove.append(('logs', file_path, size))
                    total_size += size

    # Display summary
    if not files_to_remove:
        click.echo("✅ No cache files found to clean")
        return

    # Group by type
    by_type = {}
    for file_type, file_path, size in files_to_remove:
        if file_type not in by_type:
            by_type[file_type] = {'count': 0, 'size': 0, 'files': []}
        by_type[file_type]['count'] += 1
        by_type[file_type]['size'] += size
        by_type[file_type]['files'].append(file_path)

    def format_size(size_bytes):
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    click.echo("📋 Files to be removed:")
    for file_type, info in by_type.items():
        click.echo(f"   {file_type.title()}: {info['count']} files ({format_size(info['size'])})")

    click.echo(f"\n📊 Total: {len(files_to_remove)} files ({format_size(total_size)})")

    if dry_run:
        click.echo("\n🔍 Dry run complete - no files removed")
        click.echo("   Run without --dry-run to actually remove files")
        return

    # Confirmation
    if not force:
        click.echo()
        if not click.confirm(f"Are you sure you want to remove {len(files_to_remove)} files ({format_size(total_size)})?"):
            click.echo("❌ Cache cleaning cancelled")
            return

    # Remove files
    click.echo("\n🗑️  Removing files...")
    removed_count = 0
    removed_size = 0
    errors = []

    for file_type, file_path, size in files_to_remove:
        try:
            file_path.unlink()
            removed_count += 1
            removed_size += size
        except Exception as e:
            errors.append(f"{file_path}: {e}")

    # Summary
    click.echo(f"✅ Removed {removed_count} files ({format_size(removed_size)})")

    if errors:
        click.echo(f"\n⚠️  {len(errors)} errors encountered:")
        for error in errors[:5]:  # Show first 5 errors
            click.echo(f"   {error}")
        if len(errors) > 5:
            click.echo(f"   ... and {len(errors) - 5} more errors")

    # Remove empty directories
    click.echo("\n📁 Cleaning empty directories...")
    empty_dirs_removed = 0
    for root, dirs, files in os.walk(tests_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):  # Directory is empty
                    os.rmdir(dir_path)
                    empty_dirs_removed += 1
            except OSError:
                pass  # Directory not empty or permission error

    if empty_dirs_removed > 0:
        click.echo(f"✅ Removed {empty_dirs_removed} empty directories")

    click.echo("\n💡 Cache cleaning complete!")


@click.command("export-examples")
@click.option("--type", "example_type", type=click.Choice(['config', 'strategy', 'indicator', 'all']),
              default='all', help="Type of examples to export (default: all)")
@click.option("--output-dir", help="Output directory for examples (default: examples/)")
@click.option("--overwrite", is_flag=True, help="Overwrite existing example files")
@click.pass_context
def export_examples(ctx, example_type: str, output_dir: Optional[str], overwrite: bool):
    """
    Generate example configuration files and templates.

    Creates example configurations for strategies, indicators, and full configs
    to help users get started with their own configurations.

    Examples:
        # Export all examples
        python cli/main.py export-examples

        # Export only strategy examples
        python cli/main.py export-examples --type strategy

        # Export to custom directory
        python cli/main.py export-examples --output-dir my-examples/

        # Overwrite existing examples
        python cli/main.py export-examples --overwrite
    """
    # Determine output directory
    if not output_dir:
        tests_dir = os.path.dirname(os.path.dirname(__file__))
        output_dir = os.path.join(tests_dir, "examples")

    os.makedirs(output_dir, exist_ok=True)

    click.echo("📄 Exporting Examples")
    click.echo("=" * 30)
    click.echo(f"📁 Output directory: {output_dir}")
    click.echo(f"📋 Example type: {example_type}")
    click.echo()

    files_created = []

    # Configuration examples
    if example_type in ['all', 'config']:
        config_examples = {
            'minimal-config.json': {
                "version": "1.0.0",
                "strategies": [],
                "indicators": [],
                "exchanges": [
                    {
                        "name": "hyperliquid",
                        "enabled": true,
                        "api_key": "${HYPERLIQUID_API_KEY}",
                        "secret": "${HYPERLIQUID_SECRET}"
                    }
                ]
            },
            'full-config-example.json': {
                "version": "1.0.0",
                "global_settings": {
                    "default_timeframe": "1h",
                    "default_position_size": 0.1,
                    "risk_per_trade_pct": 0.02
                },
                "strategies": [
                    {
                        "name": "MACD_ETH_Long",
                        "market": "ETH-USD",
                        "exchange": "hyperliquid",
                        "timeframe": "1h",
                        "enabled": true,
                        "indicators": ["MACD_ETH_1h"],
                        "max_position_size": 0.2,
                        "risk_per_trade_pct": 0.015,
                        "stop_loss_pct": 3.0,
                        "take_profit_pct": 6.0
                    }
                ],
                "indicators": [
                    {
                        "name": "MACD_ETH_1h",
                        "type": "MACD",
                        "symbol": "ETHUSDT",
                        "timeframe": "1h",
                        "enabled": true,
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9
                    }
                ],
                "exchanges": [
                    {
                        "name": "hyperliquid",
                        "enabled": true,
                        "api_key": "${HYPERLIQUID_API_KEY}",
                        "secret": "${HYPERLIQUID_SECRET}",
                        "testnet": false
                    }
                ]
            }
        }

        for filename, config in config_examples.items():
            file_path = os.path.join(output_dir, filename)
            if os.path.exists(file_path) and not overwrite:
                click.echo(f"⚠️  Skipping {filename} (already exists, use --overwrite)")
                continue

            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                files_created.append(filename)
                click.echo(f"✅ Created {filename}")
            except Exception as e:
                click.echo(f"❌ Failed to create {filename}: {e}")

    # Strategy examples
    if example_type in ['all', 'strategy']:
        strategy_examples = {
            'strategy-template.json': {
                "name": "YOUR_STRATEGY_NAME",
                "market": "ETH-USD",
                "exchange": "hyperliquid",
                "timeframe": "1h",
                "enabled": true,
                "indicators": ["YOUR_INDICATOR_NAME"],
                "max_position_size": 0.1,
                "risk_per_trade_pct": 0.02,
                "main_leverage": 1.0,
                "hedge_leverage": 1.0,
                "hedge_ratio": 0.0,
                "stop_loss_pct": 2.0,
                "take_profit_pct": 4.0
            },
            'popular-strategies.json': [
                {
                    "name": "MACD_Crossover_Long",
                    "description": "Long positions based on MACD crossovers",
                    "market": "ETH-USD",
                    "timeframe": "4h",
                    "indicators": ["MACD_ETH_4h"],
                    "risk_per_trade_pct": 0.015
                },
                {
                    "name": "RSI_Oversold_Bounce",
                    "description": "Buy oversold conditions using RSI",
                    "market": "BTC-USD",
                    "timeframe": "1h",
                    "indicators": ["RSI_BTC_1h"],
                    "risk_per_trade_pct": 0.02
                }
            ]
        }

        for filename, strategy_data in strategy_examples.items():
            file_path = os.path.join(output_dir, filename)
            if os.path.exists(file_path) and not overwrite:
                click.echo(f"⚠️  Skipping {filename} (already exists, use --overwrite)")
                continue

            try:
                with open(file_path, 'w') as f:
                    json.dump(strategy_data, f, indent=2)
                files_created.append(filename)
                click.echo(f"✅ Created {filename}")
            except Exception as e:
                click.echo(f"❌ Failed to create {filename}: {e}")

    # Indicator examples
    if example_type in ['all', 'indicator']:
        indicator_examples = {
            'indicator-template.json': {
                "name": "YOUR_INDICATOR_NAME",
                "type": "MACD",
                "symbol": "ETHUSDT",
                "timeframe": "1h",
                "enabled": true,
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9
            },
            'common-indicators.json': [
                {
                    "name": "MACD_ETH_1h",
                    "type": "MACD",
                    "symbol": "ETHUSDT",
                    "timeframe": "1h",
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9
                },
                {
                    "name": "RSI_BTC_1h",
                    "type": "RSI",
                    "symbol": "BTCUSDT",
                    "timeframe": "1h",
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30
                },
                {
                    "name": "EMA_Cross_ETH_4h",
                    "type": "EMA",
                    "symbol": "ETHUSDT",
                    "timeframe": "4h",
                    "fast_period": 9,
                    "slow_period": 21
                }
            ]
        }

        for filename, indicator_data in indicator_examples.items():
            file_path = os.path.join(output_dir, filename)
            if os.path.exists(file_path) and not overwrite:
                click.echo(f"⚠️  Skipping {filename} (already exists, use --overwrite)")
                continue

            try:
                with open(file_path, 'w') as f:
                    json.dump(indicator_data, f, indent=2)
                files_created.append(filename)
                click.echo(f"✅ Created {filename}")
            except Exception as e:
                click.echo(f"❌ Failed to create {filename}: {e}")

    # Create README
    readme_path = os.path.join(output_dir, "README.md")
    if not os.path.exists(readme_path) or overwrite:
        readme_content = """# Spark-App Configuration Examples

This directory contains example configuration files to help you get started with Spark-App.

## Files Overview

### Configuration Examples
- `minimal-config.json` - Minimal configuration with basic structure
- `full-config-example.json` - Complete configuration with all sections

### Strategy Examples
- `strategy-template.json` - Template for creating new strategies
- `popular-strategies.json` - Examples of popular trading strategies

### Indicator Examples
- `indicator-template.json` - Template for creating new indicators
- `common-indicators.json` - Examples of commonly used indicators

## Usage

1. Copy an example file to your project directory
2. Rename it to `config.json` or use `--config path/to/config.json`
3. Modify the values to match your trading setup
4. Replace `${VARIABLE}` placeholders with actual values or environment variables

## Environment Variables

Make sure to set these environment variables for API access:
- `HYPERLIQUID_API_KEY` - Your Hyperliquid API key
- `HYPERLIQUID_SECRET` - Your Hyperliquid secret

## Next Steps

- Use `validate-config` to check your configuration
- Use `list-strategies` to see available strategies
- Use `strategy --strategy-name <name>` to run backtests

For more information, see the main documentation.
"""

        try:
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            files_created.append("README.md")
            click.echo(f"✅ Created README.md")
        except Exception as e:
            click.echo(f"❌ Failed to create README.md: {e}")

    # Summary
    if files_created:
        click.echo(f"\n📊 Summary: Created {len(files_created)} example files")
        click.echo("\n💡 Next steps:")
        click.echo(f"   • Review examples in: {output_dir}")
        click.echo("   • Copy and modify examples for your use case")
        click.echo("   • Use 'validate-config' to check your configurations")
    else:
        click.echo("\n⚠️  No files created (all examples already exist)")
        click.echo("   Use --overwrite to replace existing examples")


@click.command()
@click.option('--exchange', help='Test specific exchange (hyperliquid, coinbase, kraken)')
@click.option('--symbol', default='ETH-USD', help='Symbol to test data fetching for')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def test_real_data(ctx, exchange: Optional[str], symbol: str, verbose: bool):
    """Test real data connectivity and diagnose issues."""
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    click.echo("🔍 Testing Real Data Connectivity...")
    click.echo("=" * 50)

    try:
        from tests._utils.test_real_data_isolation import (
            load_environment_variables, test_config_based_initialization,
            test_connector_creation_isolated, test_real_data_fetching)

        # Load and check environment variables
        click.echo("\n📋 Checking Environment Variables...")
        load_environment_variables()

        # Test connector creation
        click.echo("\n🔧 Testing Connector Creation...")
        connectors = test_connector_creation_isolated()

        if exchange and exchange in connectors:
            # Test specific exchange
            specific_connectors = {exchange: connectors[exchange]}
            click.echo(f"\n📊 Testing {exchange.upper()} Data Fetching...")
            results = test_real_data_fetching(specific_connectors)
        elif connectors:
            # Test all available connectors
            click.echo(f"\n📊 Testing Data Fetching for {len(connectors)} connectors...")
            results = test_real_data_fetching(connectors)
        else:
            click.echo("\n❌ No connectors available for testing")
            click.echo("\nTo set up real data access, create a .env file with exchange credentials:")
            click.echo("- packages/shared/.env (recommended)")
            click.echo("- packages/spark-app/.env")

            click.echo("\nRequired environment variables:")
            click.echo("HYPERLIQUID: WALLET_ADDRESS, PRIVATE_KEY")
            click.echo("COINBASE: COINBASE_API_KEY, COINBASE_API_SECRET")
            click.echo("KRAKEN: KRAKEN_API_KEY, KRAKEN_API_SECRET")
            return

        # Display results
        click.echo("\n📈 Data Fetching Results:")
        for exchange_name, exchange_results in results.items():
            click.echo(f"\n{exchange_name.upper()}:")
            for key, value in exchange_results.items():
                if 'error' in key:
                    click.echo(f"  ❌ {key}: {value}")
                else:
                    click.echo(f"  ✅ {key}: {value} candles")

        # Test config-based initialization
        click.echo("\n⚙️  Testing Config-Based Initialization...")
        config_connectors = test_config_based_initialization()

        if config_connectors:
            click.echo(f"✅ Config-based initialization successful: {len(config_connectors)} connectors")
        else:
            click.echo("❌ Config-based initialization failed")

        # Summary
        click.echo("\n📊 Summary:")
        click.echo(f"Direct connectors: {len(connectors)}")
        click.echo(f"Config connectors: {len(config_connectors)}")

        if connectors or config_connectors:
            click.echo("✅ Real data collection is working!")

            # Provide CLI usage hint
            click.echo("\n💡 To use real data in CLI commands:")
            click.echo("python tests/_utils/cli/main.py real-data RSI --symbol ETH-USD --days 7")
        else:
            click.echo("❌ Real data collection needs setup")

    except Exception as e:
        click.echo(f"❌ Test failed: {e}")
        if verbose:
            import traceback
            click.echo(traceback.format_exc())


@click.command()
@click.option('--check-env', is_flag=True, help='Check environment variables only')
@click.option('--check-config', is_flag=True, help='Check configuration only')
@click.option('--check-connectors', is_flag=True, help='Check connector creation only')
@click.pass_context
def diagnose_data_issues(ctx, check_env: bool, check_config: bool, check_connectors: bool):
    """Diagnose common data fetching issues."""
    click.echo("🔬 Diagnosing Data Fetching Issues...")
    click.echo("=" * 40)

    if not any([check_env, check_config, check_connectors]):
        # Run all checks by default
        check_env = check_config = check_connectors = True

    try:
        if check_env:
            click.echo("\n📋 Environment Variables Check:")
            env_vars = {
                'WALLET_ADDRESS': 'Hyperliquid wallet address',
                'PRIVATE_KEY': 'Hyperliquid private key',
                'COINBASE_API_KEY': 'Coinbase API key',
                'COINBASE_API_SECRET': 'Coinbase API secret',
                'KRAKEN_API_KEY': 'Kraken API key',
                'KRAKEN_API_SECRET': 'Kraken API secret'
            }

            missing_vars = []
            for var, description in env_vars.items():
                value = os.environ.get(var, '')
                if value:
                    click.echo(f"  ✅ {var}: Set ({len(value)} chars)")
                else:
                    click.echo(f"  ❌ {var}: Not set ({description})")
                    missing_vars.append(var)

            if missing_vars:
                click.echo(f"\n⚠️  Missing {len(missing_vars)} environment variables")
                click.echo("   These are needed for real data fetching")

        if check_config:
            click.echo("\n⚙️  Configuration Check:")
            try:
                from pathlib import Path
                config_path = Path(__file__).parents[4] / "shared" / "config.json"

                if config_path.exists():
                    click.echo(f"  ✅ Config file found: {config_path}")

                    import json
                    with open(config_path) as f:
                        config = json.load(f)

                    exchanges = config.get('exchanges', [])
                    enabled_exchanges = [ex for ex in exchanges if ex.get('enabled', False)]

                    click.echo(f"  📊 Total exchanges configured: {len(exchanges)}")
                    click.echo(f"  ✅ Enabled exchanges: {len(enabled_exchanges)}")

                    for ex in enabled_exchanges:
                        click.echo(f"    - {ex.get('name', 'unknown')}: {ex.get('exchange_type', 'unknown')}")

                    if not enabled_exchanges:
                        click.echo("  ⚠️  No exchanges are enabled in config")
                        click.echo("     Set 'enabled': true for at least one exchange")
                else:
                    click.echo(f"  ❌ Config file not found: {config_path}")

            except Exception as e:
                click.echo(f"  ❌ Config check failed: {e}")

        if check_connectors:
            click.echo("\n🔧 Connector Creation Check:")
            try:
                from app.connectors.connector_factory import ConnectorFactory
                available_types = ConnectorFactory.get_available_connectors()
                click.echo(f"  📋 Available connector types: {', '.join(available_types)}")

                # Test creating connectors without credentials
                for exchange_type in available_types:
                    try:
                        connector = ConnectorFactory.create_connector(
                            exchange_type=exchange_type,
                            name=f"test_{exchange_type}",
                            wallet_address="test",
                            private_key="test",
                            api_key="test",
                            api_secret="test",
                            testnet=True
                        )
                        if connector:
                            click.echo(f"  ✅ {exchange_type}: Connector creation possible")
                        else:
                            click.echo(f"  ❌ {exchange_type}: Connector creation failed")
                    except Exception as e:
                        click.echo(f"  ⚠️  {exchange_type}: {str(e)[:50]}...")

            except Exception as e:
                click.echo(f"  ❌ Connector check failed: {e}")

        # Provide recommendations
        click.echo("\n💡 Recommendations:")
        click.echo("1. Create .env file with exchange credentials")
        click.echo("2. Enable at least one exchange in config.json")
        click.echo("3. Use testnet/sandbox credentials for testing")
        click.echo("4. Run 'test-real-data' command to verify setup")

    except Exception as e:
        click.echo(f"❌ Diagnosis failed: {e}")
