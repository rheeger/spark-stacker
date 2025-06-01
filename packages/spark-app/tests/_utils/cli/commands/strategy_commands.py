"""
Strategy Command Handlers

This module contains CLI commands for strategy backtesting and management:
- strategy: Run backtest for a specific strategy from config
- compare-strategies: Compare multiple strategies with multi-scenario testing
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

# Add the CLI directory to the path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
cli_dir = os.path.dirname(current_dir)
spark_app_dir = os.path.dirname(os.path.dirname(os.path.dirname(cli_dir)))
sys.path.insert(0, spark_app_dir)
sys.path.insert(0, cli_dir)

# Import app-level components directly
from app.backtesting.backtest_engine import BacktestEngine
from app.backtesting.data_manager import DataManager as AppDataManager
from app.backtesting.indicator_backtest_manager import IndicatorBacktestManager
# Now use absolute imports from the CLI modules that are available
from core.config_manager import ConfigManager
from core.data_manager import DataManager as CLIDataManager
from core.scenario_manager import ScenarioManager
# Import managers directly since they're not in __init__.py yet
from managers.comparison_manager import ComparisonManager
from managers.scenario_backtest_manager import ScenarioBacktestManager
from managers.strategy_backtest_manager import StrategyBacktestManager
# Import reporting modules (these are properly exported)
from reporting.comparison_reporter import ComparisonReporter
from reporting.scenario_reporter import ScenarioReporter
from reporting.strategy_reporter import StrategyReporter
# Import CLI utility functions and error classes that exist
from utils import (ConfigurationError, DataFetchingError, StrategyError,
                   ValidationError, data_fetch_retry, graceful_degradation,
                   strategy_error_handler, validate_required_params)
# Import validation modules (these are properly exported)
from validation.strategy_validator import StrategyValidator

logger = logging.getLogger(__name__)


def register_strategy_commands(cli_group):
    """Register strategy-related commands with the CLI group."""
    cli_group.add_command(strategy)
    cli_group.add_command(compare_strategies)


def _validate_strategy_name(strategy_name: str) -> bool:
    """Validate that strategy name is not empty."""
    return strategy_name and strategy_name.strip()


def _validate_days(days: int) -> bool:
    """Validate that days is a positive integer."""
    return isinstance(days, int) and days > 0


def _validate_scenarios(scenarios: str) -> bool:
    """Validate that scenarios string is valid."""
    if not scenarios:
        return False

    valid_scenarios = {"all", "bull", "bear", "sideways", "volatile", "low-vol", "choppy", "gaps", "real"}
    if scenarios.lower() == "all":
        return True

    scenario_list = [s.strip().lower() for s in scenarios.split(",")]
    return all(s in valid_scenarios for s in scenario_list)


@click.command("strategy")
@click.option("--strategy-name", required=True, help="Name of strategy from config.json")
@click.option("--days", default=30, help="Number of days to test (default: 30)")
@click.option("--scenarios", default="all", help="Scenarios to run: all, bull, bear, sideways, volatile, low-vol, choppy, gaps, real, or comma-separated list")
@click.option("--scenario-only", is_flag=True, help="Run single scenario instead of full suite")
@click.option("--override-timeframe", help="Override strategy timeframe for testing")
@click.option("--override-market", help="Override strategy market for testing")
@click.option("--override-position-size", type=float, help="Override position sizing for testing")
@click.option("--use-real-data", is_flag=True, help="Use real data instead of synthetic (legacy compatibility)")
@click.option("--export-data", is_flag=True, help="Export scenario data for external analysis")
@click.option("--sensitivity-analysis", is_flag=True, help="Run configuration sensitivity analysis")
@click.option("--output-dir", help="Output directory for reports (default: tests/results)")
@click.pass_context
@strategy_error_handler
@validate_required_params(
    strategy_name=_validate_strategy_name,
    days=_validate_days,
    scenarios=_validate_scenarios
)
def strategy(ctx, strategy_name: str, days: int, scenarios: str, scenario_only: bool,
             override_timeframe: Optional[str], override_market: Optional[str],
             override_position_size: Optional[float], use_real_data: bool,
             export_data: bool, sensitivity_analysis: bool, output_dir: Optional[str]):
    """
    Run backtest for a specific strategy from config.json.

    By default, runs multi-scenario testing (all 7 synthetic scenarios + real data)
    to evaluate strategy robustness across different market conditions.

    Use --sensitivity-analysis to run comprehensive configuration sensitivity analysis
    that tests different position sizing methods, timeframes, indicator parameters,
    and risk management settings to identify optimization opportunities.

    Examples:
        # Run full multi-scenario test for a strategy
        python cli/main.py strategy --strategy-name "MACD_ETH_Long"

        # Test specific scenarios only
        python cli/main.py strategy --strategy-name "MACD_ETH_Long" --scenarios "bull,bear,real"

        # Run single scenario with overrides
        python cli/main.py strategy --strategy-name "MACD_ETH_Long" --scenario-only --scenarios "bull" --override-timeframe "4h"

        # Run with configuration sensitivity analysis
        python cli/main.py strategy --strategy-name "MACD_ETH_Long" --sensitivity-analysis
    """
    # Initialize managers and components with error handling
    try:
        config_manager = _initialize_config_manager(ctx)
        config = config_manager.load_config()

        # Create both data managers - app's for BacktestEngine, CLI's for strategy manager
        app_data_manager = AppDataManager()  # App's DataManager for BacktestEngine
        cli_data_manager = CLIDataManager(config=config)  # CLI's DataManager for data fetching

        strategy_validator = StrategyValidator(config_manager)
        backtest_engine = BacktestEngine(app_data_manager)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to initialize backtesting components: {str(e)}",
            fix_suggestions=[
                "Check that config.json exists and is valid",
                "Ensure all required dependencies are installed",
                "Verify database connections if required",
                "Check file permissions for config files"
            ],
            context={
                "config_path": ctx.obj.get('config_path', 'Not specified'),
                "working_dir": os.getcwd()
            }
        )

    # Determine and validate output directory
    output_path = _setup_output_directory(output_dir)

    # Initialize strategy backtest manager with enhanced error handling
    strategy_manager = _initialize_strategy_manager(
        backtest_engine, config_manager, cli_data_manager,  # Use CLI data manager
        strategy_validator, output_path
    )

    # Load and validate strategy configuration
    strategy_config = _load_strategy_config(strategy_manager, strategy_name)

    # Build and apply configuration overrides
    overrides, position_sizing_overrides = _build_overrides(
        strategy_manager, override_timeframe, override_market,
        override_position_size, strategy_config
    )

    # Initialize strategy components with comprehensive error handling
    _initialize_strategy_components(strategy_manager, overrides, position_sizing_overrides)

    # Display strategy information
    _display_strategy_info(strategy_name, strategy_config, days)

    # Parse and validate scenarios
    scenario_list = _parse_and_validate_scenarios(scenarios, use_real_data, scenario_only)

    # Execute backtesting with appropriate error handling
    _execute_strategy_backtest(
        strategy_manager, strategy_name, days, scenario_list,
        config_manager, cli_data_manager, output_path, export_data, sensitivity_analysis
    )


@graceful_degradation(fallback_value=None, log_level=logging.ERROR)
def _initialize_config_manager(ctx) -> ConfigManager:
    """Initialize configuration manager with error handling."""
    config_path = ctx.obj.get('config_path')

    # Allow None config_path - ConfigManager has fallback logic to find default configs
    try:
        config_manager = ConfigManager(config_path)
        # Test if we can actually load the config to validate the setup
        config_manager.load_config()
        return config_manager
    except Exception as e:
        # If config loading fails, provide helpful error message
        raise ConfigurationError(
            f"Failed to load configuration: {str(e)}",
            fix_suggestions=[
                "Ensure config.json exists in the shared directory",
                "Provide --config option with path to config.json",
                "Check config.json format and validity",
                "Verify file permissions for config files"
            ],
            context={
                "attempted_config_path": config_path or "default fallback paths",
                "working_dir": os.getcwd()
            }
        )


def _setup_output_directory(output_dir: Optional[str]) -> Path:
    """Setup and validate output directory."""
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "__test_results__")

    output_path = Path(output_dir)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise ConfigurationError(
            f"Cannot create output directory: {output_path}",
            fix_suggestions=[
                "Check directory permissions",
                "Choose a different output directory with write access",
                "Run with appropriate user privileges"
            ],
            context={"requested_dir": str(output_path)}
        )

    return output_path


def _initialize_strategy_manager(backtest_engine, config_manager, data_manager,
                               strategy_validator, output_path) -> StrategyBacktestManager:
    """Initialize strategy backtest manager with error handling."""
    try:
        return StrategyBacktestManager(
            backtest_engine=backtest_engine,
            config_manager=config_manager,
            data_manager=data_manager,
            strategy_validator=strategy_validator,
            output_dir=output_path
        )
    except Exception as e:
        raise StrategyError(
            f"Failed to initialize strategy backtest manager: {str(e)}",
            fix_suggestions=[
                "Check strategy configuration format",
                "Verify all required strategy components are available",
                "Ensure database connections are working"
            ]
        )


def _load_strategy_config(strategy_manager: StrategyBacktestManager, strategy_name: str):
    """Load and validate strategy configuration."""
    click.echo(f"🔍 Loading strategy: {strategy_name}")

    try:
        strategy_config = strategy_manager.load_strategy_from_config(strategy_name)
        if not strategy_config:
            raise StrategyError(
                f"Strategy '{strategy_name}' not found in configuration",
                fix_suggestions=[
                    "Check spelling of strategy name",
                    "List available strategies with: list-strategies",
                    "Verify strategy is defined in config.json",
                    "Ensure strategy is enabled in configuration"
                ],
                context={"requested_strategy": strategy_name}
            )
        return strategy_config
    except Exception as e:
        if isinstance(e, StrategyError):
            raise
        raise StrategyError(
            f"Failed to load strategy configuration: {str(e)}",
            fix_suggestions=[
                "Check config.json format and validity",
                "Verify strategy section exists in config",
                "Ensure all required strategy fields are present"
            ],
            context={"strategy_name": strategy_name}
        )


def _build_overrides(strategy_manager, override_timeframe, override_market,
                    override_position_size, strategy_config):
    """Build configuration overrides with validation."""
    overrides = {}
    position_sizing_overrides = {}

    if override_timeframe:
        # Validate timeframe format
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        if override_timeframe not in valid_timeframes:
            raise ValidationError(
                f"Invalid timeframe: {override_timeframe}",
                fix_suggestions=[
                    f"Use one of the valid timeframes: {', '.join(valid_timeframes)}",
                    "Check timeframe format (e.g., '1h', '4h', '1d')"
                ],
                context={"provided_timeframe": override_timeframe}
            )

        overrides['timeframe'] = override_timeframe
        click.echo(f"⚠️  Overriding timeframe: {strategy_config.timeframe} → {override_timeframe}")

    if override_market:
        # Basic market format validation
        if "-" not in override_market:
            raise ValidationError(
                f"Invalid market format: {override_market}",
                fix_suggestions=[
                    "Use format like 'ETH-USD' or 'BTC-USD'",
                    "Ensure market pair is separated by hyphen"
                ],
                context={"provided_market": override_market}
            )

        overrides['market'] = override_market
        click.echo(f"⚠️  Overriding market: {strategy_config.market} → {override_market}")

    if override_position_size:
        if override_position_size <= 0:
            raise ValidationError(
                f"Position size must be positive: {override_position_size}",
                fix_suggestions=[
                    "Provide a positive number for position size",
                    "Check position sizing method and appropriate range"
                ]
            )

        position_sizing_overrides = _handle_position_size_override(
            strategy_manager, override_position_size
        )

    return overrides, position_sizing_overrides


@graceful_degradation(fallback_value={}, log_level=logging.WARNING)
def _handle_position_size_override(strategy_manager, override_position_size):
    """Handle position sizing override with graceful degradation."""
    current_position_info = strategy_manager.get_current_position_sizing_info()

    if 'error' in current_position_info:
        click.echo(f"⚠️  Warning: Could not get current position sizing info: {current_position_info['error']}")
        # Return default override
        return {'fixed_usd_amount': override_position_size}

    method = current_position_info.get('method', 'fixed_usd')
    position_sizing_overrides = {}

    if method == 'fixed_usd':
        position_sizing_overrides['fixed_usd_amount'] = override_position_size
        click.echo(f"⚠️  Overriding fixed USD amount: ${current_position_info.get('fixed_usd_amount', 'unknown')} → ${override_position_size}")
    elif method == 'percentage_equity':
        position_sizing_overrides['equity_percentage'] = override_position_size / 100.0
        click.echo(f"⚠️  Overriding equity percentage: {current_position_info.get('equity_percentage', 'unknown')*100:.1f}% → {override_position_size}%")
    else:
        position_sizing_overrides['max_position_size_usd'] = override_position_size
        click.echo(f"⚠️  Overriding max position size: ${current_position_info.get('max_position_size_usd', 'unknown')} → ${override_position_size}")

    return position_sizing_overrides


def _initialize_strategy_components(strategy_manager, overrides, position_sizing_overrides):
    """Initialize strategy components with error handling."""
    try:
        click.echo("🔧 Initializing strategy components...")
        strategy_manager.initialize_strategy_components(overrides)
    except Exception as e:
        raise StrategyError(
            f"Failed to initialize strategy components: {str(e)}",
            fix_suggestions=[
                "Check strategy configuration validity",
                "Verify all indicators are available",
                "Ensure position sizing configuration is correct",
                "Check exchange connectivity if required"
            ]
        )

    # Apply position sizing overrides
    if position_sizing_overrides:
        try:
            click.echo("💰 Applying position sizing overrides...")
            strategy_manager.apply_position_sizing_overrides(position_sizing_overrides)

            # Display final position sizing configuration
            final_position_info = strategy_manager.get_current_position_sizing_info()
            if 'error' not in final_position_info:
                click.echo(f"💰 Position sizing method: {final_position_info['method']}")
                if final_position_info['method'] == 'fixed_usd':
                    click.echo(f"💰 Fixed USD amount: ${final_position_info['fixed_usd_amount']}")
                elif final_position_info['method'] == 'percentage_equity':
                    click.echo(f"💰 Equity percentage: {final_position_info['equity_percentage']*100:.1f}%")
                click.echo(f"💰 Max position size: ${final_position_info['max_position_size_usd']}")
        except Exception as e:
            raise StrategyError(
                f"Failed to apply position sizing overrides: {str(e)}",
                fix_suggestions=[
                    "Check position sizing override values",
                    "Verify position sizing method compatibility",
                    "Ensure position sizing configuration is valid"
                ]
            )


def _display_strategy_info(strategy_name, strategy_config, days):
    """Display strategy information."""
    click.echo(f"🎯 Running backtest for strategy: {strategy_name}")
    click.echo(f"📊 Market: {strategy_config.market}")
    click.echo(f"⏱️  Timeframe: {strategy_config.timeframe}")
    click.echo(f"📅 Duration: {days} days")


def _parse_and_validate_scenarios(scenarios: str, use_real_data: bool, scenario_only: bool) -> List[str]:
    """Parse and validate scenario configuration."""
    # Parse scenarios
    if scenarios.lower() == "all":
        scenario_list = ["bull", "bear", "sideways", "volatile", "low-vol", "choppy", "gaps", "real"]
    else:
        scenario_list = [s.strip() for s in scenarios.split(",")]

    # Validate scenario names
    valid_scenarios = ["bull", "bear", "sideways", "volatile", "low-vol", "choppy", "gaps", "real"]
    invalid_scenarios = [s for s in scenario_list if s not in valid_scenarios]
    if invalid_scenarios:
        raise ValidationError(
            f"Invalid scenarios: {', '.join(invalid_scenarios)}",
            fix_suggestions=[
                f"Use valid scenarios: {', '.join(valid_scenarios)}",
                "Check spelling of scenario names",
                "Use comma-separated list for multiple scenarios"
            ],
            context={
                "provided_scenarios": scenarios,
                "invalid_scenarios": invalid_scenarios
            }
        )

    click.echo(f"🎲 Scenarios: {', '.join(scenario_list)}")

    # Handle legacy use-real-data flag
    if use_real_data and "real" not in scenario_list:
        scenario_list = ["real"]
        click.echo("⚠️  Legacy --use-real-data flag detected, running real data scenario only")

    # Check for multi-scenario vs single scenario
    if scenario_only and len(scenario_list) > 1:
        click.echo("⚠️  --scenario-only flag specified but multiple scenarios selected. Running all specified scenarios.")

    return scenario_list


@data_fetch_retry
def _execute_strategy_backtest(strategy_manager, strategy_name, days, scenario_list,
                             config_manager, data_manager, output_path, export_data, sensitivity_analysis):
    """Execute strategy backtesting with retry and error handling."""
    try:
        if len(scenario_list) == 1 and scenario_list[0] == "real":
            # Single real data backtest (legacy compatibility)
            result = _execute_single_scenario_backtest(
                strategy_manager, strategy_name, days, config_manager, output_path
            )
        else:
            # Multi-scenario testing
            result = _execute_multi_scenario_backtest(
                strategy_manager, strategy_name, days, scenario_list,
                config_manager, data_manager, output_path
            )

        # Run configuration sensitivity analysis if requested
        if sensitivity_analysis:
            _execute_configuration_sensitivity_analysis(
                strategy_manager, strategy_name, result, config_manager, output_path
            )

        if export_data:
            click.echo(f"📁 Scenario data exported to: {output_path}/scenario_data/")

    except Exception as e:
        raise DataFetchingError(
            f"Strategy backtesting execution failed: {str(e)}",
            fix_suggestions=[
                "Check network connectivity for data fetching",
                "Verify exchange API access and rate limits",
                "Ensure sufficient disk space for results",
                "Try with fewer scenarios or shorter duration"
            ],
            context={
                "strategy_name": strategy_name,
                "days": days,
                "scenarios": scenario_list
            }
        )


def _execute_configuration_sensitivity_analysis(strategy_manager, strategy_name, base_results,
                                               config_manager, output_path):
    """Execute configuration sensitivity analysis for the strategy."""
    click.echo("\n🔬 Running configuration sensitivity analysis...")

    try:
        # Initialize strategy reporter
        strategy_reporter = StrategyReporter(config_manager)

        # Convert base results to the format expected by sensitivity analysis
        if hasattr(base_results, 'backtest_result'):
            backtest_data = {
                "trades": base_results.backtest_result.trades,
                "metrics": base_results.backtest_result.metrics,
                "equity_curve": getattr(base_results.backtest_result, 'equity_curve', [])
            }
        elif isinstance(base_results, dict):
            # Multi-scenario results - use the first scenario as base
            first_scenario = next(iter(base_results.values()))
            if hasattr(first_scenario, 'backtest_result'):
                backtest_data = {
                    "trades": first_scenario.backtest_result.trades,
                    "metrics": first_scenario.backtest_result.metrics,
                    "equity_curve": getattr(first_scenario.backtest_result, 'equity_curve', [])
                }
            else:
                backtest_data = {
                    "trades": getattr(first_scenario, 'trades', []),
                    "metrics": getattr(first_scenario, 'metrics', {}),
                    "equity_curve": getattr(first_scenario, 'equity_curve', [])
                }
        else:
            # Fallback for other result formats
            backtest_data = {
                "trades": getattr(base_results, 'trades', []),
                "metrics": getattr(base_results, 'metrics', {}),
                "equity_curve": getattr(base_results, 'equity_curve', [])
            }

        # Run sensitivity analysis
        analysis_results = strategy_reporter.generate_configuration_sensitivity_analysis(
            strategy_name=strategy_name,
            base_backtest_results=backtest_data,
            analysis_options={
                "test_position_sizing": True,
                "test_timeframes": True,
                "test_indicator_parameters": True,
                "test_risk_parameters": True
            }
        )

        # Save sensitivity analysis report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sensitivity_report_path = output_path / f"sensitivity_analysis_{strategy_name}_{timestamp}.json"

        with open(sensitivity_report_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)

        # Display summary of findings
        _display_sensitivity_analysis_summary(analysis_results)

        click.echo(f"📊 Sensitivity analysis completed!")
        click.echo(f"📋 Detailed report saved to: {sensitivity_report_path}")

    except Exception as e:
        click.echo(f"⚠️  Sensitivity analysis failed: {str(e)}")
        logger.error(f"Configuration sensitivity analysis error: {e}")


def _display_sensitivity_analysis_summary(analysis_results):
    """Display a summary of sensitivity analysis findings."""
    click.echo("\n📈 Configuration Sensitivity Analysis Summary:")

    # Display optimization suggestions
    suggestions = analysis_results.get("optimization_suggestions", [])
    if suggestions:
        click.echo(f"\n🎯 Top optimization opportunities ({len(suggestions)} found):")
        for i, suggestion in enumerate(suggestions[:3], 1):  # Show top 3
            category = suggestion.get("category", "unknown")
            description = suggestion.get("description", "No description")
            priority = suggestion.get("priority", "medium")
            click.echo(f"  {i}. [{priority.upper()}] {category}: {description}")

    # Display feasibility assessment
    feasibility = analysis_results.get("feasibility_analysis", {})
    overall_assessment = feasibility.get("overall_assessment", {})
    if overall_assessment:
        feasibility_level = overall_assessment.get("overall_feasibility", "unknown")
        feasible_count = overall_assessment.get("feasible_suggestions", 0)
        total_count = overall_assessment.get("total_suggestions", 0)
        effort_estimate = overall_assessment.get("effort_estimate", "unknown")

        click.echo(f"\n✅ Feasibility Assessment:")
        click.echo(f"  Overall feasibility: {feasibility_level}")
        click.echo(f"  Feasible suggestions: {feasible_count}/{total_count}")
        click.echo(f"  Estimated effort: {effort_estimate}")

    # Display sensitivity test results summary
    sensitivity_tests = analysis_results.get("sensitivity_tests", {})
    if sensitivity_tests:
        click.echo(f"\n🔍 Sensitivity Tests Completed:")
        for test_name, test_results in sensitivity_tests.items():
            if isinstance(test_results, dict):
                variations_count = len(test_results.get("variations", {}))
                recommendations_count = len(test_results.get("recommendations", []))
                click.echo(f"  {test_name}: {variations_count} variations tested, {recommendations_count} recommendations")


def _execute_single_scenario_backtest(strategy_manager, strategy_name, days, config_manager, output_path):
    """Execute single scenario backtest."""
    click.echo("\n🚀 Running single real data backtest...")

    try:
        result = strategy_manager.backtest_strategy(
            days=days,
            use_real_data=True,
            leverage=1.0
        )
    except Exception as e:
        error_msg = str(e)
        if "Unable to fetch data" in error_msg and "from any source" in error_msg:
            click.echo("❌ Failed to fetch real market data. No exchange connectors are available.")
            click.echo("\n💡 To use real data, you need to:")
            click.echo("   1. Set exchange credentials in environment variables:")
            click.echo("      - For Hyperliquid: WALLET_ADDRESS, PRIVATE_KEY")
            click.echo("      - For Coinbase: COINBASE_API_KEY, COINBASE_API_SECRET")
            click.echo("      - For Kraken: KRAKEN_API_KEY, KRAKEN_API_SECRET")
            click.echo("   2. Or run with synthetic data scenarios instead:")
            click.echo(f"      python cli/main.py strategy --strategy-name {strategy_name} --scenarios bull,bear,sideways")
            raise click.ClickException("Cannot run real data backtest without exchange credentials")
        else:
            raise

    # Generate strategy report
    strategy_reporter = StrategyReporter(config_manager)

    # Create a report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"strategy_report_{strategy_name}_{timestamp}.json"
    report_path = output_path / report_filename

    # Generate the report with the output path
    report_data = strategy_reporter.generate_strategy_report(
        strategy_name=strategy_name,
        backtest_results=result,
        output_path=report_path
    )

    click.echo(f"✅ Backtest completed successfully!")
    click.echo(f"📊 Total trades: {result.metrics.get('total_trades', 0)}")
    click.echo(f"💰 Total return: {result.metrics.get('total_return_pct', 0):.2f}%")
    click.echo(f"📈 Win rate: {result.metrics.get('win_rate', 0):.1f}%")
    click.echo(f"📋 Report saved to: {report_path}")

    return result


def _execute_multi_scenario_backtest(strategy_manager, strategy_name, days, scenario_list,
                                   config_manager, data_manager, output_path):
    """Execute multi-scenario backtest."""
    # Check if real data is available
    if "real" in scenario_list:
        # Check if any connectors are available in the DataManager
        if not hasattr(data_manager, '_connectors') or not data_manager._connectors:
            click.echo("⚠️  No exchange connectors available - removing 'real' scenario from test")
            click.echo("💡 To enable real data scenarios, set exchange credentials in environment variables")
            scenario_list = [s for s in scenario_list if s != "real"]
            if not scenario_list:
                click.echo("⚠️  No scenarios remaining after removing 'real'. Adding synthetic scenarios.")
                scenario_list = ["bull", "bear", "sideways"]

    click.echo(f"\n🚀 Running multi-scenario backtest across {len(scenario_list)} scenarios...")

    # Initialize scenario manager and scenario backtest manager
    scenario_manager = ScenarioManager(
        data_manager=data_manager,
        config_manager=config_manager,
        default_duration_days=days
    )

    scenario_backtest_manager = ScenarioBacktestManager(
        config_manager=config_manager,
        data_manager=data_manager,
        scenario_manager=scenario_manager,
        strategy_manager=strategy_manager,
        output_dir=output_path
    )

    # Run scenario testing
    include_real_data = "real" in scenario_list
    scenario_results = scenario_backtest_manager.run_strategy_scenarios(
        strategy_name=strategy_name,
        days=days,
        scenario_filter=scenario_list,
        include_real_data=include_real_data
    )

    # Generate scenario report
    scenario_reporter = ScenarioReporter(config_manager, scenario_manager)
    report_path = scenario_reporter.generate_multi_scenario_report(
        strategy_name=strategy_name,
        scenario_results=scenario_results,
        output_path=output_path
    )

    # Display summary
    click.echo(f"✅ Multi-scenario backtest completed!")
    click.echo(f"📊 Scenarios tested: {len(scenario_results)}")

    # Show key metrics across scenarios
    for scenario_name, result in scenario_results.items():
        if hasattr(result, 'backtest_result'):
            metrics = result.backtest_result.metrics
            click.echo(f"  {scenario_name}: {metrics.get('total_return_pct', 0):.1f}% return, {metrics.get('total_trades', 0)} trades")
        elif hasattr(result, 'total_return'):
            click.echo(f"  {scenario_name}: {result.total_return:.1f}% return, {result.total_trades} trades")

    click.echo(f"📋 Comprehensive report saved to: {report_path}")

    # Calculate and display robustness score
    _calculate_and_display_robustness_score(scenario_results)

    return scenario_results


@graceful_degradation(fallback_value=None, log_level=logging.WARNING)
def _calculate_and_display_robustness_score(scenario_results):
    """Calculate and display strategy robustness score."""
    total_returns = []
    for result in scenario_results.values():
        if hasattr(result, 'backtest_result'):
            total_returns.append(result.backtest_result.metrics.get('total_return_pct', 0))
        elif hasattr(result, 'total_return'):
            total_returns.append(result.total_return)

    if total_returns:
        avg_return = sum(total_returns) / len(total_returns)
        return_variance = sum((r - avg_return) ** 2 for r in total_returns) / len(total_returns) if len(total_returns) > 1 else 0
        consistency_score = max(0, 100 - (return_variance / (avg_return ** 2) * 100)) if avg_return != 0 else 0
        click.echo(f"🎯 Strategy robustness score: {consistency_score:.1f}% (higher is better)")


@click.command("compare-strategies")
@click.option("--strategy-names", help="Comma-separated list of strategy names to compare")
@click.option("--all-strategies", is_flag=True, help="Compare all enabled strategies")
@click.option("--same-market", help="Filter strategies to specific market (e.g., ETH-USD)")
@click.option("--same-exchange", help="Filter strategies to specific exchange")
@click.option("--days", default=30, help="Number of days to test (default: 30)")
@click.option("--scenarios", default="all", help="Scenarios for comparison: all, bull, bear, etc.")
@click.option("--output-dir", help="Output directory for reports (default: tests/results)")
@click.pass_context
@strategy_error_handler
def compare_strategies(ctx, strategy_names: Optional[str], all_strategies: bool,
                      same_market: Optional[str], same_exchange: Optional[str],
                      days: int, scenarios: str, output_dir: Optional[str]):
    """
    Compare multiple strategies with multi-scenario testing.

    Runs all strategies through the same set of market scenarios for fair comparison,
    including cross-scenario robustness scoring.

    Examples:
        # Compare specific strategies
        python cli/main.py compare-strategies --strategy-names "MACD_ETH_Long,RSI_ETH_Short"

        # Compare all enabled strategies
        python cli/main.py compare-strategies --all-strategies

        # Compare strategies on same market
        python cli/main.py compare-strategies --all-strategies --same-market "ETH-USD"

        # Compare with specific scenarios
        python cli/main.py compare-strategies --all-strategies --scenarios "bull,bear,real"
    """
    try:
        # Initialize managers and components
        config_manager = ConfigManager(ctx.obj.get('config_path'))
        config = config_manager.load_config()

        # Create both data managers - app's for BacktestEngine, CLI's for scenario manager
        app_data_manager = AppDataManager()  # App's DataManager for BacktestEngine
        cli_data_manager = CLIDataManager(config=config)  # CLI's DataManager for data fetching

        strategy_validator = StrategyValidator(config_manager)
        backtest_engine = BacktestEngine(app_data_manager)

        # Determine output directory
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "__test_results__")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine strategies to compare
        if all_strategies:
            strategies_to_compare = config_manager.list_strategies(
                filter_exchange=same_exchange,
                filter_market=same_market,
                filter_enabled=True
            )
        elif strategy_names:
            strategy_name_list = [name.strip() for name in strategy_names.split(",")]
            strategies_to_compare = []

            for name in strategy_name_list:
                strategy_config = config_manager.get_strategy_config(name)
                if not strategy_config:
                    available_strategies = [s.get('name', 'unnamed') for s in config_manager.list_strategies()]
                    raise click.ClickException(
                        f"Strategy '{name}' not found in configuration.\n"
                        f"Available strategies: {', '.join(available_strategies)}"
                    )

                # Apply filters if specified
                if same_market and strategy_config.get('market', '').upper() != same_market.upper():
                    continue
                if same_exchange and strategy_config.get('exchange', '').lower() != same_exchange.lower():
                    continue

                strategies_to_compare.append(strategy_config)
        else:
            raise click.ClickException("Must specify either --strategy-names or --all-strategies")

        if not strategies_to_compare:
            raise click.ClickException("No strategies found matching the specified criteria")

        # Parse scenarios
        if scenarios.lower() == "all":
            scenario_list = ["bull", "bear", "sideways", "volatile", "low-vol", "choppy", "gaps", "real"]
        else:
            scenario_list = [s.strip() for s in scenarios.split(",")]

        # Validate scenario names
        valid_scenarios = ["bull", "bear", "sideways", "volatile", "low-vol", "choppy", "gaps", "real"]
        invalid_scenarios = [s for s in scenario_list if s not in valid_scenarios]
        if invalid_scenarios:
            raise click.ClickException(
                f"Invalid scenarios: {', '.join(invalid_scenarios)}\n"
                f"Valid scenarios: {', '.join(valid_scenarios)}"
            )

        # Check if real data is available
        if "real" in scenario_list:
            # Check if any connectors are available in the DataManager
            if not hasattr(cli_data_manager, '_connectors') or not cli_data_manager._connectors:
                click.echo("⚠️  No exchange connectors available - removing 'real' scenario from comparison")
                click.echo("💡 To enable real data scenarios, set exchange credentials in environment variables")
                scenario_list = [s for s in scenario_list if s != "real"]
                if not scenario_list:
                    click.echo("⚠️  No scenarios remaining after removing 'real'. Using synthetic scenarios.")
                    scenario_list = ["bull", "bear", "sideways", "volatile"]

        # Display comparison setup
        click.echo(f"🔍 Comparing {len(strategies_to_compare)} strategies:")
        for i, strategy in enumerate(strategies_to_compare, 1):
            click.echo(f"   {i}. {strategy.get('name', 'unnamed')} ({strategy.get('market', 'unknown')}, {strategy.get('exchange', 'unknown')})")

        click.echo(f"\n📅 Duration: {days} days")
        click.echo(f"🎲 Scenarios: {', '.join(scenario_list)}")

        if same_market:
            click.echo(f"🎯 Market filter: {same_market}")
        if same_exchange:
            click.echo(f"🏢 Exchange filter: {same_exchange}")

        # Initialize comparison manager
        comparison_manager = ComparisonManager(
            config_manager=config_manager,
            data_manager=cli_data_manager,
            output_dir=output_path
        )

        # Run strategy comparison
        click.echo(f"\n🚀 Running strategy comparison across {len(scenario_list)} scenarios...")
        strategy_names_list = [s['name'] for s in strategies_to_compare]

        comparison_results = comparison_manager.run_strategy_comparison(
            strategy_names=strategy_names_list,
            scenarios=scenario_list,
            days=days
        )

        # Generate comparison report using correct initialization and method
        comparison_reporter = ComparisonReporter(config_manager, comparison_manager)

        # Prepare output path for report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"strategy_comparison_{timestamp}.html"
        report_path = output_path / report_filename

        # Generate the strategy comparison report
        report_data = comparison_reporter.generate_strategy_comparison_report(
            strategy_names=strategy_names_list,
            comparison_results=comparison_results.strategy_results,
            output_path=report_path
        )

        # Display summary results
        click.echo(f"✅ Strategy comparison completed!")
        click.echo(f"📊 Strategies compared: {len(comparison_results.strategy_results)}")
        click.echo(f"🎲 Scenarios tested: {len(scenario_list)}")

        # Show ranking summary from report data
        click.echo(f"\n🏆 Strategy Rankings:")
        ranking_analysis = report_data.get('ranking_analysis', {})
        final_ranking = ranking_analysis.get('final_ranking', [])

        for i, (strategy_name, score_data) in enumerate(final_ranking[:5], 1):  # Top 5
            total_score = score_data.get('total_score', 0)
            click.echo(f"   {i}. {strategy_name}: Score {total_score}")

        # Show performance comparison summary
        performance_comparison = report_data.get('performance_comparison', {})
        best_performers = performance_comparison.get('best_performers', {})

        if best_performers:
            click.echo(f"\n🥇 Best Performers by Metric:")
            for metric, performer_data in best_performers.items():
                strategy = performer_data.get('strategy', 'Unknown')
                value = performer_data.get('value', 0)
                click.echo(f"   • {metric.replace('_', ' ').title()}: {strategy} ({value:.2f})")

        # Show diversification insights
        diversification_analysis = report_data.get('diversification_analysis', {})
        diversification_insights = diversification_analysis.get('diversification_insights', [])

        if diversification_insights:
            click.echo(f"\n🔄 Diversification Insights:")
            for insight in diversification_insights[:3]:  # Top 3 insights
                click.echo(f"   • {insight}")

        # Show portfolio allocation suggestions
        allocation_suggestions = report_data.get('allocation_suggestions', {})
        recommended_allocations = allocation_suggestions.get('recommended_allocations', {})

        if recommended_allocations:
            click.echo(f"\n💼 Recommended Portfolio Allocation:")
            for strategy, allocation in recommended_allocations.items():
                if allocation > 0:
                    click.echo(f"   • {strategy}: {allocation}%")

        # Show market condition analysis
        market_condition_analysis = report_data.get('market_condition_analysis', {})
        best_performers_by_condition = market_condition_analysis.get('best_performers_by_condition', {})

        if best_performers_by_condition:
            click.echo(f"\n🌊 Best Performers by Market Condition:")
            for condition, performer_data in best_performers_by_condition.items():
                strategy = performer_data.get('strategy', 'Unknown')
                return_pct = performer_data.get('return', 0)
                click.echo(f"   • {condition.title()}: {strategy} ({return_pct:.1f}%)")

        # Show adaptability scores
        adaptability_scores = market_condition_analysis.get('adaptability_scores', {})
        if adaptability_scores:
            click.echo(f"\n🎯 Strategy Adaptability (Consistency Across Conditions):")
            sorted_adaptability = sorted(adaptability_scores.items(),
                                       key=lambda x: x[1].get('score', 0), reverse=True)
            for strategy, score_data in sorted_adaptability[:3]:  # Top 3
                score = score_data.get('score', 0)
                mean_return = score_data.get('mean_return', 0)
                click.echo(f"   • {strategy}: {score:.1f}/100 (avg {mean_return:.1f}% across conditions)")

        # Show sensitivity analysis highlights
        sensitivity_analysis = report_data.get('sensitivity_analysis', {})
        optimization_potential = sensitivity_analysis.get('optimization_potential', {})

        high_priority_strategies = sensitivity_analysis.get('parameter_impact_rankings', {}).get('high_priority_strategies', [])
        if high_priority_strategies:
            click.echo(f"\n⚡ High Optimization Potential:")
            for strategy in high_priority_strategies[:3]:  # Top 3
                strategy_data = optimization_potential.get(strategy, {})
                improvement = strategy_data.get('max_estimated_improvement', 0)
                optimization = strategy_data.get('best_optimization', 'Unknown')
                click.echo(f"   • {strategy}: +{improvement:.1f}% potential via {optimization}")

        click.echo(f"\n📋 Detailed comparison report saved to: {report_path}")

    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        raise click.ClickException(f"Strategy comparison failed: {str(e)}")
