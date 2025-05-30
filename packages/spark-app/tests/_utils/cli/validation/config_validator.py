"""
Configuration Validator

This module provides comprehensive configuration validation:
- Add comprehensive configuration validation
- Add configuration repair suggestions
- Add configuration compatibility checking
- Add configuration performance analysis
- Add configuration optimization recommendations
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from app.core.strategy_config import StrategyConfig
from app.indicators.indicator_factory import IndicatorFactory
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class ConfigIssue:
    """Represents a configuration issue with severity and fix suggestions."""
    severity: str  # 'error', 'warning', 'suggestion'
    category: str  # 'structure', 'compatibility', 'performance', 'best_practice'
    message: str
    location: str  # Where in config the issue occurs
    fix_suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class ConfigValidationResult:
    """Result of configuration validation with detailed issues and suggestions."""
    is_valid: bool
    issues: List[ConfigIssue]
    performance_score: float  # 0-100 score for configuration quality
    optimization_suggestions: List[str]
    repair_suggestions: List[str]

    def add_issue(self, issue: ConfigIssue) -> None:
        """Add a configuration issue."""
        self.issues.append(issue)
        if issue.severity == 'error':
            self.is_valid = False

    def get_errors(self) -> List[ConfigIssue]:
        """Get all error-level issues."""
        return [issue for issue in self.issues if issue.severity == 'error']

    def get_warnings(self) -> List[ConfigIssue]:
        """Get all warning-level issues."""
        return [issue for issue in self.issues if issue.severity == 'warning']

    def get_suggestions(self) -> List[ConfigIssue]:
        """Get all suggestion-level issues."""
        return [issue for issue in self.issues if issue.severity == 'suggestion']


class ConfigValidator:
    """
    Validates configuration files for correctness, compatibility,
    performance, and provides optimization recommendations.
    """

    # Required global configuration sections
    REQUIRED_GLOBAL_SECTIONS = {
        'exchanges', 'position_sizing', 'risk_management', 'strategies'
    }

    # Required strategy fields
    REQUIRED_STRATEGY_FIELDS = {
        'name', 'market', 'exchange', 'timeframe', 'indicators', 'enabled'
    }

    # Valid timeframes in order of granularity
    VALID_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']

    # Valid exchanges
    VALID_EXCHANGES = ['hyperliquid', 'coinbase', 'kraken', 'binance']

    # Valid position sizing methods
    VALID_POSITION_SIZING_METHODS = [
        'fixed_usd', 'percentage_equity', 'kelly_criterion', 'volatility_adjusted'
    ]

    # Performance thresholds
    PERFORMANCE_THRESHOLDS = {
        'max_strategies_per_exchange': 10,
        'max_indicators_per_strategy': 5,
        'max_timeframe_spread': 3,  # Max difference in timeframe indices
        'recommended_indicator_types': 3,  # Different types per strategy
    }

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the configuration validator.

        Args:
            config_manager: ConfigManager instance for configuration access
        """
        self.config_manager = config_manager
        # Import StrategyValidator here to avoid circular imports
        from .strategy_validator import StrategyValidator
        self.strategy_validator = StrategyValidator(config_manager)

    def get_available_indicator_types(self) -> List[str]:
        """
        Get dynamically available indicator types from IndicatorFactory.

        Returns:
            List of available indicator type names
        """
        try:
            return IndicatorFactory.get_available_indicators()
        except Exception as e:
            logger.warning(f"Could not get available indicators from factory: {e}")
            # Fallback to common indicator types if factory fails
            return ['rsi', 'macd', 'bollinger', 'ma', 'sma', 'ema']

    def _check_indicator_availability(self, strategy_name: str, result: ConfigValidationResult) -> None:
        """Check if all strategy indicators are available in the IndicatorFactory."""
        try:
            config = self.config_manager.load_config()
            strategy_config = config.get('strategies', {}).get(strategy_name, {})
            indicators = strategy_config.get('indicators', {})

            # Get currently available indicator types from the factory
            available_types = self.get_available_indicator_types()

            logger.debug(f"Available indicator types: {available_types}")

            for indicator_name, indicator_config in indicators.items():
                indicator_type = indicator_config.get('type')
                if not indicator_type:
                    result.add_issue(ConfigIssue(
                        severity='error',
                        category='structure',
                        message=f'Indicator {indicator_name} missing type specification',
                        location=f'strategies.{strategy_name}.indicators.{indicator_name}',
                        fix_suggestion='Add indicator type',
                        auto_fixable=True
                    ))
                elif indicator_type not in available_types:
                    result.add_issue(ConfigIssue(
                        severity='error',
                        category='compatibility',
                        message=f'Unknown indicator type: {indicator_type}. Available types: {", ".join(available_types)}',
                        location=f'strategies.{strategy_name}.indicators.{indicator_name}',
                        fix_suggestion=f'Use one of the available indicator types: {", ".join(available_types)}',
                        auto_fixable=False
                    ))
                else:
                    # Indicator type is valid, check if it can actually be created
                    try:
                        test_indicator = IndicatorFactory.create_indicator(
                            name=f"test_{indicator_name}",
                            indicator_type=indicator_type,
                            params=indicator_config.get('parameters', {})
                        )
                        if test_indicator is None:
                            result.add_issue(ConfigIssue(
                                severity='warning',
                                category='compatibility',
                                message=f'Indicator {indicator_name} of type {indicator_type} could not be created with given parameters',
                                location=f'strategies.{strategy_name}.indicators.{indicator_name}',
                                fix_suggestion='Check indicator parameters and ensure they are valid',
                                auto_fixable=False
                            ))
                        else:
                            logger.debug(f"Successfully validated indicator {indicator_name} ({indicator_type})")
                    except Exception as e:
                        result.add_issue(ConfigIssue(
                            severity='warning',
                            category='compatibility',
                            message=f'Error creating indicator {indicator_name}: {e}',
                            location=f'strategies.{strategy_name}.indicators.{indicator_name}',
                            fix_suggestion='Review indicator configuration and parameters',
                            auto_fixable=False
                        ))

        except Exception as e:
            logger.warning(f"Could not check indicator availability for {strategy_name}: {e}")

    def validate_full_config(self, config_path: Optional[str] = None) -> ConfigValidationResult:
        """
        Validate the complete configuration file.

        Args:
            config_path: Optional path to config file, uses default if None

        Returns:
            ConfigValidationResult with comprehensive validation results
        """
        result = ConfigValidationResult(
            is_valid=True,
            issues=[],
            performance_score=0.0,
            optimization_suggestions=[],
            repair_suggestions=[]
        )

        try:
            # Load configuration using ConfigManager
            if config_path:
                # For custom config paths, we need to temporarily set the path
                # Store original config path
                original_path = getattr(self.config_manager, 'config_path', None)
                try:
                    # Set custom path and load
                    self.config_manager.config_path = config_path
                    config = self.config_manager.load_config(force_reload=True)
                finally:
                    # Restore original path
                    if original_path:
                        self.config_manager.config_path = original_path
            else:
                config = self.config_manager.load_config()

            if not config:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='structure',
                    message='Configuration file could not be loaded or is empty',
                    location='root',
                    fix_suggestion='Ensure config file exists and contains valid JSON',
                    auto_fixable=False
                ))
                return result

            # Validate global structure
            self._validate_global_structure(config, result)

            # Validate exchange configurations
            self._validate_exchange_configs(config, result)

            # Validate global position sizing
            self._validate_global_position_sizing(config, result)

            # Validate global risk management
            self._validate_global_risk_management(config, result)

            # Validate strategies section
            self._validate_strategies_section(config, result)

            # Validate individual strategies using StrategyValidator integration
            self._validate_individual_strategies_with_strategy_validator(config, result)

            # Use utility validation functions
            self._validate_strategy_indicator_consistency_all(config, result)
            self._validate_timeframe_consistency_all(config, result)
            self._validate_market_exchange_compatibility_all(config, result)

            # Check compatibility between components
            self._validate_component_compatibility(config, result)

            # Analyze performance characteristics
            self._analyze_performance_characteristics(config, result)

            # Generate optimization suggestions
            self._generate_optimization_suggestions(config, result)

            # Generate repair suggestions
            self._generate_repair_suggestions(result)

            # Calculate performance score
            result.performance_score = self._calculate_performance_score(result)

            logger.debug(f"Configuration validation completed. Valid: {result.is_valid}, "
                        f"Issues: {len(result.issues)}, Score: {result.performance_score:.1f}")

        except Exception as e:
            result.add_issue(ConfigIssue(
                severity='error',
                category='structure',
                message=f'Unexpected error during validation: {e}',
                location='validation_process',
                fix_suggestion='Check configuration file format and content',
                auto_fixable=False
            ))
            logger.error(f"Configuration validation failed: {e}")

        return result

    def _validate_global_structure(self, config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """Validate overall configuration structure."""
        # Check required sections
        for section in self.REQUIRED_GLOBAL_SECTIONS:
            if section not in config:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='structure',
                    message=f'Required section "{section}" missing from configuration',
                    location='root',
                    fix_suggestion=f'Add "{section}" section to configuration',
                    auto_fixable=True
                ))

        # Check for unknown sections
        known_sections = self.REQUIRED_GLOBAL_SECTIONS | {
            'environment', 'logging', 'data_sources', 'backtesting'
        }
        for section in config.keys():
            if section not in known_sections:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='structure',
                    message=f'Unknown configuration section: "{section}"',
                    location='root',
                    fix_suggestion=f'Verify "{section}" section is needed or remove if unused',
                    auto_fixable=False
                ))

    def _validate_exchange_configs(self, config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """Validate exchange configurations."""
        exchanges = config.get('exchanges', {})

        if not exchanges:
            result.add_issue(ConfigIssue(
                severity='error',
                category='structure',
                message='No exchange configurations found',
                location='exchanges',
                fix_suggestion='Add at least one exchange configuration',
                auto_fixable=False
            ))
            return

        for exchange_name, exchange_config in exchanges.items():
            location = f'exchanges.{exchange_name}'

            # Validate exchange name
            if exchange_name not in self.VALID_EXCHANGES:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='compatibility',
                    message=f'Unknown exchange: {exchange_name}',
                    location=location,
                    fix_suggestion=f'Verify exchange is supported. Valid exchanges: {", ".join(self.VALID_EXCHANGES)}',
                    auto_fixable=False
                ))

            # Check required fields
            required_fields = {'api_key', 'api_secret', 'sandbox'}
            for field in required_fields:
                if field not in exchange_config:
                    result.add_issue(ConfigIssue(
                        severity='error',
                        category='structure',
                        message=f'Exchange {exchange_name} missing required field: {field}',
                        location=f'{location}.{field}',
                        fix_suggestion=f'Add {field} configuration for {exchange_name}',
                        auto_fixable=True
                    ))

            # Validate environment variables
            api_key = exchange_config.get('api_key', '')
            if isinstance(api_key, str) and api_key.startswith('${') and api_key.endswith('}'):
                env_var = api_key[2:-1]
                if not os.getenv(env_var):
                    result.add_issue(ConfigIssue(
                        severity='warning',
                        category='compatibility',
                        message=f'Environment variable {env_var} not set for {exchange_name}',
                        location=f'{location}.api_key',
                        fix_suggestion=f'Set environment variable {env_var} or update configuration',
                        auto_fixable=False
                    ))

    def _validate_global_position_sizing(self, config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """Validate global position sizing configuration."""
        position_sizing = config.get('position_sizing', {})

        if not position_sizing:
            result.add_issue(ConfigIssue(
                severity='warning',
                category='structure',
                message='No global position sizing configuration found',
                location='position_sizing',
                fix_suggestion='Add global position sizing configuration as fallback for strategies',
                auto_fixable=True
            ))
            return

        # Validate method
        method = position_sizing.get('method') or position_sizing.get('position_sizing_method')
        if not method:
            result.add_issue(ConfigIssue(
                severity='error',
                category='structure',
                message='Position sizing method not specified',
                location='position_sizing.method',
                fix_suggestion='Add position sizing method',
                auto_fixable=True
            ))
        elif method not in self.VALID_POSITION_SIZING_METHODS:
            result.add_issue(ConfigIssue(
                severity='error',
                category='compatibility',
                message=f'Invalid position sizing method: {method}',
                location='position_sizing.method',
                fix_suggestion=f'Use one of: {", ".join(self.VALID_POSITION_SIZING_METHODS)}',
                auto_fixable=True
            ))

        # Validate method-specific parameters
        if method == 'fixed_usd':
            amount = position_sizing.get('fixed_usd_amount')
            if not amount or amount <= 0:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='structure',
                    message='Fixed USD position sizing requires positive amount',
                    location='position_sizing.fixed_usd_amount',
                    fix_suggestion='Add positive fixed_usd_amount value',
                    auto_fixable=True
                ))

        elif method == 'percentage_equity':
            percentage = position_sizing.get('percentage')
            if not percentage or percentage <= 0 or percentage > 100:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='structure',
                    message='Percentage equity sizing requires percentage between 0 and 100',
                    location='position_sizing.percentage',
                    fix_suggestion='Set percentage between 0 and 100',
                    auto_fixable=True
                ))

    def _validate_global_risk_management(self, config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """Validate global risk management configuration."""
        risk_mgmt = config.get('risk_management', {})

        if not risk_mgmt:
            result.add_issue(ConfigIssue(
                severity='suggestion',
                category='best_practice',
                message='No global risk management configuration found',
                location='risk_management',
                fix_suggestion='Consider adding global risk management settings',
                auto_fixable=True
            ))
            return

        # Validate stop loss
        stop_loss = risk_mgmt.get('stop_loss_pct')
        if stop_loss is not None:
            if stop_loss <= 0 or stop_loss > 50:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='best_practice',
                    message=f'Stop loss of {stop_loss}% may be too extreme',
                    location='risk_management.stop_loss_pct',
                    fix_suggestion='Consider stop loss between 1% and 10% for most strategies',
                    auto_fixable=False
                ))

        # Validate take profit
        take_profit = risk_mgmt.get('take_profit_pct')
        if take_profit is not None:
            if take_profit <= 0 or take_profit > 200:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='best_practice',
                    message=f'Take profit of {take_profit}% may be unrealistic',
                    location='risk_management.take_profit_pct',
                    fix_suggestion='Consider take profit between 2% and 20% for most strategies',
                    auto_fixable=False
                ))

        # Check risk/reward ratio
        if stop_loss and take_profit:
            ratio = take_profit / stop_loss
            if ratio < 1.0:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='best_practice',
                    message=f'Risk/reward ratio of {ratio:.2f} is less than 1:1',
                    location='risk_management',
                    fix_suggestion='Consider improving risk/reward ratio to at least 1:1',
                    auto_fixable=False
                ))

    def _validate_strategies_section(self, config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """Validate strategies section structure."""
        strategies = config.get('strategies', {})

        if not strategies:
            result.add_issue(ConfigIssue(
                severity='error',
                category='structure',
                message='No strategies configured',
                location='strategies',
                fix_suggestion='Add at least one strategy configuration',
                auto_fixable=False
            ))
            return

        # Check for enabled strategies
        enabled_strategies = [
            name for name, config in strategies.items()
            if config.get('enabled', False)
        ]

        if not enabled_strategies:
            result.add_issue(ConfigIssue(
                severity='warning',
                category='best_practice',
                message='No strategies are enabled',
                location='strategies',
                fix_suggestion='Enable at least one strategy for trading',
                auto_fixable=True
            ))

        # Check strategy name uniqueness
        strategy_names = list(strategies.keys())
        if len(strategy_names) != len(set(strategy_names)):
            result.add_issue(ConfigIssue(
                severity='error',
                category='structure',
                message='Duplicate strategy names found',
                location='strategies',
                fix_suggestion='Ensure all strategy names are unique',
                auto_fixable=False
            ))

    def _validate_individual_strategies_with_strategy_validator(self, config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """Validate individual strategy configurations using StrategyValidator."""
        strategies = config.get('strategies', {})

        for strategy_name, strategy_config in strategies.items():
            location = f'strategies.{strategy_name}'

            # Check required fields
            for field in self.REQUIRED_STRATEGY_FIELDS:
                if field not in strategy_config:
                    result.add_issue(ConfigIssue(
                        severity='error',
                        category='structure',
                        message=f'Strategy {strategy_name} missing required field: {field}',
                        location=f'{location}.{field}',
                        fix_suggestion=f'Add {field} to strategy configuration',
                        auto_fixable=True
                    ))

            # Validate timeframe
            timeframe = strategy_config.get('timeframe')
            if timeframe and timeframe not in self.VALID_TIMEFRAMES:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='compatibility',
                    message=f'Invalid timeframe for strategy {strategy_name}: {timeframe}',
                    location=f'{location}.timeframe',
                    fix_suggestion=f'Use one of: {", ".join(self.VALID_TIMEFRAMES)}',
                    auto_fixable=True
                ))

            # Validate exchange
            exchange = strategy_config.get('exchange')
            if exchange and exchange not in self.VALID_EXCHANGES:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='compatibility',
                    message=f'Unknown exchange for strategy {strategy_name}: {exchange}',
                    location=f'{location}.exchange',
                    fix_suggestion=f'Verify exchange is supported. Valid: {", ".join(self.VALID_EXCHANGES)}',
                    auto_fixable=False
                ))

            # Validate indicators using StrategyValidator integration
            indicators = strategy_config.get('indicators', {})
            if not indicators:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='structure',
                    message=f'Strategy {strategy_name} has no indicators',
                    location=f'{location}.indicators',
                    fix_suggestion='Add at least one indicator to the strategy',
                    auto_fixable=False
                ))
            else:
                # Use StrategyValidator for detailed indicator validation
                try:
                    # Create StrategyConfig object for validation
                    strategy_config_obj = StrategyConfig(
                        name=strategy_name,
                        **strategy_config
                    )
                    strategy_result = self.strategy_validator.validate_strategy_config(strategy_config_obj)

                    # Convert validation results to ConfigIssues
                    for error in strategy_result.errors:
                        result.add_issue(ConfigIssue(
                            severity='error',
                            category='structure',
                            message=f'Strategy {strategy_name}: {error}',
                            location=location,
                            fix_suggestion='Fix strategy configuration error',
                            auto_fixable=False
                        ))

                    for warning in strategy_result.warnings:
                        result.add_issue(ConfigIssue(
                            severity='warning',
                            category='best_practice',
                            message=f'Strategy {strategy_name}: {warning}',
                            location=location,
                            fix_suggestion='Consider optimization',
                            auto_fixable=False
                        ))

                except Exception as e:
                    logger.warning(f"Could not validate strategy {strategy_name} with StrategyValidator: {e}")
                    # Fall back to basic validation
                    self._validate_strategy_indicators(strategy_name, indicators, location, result)

    def _validate_strategy_indicators(
        self,
        strategy_name: str,
        indicators: Dict[str, Any],
        location: str,
        result: ConfigValidationResult
    ) -> None:
        """Validate indicators within a strategy."""
        for indicator_name, indicator_config in indicators.items():
            indicator_location = f'{location}.indicators.{indicator_name}'

            # Check indicator type
            if 'type' not in indicator_config:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='structure',
                    message=f'Indicator {indicator_name} in strategy {strategy_name} missing type',
                    location=f'{indicator_location}.type',
                    fix_suggestion='Add indicator type specification',
                    auto_fixable=True
                ))

            # Check for indicator naming consistency
            indicator_type = indicator_config.get('type', '')
            if indicator_type and not indicator_name.lower().startswith(indicator_type.lower()):
                result.add_issue(ConfigIssue(
                    severity='suggestion',
                    category='best_practice',
                    message=f'Indicator name "{indicator_name}" doesn\'t match type "{indicator_type}"',
                    location=indicator_location,
                    fix_suggestion=f'Consider renaming to "{indicator_type}_1" or similar',
                    auto_fixable=True
                ))

    def _validate_component_compatibility(self, config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """Validate compatibility between different configuration components."""
        strategies = config.get('strategies', {})
        exchanges = config.get('exchanges', {})

        # Check strategy-exchange compatibility
        for strategy_name, strategy_config in strategies.items():
            exchange = strategy_config.get('exchange')
            if exchange and exchange not in exchanges:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='compatibility',
                    message=f'Strategy {strategy_name} references unconfigured exchange: {exchange}',
                    location=f'strategies.{strategy_name}.exchange',
                    fix_suggestion=f'Add configuration for exchange {exchange} or change strategy exchange',
                    auto_fixable=False
                ))

    def _analyze_performance_characteristics(self, config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """Analyze configuration for performance characteristics."""
        strategies = config.get('strategies', {})

        # Analyze exchange distribution
        exchange_counts = {}
        for strategy_config in strategies.values():
            exchange = strategy_config.get('exchange')
            if exchange:
                exchange_counts[exchange] = exchange_counts.get(exchange, 0) + 1

        # Check for exchange concentration
        for exchange, count in exchange_counts.items():
            if count > self.PERFORMANCE_THRESHOLDS['max_strategies_per_exchange']:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='performance',
                    message=f'High concentration of strategies on {exchange}: {count} strategies',
                    location='strategies',
                    fix_suggestion='Consider distributing strategies across multiple exchanges',
                    auto_fixable=False
                ))

        # Analyze indicator complexity
        for strategy_name, strategy_config in strategies.items():
            indicators = strategy_config.get('indicators', {})
            indicator_count = len(indicators)

            if indicator_count > self.PERFORMANCE_THRESHOLDS['max_indicators_per_strategy']:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='performance',
                    message=f'Strategy {strategy_name} has many indicators: {indicator_count}',
                    location=f'strategies.{strategy_name}.indicators',
                    fix_suggestion='Consider reducing indicator count to improve performance and reduce overfitting',
                    auto_fixable=False
                ))

    def _generate_optimization_suggestions(self, config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """Generate optimization suggestions based on configuration analysis."""
        strategies = config.get('strategies', {})

        # Suggest timeframe diversification
        timeframes = [s.get('timeframe') for s in strategies.values() if s.get('timeframe')]
        unique_timeframes = set(timeframes)

        if len(unique_timeframes) == 1:
            result.optimization_suggestions.append(
                "Consider adding strategies with different timeframes for diversification"
            )

        # Suggest exchange diversification
        exchanges = [s.get('exchange') for s in strategies.values() if s.get('exchange')]
        unique_exchanges = set(exchanges)

        if len(unique_exchanges) == 1:
            result.optimization_suggestions.append(
                "Consider adding strategies on different exchanges for risk distribution"
            )

        # Suggest indicator type diversification
        for strategy_name, strategy_config in strategies.items():
            indicators = strategy_config.get('indicators', {})
            indicator_types = [i.get('type') for i in indicators.values() if i.get('type')]
            unique_types = set(indicator_types)

            if len(unique_types) < 2 and len(indicators) > 1:
                result.optimization_suggestions.append(
                    f"Strategy {strategy_name}: Consider using different types of indicators for better signal confirmation"
                )

    def _generate_repair_suggestions(self, result: ConfigValidationResult) -> None:
        """Generate repair suggestions for auto-fixable issues."""
        auto_fixable_issues = [issue for issue in result.issues if issue.auto_fixable]

        if auto_fixable_issues:
            result.repair_suggestions.append(
                f"Found {len(auto_fixable_issues)} auto-fixable issues that can be resolved automatically"
            )

            # Group by category
            by_category = {}
            for issue in auto_fixable_issues:
                category = issue.category
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(issue)

            for category, issues in by_category.items():
                result.repair_suggestions.append(
                    f"  {category.title()}: {len(issues)} issues can be auto-fixed"
                )

    def _calculate_performance_score(self, result: ConfigValidationResult) -> float:
        """Calculate a performance score based on validation results."""
        base_score = 100.0

        # Deduct points for issues
        for issue in result.issues:
            if issue.severity == 'error':
                base_score -= 20.0
            elif issue.severity == 'warning':
                base_score -= 10.0
            elif issue.severity == 'suggestion':
                base_score -= 2.0

        # Bonus for optimization opportunities
        if result.optimization_suggestions:
            base_score += min(5.0, len(result.optimization_suggestions))

        return max(0.0, min(100.0, base_score))

    def validate_config_compatibility(self, config1_path: str, config2_path: str) -> ConfigValidationResult:
        """
        Check compatibility between two configuration files.

        Args:
            config1_path: Path to first config file
            config2_path: Path to second config file

        Returns:
            ConfigValidationResult with compatibility analysis
        """
        result = ConfigValidationResult(
            is_valid=True,
            issues=[],
            performance_score=0.0,
            optimization_suggestions=[],
            repair_suggestions=[]
        )

        try:
            config1 = self.config_manager.load_config_from_path(config1_path)
            config2 = self.config_manager.load_config_from_path(config2_path)

            if not config1 or not config2:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='compatibility',
                    message='One or both configuration files could not be loaded',
                    location='comparison',
                    fix_suggestion='Ensure both configuration files exist and are valid',
                    auto_fixable=False
                ))
                return result

            # Compare structure
            self._compare_config_structures(config1, config2, result)

            # Compare exchange configurations
            self._compare_exchange_configs(config1, config2, result)

            # Compare strategy configurations
            self._compare_strategy_configs(config1, config2, result)

            logger.debug("Configuration compatibility validation completed")

        except Exception as e:
            result.add_issue(ConfigIssue(
                severity='error',
                category='compatibility',
                message=f'Error during compatibility check: {e}',
                location='comparison',
                fix_suggestion='Check configuration file formats and paths',
                auto_fixable=False
            ))

        return result

    def _compare_config_structures(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        result: ConfigValidationResult
    ) -> None:
        """Compare the structures of two configurations."""
        sections1 = set(config1.keys())
        sections2 = set(config2.keys())

        # Check for missing sections
        missing_in_2 = sections1 - sections2
        missing_in_1 = sections2 - sections1

        for section in missing_in_2:
            result.add_issue(ConfigIssue(
                severity='warning',
                category='compatibility',
                message=f'Section "{section}" present in first config but missing in second',
                location='structure_comparison',
                fix_suggestion=f'Add "{section}" section to second configuration',
                auto_fixable=True
            ))

        for section in missing_in_1:
            result.add_issue(ConfigIssue(
                severity='warning',
                category='compatibility',
                message=f'Section "{section}" present in second config but missing in first',
                location='structure_comparison',
                fix_suggestion=f'Add "{section}" section to first configuration',
                auto_fixable=True
            ))

    def _compare_exchange_configs(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        result: ConfigValidationResult
    ) -> None:
        """Compare exchange configurations between two configs."""
        exchanges1 = set(config1.get('exchanges', {}).keys())
        exchanges2 = set(config2.get('exchanges', {}).keys())

        if exchanges1 != exchanges2:
            result.add_issue(ConfigIssue(
                severity='warning',
                category='compatibility',
                message=f'Different exchanges configured: {exchanges1} vs {exchanges2}',
                location='exchanges',
                fix_suggestion='Align exchange configurations for consistency',
                auto_fixable=False
            ))

    def _compare_strategy_configs(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        result: ConfigValidationResult
    ) -> None:
        """Compare strategy configurations between two configs."""
        strategies1 = set(config1.get('strategies', {}).keys())
        strategies2 = set(config2.get('strategies', {}).keys())

        common_strategies = strategies1 & strategies2

        if common_strategies:
            result.add_issue(ConfigIssue(
                severity='suggestion',
                category='compatibility',
                message=f'Found {len(common_strategies)} common strategies that could be compared in detail',
                location='strategies',
                fix_suggestion='Consider detailed strategy parameter comparison',
                auto_fixable=False
            ))

        unique_to_1 = strategies1 - strategies2
        unique_to_2 = strategies2 - strategies1

        if unique_to_1 or unique_to_2:
            result.add_issue(ConfigIssue(
                severity='suggestion',
                category='compatibility',
                message=f'Unique strategies - Config1: {len(unique_to_1)}, Config2: {len(unique_to_2)}',
                location='strategies',
                fix_suggestion='Consider strategy portfolio alignment between configurations',
                auto_fixable=False
            ))

    def _validate_strategy_indicator_consistency_all(self, config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """Validate strategy-indicator consistency for all strategies."""
        strategies = config.get('strategies', {})

        for strategy_name in strategies.keys():
            try:
                consistency_result = self.validate_strategy_indicator_consistency(strategy_name)
                # Merge results
                for issue in consistency_result.issues:
                    result.add_issue(issue)
            except Exception as e:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='compatibility',
                    message=f'Could not validate strategy-indicator consistency for {strategy_name}: {e}',
                    location=f'strategies.{strategy_name}',
                    fix_suggestion='Check strategy configuration',
                    auto_fixable=False
                ))

    def _validate_timeframe_consistency_all(self, config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """Validate timeframe consistency for all strategies."""
        strategies = config.get('strategies', {})

        for strategy_name in strategies.keys():
            try:
                timeframe_result = self.validate_timeframe_consistency(strategy_name)
                # Merge results
                for issue in timeframe_result.issues:
                    result.add_issue(issue)
            except Exception as e:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='compatibility',
                    message=f'Could not validate timeframe consistency for {strategy_name}: {e}',
                    location=f'strategies.{strategy_name}',
                    fix_suggestion='Check strategy timeframe configuration',
                    auto_fixable=False
                ))

    def _validate_market_exchange_compatibility_all(self, config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """Validate market and exchange compatibility for all strategies."""
        strategies = config.get('strategies', {})

        for strategy_name in strategies.keys():
            try:
                compatibility_result = self.validate_market_exchange_compatibility(strategy_name)
                # Merge results
                for issue in compatibility_result.issues:
                    result.add_issue(issue)
            except Exception as e:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='compatibility',
                    message=f'Could not validate market-exchange compatibility for {strategy_name}: {e}',
                    location=f'strategies.{strategy_name}',
                    fix_suggestion='Check strategy market and exchange configuration',
                    auto_fixable=False
                ))

    def validate_strategy_indicator_consistency(self, strategy_name: str) -> ConfigValidationResult:
        """
        Validate strategy-indicator consistency for a specific strategy.

        Args:
            strategy_name: Name of strategy to validate

        Returns:
            ConfigValidationResult with consistency validation results
        """
        result = ConfigValidationResult(
            is_valid=True,
            issues=[],
            performance_score=0.0,
            optimization_suggestions=[],
            repair_suggestions=[]
        )

        try:
            # Use StrategyValidator for detailed strategy-indicator consistency checking
            strategy_result = self.strategy_validator.validate_strategy_indicator_consistency(strategy_name)

            # Convert StrategyValidator results to ConfigValidationResult format
            for error in strategy_result.errors:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='compatibility',
                    message=error,
                    location=f'strategies.{strategy_name}.indicators',
                    fix_suggestion='Review strategy-indicator compatibility',
                    auto_fixable=False
                ))

            for warning in strategy_result.warnings:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='compatibility',
                    message=warning,
                    location=f'strategies.{strategy_name}.indicators',
                    fix_suggestion='Consider optimizing strategy-indicator configuration',
                    auto_fixable=False
                ))

            for suggestion in strategy_result.suggestions:
                result.add_issue(ConfigIssue(
                    severity='suggestion',
                    category='best_practice',
                    message=suggestion,
                    location=f'strategies.{strategy_name}.indicators',
                    fix_suggestion='Consider applying this optimization',
                    auto_fixable=False
                ))

            # Additional consistency checks specific to configuration
            self._check_indicator_availability(strategy_name, result)
            self._check_indicator_parameter_consistency(strategy_name, result)

        except Exception as e:
            result.add_issue(ConfigIssue(
                severity='error',
                category='compatibility',
                message=f'Error validating strategy-indicator consistency: {e}',
                location=f'strategies.{strategy_name}',
                fix_suggestion='Check strategy configuration and indicators',
                auto_fixable=False
            ))
            logger.error(f"Strategy-indicator consistency validation failed for {strategy_name}: {e}")

        return result

    def validate_position_sizing_config(self, position_sizing_config: Dict[str, Any]) -> ConfigValidationResult:
        """
        Validate position sizing configuration.

        Args:
            position_sizing_config: Position sizing configuration to validate

        Returns:
            ConfigValidationResult with position sizing validation results
        """
        result = ConfigValidationResult(
            is_valid=True,
            issues=[],
            performance_score=0.0,
            optimization_suggestions=[],
            repair_suggestions=[]
        )

        try:
            # Use StrategyValidator for detailed position sizing validation
            strategy_result = self.strategy_validator.validate_position_sizing_config(position_sizing_config)

            # Convert results
            for error in strategy_result.errors:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='structure',
                    message=error,
                    location='position_sizing',
                    fix_suggestion='Fix position sizing configuration',
                    auto_fixable=True
                ))

            for warning in strategy_result.warnings:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='best_practice',
                    message=warning,
                    location='position_sizing',
                    fix_suggestion='Consider optimizing position sizing parameters',
                    auto_fixable=False
                ))

            # Additional position sizing checks
            self._validate_position_sizing_parameters(position_sizing_config, result)
            self._check_position_sizing_risk_compatibility(position_sizing_config, result)

        except Exception as e:
            result.add_issue(ConfigIssue(
                severity='error',
                category='structure',
                message=f'Error validating position sizing config: {e}',
                location='position_sizing',
                fix_suggestion='Check position sizing configuration format',
                auto_fixable=False
            ))
            logger.error(f"Position sizing validation failed: {e}")

        return result

    def validate_market_exchange_compatibility(self, strategy_name: str) -> ConfigValidationResult:
        """
        Validate market and exchange compatibility for a strategy.

        Args:
            strategy_name: Name of strategy to validate

        Returns:
            ConfigValidationResult with market-exchange compatibility results
        """
        result = ConfigValidationResult(
            is_valid=True,
            issues=[],
            performance_score=0.0,
            optimization_suggestions=[],
            repair_suggestions=[]
        )

        try:
            # Use StrategyValidator for market-exchange compatibility
            strategy_result = self.strategy_validator.validate_market_exchange_compatibility(strategy_name)

            # Convert results
            for error in strategy_result.errors:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='compatibility',
                    message=error,
                    location=f'strategies.{strategy_name}',
                    fix_suggestion='Fix market-exchange compatibility issues',
                    auto_fixable=False
                ))

            for warning in strategy_result.warnings:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='compatibility',
                    message=warning,
                    location=f'strategies.{strategy_name}',
                    fix_suggestion='Consider market-exchange optimization',
                    auto_fixable=False
                ))

            # Additional market-exchange checks
            self._check_market_symbol_format(strategy_name, result)
            self._check_exchange_market_support(strategy_name, result)
            self._check_liquidity_requirements(strategy_name, result)

        except Exception as e:
            result.add_issue(ConfigIssue(
                severity='error',
                category='compatibility',
                message=f'Error validating market-exchange compatibility: {e}',
                location=f'strategies.{strategy_name}',
                fix_suggestion='Check strategy market and exchange configuration',
                auto_fixable=False
            ))
            logger.error(f"Market-exchange compatibility validation failed for {strategy_name}: {e}")

        return result

    def validate_timeframe_consistency(self, strategy_name: str) -> ConfigValidationResult:
        """
        Validate timeframe consistency for a strategy.

        Args:
            strategy_name: Name of strategy to validate

        Returns:
            ConfigValidationResult with timeframe consistency results
        """
        result = ConfigValidationResult(
            is_valid=True,
            issues=[],
            performance_score=0.0,
            optimization_suggestions=[],
            repair_suggestions=[]
        )

        try:
            # Use StrategyValidator for timeframe consistency
            strategy_result = self.strategy_validator.validate_timeframe_consistency(strategy_name)

            # Convert results
            for error in strategy_result.errors:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='compatibility',
                    message=error,
                    location=f'strategies.{strategy_name}.timeframe',
                    fix_suggestion='Fix timeframe consistency issues',
                    auto_fixable=True
                ))

            for warning in strategy_result.warnings:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='performance',
                    message=warning,
                    location=f'strategies.{strategy_name}.timeframe',
                    fix_suggestion='Consider timeframe optimization',
                    auto_fixable=False
                ))

            # Additional timeframe consistency checks
            self._check_strategy_indicator_timeframe_harmony(strategy_name, result)
            self._check_timeframe_data_availability(strategy_name, result)

        except Exception as e:
            result.add_issue(ConfigIssue(
                severity='error',
                category='compatibility',
                message=f'Error validating timeframe consistency: {e}',
                location=f'strategies.{strategy_name}.timeframe',
                fix_suggestion='Check strategy timeframe configuration',
                auto_fixable=False
            ))
            logger.error(f"Timeframe consistency validation failed for {strategy_name}: {e}")

        return result

    def _check_indicator_parameter_consistency(self, strategy_name: str, result: ConfigValidationResult) -> None:
        """Check indicator parameter consistency within strategy."""
        try:
            config = self.config_manager.load_config()
            strategy_config = config.get('strategies', {}).get(strategy_name, {})
            indicators = strategy_config.get('indicators', {})

            # Check for conflicting timeframes in indicators
            indicator_timeframes = {}
            for indicator_name, indicator_config in indicators.items():
                timeframe = indicator_config.get('timeframe')
                if timeframe:
                    indicator_timeframes[indicator_name] = timeframe

            if len(set(indicator_timeframes.values())) > 1:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='performance',
                    message=f'Strategy {strategy_name} has indicators with different timeframes: {indicator_timeframes}',
                    location=f'strategies.{strategy_name}.indicators',
                    fix_suggestion='Consider aligning indicator timeframes for better performance',
                    auto_fixable=False
                ))

        except Exception as e:
            logger.warning(f"Could not check indicator parameter consistency for {strategy_name}: {e}")

    def _validate_position_sizing_parameters(self, position_sizing_config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """Validate position sizing parameters."""
        method = position_sizing_config.get('method') or position_sizing_config.get('position_sizing_method')

        if method == 'fixed_usd':
            amount = position_sizing_config.get('fixed_usd_amount') or position_sizing_config.get('amount')
            if not amount or amount <= 0:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='structure',
                    message='Fixed USD amount must be positive',
                    location='position_sizing.fixed_usd_amount',
                    fix_suggestion='Set positive amount value',
                    auto_fixable=True
                ))

        elif method == 'percentage_equity':
            percentage = position_sizing_config.get('percentage')
            if not percentage or percentage <= 0 or percentage > 100:
                result.add_issue(ConfigIssue(
                    severity='error',
                    category='structure',
                    message='Percentage must be between 0 and 100',
                    location='position_sizing.percentage',
                    fix_suggestion='Set percentage between 0 and 100',
                    auto_fixable=True
                ))

    def _check_position_sizing_risk_compatibility(self, position_sizing_config: Dict[str, Any], result: ConfigValidationResult) -> None:
        """Check position sizing compatibility with risk management."""
        try:
            config = self.config_manager.load_config()
            risk_mgmt = config.get('risk_management', {})

            # Check if position sizing is compatible with risk limits
            method = position_sizing_config.get('method')
            if method == 'fixed_usd':
                amount = position_sizing_config.get('fixed_usd_amount', 0)
                max_position = risk_mgmt.get('max_position_usd')
                if max_position and amount > max_position:
                    result.add_issue(ConfigIssue(
                        severity='warning',
                        category='compatibility',
                        message=f'Fixed position size ({amount}) exceeds max position limit ({max_position})',
                        location='position_sizing',
                        fix_suggestion='Reduce position size or increase risk limit',
                        auto_fixable=False
                    ))

        except Exception as e:
            logger.warning(f"Could not check position sizing risk compatibility: {e}")

    def _check_market_symbol_format(self, strategy_name: str, result: ConfigValidationResult) -> None:
        """Check market symbol format validity."""
        try:
            config = self.config_manager.load_config()
            strategy_config = config.get('strategies', {}).get(strategy_name, {})
            market = strategy_config.get('market', '')

            # Basic symbol format validation
            if market and '-' not in market:
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='compatibility',
                    message=f'Market symbol {market} may not be in standard format (expected: BASE-QUOTE)',
                    location=f'strategies.{strategy_name}.market',
                    fix_suggestion='Use format like BTC-USD, ETH-USDC',
                    auto_fixable=False
                ))

        except Exception as e:
            logger.warning(f"Could not check market symbol format for {strategy_name}: {e}")

    def _check_exchange_market_support(self, strategy_name: str, result: ConfigValidationResult) -> None:
        """Check if exchange supports the specified market."""
        try:
            config = self.config_manager.load_config()
            strategy_config = config.get('strategies', {}).get(strategy_name, {})
            exchange = strategy_config.get('exchange')
            market = strategy_config.get('market')

            # Basic exchange-market compatibility checks
            if exchange == 'hyperliquid' and market and not market.endswith('-USD'):
                result.add_issue(ConfigIssue(
                    severity='warning',
                    category='compatibility',
                    message=f'Hyperliquid typically supports USD pairs, but strategy uses {market}',
                    location=f'strategies.{strategy_name}',
                    fix_suggestion='Verify market is supported on Hyperliquid',
                    auto_fixable=False
                ))

        except Exception as e:
            logger.warning(f"Could not check exchange market support for {strategy_name}: {e}")

    def _check_liquidity_requirements(self, strategy_name: str, result: ConfigValidationResult) -> None:
        """Check liquidity requirements for strategy."""
        try:
            config = self.config_manager.load_config()
            strategy_config = config.get('strategies', {}).get(strategy_name, {})
            market = strategy_config.get('market', '')

            # Check if market might have liquidity concerns
            minor_pairs = ['LTC-USD', 'ADA-USD', 'DOGE-USD', 'XRP-USD']
            if market in minor_pairs:
                result.add_issue(ConfigIssue(
                    severity='suggestion',
                    category='performance',
                    message=f'Market {market} may have lower liquidity, consider monitoring spreads',
                    location=f'strategies.{strategy_name}.market',
                    fix_suggestion='Monitor bid-ask spreads and consider major pairs for better liquidity',
                    auto_fixable=False
                ))

        except Exception as e:
            logger.warning(f"Could not check liquidity requirements for {strategy_name}: {e}")

    def _check_strategy_indicator_timeframe_harmony(self, strategy_name: str, result: ConfigValidationResult) -> None:
        """Check timeframe harmony between strategy and indicators."""
        try:
            config = self.config_manager.load_config()
            strategy_config = config.get('strategies', {}).get(strategy_name, {})
            strategy_timeframe = strategy_config.get('timeframe')
            indicators = strategy_config.get('indicators', {})

            if not strategy_timeframe:
                return

            strategy_tf_index = self.VALID_TIMEFRAMES.index(strategy_timeframe)

            for indicator_name, indicator_config in indicators.items():
                indicator_timeframe = indicator_config.get('timeframe', strategy_timeframe)
                if indicator_timeframe in self.VALID_TIMEFRAMES:
                    indicator_tf_index = self.VALID_TIMEFRAMES.index(indicator_timeframe)

                    # Check if indicator timeframe is much larger than strategy timeframe
                    if indicator_tf_index - strategy_tf_index > 2:
                        result.add_issue(ConfigIssue(
                            severity='warning',
                            category='performance',
                            message=f'Indicator {indicator_name} timeframe ({indicator_timeframe}) much larger than strategy timeframe ({strategy_timeframe})',
                            location=f'strategies.{strategy_name}.indicators.{indicator_name}',
                            fix_suggestion='Consider aligning timeframes for better signal quality',
                            auto_fixable=False
                        ))

        except Exception as e:
            logger.warning(f"Could not check timeframe harmony for {strategy_name}: {e}")

    def _check_timeframe_data_availability(self, strategy_name: str, result: ConfigValidationResult) -> None:
        """Check if timeframe has good data availability."""
        try:
            config = self.config_manager.load_config()
            strategy_config = config.get('strategies', {}).get(strategy_name, {})
            timeframe = strategy_config.get('timeframe')

            # Check for very high frequency timeframes that might have data gaps
            if timeframe in ['1m', '5m']:
                result.add_issue(ConfigIssue(
                    severity='suggestion',
                    category='performance',
                    message=f'Very high frequency timeframe ({timeframe}) may have data gaps or noise',
                    location=f'strategies.{strategy_name}.timeframe',
                    fix_suggestion='Consider using 15m or higher timeframes for more reliable signals',
                    auto_fixable=False
                ))

        except Exception as e:
            logger.warning(f"Could not check timeframe data availability for {strategy_name}: {e}")
