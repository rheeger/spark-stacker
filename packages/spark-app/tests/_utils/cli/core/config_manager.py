"""
Configuration Manager

This module provides centralized configuration management for the CLI with:
- Configuration loading with intelligent fallback logic
- Caching and reload functionality for performance
- Environment variable expansion support
- Configuration merging (global + strategy overrides)
- Configuration versioning and migration support
- Comprehensive validation and repair utilities
"""

import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from app.utils.config import ConfigManager as AppConfigManager

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigManager:
    """
    Centralized configuration manager for CLI operations.

    Features:
    - Intelligent configuration loading with fallback paths
    - Configuration caching and reload functionality
    - Environment variable expansion
    - Configuration merging and inheritance
    - Version compatibility and migration support
    - Comprehensive validation and repair suggestions
    """

    # Configuration file version for migration support
    CURRENT_CONFIG_VERSION = "1.0.0"
    SUPPORTED_VERSIONS = ["1.0.0", "0.9.0"]

    def __init__(self, config_path: Optional[str] = None,
                 enable_caching: bool = True,
                 cache_ttl_seconds: int = 300):
        """
        Initialize the configuration manager.

        Args:
            config_path: Optional explicit path to config file
            enable_caching: Whether to enable configuration caching
            cache_ttl_seconds: Time-to-live for cached configuration (default 5 minutes)
        """
        self.config_path = config_path
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds

        # Internal state
        self._cached_config = None
        self._cache_timestamp = None
        self._resolved_config_path = None
        self._config_lock = threading.Lock()

        # Environment variable pattern for expansion
        self._env_pattern = re.compile(r'\$\{([^}]+)\}')

        logger.debug(f"ConfigManager initialized with caching={'enabled' if enable_caching else 'disabled'}")

    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load configuration with caching and fallback logic.

        Args:
            force_reload: Whether to force reload even if cached version is available

        Returns:
            Dictionary containing the loaded and processed configuration

        Raises:
            ConfigurationError: If configuration cannot be loaded
        """
        with self._config_lock:
            # Check if we can use cached configuration
            if self._can_use_cached_config() and not force_reload:
                logger.debug("Using cached configuration")
                return self._cached_config.copy()

            # Resolve configuration file path
            config_file_path = self._resolve_config_path()

            try:
                # Load raw configuration
                raw_config = self._load_raw_config(config_file_path)

                # Process configuration (expand env vars, validate version, etc.)
                processed_config = self._process_config(raw_config, config_file_path)

                # Cache the processed configuration
                if self.enable_caching:
                    self._cached_config = processed_config.copy()
                    self._cache_timestamp = time.time()
                    self._resolved_config_path = config_file_path

                logger.info(f"Successfully loaded configuration from {config_file_path}")
                logger.debug(f"Loaded {len(processed_config.get('strategies', []))} strategies "
                           f"and {len(processed_config.get('indicators', []))} indicators")

                return processed_config

            except Exception as e:
                logger.error(f"Failed to load configuration from {config_file_path}: {e}")
                raise ConfigurationError(f"Configuration loading failed: {e}") from e

    def reload_config(self) -> Dict[str, Any]:
        """
        Force reload configuration from disk, bypassing cache.

        Returns:
            Dictionary containing the reloaded configuration
        """
        logger.info("Forcing configuration reload")
        return self.load_config(force_reload=True)

    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Comprehensive configuration validation.

        Args:
            config: Optional configuration to validate. If None, loads current config.

        Returns:
            List of validation error messages (empty if valid)
        """
        if config is None:
            config = self.load_config()

        errors = []

        # Check configuration version compatibility
        version_errors = self._validate_version(config)
        errors.extend(version_errors)

        # Check required top-level sections
        section_errors = self._validate_sections(config)
        errors.extend(section_errors)

        # Validate strategies
        strategy_errors = self._validate_strategies(config)
        errors.extend(strategy_errors)

        # Validate indicators
        indicator_errors = self._validate_indicators(config)
        errors.extend(indicator_errors)

        # Validate exchanges
        exchange_errors = self._validate_exchanges(config)
        errors.extend(exchange_errors)

        # Validate cross-references (strategy-indicator relationships)
        reference_errors = self._validate_cross_references(config)
        errors.extend(reference_errors)

        if errors:
            logger.warning(f"Configuration validation found {len(errors)} issues")
        else:
            logger.info("Configuration validation passed")

        return errors

    def get_strategy_config(self, strategy_name: str,
                           config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific strategy with inheritance and merging.

        Args:
            strategy_name: Name of the strategy to retrieve
            config: Optional configuration dict. If None, loads current config.

        Returns:
            Strategy configuration with merged global settings, or None if not found
        """
        if config is None:
            config = self.load_config()

        # Find the strategy
        strategies = config.get('strategies', [])
        strategy = None

        for strat in strategies:
            if strat.get('name') == strategy_name:
                strategy = strat.copy()
                break

        if not strategy:
            return None

        # Apply configuration inheritance and merging
        merged_strategy = self._merge_strategy_config(strategy, config)

        return merged_strategy

    def list_strategies(self, config: Optional[Dict[str, Any]] = None,
                       filter_exchange: Optional[str] = None,
                       filter_market: Optional[str] = None,
                       filter_enabled: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        List strategies with optional filtering and enhanced metadata.

        Args:
            config: Optional configuration dict. If None, loads current config.
            filter_exchange: Optional exchange name filter
            filter_market: Optional market filter (e.g., "ETH-USD")
            filter_enabled: Optional enabled status filter

        Returns:
            List of strategy dictionaries with merged configurations
        """
        if config is None:
            config = self.load_config()

        strategies = config.get('strategies', [])
        filtered_strategies = []

        for strategy in strategies:
            # Apply filters
            if filter_enabled is not None and strategy.get('enabled', True) != filter_enabled:
                continue

            if filter_exchange and strategy.get('exchange', '').lower() != filter_exchange.lower():
                continue

            if filter_market and strategy.get('market', '').upper() != filter_market.upper():
                continue

            # Apply configuration merging for complete strategy context
            merged_strategy = self._merge_strategy_config(strategy.copy(), config)
            filtered_strategies.append(merged_strategy)

        return filtered_strategies

    def get_repair_suggestions(self, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate configuration repair suggestions for common issues.

        Args:
            config: Optional configuration to analyze. If None, loads current config.

        Returns:
            List of repair suggestion dictionaries with issue, severity, and fix information
        """
        if config is None:
            config = self.load_config()

        suggestions = []

        # Check for missing strategies
        if not config.get('strategies'):
            suggestions.append({
                'issue': 'No strategies defined',
                'severity': 'high',
                'fix': 'Add at least one strategy configuration to the strategies section',
                'example': {
                    'name': 'example_strategy',
                    'market': 'ETH-USD',
                    'exchange': 'hyperliquid',
                    'timeframe': '1h',
                    'indicators': ['rsi', 'macd'],
                    'enabled': True
                }
            })

        # Check for disabled exchanges
        exchanges = config.get('exchanges', [])
        enabled_exchanges = [ex for ex in exchanges if ex.get('enabled', False)]
        if not enabled_exchanges:
            suggestions.append({
                'issue': 'No exchanges are enabled',
                'severity': 'high',
                'fix': 'Enable at least one exchange in the exchanges section',
                'example': {'name': 'hyperliquid', 'enabled': True}
            })

        # Check for strategies without indicators
        strategies = config.get('strategies', [])
        for strategy in strategies:
            if not strategy.get('indicators'):
                suggestions.append({
                    'issue': f"Strategy '{strategy.get('name')}' has no indicators",
                    'severity': 'medium',
                    'fix': f"Add indicators to strategy '{strategy.get('name')}'",
                    'example': {'indicators': ['rsi', 'macd']}
                })

        # Check for orphaned indicators
        used_indicators = set()
        for strategy in strategies:
            used_indicators.update(strategy.get('indicators', []))

        all_indicators = {ind.get('name') for ind in config.get('indicators', [])}
        orphaned = all_indicators - used_indicators

        for indicator_name in orphaned:
            suggestions.append({
                'issue': f"Indicator '{indicator_name}' is not used by any strategy",
                'severity': 'low',
                'fix': f"Either remove unused indicator '{indicator_name}' or add it to a strategy",
                'example': None
            })

        return suggestions

    def migrate_config(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Migrate configuration to current version if needed.

        Args:
            config: Configuration dictionary to migrate

        Returns:
            Tuple of (migrated_config, migration_notes)
        """
        current_version = config.get('version', '0.9.0')  # Default to old version
        migration_notes = []

        if current_version == self.CURRENT_CONFIG_VERSION:
            logger.debug("Configuration is already at current version")
            return config.copy(), migration_notes

        if current_version not in self.SUPPORTED_VERSIONS:
            raise ConfigurationError(f"Unsupported configuration version: {current_version}")

        migrated_config = config.copy()

        # Migrate from 0.9.0 to 1.0.0
        if current_version == '0.9.0':
            migrated_config = self._migrate_from_0_9_0(migrated_config)
            migration_notes.append("Migrated from version 0.9.0 to 1.0.0")
            migration_notes.append("Added position sizing defaults")
            migration_notes.append("Normalized timeframe formats")

        # Update version
        migrated_config['version'] = self.CURRENT_CONFIG_VERSION
        migration_notes.append(f"Updated configuration version to {self.CURRENT_CONFIG_VERSION}")

        logger.info(f"Configuration migrated from {current_version} to {self.CURRENT_CONFIG_VERSION}")
        return migrated_config, migration_notes

    def export_config(self, output_path: str, config: Optional[Dict[str, Any]] = None,
                     pretty_print: bool = True) -> None:
        """
        Export configuration to a file with optional formatting.

        Args:
            output_path: Path where to save the configuration
            config: Optional configuration to export. If None, loads current config.
            pretty_print: Whether to format JSON with indentation
        """
        if config is None:
            config = self.load_config()

        # Add export metadata
        export_config = config.copy()
        export_config['_export_metadata'] = {
            'exported_at': datetime.utcnow().isoformat(),
            'exported_by': 'CLI ConfigManager',
            'original_path': self._resolved_config_path
        }

        # Write to file
        with open(output_path, 'w') as f:
            if pretty_print:
                json.dump(export_config, f, indent=2, sort_keys=True)
            else:
                json.dump(export_config, f)

        logger.info(f"Configuration exported to {output_path}")

    # Private methods for internal functionality

    def _can_use_cached_config(self) -> bool:
        """Check if cached configuration is still valid."""
        if not self.enable_caching or self._cached_config is None:
            return False

        if self._cache_timestamp is None:
            return False

        # Check if cache has expired
        cache_age = time.time() - self._cache_timestamp
        if cache_age > self.cache_ttl_seconds:
            logger.debug(f"Cache expired (age: {cache_age:.1f}s)")
            return False

        # Check if config file has been modified
        if self._resolved_config_path and os.path.exists(self._resolved_config_path):
            file_mtime = os.path.getmtime(self._resolved_config_path)
            if file_mtime > self._cache_timestamp:
                logger.debug("Config file modified since cache, reloading")
                return False

        return True

    def _resolve_config_path(self) -> str:
        """Resolve the configuration file path with fallback logic."""
        if self.config_path:
            if os.path.exists(self.config_path):
                return os.path.abspath(self.config_path)
            else:
                raise ConfigurationError(f"Specified config file not found: {self.config_path}")

        # Try relative path to shared config first
        current_dir = os.path.dirname(os.path.abspath(__file__))
        shared_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))),
            "shared",
            "config.json"
        )

        if os.path.exists(shared_config_path):
            return os.path.abspath(shared_config_path)

        # Fallback to config.json in current directory
        local_config_path = os.path.abspath("config.json")
        if os.path.exists(local_config_path):
            return local_config_path

        # If no config found, raise error
        raise ConfigurationError(
            f"Configuration file not found. Tried: {shared_config_path}, {local_config_path}"
        )

    def _load_raw_config(self, config_path: str) -> Dict[str, Any]:
        """Load raw configuration from file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to read {config_path}: {e}")

    def _process_config(self, raw_config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
        """Process raw configuration (expand env vars, validate, etc.)."""
        # Deep copy to avoid modifying original
        config = json.loads(json.dumps(raw_config))

        # Expand environment variables
        config = self._expand_environment_variables(config)

        # Check and migrate version if needed
        migrated_config, migration_notes = self.migrate_config(config)
        if migration_notes:
            for note in migration_notes:
                logger.info(f"Migration: {note}")

        return migrated_config

    def _expand_environment_variables(self, obj: Any) -> Any:
        """Recursively expand environment variables in configuration."""
        if isinstance(obj, dict):
            return {key: self._expand_environment_variables(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_environment_variables(item) for item in obj]
        elif isinstance(obj, str):
            return self._env_pattern.sub(self._env_replacer, obj)
        else:
            return obj

    def _env_replacer(self, match) -> str:
        """Replace environment variable with its value."""
        var_name = match.group(1)
        default_value = None

        # Support ${VAR:-default} syntax
        if ':-' in var_name:
            var_name, default_value = var_name.split(':-', 1)

        value = os.environ.get(var_name, default_value)
        if value is None:
            logger.warning(f"Environment variable ${{{var_name}}} not found and no default provided")
            return match.group(0)  # Return original if not found

        return value

    def _merge_strategy_config(self, strategy: Dict[str, Any],
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge strategy configuration with global defaults."""
        # Get global defaults
        global_defaults = config.get('global_defaults', {})
        strategy_defaults = global_defaults.get('strategy', {})

        # Merge in order: strategy_defaults -> strategy
        merged = strategy_defaults.copy()
        merged.update(strategy)

        # Handle position sizing inheritance specifically
        merged = self._merge_position_sizing_config(merged, config)

        # Add computed fields
        merged['_computed'] = {
            'has_all_indicators': self._strategy_has_all_indicators(strategy, config),
            'indicator_count': len(strategy.get('indicators', [])),
            'exchange_enabled': self._is_exchange_enabled(strategy.get('exchange'), config)
        }

        return merged

    def _merge_position_sizing_config(self, strategy: Dict[str, Any],
                                    global_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge position sizing configuration with inheritance from global config.

        Strategy-specific position sizing takes precedence over global defaults.
        Individual strategy fields (max_position_size, risk_per_trade_pct) can override
        specific aspects of position sizing configuration.

        Args:
            strategy: Strategy configuration dictionary
            global_config: Global configuration dictionary

        Returns:
            Strategy configuration with merged position sizing
        """
        # Get global position sizing configuration
        global_position_sizing = global_config.get('position_sizing', {})

        # Get strategy-specific position sizing (if any)
        strategy_position_sizing = strategy.get('position_sizing', {})

        # Start with global position sizing as base
        merged_position_sizing = global_position_sizing.copy()

        # Override with strategy-specific position sizing
        merged_position_sizing.update(strategy_position_sizing)

        # Handle individual strategy fields that map to position sizing
        # These provide backward compatibility and convenience
        strategy_overrides = {}

        if 'max_position_size' in strategy:
            # Don't multiply by 1000 - assume it's already in the correct format
            # The max_position_size in strategy could be in units or USD depending on context
            max_pos = strategy['max_position_size']
            if max_pos < 100:  # Likely in units, convert to reasonable USD value
                strategy_overrides['max_position_size_usd'] = max_pos * 1000  # Convert to USD
            else:  # Already in USD
                strategy_overrides['max_position_size_usd'] = max_pos

        if 'risk_per_trade_pct' in strategy:
            strategy_overrides['risk_per_trade_pct'] = strategy['risk_per_trade_pct'] / 100.0

        if 'stop_loss_pct' in strategy:
            strategy_overrides['default_stop_loss_pct'] = strategy['stop_loss_pct'] / 100.0

        # Apply strategy field overrides to position sizing
        merged_position_sizing.update(strategy_overrides)

        # Store the merged position sizing back in strategy
        strategy['position_sizing'] = merged_position_sizing

        return strategy

    def get_effective_position_sizing_config(self, strategy_name: str,
                                           config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get the effective position sizing configuration for a strategy.

        This method resolves all inheritance and returns the final position sizing
        configuration that will be used for the strategy.

        Args:
            strategy_name: Name of the strategy
            config: Optional configuration dict. If None, loads current config.

        Returns:
            Dictionary containing the effective position sizing configuration

        Raises:
            ValueError: If strategy not found
        """
        strategy_config = self.get_strategy_config(strategy_name, config)
        if not strategy_config:
            raise ValueError(f"Strategy '{strategy_name}' not found in configuration")

        return strategy_config.get('position_sizing', {})

    def validate_position_sizing_inheritance(self, strategy_name: str,
                                           config: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Validate position sizing configuration inheritance for a strategy.

        Args:
            strategy_name: Name of the strategy to validate
            config: Optional configuration dict. If None, loads current config.

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        try:
            if config is None:
                config = self.load_config()

            # Check if strategy exists
            strategy_config = self.get_strategy_config(strategy_name, config)
            if not strategy_config:
                issues.append(f"Strategy '{strategy_name}' not found")
                return issues

            # Check global position sizing exists
            global_position_sizing = config.get('position_sizing')
            if not global_position_sizing:
                issues.append("No global position sizing configuration found")

            # Get effective position sizing
            effective_config = strategy_config.get('position_sizing', {})

            # Check required fields are present
            required_fields = ['method']
            for field in required_fields:
                if field not in effective_config:
                    issues.append(f"Missing required position sizing field: {field}")

            # Validate method-specific requirements
            method = effective_config.get('method')
            if method == 'fixed_usd' and not effective_config.get('fixed_usd_amount'):
                issues.append("Fixed USD method requires 'fixed_usd_amount' parameter")
            elif method == 'percentage_equity' and not effective_config.get('equity_percentage'):
                issues.append("Percentage equity method requires 'equity_percentage' parameter")
            elif method == 'risk_based' and not effective_config.get('risk_per_trade_pct'):
                issues.append("Risk-based method requires 'risk_per_trade_pct' parameter")

        except Exception as e:
            issues.append(f"Error validating position sizing inheritance: {e}")

        return issues

    def _strategy_has_all_indicators(self, strategy: Dict[str, Any],
                                   config: Dict[str, Any]) -> bool:
        """Check if all strategy indicators exist in config."""
        strategy_indicators = set(strategy.get('indicators', []))
        available_indicators = {ind.get('name') for ind in config.get('indicators', [])}
        return strategy_indicators.issubset(available_indicators)

    def _is_exchange_enabled(self, exchange_name: str, config: Dict[str, Any]) -> bool:
        """Check if specified exchange is enabled."""
        if not exchange_name:
            return False

        exchanges = config.get('exchanges', [])
        for exchange in exchanges:
            if exchange.get('name') == exchange_name:
                return exchange.get('enabled', False)

        return False

    def _validate_version(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration version."""
        errors = []
        version = config.get('version')

        if not version:
            errors.append("Configuration missing version field")
        elif version not in self.SUPPORTED_VERSIONS:
            errors.append(f"Unsupported configuration version: {version}. "
                         f"Supported versions: {', '.join(self.SUPPORTED_VERSIONS)}")

        return errors

    def _validate_sections(self, config: Dict[str, Any]) -> List[str]:
        """Validate required configuration sections."""
        errors = []
        required_sections = ['strategies', 'indicators', 'exchanges']

        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: '{section}'")
            elif not isinstance(config[section], list):
                errors.append(f"Section '{section}' must be a list")

        return errors

    def _validate_strategies(self, config: Dict[str, Any]) -> List[str]:
        """Validate strategy configurations."""
        errors = []
        strategies = config.get('strategies', [])

        for i, strategy in enumerate(strategies):
            prefix = f"Strategy {i+1} ({strategy.get('name', 'unnamed')})"

            # Required fields
            required_fields = ['name', 'market', 'exchange', 'timeframe']
            for field in required_fields:
                if field not in strategy:
                    errors.append(f"{prefix}: Missing required field '{field}'")

            # Numeric field validation
            numeric_fields = {
                'main_leverage': (0.1, 100.0),
                'hedge_leverage': (0.1, 100.0),
                'hedge_ratio': (0.0, 1.0),
                'stop_loss_pct': (0.1, 50.0),
                'take_profit_pct': (0.1, 100.0),
                'max_position_size': (0.001, 10.0),
                'risk_per_trade_pct': (0.001, 0.1)
            }

            for field, (min_val, max_val) in numeric_fields.items():
                if field in strategy:
                    value = strategy[field]
                    if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                        errors.append(f"{prefix}: '{field}' must be between {min_val} and {max_val}")

        return errors

    def _validate_indicators(self, config: Dict[str, Any]) -> List[str]:
        """Validate indicator configurations."""
        errors = []
        indicators = config.get('indicators', [])
        indicator_names = set()

        for i, indicator in enumerate(indicators):
            prefix = f"Indicator {i+1} ({indicator.get('name', 'unnamed')})"

            # Required fields
            required_fields = ['name', 'type', 'timeframe', 'symbol']
            for field in required_fields:
                if field not in indicator:
                    errors.append(f"{prefix}: Missing required field '{field}'")

            # Check for duplicate names
            name = indicator.get('name')
            if name:
                if name in indicator_names:
                    errors.append(f"{prefix}: Duplicate indicator name '{name}'")
                indicator_names.add(name)

        return errors

    def _validate_exchanges(self, config: Dict[str, Any]) -> List[str]:
        """Validate exchange configurations."""
        errors = []
        exchanges = config.get('exchanges', [])

        if not any(ex.get('enabled', False) for ex in exchanges):
            errors.append("No exchanges are enabled")

        exchange_names = set()
        for i, exchange in enumerate(exchanges):
            prefix = f"Exchange {i+1} ({exchange.get('name', 'unnamed')})"

            if 'name' not in exchange:
                errors.append(f"{prefix}: Missing required field 'name'")
            else:
                name = exchange['name']
                if name in exchange_names:
                    errors.append(f"{prefix}: Duplicate exchange name '{name}'")
                exchange_names.add(name)

        return errors

    def _validate_cross_references(self, config: Dict[str, Any]) -> List[str]:
        """Validate cross-references between strategies, indicators, and exchanges."""
        errors = []

        # Get available names
        indicator_names = {ind.get('name') for ind in config.get('indicators', [])}
        exchange_names = {ex.get('name') for ex in config.get('exchanges', [])}

        # Validate strategy references
        strategies = config.get('strategies', [])
        for strategy in strategies:
            strategy_name = strategy.get('name', 'unnamed')

            # Check indicator references
            for indicator_name in strategy.get('indicators', []):
                if indicator_name not in indicator_names:
                    errors.append(f"Strategy '{strategy_name}': "
                                f"Referenced indicator '{indicator_name}' not found")

            # Check exchange reference
            exchange_name = strategy.get('exchange')
            if exchange_name and exchange_name not in exchange_names:
                errors.append(f"Strategy '{strategy_name}': "
                            f"Referenced exchange '{exchange_name}' not found")

        return errors

    def _migrate_from_0_9_0(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate configuration from version 0.9.0 to 1.0.0."""
        migrated = config.copy()

        # Add default position sizing if missing
        strategies = migrated.get('strategies', [])
        for strategy in strategies:
            if 'max_position_size' not in strategy:
                strategy['max_position_size'] = 1.0
            if 'risk_per_trade_pct' not in strategy:
                strategy['risk_per_trade_pct'] = 0.02

        # Normalize timeframe formats
        timeframe_mapping = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h', '1d': '1d',
            'minute': '1m', 'hour': '1h', 'day': '1d'
        }

        for strategy in strategies:
            old_timeframe = strategy.get('timeframe')
            if old_timeframe in timeframe_mapping:
                strategy['timeframe'] = timeframe_mapping[old_timeframe]

        for indicator in migrated.get('indicators', []):
            old_timeframe = indicator.get('timeframe')
            if old_timeframe in timeframe_mapping:
                indicator['timeframe'] = timeframe_mapping[old_timeframe]

        return migrated
