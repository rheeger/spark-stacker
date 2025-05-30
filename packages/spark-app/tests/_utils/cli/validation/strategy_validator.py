"""
Strategy Validator

This module provides strategy-specific validation logic:
- Add strategy-specific validation logic
- Validate strategy-indicator compatibility
- Validate strategy timeframe consistency
- Validate strategy position sizing
- Add strategy feasibility analysis
- Add strategy risk assessment
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.strategy_config import StrategyConfig

from ..core.config_manager import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

    def add_error(self, error: str) -> None:
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning to the validation result."""
        self.warnings.append(warning)

    def add_suggestion(self, suggestion: str) -> None:
        """Add a suggestion to the validation result."""
        self.suggestions.append(suggestion)


class StrategyValidator:
    """
    Validates strategy configurations for correctness, compatibility,
    and feasibility.
    """

    # Valid timeframes in order of granularity
    VALID_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']

    # Valid exchanges
    VALID_EXCHANGES = ['hyperliquid', 'coinbase', 'kraken', 'binance']

    # Valid position sizing methods
    VALID_POSITION_SIZING_METHODS = [
        'fixed_usd', 'percentage_equity', 'kelly_criterion', 'volatility_adjusted'
    ]

    # Valid indicator types
    VALID_INDICATOR_TYPES = [
        'rsi', 'macd', 'sma', 'ema', 'bollinger_bands', 'stochastic',
        'atr', 'williams_r', 'cci', 'mfi'
    ]

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the strategy validator.

        Args:
            config_manager: ConfigManager instance for configuration access
        """
        self.config_manager = config_manager

    def validate_strategy_config(self, strategy_config: StrategyConfig) -> ValidationResult:
        """
        Validate a complete strategy configuration.

        Args:
            strategy_config: Strategy configuration to validate

        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        # Basic configuration validation
        self._validate_basic_config(strategy_config, result)

        # Indicator validation
        self._validate_indicators(strategy_config, result)

        # Position sizing validation
        self._validate_position_sizing(strategy_config, result)

        # Timeframe consistency validation
        self._validate_timeframe_consistency(strategy_config, result)

        # Market and exchange compatibility
        self._validate_market_exchange_compatibility(strategy_config, result)

        # Risk assessment
        self._validate_risk_parameters(strategy_config, result)

        # Feasibility analysis
        self._analyze_strategy_feasibility(strategy_config, result)

        logger.debug(f"Strategy validation completed. Valid: {result.is_valid}, "
                    f"Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")

        return result

    def _validate_basic_config(self, strategy_config: StrategyConfig, result: ValidationResult) -> None:
        """Validate basic strategy configuration fields."""
        # Check required fields
        if not strategy_config.name:
            result.add_error("Strategy name is required")

        if not strategy_config.market:
            result.add_error("Strategy market is required")

        if not strategy_config.exchange:
            result.add_error("Strategy exchange is required")

        if not strategy_config.timeframe:
            result.add_error("Strategy timeframe is required")

        # Validate exchange
        if strategy_config.exchange and strategy_config.exchange not in self.VALID_EXCHANGES:
            result.add_error(f"Invalid exchange: {strategy_config.exchange}. "
                           f"Valid exchanges: {', '.join(self.VALID_EXCHANGES)}")

        # Validate timeframe
        if strategy_config.timeframe and strategy_config.timeframe not in self.VALID_TIMEFRAMES:
            result.add_error(f"Invalid timeframe: {strategy_config.timeframe}. "
                           f"Valid timeframes: {', '.join(self.VALID_TIMEFRAMES)}")

        # Validate market format
        if strategy_config.market and '-' not in strategy_config.market:
            result.add_warning(f"Market symbol '{strategy_config.market}' should contain a dash (e.g., 'BTC-USD')")

    def _validate_indicators(self, strategy_config: StrategyConfig, result: ValidationResult) -> None:
        """Validate strategy indicators."""
        if not strategy_config.indicators:
            result.add_error("Strategy must have at least one indicator")
            return

        for indicator_name, indicator_config in strategy_config.indicators.items():
            # Check indicator type
            indicator_type = indicator_config.get('type')
            if not indicator_type:
                result.add_error(f"Indicator '{indicator_name}' missing type")
                continue

            if indicator_type not in self.VALID_INDICATOR_TYPES:
                result.add_error(f"Invalid indicator type '{indicator_type}' for indicator '{indicator_name}'. "
                               f"Valid types: {', '.join(self.VALID_INDICATOR_TYPES)}")

            # Validate indicator parameters
            self._validate_indicator_parameters(indicator_name, indicator_config, result)

            # Check indicator timeframe compatibility
            indicator_timeframe = indicator_config.get('timeframe')
            if indicator_timeframe and indicator_timeframe != strategy_config.timeframe:
                self._validate_timeframe_relationship(
                    strategy_config.timeframe, indicator_timeframe, indicator_name, result
                )

    def _validate_indicator_parameters(
        self,
        indicator_name: str,
        indicator_config: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Validate indicator-specific parameters."""
        indicator_type = indicator_config.get('type')

        if indicator_type == 'rsi':
            period = indicator_config.get('period', 14)
            if not isinstance(period, int) or period < 2 or period > 100:
                result.add_error(f"RSI indicator '{indicator_name}' period must be between 2 and 100")

        elif indicator_type == 'macd':
            fast_period = indicator_config.get('fast_period', 12)
            slow_period = indicator_config.get('slow_period', 26)
            signal_period = indicator_config.get('signal_period', 9)

            if fast_period >= slow_period:
                result.add_error(f"MACD indicator '{indicator_name}' fast_period must be less than slow_period")

            if signal_period < 1:
                result.add_error(f"MACD indicator '{indicator_name}' signal_period must be at least 1")

        elif indicator_type in ['sma', 'ema']:
            period = indicator_config.get('period', 20)
            if not isinstance(period, int) or period < 2 or period > 200:
                result.add_error(f"{indicator_type.upper()} indicator '{indicator_name}' period must be between 2 and 200")

        elif indicator_type == 'bollinger_bands':
            period = indicator_config.get('period', 20)
            std_dev = indicator_config.get('std_dev', 2.0)

            if not isinstance(period, int) or period < 5 or period > 100:
                result.add_error(f"Bollinger Bands indicator '{indicator_name}' period must be between 5 and 100")

            if not isinstance(std_dev, (int, float)) or std_dev < 0.5 or std_dev > 4.0:
                result.add_error(f"Bollinger Bands indicator '{indicator_name}' std_dev must be between 0.5 and 4.0")

    def _validate_position_sizing(self, strategy_config: StrategyConfig, result: ValidationResult) -> None:
        """Validate position sizing configuration."""
        position_sizing = strategy_config.position_sizing

        if not position_sizing:
            # Check if global position sizing exists
            global_config = self.config_manager.get_global_config()
            global_position_sizing = global_config.get('position_sizing')

            if not global_position_sizing:
                result.add_error("No position sizing configuration found (strategy or global)")
                return
            else:
                result.add_suggestion("Strategy inherits position sizing from global configuration")
                position_sizing = global_position_sizing

        # Validate position sizing method
        method = position_sizing.get('method') or position_sizing.get('position_sizing_method')
        if not method:
            result.add_error("Position sizing method is required")
            return

        if method not in self.VALID_POSITION_SIZING_METHODS:
            result.add_error(f"Invalid position sizing method: {method}. "
                           f"Valid methods: {', '.join(self.VALID_POSITION_SIZING_METHODS)}")

        # Validate method-specific parameters
        if method == 'fixed_usd':
            amount = position_sizing.get('fixed_usd_amount')
            if not amount or amount <= 0:
                result.add_error("Fixed USD position sizing requires positive fixed_usd_amount")

        elif method == 'percentage_equity':
            percentage = position_sizing.get('percentage')
            if not percentage or percentage <= 0 or percentage > 100:
                result.add_error("Percentage equity position sizing requires percentage between 0 and 100")

        elif method == 'kelly_criterion':
            lookback_period = position_sizing.get('lookback_period', 100)
            if lookback_period < 10 or lookback_period > 1000:
                result.add_warning(f"Kelly criterion lookback_period of {lookback_period} may be suboptimal. "
                                 "Consider 50-200 period range.")

    def _validate_timeframe_consistency(self, strategy_config: StrategyConfig, result: ValidationResult) -> None:
        """Validate timeframe consistency across strategy components."""
        strategy_timeframe = strategy_config.timeframe

        # Check indicator timeframes
        for indicator_name, indicator_config in strategy_config.indicators.items():
            indicator_timeframe = indicator_config.get('timeframe')

            if indicator_timeframe and indicator_timeframe != strategy_timeframe:
                self._validate_timeframe_relationship(
                    strategy_timeframe, indicator_timeframe, indicator_name, result
                )

    def _validate_timeframe_relationship(
        self,
        strategy_timeframe: str,
        indicator_timeframe: str,
        indicator_name: str,
        result: ValidationResult
    ) -> None:
        """Validate the relationship between strategy and indicator timeframes."""
        try:
            strategy_idx = self.VALID_TIMEFRAMES.index(strategy_timeframe)
            indicator_idx = self.VALID_TIMEFRAMES.index(indicator_timeframe)

            # Indicator timeframe should typically be <= strategy timeframe
            if indicator_idx > strategy_idx:
                result.add_warning(f"Indicator '{indicator_name}' timeframe '{indicator_timeframe}' "
                                 f"is longer than strategy timeframe '{strategy_timeframe}'. "
                                 "This may cause signal delays.")
            elif indicator_idx < strategy_idx - 2:  # More than 2 levels below
                result.add_warning(f"Indicator '{indicator_name}' timeframe '{indicator_timeframe}' "
                                 f"is much shorter than strategy timeframe '{strategy_timeframe}'. "
                                 "This may cause excessive noise.")

        except ValueError:
            result.add_error(f"Invalid timeframe comparison: {strategy_timeframe} vs {indicator_timeframe}")

    def _validate_market_exchange_compatibility(self, strategy_config: StrategyConfig, result: ValidationResult) -> None:
        """Validate market and exchange compatibility."""
        market = strategy_config.market
        exchange = strategy_config.exchange

        if not market or not exchange:
            return  # Already handled in basic validation

        # Check exchange-specific market format requirements
        if exchange == 'hyperliquid':
            if not market.endswith('-USD'):
                result.add_warning(f"Hyperliquid typically uses USD-denominated pairs. "
                                 f"Market '{market}' may not be available.")

        elif exchange == 'coinbase':
            if not ('-USD' in market or '-EUR' in market or '-GBP' in market):
                result.add_warning(f"Coinbase typically uses fiat-denominated pairs. "
                                 f"Market '{market}' may not be available.")

        # Check for common market symbol issues
        if market.count('-') != 1:
            result.add_error(f"Market symbol '{market}' should have exactly one dash (e.g., 'BTC-USD')")

    def _validate_risk_parameters(self, strategy_config: StrategyConfig, result: ValidationResult) -> None:
        """Validate risk management parameters."""
        risk_config = strategy_config.risk_management

        if not risk_config:
            result.add_suggestion("Consider adding risk management parameters (stop_loss, take_profit)")
            return

        # Validate stop loss
        stop_loss = risk_config.get('stop_loss_pct')
        if stop_loss is not None:
            if stop_loss <= 0 or stop_loss > 50:
                result.add_error(f"Stop loss percentage must be between 0 and 50, got {stop_loss}")

        # Validate take profit
        take_profit = risk_config.get('take_profit_pct')
        if take_profit is not None:
            if take_profit <= 0 or take_profit > 200:
                result.add_error(f"Take profit percentage must be between 0 and 200, got {take_profit}")

        # Validate risk/reward ratio
        if stop_loss and take_profit:
            risk_reward_ratio = take_profit / stop_loss
            if risk_reward_ratio < 1.0:
                result.add_warning(f"Risk/reward ratio of {risk_reward_ratio:.2f} is less than 1:1. "
                                 "Consider adjusting stop loss or take profit levels.")

    def _analyze_strategy_feasibility(self, strategy_config: StrategyConfig, result: ValidationResult) -> None:
        """Analyze overall strategy feasibility."""
        # Check for indicator conflicts
        indicator_types = [config.get('type') for config in strategy_config.indicators.values()]

        # Warn about too many trend-following indicators
        trend_indicators = [t for t in indicator_types if t in ['sma', 'ema', 'macd']]
        if len(trend_indicators) > 3:
            result.add_warning(f"Strategy uses {len(trend_indicators)} trend-following indicators. "
                             "Consider diversifying with momentum or volatility indicators.")

        # Warn about too many oscillators
        oscillator_indicators = [t for t in indicator_types if t in ['rsi', 'stochastic', 'williams_r', 'cci']]
        if len(oscillator_indicators) > 2:
            result.add_warning(f"Strategy uses {len(oscillator_indicators)} oscillator indicators. "
                             "Multiple oscillators may provide redundant signals.")

        # Check for minimum data requirements
        max_indicator_period = self._get_max_indicator_period(strategy_config)
        if max_indicator_period > 100:
            result.add_warning(f"Strategy requires {max_indicator_period} periods for indicators. "
                             "Ensure sufficient historical data is available.")

        # Validate strategy complexity
        total_indicators = len(strategy_config.indicators)
        if total_indicators > 5:
            result.add_warning(f"Strategy uses {total_indicators} indicators. "
                             "Complex strategies may be prone to overfitting.")
        elif total_indicators == 1:
            result.add_suggestion("Single-indicator strategies may benefit from confirmation signals.")

    def _get_max_indicator_period(self, strategy_config: StrategyConfig) -> int:
        """Get the maximum period required by any indicator."""
        max_period = 0

        for indicator_config in strategy_config.indicators.values():
            indicator_type = indicator_config.get('type')

            if indicator_type in ['rsi', 'sma', 'ema']:
                period = indicator_config.get('period', 20)
                max_period = max(max_period, period)

            elif indicator_type == 'macd':
                slow_period = indicator_config.get('slow_period', 26)
                signal_period = indicator_config.get('signal_period', 9)
                max_period = max(max_period, slow_period + signal_period)

            elif indicator_type == 'bollinger_bands':
                period = indicator_config.get('period', 20)
                max_period = max(max_period, period)

        return max_period

    def validate_strategy_indicator_consistency(self, strategy_name: str) -> ValidationResult:
        """
        Validate consistency between strategy configuration and indicator implementations.

        Args:
            strategy_name: Name of strategy to validate

        Returns:
            ValidationResult with consistency validation
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        try:
            strategy_config_dict = self.config_manager.get_strategy_config(strategy_name)
            if not strategy_config_dict:
                result.add_error(f"Strategy '{strategy_name}' not found in configuration")
                return result

            strategy_config = StrategyConfig.from_config_dict(strategy_config_dict)

            # Validate that all configured indicators can be created
            # This would typically involve checking with an IndicatorFactory
            # For now, we'll do basic validation

            for indicator_name, indicator_config in strategy_config.indicators.items():
                indicator_type = indicator_config.get('type')
                if not indicator_type:
                    result.add_error(f"Indicator '{indicator_name}' missing type specification")
                    continue

                # Check if indicator type is supported
                if indicator_type not in self.VALID_INDICATOR_TYPES:
                    result.add_error(f"Unsupported indicator type: {indicator_type}")

            logger.debug(f"Strategy-indicator consistency validation completed for {strategy_name}")

        except Exception as e:
            result.add_error(f"Failed to validate strategy-indicator consistency: {e}")

        return result

    def validate_timeframe_consistency(self, strategy_name: str) -> ValidationResult:
        """
        Validate timeframe consistency for a strategy.

        Args:
            strategy_name: Name of strategy to validate

        Returns:
            ValidationResult with timeframe validation
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        try:
            strategy_config_dict = self.config_manager.get_strategy_config(strategy_name)
            if not strategy_config_dict:
                result.add_error(f"Strategy '{strategy_name}' not found in configuration")
                return result

            strategy_config = StrategyConfig.from_config_dict(strategy_config_dict)
            self._validate_timeframe_consistency(strategy_config, result)

            logger.debug(f"Timeframe consistency validation completed for {strategy_name}")

        except Exception as e:
            result.add_error(f"Failed to validate timeframe consistency: {e}")

        return result

    def validate_market_exchange_compatibility(self, strategy_name: str) -> ValidationResult:
        """
        Validate market and exchange compatibility for a strategy.

        Args:
            strategy_name: Name of strategy to validate

        Returns:
            ValidationResult with market-exchange validation
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        try:
            strategy_config_dict = self.config_manager.get_strategy_config(strategy_name)
            if not strategy_config_dict:
                result.add_error(f"Strategy '{strategy_name}' not found in configuration")
                return result

            strategy_config = StrategyConfig.from_config_dict(strategy_config_dict)
            self._validate_market_exchange_compatibility(strategy_config, result)

            logger.debug(f"Market-exchange compatibility validation completed for {strategy_name}")

        except Exception as e:
            result.add_error(f"Failed to validate market-exchange compatibility: {e}")

        return result

    def validate_position_sizing_config(self, position_sizing_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate a position sizing configuration dictionary.

        Args:
            position_sizing_config: Position sizing configuration to validate

        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])

        # Check required method field
        method = position_sizing_config.get('method')
        if not method:
            result.add_error("Position sizing method is required")
            return result

        # Validate method value
        if method not in self.VALID_POSITION_SIZING_METHODS:
            result.add_error(f"Invalid position sizing method: {method}. "
                           f"Valid methods: {', '.join(self.VALID_POSITION_SIZING_METHODS)}")

        # Method-specific validation
        if method == 'fixed_usd':
            amount = position_sizing_config.get('fixed_usd_amount')
            if not amount or amount <= 0:
                result.add_error("Fixed USD method requires positive 'fixed_usd_amount'")
            elif amount < 10:
                result.add_warning(f"Fixed USD amount ${amount} is very small, consider increasing")

        elif method == 'percentage_equity':
            percentage = position_sizing_config.get('equity_percentage')
            if not percentage or percentage <= 0 or percentage > 1:
                result.add_error("Percentage equity method requires 'equity_percentage' between 0 and 1")
            elif percentage > 0.1:
                result.add_warning(f"Equity percentage {percentage*100:.1f}% is high, consider risk management")

        elif method == 'risk_based':
            risk_pct = position_sizing_config.get('risk_per_trade_pct')
            if not risk_pct or risk_pct <= 0 or risk_pct > 0.1:
                result.add_error("Risk-based method requires 'risk_per_trade_pct' between 0 and 0.1 (10%)")
            stop_loss_pct = position_sizing_config.get('default_stop_loss_pct')
            if not stop_loss_pct or stop_loss_pct <= 0:
                result.add_error("Risk-based method requires positive 'default_stop_loss_pct'")

        elif method == 'kelly_criterion':
            win_rate = position_sizing_config.get('kelly_win_rate')
            if not win_rate or win_rate <= 0 or win_rate >= 1:
                result.add_error("Kelly criterion requires 'kelly_win_rate' between 0 and 1")
            avg_win = position_sizing_config.get('kelly_avg_win')
            avg_loss = position_sizing_config.get('kelly_avg_loss')
            if not avg_win or avg_win <= 0:
                result.add_error("Kelly criterion requires positive 'kelly_avg_win'")
            if not avg_loss or avg_loss <= 0:
                result.add_error("Kelly criterion requires positive 'kelly_avg_loss'")

        # Validate limit fields
        max_position = position_sizing_config.get('max_position_size_usd')
        if max_position and max_position <= 0:
            result.add_error("max_position_size_usd must be positive if specified")

        min_position = position_sizing_config.get('min_position_size_usd')
        if min_position and min_position <= 0:
            result.add_error("min_position_size_usd must be positive if specified")

        if max_position and min_position and min_position > max_position:
            result.add_error("min_position_size_usd cannot be greater than max_position_size_usd")

        max_leverage = position_sizing_config.get('max_leverage')
        if max_leverage and max_leverage <= 0:
            result.add_error("max_leverage must be positive if specified")
        elif max_leverage and max_leverage > 10:
            result.add_warning(f"Max leverage {max_leverage}x is very high, consider risk management")

        return result

    def merge_position_sizing_config(self, strategy_config: Dict[str, Any],
                                   global_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge position sizing configuration from strategy and global config.

        Args:
            strategy_config: Strategy-specific configuration
            global_config: Global configuration

        Returns:
            Merged position sizing configuration
        """
        # Get global position sizing as base
        global_position_sizing = global_config.get('position_sizing', {})
        merged = global_position_sizing.copy()

        # Override with strategy-specific position sizing
        strategy_position_sizing = strategy_config.get('position_sizing', {})
        merged.update(strategy_position_sizing)

        # Handle individual strategy fields that map to position sizing
        if 'max_position_size' in strategy_config:
            merged['max_position_size_usd'] = strategy_config['max_position_size']

        if 'risk_per_trade_pct' in strategy_config:
            merged['risk_per_trade_pct'] = strategy_config['risk_per_trade_pct'] / 100.0

        if 'stop_loss_pct' in strategy_config:
            merged['default_stop_loss_pct'] = strategy_config['stop_loss_pct'] / 100.0

        return merged

    def calculate_effective_position_size(self, position_sizing_config: Dict[str, Any],
                                        current_price: float,
                                        current_equity: float = 10000.0,
                                        signal_strength: float = 1.0) -> Dict[str, Any]:
        """
        Calculate effective position size for given parameters.

        Args:
            position_sizing_config: Position sizing configuration
            current_price: Current asset price
            current_equity: Current account equity
            signal_strength: Signal strength (0-1)

        Returns:
            Dictionary with position size calculations and analysis
        """
        try:
            from app.risk_management.position_sizing import (
                PositionSizer, PositionSizingConfig)

            # Create position sizer from config
            sizing_config = PositionSizingConfig.from_config_dict(position_sizing_config)
            sizer = PositionSizer(sizing_config)

            # Calculate position size
            position_size = sizer.calculate_position_size(
                current_price=current_price,
                current_equity=current_equity,
                signal_strength=signal_strength
            )

            # Calculate additional metrics
            position_value = position_size * current_price
            equity_percentage = (position_value / current_equity) * 100

            return {
                'position_size_units': position_size,
                'position_value_usd': position_value,
                'equity_percentage': equity_percentage,
                'current_price': current_price,
                'current_equity': current_equity,
                'signal_strength': signal_strength,
                'method': sizing_config.method.value,
                'is_valid': position_size > 0,
                'analysis': {
                    'within_limits': position_value <= sizing_config.max_position_size_usd,
                    'above_minimum': position_value >= sizing_config.min_position_size_usd,
                    'leverage_used': position_value / current_equity,
                    'max_leverage_limit': sizing_config.max_leverage
                }
            }

        except Exception as e:
            return {
                'error': f"Failed to calculate position size: {e}",
                'position_size_units': 0,
                'position_value_usd': 0,
                'equity_percentage': 0,
                'is_valid': False
            }

    def compare_position_sizing_configs(self, config1: Dict[str, Any],
                                      config2: Dict[str, Any],
                                      test_scenarios: List[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Compare two position sizing configurations across multiple scenarios.

        Args:
            config1: First position sizing configuration
            config2: Second position sizing configuration
            test_scenarios: Optional list of test scenarios with price and equity

        Returns:
            Dictionary with detailed comparison analysis
        """
        if test_scenarios is None:
            test_scenarios = [
                {'price': 100, 'equity': 10000, 'signal': 1.0},
                {'price': 200, 'equity': 10000, 'signal': 0.8},
                {'price': 50, 'equity': 10000, 'signal': 0.6},
                {'price': 100, 'equity': 5000, 'signal': 1.0},
                {'price': 100, 'equity': 20000, 'signal': 1.0},
            ]

        comparison = {
            'config1_method': config1.get('method', 'unknown'),
            'config2_method': config2.get('method', 'unknown'),
            'scenarios': [],
            'summary': {}
        }

        config1_values = []
        config2_values = []

        for i, scenario in enumerate(test_scenarios):
            result1 = self.calculate_effective_position_size(
                config1, scenario['price'], scenario['equity'], scenario['signal']
            )
            result2 = self.calculate_effective_position_size(
                config2, scenario['price'], scenario['equity'], scenario['signal']
            )

            config1_values.append(result1['position_value_usd'])
            config2_values.append(result2['position_value_usd'])

            comparison['scenarios'].append({
                'scenario': i + 1,
                'price': scenario['price'],
                'equity': scenario['equity'],
                'signal': scenario['signal'],
                'config1': result1,
                'config2': result2,
                'difference_usd': result2['position_value_usd'] - result1['position_value_usd'],
                'difference_pct': ((result2['position_value_usd'] / result1['position_value_usd']) - 1) * 100
                                 if result1['position_value_usd'] > 0 else 0
            })

        # Calculate summary statistics
        if config1_values and config2_values:
            comparison['summary'] = {
                'avg_config1_usd': sum(config1_values) / len(config1_values),
                'avg_config2_usd': sum(config2_values) / len(config2_values),
                'max_difference_usd': max(abs(v2 - v1) for v1, v2 in zip(config1_values, config2_values)),
                'avg_difference_usd': sum(v2 - v1 for v1, v2 in zip(config1_values, config2_values)) / len(config1_values),
                'config2_larger_scenarios': sum(1 for v1, v2 in zip(config1_values, config2_values) if v2 > v1),
                'config1_larger_scenarios': sum(1 for v1, v2 in zip(config1_values, config2_values) if v1 > v2),
            }

        return comparison

    def analyze_position_sizing_impact(self, position_sizing_config: Dict[str, Any],
                                     parameter_variations: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Analyze the impact of changing position sizing parameters.

        Args:
            position_sizing_config: Base position sizing configuration
            parameter_variations: Dictionary of parameter names to lists of test values

        Returns:
            Dictionary with impact analysis results
        """
        base_result = self.calculate_effective_position_size(
            position_sizing_config, current_price=100, current_equity=10000
        )

        analysis = {
            'base_configuration': position_sizing_config.copy(),
            'base_result': base_result,
            'parameter_impacts': {},
            'sensitivity_ranking': []
        }

        sensitivities = []

        for param_name, test_values in parameter_variations.items():
            if param_name not in position_sizing_config:
                continue

            original_value = position_sizing_config[param_name]
            impacts = []

            for test_value in test_values:
                # Create modified config
                modified_config = position_sizing_config.copy()
                modified_config[param_name] = test_value

                # Calculate result with modified parameter
                result = self.calculate_effective_position_size(
                    modified_config, current_price=100, current_equity=10000
                )

                # Calculate impact
                if base_result['position_value_usd'] > 0:
                    impact_pct = ((result['position_value_usd'] / base_result['position_value_usd']) - 1) * 100
                else:
                    impact_pct = 0

                impacts.append({
                    'test_value': test_value,
                    'result': result,
                    'impact_pct': impact_pct,
                    'difference_usd': result['position_value_usd'] - base_result['position_value_usd']
                })

            # Calculate sensitivity (max absolute impact)
            max_impact = max(abs(impact['impact_pct']) for impact in impacts) if impacts else 0
            sensitivities.append((param_name, max_impact))

            analysis['parameter_impacts'][param_name] = {
                'original_value': original_value,
                'impacts': impacts,
                'max_impact_pct': max_impact,
                'avg_impact_pct': sum(abs(impact['impact_pct']) for impact in impacts) / len(impacts) if impacts else 0
            }

        # Rank parameters by sensitivity
        analysis['sensitivity_ranking'] = sorted(sensitivities, key=lambda x: x[1], reverse=True)

        return analysis
