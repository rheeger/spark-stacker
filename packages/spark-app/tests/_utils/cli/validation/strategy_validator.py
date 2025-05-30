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

from ....app.core.strategy_config import StrategyConfig
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
