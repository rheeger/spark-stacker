"""Strategy configuration schema and validation."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Strategy configuration data class with comprehensive validation."""

    name: str
    market: str
    exchange: str
    indicators: List[str]
    enabled: bool = True
    timeframe: str = "1h"
    main_leverage: float = 1.0
    hedge_leverage: float = 1.0
    hedge_ratio: float = 0.0
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 10.0
    max_position_size: float = 0.1
    max_position_size_usd: float = 1000.0
    risk_per_trade_pct: float = 0.02
    position_sizing: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate strategy configuration after initialization."""
        logger.debug(f"Validating strategy configuration: {self.name}")

        # Validate market format (must contain "-")
        if not self.market or "-" not in self.market:
            raise ValueError(
                f"Strategy '{self.name}' has invalid market format: '{self.market}'. "
                "Must use format like 'ETH-USD', 'BTC-USD', etc."
            )

        # Validate exchange is specified
        if not self.exchange:
            raise ValueError(f"Strategy '{self.name}' missing required 'exchange' field")

        # Validate indicators list is not empty
        if not self.indicators:
            raise ValueError(f"Strategy '{self.name}' must specify at least one indicator")

        # Validate timeframe format
        if not self._is_valid_timeframe(self.timeframe):
            raise ValueError(
                f"Strategy '{self.name}' has invalid timeframe '{self.timeframe}'. "
                "Valid formats: '1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '1w'"
            )

        # Validate numeric parameters
        if self.main_leverage <= 0:
            raise ValueError(f"Strategy '{self.name}' main_leverage must be positive")

        if self.hedge_leverage <= 0:
            raise ValueError(f"Strategy '{self.name}' hedge_leverage must be positive")

        if not (0 <= self.hedge_ratio <= 1):
            raise ValueError(f"Strategy '{self.name}' hedge_ratio must be between 0 and 1")

        if self.stop_loss_pct <= 0:
            raise ValueError(f"Strategy '{self.name}' stop_loss_pct must be positive")

        if self.take_profit_pct <= 0:
            raise ValueError(f"Strategy '{self.name}' take_profit_pct must be positive")

        if self.max_position_size <= 0:
            raise ValueError(f"Strategy '{self.name}' max_position_size must be positive")

        if self.max_position_size_usd <= 0:
            raise ValueError(f"Strategy '{self.name}' max_position_size_usd must be positive")

        if not (0 < self.risk_per_trade_pct <= 1):
            raise ValueError(f"Strategy '{self.name}' risk_per_trade_pct must be between 0 and 1")

        logger.info(
            f"Strategy config validated: {self.name} -> {self.market} on {self.exchange} "
            f"(timeframe: {self.timeframe}, indicators: {len(self.indicators)})"
        )

    def _is_valid_timeframe(self, timeframe: str) -> bool:
        """
        Validate timeframe format.

        Args:
            timeframe: Timeframe string to validate

        Returns:
            bool: True if valid timeframe format
        """
        # Valid timeframe pattern: number followed by unit (m, h, d, w)
        pattern = r'^(\d+)([mhdw])$'
        match = re.match(pattern, timeframe.lower())

        if not match:
            return False

        number, unit = match.groups()
        number = int(number)

        # Validate ranges for different units
        if unit == 'm':  # minutes
            return number in [1, 5, 15, 30]
        elif unit == 'h':  # hours
            return number in [1, 4, 12]
        elif unit == 'd':  # days
            return number in [1]
        elif unit == 'w':  # weeks
            return number in [1]

        return False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """
        Create a StrategyConfig from a dictionary.

        Args:
            data: Dictionary containing strategy configuration

        Returns:
            StrategyConfig: The created strategy configuration
        """
        return cls(
            name=data.get('name', ''),
            market=data.get('market', ''),
            exchange=data.get('exchange', ''),
            indicators=data.get('indicators', []),
            enabled=data.get('enabled', True),
            timeframe=data.get('timeframe', '1h'),
            main_leverage=float(data.get('main_leverage', 1.0)),
            hedge_leverage=float(data.get('hedge_leverage', 1.0)),
            hedge_ratio=float(data.get('hedge_ratio', 0.0)),
            stop_loss_pct=float(data.get('stop_loss_pct', 5.0)),
            take_profit_pct=float(data.get('take_profit_pct', 10.0)),
            max_position_size=float(data.get('max_position_size', 0.1)),
            max_position_size_usd=float(data.get('max_position_size_usd', 1000.0)),
            risk_per_trade_pct=float(data.get('risk_per_trade_pct', 0.02)),
            position_sizing=data.get('position_sizing')
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert strategy configuration to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'name': self.name,
            'market': self.market,
            'exchange': self.exchange,
            'indicators': self.indicators,
            'enabled': self.enabled,
            'timeframe': self.timeframe,
            'main_leverage': self.main_leverage,
            'hedge_leverage': self.hedge_leverage,
            'hedge_ratio': self.hedge_ratio,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_position_size': self.max_position_size,
            'max_position_size_usd': self.max_position_size_usd,
            'risk_per_trade_pct': self.risk_per_trade_pct,
            'position_sizing': self.position_sizing
        }


class StrategyConfigLoader:
    """Loader and validator for strategy configurations."""

    @staticmethod
    def load_strategies(strategies_data: List[Dict[str, Any]]) -> List[StrategyConfig]:
        """
        Load and validate strategy configurations from data.

        Args:
            strategies_data: List of strategy configuration dictionaries

        Returns:
            List[StrategyConfig]: List of validated strategy configurations

        Raises:
            ValueError: If strategy configuration is invalid
        """
        logger.info(f"Loading {len(strategies_data)} strategy configurations...")

        strategies = []
        strategy_names = set()

        for i, strategy_data in enumerate(strategies_data):
            try:
                # Create strategy config with validation
                strategy = StrategyConfig.from_dict(strategy_data)

                # Check for duplicate strategy names
                if strategy.name in strategy_names:
                    raise ValueError(f"Duplicate strategy name: '{strategy.name}'")

                strategy_names.add(strategy.name)
                strategies.append(strategy)

                logger.debug(f"Successfully loaded strategy '{strategy.name}'")

            except Exception as e:
                logger.error(f"Failed to load strategy at index {i}: {str(e)}")
                raise ValueError(f"Strategy configuration error at index {i}: {str(e)}")

        logger.info(f"Successfully loaded {len(strategies)} strategies")
        return strategies

    @staticmethod
    def validate_indicators(
        strategies: List[StrategyConfig],
        indicators: Dict[str, Any]
    ) -> None:
        """
        Validate that all strategy indicators exist and are properly configured.

        Args:
            strategies: List of strategy configurations
            indicators: Dictionary of loaded indicators

        Raises:
            ValueError: If indicator validation fails
        """
        logger.info("Validating strategy-indicator relationships...")

        # Track which indicators are referenced by strategies
        referenced_indicators = set()

        for strategy in strategies:
            logger.debug(f"Validating indicators for strategy '{strategy.name}'")

            # Validate all strategy indicators exist
            for indicator_name in strategy.indicators:
                if indicator_name not in indicators:
                    raise ValueError(
                        f"Strategy '{strategy.name}' references unknown indicator: '{indicator_name}'. "
                        f"Available indicators: {list(indicators.keys())}"
                    )

                referenced_indicators.add(indicator_name)
                logger.debug(f"Strategy '{strategy.name}' -> indicator '{indicator_name}' ✓")

        # Log summary
        total_indicators = len(indicators)
        used_indicators = len(referenced_indicators)
        unused_indicators = set(indicators.keys()) - referenced_indicators

        logger.info(f"Indicator validation summary:")
        logger.info(f"  Total indicators configured: {total_indicators}")
        logger.info(f"  Indicators used by strategies: {used_indicators}")
        logger.info(f"  Unused indicators: {len(unused_indicators)}")

        if unused_indicators:
            logger.warning(f"Unused indicators: {sorted(unused_indicators)}")

        logger.info("Strategy-indicator validation completed successfully")

    @staticmethod
    def validate_position_sizing_configs(
        strategies: List[StrategyConfig],
        global_position_sizing: Dict[str, Any]
    ) -> None:
        """
        Validate position sizing configurations for strategies.

        Args:
            strategies: List of strategy configurations
            global_position_sizing: Global position sizing configuration

        Raises:
            ValueError: If position sizing configuration is invalid
        """
        logger.info("Validating strategy position sizing configurations...")

        valid_methods = {'fixed_usd', 'equity_percentage', 'risk_based', 'fixed_units', 'kelly'}

        for strategy in strategies:
            if strategy.position_sizing:
                logger.debug(f"Validating position sizing for strategy '{strategy.name}'")

                method = strategy.position_sizing.get('method')
                if not method:
                    raise ValueError(
                        f"Strategy '{strategy.name}' position sizing config missing 'method' field"
                    )

                if method not in valid_methods:
                    raise ValueError(
                        f"Strategy '{strategy.name}' has invalid position sizing method '{method}'. "
                        f"Valid methods: {valid_methods}"
                    )

                # Validate method-specific parameters
                StrategyConfigLoader._validate_position_sizing_method(strategy, method)

                logger.debug(f"Strategy '{strategy.name}' position sizing method '{method}' ✓")
            else:
                logger.debug(f"Strategy '{strategy.name}' will use global position sizing")

        logger.info("Position sizing validation completed successfully")

    @staticmethod
    def _validate_position_sizing_method(strategy: StrategyConfig, method: str) -> None:
        """
        Validate method-specific position sizing parameters.

        Args:
            strategy: Strategy configuration
            method: Position sizing method

        Raises:
            ValueError: If method-specific parameters are invalid
        """
        config = strategy.position_sizing

        if method == 'fixed_usd':
            if 'fixed_usd_amount' not in config or config['fixed_usd_amount'] <= 0:
                raise ValueError(
                    f"Strategy '{strategy.name}' fixed_usd method requires positive 'fixed_usd_amount'"
                )

        elif method == 'equity_percentage':
            percentage = config.get('equity_percentage', 0)
            if not (0 < percentage <= 1):
                raise ValueError(
                    f"Strategy '{strategy.name}' equity_percentage method requires "
                    "'equity_percentage' between 0 and 1"
                )

        elif method == 'risk_based':
            risk_pct = config.get('risk_per_trade_pct', 0)
            if not (0 < risk_pct <= 1):
                raise ValueError(
                    f"Strategy '{strategy.name}' risk_based method requires "
                    "'risk_per_trade_pct' between 0 and 1"
                )

        elif method == 'fixed_units':
            if 'fixed_units' not in config or config['fixed_units'] <= 0:
                raise ValueError(
                    f"Strategy '{strategy.name}' fixed_units method requires positive 'fixed_units'"
                )

        elif method == 'kelly':
            required_fields = ['kelly_win_rate', 'kelly_avg_win', 'kelly_avg_loss']
            for field in required_fields:
                if field not in config or config[field] <= 0:
                    raise ValueError(
                        f"Strategy '{strategy.name}' kelly method requires positive '{field}'"
                    )
