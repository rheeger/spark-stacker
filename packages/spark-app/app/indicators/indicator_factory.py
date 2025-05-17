import logging
from typing import Any, Dict, List, Optional, Type

from app.indicators.adaptive_supertrend_indicator import \
    AdaptiveSupertrendIndicator
from app.indicators.adaptive_trend_finder_indicator import \
    AdaptiveTrendFinderIndicator
from app.indicators.base_indicator import BaseIndicator
from app.indicators.bollinger_bands_indicator import BollingerBandsIndicator
from app.indicators.macd_indicator import MACDIndicator
from app.indicators.moving_average_indicator import MovingAverageIndicator
from app.indicators.rsi_indicator import RSIIndicator
from app.indicators.ultimate_ma_indicator import UltimateMAIndicator

# Import additional indicators as they are implemented
# from app.indicators.macd_indicator import MACDIndicator
# from app.indicators.bollinger_bands_indicator import BollingerBandsIndicator

logger = logging.getLogger(__name__)


class IndicatorFactory:
    """
    Factory for creating technical indicator instances.

    This class provides methods to create and manage indicator instances
    based on configuration parameters.
    """

    # Registry of available indicator types
    _indicator_registry: Dict[str, Type[BaseIndicator]] = {
        "rsi": RSIIndicator,
        "macd": MACDIndicator,
        "bollinger": BollingerBandsIndicator,
        "ma": MovingAverageIndicator,
        "adaptive_supertrend": AdaptiveSupertrendIndicator,
        "adaptive_trend_finder": AdaptiveTrendFinderIndicator,
        "ultimate_ma": UltimateMAIndicator,
    }

    @classmethod
    def create_indicator(
        cls, name: str, indicator_type: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseIndicator]:
        """
        Create a new indicator instance.

        Args:
            name: Unique name for the indicator instance
            indicator_type: Type of indicator to create (e.g., 'rsi', 'macd')
            params: Configuration parameters for the indicator

        Returns:
            Indicator instance or None if indicator_type is not supported
        """
        if indicator_type not in cls._indicator_registry:
            logger.error(
                f"Indicator type '{indicator_type}' not supported. Available types: {list(cls._indicator_registry.keys())}"
            )
            return None

        try:
            indicator_class = cls._indicator_registry[indicator_type]
            return indicator_class(name=name, params=params)
        except Exception as e:
            logger.error(
                f"Failed to create indicator '{name}' of type '{indicator_type}': {e}"
            )
            return None

    @classmethod
    def create_indicators_from_config(
        cls, configs: List[Any]
    ) -> Dict[str, BaseIndicator]:
        """
        Create multiple indicators from configuration.

        Args:
            configs: List of indicator configurations
                Each config can be a dictionary or an IndicatorConfig object

        Returns:
            Dictionary of indicator instances, keyed by name
        """
        indicators = {}

        for config in configs:
            # Check if config is a dictionary or an IndicatorConfig object
            if hasattr(config, "name") and hasattr(config, "type"):
                # It's an IndicatorConfig object
                name = config.name
                indicator_type = config.type
                enabled = getattr(config, "enabled", True)
                params = getattr(config, "parameters", {}) or {}
            elif isinstance(config, dict):
                # It's a dictionary
                name = config.get("name")
                indicator_type = config.get("type")
                enabled = config.get("enabled", True)
                params = config.get("parameters", {}) or config.get("params", {})
            else:
                logger.warning(
                    f"Skipping invalid indicator config type: {type(config)}"
                )
                continue

            logger.debug(
                f"Processing indicator config: name={name}, type={indicator_type}, enabled={enabled}"
            )

            if not name or not indicator_type:
                logger.warning(
                    f"Skipping invalid indicator config: missing name or type"
                )
                continue

            if not enabled:
                logger.info(f"Skipping disabled indicator: {name}")
                continue

            indicator = cls.create_indicator(
                name=name, indicator_type=indicator_type, params=params
            )

            if indicator:
                indicators[name] = indicator
                logger.info(f"Created indicator: {indicator}")

        return indicators

    @classmethod
    def register_indicator(
        cls, indicator_type: str, indicator_class: Type[BaseIndicator]
    ) -> None:
        """
        Register a new indicator type.

        Args:
            indicator_type: String identifier for the indicator type
            indicator_class: Indicator class to register
        """
        # Check if it's a BaseIndicator by name rather than direct class check
        # This allows for different import paths
        if not hasattr(indicator_class, '__mro__') or not any(
            base.__name__ == 'BaseIndicator' for base in indicator_class.__mro__
        ):
            raise TypeError(
                f"Class {indicator_class.__name__} is not a subclass of BaseIndicator"
            )

        cls._indicator_registry[indicator_type] = indicator_class
        logger.info(f"Registered indicator type: {indicator_type}")

    @classmethod
    def get_available_indicators(cls) -> List[str]:
        """
        Get list of available indicator types.

        Returns:
            List of indicator type names
        """
        return list(cls._indicator_registry.keys())
