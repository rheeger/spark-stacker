import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Available position sizing methods."""
    FIXED_USD = "fixed_usd"                    # Fixed dollar amount per trade
    PERCENT_EQUITY = "percent_equity"          # Percentage of current equity
    RISK_BASED = "risk_based"                  # Risk percentage with stop loss
    FIXED_UNITS = "fixed_units"                # Fixed number of units/coins
    KELLY_CRITERION = "kelly_criterion"        # Kelly criterion optimal sizing


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing strategy."""
    method: PositionSizingMethod

    # Fixed USD amount per trade
    fixed_usd_amount: float = 1000.0

    # Percentage of equity per trade (0.01 = 1%)
    equity_percentage: float = 0.05

    # Risk-based parameters
    risk_per_trade_pct: float = 0.02  # 2% risk per trade
    default_stop_loss_pct: float = 0.05  # 5% stop loss if not specified

    # Fixed units
    fixed_units: float = 0.1

    # Kelly criterion parameters
    kelly_win_rate: float = 0.6
    kelly_avg_win: float = 0.03
    kelly_avg_loss: float = 0.02
    kelly_max_position_pct: float = 0.25  # Cap Kelly at 25%

    # Global limits
    max_position_size_usd: float = 10000.0
    min_position_size_usd: float = 10.0
    max_leverage: float = 1.0

    @classmethod
    def from_config_dict(cls, config: Dict) -> 'PositionSizingConfig':
        """Create PositionSizingConfig from configuration dictionary."""
        # Support both 'method' and 'position_sizing_method' keys for flexibility
        method_str = config.get('method') or config.get('position_sizing_method', 'fixed_usd')
        try:
            method = PositionSizingMethod(method_str)
        except ValueError:
            logger.warning(f"Unknown position sizing method: {method_str}, using fixed_usd")
            method = PositionSizingMethod.FIXED_USD

        return cls(
            method=method,
            fixed_usd_amount=config.get('fixed_usd_amount', 1000.0),
            equity_percentage=config.get('equity_percentage', 0.05),
            risk_per_trade_pct=config.get('risk_per_trade_pct', 0.02),
            default_stop_loss_pct=config.get('default_stop_loss_pct', 0.05),
            fixed_units=config.get('fixed_units', 0.1),
            kelly_win_rate=config.get('kelly_win_rate', 0.6),
            kelly_avg_win=config.get('kelly_avg_win', 0.03),
            kelly_avg_loss=config.get('kelly_avg_loss', 0.02),
            kelly_max_position_pct=config.get('kelly_max_position_pct', 0.25),
            max_position_size_usd=config.get('max_position_size_usd', 10000.0),
            min_position_size_usd=config.get('min_position_size_usd', 10.0),
            max_leverage=config.get('max_leverage', 1.0)
        )


class PositionSizer:
    """Calculates position sizes based on configuration."""

    def __init__(self, config: PositionSizingConfig):
        """Initialize with position sizing configuration."""
        self.config = config
        logger.info(f"Initialized position sizer with method: {config.method.value}")

    def calculate_position_size(
        self,
        current_equity: float,
        current_price: float,
        stop_loss_price: Optional[float] = None,
        signal_strength: float = 1.0
    ) -> float:
        """
        Calculate position size based on configuration.

        Args:
            current_equity: Current account equity in USD
            current_price: Current price of the asset
            stop_loss_price: Stop loss price (if using risk-based sizing)
            signal_strength: Signal strength modifier (0.0 to 1.0)

        Returns:
            Position size in asset units (e.g., number of BTC)
        """
        if current_equity <= 0 or current_price <= 0:
            logger.warning(f"Invalid inputs: equity={current_equity}, price={current_price}")
            return 0.0

        # Calculate base position size
        if self.config.method == PositionSizingMethod.FIXED_USD:
            position_size = self._calculate_fixed_usd(current_price, signal_strength)

        elif self.config.method == PositionSizingMethod.PERCENT_EQUITY:
            position_size = self._calculate_percent_equity(current_equity, current_price, signal_strength)

        elif self.config.method == PositionSizingMethod.RISK_BASED:
            position_size = self._calculate_risk_based(current_equity, current_price, stop_loss_price, signal_strength)

        elif self.config.method == PositionSizingMethod.FIXED_UNITS:
            position_size = self._calculate_fixed_units(signal_strength)

        elif self.config.method == PositionSizingMethod.KELLY_CRITERION:
            position_size = self._calculate_kelly(current_equity, current_price, signal_strength)

        else:
            logger.error(f"Unsupported position sizing method: {self.config.method}")
            position_size = 0.0

        # Apply global limits
        position_size = self._apply_limits(position_size, current_price, current_equity)

        logger.debug(f"Calculated position size: {position_size:.6f} units at ${current_price:.2f}")
        return position_size

    def _calculate_fixed_usd(self, current_price: float, signal_strength: float) -> float:
        """Calculate position size using fixed USD amount."""
        usd_amount = self.config.fixed_usd_amount * signal_strength
        return usd_amount / current_price

    def _calculate_percent_equity(self, current_equity: float, current_price: float, signal_strength: float) -> float:
        """Calculate position size using percentage of equity."""
        usd_amount = current_equity * self.config.equity_percentage * signal_strength
        return usd_amount / current_price

    def _calculate_risk_based(
        self,
        current_equity: float,
        current_price: float,
        stop_loss_price: Optional[float],
        signal_strength: float
    ) -> float:
        """Calculate position size based on risk percentage."""
        if stop_loss_price is None:
            # Use default stop loss percentage
            stop_loss_price = current_price * (1 - self.config.default_stop_loss_pct)

        # Calculate risk per unit
        risk_per_unit = abs(current_price - stop_loss_price)

        if risk_per_unit <= 0:
            logger.warning("Invalid stop loss price, using fixed USD method")
            return self._calculate_fixed_usd(current_price, signal_strength)

        # Calculate position size based on risk
        risk_amount = current_equity * self.config.risk_per_trade_pct * signal_strength
        position_size = risk_amount / risk_per_unit

        return position_size

    def _calculate_fixed_units(self, signal_strength: float) -> float:
        """Calculate position size using fixed number of units."""
        return self.config.fixed_units * signal_strength

    def _calculate_kelly(self, current_equity: float, current_price: float, signal_strength: float) -> float:
        """Calculate position size using Kelly criterion."""
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        win_rate = self.config.kelly_win_rate
        avg_win = self.config.kelly_avg_win
        avg_loss = self.config.kelly_avg_loss

        if avg_loss <= 0:
            logger.warning("Invalid Kelly parameters, using fixed USD method")
            return self._calculate_fixed_usd(current_price, signal_strength)

        # Calculate Kelly percentage
        odds = avg_win / avg_loss
        kelly_pct = (odds * win_rate - (1 - win_rate)) / odds

        # Cap at maximum position percentage
        kelly_pct = min(kelly_pct, self.config.kelly_max_position_pct)
        kelly_pct = max(kelly_pct, 0)  # Don't go negative

        # Calculate position size
        usd_amount = current_equity * kelly_pct * signal_strength
        return usd_amount / current_price

    def _apply_limits(self, position_size: float, current_price: float, current_equity: float) -> float:
        """Apply global position size limits."""
        if position_size <= 0:
            return 0.0

        # Calculate USD value of position
        position_value_usd = position_size * current_price

        # Apply maximum USD limit
        if position_value_usd > self.config.max_position_size_usd:
            position_size = self.config.max_position_size_usd / current_price
            logger.debug(f"Position size capped at max USD limit: {self.config.max_position_size_usd}")

        # Apply minimum USD limit
        if position_value_usd < self.config.min_position_size_usd:
            position_size = self.config.min_position_size_usd / current_price
            logger.debug(f"Position size increased to min USD limit: {self.config.min_position_size_usd}")

        # Apply leverage limit (for leveraged positions)
        max_position_value = current_equity * self.config.max_leverage
        if position_value_usd > max_position_value:
            position_size = max_position_value / current_price
            logger.debug(f"Position size capped at max leverage: {self.config.max_leverage}x")

        return position_size

    def get_position_value_usd(self, position_size: float, current_price: float) -> float:
        """Get USD value of a position size."""
        return position_size * current_price

    def validate_position_size(self, position_size: float, current_price: float, current_equity: float) -> bool:
        """Validate if a position size is within acceptable limits."""
        if position_size <= 0:
            return False

        position_value = position_size * current_price

        # Check limits
        if position_value > self.config.max_position_size_usd:
            return False
        if position_value < self.config.min_position_size_usd:
            return False
        if position_value > current_equity * self.config.max_leverage:
            return False

        return True


def create_position_sizer_from_app_config(app_config) -> PositionSizer:
    """Create a PositionSizer from the application configuration."""
    # Extract position sizing config from app config
    position_config = {}

    # Map app config fields to position sizing config
    if hasattr(app_config, 'max_position_size_usd'):
        position_config['max_position_size_usd'] = app_config.max_position_size_usd

    if hasattr(app_config, 'max_leverage'):
        position_config['max_leverage'] = app_config.max_leverage

    if hasattr(app_config, 'max_account_risk_pct'):
        position_config['risk_per_trade_pct'] = app_config.max_account_risk_pct

    # Check for strategy-specific configs
    if hasattr(app_config, 'strategies') and app_config.strategies:
        strategy = app_config.strategies[0]  # Use first strategy
        if hasattr(strategy, 'risk_per_trade_pct'):
            position_config['risk_per_trade_pct'] = strategy.risk_per_trade_pct

    # Check for position sizing specific config
    if hasattr(app_config, 'position_sizing'):
        position_config.update(app_config.position_sizing)

    # Create and return position sizer
    sizing_config = PositionSizingConfig.from_config_dict(position_config)
    return PositionSizer(sizing_config)
