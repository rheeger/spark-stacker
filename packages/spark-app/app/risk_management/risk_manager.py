import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from app.connectors.base_connector import BaseConnector, OrderSide

from .position_sizing import (PositionSizer, PositionSizingConfig,
                              PositionSizingMethod)

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk management system to control position size, leverage, and risk.

    This class handles risk-related calculations and validations before placing trades.
    It helps protect the principal by limiting exposure and implementing smart risk controls.

    Supports strategy-specific position sizing configurations.
    """

    def __init__(
        self,
        max_account_risk_pct: float = 2.0,
        max_leverage: float = 25.0,
        max_position_size_usd: Optional[float] = None,
        max_positions: int = 3,
        min_margin_buffer_pct: float = 20.0,
        position_sizer: Optional[PositionSizer] = None,
        position_sizing_config: Optional[Dict[str, Any]] = None,
        strategy_position_sizers: Optional[Dict[str, PositionSizer]] = None,
    ):
        """
        Initialize the risk manager.

        Args:
            max_account_risk_pct: Maximum percentage of account to risk per trade
            max_leverage: Maximum allowed leverage
            max_position_size_usd: Maximum position size in USD (None for no limit)
            max_positions: Maximum number of concurrent positions
            min_margin_buffer_pct: Minimum margin buffer percentage
            position_sizer: Optional custom position sizer instance (used as default)
            position_sizing_config: Optional position sizing configuration dict (used as default)
            strategy_position_sizers: Optional dict of strategy name -> PositionSizer mappings
        """
        self.max_account_risk_pct = max_account_risk_pct
        self.max_leverage = max_leverage
        self.max_position_size_usd = max_position_size_usd
        self.max_positions = max_positions
        self.min_margin_buffer_pct = min_margin_buffer_pct

        # Initialize default position sizer
        if position_sizer:
            self.position_sizer = position_sizer
            logger.info(f"Using provided position sizer with method: {position_sizer.config.method.value}")
        elif position_sizing_config:
            # Create position sizer from config
            sizing_config = PositionSizingConfig.from_config_dict(position_sizing_config)
            self.position_sizer = PositionSizer(sizing_config)
            logger.info(f"Created position sizer from config with method: {sizing_config.method.value}")
        else:
            # Create default position sizer with fallback configuration
            default_config = {
                'position_sizing_method': 'fixed_usd',
                'fixed_usd_amount': max_position_size_usd or 1000.0,
                'max_position_size_usd': max_position_size_usd or 10000.0,
                'min_position_size_usd': 10.0,
                'max_leverage': max_leverage,
                'risk_per_trade_pct': max_account_risk_pct / 100.0,
            }
            sizing_config = PositionSizingConfig.from_config_dict(default_config)
            self.position_sizer = PositionSizer(sizing_config)
            logger.info("Created default position sizer with fixed USD method")

        # Initialize strategy-specific position sizers
        self.strategy_position_sizers: Dict[str, PositionSizer] = strategy_position_sizers or {}

        if self.strategy_position_sizers:
            logger.info(f"Initialized with {len(self.strategy_position_sizers)} strategy-specific position sizers")
            for strategy_name, sizer in self.strategy_position_sizers.items():
                logger.info(f"  {strategy_name}: {sizer.config.method.value}")

        # Track positions and risk metrics
        self.positions = {}
        self.total_exposure = 0.0

    def calculate_position_size(
        self,
        exchange: BaseConnector,
        symbol: str,
        available_balance: Union[float, Dict[str, Any]],
        confidence: float,
        signal_side: OrderSide,
        leverage: float,
        price: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        strategy_name: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Calculate position size using the integrated position sizer.

        Uses strategy-specific position sizer if available, otherwise falls back to default.

        Args:
            exchange: Exchange connector
            symbol: Market symbol (e.g., 'ETH')
            available_balance: Available balance (either as float or dictionary of balances)
            confidence: Signal confidence (0-1)
            signal_side: Order side (BUY/SELL)
            leverage: Initial leverage
            price: Optional price override
            stop_loss_pct: Stop loss percentage
            strategy_name: Optional strategy name for strategy-specific position sizing

        Returns:
            Tuple of (position_size_usd, adjusted_leverage) where position_size_usd is in USD
        """
        # Select appropriate position sizer based on strategy
        if strategy_name and strategy_name in self.strategy_position_sizers:
            position_sizer = self.strategy_position_sizers[strategy_name]
            logger.debug(f"Using strategy-specific position sizer for '{strategy_name}': {position_sizer.config.method.value}")
        else:
            position_sizer = self.position_sizer
            if strategy_name:
                logger.debug(f"No specific position sizer for strategy '{strategy_name}', using default: {position_sizer.config.method.value}")
            else:
                logger.debug(f"No strategy specified, using default position sizer: {position_sizer.config.method.value}")

        # Handle balance input - convert dictionary to total if needed
        try:
            if isinstance(available_balance, dict):
                current_equity = sum(float(val) for val in available_balance.values())
                logger.debug(f"Converted balance dictionary {available_balance} to total: {current_equity}")
            else:
                current_equity = float(available_balance)
                logger.debug(f"Using direct balance value: {current_equity}")

            if current_equity <= 0:
                logger.warning("Available balance is zero or negative")
                return 0.0, 1.0
        except (TypeError, ValueError) as e:
            logger.error(f"Error processing available balance: {e}")
            return 0.0, 1.0

        # Get current price if not provided
        current_price = price
        if not current_price:
            try:
                ticker = exchange.get_ticker(symbol)
                if ticker and ticker.get("last_price"):
                    current_price = ticker["last_price"]
                    logger.debug(f"Using current market price: {current_price}")
                else:
                    logger.warning(f"Could not fetch current market price for {symbol}")
                    return 0.0, 1.0
            except Exception as e:
                logger.warning(f"Could not fetch current market price: {e}")
                return 0.0, 1.0

        # Calculate stop loss price if percentage is provided
        stop_loss_price = None
        if stop_loss_pct and stop_loss_pct > 0:
            if signal_side == OrderSide.BUY:
                # For long positions, stop loss is below entry price
                stop_loss_price = current_price * (1 - stop_loss_pct / 100.0)
            else:
                # For short positions, stop loss is above entry price
                stop_loss_price = current_price * (1 + stop_loss_pct / 100.0)

        # Determine max leverage from exchange
        try:
            exchange_max_leverage = 25.0  # Default max leverage
            leverage_tiers = exchange.get_leverage_tiers(symbol)

            if leverage_tiers:
                max_leverages = []
                for tier in leverage_tiers:
                    if isinstance(tier, dict) and 'max_leverage' in tier:
                        max_leverages.append(float(tier['max_leverage']))

                if max_leverages:
                    exchange_max_leverage = max(max_leverages)
                    logger.debug(f"Found max exchange leverage: {exchange_max_leverage}")
                else:
                    logger.warning(f"Empty or invalid leverage tiers for {symbol}, using default max_leverage: {exchange_max_leverage}")
            else:
                logger.warning(f"No leverage tiers found for {symbol}, using default max_leverage: {exchange_max_leverage}")

            # Apply the leverage cap based on our findings
            adjusted_leverage = min(leverage, exchange_max_leverage, self.max_leverage, position_sizer.config.max_leverage)
            logger.debug(f"Adjusted leverage to {adjusted_leverage} based on limits")

        except Exception as e:
            logger.error(f"Unexpected error determining leverage for {symbol}: {e}")
            adjusted_leverage = min(leverage, self.max_leverage, position_sizer.config.max_leverage)
            logger.debug(f"Using fallback leverage: {adjusted_leverage}")

        # Use position sizer to calculate position size in asset units
        try:
            position_size_units = position_sizer.calculate_position_size(
                current_equity=current_equity,
                current_price=current_price,
                stop_loss_price=stop_loss_price,
                signal_strength=confidence
            )

            # Convert position size from asset units to USD value
            position_size_usd = position_size_units * current_price

            logger.debug(f"Position sizer result: {position_size_units:.6f} units = ${position_size_usd:.2f} USD")

            # Apply additional risk management limits
            if self.max_position_size_usd and position_size_usd > self.max_position_size_usd:
                logger.debug(f"Reducing position size from ${position_size_usd:.2f} to max limit: ${self.max_position_size_usd}")
                position_size_usd = self.max_position_size_usd

            logger.info(f"Final position calculation for {symbol}: ${position_size_usd:.2f} USD, leverage={adjusted_leverage:.1f}x")
            if strategy_name:
                logger.info(f"  Strategy: {strategy_name}, Position sizer: {position_sizer.config.method.value}")
            return position_size_usd, adjusted_leverage

        except Exception as e:
            logger.error(f"Error in position size calculation for {symbol}: {e}", exc_info=True)
            # Return conservative values
            default_position = current_equity * 0.05  # Use 5% of balance as safe default
            default_leverage = min(2.0, self.max_leverage)  # Use low leverage as safe default
            logger.warning(f"Using fallback position size: ${default_position:.2f} with leverage: {default_leverage:.1f}x")
            return default_position, default_leverage

    def calculate_hedge_parameters(
        self,
        main_position_size: float,
        main_leverage: float,
        hedge_ratio: float,
        max_hedge_leverage: Optional[float] = None,
        strategy_name: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Calculate hedge position size and leverage.

        Args:
            main_position_size: Size of the main position
            main_leverage: Leverage of the main position
            hedge_ratio: Ratio of hedge to main position (0-1)
            max_hedge_leverage: Maximum leverage for the hedge position
            strategy_name: Optional strategy name for strategy-specific hedge logic

        Returns:
            Tuple of (hedge_position_size, hedge_leverage)
        """
        if hedge_ratio < 0 or hedge_ratio > 1:
            logger.warning(f"Invalid hedge ratio {hedge_ratio}, using default of 0.2")
            hedge_ratio = 0.2

        # Calculate main position's notional value
        main_notional = main_position_size * main_leverage

        # Determine hedge notional based on ratio
        hedge_notional = main_notional * hedge_ratio

        # Calculate hedge leverage, defaulting to same as main if not specified
        if max_hedge_leverage is None:
            hedge_leverage = main_leverage
        else:
            hedge_leverage = min(main_leverage, max_hedge_leverage, self.max_leverage)

        # Calculate hedge position size based on leverage
        hedge_position_size = hedge_notional / hedge_leverage

        if strategy_name:
            logger.debug(f"Calculated hedge parameters for strategy '{strategy_name}': size={hedge_position_size:.2f}, leverage={hedge_leverage:.1f}x")

        return hedge_position_size, hedge_leverage

    def validate_trade(
        self,
        exchange: BaseConnector,
        symbol: str,
        position_size: float,
        leverage: float,
        side: OrderSide,
    ) -> Tuple[bool, str]:
        """
        Validate if a trade meets risk criteria.

        Args:
            exchange: Exchange connector
            symbol: Market symbol (e.g., 'ETH')
            position_size: Position size (margin) in USD
            leverage: Position leverage
            side: Trade direction (BUY or SELL)

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check if we're already at max positions
        if len(self.positions) >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) already reached"

        # Fetch account balances
        try:
            balances = exchange.get_account_balance()
            available_balance = sum(balances.values())

            if position_size > available_balance:
                return (
                    False,
                    f"Position size ${position_size:.2f} exceeds available balance ${available_balance:.2f}",
                )
        except Exception as e:
            logger.error(f"Failed to fetch account balance: {e}")
            return False, "Unable to validate trade due to balance fetch error"

        # Check if leverage exceeds limit
        if leverage > self.max_leverage:
            return (
                False,
                f"Requested leverage {leverage:.1f}x exceeds maximum {self.max_leverage:.1f}x",
            )

        # Calculate notional value
        notional_value = position_size * leverage

        # Fetch current price to check max position size
        try:
            ticker = exchange.get_ticker(symbol)
            price = ticker.get("last_price", 0.0)

            # Ensure price is valid
            if price <= 0:
                logger.warning(
                    f"Invalid price for {symbol}: {price}, using dummy price for validation"
                )
                # Use a placeholder price to avoid division by zero
                price = 1.0

            # Fetch market info for min/max position constraints
            markets = exchange.get_markets()
            market_info = next((m for m in markets if m.get("symbol") == symbol), None)

            if market_info:
                min_size = market_info.get("min_size", 0.001)
                # Convert position size to asset units
                base_units = notional_value / price

                if base_units < min_size:
                    return (
                        False,
                        f"Position size {base_units} {symbol} is below minimum {min_size}",
                    )
        except ZeroDivisionError:
            logger.warning(
                f"Division by zero error validating position size for {symbol}, using placeholder value"
            )
            # Skip this validation step but continue
        except Exception as e:
            logger.warning(
                f"Failed to validate position size against market constraints: {e}"
            )
            # Continue validation, this is not critical

        # Check the total exposure including this new position
        new_total_exposure = self.total_exposure + notional_value
        max_exposure = available_balance * self.max_leverage

        if new_total_exposure > max_exposure:
            return (
                False,
                f"New total exposure ${new_total_exposure:.2f} would exceed maximum ${max_exposure:.2f}",
            )

        # All checks passed
        return True, "Trade validated"

    def update_positions(self, positions: List[Dict[str, Any]]) -> None:
        """
        Update the internal positions tracking.

        Args:
            positions: List of position dictionaries from exchange connector
        """
        self.positions = {}
        self.total_exposure = 0.0

        for position in positions:
            symbol = position.get("symbol")
            size = position.get("size", 0.0)

            if not symbol or size == 0:
                continue

            self.positions[symbol] = position
            notional = abs(size) * position.get("mark_price", 0.0)
            self.total_exposure += notional

    def should_close_position(
        self, position: Dict[str, Any], stop_loss_pct: float, take_profit_pct: float
    ) -> Tuple[bool, str]:
        """
        Check if a position should be closed based on risk parameters.

        Args:
            position: Position information dictionary
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage

        Returns:
            Tuple of (should_close, reason)
        """
        symbol = position.get("symbol")
        entry_price = position.get("entry_price", 0.0)
        current_price = position.get("mark_price", 0.0)
        side = position.get("side", "")
        unrealized_pnl_pct = (
            position.get("unrealized_pnl", 0.0) / position.get("margin", 1.0) * 100
        )

        if not symbol or entry_price <= 0 or current_price <= 0:
            return False, "Invalid position data"

        # Check stop loss first
        if unrealized_pnl_pct <= -stop_loss_pct:
            return True, f"Stop loss triggered at {unrealized_pnl_pct:.2f}%"

        # Check take profit
        if unrealized_pnl_pct >= take_profit_pct:
            return True, f"Take profit triggered at {unrealized_pnl_pct:.2f}%"

        # Check if position is approaching liquidation
        liquidation_price = position.get("liquidation_price", 0.0)

        if liquidation_price > 0:
            # For long positions
            if side == "LONG" and current_price < entry_price:
                price_diff_pct = (
                    (current_price - liquidation_price) / current_price * 100
                )
                if price_diff_pct < self.min_margin_buffer_pct:
                    return True, f"Close to liquidation (buffer: {price_diff_pct:.2f}%)"

            # For short positions
            elif side == "SHORT" and current_price > entry_price:
                price_diff_pct = (
                    (liquidation_price - current_price) / current_price * 100
                )
                if price_diff_pct < self.min_margin_buffer_pct:
                    return True, f"Close to liquidation (buffer: {price_diff_pct:.2f}%)"

        # Position is still valid
        return False, "Position within risk parameters"

    def manage_hedge_position(
        self, main_position: Dict[str, Any], hedge_position: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Determine if a hedge position should be adjusted.

        Args:
            main_position: Main position information
            hedge_position: Hedge position information

        Returns:
            Tuple of (should_adjust, reason, adjustment_details)
        """
        # Calculate P&L for both positions
        main_pnl = main_position.get("unrealized_pnl", 0.0)
        hedge_pnl = hedge_position.get("unrealized_pnl", 0.0)

        # Get position details
        main_side = main_position.get("side", "")
        hedge_side = hedge_position.get("side", "")

        # Ensure we have a proper hedge (opposite sides)
        if main_side == hedge_side:
            logger.warning(
                f"Main and hedge positions have the same side ({main_side}), not a proper hedge"
            )
            return False, "Not a valid hedge", {}

        # Check if main position is losing money significantly
        main_pnl_pct = main_pnl / main_position.get("margin", 1.0) * 100
        hedge_pnl_pct = hedge_pnl / hedge_position.get("margin", 1.0) * 100

        # Calculate combined P&L
        combined_pnl = main_pnl + hedge_pnl
        combined_margin = main_position.get("margin", 0.0) + hedge_position.get(
            "margin", 0.0
        )
        combined_pnl_pct = (
            combined_pnl / combined_margin * 100 if combined_margin > 0 else 0
        )

        # Scenario 1: Main position is profitable, hedge is lossy
        # Action: Consider reducing hedge size to let profits run
        if main_pnl > 0 and main_pnl_pct > 10 and hedge_pnl < 0:
            return (
                True,
                "Main position profitable, reduce hedge",
                {
                    "action": "reduce",
                    "position": "hedge",
                    "reduction_pct": min(50, main_pnl_pct),  # Reduce hedge by up to 50%
                },
            )

        # Scenario 2: Main position is losing badly, hedge is profitable
        # Action: Consider closing main and keeping hedge
        if main_pnl < 0 and main_pnl_pct < -15 and hedge_pnl > 0:
            return (
                True,
                "Main position losing badly, close it",
                {"action": "close", "position": "main"},
            )

        # Scenario 3: Both are losing money (rare but possible)
        # Action: Close both to prevent further losses
        if (
            main_pnl < 0
            and hedge_pnl < 0
            and (main_pnl_pct < -10 or hedge_pnl_pct < -10)
        ):
            return (
                True,
                "Both positions losing, close all",
                {"action": "close", "position": "both"},
            )

        # No adjustment needed
        return False, "No adjustment needed", {}

    @classmethod
    def from_config(cls, config: Dict[str, Any], strategies: Optional[List[Any]] = None) -> 'RiskManager':
        """
        Create a RiskManager from configuration dictionary.

        Args:
            config: Configuration dictionary containing risk management and position sizing settings
            strategies: Optional list of strategy configuration objects for strategy-specific position sizing

        Returns:
            RiskManager instance
        """
        # Extract position sizing configuration if available
        position_sizing_config = config.get('position_sizing')

        # Extract risk management parameters
        max_account_risk_pct = config.get('max_account_risk_pct', 2.0)
        max_leverage = config.get('max_leverage', 25.0)
        max_position_size_usd = config.get('max_position_size_usd')
        max_positions = config.get('max_positions', 3)
        min_margin_buffer_pct = config.get('min_margin_buffer_pct', 20.0)

        # If position sizing config exists, merge risk parameters into it
        if position_sizing_config:
            # Merge risk parameters into position sizing config
            merged_config = position_sizing_config.copy()

            # Use position sizing config values, but fallback to risk management values if not specified
            if 'max_position_size_usd' not in merged_config and max_position_size_usd:
                merged_config['max_position_size_usd'] = max_position_size_usd
            if 'max_leverage' not in merged_config:
                merged_config['max_leverage'] = max_leverage
            if 'risk_per_trade_pct' not in merged_config:
                merged_config['risk_per_trade_pct'] = max_account_risk_pct / 100.0

            logger.info("Creating RiskManager with position sizing configuration from config")
            risk_manager = cls(
                max_account_risk_pct=max_account_risk_pct,
                max_leverage=max_leverage,
                max_position_size_usd=max_position_size_usd,
                max_positions=max_positions,
                min_margin_buffer_pct=min_margin_buffer_pct,
                position_sizing_config=merged_config
            )
        else:
            logger.info("Creating RiskManager with default position sizing configuration")
            risk_manager = cls(
                max_account_risk_pct=max_account_risk_pct,
                max_leverage=max_leverage,
                max_position_size_usd=max_position_size_usd,
                max_positions=max_positions,
                min_margin_buffer_pct=min_margin_buffer_pct
            )

        # Create strategy-specific position sizers if strategies are provided
        if strategies:
            risk_manager._create_strategy_position_sizers(strategies)
            logger.info(f"RiskManager created with {len(risk_manager.strategy_position_sizers)} strategy-specific position sizers")
        else:
            logger.info("RiskManager created without strategy-specific position sizing")

        return risk_manager

    def _create_strategy_position_sizers(self, strategies: List[Any]) -> None:
        """
        Create strategy-specific position sizers from strategy configurations.

        Args:
            strategies: List of strategy configuration objects with position_sizing attributes
        """
        logger.info("Creating strategy-specific position sizers...")

        for strategy in strategies:
            if hasattr(strategy, 'position_sizing') and strategy.position_sizing:
                try:
                    strategy_sizer = self._create_position_sizer_for_strategy(strategy)
                    self.strategy_position_sizers[strategy.name] = strategy_sizer
                    logger.info(f"Created position sizer for strategy '{strategy.name}': {strategy_sizer.config.method.value}")
                except Exception as e:
                    logger.error(f"Failed to create position sizer for strategy '{strategy.name}': {e}")
                    logger.info(f"Strategy '{strategy.name}' will use default position sizer")
            else:
                logger.debug(f"Strategy '{strategy.name}' has no specific position sizing config, will use default")

    def _create_position_sizer_for_strategy(self, strategy: Any) -> PositionSizer:
        """
        Create a position sizer for a specific strategy.

        Merges strategy-specific config with global defaults.

        Args:
            strategy: Strategy configuration object

        Returns:
            PositionSizer: Position sizer configured for the strategy
        """
        # Start with default position sizing config as the baseline
        # This ensures strategies inherit global settings unless explicitly overridden
        merged_config = {
            'position_sizing_method': self.position_sizer.config.method.value,
            'max_position_size_usd': self.position_sizer.config.max_position_size_usd,
            'min_position_size_usd': self.position_sizer.config.min_position_size_usd,
            'max_leverage': self.position_sizer.config.max_leverage,
            'risk_per_trade_pct': self.position_sizer.config.risk_per_trade_pct,
        }

        # Add method-specific defaults from global config
        # Each position sizing method has different required parameters
        if self.position_sizer.config.method == PositionSizingMethod.FIXED_USD:
            # Fixed USD method needs the fixed amount parameter
            merged_config['fixed_usd_amount'] = self.position_sizer.config.fixed_usd_amount
        elif self.position_sizer.config.method == PositionSizingMethod.PERCENT_EQUITY:
            # Percentage method needs the equity percentage parameter
            merged_config['percent_equity'] = self.position_sizer.config.percent_equity
        elif self.position_sizer.config.method == PositionSizingMethod.RISK_BASED:
            # Risk-based method uses risk per trade percentage
            merged_config['risk_per_trade_pct'] = self.position_sizer.config.risk_per_trade_pct
        elif self.position_sizer.config.method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            # Volatility adjusted method needs base percentage
            merged_config['base_percent_equity'] = self.position_sizer.config.base_percent_equity

        # Override inheritance: Strategy-specific config takes precedence
        # This allows strategies to customize any aspect of position sizing
        if strategy.position_sizing:
            for key, value in strategy.position_sizing.items():
                merged_config[key] = value
                logger.debug(f"Strategy '{strategy.name}' overrides {key} = {value}")

        # Create and validate the merged configuration
        try:
            # Parse merged config into validated PositionSizingConfig object
            sizing_config = PositionSizingConfig.from_config_dict(merged_config)
            return PositionSizer(sizing_config)
        except Exception as e:
            # If strategy config is invalid, log error and fall back to default
            # This prevents one bad strategy from breaking the entire system
            logger.error(f"Invalid position sizing config for strategy '{strategy.name}': {e}")
            logger.info(f"Using default position sizer for strategy '{strategy.name}'")
            return self.position_sizer
