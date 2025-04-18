import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from connectors.base_connector import BaseConnector, OrderSide

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk management system to control position size, leverage, and risk.

    This class handles risk-related calculations and validations before placing trades.
    It helps protect the principal by limiting exposure and implementing smart risk controls.
    """

    def __init__(
        self,
        max_account_risk_pct: float = 2.0,
        max_leverage: float = 25.0,
        max_position_size_usd: Optional[float] = None,
        max_positions: int = 3,
        min_margin_buffer_pct: float = 20.0,
    ):
        """
        Initialize the risk manager.

        Args:
            max_account_risk_pct: Maximum percentage of account to risk per trade
            max_leverage: Maximum allowed leverage
            max_position_size_usd: Maximum position size in USD (None for no limit)
            max_positions: Maximum number of concurrent positions
            min_margin_buffer_pct: Minimum margin buffer percentage
        """
        self.max_account_risk_pct = max_account_risk_pct
        self.max_leverage = max_leverage
        self.max_position_size_usd = max_position_size_usd
        self.max_positions = max_positions
        self.min_margin_buffer_pct = min_margin_buffer_pct

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
    ) -> Tuple[float, float]:
        """
        Calculate position size based on risk parameters.

        Args:
            exchange: Exchange connector
            symbol: Market symbol (e.g., 'ETH')
            available_balance: Available balance (either as float or dictionary of balances)
            confidence: Signal confidence (0-1)
            signal_side: Order side (BUY/SELL)
            leverage: Initial leverage
            price: Optional price override
            stop_loss_pct: Stop loss percentage

        Returns:
            Tuple of (position_size, adjusted_leverage)
        """
        # Handle balance input - convert dictionary to total if needed
        try:
            if isinstance(available_balance, dict):
                total_balance = sum(float(val) for val in available_balance.values())
                logger.debug(f"Converted balance dictionary {available_balance} to total: {total_balance}")
            else:
                total_balance = float(available_balance)
                logger.debug(f"Using direct balance value: {total_balance}")

            if total_balance <= 0:
                logger.warning("Available balance is zero or negative")
                return 0.0, 1.0
        except (TypeError, ValueError) as e:
            logger.error(f"Error processing available balance: {e}")
            return 0.0, 1.0

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
            adjusted_leverage = min(leverage, exchange_max_leverage, self.max_leverage)
            logger.debug(f"Adjusted leverage to {adjusted_leverage} based on limits")

        except Exception as e:
            logger.error(f"Unexpected error determining leverage for {symbol}: {e}")
            adjusted_leverage = min(leverage, self.max_leverage)
            logger.debug(f"Using fallback leverage: {adjusted_leverage}")

        # Calculate max position size based on risk percentage
        try:
            risk_amount = total_balance * (self.max_account_risk_pct / 100.0)
            logger.debug(f"Risk amount: {risk_amount} (from {self.max_account_risk_pct}% of {total_balance})")

            # Adjust by confidence
            # Higher confidence = use more of the available risk amount
            confidence_adjusted_risk = risk_amount * (0.5 + confidence * 0.5)
            logger.debug(f"Confidence-adjusted risk: {confidence_adjusted_risk} (confidence: {confidence})")

            # Calculate position size from risk amount and leverage
            # If we have a stop loss percentage, use that to determine max position size
            if stop_loss_pct and stop_loss_pct > 0:
                # Determine how much we can lose before hitting the stop loss
                max_loss_pct = stop_loss_pct / 100.0
                # Calculate position size based on risk amount and potential loss
                position_size = confidence_adjusted_risk / (max_loss_pct / adjusted_leverage)
                logger.debug(f"Using stop-loss calculation: position_size = {position_size}")
            else:
                # Without specific stop-loss, use a more conservative approach
                position_size = confidence_adjusted_risk * adjusted_leverage
                logger.debug(f"Using conservative calculation: position_size = {position_size}")

            # Apply absolute position size limit if configured
            if self.max_position_size_usd and position_size > self.max_position_size_usd:
                logger.debug(f"Reducing position size from {position_size} to max limit: {self.max_position_size_usd}")
                position_size = self.max_position_size_usd

            logger.info(f"Final position calculation for {symbol}: size={position_size:.2f}, leverage={adjusted_leverage:.1f}x")
            return position_size, adjusted_leverage

        except Exception as e:
            logger.error(f"Error in position size calculation for {symbol}: {e}", exc_info=True)
            # Return conservative values
            default_position = total_balance * 0.05  # Use 5% of balance as safe default
            default_leverage = min(2.0, self.max_leverage)  # Use low leverage as safe default
            logger.warning(f"Using fallback position size: {default_position:.2f} with leverage: {default_leverage:.1f}x")
            return default_position, default_leverage

    def calculate_hedge_parameters(
        self,
        main_position_size: float,
        main_leverage: float,
        hedge_ratio: float,
        max_hedge_leverage: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculate hedge position size and leverage.

        Args:
            main_position_size: Size of the main position
            main_leverage: Leverage of the main position
            hedge_ratio: Ratio of hedge to main position (0-1)
            max_hedge_leverage: Maximum leverage for the hedge position

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
