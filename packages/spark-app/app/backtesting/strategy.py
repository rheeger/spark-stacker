import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..connectors.base_connector import OrderSide, OrderType
from .simulation_engine import SimulationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def simple_moving_average_crossover_strategy(
    data: pd.DataFrame,
    simulation: SimulationEngine,
    params: Dict[str, Any]
) -> None:
    """
    Simple moving average crossover strategy.

    Buy when fast MA crosses above slow MA, sell when fast MA crosses below slow MA.

    Args:
        data: Historical price data
        simulation: Simulation engine instance
        params: Strategy parameters including:
            - symbol: Market symbol
            - current_candle: Current price candle
            - fast_period: Fast moving average period
            - slow_period: Slow moving average period
            - leverage: Leverage multiplier
            - position_size: Position size as percentage of equity (0.0-1.0)
    """
    # Extract parameters
    symbol = params['symbol']
    current_candle = params['current_candle']
    fast_period = params.get('fast_period', 10)
    slow_period = params.get('slow_period', 50)
    leverage = params.get('leverage', 1.0)
    position_size = params.get('position_size', 0.1)  # 10% of equity by default

    # Need at least slow_period + 1 candles to calculate crossover
    if len(data) <= slow_period:
        return

    # Calculate moving averages
    data['fast_ma'] = data['close'].rolling(window=fast_period).mean()
    data['slow_ma'] = data['close'].rolling(window=slow_period).mean()

    # Get current and previous MA values
    current_fast_ma = data['fast_ma'].iloc[-1]
    current_slow_ma = data['slow_ma'].iloc[-1]

    # Check for NaN values
    if pd.isna(current_fast_ma) or pd.isna(current_slow_ma):
        return

    # Get previous MA values (for crossover detection)
    if len(data) > 1:
        prev_fast_ma = data['fast_ma'].iloc[-2]
        prev_slow_ma = data['slow_ma'].iloc[-2]

        # Check for NaN values in previous data point
        if pd.isna(prev_fast_ma) or pd.isna(prev_slow_ma):
            return
    else:
        # Not enough data to detect crossover
        return

    # Check for crossovers
    fast_above_slow = current_fast_ma > current_slow_ma
    prev_fast_above_slow = prev_fast_ma > prev_slow_ma

    # Get current positions for this symbol
    current_positions = simulation.get_positions(symbol)
    has_position = len(current_positions) > 0

    # Calculate position size based on current equity
    current_prices = {symbol: current_candle['close']}
    equity = simulation.calculate_equity(current_prices)
    position_value = equity * position_size

    # Position amount (units of the asset)
    amount = position_value / current_candle['close']

    # Buy signal: Fast MA crosses above Slow MA
    if fast_above_slow and not prev_fast_above_slow:
        logger.info(f"Buy signal detected at {pd.to_datetime(current_candle['timestamp'], unit='ms')}")

        # If we already have a position, don't do anything
        if not has_position:
            # Place a market buy order
            simulation.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=amount,
                leverage=leverage,
                timestamp=current_candle['timestamp'],
                current_candle=current_candle
            )

    # Sell signal: Fast MA crosses below Slow MA
    elif not fast_above_slow and prev_fast_above_slow:
        logger.info(f"Sell signal detected at {pd.to_datetime(current_candle['timestamp'], unit='ms')}")

        # If we have a long position, close it
        if has_position and current_positions[0].side == OrderSide.BUY:
            # Place a market sell order with the same amount as our position
            simulation.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                amount=current_positions[0].amount,
                leverage=leverage,
                timestamp=current_candle['timestamp'],
                current_candle=current_candle
            )


def bollinger_bands_mean_reversion_strategy(
    data: pd.DataFrame,
    simulation: SimulationEngine,
    params: Dict[str, Any]
) -> None:
    """
    Bollinger Bands mean reversion strategy.

    Buy when price touches lower band, sell when price touches upper band.

    Args:
        data: Historical price data
        simulation: Simulation engine instance
        params: Strategy parameters including:
            - symbol: Market symbol
            - current_candle: Current price candle
            - bb_period: Bollinger Bands period
            - bb_std: Number of standard deviations for bands
            - leverage: Leverage multiplier
            - position_size: Position size as percentage of equity (0.0-1.0)
    """
    # Extract parameters
    symbol = params['symbol']
    current_candle = params['current_candle']
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    leverage = params.get('leverage', 1.0)
    position_size = params.get('position_size', 0.1)  # 10% of equity by default

    # Need at least bb_period candles
    if len(data) < bb_period:
        return

    # Calculate Bollinger Bands
    data['sma'] = data['close'].rolling(window=bb_period).mean()
    data['std'] = data['close'].rolling(window=bb_period).std()
    data['upper_band'] = data['sma'] + (data['std'] * bb_std)
    data['lower_band'] = data['sma'] - (data['std'] * bb_std)

    # Get current values
    current_close = current_candle['close']
    current_upper = data['upper_band'].iloc[-1]
    current_lower = data['lower_band'].iloc[-1]

    # Check for NaN values
    if pd.isna(current_upper) or pd.isna(current_lower):
        return

    # Get current positions for this symbol
    current_positions = simulation.get_positions(symbol)
    has_position = len(current_positions) > 0

    # Calculate position size based on current equity
    current_prices = {symbol: current_candle['close']}
    equity = simulation.calculate_equity(current_prices)
    position_value = equity * position_size

    # Position amount (units of the asset)
    amount = position_value / current_candle['close']

    # Buy signal: Price touches or crosses below lower band
    if current_close <= current_lower:
        logger.info(f"Buy signal detected at {pd.to_datetime(current_candle['timestamp'], unit='ms')} - Price below lower band")

        # If we don't have a position, buy
        if not has_position:
            # Place a market buy order
            simulation.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=amount,
                leverage=leverage,
                timestamp=current_candle['timestamp'],
                current_candle=current_candle
            )

    # Sell signal: Price touches or crosses above upper band
    elif current_close >= current_upper:
        logger.info(f"Sell signal detected at {pd.to_datetime(current_candle['timestamp'], unit='ms')} - Price above upper band")

        # If we have a long position, close it
        if has_position and current_positions[0].side == OrderSide.BUY:
            # Place a market sell order with the same amount as our position
            simulation.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                amount=current_positions[0].amount,
                leverage=leverage,
                timestamp=current_candle['timestamp'],
                current_candle=current_candle
            )


def rsi_strategy(
    data: pd.DataFrame,
    simulation: SimulationEngine,
    params: Dict[str, Any]
) -> None:
    """
    RSI (Relative Strength Index) strategy.

    Buy when RSI is below oversold level, sell when RSI is above overbought level.

    Args:
        data: Historical price data
        simulation: Simulation engine instance
        params: Strategy parameters including:
            - symbol: Market symbol
            - current_candle: Current price candle
            - rsi_period: RSI calculation period
            - oversold: RSI oversold level
            - overbought: RSI overbought level
            - leverage: Leverage multiplier
            - position_size: Position size as percentage of equity (0.0-1.0)
    """
    # Extract parameters
    symbol = params['symbol']
    current_candle = params['current_candle']
    rsi_period = params.get('rsi_period', 14)
    oversold = params.get('oversold', 30)
    overbought = params.get('overbought', 70)
    leverage = params.get('leverage', 1.0)
    position_size = params.get('position_size', 0.1)  # 10% of equity by default

    # Need at least rsi_period + 1 candles to calculate RSI
    if len(data) <= rsi_period + 1:
        return

    # Calculate RSI
    # First calculate price changes
    delta = data['close'].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gain and loss
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()

    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss

    # Calculate RSI
    data['rsi'] = 100 - (100 / (1 + rs))

    # Get current RSI value
    current_rsi = data['rsi'].iloc[-1]

    # Get previous RSI value
    if len(data) > 1:
        prev_rsi = data['rsi'].iloc[-2]
    else:
        prev_rsi = None

    # Check for NaN values
    if pd.isna(current_rsi):
        return

    # Get current positions for this symbol
    current_positions = simulation.get_positions(symbol)
    has_position = len(current_positions) > 0

    # Calculate position size based on current equity
    current_prices = {symbol: current_candle['close']}
    equity = simulation.calculate_equity(current_prices)
    position_value = equity * position_size

    # Position amount (units of the asset)
    amount = position_value / current_candle['close']

    # Buy signal: RSI crosses below oversold level
    if current_rsi < oversold and (prev_rsi is None or prev_rsi >= oversold):
        logger.info(f"Buy signal detected at {pd.to_datetime(current_candle['timestamp'], unit='ms')} - RSI below oversold")

        # If we don't have a position, buy
        if not has_position:
            # Place a market buy order
            simulation.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=amount,
                leverage=leverage,
                timestamp=current_candle['timestamp'],
                current_candle=current_candle
            )

    # Sell signal: RSI crosses above overbought level
    elif current_rsi > overbought and (prev_rsi is None or prev_rsi <= overbought):
        logger.info(f"Sell signal detected at {pd.to_datetime(current_candle['timestamp'], unit='ms')} - RSI above overbought")

        # If we have a long position, close it
        if has_position and current_positions[0].side == OrderSide.BUY:
            # Place a market sell order with the same amount as our position
            simulation.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                amount=current_positions[0].amount,
                leverage=leverage,
                timestamp=current_candle['timestamp'],
                current_candle=current_candle
            )


def macd_strategy(
    data: pd.DataFrame,
    simulation: SimulationEngine,
    params: Dict[str, Any]
) -> None:
    """
    MACD (Moving Average Convergence Divergence) strategy.

    Buy when MACD line crosses above signal line, sell when MACD line crosses below signal line.

    Args:
        data: Historical price data
        simulation: Simulation engine instance
        params: Strategy parameters including:
            - symbol: Market symbol
            - current_candle: Current price candle
            - fast_period: Fast EMA period
            - slow_period: Slow EMA period
            - signal_period: Signal EMA period
            - leverage: Leverage multiplier
            - position_size: Position size as percentage of equity (0.0-1.0)
    """
    # Extract parameters
    symbol = params['symbol']
    current_candle = params['current_candle']
    fast_period = params.get('fast_period', 12)
    slow_period = params.get('slow_period', 26)
    signal_period = params.get('signal_period', 9)
    leverage = params.get('leverage', 1.0)
    position_size = params.get('position_size', 0.1)  # 10% of equity by default

    # Need at least slow_period + signal_period candles
    if len(data) < slow_period + signal_period:
        return

    # Calculate MACD
    # Fast EMA
    data['fast_ema'] = data['close'].ewm(span=fast_period, adjust=False).mean()

    # Slow EMA
    data['slow_ema'] = data['close'].ewm(span=slow_period, adjust=False).mean()

    # MACD Line
    data['macd'] = data['fast_ema'] - data['slow_ema']

    # Signal Line
    data['signal'] = data['macd'].ewm(span=signal_period, adjust=False).mean()

    # Histogram
    data['histogram'] = data['macd'] - data['signal']

    # Get current values
    current_macd = data['macd'].iloc[-1]
    current_signal = data['signal'].iloc[-1]

    # Get previous values
    if len(data) > 1:
        prev_macd = data['macd'].iloc[-2]
        prev_signal = data['signal'].iloc[-2]
    else:
        prev_macd = None
        prev_signal = None

    # Check for NaN values
    if pd.isna(current_macd) or pd.isna(current_signal):
        return

    # Get current positions for this symbol
    current_positions = simulation.get_positions(symbol)
    has_position = len(current_positions) > 0

    # Calculate position size based on current equity
    current_prices = {symbol: current_candle['close']}
    equity = simulation.calculate_equity(current_prices)
    position_value = equity * position_size

    # Position amount (units of the asset)
    amount = position_value / current_candle['close']

    # Check for crossovers
    macd_above_signal = current_macd > current_signal
    if prev_macd is not None and prev_signal is not None:
        prev_macd_above_signal = prev_macd > prev_signal
    else:
        prev_macd_above_signal = None

    # Buy signal: MACD crosses above signal line
    if macd_above_signal and prev_macd_above_signal is not None and not prev_macd_above_signal:
        logger.info(f"Buy signal detected at {pd.to_datetime(current_candle['timestamp'], unit='ms')} - MACD crossed above signal")

        # If we don't have a position, buy
        if not has_position:
            # Place a market buy order
            simulation.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=amount,
                leverage=leverage,
                timestamp=current_candle['timestamp'],
                current_candle=current_candle
            )

    # Sell signal: MACD crosses below signal line
    elif not macd_above_signal and prev_macd_above_signal is not None and prev_macd_above_signal:
        logger.info(f"Sell signal detected at {pd.to_datetime(current_candle['timestamp'], unit='ms')} - MACD crossed below signal")

        # If we have a long position, close it
        if has_position and current_positions[0].side == OrderSide.BUY:
            # Place a market sell order with the same amount as our position
            simulation.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                amount=current_positions[0].amount,
                leverage=leverage,
                timestamp=current_candle['timestamp'],
                current_candle=current_candle
            )


def multi_indicator_strategy(
    data: pd.DataFrame,
    simulation: SimulationEngine,
    params: Dict[str, Any]
) -> None:
    """
    Multi-indicator strategy combining RSI, MACD, and Bollinger Bands.

    Buy when at least 2 out of 3 indicators give a buy signal.
    Sell when at least 2 out of 3 indicators give a sell signal.

    Args:
        data: Historical price data
        simulation: Simulation engine instance
        params: Strategy parameters including:
            - symbol: Market symbol
            - current_candle: Current price candle
            - rsi_period: RSI calculation period
            - oversold: RSI oversold level
            - overbought: RSI overbought level
            - fast_period: Fast EMA period for MACD
            - slow_period: Slow EMA period for MACD
            - signal_period: Signal EMA period for MACD
            - bb_period: Bollinger Bands period
            - bb_std: Number of standard deviations for bands
            - leverage: Leverage multiplier
            - position_size: Position size as percentage of equity (0.0-1.0)
    """
    # Extract parameters
    symbol = params['symbol']
    current_candle = params['current_candle']

    # RSI parameters
    rsi_period = params.get('rsi_period', 14)
    oversold = params.get('oversold', 30)
    overbought = params.get('overbought', 70)

    # MACD parameters
    fast_period = params.get('fast_period', 12)
    slow_period = params.get('slow_period', 26)
    signal_period = params.get('signal_period', 9)

    # Bollinger Bands parameters
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)

    # Other parameters
    leverage = params.get('leverage', 1.0)
    position_size = params.get('position_size', 0.1)  # 10% of equity by default

    # Need at least slow_period + signal_period candles (MACD requires the most data)
    if len(data) < slow_period + signal_period:
        return

    # CALCULATE INDICATORS

    # 1. RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # 2. MACD
    data['fast_ema'] = data['close'].ewm(span=fast_period, adjust=False).mean()
    data['slow_ema'] = data['close'].ewm(span=slow_period, adjust=False).mean()
    data['macd'] = data['fast_ema'] - data['slow_ema']
    data['signal'] = data['macd'].ewm(span=signal_period, adjust=False).mean()
    data['histogram'] = data['macd'] - data['signal']

    # 3. Bollinger Bands
    data['sma'] = data['close'].rolling(window=bb_period).mean()
    data['std'] = data['close'].rolling(window=bb_period).std()
    data['upper_band'] = data['sma'] + (data['std'] * bb_std)
    data['lower_band'] = data['sma'] - (data['std'] * bb_std)

    # CHECK FOR SIGNALS

    # Current values
    current_close = current_candle['close']
    current_rsi = data['rsi'].iloc[-1]
    current_macd = data['macd'].iloc[-1]
    current_signal = data['signal'].iloc[-1]
    current_upper = data['upper_band'].iloc[-1]
    current_lower = data['lower_band'].iloc[-1]

    # Previous values
    if len(data) > 1:
        prev_rsi = data['rsi'].iloc[-2]
        prev_macd = data['macd'].iloc[-2]
        prev_signal = data['signal'].iloc[-2]
    else:
        prev_rsi = None
        prev_macd = None
        prev_signal = None

    # Check for NaN values
    if pd.isna(current_rsi) or pd.isna(current_macd) or pd.isna(current_signal) or pd.isna(current_upper) or pd.isna(current_lower):
        return

    # Count buy and sell signals
    buy_signals = 0
    sell_signals = 0

    # RSI Signals
    if current_rsi < oversold and (prev_rsi is None or prev_rsi >= oversold):
        buy_signals += 1
    elif current_rsi > overbought and (prev_rsi is None or prev_rsi <= overbought):
        sell_signals += 1

    # MACD Signals
    macd_above_signal = current_macd > current_signal
    if prev_macd is not None and prev_signal is not None:
        prev_macd_above_signal = prev_macd > prev_signal

        if macd_above_signal and not prev_macd_above_signal:
            buy_signals += 1
        elif not macd_above_signal and prev_macd_above_signal:
            sell_signals += 1

    # Bollinger Bands Signals
    if current_close <= current_lower:
        buy_signals += 1
    elif current_close >= current_upper:
        sell_signals += 1

    # Get current positions for this symbol
    current_positions = simulation.get_positions(symbol)
    has_position = len(current_positions) > 0

    # Calculate position size based on current equity
    current_prices = {symbol: current_candle['close']}
    equity = simulation.calculate_equity(current_prices)
    position_value = equity * position_size

    # Position amount (units of the asset)
    amount = position_value / current_candle['close']

    # Execute trades based on signals
    if buy_signals >= 2:
        logger.info(f"Buy signal detected at {pd.to_datetime(current_candle['timestamp'], unit='ms')} - {buy_signals} buy signals active")

        # If we don't have a position, buy
        if not has_position:
            # Place a market buy order
            simulation.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=amount,
                leverage=leverage,
                timestamp=current_candle['timestamp'],
                current_candle=current_candle
            )

    elif sell_signals >= 2:
        logger.info(f"Sell signal detected at {pd.to_datetime(current_candle['timestamp'], unit='ms')} - {sell_signals} sell signals active")

        # If we have a long position, close it
        if has_position and current_positions[0].side == OrderSide.BUY:
            # Place a market sell order with the same amount as our position
            simulation.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                amount=current_positions[0].amount,
                leverage=leverage,
                timestamp=current_candle['timestamp'],
                current_candle=current_candle
            )
