from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd


def make_price_dataframe(rows: int = 100, pattern: str = "trend", noise: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """
    Creates a synthetic price dataframe with configurable patterns for testing indicators.

    Args:
        rows: Number of rows (candles) to generate
        pattern: The price pattern to generate. One of:
            - "trend": Upward trending market with some fluctuations
            - "mean_revert": Price oscillates around a mean value
            - "sideways": Flat market with small fluctuations
        noise: Amount of randomness to add (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame: A DataFrame with OHLCV price data with columns:
            - timestamp: Unix timestamp in milliseconds
            - open: Opening price
            - high: Highest price during the period
            - low: Lowest price during the period
            - close: Closing price
            - volume: Trading volume
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Create a date range
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(rows)]
    timestamps = [int(date.timestamp() * 1000) for date in dates]

    # Generate base close prices according to the selected pattern
    closes = generate_pattern(rows, pattern, noise)

    # Ensure all prices are positive
    closes = [max(price, 1.0) for price in closes]

    # Generate realistic OHLC based on close prices
    opens, highs, lows = generate_ohlc(closes, noise)

    # Generate volume data with some correlation to price changes
    volumes = generate_volumes(closes, rows, noise)

    # Create the DataFrame
    data = {
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    }

    return pd.DataFrame(data)

def generate_pattern(rows: int, pattern: str, noise: float) -> List[float]:
    """Generate price data based on the specified pattern."""
    closes = []
    base_price = 100.0

    if pattern == "trend":
        # Upward trending market
        for i in range(rows):
            # Positive drift with noise
            drift = 0.1  # Daily upward drift
            random_component = np.random.normal(0, noise)
            change = drift + random_component
            base_price = base_price * (1 + change / 100)
            closes.append(base_price)

    elif pattern == "mean_revert":
        # Mean-reverting market
        mean_price = 100.0
        for i in range(rows):
            # Mean reversion with noise
            reversion_strength = 0.05  # Mean reversion factor
            distance_from_mean = mean_price - base_price
            reversion = distance_from_mean * reversion_strength
            random_component = np.random.normal(0, noise)
            change = reversion + random_component
            base_price = base_price * (1 + change / 100)
            closes.append(base_price)

    elif pattern == "sideways":
        # Sideways market
        for i in range(rows):
            # Pure noise around the base price
            random_component = np.random.normal(0, noise / 2)  # Lower noise for sideways
            base_price = 100.0 * (1 + random_component / 100)
            closes.append(base_price)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return closes

def generate_ohlc(closes: List[float], noise: float) -> Tuple[List[float], List[float], List[float]]:
    """Generate open, high, low prices based on close prices."""
    opens = []
    highs = []
    lows = []

    prev_close = closes[0]  # For the first candle

    for i, close in enumerate(closes):
        # Open price typically around previous close
        if i == 0:
            # First candle open is close with small random adjustment
            open_price = close * (1 + np.random.normal(0, noise / 10))
        else:
            # Subsequent opens are based on previous close
            open_price = prev_close * (1 + np.random.normal(0, noise / 10))

        # Determine candle direction
        is_bullish = close > open_price

        # Calculate price range based on volatility (represented by noise)
        price_range = abs(close - open_price) + (close * noise / 50)

        if is_bullish:
            # Bullish candle: high above close, low below open
            high_price = close + np.random.uniform(0, price_range / 2)
            low_price = open_price - np.random.uniform(0, price_range / 2)
        else:
            # Bearish candle: high above open, low below close
            high_price = open_price + np.random.uniform(0, price_range / 2)
            low_price = close - np.random.uniform(0, price_range / 2)

        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)

        prev_close = close

    return opens, highs, lows

def generate_volumes(closes: List[float], rows: int, noise: float) -> List[float]:
    """Generate trading volume with some correlation to price movements."""
    volumes = []
    base_volume = 10000.0

    for i in range(1, rows):
        if i == 0:
            # First candle has base volume with noise
            volume = base_volume * (1 + np.random.normal(0, noise))
        else:
            # Volume often increases with larger price changes
            price_change_pct = abs((closes[i] - closes[i-1]) / closes[i-1])
            volume_change = 1.0 + (price_change_pct * 10 * noise) + np.random.normal(0, noise)
            volume = base_volume * volume_change

        # Ensure volume is positive
        volumes.append(max(volume, 100.0))

    # Handle the case for the first element if rows > 0
    if rows > 0:
        volumes.insert(0, base_volume * (1 + np.random.normal(0, noise)))

    return volumes
