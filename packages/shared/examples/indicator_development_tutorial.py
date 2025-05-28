#!/usr/bin/env python3
"""
Indicator Development Tutorial for Spark Stacker

This tutorial demonstrates how to develop a custom technical indicator using the
actual Spark Stacker framework, including calculation logic, signal generation,
testing, and integration with the trading system.

We'll create a Bollinger Bands indicator as a comprehensive example using the
real BaseIndicator interface.

Author: Spark Stacker Development Team
Usage: python indicator_development_tutorial.py
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the spark-app directory to Python path so we can import from the real codebase
spark_app_path = os.path.join(os.path.dirname(__file__), '..', '..', 'spark-app')
sys.path.insert(0, spark_app_path)

# Import from the REAL Spark Stacker codebase
from app.indicators.base_indicator import (BaseIndicator, Signal,
                                           SignalDirection)
from app.indicators.indicator_factory import IndicatorFactory

# Setup logging for the tutorial
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TutorialBollingerBandsIndicator(BaseIndicator):
    """
    Tutorial Bollinger Bands indicator implementation using the real BaseIndicator.

    This demonstrates how to create a custom indicator that integrates with the
    actual Spark Stacker framework. Bollinger Bands consist of:
    - Middle Band: Simple Moving Average (SMA)
    - Upper Band: SMA + (2 * Standard Deviation)
    - Lower Band: SMA - (2 * Standard Deviation)

    Signals are generated when price touches or crosses the bands:
    - Buy signal: Price touches lower band (oversold)
    - Sell signal: Price touches upper band (overbought)
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Bollinger Bands indicator using the real BaseIndicator.

        Args:
            name: Indicator name
            params: Parameters including:
                - period: Period for SMA calculation (default: 20)
                - std_dev_multiplier: Standard deviation multiplier (default: 2.0)
                - price_column: Column to use for calculation (default: 'close')
        """
        super().__init__(name, params)

        # Extract and validate parameters
        self.period = self.params.get("period", 20)
        self.std_dev_multiplier = self.params.get("std_dev_multiplier", 2.0)
        self.price_column = self.params.get("price_column", "close")

        # Validation
        if self.period < 2:
            raise ValueError("Period must be at least 2")
        if self.std_dev_multiplier <= 0:
            raise ValueError("Standard deviation multiplier must be positive")

        # Update params with actual values being used (including defaults)
        self.params.update({
            "period": self.period,
            "std_dev_multiplier": self.std_dev_multiplier,
            "price_column": self.price_column
        })

        logger.info(f"Tutorial Bollinger Bands configured: period={self.period}, "
                   f"std_dev={self.std_dev_multiplier}, price_column={self.price_column}")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands values using the real BaseIndicator interface.

        Args:
            data: Price data DataFrame

        Returns:
            DataFrame with Bollinger Bands columns added:
            - bb_middle: Middle band (SMA)
            - bb_upper: Upper band
            - bb_lower: Lower band
            - bb_width: Band width (upper - lower)
            - bb_percent: %B indicator (position within bands)
        """
        # Validate required columns
        required_columns = [self.price_column]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"DataFrame must contain a '{col}' column")

        if len(data) < self.period:
            logger.warning(
                f"Not enough data points for Bollinger Bands calculation. "
                f"Need at least {self.period}, got {len(data)}"
            )
            # Add empty columns
            for col in ["bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_percent"]:
                data[col] = np.nan
            return data

        # Create a copy to avoid modifying the original
        df = data.copy()

        # Calculate the middle band (Simple Moving Average)
        df["bb_middle"] = df[self.price_column].rolling(window=self.period).mean()

        # Calculate the standard deviation
        rolling_std = df[self.price_column].rolling(window=self.period).std()

        # Calculate upper and lower bands
        df["bb_upper"] = df["bb_middle"] + (self.std_dev_multiplier * rolling_std)
        df["bb_lower"] = df["bb_middle"] - (self.std_dev_multiplier * rolling_std)

        # Calculate band width (volatility measure)
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]

        # Calculate %B (position within bands)
        # %B = (Price - Lower Band) / (Upper Band - Lower Band)
        df["bb_percent"] = (df[self.price_column] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        logger.debug(f"Calculated Bollinger Bands for {len(df)} data points")
        return df

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate trading signals using the real Signal class from the framework.

        Signal Logic:
        - BUY: Price touches or goes below lower band (%B <= 0.05)
        - SELL: Price touches or goes above upper band (%B >= 0.95)
        - Higher confidence when %B is more extreme

        Args:
            data: DataFrame with Bollinger Bands values

        Returns:
            Real Signal object if conditions are met, None otherwise
        """
        required_columns = ["bb_percent", "bb_upper", "bb_lower", self.price_column]
        for col in required_columns:
            if col not in data.columns:
                logger.warning(f"No {col} column in data, cannot generate signal")
                return None

        if len(data) < 2:
            logger.warning("Not enough data points to generate a signal")
            return None

        # Get the latest data point
        latest = data.iloc[-1]

        # Skip if we have NaN values
        if pd.isna(latest["bb_percent"]):
            return None

        symbol = latest.get("symbol", "UNKNOWN")
        current_price = latest[self.price_column]
        bb_percent = latest["bb_percent"]

        # Define thresholds for signal generation
        oversold_threshold = 0.05  # Below 5% of band range
        overbought_threshold = 0.95  # Above 95% of band range

        # Generate BUY signal when price is near lower band
        if bb_percent <= oversold_threshold:
            # Higher confidence the lower the %B value
            confidence = max(0.5, 1.0 - (bb_percent / oversold_threshold)) * 0.8

            logger.info(f"BUY signal generated: {symbol} at {current_price}, %B={bb_percent:.3f}")

            return Signal(
                direction=SignalDirection.BUY,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                timestamp=latest.get("timestamp"),
                params={
                    "bb_percent": bb_percent,
                    "price": current_price,
                    "bb_lower": latest["bb_lower"],
                    "trigger": "oversold"
                }
            )

        # Generate SELL signal when price is near upper band
        elif bb_percent >= overbought_threshold:
            # Higher confidence the higher the %B value
            confidence = max(0.5, (bb_percent - overbought_threshold) / (1.0 - overbought_threshold)) * 0.8

            logger.info(f"SELL signal generated: {symbol} at {current_price}, %B={bb_percent:.3f}")

            return Signal(
                direction=SignalDirection.SELL,
                symbol=symbol,
                indicator=self.name,
                confidence=confidence,
                timestamp=latest.get("timestamp"),
                params={
                    "bb_percent": bb_percent,
                    "price": current_price,
                    "bb_upper": latest["bb_upper"],
                    "trigger": "overbought"
                }
            )

        # No signal generated
        return None

    def __str__(self) -> str:
        """String representation of the indicator"""
        return f"TutorialBollingerBands(period={self.period}, std_dev={self.std_dev_multiplier})"


# Tutorial helper functions using real framework components

def generate_sample_data(symbol: str = "ETH-USD", days: int = 100) -> pd.DataFrame:
    """
    Generate sample price data for testing indicators.
    This creates data compatible with the real Spark Stacker data format.

    Args:
        symbol: Trading symbol
        days: Number of days of data to generate

    Returns:
        DataFrame with realistic price data in Spark Stacker format
    """
    # Create timestamp range
    timestamps = pd.date_range(
        start="2023-01-01",
        periods=days * 24,  # Hourly data
        freq="1H"
    )

    # Generate realistic price data using random walk with trend
    np.random.seed(42)  # For reproducible results

    base_price = 1500.0  # Starting price for ETH
    returns = np.random.normal(0.0001, 0.02, len(timestamps))  # Small upward bias with volatility

    # Add some trend and volatility clustering
    trend = np.sin(np.arange(len(timestamps)) * 2 * np.pi / (24 * 7)) * 0.001  # Weekly cycle
    volatility = 0.015 + 0.01 * np.sin(np.arange(len(timestamps)) * 2 * np.pi / (24 * 30))  # Monthly volatility cycle

    returns = returns + trend
    returns = returns * volatility

    # Calculate prices
    log_prices = np.log(base_price) + np.cumsum(returns)
    prices = np.exp(log_prices)

    # Generate OHLCV data in the format expected by Spark Stacker
    df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": symbol,
        "close": prices
    })

    # Generate open, high, low based on close
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])

    # High and low with some random variation
    high_low_range = df["close"] * 0.01  # 1% range
    df["high"] = df["close"] + np.random.uniform(0, 1, len(df)) * high_low_range
    df["low"] = df["close"] - np.random.uniform(0, 1, len(df)) * high_low_range

    # Ensure OHLC relationship is valid
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    # Generate volume
    df["volume"] = np.random.lognormal(mean=10, sigma=1, size=len(df))

    logger.info(f"Generated {len(df)} data points for {symbol}")
    return df


def test_with_real_indicators():
    """
    Test our tutorial indicator alongside real indicators from the framework.
    This demonstrates how indicators integrate with the real system.
    """
    logger.info("Testing tutorial indicator with real framework indicators...")

    # Create sample data
    data = generate_sample_data("ETH-USD", days=30)

    # Create our tutorial indicator
    tutorial_bb = TutorialBollingerBandsIndicator(
        name="tutorial_bollinger",
        params={"period": 20, "std_dev_multiplier": 2.0}
    )

    # Create real indicators from the framework for comparison
    try:
        real_rsi = IndicatorFactory.create_indicator(
            name="comparison_rsi",
            indicator_type="rsi",
            params={"period": 14}
        )

        real_bb = IndicatorFactory.create_indicator(
            name="comparison_bollinger",
            indicator_type="bollinger",
            params={"period": 20, "std_dev": 2.0}
        )

        # Test our tutorial indicator
        processed_data, tutorial_signal = tutorial_bb.process(data)
        logger.info(f"Tutorial Bollinger Bands: {len(processed_data)} data points processed")
        if tutorial_signal:
            logger.info(f"Tutorial signal: {tutorial_signal.direction.value} with confidence {tutorial_signal.confidence:.3f}")

        # Test real indicators
        if real_rsi:
            rsi_data, rsi_signal = real_rsi.process(data)
            logger.info(f"Real RSI: {len(rsi_data)} data points processed")
            if rsi_signal:
                logger.info(f"RSI signal: {rsi_signal.direction.value} with confidence {rsi_signal.confidence:.3f}")

        if real_bb:
            bb_data, bb_signal = real_bb.process(data)
            logger.info(f"Real Bollinger Bands: {len(bb_data)} data points processed")
            if bb_signal:
                logger.info(f"Real BB signal: {bb_signal.direction.value} with confidence {bb_signal.confidence:.3f}")

        return {
            "tutorial_indicator": tutorial_bb,
            "real_indicators": [real_rsi, real_bb],
            "data": processed_data,
            "signals": [tutorial_signal, rsi_signal, bb_signal]
        }

    except Exception as e:
        logger.error(f"Error testing with real indicators: {e}")
        logger.info("This might happen if the full Spark Stacker environment isn't available")
        return None


def register_tutorial_indicator():
    """
    Demonstrate how to register a custom indicator with the real IndicatorFactory.
    """
    logger.info("Registering tutorial indicator with IndicatorFactory...")

    try:
        # Register our tutorial indicator
        IndicatorFactory.register_indicator(
            indicator_type="tutorial_bollinger",
            indicator_class=TutorialBollingerBandsIndicator
        )

        # Test creating it through the factory
        factory_indicator = IndicatorFactory.create_indicator(
            name="factory_tutorial_bb",
            indicator_type="tutorial_bollinger",
            params={"period": 15, "std_dev_multiplier": 1.8}
        )

        if factory_indicator:
            logger.info(f"✓ Successfully created indicator through factory: {factory_indicator}")
            return factory_indicator
        else:
            logger.error("✗ Failed to create indicator through factory")
            return None

    except Exception as e:
        logger.error(f"Error registering indicator: {e}")
        return None


def main():
    """
    Main tutorial function demonstrating indicator development with the real framework.
    """
    print("=" * 60)
    print("Spark Stacker Indicator Development Tutorial")
    print("Using REAL Framework Components")
    print("=" * 60)

    # Step 1: Generate sample data
    print("\n1. Generating sample market data...")
    sample_data = generate_sample_data("ETH-USD", days=30)
    print(f"Generated {len(sample_data)} data points")
    print("Sample data columns:", list(sample_data.columns))
    print(sample_data.head())

    # Step 2: Create tutorial indicator using real BaseIndicator
    print("\n2. Creating tutorial Bollinger Bands indicator using real BaseIndicator...")
    tutorial_indicator = TutorialBollingerBandsIndicator(
        name="tutorial_bollinger_bands",
        params={
            "period": 20,
            "std_dev_multiplier": 2.0,
            "price_column": "close"
        }
    )
    print(f"Created indicator: {tutorial_indicator}")

    # Step 3: Test basic functionality with real framework
    print("\n3. Testing with real framework...")
    try:
        processed_data, signal = tutorial_indicator.process(sample_data)
        print("✓ Calculation successful using real BaseIndicator")
        print(f"Added columns: {[col for col in processed_data.columns if col.startswith('bb_')]}")

        if signal:
            print(f"✓ Real Signal generated: {signal.direction.value} with confidence {signal.confidence:.3f}")
            print(f"Signal params: {signal.params}")
        else:
            print("ℹ No signal generated for final data point")

    except Exception as e:
        print(f"✗ Error in calculation: {e}")
        return

    # Step 4: Test with real indicators from the framework
    print("\n4. Testing alongside real framework indicators...")
    comparison_results = test_with_real_indicators()

    # Step 5: Register with IndicatorFactory
    print("\n5. Registering with real IndicatorFactory...")
    factory_indicator = register_tutorial_indicator()

    if factory_indicator:
        print("✓ Successfully integrated with IndicatorFactory")
        print("Available indicators:", IndicatorFactory.get_available_indicators())

    # Step 6: Integration example
    print("\n6. Real integration example...")
    print("Example configuration for using with the real trading system:")

    config_example = {
        "name": "my_tutorial_bollinger",
        "type": "tutorial_bollinger",
        "enabled": True,
        "parameters": {
            "period": 20,
            "std_dev_multiplier": 2.0,
            "price_column": "close"
        }
    }

    print(f"Config: {config_example}")

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)
    print("\nKey Achievements:")
    print("✓ Created indicator using real BaseIndicator interface")
    print("✓ Generated real Signal objects")
    print("✓ Integrated with real IndicatorFactory")
    print("✓ Tested alongside real framework indicators")
    print("\nNext Steps:")
    print("1. Move your indicator to app/indicators/ for production use")
    print("2. Add comprehensive tests in tests/indicators/unit/")
    print("3. Register in IndicatorFactory.register_defaults()")
    print("4. Use with real BacktestEngine for validation")
    print("5. Deploy in live trading strategies")


if __name__ == "__main__":
    main()
