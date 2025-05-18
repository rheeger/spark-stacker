"""
Integration test that demonstrates using real market data with trading indicators.

This test:
1. Uses cached market data from the test_data directory
2. Applies MACD indicator to generate trading signals
3. Mocks the trading engine to verify signal processing
"""

import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from app.connectors.base_connector import MarketType, OrderSide, OrderType
from app.core.trading_engine import TradingEngine
from app.indicators.base_indicator import Signal, SignalDirection
from app.indicators.macd_indicator import MACDIndicator
from app.risk_management.risk_manager import RiskManager
from tests.conftest import get_cached_market_data

# Directory to save visualization results
TEST_RESULTS_DIR = Path(__file__).parent.parent / "test_results"
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class TestRealMarketDataUsage:
    """Test using real market data with trading logic."""

    def setup_method(self):
        """Setup method that runs before each test."""
        # Create mock connectors
        self.mock_main_connector = self._create_mock_connector("hyperliquid")
        self.mock_hedge_connector = self._create_mock_connector("coinbase")

        # Create risk manager
        self.risk_manager = MagicMock(spec=RiskManager)
        self.risk_manager.calculate_position_size.return_value = (
            1.0,
            5.0,
        )  # (size, leverage)
        self.risk_manager.calculate_hedge_parameters.return_value = (
            0.2,
            2.0,
        )  # (size, leverage)
        self.risk_manager.validate_trade.return_value = (True, "Trade validated")

        # Create trading engine
        self.engine = TradingEngine(
            main_connector=self.mock_main_connector,
            hedge_connector=self.mock_hedge_connector,
            risk_manager=self.risk_manager,
            dry_run=False,  # Set to False to ensure actual execution
            polling_interval=1,
            max_parallel_trades=1,
        )

        # Create MACD indicator with correct parameters format
        self.macd = MACDIndicator(
            name="MACD",
            params={"fast_period": 12, "slow_period": 26, "signal_period": 9},
        )

        # Start the trading engine
        self.engine.start()

    def teardown_method(self):
        """Teardown method that runs after each test."""
        self.engine.stop()

    def _create_mock_connector(self, name):
        """Create a mock connector with necessary methods for testing."""
        connector = MagicMock()
        connector.name = name
        connector.is_connected = True
        connector.supports_derivatives = True

        # Setup market types
        connector.market_types = (
            MarketType.PERPETUAL if name == "hyperliquid" else MarketType.SPOT
        )

        # Define account data
        connector.get_account_balance.return_value = {"USD": 10000.0, "ETH": 5.0}

        # Setup place_order to simulate successful order execution
        connector.place_order = AsyncMock(return_value={
            "order_id": f"{name}_order_123",
            "symbol": "ETH-USD",
            "side": "BUY",
            "size": 1.0,
            "price": 2000.0,
            "status": "FILLED",
        })

        return connector

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_macd_signal_from_real_market_data(self):
        """
        Test generating MACD signals from real market data and
        processing them with the trading engine.
        """
        # Get test market data from cache
        try:
            # Try to get real cached data
            market_data = get_cached_market_data(
                exchange_name="hyperliquid", symbol="ETH-USD", timeframe="1h"
            )
        except FileNotFoundError:
            # Fall back to generating synthetic data
            market_data = self._generate_test_data()

        # Print data info
        print(
            f"\nUsing market data: {len(market_data)} rows, "
            f"date range {market_data.index[0]} to {market_data.index[-1]}"
        )

        # Calculate MACD values and look for signals using the process method
        processed_data, signal = self.macd.process(market_data)

        # If this specific data point didn't generate a signal, process each row
        # to find all signals
        all_signals = []
        for i in range(len(processed_data)):
            # Get data up to this point
            row_data = processed_data.iloc[: i + 1]
            # Generate signal
            row_signal = self.macd.generate_signal(row_data)
            if row_signal:
                all_signals.append((row_data.index[i], row_signal))

        # Print signal info
        print(f"Found {len(all_signals)} signals in the data")

        # Check if we have any signals
        if len(all_signals) == 0:
            # If no signals were found in the real data, create a synthetic signal for testing
            print("Creating synthetic signal for testing")
            last_row = processed_data.iloc[-1]
            signal = Signal(
                direction=SignalDirection.BUY,  # Let's assume a BUY signal
                symbol="ETH-USD",
                indicator="MACD",
                confidence=0.8,
                timestamp=int(last_row.name.timestamp() * 1000),
                params={
                    "macd": last_row["macd"],
                    "signal": last_row["macd_signal"],
                    "histogram": last_row["macd_histogram"]
                    if "macd_histogram" in last_row
                    else 0,
                },
            )
        else:
            # Use the most recent signal
            timestamp, signal = all_signals[-1]
            print(f"Using {signal.direction} signal from {timestamp}")

        # Mock the current market price based on the last data point
        self.mock_main_connector.get_ticker.return_value = {
            "symbol": "ETH-USD",
            "last_price": market_data["close"].iloc[-1],
            "bid": market_data["close"].iloc[-1] * 0.999,
            "ask": market_data["close"].iloc[-1] * 1.001,
            "volume": market_data["volume"].iloc[-1],
        }
        self.mock_main_connector.place_order = AsyncMock(return_value={
            "status": "FILLED",
            "order_id": "test_order_123",
            "symbol": "ETH-USD",
            "side": "BUY",
            "size": 1.0,
            "price": market_data["close"].iloc[-1],
            "timestamp": int(time.time() * 1000)
        })

        # Process the signal
        result = await self.engine.process_signal(signal)

        # Verify signal processing
        assert result is True, "Signal processing should succeed"

        # Create and save visualization
        self._visualize_macd_data(
            market_data, processed_data, all_signals, "macd_signals_from_real_data"
        )

    def _generate_test_data(self):
        """Generate synthetic price data for testing."""
        print("Using synthetic data as real market data is not available")

        # Create 100 days of hourly data
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100 * 24, freq="1H")

        # Generate random walk for close prices
        np.random.seed(42)  # For reproducibility
        random_walk = np.random.normal(0, 50, len(dates)).cumsum()
        close_prices = 2000 + random_walk

        # Ensure no negative prices
        close_prices = np.maximum(close_prices, 10)

        # Generate OHLCV data
        data = {
            "symbol": "ETH-USD",
            "open": close_prices * (1 + np.random.normal(0, 0.005, len(dates))),
            "high": close_prices * (1 + np.random.uniform(0.001, 0.02, len(dates))),
            "low": close_prices * (1 - np.random.uniform(0.001, 0.02, len(dates))),
            "close": close_prices,
            "volume": np.random.normal(1000, 200, len(dates)),
        }

        # Ensure high is always the highest and low is always the lowest
        df = pd.DataFrame(data, index=dates)
        for i in range(len(df)):
            df.loc[df.index[i], "high"] = max(
                df.loc[df.index[i], "open"],
                df.loc[df.index[i], "close"],
                df.loc[df.index[i], "high"],
            )
            df.loc[df.index[i], "low"] = min(
                df.loc[df.index[i], "open"],
                df.loc[df.index[i], "close"],
                df.loc[df.index[i], "low"],
            )

        return df

    def _visualize_macd_data(self, price_data, macd_data, signals, filename):
        """Create and save a visualization of the price data and MACD signals."""
        try:
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 2]}
            )

            # Plot price chart
            ax1.plot(
                price_data.index, price_data["close"], color="blue", label="Close Price"
            )

            # Plot buy and sell signals on price chart
            buy_signals = [
                (t, s) for t, s in signals if s.direction == SignalDirection.BUY
            ]
            sell_signals = [
                (t, s) for t, s in signals if s.direction == SignalDirection.SELL
            ]

            if buy_signals:
                buy_timestamps = [t for t, _ in buy_signals]
                buy_prices = price_data.loc[buy_timestamps]["close"]
                ax1.scatter(
                    buy_timestamps,
                    buy_prices,
                    color="green",
                    label="Buy Signal",
                    marker="^",
                    s=100,
                )

            if sell_signals:
                sell_timestamps = [t for t, _ in sell_signals]
                sell_prices = price_data.loc[sell_timestamps]["close"]
                ax1.scatter(
                    sell_timestamps,
                    sell_prices,
                    color="red",
                    label="Sell Signal",
                    marker="v",
                    s=100,
                )

            ax1.set_title("ETH-USD Price with MACD Signals")
            ax1.set_ylabel("Price")
            ax1.legend()
            ax1.grid(True)

            # Plot MACD indicator
            if "macd" in macd_data.columns and "macd_signal" in macd_data.columns:
                ax2.plot(
                    macd_data.index, macd_data["macd"], color="blue", label="MACD Line"
                )
                ax2.plot(
                    macd_data.index,
                    macd_data["macd_signal"],
                    color="red",
                    label="Signal Line",
                )

                # Plot histogram if available
                if "macd_histogram" in macd_data.columns:
                    # Plot histogram as bar chart
                    positive_hist = macd_data["macd_histogram"].copy()
                    negative_hist = macd_data["macd_histogram"].copy()
                    positive_hist[positive_hist <= 0] = np.nan
                    negative_hist[negative_hist > 0] = np.nan

                    ax2.bar(
                        macd_data.index,
                        positive_hist,
                        color="green",
                        alpha=0.5,
                        label="Positive Histogram",
                    )
                    ax2.bar(
                        macd_data.index,
                        negative_hist,
                        color="red",
                        alpha=0.5,
                        label="Negative Histogram",
                    )

                # Add zero line
                ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

                ax2.set_title("MACD Indicator")
                ax2.set_ylabel("MACD Value")
                ax2.legend()
                ax2.grid(True)

            # Format and save the chart
            plt.tight_layout()
            chart_path = TEST_RESULTS_DIR / f"{filename}.png"
            plt.savefig(chart_path)
            plt.close(fig)

            print(f"Chart saved to {chart_path}")
        except Exception as e:
            # Don't fail the test if chart creation fails
            print(f"Error creating chart: {e}")
