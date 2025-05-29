import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from app.indicators.base_indicator import SignalDirection
from app.indicators.macd_indicator import MACDIndicator

# Directory to save charts for visual inspection
TEST_RESULTS_DIR = Path(__file__).parent.parent / "__test_results__"
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class TestMACDIndicatorWithRealData:
    """Test MACD indicator implementation with real market data."""

    @pytest.mark.slow
    def test_macd_signal_generation_with_default_params(self, sample_price_data):
        """Test MACD indicator with default parameters on real market data."""
        # Create MACD indicator with default parameters
        macd = MACDIndicator(fast_period=12, slow_period=26, signal_period=9)

        # Generate signals
        signals = macd.generate_signals(sample_price_data)

        # Basic assertions
        assert signals is not None
        assert len(signals) > 0
        assert "signal" in signals.columns
        assert "confidence" in signals.columns
        assert "macd" in signals.columns
        assert "macd_signal" in signals.columns
        assert "macd_histogram" in signals.columns

        # Verify we have both BUY and SELL signals
        buy_signals = signals[signals["signal"] == SignalDirection.BUY]
        sell_signals = signals[signals["signal"] == SignalDirection.SELL]

        # Log signal counts
        print(
            f"Found {len(buy_signals)} BUY signals and {len(sell_signals)} SELL signals"
        )

        # There should be some signals (can't guarantee both buy and sell with real data)
        assert len(buy_signals) + len(sell_signals) > 0

        # Optional: Save a chart for visual inspection
        self._save_chart(sample_price_data, signals, "macd_default_params")

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "market_data_params",
        [
            {"exchange": "hyperliquid", "symbol": "BTC-USD", "timeframe": "1h"},
            {"exchange": "hyperliquid", "symbol": "ETH-USD", "timeframe": "1h"},
            {"exchange": "hyperliquid", "symbol": "SOL-USD", "timeframe": "1h"},
        ],
        indirect=True,
    )
    def test_macd_performance_across_assets(self, real_market_data, market_data_params):
        """Test MACD indicator across different assets to compare performance."""
        # Skip test if data couldn't be fetched
        if len(real_market_data) < 30:
            pytest.skip(f"Not enough data for {market_data_params['symbol']}")

        # Get symbol for reporting
        symbol = market_data_params["symbol"]

        # Create MACD indicator
        macd = MACDIndicator()

        # Generate signals
        signals = macd.generate_signals(real_market_data)

        # Count signals
        buy_signals = signals[signals["signal"] == SignalDirection.BUY]
        sell_signals = signals[signals["signal"] == SignalDirection.SELL]

        # Print signal statistics
        buy_pct = 100 * len(buy_signals) / len(signals) if len(signals) > 0 else 0
        sell_pct = 100 * len(sell_signals) / len(signals) if len(signals) > 0 else 0
        print(
            f"{symbol}: BUY: {len(buy_signals)} ({buy_pct:.1f}%), SELL: {len(sell_signals)} ({sell_pct:.1f}%)"
        )

        # Optional: Save a chart for this asset
        symbol_safe = symbol.replace("/", "_").replace("-", "_")
        self._save_chart(real_market_data, signals, f"macd_{symbol_safe}")

        # Basic assertions
        assert signals is not None
        assert "macd" in signals.columns

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "params",
        [
            {
                "fast_period": 8,
                "slow_period": 21,
                "signal_period": 5,
            },  # More responsive
            {
                "fast_period": 16,
                "slow_period": 32,
                "signal_period": 12,
            },  # Less responsive
        ],
    )
    def test_macd_with_various_parameters(self, sample_price_data, params):
        """Test MACD indicator with different parameters on real market data."""
        # Create MACD indicator with custom parameters
        macd = MACDIndicator(**params)

        # Generate signals
        signals = macd.generate_signals(sample_price_data)

        # Log parameter and signal counts
        buy_signals = signals[signals["signal"] == SignalDirection.BUY]
        sell_signals = signals[signals["signal"] == SignalDirection.SELL]
        print(
            f"Fast: {params['fast_period']}, Slow: {params['slow_period']}, Signal: {params['signal_period']}"
        )
        print(f"Found {len(buy_signals)} BUY and {len(sell_signals)} SELL signals")

        # Save chart for visual comparison
        param_str = f"fast{params['fast_period']}_slow{params['slow_period']}_signal{params['signal_period']}"
        self._save_chart(sample_price_data, signals, f"macd_{param_str}")

        # Basic assertions
        assert signals is not None
        assert "macd" in signals.columns

    def _save_chart(self, price_data, signals, filename_prefix):
        """Save a chart of price data with MACD and signals for visual inspection."""
        try:
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 2]}
            )

            # Plot price chart
            ax1.plot(
                price_data.index, price_data["close"], color="blue", label="Close Price"
            )

            # Plot buy signals on price chart
            buy_signals = signals[signals["signal"] == SignalDirection.BUY]
            sell_signals = signals[signals["signal"] == SignalDirection.SELL]

            if not buy_signals.empty:
                ax1.scatter(
                    buy_signals.index,
                    price_data.loc[buy_signals.index]["close"],
                    color="green",
                    label="Buy Signal",
                    marker="^",
                    s=100,
                )

            if not sell_signals.empty:
                ax1.scatter(
                    sell_signals.index,
                    price_data.loc[sell_signals.index]["close"],
                    color="red",
                    label="Sell Signal",
                    marker="v",
                    s=100,
                )

            ax1.set_title("Price Chart with MACD Signals")
            ax1.set_ylabel("Price")
            ax1.legend()
            ax1.grid(True)

            # Plot MACD indicator
            ax2.plot(signals.index, signals["macd"], color="blue", label="MACD Line")
            ax2.plot(
                signals.index, signals["macd_signal"], color="red", label="Signal Line"
            )

            # Plot histogram as bar chart
            positive_hist = signals["macd_histogram"].copy()
            negative_hist = signals["macd_histogram"].copy()
            positive_hist[positive_hist <= 0] = np.nan
            negative_hist[negative_hist > 0] = np.nan

            ax2.bar(
                signals.index,
                positive_hist,
                color="green",
                alpha=0.5,
                label="Positive Histogram",
            )
            ax2.bar(
                signals.index,
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
            chart_path = TEST_RESULTS_DIR / f"{filename_prefix}.png"
            plt.savefig(chart_path)
            plt.close(fig)

            print(f"Chart saved to {chart_path}")
        except Exception as e:
            # Don't fail the test if chart creation fails
            print(f"Error creating chart: {e}")
