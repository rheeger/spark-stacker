#!/usr/bin/env python3
"""
Backtesting Tutorial for Spark Stacker

This tutorial demonstrates how to use the REAL Spark Stacker backtesting framework
to test trading strategies with actual indicators and real performance metrics.

This uses the actual BacktestEngine, DataManager, and other components from the
production system.

Author: Spark Stacker Development Team
Usage: python backtesting_tutorial.py
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the spark-app directory to Python path so we can import from the real codebase
spark_app_path = os.path.join(os.path.dirname(__file__), '..', '..', 'spark-app')
sys.path.insert(0, spark_app_path)

# Import from the REAL Spark Stacker codebase
from app.backtesting.backtest_engine import BacktestEngine
from app.backtesting.data_manager import CSVDataSource, DataManager
from app.backtesting.strategy import simple_moving_average_crossover_strategy
from app.indicators.indicator_factory import IndicatorFactory
from app.indicators.moving_average_indicator import MovingAverageIndicator
from app.indicators.rsi_indicator import RSIIndicator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data_file(file_path: str, symbol: str = "ETH-USD", days: int = 90) -> None:
    """
    Create a sample CSV file with realistic market data for backtesting.
    This creates data in the format expected by the real Spark Stacker system.

    Args:
        file_path: Path where to save the CSV file
        symbol: Trading symbol
        days: Number of days of data to generate
    """
    logger.info(f"Creating sample data file at {file_path}")

    # Generate timestamps (hourly data)
    timestamps = pd.date_range(start="2023-01-01", periods=days * 24, freq="1H")

    # Generate price data with multiple market regimes
    np.random.seed(42)
    base_price = 1500.0

    # Create regime changes for more realistic testing
    regime_length = len(timestamps) // 3

    # Regime 1: Uptrend
    returns1 = np.random.normal(0.0005, 0.015, regime_length)

    # Regime 2: Sideways with high volatility
    returns2 = np.random.normal(0.0001, 0.025, regime_length)

    # Regime 3: Downtrend
    returns3 = np.random.normal(-0.0003, 0.018, len(timestamps) - 2 * regime_length)

    returns = np.concatenate([returns1, returns2, returns3])

    # Calculate prices
    log_prices = np.log(base_price) + np.cumsum(returns)
    prices = np.exp(log_prices)

    # Create OHLCV data in the format expected by Spark Stacker
    df = pd.DataFrame({
        "timestamp": [int(ts.timestamp() * 1000) for ts in timestamps],  # Milliseconds timestamp
        "open": np.concatenate([[base_price], prices[:-1]]),
        "high": prices * (1 + np.random.uniform(0, 0.005, len(prices))),
        "low": prices * (1 - np.random.uniform(0, 0.005, len(prices))),
        "close": prices,
        "volume": np.random.lognormal(mean=10, sigma=0.5, size=len(timestamps))
    })

    # Ensure OHLC relationships are valid
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    # Save to CSV
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info(f"Sample data saved: {len(df)} data points")


def create_rsi_strategy_function():
    """
    Create a strategy function that uses RSI for signals.
    This demonstrates how to write strategy functions for the real BacktestEngine.
    """
    def rsi_strategy(data: pd.DataFrame, simulation_engine, params: Dict) -> None:
        """
        RSI-based trading strategy for use with the real BacktestEngine.

        Args:
            data: Historical price data with RSI values calculated
            simulation_engine: Real SimulationEngine for executing trades
            params: Strategy parameters including RSI thresholds
        """
        symbol = params.get("symbol", "ETH-USD")
        current_candle = params.get("current_candle")

        if current_candle is None or len(data) < 2:
            return

        # Get RSI value (should be calculated by indicator)
        rsi_value = current_candle.get("rsi", None)
        if rsi_value is None or pd.isna(rsi_value):
            return

        # Get strategy parameters
        oversold_threshold = params.get("oversold_threshold", 30)
        overbought_threshold = params.get("overbought_threshold", 70)
        position_size = params.get("position_size", 0.1)

        current_price = current_candle["close"]

        # Check current positions
        current_position = simulation_engine.get_position(symbol)

        # Buy signal: RSI oversold and no current position
        if rsi_value < oversold_threshold and (not current_position or current_position.size == 0):
            # Calculate position size based on available balance
            balance = simulation_engine.get_balance()
            available_usd = balance.get("USD", 0)

            if available_usd > 100:  # Minimum trade size
                trade_size = (available_usd * position_size) / current_price

                simulation_engine.place_order(
                    symbol=symbol,
                    side="BUY",
                    order_type="MARKET",
                    size=trade_size,
                    price=None  # Market order
                )
                logger.debug(f"RSI Buy signal: {symbol} at {current_price}, RSI={rsi_value:.2f}")

        # Sell signal: RSI overbought and have position
        elif rsi_value > overbought_threshold and current_position and current_position.size > 0:
            simulation_engine.place_order(
                symbol=symbol,
                side="SELL",
                order_type="MARKET",
                size=current_position.size,
                price=None  # Market order
            )
            logger.debug(f"RSI Sell signal: {symbol} at {current_price}, RSI={rsi_value:.2f}")

    return rsi_strategy


def run_real_backtest_tutorial():
    """
    Run a comprehensive backtesting tutorial using the real Spark Stacker framework.
    """
    logger.info("Starting Real Backtesting Tutorial")

    # Step 1: Set up data directory and sample data
    tutorial_dir = Path(__file__).parent / "tutorial_data"
    tutorial_dir.mkdir(exist_ok=True)

    data_file = tutorial_dir / "ETH-USD_1h.csv"
    if not data_file.exists():
        create_sample_data_file(str(data_file), "ETH-USD", days=90)

    # Step 2: Initialize real DataManager and data source
    logger.info("Setting up real DataManager...")
    data_manager = DataManager()
    csv_source = CSVDataSource(data_directory=str(tutorial_dir))
    data_manager.register_source("csv", csv_source)

    # Step 3: Create real indicators
    logger.info("Creating real indicators...")
    rsi_indicator = RSIIndicator(
        name="tutorial_rsi",
        params={"period": 14, "overbought": 70, "oversold": 30}
    )

    ma_indicator = MovingAverageIndicator(
        name="tutorial_ma",
        params={"fast_period": 10, "slow_period": 30, "ma_type": "ema"}
    )

    # Step 4: Initialize real BacktestEngine
    logger.info("Initializing real BacktestEngine...")
    engine = BacktestEngine(
        data_manager=data_manager,
        initial_balance={"USD": 10000.0},
        maker_fee=0.0001,  # 0.01%
        taker_fee=0.0005,  # 0.05%
        slippage_model="random"
    )

    # Step 5: Run backtest with RSI strategy
    logger.info("Running RSI strategy backtest...")

    rsi_strategy = create_rsi_strategy_function()

    rsi_result = engine.run_backtest(
        strategy_func=rsi_strategy,
        symbol="ETH-USD",
        interval="1h",
        start_date="2023-01-01",
        end_date="2023-03-31",
        data_source_name="csv",
        strategy_params={
            "oversold_threshold": 30,
            "overbought_threshold": 70,
            "position_size": 0.2
        },
        leverage=1.0,
        indicators=[rsi_indicator]
    )

    # Step 6: Run backtest with Moving Average strategy
    logger.info("Running Moving Average strategy backtest...")

    ma_result = engine.run_backtest(
        strategy_func=simple_moving_average_crossover_strategy,
        symbol="ETH-USD",
        interval="1h",
        start_date="2023-01-01",
        end_date="2023-03-31",
        data_source_name="csv",
        strategy_params={
            "fast_period": 10,
            "slow_period": 30,
            "position_size": 0.2
        },
        leverage=1.0,
        indicators=[ma_indicator]
    )

    # Step 7: Display comprehensive results
    display_backtest_results("RSI Strategy", rsi_result)
    display_backtest_results("Moving Average Strategy", ma_result)

    # Step 8: Compare strategies
    compare_strategies({
        "RSI Strategy": rsi_result,
        "MA Strategy": ma_result
    })

    # Step 9: Parameter optimization example
    logger.info("Running parameter optimization...")
    optimization_result = run_optimization_example(engine)

    return {
        "rsi_result": rsi_result,
        "ma_result": ma_result,
        "optimization_result": optimization_result
    }


def display_backtest_results(strategy_name: str, result) -> None:
    """
    Display comprehensive backtest results from the real BacktestEngine.

    Args:
        strategy_name: Name of the strategy
        result: BacktestResult object from the real engine
    """
    print(f"\n{'='*60}")
    print(f"{strategy_name} - Backtest Results")
    print(f"{'='*60}")

    metrics = result.metrics

    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"Symbol: {result.symbol}")
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
    print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    print(f"  Winning Trades: {metrics.get('winning_trades', 0)}")
    print(f"  Losing Trades: {metrics.get('losing_trades', 0)}")
    print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
    print(f"  Average Profit: ${metrics.get('avg_profit', 0):.2f}")
    print(f"  Average Loss: ${metrics.get('avg_loss', 0):.2f}")
    print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"  Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    print(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")

    # Display initial vs final balance
    print(f"\nBalance Changes:")
    for currency, initial_amount in result.initial_balance.items():
        final_amount = result.final_balance.get(currency, 0)
        change = final_amount - initial_amount
        change_pct = (change / initial_amount) * 100 if initial_amount > 0 else 0
        print(f"  {currency}: ${initial_amount:,.2f} → ${final_amount:,.2f} ({change_pct:+.2f}%)")


def compare_strategies(results: Dict[str, any]) -> None:
    """
    Compare multiple strategy results side by side.

    Args:
        results: Dictionary of strategy name -> BacktestResult
    """
    print(f"\n{'='*80}")
    print("Strategy Comparison")
    print(f"{'='*80}")

    # Key metrics to compare
    metrics_to_compare = [
        ("Total Return", "total_return", ":.2%"),
        ("Sharpe Ratio", "sharpe_ratio", ":.2f"),
        ("Max Drawdown", "max_drawdown", ":.2%"),
        ("Win Rate", "win_rate", ":.2%"),
        ("Total Trades", "total_trades", ""),
        ("Profit Factor", "profit_factor", ":.2f")
    ]

    for metric_name, metric_key, format_str in metrics_to_compare:
        print(f"\n{metric_name}:")
        for strategy_name, result in results.items():
            value = result.metrics.get(metric_key, 0)
            if format_str:
                formatted_value = format(value, format_str)
            else:
                formatted_value = str(value)
            print(f"  {strategy_name:25}: {formatted_value}")


def run_optimization_example(engine: BacktestEngine):
    """
    Demonstrate parameter optimization using the real BacktestEngine.

    Args:
        engine: Real BacktestEngine instance

    Returns:
        Optimization results
    """
    logger.info("Running parameter optimization example...")

    # Define parameter grid for optimization
    param_grid = {
        "fast_period": [5, 10, 15],
        "slow_period": [20, 30, 40],
        "position_size": [0.1, 0.2, 0.3]
    }

    # Use the real optimization method
    try:
        best_params, best_result = engine.optimize_parameters(
            strategy_func=simple_moving_average_crossover_strategy,
            symbol="ETH-USD",
            interval="1h",
            start_date="2023-01-01",
            end_date="2023-02-28",
            data_source_name="csv",
            param_grid=param_grid,
            metric_to_optimize="sharpe_ratio",
            leverage=1.0,
            indicators=[MovingAverageIndicator("opt_ma", {})]
        )

        logger.info(f"Optimization complete!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best Sharpe ratio: {best_result.metrics.get('sharpe_ratio', 0):.3f}")

        return {"best_params": best_params, "best_result": best_result}

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        logger.info("This might happen if optimization methods aren't fully available")
        return None


def create_visualization(results: Dict[str, any]) -> None:
    """
    Create visualizations of backtest results using real data.

    Args:
        results: Dictionary of strategy results
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Spark Stacker Backtesting Results", fontsize=16)

        # Plot 1: Equity curves
        ax1 = axes[0, 0]
        for strategy_name, result in results.items():
            equity_curve = result.equity_curve
            if not equity_curve.empty:
                ax1.plot(equity_curve['datetime'], equity_curve['equity'],
                        label=strategy_name, linewidth=2)

        ax1.set_title("Equity Curves")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Drawdown
        ax2 = axes[0, 1]
        for strategy_name, result in results.items():
            equity_curve = result.equity_curve
            if not equity_curve.empty and 'drawdown' in equity_curve.columns:
                ax2.plot(equity_curve['datetime'], equity_curve['drawdown'] * 100,
                        label=strategy_name, linewidth=2)

        ax2.set_title("Drawdown (%)")
        ax2.set_ylabel("Drawdown (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Returns comparison
        ax3 = axes[1, 0]
        strategy_names = list(results.keys())
        returns = [results[name].metrics.get('total_return', 0) * 100 for name in strategy_names]

        bars = ax3.bar(strategy_names, returns, alpha=0.7)
        ax3.set_title("Total Returns Comparison")
        ax3.set_ylabel("Return (%)")
        ax3.grid(True, alpha=0.3)

        # Color bars based on performance
        for bar, ret in zip(bars, returns):
            bar.set_color('green' if ret > 0 else 'red')

        # Plot 4: Risk-Return scatter
        ax4 = axes[1, 1]
        for strategy_name, result in results.items():
            ret = result.metrics.get('total_return', 0) * 100
            risk = result.metrics.get('max_drawdown', 0) * 100
            ax4.scatter(risk, ret, s=100, label=strategy_name, alpha=0.7)
            ax4.annotate(strategy_name, (risk, ret), xytext=(5, 5),
                        textcoords='offset points', fontsize=10)

        ax4.set_xlabel("Max Drawdown (%)")
        ax4.set_ylabel("Total Return (%)")
        ax4.set_title("Risk vs Return")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        logger.info("✓ Visualizations created successfully")

    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")


def main():
    """
    Main tutorial function demonstrating real backtesting workflow.
    """
    print("=" * 60)
    print("Spark Stacker Real Backtesting Tutorial")
    print("Using REAL Framework Components")
    print("=" * 60)

    try:
        # Run the comprehensive tutorial
        results = run_real_backtest_tutorial()

        # Create visualizations
        print("\nCreating visualizations...")
        if results:
            strategy_results = {k: v for k, v in results.items() if k.endswith('_result')}
            if strategy_results:
                create_visualization(strategy_results)

        print("\n" + "=" * 60)
        print("Tutorial Complete!")
        print("=" * 60)
        print("\nKey Achievements:")
        print("✓ Used real BacktestEngine for realistic simulation")
        print("✓ Integrated real indicators with strategies")
        print("✓ Generated comprehensive performance metrics")
        print("✓ Demonstrated parameter optimization")
        print("✓ Compared multiple strategies")
        print("\nNext Steps:")
        print("1. Experiment with different strategy parameters")
        print("2. Create your own strategy functions")
        print("3. Use walk-forward analysis for robust testing")
        print("4. Test on different time periods and markets")
        print("5. Integrate with real exchange connectors for live trading")

    except Exception as e:
        logger.error(f"Tutorial failed: {e}")
        print("\nTutorial encountered an error. This might happen if:")
        print("1. The full Spark Stacker environment isn't available")
        print("2. Required dependencies are missing")
        print("3. Data files couldn't be created")
        print(f"\nError details: {e}")


if __name__ == "__main__":
    main()
