#!/usr/bin/env python3
"""
Genetic Algorithm Optimization Example

This script demonstrates how to use the genetic algorithm optimization feature
to find optimal parameters for a trading strategy using the backtesting framework.
"""

import logging
import os
from datetime import datetime, timedelta

import pandas as pd

from app.backtesting import (BacktestEngine, CSVDataSource, DataManager,
                             multi_indicator_strategy)
from app.indicators.bollinger_bands_indicator import BollingerBandsIndicator
from app.indicators.rsi_indicator import RSIIndicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the genetic algorithm optimization example."""
    # Create a data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Generate sample data if needed (in real use, you'd load historical data from an exchange)
    sample_data_path = os.path.join(data_dir, 'ETH-USD_1h.csv')
    if not os.path.exists(sample_data_path):
        logger.info("Generating sample data...")
        generate_sample_data(sample_data_path)

    # Initialize DataManager
    data_manager = DataManager(data_dir=data_dir)
    data_manager.register_data_source('csv', CSVDataSource(data_dir))

    # Initialize indicators
    rsi = RSIIndicator(period=14)
    bollinger = BollingerBandsIndicator(period=20, num_std_dev=2.0)
    indicators = [rsi, bollinger]

    # Create BacktestEngine
    engine = BacktestEngine(
        data_manager=data_manager,
        initial_balance={'USD': 10000.0},
        maker_fee=0.0001,  # 0.01%
        taker_fee=0.0005,  # 0.05%
    )

    # Define parameter space for genetic optimization
    param_space = {
        # Discrete parameter options
        'rsi_buy_threshold': [20, 25, 30, 35],
        'rsi_sell_threshold': [65, 70, 75, 80],

        # Continuous parameter ranges (min, max, step)
        'bollinger_entry_pct': (0.01, 0.05, 0.01),  # 1% to 5% in 1% steps
        'position_size': (0.1, 0.5, 0.1),           # 10% to 50% in 10% steps

        # Strategy parameters
        'stop_loss_pct': [0.05, 0.1, 0.15],        # 5%, 10%, or 15%
        'take_profit_pct': [0.1, 0.15, 0.2, 0.25]  # 10%, 15%, 20%, or 25%
    }

    # Run genetic optimization
    logger.info("Starting genetic algorithm optimization...")
    best_params, best_result = engine.genetic_optimize(
        strategy_func=multi_indicator_strategy,
        symbol='ETH-USD',
        interval='1h',
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 2, 1),
        data_source_name='csv',
        param_space=param_space,
        population_size=20,         # Number of parameter combinations in each generation
        generations=5,              # Number of evolutionary generations
        crossover_rate=0.7,         # Probability of parameter crossover
        mutation_rate=0.2,          # Probability of parameter mutation
        tournament_size=3,          # Number of candidates in tournament selection
        metric_to_optimize='sharpe_ratio',  # Metric to maximize
        leverage=1.0,               # Default leverage
        indicators=indicators,      # Technical indicators to use
        random_seed=42              # For reproducible results
    )

    # Display optimization results
    logger.info(f"\nBest Parameters Found: {best_params}")
    logger.info(f"\nPerformance Metrics:")
    for metric, value in best_result.metrics.items():
        logger.info(f"  {metric}: {value}")

    # Save equity curve chart
    chart_path = os.path.join(data_dir, 'equity_curve.png')
    best_result.plot_equity_curve(save_path=chart_path)
    logger.info(f"\nEquity curve saved to {chart_path}")

    # Run validation test on different time period
    logger.info("\nRunning validation test on different time period...")
    validation_result = engine.run_backtest(
        strategy_func=multi_indicator_strategy,
        symbol='ETH-USD',
        interval='1h',
        start_date=datetime(2020, 2, 1),
        end_date=datetime(2020, 3, 1),
        data_source_name='csv',
        strategy_params=best_params,
        leverage=1.0,
        indicators=indicators
    )

    # Display validation results
    logger.info(f"\nValidation Performance Metrics:")
    for metric, value in validation_result.metrics.items():
        logger.info(f"  {metric}: {value}")

    # Compare optimization and validation results
    logger.info("\nKey Metrics Comparison (Optimization vs Validation):")
    for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
        opt_value = best_result.metrics.get(metric, 'N/A')
        val_value = validation_result.metrics.get(metric, 'N/A')
        logger.info(f"  {metric}: {opt_value} vs {val_value}")

def generate_sample_data(file_path):
    """Generate sample price data for demo purposes."""
    # Create a date range
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 3, 1)
    delta = timedelta(hours=1)

    dates = []
    current_date = start_date
    while current_date < end_date:
        dates.append(current_date)
        current_date += delta

    timestamps = [int(date.timestamp() * 1000) for date in dates]

    # Generate price data with trends and volatility
    import numpy as np

    # Start with a base price
    close_price = 200.0
    closes = []

    # Create trends with random walk
    for i in range(len(dates)):
        # Add some seasonality (24hr cycle)
        seasonal_factor = 0.5 * np.sin(2 * np.pi * (i % 24) / 24)

        # Add trend component (upward trend for the first half, downward for the second)
        if i < len(dates) / 2:
            trend = 0.1  # Upward trend
        else:
            trend = -0.08  # Downward trend

        # Random component
        noise = np.random.normal(0, 1.0)

        # Combine components and update price
        price_change = trend + seasonal_factor + noise
        close_price = max(close_price * (1 + price_change / 100), 1.0)  # Ensure price > 0
        closes.append(close_price)

    # Create OHLCV data
    data = {
        'timestamp': timestamps,
        'open': [c * (1 + np.random.uniform(-0.005, 0.005)) for c in closes],
        'high': [c * (1 + np.random.uniform(0, 0.01)) for c in closes],
        'low': [c * (1 - np.random.uniform(0, 0.01)) for c in closes],
        'close': closes,
        'volume': [np.random.uniform(1000, 10000) for _ in range(len(dates))]
    }

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    logger.info(f"Generated sample data and saved to {file_path}")

if __name__ == "__main__":
    main()
