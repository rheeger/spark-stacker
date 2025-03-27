import os
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from app.backtesting.backtest_engine import BacktestEngine
from app.backtesting.data_manager import CSVDataSource, DataManager
from app.backtesting.simulation_engine import SimulationEngine
from app.indicators.rsi_indicator import RSIIndicator


# Sample strategy for testing
def simple_strategy_with_many_params(
    historical_data: pd.DataFrame,
    simulation_engine: SimulationEngine,
    params: Dict[str, Any]
) -> None:
    """
    A simple strategy for testing genetic optimization with many parameters.
    This strategy is intentionally simplified for testing purposes only.
    """
    if len(historical_data) < 2:
        return

    # Extract current candle and symbol
    current_candle = params['current_candle']
    symbol = params['symbol']

    # Extract strategy parameters or use defaults
    param1 = params.get('param1', 5)
    param2 = params.get('param2', 10)
    param3 = params.get('param3', 15)
    param4 = params.get('param4', 0.1)
    param5 = params.get('param5', 0.2)
    param6 = params.get('param6', 0.5)
    param7 = params.get('param7', 1.0)
    param8 = params.get('param8', 2.0)
    param9 = params.get('param9', True)
    param10 = params.get('param10', 'option1')

    # Get RSI value if available
    rsi_value = params.get('RSI_rsi', 50.0)

    # Very simple logic for testing:
    # Buy if RSI is below param1 threshold
    # Sell if RSI is above param2 threshold
    # Use position size based on param6

    # Check for open positions
    positions = simulation_engine.get_positions()
    has_position = any(pos['symbol'] == symbol for pos in positions)

    if not has_position and rsi_value < param1:
        # Buy signal
        position_size = param6
        if param9:  # Boolean parameter affecting trade size
            position_size *= param7

        # Place order
        simulation_engine.place_market_order(
            symbol=symbol,
            side="BUY",
            quantity=current_candle['close'] * position_size / params.get('leverage', 1.0),
            margin=True
        )

    elif has_position and rsi_value > param2:
        # Sell signal
        for position in positions:
            if position['symbol'] == symbol:
                simulation_engine.place_market_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=abs(position['quantity']),
                    margin=True
                )
                break


class TestGeneticOptimizationStress:

    @pytest.fixture
    def sample_data(self):
        """Create sample price data with clear patterns for testing."""
        # Create a date range
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(100)]
        timestamps = [int(date.timestamp() * 1000) for date in dates]

        # Generate price data designed to have clear RSI patterns
        closes = []
        base_price = 100.0

        # Create a cyclical pattern that will result in clear RSI signals
        for i in range(100):
            # Create cycles that gradually increase in amplitude
            cycle_amplitude = 10 * (1 + i / 200)
            if i % 20 < 10:  # First half of cycle: price rises
                change = (i % 10) * 0.01 * cycle_amplitude
            else:  # Second half of cycle: price falls
                change = -((i % 10) * 0.01) * cycle_amplitude

            base_price += change
            closes.append(max(base_price, 1.0))

        # Create OHLCV data
        data = {
            'timestamp': timestamps,
            'open': [c * 0.99 for c in closes],
            'high': [c * 1.01 for c in closes],
            'low': [c * 0.98 for c in closes],
            'close': closes,
            'volume': [10000.0 for _ in range(100)]
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def backtest_engine(self, sample_data):
        """Create a BacktestEngine with sample data."""
        # Create a temporary directory for data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a DataManager
            dm = DataManager(data_dir=temp_dir)

            # Save sample data to CSV
            sample_data.to_csv(os.path.join(temp_dir, "TEST-USD_1d.csv"), index=False)

            # Register a CSV data source
            dm.register_data_source('csv', CSVDataSource(temp_dir))

            # Create BacktestEngine
            engine = BacktestEngine(
                data_manager=dm,
                initial_balance={'USD': 10000.0},
                maker_fee=0.001,
                taker_fee=0.002,
                slippage_model='fixed'
            )

            yield engine

    def test_large_parameter_space(self, backtest_engine):
        """Test genetic optimization with a large parameter space."""
        # Define a complex parameter space with many parameters
        param_space = {
            'param1': list(range(10, 40, 5)),              # 6 options (RSI buy threshold)
            'param2': list(range(60, 90, 5)),              # 6 options (RSI sell threshold)
            'param3': (10, 50, 5),                         # Range (integers)
            'param4': (0.01, 0.1, 0.01),                   # Range (floats)
            'param5': (0.1, 0.5, 0.1),                     # Range (floats)
            'param6': [0.1, 0.2, 0.3, 0.4, 0.5],           # 5 options (position size)
            'param7': [0.5, 1.0, 1.5, 2.0],                # 4 options (multiplier)
            'param8': (1.0, 5.0, 0.5),                     # Range (floats)
            'param9': [True, False],                       # Boolean option
            'param10': ['option1', 'option2', 'option3']   # String options
        }

        # This parameter space has a total of 6 * 6 * 9 * 10 * 5 * 5 * 4 * 9 * 2 * 3 = 5,832,000 possible combinations
        # Grid search would be impractical, but genetic algorithms should handle it

        # Initialize indicator
        rsi = RSIIndicator(name="RSI", params={"period": 14})

        # Run genetic optimization with a small population and few generations for testing
        best_params, best_result = backtest_engine.genetic_optimize(
            strategy_func=simple_strategy_with_many_params,
            symbol='TEST-USD',
            interval='1d',
            start_date='2020-01-01',
            end_date='2020-04-10',
            data_source_name='csv',
            param_space=param_space,
            population_size=10,       # Small population for testing
            generations=3,            # Few generations for testing
            random_seed=42            # For reproducibility
        )

        # Verify results are within expected parameter ranges
        assert isinstance(best_params, dict)
        assert len(best_params) == len(param_space)

        # Check each parameter is within its expected range
        assert 10 <= best_params['param1'] <= 35
        assert 60 <= best_params['param2'] <= 85
        assert 10 <= best_params['param3'] <= 50
        assert 0.01 <= best_params['param4'] <= 0.1
        assert 0.1 <= best_params['param5'] <= 0.5
        assert best_params['param6'] in [0.1, 0.2, 0.3, 0.4, 0.5]
        assert best_params['param7'] in [0.5, 1.0, 1.5, 2.0]
        assert 1.0 <= best_params['param8'] <= 5.0
        assert best_params['param9'] in [True, False]
        assert best_params['param10'] in ['option1', 'option2', 'option3']

    def test_edge_cases(self, backtest_engine):
        """Test genetic optimization with edge cases."""

        # 1. Test with a single parameter (minimal case)
        single_param_space = {
            'param1': list(range(5, 50, 5))  # RSI threshold
        }

        # Run with minimal parameter space
        best_params, best_result = backtest_engine.genetic_optimize(
            strategy_func=simple_strategy_with_many_params,
            symbol='TEST-USD',
            interval='1d',
            start_date='2020-01-01',
            end_date='2020-04-10',
            data_source_name='csv',
            param_space=single_param_space,
            population_size=5,
            generations=2,
            random_seed=42
        )

        # Verify result
        assert 'param1' in best_params
        assert 5 <= best_params['param1'] <= 45

        # 2. Test with very small population size (edge case)
        small_param_space = {
            'param1': list(range(20, 40, 5)),
            'param2': list(range(60, 80, 5))
        }

        # Run with minimal population
        best_params, best_result = backtest_engine.genetic_optimize(
            strategy_func=simple_strategy_with_many_params,
            symbol='TEST-USD',
            interval='1d',
            start_date='2020-01-01',
            end_date='2020-04-10',
            data_source_name='csv',
            param_space=small_param_space,
            population_size=3,        # Very small population
            generations=2,
            random_seed=42
        )

        # Verify result
        assert 'param1' in best_params and 'param2' in best_params

        # 3. Test with extreme mutation rate (edge case)
        high_mutation_params = {
            'param1': list(range(20, 40, 5)),
            'param2': list(range(60, 80, 5)),
            'param6': [0.1, 0.3, 0.5]
        }

        # Run with high mutation rate
        best_params, best_result = backtest_engine.genetic_optimize(
            strategy_func=simple_strategy_with_many_params,
            symbol='TEST-USD',
            interval='1d',
            start_date='2020-01-01',
            end_date='2020-04-10',
            data_source_name='csv',
            param_space=high_mutation_params,
            population_size=5,
            generations=3,
            mutation_rate=0.9,        # Very high mutation rate
            random_seed=42
        )

        # Verify result
        assert len(best_params) == 3
        assert best_params['param6'] in [0.1, 0.3, 0.5]

        # 4. Test with no crossover (edge case)
        no_crossover_params = {
            'param1': list(range(20, 40, 5)),
            'param2': list(range(60, 80, 5)),
            'param6': [0.1, 0.3, 0.5]
        }

        # Run with no crossover
        best_params, best_result = backtest_engine.genetic_optimize(
            strategy_func=simple_strategy_with_many_params,
            symbol='TEST-USD',
            interval='1d',
            start_date='2020-01-01',
            end_date='2020-04-10',
            data_source_name='csv',
            param_space=no_crossover_params,
            population_size=5,
            generations=3,
            crossover_rate=0.0,       # No crossover
            random_seed=42
        )

        # Verify result (should still work, just less effective)
        assert len(best_params) == 3
        assert best_params['param6'] in [0.1, 0.3, 0.5]
