"""
Scenario Manager for CLI Backtesting

Centralizes multi-scenario testing logic, coordinating scenario data generation,
execution scheduling, result aggregation, and performance analysis.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Enumeration of available market scenario types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CHOPPY = "choppy"
    GAP_HEAVY = "gap_heavy"
    REAL_DATA = "real_data"


@dataclass
class ScenarioConfig:
    """Configuration for a specific market scenario."""
    scenario_type: ScenarioType
    name: str
    description: str
    duration_days: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    weight: float = 1.0  # For weighted scoring


@dataclass
class ScenarioResult:
    """Results from running a backtest on a specific scenario."""
    scenario_type: ScenarioType
    scenario_name: str
    strategy_name: str
    execution_time: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration: float
    profit_factor: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether the scenario executed successfully."""
        return self.error is None


@dataclass
class ScenarioPerformanceAnalysis:
    """Analysis of strategy performance across scenarios."""
    strategy_name: str
    total_scenarios: int
    successful_scenarios: int
    consistency_score: float  # Low variance across scenarios
    adaptability_score: float  # Performance in diverse conditions
    robustness_score: float  # Risk-adjusted consistency
    worst_case_performance: float  # Performance in hardest conditions
    best_scenario: str
    worst_scenario: str
    scenario_correlation: Dict[str, float]
    optimization_recommendations: List[str]


class ScenarioManager:
    """
    Manages multi-scenario testing workflow including data generation,
    execution coordination, and performance analysis.
    """

    def __init__(
        self,
        data_manager=None,
        config_manager=None,
        default_duration_days: int = 30,
        max_workers: int = 4
    ):
        """
        Initialize the ScenarioManager.

        Args:
            data_manager: DataManager instance for data operations
            config_manager: ConfigManager instance for configuration access
            default_duration_days: Default testing duration for scenarios
            max_workers: Maximum concurrent workers for parallel execution
        """
        self.data_manager = data_manager
        self.config_manager = config_manager
        self.default_duration_days = default_duration_days
        self.max_workers = max_workers

        # Initialize scenario configurations
        self.scenario_configs = self._initialize_scenario_configs()

        # Results storage
        self.scenario_results: Dict[str, List[ScenarioResult]] = {}
        self.performance_analyses: Dict[str, ScenarioPerformanceAnalysis] = {}

        logger.info(f"ScenarioManager initialized with {len(self.scenario_configs)} scenarios")

    def _initialize_scenario_configs(self) -> Dict[ScenarioType, ScenarioConfig]:
        """Initialize default scenario configurations."""
        configs = {
            ScenarioType.BULL: ScenarioConfig(
                scenario_type=ScenarioType.BULL,
                name="Bull Market",
                description="Consistent uptrend with 60-80% up days",
                duration_days=self.default_duration_days,
                parameters={
                    "trend_strength": 0.7,
                    "up_day_probability": 0.7,
                    "daily_volatility": 0.02,
                    "trend_persistence": 0.8
                }
            ),
            ScenarioType.BEAR: ScenarioConfig(
                scenario_type=ScenarioType.BEAR,
                name="Bear Market",
                description="Consistent downtrend with 60-80% down days",
                duration_days=self.default_duration_days,
                parameters={
                    "trend_strength": -0.7,
                    "down_day_probability": 0.7,
                    "daily_volatility": 0.025,
                    "trend_persistence": 0.8
                }
            ),
            ScenarioType.SIDEWAYS: ScenarioConfig(
                scenario_type=ScenarioType.SIDEWAYS,
                name="Sideways Market",
                description="Range-bound oscillation within 5-10% range",
                duration_days=self.default_duration_days,
                parameters={
                    "range_percentage": 0.075,
                    "oscillation_frequency": 0.1,
                    "noise_level": 0.02,
                    "mean_reversion_strength": 0.6
                }
            ),
            ScenarioType.HIGH_VOLATILITY: ScenarioConfig(
                scenario_type=ScenarioType.HIGH_VOLATILITY,
                name="High Volatility",
                description="Large daily swings with 15-25% moves",
                duration_days=self.default_duration_days,
                parameters={
                    "daily_volatility": 0.2,
                    "spike_probability": 0.3,
                    "spike_magnitude": 0.15,
                    "volatility_clustering": 0.7
                }
            ),
            ScenarioType.LOW_VOLATILITY: ScenarioConfig(
                scenario_type=ScenarioType.LOW_VOLATILITY,
                name="Low Volatility",
                description="Minimal daily changes with <2% moves",
                duration_days=self.default_duration_days,
                parameters={
                    "daily_volatility": 0.005,
                    "trend_strength": 0.1,
                    "noise_reduction": 0.8,
                    "stability_factor": 0.9
                }
            ),
            ScenarioType.CHOPPY: ScenarioConfig(
                scenario_type=ScenarioType.CHOPPY,
                name="Choppy Market",
                description="Frequent direction changes and whipsaws",
                duration_days=self.default_duration_days,
                parameters={
                    "direction_change_probability": 0.4,
                    "whipsaw_intensity": 0.8,
                    "fake_breakout_probability": 0.3,
                    "noise_amplification": 1.5
                }
            ),
            ScenarioType.GAP_HEAVY: ScenarioConfig(
                scenario_type=ScenarioType.GAP_HEAVY,
                name="Gap Heavy",
                description="Frequent price gaps simulating news events",
                duration_days=self.default_duration_days,
                parameters={
                    "gap_probability": 0.2,
                    "gap_magnitude_range": (0.02, 0.08),
                    "gap_direction_bias": 0.0,
                    "post_gap_behavior": "continuation"
                }
            ),
            ScenarioType.REAL_DATA: ScenarioConfig(
                scenario_type=ScenarioType.REAL_DATA,
                name="Real Market Data",
                description="Historical market data for validation",
                duration_days=self.default_duration_days,
                parameters={
                    "data_source": "live",
                    "validation_mode": True
                }
            )
        }

        return configs

    def get_available_scenarios(self) -> List[str]:
        """Get list of available scenario names."""
        return [config.name for config in self.scenario_configs.values() if config.enabled]

    def get_scenario_config(self, scenario_type: Union[str, ScenarioType]) -> Optional[ScenarioConfig]:
        """Get configuration for a specific scenario."""
        if isinstance(scenario_type, str):
            # Try to match by name or enum value
            for config in self.scenario_configs.values():
                if config.name.lower() == scenario_type.lower() or config.scenario_type.value == scenario_type:
                    return config
            return None
        else:
            return self.scenario_configs.get(scenario_type)

    def update_scenario_duration(self, duration_days: int) -> None:
        """Update duration for all scenarios."""
        for config in self.scenario_configs.values():
            config.duration_days = duration_days
        logger.info(f"Updated all scenario durations to {duration_days} days")

    def enable_scenarios(self, scenario_names: List[str]) -> None:
        """Enable only the specified scenarios."""
        # Disable all first
        for config in self.scenario_configs.values():
            config.enabled = False

        # Enable specified ones
        for name in scenario_names:
            config = self.get_scenario_config(name)
            if config:
                config.enabled = True
                logger.info(f"Enabled scenario: {config.name}")
            else:
                logger.warning(f"Unknown scenario name: {name}")

    def generate_scenario_data(
        self,
        scenario_type: ScenarioType,
        symbol: str,
        timeframe: str,
        duration_days: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Generate synthetic market data for a specific scenario.

        Args:
            scenario_type: Type of scenario to generate
            symbol: Trading symbol
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            duration_days: Override default duration

        Returns:
            DataFrame with OHLCV data or None if generation failed
        """
        config = self.scenario_configs.get(scenario_type)
        if not config:
            logger.error(f"Unknown scenario type: {scenario_type}")
            return None

        days = duration_days or config.duration_days
        logger.info(f"Generating {config.name} data for {symbol} ({timeframe}, {days} days)")

        try:
            if scenario_type == ScenarioType.REAL_DATA:
                return self._generate_real_data(symbol, timeframe, days)
            else:
                return self._generate_synthetic_data(config, symbol, timeframe, days)
        except Exception as e:
            logger.error(f"Failed to generate scenario data: {e}")
            return None

    def _generate_real_data(self, symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
        """Generate real market data using DataManager."""
        if not self.data_manager:
            logger.error("DataManager not available for real data generation")
            return None

        try:
            # Use DataManager to fetch real market data
            return self.data_manager.fetch_market_data(
                symbol=symbol,
                timeframe=timeframe,
                days=days,
                source="live"
            )
        except Exception as e:
            logger.error(f"Failed to fetch real data: {e}")
            return None

    def _generate_synthetic_data(
        self,
        config: ScenarioConfig,
        symbol: str,
        timeframe: str,
        days: int
    ) -> pd.DataFrame:
        """Generate synthetic market data based on scenario configuration."""
        # Calculate number of candles based on timeframe
        candles_per_day = self._get_candles_per_day(timeframe)
        total_candles = int(days * candles_per_day)

        # Generate base price movement
        base_price = 100.0  # Starting price
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=total_candles,
            freq=self._timeframe_to_freq(timeframe)
        )

        # Generate scenario-specific price data
        if config.scenario_type == ScenarioType.BULL:
            prices = self._generate_bull_market(base_price, total_candles, config.parameters)
        elif config.scenario_type == ScenarioType.BEAR:
            prices = self._generate_bear_market(base_price, total_candles, config.parameters)
        elif config.scenario_type == ScenarioType.SIDEWAYS:
            prices = self._generate_sideways_market(base_price, total_candles, config.parameters)
        elif config.scenario_type == ScenarioType.HIGH_VOLATILITY:
            prices = self._generate_high_volatility_market(base_price, total_candles, config.parameters)
        elif config.scenario_type == ScenarioType.LOW_VOLATILITY:
            prices = self._generate_low_volatility_market(base_price, total_candles, config.parameters)
        elif config.scenario_type == ScenarioType.CHOPPY:
            prices = self._generate_choppy_market(base_price, total_candles, config.parameters)
        elif config.scenario_type == ScenarioType.GAP_HEAVY:
            prices = self._generate_gap_heavy_market(base_price, total_candles, config.parameters)
        else:
            raise ValueError(f"Unknown scenario type: {config.scenario_type}")

        # Convert to OHLCV format
        df = self._prices_to_ohlcv(timestamps, prices, symbol)

        logger.info(f"Generated {len(df)} candles for {config.name} scenario")
        return df

    def _get_candles_per_day(self, timeframe: str) -> float:
        """Calculate candles per day for a given timeframe."""
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '1w': 10080
        }

        minutes = timeframe_minutes.get(timeframe, 60)  # Default to 1h
        return 1440 / minutes  # 1440 minutes per day

    def _timeframe_to_freq(self, timeframe: str) -> str:
        """Convert timeframe to pandas frequency string."""
        freq_map = {
            '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
            '1d': '1D', '1w': '1W'
        }
        return freq_map.get(timeframe, '1H')

    def _generate_bull_market(self, base_price: float, candles: int, params: Dict) -> np.ndarray:
        """Generate bull market price series."""
        trend_strength = params.get('trend_strength', 0.7)
        up_probability = params.get('up_day_probability', 0.7)
        volatility = params.get('daily_volatility', 0.02)
        persistence = params.get('trend_persistence', 0.8)

        prices = [base_price]
        trend = 1.0

        for i in range(candles - 1):
            # Trend persistence - trend tends to continue
            if np.random.random() > persistence:
                trend *= -1

            # Bias towards up moves in bull market
            if np.random.random() < up_probability:
                move = abs(np.random.normal(trend_strength, volatility))
            else:
                move = -abs(np.random.normal(trend_strength * 0.3, volatility))

            new_price = prices[-1] * (1 + move / 100)
            prices.append(max(new_price, base_price * 0.1))  # Prevent negative prices

        return np.array(prices)

    def _generate_bear_market(self, base_price: float, candles: int, params: Dict) -> np.ndarray:
        """Generate bear market price series."""
        trend_strength = abs(params.get('trend_strength', -0.7))
        down_probability = params.get('down_day_probability', 0.7)
        volatility = params.get('daily_volatility', 0.025)
        persistence = params.get('trend_persistence', 0.8)

        prices = [base_price]
        trend = -1.0

        for i in range(candles - 1):
            # Trend persistence
            if np.random.random() > persistence:
                trend *= -1

            # Bias towards down moves in bear market
            if np.random.random() < down_probability:
                move = -abs(np.random.normal(trend_strength, volatility))
            else:
                move = abs(np.random.normal(trend_strength * 0.3, volatility))

            new_price = prices[-1] * (1 + move / 100)
            prices.append(max(new_price, base_price * 0.1))

        return np.array(prices)

    def _generate_sideways_market(self, base_price: float, candles: int, params: Dict) -> np.ndarray:
        """Generate sideways/range-bound market."""
        range_pct = params.get('range_percentage', 0.075)
        oscillation_freq = params.get('oscillation_frequency', 0.1)
        noise_level = params.get('noise_level', 0.02)
        mean_reversion = params.get('mean_reversion_strength', 0.6)

        upper_bound = base_price * (1 + range_pct)
        lower_bound = base_price * (1 - range_pct)

        prices = [base_price]

        for i in range(candles - 1):
            current_price = prices[-1]

            # Mean reversion force
            distance_from_center = (current_price - base_price) / base_price
            reversion_force = -distance_from_center * mean_reversion

            # Oscillation component
            oscillation = np.sin(i * oscillation_freq) * range_pct * 0.3

            # Random noise
            noise = np.random.normal(0, noise_level)

            # Combine forces
            total_move = reversion_force + oscillation + noise
            new_price = current_price * (1 + total_move)

            # Enforce range bounds with bouncing
            if new_price > upper_bound:
                new_price = upper_bound - (new_price - upper_bound) * 0.5
            elif new_price < lower_bound:
                new_price = lower_bound + (lower_bound - new_price) * 0.5

            prices.append(new_price)

        return np.array(prices)

    def _generate_high_volatility_market(self, base_price: float, candles: int, params: Dict) -> np.ndarray:
        """Generate high volatility market with large swings."""
        daily_vol = params.get('daily_volatility', 0.2)
        spike_prob = params.get('spike_probability', 0.3)
        spike_magnitude = params.get('spike_magnitude', 0.15)
        vol_clustering = params.get('volatility_clustering', 0.7)

        prices = [base_price]
        current_vol = daily_vol

        for i in range(candles - 1):
            # Volatility clustering - high vol tends to follow high vol
            vol_change = np.random.normal(0, daily_vol * 0.1)
            if np.random.random() < vol_clustering:
                current_vol = max(daily_vol * 0.5, min(daily_vol * 2, current_vol + vol_change))
            else:
                current_vol = daily_vol

            # Normal move
            move = np.random.normal(0, current_vol)

            # Occasional large spikes
            if np.random.random() < spike_prob:
                spike = np.random.choice([-1, 1]) * spike_magnitude
                move += spike

            new_price = prices[-1] * (1 + move)
            prices.append(max(new_price, base_price * 0.1))

        return np.array(prices)

    def _generate_low_volatility_market(self, base_price: float, candles: int, params: Dict) -> np.ndarray:
        """Generate low volatility market with minimal changes."""
        daily_vol = params.get('daily_volatility', 0.005)
        trend_strength = params.get('trend_strength', 0.1)
        noise_reduction = params.get('noise_reduction', 0.8)
        stability_factor = params.get('stability_factor', 0.9)

        prices = [base_price]

        for i in range(candles - 1):
            # Very small random moves
            move = np.random.normal(trend_strength / 100, daily_vol)

            # Apply noise reduction
            if np.random.random() < noise_reduction:
                move *= stability_factor

            new_price = prices[-1] * (1 + move)
            prices.append(new_price)

        return np.array(prices)

    def _generate_choppy_market(self, base_price: float, candles: int, params: Dict) -> np.ndarray:
        """Generate choppy market with frequent direction changes."""
        direction_change_prob = params.get('direction_change_probability', 0.4)
        whipsaw_intensity = params.get('whipsaw_intensity', 0.8)
        fake_breakout_prob = params.get('fake_breakout_probability', 0.3)
        noise_amplification = params.get('noise_amplification', 1.5)

        prices = [base_price]
        direction = 1

        for i in range(candles - 1):
            # Frequent direction changes
            if np.random.random() < direction_change_prob:
                direction *= -1

            # Base move
            move = direction * abs(np.random.normal(0.01, 0.02))

            # Add whipsaws - sudden reversals
            if np.random.random() < whipsaw_intensity * 0.1:
                move *= -2

            # Fake breakouts - strong move followed by reversal
            if np.random.random() < fake_breakout_prob * 0.05:
                move *= 3
                direction *= -1  # Set up for reversal next candle

            # Amplify noise
            noise = np.random.normal(0, 0.01) * noise_amplification
            move += noise

            new_price = prices[-1] * (1 + move)
            prices.append(max(new_price, base_price * 0.1))

        return np.array(prices)

    def _generate_gap_heavy_market(self, base_price: float, candles: int, params: Dict) -> np.ndarray:
        """Generate market with frequent price gaps."""
        gap_prob = params.get('gap_probability', 0.2)
        gap_magnitude_range = params.get('gap_magnitude_range', (0.02, 0.08))
        gap_direction_bias = params.get('gap_direction_bias', 0.0)
        post_gap_behavior = params.get('post_gap_behavior', 'continuation')

        prices = [base_price]
        post_gap_counter = 0
        gap_direction = 1

        for i in range(candles - 1):
            current_price = prices[-1]

            # Normal move
            move = np.random.normal(0, 0.02)

            # Gap logic
            if np.random.random() < gap_prob / 100:  # Adjust probability for realistic gaps
                gap_size = np.random.uniform(*gap_magnitude_range)
                gap_direction = np.random.choice([-1, 1])

                # Apply direction bias
                if gap_direction_bias != 0:
                    if np.random.random() < abs(gap_direction_bias):
                        gap_direction = 1 if gap_direction_bias > 0 else -1

                gap_move = gap_direction * gap_size
                move += gap_move
                post_gap_counter = 3  # Track for post-gap behavior

                logger.debug(f"Generated gap: {gap_move:.3f} at candle {i}")

            # Post-gap behavior
            if post_gap_counter > 0:
                if post_gap_behavior == 'continuation':
                    move += gap_direction * 0.005  # Slight continuation
                elif post_gap_behavior == 'reversal':
                    move -= gap_direction * 0.01  # Partial reversal
                post_gap_counter -= 1

            new_price = current_price * (1 + move)
            prices.append(max(new_price, base_price * 0.1))

        return np.array(prices)

    def _prices_to_ohlcv(self, timestamps: pd.DatetimeIndex, prices: np.ndarray, symbol: str) -> pd.DataFrame:
        """Convert price series to OHLCV format."""
        df = pd.DataFrame(index=timestamps)

        # For simplicity, use price as close and generate OHLV around it
        df['close'] = prices
        df['open'] = np.roll(prices, 1)
        df['open'].iloc[0] = prices[0]

        # Generate high/low with some randomness around price
        volatility = 0.005  # Small intrabar volatility
        high_noise = np.random.uniform(0, volatility, len(prices))
        low_noise = np.random.uniform(0, volatility, len(prices))

        df['high'] = np.maximum(df['open'], df['close']) * (1 + high_noise)
        df['low'] = np.minimum(df['open'], df['close']) * (1 - low_noise)

        # Ensure OHLC consistency
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))

        # Generate volume (somewhat correlated with price movement)
        price_changes = np.abs(np.diff(prices, prepend=prices[0]))
        base_volume = 1000000
        volume_multiplier = 1 + price_changes * 10  # Higher volume on larger moves
        df['volume'] = base_volume * volume_multiplier

        # Add symbol column
        df['symbol'] = symbol

        return df[['symbol', 'open', 'high', 'low', 'close', 'volume']]

    def execute_scenarios(
        self,
        strategy_name: str,
        backtest_function,
        scenario_filter: Optional[List[str]] = None,
        parallel_execution: bool = True
    ) -> Dict[str, ScenarioResult]:
        """
        Execute backtesting across multiple scenarios.

        Args:
            strategy_name: Name of strategy being tested
            backtest_function: Function to execute backtesting
            scenario_filter: Optional list of scenario names to run
            parallel_execution: Whether to run scenarios in parallel

        Returns:
            Dictionary mapping scenario names to results
        """
        enabled_scenarios = [
            config for config in self.scenario_configs.values()
            if config.enabled and (not scenario_filter or config.name in scenario_filter)
        ]

        if not enabled_scenarios:
            logger.warning("No enabled scenarios found")
            return {}

        logger.info(f"Executing {len(enabled_scenarios)} scenarios for strategy: {strategy_name}")

        results = {}

        if parallel_execution and len(enabled_scenarios) > 1:
            results = self._execute_scenarios_parallel(
                strategy_name, backtest_function, enabled_scenarios
            )
        else:
            results = self._execute_scenarios_sequential(
                strategy_name, backtest_function, enabled_scenarios
            )

        # Store results for analysis
        if strategy_name not in self.scenario_results:
            self.scenario_results[strategy_name] = []

        self.scenario_results[strategy_name].extend(results.values())

        # Generate performance analysis
        self._generate_performance_analysis(strategy_name)

        return results

    def _execute_scenarios_parallel(
        self,
        strategy_name: str,
        backtest_function,
        scenarios: List[ScenarioConfig]
    ) -> Dict[str, ScenarioResult]:
        """Execute scenarios in parallel using ThreadPoolExecutor."""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all scenarios
            future_to_scenario = {
                executor.submit(self._execute_single_scenario, strategy_name, backtest_function, scenario): scenario
                for scenario in scenarios
            }

            # Collect results as they complete
            for future in as_completed(future_to_scenario):
                scenario = future_to_scenario[future]
                try:
                    result = future.result()
                    results[scenario.name] = result
                    logger.info(f"Completed scenario: {scenario.name}")
                except Exception as e:
                    logger.error(f"Scenario {scenario.name} failed: {e}")
                    results[scenario.name] = ScenarioResult(
                        scenario_type=scenario.scenario_type,
                        scenario_name=scenario.name,
                        strategy_name=strategy_name,
                        execution_time=0.0,
                        total_return=0.0,
                        total_trades=0,
                        winning_trades=0,
                        losing_trades=0,
                        win_rate=0.0,
                        sharpe_ratio=0.0,
                        max_drawdown=0.0,
                        avg_trade_duration=0.0,
                        profit_factor=0.0,
                        error=str(e)
                    )

        return results

    def _execute_scenarios_sequential(
        self,
        strategy_name: str,
        backtest_function,
        scenarios: List[ScenarioConfig]
    ) -> Dict[str, ScenarioResult]:
        """Execute scenarios sequentially."""
        results = {}

        for scenario in scenarios:
            logger.info(f"Executing scenario: {scenario.name}")
            try:
                result = self._execute_single_scenario(strategy_name, backtest_function, scenario)
                results[scenario.name] = result
            except Exception as e:
                logger.error(f"Scenario {scenario.name} failed: {e}")
                results[scenario.name] = ScenarioResult(
                    scenario_type=scenario.scenario_type,
                    scenario_name=scenario.name,
                    strategy_name=strategy_name,
                    execution_time=0.0,
                    total_return=0.0,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    avg_trade_duration=0.0,
                    profit_factor=0.0,
                    error=str(e)
                )

        return results

    def _execute_single_scenario(
        self,
        strategy_name: str,
        backtest_function,
        scenario: ScenarioConfig
    ) -> ScenarioResult:
        """Execute a single scenario backtest."""
        start_time = time.time()

        try:
            # Generate scenario data
            # For now, we'll pass scenario info to backtest function
            # The actual data generation will be handled by the calling code
            # This maintains separation of concerns

            result = backtest_function(
                scenario_type=scenario.scenario_type,
                scenario_config=scenario,
                strategy_name=strategy_name
            )

            execution_time = time.time() - start_time

            # Extract metrics from backtest result
            # This assumes the backtest function returns a structured result
            scenario_result = ScenarioResult(
                scenario_type=scenario.scenario_type,
                scenario_name=scenario.name,
                strategy_name=strategy_name,
                execution_time=execution_time,
                total_return=getattr(result, 'total_return', 0.0),
                total_trades=getattr(result, 'total_trades', 0),
                winning_trades=getattr(result, 'winning_trades', 0),
                losing_trades=getattr(result, 'losing_trades', 0),
                win_rate=getattr(result, 'win_rate', 0.0),
                sharpe_ratio=getattr(result, 'sharpe_ratio', 0.0),
                max_drawdown=getattr(result, 'max_drawdown', 0.0),
                avg_trade_duration=getattr(result, 'avg_trade_duration', 0.0),
                profit_factor=getattr(result, 'profit_factor', 0.0),
                metadata=getattr(result, 'metadata', {})
            )

            return scenario_result

        except Exception as e:
            execution_time = time.time() - start_time
            raise Exception(f"Scenario execution failed after {execution_time:.2f}s: {e}")

    def _generate_performance_analysis(self, strategy_name: str) -> None:
        """Generate comprehensive performance analysis for a strategy."""
        if strategy_name not in self.scenario_results:
            return

        results = self.scenario_results[strategy_name]
        successful_results = [r for r in results if r.success]

        if not successful_results:
            logger.warning(f"No successful results for strategy: {strategy_name}")
            return

        # Calculate consistency score (inverse of coefficient of variation)
        returns = [r.total_return for r in successful_results]
        consistency_score = 1.0 / (np.std(returns) / np.mean(returns)) if returns and np.mean(returns) != 0 else 0.0

        # Calculate adaptability score (performance across diverse conditions)
        scenario_types = set(r.scenario_type for r in successful_results)
        adaptability_score = len(scenario_types) / len(ScenarioType) * np.mean(returns) if returns else 0.0

        # Calculate risk-adjusted robustness
        sharpe_ratios = [r.sharpe_ratio for r in successful_results if r.sharpe_ratio != 0]
        robustness_score = np.mean(sharpe_ratios) if sharpe_ratios else 0.0

        # Find best and worst scenarios
        best_scenario = max(successful_results, key=lambda x: x.total_return).scenario_name
        worst_scenario = min(successful_results, key=lambda x: x.total_return).scenario_name
        worst_case_performance = min(r.total_return for r in successful_results)

        # Calculate scenario correlations (simplified)
        scenario_correlation = {}
        for result in successful_results:
            scenario_correlation[result.scenario_name] = result.total_return / max(returns) if returns else 0.0

        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(successful_results)

        analysis = ScenarioPerformanceAnalysis(
            strategy_name=strategy_name,
            total_scenarios=len(results),
            successful_scenarios=len(successful_results),
            consistency_score=consistency_score,
            adaptability_score=adaptability_score,
            robustness_score=robustness_score,
            worst_case_performance=worst_case_performance,
            best_scenario=best_scenario,
            worst_scenario=worst_scenario,
            scenario_correlation=scenario_correlation,
            optimization_recommendations=recommendations
        )

        self.performance_analyses[strategy_name] = analysis
        logger.info(f"Generated performance analysis for {strategy_name}")

    def _generate_optimization_recommendations(self, results: List[ScenarioResult]) -> List[str]:
        """Generate optimization recommendations based on scenario performance."""
        recommendations = []

        # Analyze win rates across scenarios
        win_rates = [r.win_rate for r in results if r.win_rate > 0]
        if win_rates and np.mean(win_rates) < 0.4:
            recommendations.append("Consider tightening entry criteria to improve win rate")

        # Analyze max drawdowns
        drawdowns = [abs(r.max_drawdown) for r in results if r.max_drawdown != 0]
        if drawdowns and np.mean(drawdowns) > 0.15:
            recommendations.append("Consider implementing stricter risk management to reduce drawdowns")

        # Analyze Sharpe ratios
        sharpe_ratios = [r.sharpe_ratio for r in results if r.sharpe_ratio != 0]
        if sharpe_ratios and np.mean(sharpe_ratios) < 1.0:
            recommendations.append("Consider optimizing risk-adjusted returns through position sizing")

        # Scenario-specific recommendations
        scenario_performance = {r.scenario_name: r.total_return for r in results}

        if scenario_performance.get('Bull Market', 0) < 0:
            recommendations.append("Strategy underperforms in bull markets - consider trend-following components")

        if scenario_performance.get('Bear Market', 0) > -0.1:
            recommendations.append("Strategy shows good bear market resilience")

        if scenario_performance.get('Choppy Market', 0) < -0.2:
            recommendations.append("Strategy struggles in choppy conditions - consider filters for market regime")

        return recommendations[:5]  # Limit to top 5 recommendations

    def get_performance_analysis(self, strategy_name: str) -> Optional[ScenarioPerformanceAnalysis]:
        """Get performance analysis for a strategy."""
        return self.performance_analyses.get(strategy_name)

    def get_scenario_results(self, strategy_name: str) -> List[ScenarioResult]:
        """Get all scenario results for a strategy."""
        return self.scenario_results.get(strategy_name, [])

    def export_scenario_data(
        self,
        strategy_name: str,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export scenario results and analysis to JSON file.

        Args:
            strategy_name: Strategy to export data for
            output_path: Optional custom output path

        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"scenario_analysis_{strategy_name}_{timestamp}.json")

        export_data = {
            'strategy_name': strategy_name,
            'export_timestamp': datetime.now().isoformat(),
            'scenario_results': [
                {
                    'scenario_type': result.scenario_type.value,
                    'scenario_name': result.scenario_name,
                    'execution_time': result.execution_time,
                    'total_return': result.total_return,
                    'total_trades': result.total_trades,
                    'winning_trades': result.winning_trades,
                    'losing_trades': result.losing_trades,
                    'win_rate': result.win_rate,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'avg_trade_duration': result.avg_trade_duration,
                    'profit_factor': result.profit_factor,
                    'success': result.success,
                    'error': result.error,
                    'metadata': result.metadata
                }
                for result in self.get_scenario_results(strategy_name)
            ]
        }

        # Add performance analysis if available
        analysis = self.get_performance_analysis(strategy_name)
        if analysis:
            export_data['performance_analysis'] = {
                'total_scenarios': analysis.total_scenarios,
                'successful_scenarios': analysis.successful_scenarios,
                'consistency_score': analysis.consistency_score,
                'adaptability_score': analysis.adaptability_score,
                'robustness_score': analysis.robustness_score,
                'worst_case_performance': analysis.worst_case_performance,
                'best_scenario': analysis.best_scenario,
                'worst_scenario': analysis.worst_scenario,
                'scenario_correlation': analysis.scenario_correlation,
                'optimization_recommendations': analysis.optimization_recommendations
            }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported scenario data to: {output_path}")
        return output_path

    def generate_scenario_summary(self, strategy_name: str) -> str:
        """Generate a human-readable summary of scenario performance."""
        results = self.get_scenario_results(strategy_name)
        analysis = self.get_performance_analysis(strategy_name)

        if not results:
            return f"No scenario results available for {strategy_name}"

        successful_results = [r for r in results if r.success]

        summary = f"""
Scenario Analysis Summary for {strategy_name}
{'=' * 50}

Execution Overview:
- Total Scenarios: {len(results)}
- Successful: {len(successful_results)}
- Failed: {len(results) - len(successful_results)}

Performance Metrics:
"""

        if successful_results:
            returns = [r.total_return for r in successful_results]
            win_rates = [r.win_rate for r in successful_results]
            sharpe_ratios = [r.sharpe_ratio for r in successful_results if r.sharpe_ratio != 0]

            avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else None
            sharpe_display = f"{avg_sharpe:.2f}" if avg_sharpe is not None else "N/A"

            summary += f"""- Average Return: {np.mean(returns):.2%}
- Return Range: {min(returns):.2%} to {max(returns):.2%}
- Average Win Rate: {np.mean(win_rates):.1%}
- Average Sharpe Ratio: {sharpe_display}
"""

        if analysis:
            summary += f"""
Robustness Analysis:
- Consistency Score: {analysis.consistency_score:.2f}
- Adaptability Score: {analysis.adaptability_score:.2f}
- Risk-Adjusted Robustness: {analysis.robustness_score:.2f}
- Best Scenario: {analysis.best_scenario}
- Worst Scenario: {analysis.worst_scenario}

Optimization Recommendations:
"""
            for i, rec in enumerate(analysis.optimization_recommendations, 1):
                summary += f"{i}. {rec}\n"

        # Individual scenario results
        summary += "\nScenario Results:\n"
        for result in sorted(successful_results, key=lambda x: x.total_return, reverse=True):
            summary += f"- {result.scenario_name}: {result.total_return:.2%} return, {result.win_rate:.1%} win rate\n"

    def _analyze_individual_worst_case(self, result: ScenarioResult) -> Dict[str, Any]:
        """Analyze an individual worst-case scenario result."""
        analysis = {
            "primary_issues": [],
            "performance_breakdown": {
                "total_return": result.total_return,
                "win_rate": result.win_rate,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "total_trades": result.total_trades
            }
        }

        # Identify primary issues
        if result.total_return < -0.1:
            analysis["primary_issues"].append("Significant negative returns")
        if result.win_rate < 0.4:
            analysis["primary_issues"].append("Low win rate")
        if result.sharpe_ratio and result.sharpe_ratio < 0:
            analysis["primary_issues"].append("Negative risk-adjusted returns")
        if result.max_drawdown and abs(result.max_drawdown) > 0.2:
            analysis["primary_issues"].append("High maximum drawdown")

        return analysis

    def _identify_systematic_weaknesses(self, scenario_results: List[ScenarioResult]) -> List[str]:
        """Identify systematic weaknesses across scenarios."""
        weaknesses = []

        # Analyze patterns across all scenarios
        negative_returns = sum(1 for r in scenario_results if r.total_return < 0)
        low_win_rates = sum(1 for r in scenario_results if r.win_rate < 0.45)
        high_drawdowns = sum(1 for r in scenario_results if r.max_drawdown and abs(r.max_drawdown) > 0.15)
        negative_sharpes = sum(1 for r in scenario_results if r.sharpe_ratio and r.sharpe_ratio < 0)

        total_scenarios = len(scenario_results)

        if negative_returns / total_scenarios > 0.4:
            weaknesses.append("Frequently produces negative returns across different market conditions")
        if low_win_rates / total_scenarios > 0.5:
            weaknesses.append("Consistently low win rates suggest poor entry/exit timing")
        if high_drawdowns / total_scenarios > 0.3:
            weaknesses.append("Prone to large drawdowns indicating poor risk management")
        if negative_sharpes / total_scenarios > 0.4:
            weaknesses.append("Poor risk-adjusted returns suggest inefficient risk-taking")

        return weaknesses

    def _calculate_worst_case_impact(self, scenario_results: List[ScenarioResult]) -> Dict[str, Any]:
        """Calculate the impact of worst-case scenarios."""
        returns = [r.total_return for r in scenario_results]
        worst_return = min(returns)
        avg_return = np.mean(returns)

        impact = {
            "worst_case_return": worst_return,
            "average_return": avg_return,
            "impact_magnitude": abs(worst_return - avg_return),
            "worst_case_frequency": sum(1 for r in returns if r <= worst_return * 1.1) / len(returns),
            "severity_level": "high" if worst_return < -0.2 else "medium" if worst_return < -0.1 else "low"
        }

        return impact

    def _generate_worst_case_improvements(self, scenario_results: List[ScenarioResult]) -> List[str]:
        """Generate specific improvements for worst-case scenarios."""
        improvements = []

        worst_result = min(scenario_results, key=lambda x: x.total_return)

        if worst_result.win_rate < 0.4:
            improvements.append("Improve signal quality to increase win rate")
        if worst_result.max_drawdown and abs(worst_result.max_drawdown) > 0.2:
            improvements.append("Implement tighter stop-loss or position sizing controls")
        if worst_result.sharpe_ratio and worst_result.sharpe_ratio < -0.5:
            improvements.append("Review risk-reward ratio of trades")
        if worst_result.total_trades and worst_result.total_trades < 5:
            improvements.append("Strategy may be too selective, consider relaxing entry criteria")

        return improvements

    def _identify_scenario_issues(self, result: ScenarioResult) -> List[str]:
        """Identify specific issues with a scenario result."""
        issues = []

        if result.total_return < -0.05:
            issues.append("negative_returns")
        if result.win_rate < 0.45:
            issues.append("low_win_rate")
        if result.sharpe_ratio and result.sharpe_ratio < 0:
            issues.append("poor_risk_adjustment")
        if result.max_drawdown and abs(result.max_drawdown) > 0.15:
            issues.append("high_drawdown")

        return issues

    def _identify_failure_patterns(self, scenario_results: List[ScenarioResult]) -> Dict[str, Any]:
        """Identify patterns in scenario failures."""
        patterns = {
            "volatility_sensitivity": False,
            "trend_dependency": False,
            "market_regime_issues": False
        }

        # Check volatility sensitivity
        high_vol_scenarios = [r for r in scenario_results if r.scenario_type in [ScenarioType.HIGH_VOLATILITY, ScenarioType.CHOPPY]]
        if high_vol_scenarios:
            high_vol_performance = np.mean([self._calculate_scenario_performance_score(r) for r in high_vol_scenarios])
            if high_vol_performance < 40:
                patterns["volatility_sensitivity"] = True

        # Check trend dependency
        trend_scenarios = [r for r in scenario_results if r.scenario_type in [ScenarioType.BULL, ScenarioType.BEAR]]
        sideways_scenarios = [r for r in scenario_results if r.scenario_type == ScenarioType.SIDEWAYS]

        if trend_scenarios and sideways_scenarios:
            trend_performance = np.mean([self._calculate_scenario_performance_score(r) for r in trend_scenarios])
            sideways_performance = np.mean([self._calculate_scenario_performance_score(r) for r in sideways_scenarios])

            if abs(trend_performance - sideways_performance) > 30:
                patterns["trend_dependency"] = True

        return patterns

    def _generate_correlation_insights(self, scenario_results: List[ScenarioResult], struggling_scenarios: List[Dict]) -> List[str]:
        """Generate insights from correlation analysis."""
        insights = []

        if struggling_scenarios:
            struggling_types = [s["scenario_type"] for s in struggling_scenarios]

            if "high_volatility" in struggling_types or "choppy" in struggling_types:
                insights.append("Strategy struggles in high volatility conditions")
            if "sideways" in struggling_types:
                insights.append("Strategy performs poorly in range-bound markets")
            if "bear" in struggling_types:
                insights.append("Strategy has difficulty with downtrending markets")

            if len(struggling_scenarios) > len(scenario_results) / 2:
                insights.append("Strategy shows poor robustness across multiple market conditions")

        return insights

    def _generate_scenario_specific_recommendations(self, scenario_type: str, scenario_results: List[ScenarioResult]) -> List[str]:
        """Generate recommendations specific to struggling scenario types."""
        recommendations = []

        if scenario_type == "high_volatility":
            recommendations.append("Consider volatility-based position sizing")
            recommendations.append("Implement tighter stop losses for high volatility periods")
        elif scenario_type == "sideways":
            recommendations.append("Add range detection and adjust strategy for sideways markets")
            recommendations.append("Consider mean reversion components for range-bound conditions")
        elif scenario_type == "bear":
            recommendations.append("Implement bear market detection and defensive positioning")
            recommendations.append("Consider short-bias or hedging strategies")
        elif scenario_type == "choppy":
            recommendations.append("Add trend strength filters to avoid whipsaw conditions")
            recommendations.append("Increase minimum trend duration requirements")

        return recommendations

    def get_robustness_validation_report(self, strategy_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive robustness validation report for StrategyValidator integration.

        Args:
            strategy_name: Name of the strategy to validate

        Returns:
            Comprehensive robustness validation report
        """
        try:
            if strategy_name not in self.scenario_results:
                return {"error": f"No results found for strategy: {strategy_name}"}

            scenario_results = self.scenario_results[strategy_name]

            # Calculate all robustness metrics
            consistency_score = self.calculate_consistency_score(scenario_results)
            adaptability_score = self.compute_adaptability_score(scenario_results)
            risk_adjusted_robustness = self.generate_risk_adjusted_robustness(scenario_results)
            worst_case_analysis = self.analyze_worst_case_scenarios(scenario_results)
            correlation_analysis = self.calculate_scenario_correlation(scenario_results)
            optimization_recommendations = self.generate_optimization_recommendations_from_weak_scenarios(scenario_results)

            # Generate validation report
            validation_report = {
                "strategy_name": strategy_name,
                "validation_timestamp": datetime.now().isoformat(),
                "robustness_metrics": {
                    "consistency_score": consistency_score,
                    "adaptability_score": adaptability_score,
                    "risk_adjusted_robustness": risk_adjusted_robustness,
                    "overall_robustness": (consistency_score + adaptability_score + risk_adjusted_robustness) / 3
                },
                "worst_case_analysis": worst_case_analysis,
                "scenario_correlation": correlation_analysis,
                "optimization_recommendations": optimization_recommendations,
                "validation_summary": self._generate_validation_summary(
                    consistency_score, adaptability_score, risk_adjusted_robustness
                ),
                "pass_fail_assessment": self._assess_robustness_pass_fail(
                    consistency_score, adaptability_score, risk_adjusted_robustness
                )
            }

            return validation_report

        except Exception as e:
            logger.error(f"Error generating robustness validation report: {e}")
            return {"error": f"Validation report generation failed: {str(e)}"}

    def _generate_validation_summary(self, consistency: float, adaptability: float, risk_adjusted: float) -> str:
        """Generate a summary of the robustness validation."""
        overall = (consistency + adaptability + risk_adjusted) / 3

        if overall >= 80:
            level = "Excellent"
        elif overall >= 70:
            level = "Good"
        elif overall >= 60:
            level = "Acceptable"
        elif overall >= 50:
            level = "Poor"
        else:
            level = "Unacceptable"

        return f"""Robustness Assessment: {level} (Score: {overall:.1f}/100)
- Consistency: {consistency:.1f}/100
- Adaptability: {adaptability:.1f}/100
- Risk-Adjusted Performance: {risk_adjusted:.1f}/100"""

    def _assess_robustness_pass_fail(self, consistency: float, adaptability: float, risk_adjusted: float) -> Dict[str, Any]:
        """Assess whether the strategy passes robustness validation."""
        overall = (consistency + adaptability + risk_adjusted) / 3

        # Define pass/fail thresholds
        min_overall = 60
        min_individual = 40

        passes = {
            "overall_pass": overall >= min_overall,
            "consistency_pass": consistency >= min_individual,
            "adaptability_pass": adaptability >= min_individual,
            "risk_adjusted_pass": risk_adjusted >= min_individual
        }

        overall_pass = all(passes.values())

        return {
            "overall_pass": overall_pass,
            "individual_passes": passes,
            "overall_score": overall,
            "pass_threshold": min_overall,
            "individual_threshold": min_individual,
            "recommendation": "APPROVED" if overall_pass else "REQUIRES_IMPROVEMENT"
        }
