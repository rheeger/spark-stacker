"""
Strategy Backtest Manager

This module provides strategy-specific backtesting capabilities:
- Initialize with StrategyConfig object from config.json
- Load and validate all strategy indicators from config
- Set up position sizing based on strategy configuration
- Configure data sources based on strategy market and exchange
- Handle strategy-specific timeframe and data requirements
- Add comprehensive error handling and validation
- Integrate with new ConfigManager and DataManager modules
- Use new validation modules for comprehensive strategy validation
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from app.backtesting.backtest_engine import BacktestEngine, BacktestResult
from app.backtesting.data_manager import DataManager
from app.backtesting.simulation_engine import SimulationEngine
from app.connectors.base_connector import OrderSide, OrderType
from app.core.strategy_config import StrategyConfig
from app.indicators.base_indicator import (BaseIndicator, Signal,
                                           SignalDirection)
from app.indicators.indicator_factory import IndicatorFactory
from app.risk_management.position_sizing import (PositionSizer,
                                                 PositionSizingConfig)
from core.config_manager import ConfigManager
from core.data_manager import DataManager as CLIDataManager
from core.data_manager import DataRequest, DataSourceType
from validation.strategy_validator import StrategyValidator

logger = logging.getLogger(__name__)


class StrategyBacktestManager:
    """
    Manages backtesting of complete strategies with proper configuration,
    position sizing, and comprehensive reporting.

    This class provides a standardized interface for strategy-driven backtesting
    that works with strategy configurations from config.json.
    """

    def __init__(
        self,
        backtest_engine: BacktestEngine,
        config_manager: ConfigManager,
        data_manager: CLIDataManager,
        strategy_validator: Optional[StrategyValidator] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the strategy backtest manager.

        Args:
            backtest_engine: Instance of the BacktestEngine
            config_manager: ConfigManager for configuration access
            data_manager: DataManager for data operations
            strategy_validator: Optional validator for strategy validation
            output_dir: Optional output directory for results
        """
        self.backtest_engine = backtest_engine
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.strategy_validator = strategy_validator or StrategyValidator(config_manager)
        self.output_dir = output_dir or Path("./backtest_results")

        # Strategy components
        self.current_strategy: Optional[StrategyConfig] = None
        self.strategy_indicators: Dict[str, BaseIndicator] = {}
        self.position_sizer: Optional[PositionSizer] = None
        self.indicator_factory = IndicatorFactory()

        # Results storage
        self.results: Dict[str, BacktestResult] = {}
        self.performance_cache: Dict[str, Dict[str, Any]] = {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized StrategyBacktestManager with output dir: {self.output_dir}")

    def load_strategy_from_config(self, strategy_name: str) -> StrategyConfig:
        """
        Load a strategy configuration from config.json.

        Args:
            strategy_name: Name of the strategy to load

        Returns:
            StrategyConfig object

        Raises:
            ValueError: If strategy not found or invalid
        """
        try:
            # Get strategy configuration from config manager
            strategy_config_dict = self.config_manager.get_strategy_config(strategy_name)

            if not strategy_config_dict:
                raise ValueError(f"Strategy '{strategy_name}' not found in configuration")

            # Convert to StrategyConfig object
            strategy_config = StrategyConfig.from_dict(strategy_config_dict)

            # Validate the strategy configuration
            validation_result = self.strategy_validator.validate_strategy_config(strategy_config)
            if not validation_result.is_valid:
                raise ValueError(f"Strategy validation failed: {', '.join(validation_result.errors)}")

            self.current_strategy = strategy_config
            logger.info(f"Loaded strategy: {strategy_name}")
            logger.info(f"  Market: {strategy_config.market}")
            logger.info(f"  Exchange: {strategy_config.exchange}")
            logger.info(f"  Timeframe: {strategy_config.timeframe}")
            logger.info(f"  Indicators: {len(strategy_config.indicators)}")

            return strategy_config

        except Exception as e:
            logger.error(f"Failed to load strategy '{strategy_name}': {e}")
            raise

    def initialize_strategy_components(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize all strategy components (indicators, position sizer, etc.).

        Args:
            overrides: Optional configuration overrides for testing

        Raises:
            ValueError: If strategy not loaded or component initialization fails
        """
        if not self.current_strategy:
            raise ValueError("No strategy loaded. Call load_strategy_from_config first.")

        try:
            # Apply any overrides to the current strategy
            effective_config = self.current_strategy.to_dict()
            if overrides:
                # Deep merge overrides
                for key, value in overrides.items():
                    if isinstance(value, dict) and key in effective_config and isinstance(effective_config[key], dict):
                        effective_config[key].update(value)
                    else:
                        effective_config[key] = value

                # Update the current strategy with overrides
                self.current_strategy = StrategyConfig.from_dict(effective_config)
                logger.info(f"Applied configuration overrides: {overrides}")

            # Initialize indicators
            self._initialize_indicators(effective_config)

            # Initialize position sizer
            self._initialize_position_sizer(effective_config)

            logger.info("Strategy components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize strategy components: {e}")
            raise

    def _initialize_indicators(self, config: Dict[str, Any]) -> None:
        """Initialize all indicators for the strategy."""
        self.strategy_indicators = {}

        # Get the strategy's indicator names
        strategy_indicator_names = config.get('indicators', [])

        # Get the global config to access indicator definitions
        global_config = self.config_manager.load_config()
        available_indicators = {ind.get('name'): ind for ind in global_config.get('indicators', [])}

        for indicator_name in strategy_indicator_names:
            if indicator_name not in available_indicators:
                logger.warning(f"Indicator '{indicator_name}' referenced by strategy but not found in configuration")
                continue

            indicator_config = available_indicators[indicator_name]

            try:
                # Create indicator instance
                indicator = self.indicator_factory.create_indicator(
                    name=indicator_name,
                    indicator_type=indicator_config['type'],
                    params=indicator_config.get('parameters', {})
                )

                self.strategy_indicators[indicator_name] = indicator
                logger.debug(f"Initialized indicator: {indicator_name} ({indicator_config['type']})")

            except Exception as e:
                logger.error(f"Failed to initialize indicator '{indicator_name}': {e}")
                raise

        logger.info(f"Initialized {len(self.strategy_indicators)} indicators")

    def _initialize_position_sizer(self, config: Dict[str, Any]) -> None:
        """Initialize position sizer based on strategy configuration."""
        try:
            # Use ConfigManager to get properly merged position sizing configuration
            if self.current_strategy:
                # Get effective position sizing configuration with full inheritance
                position_sizing_config = self.config_manager.get_effective_position_sizing_config(
                    self.current_strategy.name
                )

                # Validate position sizing inheritance
                validation_issues = self.config_manager.validate_position_sizing_inheritance(
                    self.current_strategy.name
                )

                if validation_issues:
                    logger.warning(f"Position sizing validation issues for {self.current_strategy.name}: "
                                 f"{', '.join(validation_issues)}")
            else:
                # Fallback to direct config if no current strategy
                position_sizing_config = config.get('position_sizing', {})

                if not position_sizing_config:
                    global_config = self.config_manager.load_config()
                    position_sizing_config = global_config.get('position_sizing', {})

            # Create position sizing configuration
            sizing_config = PositionSizingConfig.from_config_dict(position_sizing_config)

            # Initialize position sizer
            self.position_sizer = PositionSizer(sizing_config)

            logger.info(f"Initialized position sizer: {sizing_config.method.value}")
            logger.debug(f"Position sizing config: {position_sizing_config}")

        except Exception as e:
            logger.error(f"Failed to initialize position sizer: {e}")
            raise

    def apply_position_sizing_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Apply temporary position sizing overrides for testing purposes.

        Args:
            overrides: Dictionary of position sizing parameters to override
        """
        if not self.position_sizer:
            logger.warning("Cannot apply position sizing overrides: position sizer not initialized")
            return

        try:
            # Get current configuration
            current_config = self.position_sizer.config

            # Create new configuration with overrides
            config_dict = {
                'method': current_config.method.value,
                'fixed_usd_amount': current_config.fixed_usd_amount,
                'equity_percentage': current_config.equity_percentage,
                'risk_per_trade_pct': current_config.risk_per_trade_pct,
                'default_stop_loss_pct': current_config.default_stop_loss_pct,
                'fixed_units': current_config.fixed_units,
                'kelly_win_rate': current_config.kelly_win_rate,
                'kelly_avg_win': current_config.kelly_avg_win,
                'kelly_avg_loss': current_config.kelly_avg_loss,
                'kelly_max_position_pct': current_config.kelly_max_position_pct,
                'max_position_size_usd': current_config.max_position_size_usd,
                'min_position_size_usd': current_config.min_position_size_usd,
                'max_leverage': current_config.max_leverage,
            }

            # Apply overrides
            config_dict.update(overrides)

            # Create new position sizer with overrides
            new_sizing_config = PositionSizingConfig.from_config_dict(config_dict)
            self.position_sizer = PositionSizer(new_sizing_config)

            logger.info(f"Applied position sizing overrides: {overrides}")
            logger.debug(f"New position sizing config: {config_dict}")

        except Exception as e:
            logger.error(f"Failed to apply position sizing overrides: {e}")
            raise

    def get_current_position_sizing_info(self) -> Dict[str, Any]:
        """
        Get information about the current position sizing configuration.

        Returns:
            Dictionary containing current position sizing details
        """
        if not self.position_sizer:
            return {"error": "Position sizer not initialized"}

        config = self.position_sizer.config

        return {
            "method": config.method.value,
            "fixed_usd_amount": config.fixed_usd_amount,
            "equity_percentage": config.equity_percentage,
            "risk_per_trade_pct": config.risk_per_trade_pct,
            "default_stop_loss_pct": config.default_stop_loss_pct,
            "fixed_units": config.fixed_units,
            "kelly_win_rate": config.kelly_win_rate,
            "kelly_avg_win": config.kelly_avg_win,
            "kelly_avg_loss": config.kelly_avg_loss,
            "kelly_max_position_pct": config.kelly_max_position_pct,
            "max_position_size_usd": config.max_position_size_usd,
            "min_position_size_usd": config.min_position_size_usd,
            "max_leverage": config.max_leverage,
        }

    def backtest_strategy(
        self,
        days: int = 30,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        use_real_data: bool = True,
        leverage: float = 1.0,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """
        Run a backtest for the loaded strategy.

        Args:
            days: Number of days to test (used if start/end dates not provided)
            start_date: Optional start date for backtest
            end_date: Optional end date for backtest
            use_real_data: Whether to use real market data
            leverage: Leverage to apply
            additional_params: Additional parameters for the strategy

        Returns:
            BacktestResult with performance metrics

        Raises:
            ValueError: If strategy not loaded or backtest fails
        """
        if not self.current_strategy:
            raise ValueError("No strategy loaded. Call load_strategy_from_config first.")

        if not self.strategy_indicators:
            raise ValueError("Strategy indicators not initialized. Call initialize_strategy_components first.")

        try:
            # Determine date range
            if not end_date:
                end_date = datetime.now()
            elif isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")

            if not start_date:
                start_date = end_date - timedelta(days=days)
            elif isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")

            # Get market data using CLI DataManager
            market_data = self._get_market_data(
                symbol=self.current_strategy.market,
                timeframe=self.current_strategy.timeframe,
                start_date=start_date,
                end_date=end_date,
                use_real_data=use_real_data
            )

            # Create strategy function
            strategy_func = self._create_strategy_function()

            # Prepare strategy parameters
            strategy_params = additional_params or {}

            # Run backtest
            logger.info(f"Running strategy backtest: {self.current_strategy.name}")
            logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")
            logger.info(f"  Data points: {len(market_data)}")

            start_time = time.time()

            result = self.backtest_engine.run_backtest(
                strategy_func=strategy_func,
                symbol=self.current_strategy.market,
                interval=self.current_strategy.timeframe,
                start_date=start_date,
                end_date=end_date,
                data_source_name="cli_data_manager",  # Use our CLI data manager
                strategy_params=strategy_params,
                leverage=leverage,
                indicators=list(self.strategy_indicators.values())
            )

            execution_time = time.time() - start_time

            # Enhance result with strategy-specific metadata
            result.metadata = {
                'strategy_name': self.current_strategy.name,
                'strategy_config': self.current_strategy.to_dict(),
                'execution_time': execution_time,
                'data_source': 'real' if use_real_data else 'synthetic',
                'leverage': leverage
            }

            # Store result
            result_key = f"{self.current_strategy.name}_{start_date.date()}_{end_date.date()}"
            self.results[result_key] = result

            logger.info(f"Strategy backtest completed in {execution_time:.2f}s")
            logger.info(f"  Total trades: {result.metrics.get('total_trades', 0)}")
            logger.info(f"  Total return: {result.metrics.get('total_return_pct', 0):.2f}%")

            return result

        except Exception as e:
            logger.error(f"Strategy backtest failed: {e}")
            raise

    def _get_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_real_data: bool
    ) -> pd.DataFrame:
        """Get market data for the strategy using CLI DataManager."""
        try:
            if use_real_data:
                # Use CLI data manager to fetch real data
                request = DataRequest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    source_preference=[DataSourceType.REAL_EXCHANGE, DataSourceType.CACHED]
                )
                data = self.data_manager.get_data(request)
            else:
                # Use CLI data manager to generate synthetic data
                data = self.data_manager.generate_synthetic_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    scenario_type='default'
                )

            if data.empty:
                raise ValueError(f"No market data available for {symbol} {timeframe}")

            logger.debug(f"Retrieved {len(data)} data points for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            raise

    def _create_strategy_function(self) -> callable:
        """
        Create a strategy function that integrates all strategy indicators.

        Returns:
            Strategy function compatible with BacktestEngine
        """
        def strategy_function(
            data: pd.DataFrame,
            simulation_engine: SimulationEngine,
            params: Dict[str, Any]
        ) -> None:
            """
            Integrated strategy function that uses all configured indicators.
            """
            symbol = params.get("symbol")
            current_candle = params.get("current_candle")
            leverage = params.get("leverage", 1.0)

            if not (symbol and current_candle is not None):
                logger.warning("Missing required parameters in strategy call")
                return

            # Calculate all indicator values
            data_with_indicators = data.copy()

            for indicator_name, indicator in self.strategy_indicators.items():
                try:
                    indicator_data = indicator.calculate(data)
                    # Merge indicator columns into main data
                    for col in indicator_data.columns:
                        if col not in data.columns:
                            data_with_indicators[col] = indicator_data[col]
                except Exception as e:
                    logger.warning(f"Failed to calculate indicator {indicator_name}: {e}")
                    continue

            # Generate signals from all indicators
            signals = []
            for indicator_name, indicator in self.strategy_indicators.items():
                try:
                    signal = indicator.generate_signal(data_with_indicators)
                    if signal:
                        signals.append((indicator_name, signal))
                except Exception as e:
                    logger.warning(f"Failed to generate signal from {indicator_name}: {e}")
                    continue

            # Execute strategy logic based on signals
            self._execute_strategy_signals(
                signals=signals,
                symbol=symbol,
                current_candle=current_candle,
                simulation_engine=simulation_engine,
                leverage=leverage
            )

        return strategy_function

    def _execute_strategy_signals(
        self,
        signals: List[Tuple[str, Signal]],
        symbol: str,
        current_candle: pd.Series,
        simulation_engine: SimulationEngine,
        leverage: float
    ) -> None:
        """Execute trading logic based on strategy signals."""
        if not signals:
            return

        # Simple strategy logic: execute based on consensus
        buy_signals = [s for name, s in signals if s.direction == SignalDirection.BUY]
        sell_signals = [s for name, s in signals if s.direction == SignalDirection.SELL]

        # Check current position
        positions = simulation_engine.get_positions(symbol)
        position = positions[0] if positions else None
        position_side = None if position is None else position.side

        # Execute based on signal consensus
        if len(buy_signals) > len(sell_signals) and position_side != "LONG":
            self._execute_buy_signal(symbol, current_candle, simulation_engine, position)
        elif len(sell_signals) > len(buy_signals) and position_side != "SHORT":
            self._execute_sell_signal(symbol, current_candle, simulation_engine, position)

    def _execute_buy_signal(
        self,
        symbol: str,
        current_candle: pd.Series,
        simulation_engine: SimulationEngine,
        current_position: Any
    ) -> None:
        """Execute a buy signal."""
        try:
            # Close any existing short position
            if current_position and current_position.side == "SHORT":
                simulation_engine.place_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    amount=current_position.amount,
                    timestamp=current_candle["timestamp"],
                    current_candle=current_candle
                )

            # Calculate position size using position sizer
            current_equity = simulation_engine.calculate_equity({symbol: current_candle["close"]})
            current_price = current_candle["close"]
            position_size = self.position_sizer.calculate_position_size(
                current_equity=current_equity,
                current_price=current_price,
                signal_strength=1.0  # Could be enhanced with signal strength
            )

            if position_size > 0:
                simulation_engine.place_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    amount=position_size,
                    timestamp=current_candle["timestamp"],
                    current_candle=current_candle
                )
                logger.debug(f"Opened LONG position: {position_size:.6f} units at ${current_price:.2f}")

        except Exception as e:
            logger.warning(f"Failed to execute buy signal: {e}")

    def _execute_sell_signal(
        self,
        symbol: str,
        current_candle: pd.Series,
        simulation_engine: SimulationEngine,
        current_position: Any
    ) -> None:
        """Execute a sell signal."""
        try:
            # Close any existing long position
            if current_position and current_position.side == "LONG":
                simulation_engine.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    amount=current_position.amount,
                    timestamp=current_candle["timestamp"],
                    current_candle=current_candle
                )

            # Calculate position size using position sizer
            current_equity = simulation_engine.calculate_equity({symbol: current_candle["close"]})
            current_price = current_candle["close"]
            position_size = self.position_sizer.calculate_position_size(
                current_equity=current_equity,
                current_price=current_price,
                signal_strength=1.0  # Could be enhanced with signal strength
            )

            if position_size > 0:
                simulation_engine.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    amount=position_size,
                    timestamp=current_candle["timestamp"],
                    current_candle=current_candle
                )
                logger.debug(f"Opened SHORT position: {position_size:.6f} units at ${current_price:.2f}")

        except Exception as e:
            logger.warning(f"Failed to execute sell signal: {e}")

    def get_strategy_performance_summary(self, result_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance summary for a strategy backtest result.

        Args:
            result_key: Optional specific result key, uses most recent if not provided

        Returns:
            Dictionary with performance summary
        """
        if result_key:
            result = self.results.get(result_key)
        else:
            # Get most recent result
            result = list(self.results.values())[-1] if self.results else None

        if not result:
            return {}

        return {
            'strategy_name': result.metadata.get('strategy_name', 'Unknown'),
            'total_return_pct': result.metrics.get('total_return_pct', 0),
            'total_trades': result.metrics.get('total_trades', 0),
            'win_rate': result.metrics.get('win_rate', 0),
            'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
            'max_drawdown': result.metrics.get('max_drawdown', 0),
            'execution_time': result.metadata.get('execution_time', 0),
            'data_source': result.metadata.get('data_source', 'unknown')
        }

    def clear_cache(self) -> None:
        """Clear performance cache and results."""
        self.performance_cache.clear()
        self.results.clear()
        logger.info("Cleared strategy backtest cache")

    def get_available_results(self) -> List[str]:
        """Get list of available result keys."""
        return list(self.results.keys())

    def export_result(self, result_key: str, output_path: Optional[Path] = None) -> Path:
        """
        Export a backtest result to file.

        Args:
            result_key: Key of the result to export
            output_path: Optional path for export, defaults to output_dir

        Returns:
            Path to exported file
        """
        result = self.results.get(result_key)
        if not result:
            raise ValueError(f"Result not found: {result_key}")

        if not output_path:
            output_path = self.output_dir / f"{result_key}_result.json"

        result.save_to_file(str(output_path))
        logger.info(f"Exported result to: {output_path}")
        return output_path
