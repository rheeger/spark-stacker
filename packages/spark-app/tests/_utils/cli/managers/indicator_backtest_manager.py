"""
Indicator Backtest Manager (CLI Version)

This module provides enhanced indicator backtesting capabilities for the CLI:
- Move existing IndicatorBacktestManager logic
- Add integration with new architecture
- Maintain compatibility with existing functionality
- Add enhanced reporting features
- Add indicator performance caching
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from app.backtesting.backtest_engine import BacktestEngine, BacktestResult
from app.backtesting.indicator_backtest_manager import \
    IndicatorBacktestManager as BaseIndicatorBacktestManager
from app.indicators.base_indicator import (BaseIndicator, Signal,
                                           SignalDirection)
from app.indicators.indicator_factory import IndicatorFactory
from app.risk_management.position_sizing import (PositionSizer,
                                                 PositionSizingConfig)

from ..core.config_manager import ConfigManager
from ..core.data_manager import DataManager as CLIDataManager
from ..core.data_manager import DataRequest, DataSourceType

logger = logging.getLogger(__name__)


class IndicatorBacktestManager:
    """
    Enhanced indicator backtest manager for CLI usage.

    This class extends the functionality of the base IndicatorBacktestManager
    with CLI-specific features like enhanced caching, reporting, and integration
    with the modular CLI architecture.
    """

    def __init__(
        self,
        backtest_engine: BacktestEngine,
        config_manager: ConfigManager,
        data_manager: CLIDataManager,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the enhanced indicator backtest manager.

        Args:
            backtest_engine: Instance of the BacktestEngine
            config_manager: ConfigManager for configuration access
            data_manager: CLI DataManager for data operations
            output_dir: Optional output directory for results
        """
        self.backtest_engine = backtest_engine
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.output_dir = output_dir or Path("./backtest_results")

        # Create base indicator backtest manager for compatibility
        self._base_manager = BaseIndicatorBacktestManager(backtest_engine)

        # Enhanced features
        self.indicators: Dict[str, BaseIndicator] = {}
        self.results: Dict[str, BacktestResult] = {}
        self.performance_cache: Dict[str, Dict[str, Any]] = {}
        self.indicator_factory = IndicatorFactory()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized enhanced IndicatorBacktestManager with output dir: {self.output_dir}")

    def load_indicators_from_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load indicators from configuration.

        Args:
            config_path: Optional path to config file, uses ConfigManager if not provided
        """
        if config_path:
            # Use legacy config loading for backward compatibility
            self._base_manager.load_indicators_from_config(config_path)
            self.indicators = self._base_manager.indicators
        else:
            # Use new ConfigManager-based loading
            self._load_indicators_from_config_manager()

        logger.info(f"Loaded {len(self.indicators)} indicators from configuration")

    def _load_indicators_from_config_manager(self) -> None:
        """Load indicators using the ConfigManager."""
        try:
            # Get global indicators configuration
            global_config = self.config_manager.get_global_config()
            indicators_config = global_config.get('indicators', {})

            # Also get indicators from all strategies
            all_strategies = self.config_manager.list_strategies()

            for strategy_name in all_strategies:
                strategy_config = self.config_manager.get_strategy_config(strategy_name)
                if strategy_config and 'indicators' in strategy_config:
                    indicators_config.update(strategy_config['indicators'])

            # Create indicator instances
            for indicator_name, indicator_config in indicators_config.items():
                try:
                    indicator = self.indicator_factory.create_indicator(
                        indicator_type=indicator_config['type'],
                        name=indicator_name,
                        config=indicator_config
                    )
                    self.indicators[indicator_name] = indicator
                    logger.debug(f"Loaded indicator: {indicator_name} ({indicator_config['type']})")

                except Exception as e:
                    logger.warning(f"Failed to load indicator '{indicator_name}': {e}")

        except Exception as e:
            logger.error(f"Failed to load indicators from config manager: {e}")
            raise

    def add_indicator(self, indicator: BaseIndicator) -> None:
        """
        Add an indicator to the manager.

        Args:
            indicator: Indicator instance to add
        """
        self.indicators[indicator.name] = indicator
        self._base_manager.add_indicator(indicator)
        logger.info(f"Added indicator: {indicator.name}")

    def get_indicator(self, name: str) -> Optional[BaseIndicator]:
        """
        Get an indicator by name.

        Args:
            name: Name of the indicator

        Returns:
            Indicator instance or None if not found
        """
        return self.indicators.get(name)

    def list_indicators(self) -> List[str]:
        """
        Get list of available indicator names.

        Returns:
            List of indicator names
        """
        return list(self.indicators.keys())

    def backtest_indicator(
        self,
        indicator_name: str,
        symbol: str,
        timeframe: str = "1h",
        days: int = 30,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        use_real_data: bool = True,
        leverage: float = 1.0,
        strategy_params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Optional[BacktestResult]:
        """
        Run a backtest for a specific indicator with enhanced features.

        Args:
            indicator_name: Name of the indicator to backtest
            symbol: Market symbol to backtest
            timeframe: Timeframe interval (e.g., '1m', '1h', '1d')
            days: Number of days to test (used if start/end dates not provided)
            start_date: Optional start date for backtest
            end_date: Optional end date for backtest
            use_real_data: Whether to use real market data
            leverage: Default leverage to use
            strategy_params: Additional parameters for the strategy
            use_cache: Whether to use cached results

        Returns:
            BacktestResult with performance metrics or None if indicator not found
        """
        indicator = self.get_indicator(indicator_name)
        if not indicator:
            logger.error(f"Indicator not found: {indicator_name}")
            return None

        # Generate cache key
        cache_key = self._generate_cache_key(
            indicator_name, symbol, timeframe, days, start_date, end_date, use_real_data
        )

        # Check cache first
        if use_cache and cache_key in self.performance_cache:
            logger.info(f"Using cached result for {indicator_name}")
            cached_data = self.performance_cache[cache_key]
            # Reconstruct BacktestResult from cached data
            return self._reconstruct_result_from_cache(cached_data)

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

            # Get market data using CLI data manager
            market_data = self._get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                use_real_data=use_real_data
            )

            # Use base manager for actual backtesting
            logger.info(f"Running indicator backtest: {indicator_name}")
            logger.info(f"  Symbol: {symbol}")
            logger.info(f"  Timeframe: {timeframe}")
            logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")
            logger.info(f"  Data points: {len(market_data)}")

            start_time = time.time()

            result = self._base_manager.backtest_indicator(
                indicator_name=indicator_name,
                symbol=symbol,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date,
                data_source_name="cli_data_manager",
                leverage=leverage,
                strategy_params=strategy_params
            )

            execution_time = time.time() - start_time

            if result:
                # Enhance result with CLI-specific metadata
                result.metadata = getattr(result, 'metadata', {})
                result.metadata.update({
                    'indicator_name': indicator_name,
                    'execution_time': execution_time,
                    'data_source': 'real' if use_real_data else 'synthetic',
                    'leverage': leverage,
                    'cache_key': cache_key
                })

                # Cache result for future use
                if use_cache:
                    self._cache_result(cache_key, result)

                # Store result
                self.results[cache_key] = result

                logger.info(f"Indicator backtest completed in {execution_time:.2f}s")
                logger.info(f"  Total trades: {result.metrics.get('total_trades', 0)}")
                logger.info(f"  Total return: {result.metrics.get('total_return_pct', 0):.2f}%")

            return result

        except Exception as e:
            logger.error(f"Indicator backtest failed: {e}")
            raise

    def backtest_all_indicators(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 30,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        use_real_data: bool = True,
        leverage: float = 1.0,
        strategy_params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        parallel: bool = False
    ) -> Dict[str, BacktestResult]:
        """
        Run backtests for all loaded indicators with enhanced features.

        Args:
            symbol: Market symbol to backtest
            timeframe: Timeframe interval
            days: Number of days to test
            start_date: Optional start date
            end_date: Optional end date
            use_real_data: Whether to use real market data
            leverage: Default leverage to use
            strategy_params: Additional parameters
            use_cache: Whether to use cached results
            parallel: Whether to run indicators in parallel (future enhancement)

        Returns:
            Dictionary of backtest results keyed by indicator name
        """
        results = {}

        logger.info(f"Running backtests for {len(self.indicators)} indicators")

        for indicator_name in self.indicators:
            try:
                result = self.backtest_indicator(
                    indicator_name=indicator_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    days=days,
                    start_date=start_date,
                    end_date=end_date,
                    use_real_data=use_real_data,
                    leverage=leverage,
                    strategy_params=strategy_params,
                    use_cache=use_cache
                )

                if result:
                    results[indicator_name] = result

            except Exception as e:
                logger.error(f"Failed to backtest indicator {indicator_name}: {e}")
                continue

        logger.info(f"Completed backtests for {len(results)} indicators")
        return results

    def compare_indicators(
        self,
        indicator_names: List[str],
        symbol: str,
        timeframe: str = "1h",
        days: int = 30,
        metric: str = "total_return_pct"
    ) -> Dict[str, Any]:
        """
        Compare performance of multiple indicators.

        Args:
            indicator_names: List of indicator names to compare
            symbol: Market symbol
            timeframe: Timeframe interval
            days: Number of days to test
            metric: Metric to compare by

        Returns:
            Dictionary with comparison results
        """
        results = {}

        for indicator_name in indicator_names:
            if indicator_name not in self.indicators:
                logger.warning(f"Indicator not found: {indicator_name}")
                continue

            result = self.backtest_indicator(
                indicator_name=indicator_name,
                symbol=symbol,
                timeframe=timeframe,
                days=days
            )

            if result:
                results[indicator_name] = result

        # Generate comparison analysis
        comparison = {
            'indicators': indicator_names,
            'results': results,
            'ranking': self._rank_indicators_by_metric(results, metric),
            'summary': self._generate_comparison_summary(results)
        }

        logger.info(f"Compared {len(results)} indicators by {metric}")
        return comparison

    def _get_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_real_data: bool
    ) -> pd.DataFrame:
        """Get market data using CLI data manager."""
        try:
            if use_real_data:
                request = DataRequest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    source_preference=[DataSourceType.REAL_EXCHANGE, DataSourceType.CACHED]
                )
                data = self.data_manager.get_data(request)
            else:
                data = self.data_manager.generate_synthetic_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    scenario_type='default'
                )

            if data.empty:
                raise ValueError(f"No market data available for {symbol} {timeframe}")

            return data

        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            raise

    def _generate_cache_key(
        self,
        indicator_name: str,
        symbol: str,
        timeframe: str,
        days: int,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        use_real_data: bool
    ) -> str:
        """Generate a unique cache key for the backtest parameters."""
        date_str = f"{start_date.isoformat() if start_date else 'None'}_{end_date.isoformat() if end_date else 'None'}"
        data_type = "real" if use_real_data else "synthetic"
        return f"{indicator_name}_{symbol}_{timeframe}_{days}_{date_str}_{data_type}"

    def _cache_result(self, cache_key: str, result: BacktestResult) -> None:
        """Cache a backtest result."""
        try:
            # Store essential result data for caching
            cached_data = {
                'metrics': result.metrics,
                'symbol': result.symbol,
                'start_date': result.start_date.isoformat(),
                'end_date': result.end_date.isoformat(),
                'initial_balance': result.initial_balance,
                'final_balance': result.final_balance,
                'total_trades': len(result.trades),
                'metadata': getattr(result, 'metadata', {})
            }

            self.performance_cache[cache_key] = cached_data
            logger.debug(f"Cached result for key: {cache_key}")

        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

    def _reconstruct_result_from_cache(self, cached_data: Dict[str, Any]) -> BacktestResult:
        """Reconstruct a BacktestResult from cached data."""
        # This is a simplified reconstruction - in practice you might want to
        # cache more complete data or regenerate some components

        # Create a minimal BacktestResult with cached metrics
        result = BacktestResult(
            symbol=cached_data['symbol'],
            start_date=datetime.fromisoformat(cached_data['start_date']),
            end_date=datetime.fromisoformat(cached_data['end_date']),
            initial_balance=cached_data['initial_balance'],
            final_balance=cached_data['final_balance'],
            trades=[],  # Trades not cached for simplicity
            equity_curve=pd.DataFrame(),  # Equity curve not cached
            metrics=cached_data['metrics']
        )

        result.metadata = cached_data.get('metadata', {})
        return result

    def _rank_indicators_by_metric(
        self,
        results: Dict[str, BacktestResult],
        metric: str
    ) -> List[Tuple[str, float]]:
        """Rank indicators by a specific metric."""
        rankings = []

        for indicator_name, result in results.items():
            metric_value = result.metrics.get(metric, 0)
            rankings.append((indicator_name, metric_value))

        # Sort by metric value (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def _generate_comparison_summary(self, results: Dict[str, BacktestResult]) -> Dict[str, Any]:
        """Generate a summary of comparison results."""
        if not results:
            return {}

        metrics = ['total_return_pct', 'win_rate', 'sharpe_ratio', 'max_drawdown', 'total_trades']
        summary = {}

        for metric in metrics:
            values = [result.metrics.get(metric, 0) for result in results.values()]
            summary[metric] = {
                'best': max(values) if values else 0,
                'worst': min(values) if values else 0,
                'average': sum(values) / len(values) if values else 0
            }

        return summary

    def get_performance_summary(self, indicator_name: str) -> Dict[str, Any]:
        """
        Get performance summary for an indicator.

        Args:
            indicator_name: Name of the indicator

        Returns:
            Dictionary with performance summary
        """
        # Find the most recent result for this indicator
        indicator_results = {
            k: v for k, v in self.results.items()
            if k.startswith(indicator_name)
        }

        if not indicator_results:
            return {}

        # Get most recent result
        latest_key = max(indicator_results.keys())
        result = indicator_results[latest_key]

        return {
            'indicator_name': indicator_name,
            'total_return_pct': result.metrics.get('total_return_pct', 0),
            'total_trades': result.metrics.get('total_trades', 0),
            'win_rate': result.metrics.get('win_rate', 0),
            'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
            'max_drawdown': result.metrics.get('max_drawdown', 0),
            'execution_time': result.metadata.get('execution_time', 0),
            'data_source': result.metadata.get('data_source', 'unknown')
        }

    def clear_cache(self) -> None:
        """Clear performance cache."""
        self.performance_cache.clear()
        self.results.clear()
        logger.info("Cleared indicator backtest cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_results': len(self.performance_cache),
            'stored_results': len(self.results),
            'indicators_loaded': len(self.indicators)
        }

    def export_result(self, cache_key: str, output_path: Optional[Path] = None) -> Path:
        """
        Export a backtest result to file.

        Args:
            cache_key: Key of the result to export
            output_path: Optional path for export

        Returns:
            Path to exported file
        """
        result = self.results.get(cache_key)
        if not result:
            raise ValueError(f"Result not found: {cache_key}")

        if not output_path:
            output_path = self.output_dir / f"{cache_key}_result.json"

        result.save_to_file(str(output_path))
        logger.info(f"Exported result to: {output_path}")
        return output_path

    def get_available_results(self) -> List[str]:
        """Get list of available result keys."""
        return list(self.results.keys())
