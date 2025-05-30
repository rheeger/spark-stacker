"""
Data Manager

This module provides centralized data management for the CLI with:
- Centralized data fetching logic (real and synthetic)
- Intelligent caching across multiple runs with TTL and invalidation
- Multi-timeframe data requirements handling
- Data quality validation and cleanup
- Data source failover and retry logic with exponential backoff
- Data export and import functionality for analysis
- Strategy-specific data management for enhanced backtesting
- Enhanced performance optimizations for strategy caching and parallel execution
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from app.backtesting.data_manager import DataManager as AppDataManager
from app.connectors.hyperliquid_connector import HyperliquidConnector
from app.core.strategy_config import StrategyConfig
from tests._helpers.data_factory import make_price_dataframe
# Import performance tracking
from utils.progress_trackers import (PerformanceMetricType,
                                     get_performance_tracker)

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of data sources available."""
    REAL_EXCHANGE = "real_exchange"
    SYNTHETIC = "synthetic"
    CACHED = "cached"
    IMPORTED = "imported"


class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


class MarketScenarioType(Enum):
    """Types of market scenarios for strategy testing."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_RANGING = "sideways_ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CHOPPY_WHIPSAW = "choppy_whipsaw"
    GAP_HEAVY = "gap_heavy"
    REAL_DATA = "real_data"


@dataclass
class IndicatorCacheEntry:
    """Cache entry for indicator calculations."""
    indicator_name: str
    data_hash: str
    parameters: Dict[str, Any]
    result: pd.DataFrame
    timestamp: float
    computation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyDataRequest:
    """Represents a strategy-specific data request with all parameters."""
    strategy_config: StrategyConfig
    scenario_type: Optional[MarketScenarioType] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    days_back: Optional[int] = None
    include_volume: bool = True
    source_preference: List[DataSourceType] = field(default_factory=lambda: [
        DataSourceType.CACHED, DataSourceType.REAL_EXCHANGE, DataSourceType.SYNTHETIC
    ])


@dataclass
class MultiTimeframeDataSet:
    """Container for multi-timeframe data required by strategies."""
    primary_timeframe: str
    data: Dict[str, pd.DataFrame]  # timeframe -> DataFrame
    metadata: Dict[str, Any]
    quality_reports: Dict[str, 'DataQualityReport']

    def get_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data for a specific timeframe."""
        return self.data.get(timeframe)

    def get_primary_data(self) -> pd.DataFrame:
        """Get data for the primary timeframe."""
        return self.data[self.primary_timeframe]


@dataclass
class ScenarioDataSet:
    """Container for scenario-specific data for strategy testing."""
    scenario_type: MarketScenarioType
    symbol: str
    timeframe: str
    data: pd.DataFrame
    characteristics: Dict[str, Any]
    generated_at: datetime


@dataclass
class DataRequest:
    """Represents a data request with all parameters."""
    symbol: str
    timeframe: str
    exchange: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    days_back: Optional[int] = None
    source_preference: List[DataSourceType] = field(default_factory=lambda: [
        DataSourceType.CACHED, DataSourceType.REAL_EXCHANGE, DataSourceType.SYNTHETIC
    ])


@dataclass
class DataQualityReport:
    """Report on data quality assessment."""
    quality_level: DataQuality
    total_records: int
    missing_records: int
    duplicate_records: int
    data_gaps: List[Tuple[datetime, datetime]]
    timestamp_issues: List[str]
    price_anomalies: List[str]
    recommendations: List[str]


@dataclass
class CacheEntry:
    """Represents a cached data entry."""
    data: pd.DataFrame
    timestamp: float
    source: DataSourceType
    quality_report: DataQualityReport
    metadata: Dict[str, Any]


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_entries: int = 0
    memory_entries: int = 0
    disk_entries: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_mb: float = 0.0
    hit_rate: float = 0.0
    memory_usage_mb: float = 0.0


class DataFetchError(Exception):
    """Raised when data fetching fails."""
    pass


class DataQualityError(Exception):
    """Raised when data quality is insufficient."""
    pass


class DataManager:
    """
    Centralized data manager for CLI operations.

    Features:
    - Unified data fetching from multiple sources with intelligent fallback
    - Multi-level caching (memory + disk) with TTL and invalidation
    - Multi-timeframe data handling and aggregation
    - Comprehensive data quality validation and cleanup
    - Retry logic with exponential backoff for reliability
    - Data export/import for external analysis
    - Enhanced performance optimizations for strategy caching and parallel execution
    - Indicator calculation caching for reuse across strategies
    - Coordinated cache management across all manager modules
    """

    # Default cache settings
    DEFAULT_CACHE_TTL_SECONDS = 3600  # 1 hour
    DEFAULT_RETRY_ATTEMPTS = 3
    DEFAULT_RETRY_DELAY = 1.0  # seconds

    # Performance optimization settings
    MAX_PARALLEL_FETCHES = 4
    INDICATOR_CACHE_SIZE = 200
    STRATEGY_CACHE_SIZE = 50

    def __init__(self,
                 cache_dir: Optional[str] = None,
                 memory_cache_size: int = 100,
                 disk_cache_ttl_hours: int = 24,
                 enable_real_data: bool = True,
                 retry_attempts: int = 3,
                 enable_parallel_fetching: bool = True,
                 enable_indicator_caching: bool = True,
                 enable_performance_tracking: bool = True):
        """
        Initialize the data manager.

        Args:
            cache_dir: Directory for disk cache. If None, uses default location.
            memory_cache_size: Maximum number of entries in memory cache
            disk_cache_ttl_hours: Time-to-live for disk cache entries
            enable_real_data: Whether to enable real exchange data fetching
            retry_attempts: Number of retry attempts for failed requests
            enable_parallel_fetching: Enable parallel data fetching for performance
            enable_indicator_caching: Enable caching of indicator calculations
            enable_performance_tracking: Enable performance monitoring
        """
        self.memory_cache_size = memory_cache_size
        self.disk_cache_ttl_hours = disk_cache_ttl_hours
        self.enable_real_data = enable_real_data
        self.retry_attempts = retry_attempts
        self.enable_parallel_fetching = enable_parallel_fetching
        self.enable_indicator_caching = enable_indicator_caching
        self.enable_performance_tracking = enable_performance_tracking

        # Set up cache directory
        if cache_dir is None:
            base_dir = Path(__file__).parent.parent.parent.parent
            self.cache_dir = base_dir / "__test_data__" / "data_manager_cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.Lock()
        self._connectors: Dict[str, Any] = {}

        # Enhanced caching for performance
        self._indicator_cache: Dict[str, IndicatorCacheEntry] = {}
        self._strategy_data_cache: Dict[str, MultiTimeframeDataSet] = {}
        self._cross_strategy_cache: Dict[str, pd.DataFrame] = {}  # Shared data across strategies
        self._cache_stats = CacheStats()

        # Performance tracking
        self._performance_tracker = get_performance_tracker() if enable_performance_tracking else None

        # Parallel execution
        self._fetch_executor: Optional[ThreadPoolExecutor] = None
        if enable_parallel_fetching:
            self._fetch_executor = ThreadPoolExecutor(max_workers=self.MAX_PARALLEL_FETCHES)

        # Initialize data sources
        self._initialize_data_sources()

        logger.info(f"DataManager initialized with cache at {self.cache_dir}")
        logger.info(f"Real data enabled: {enable_real_data}")
        logger.info(f"Parallel fetching enabled: {enable_parallel_fetching}")
        logger.info(f"Indicator caching enabled: {enable_indicator_caching}")

    def get_data(self, request: DataRequest) -> pd.DataFrame:
        """
        Get data according to the request specifications.

        Args:
            request: DataRequest object specifying what data to fetch

        Returns:
            DataFrame containing the requested data

        Raises:
            DataFetchError: If data cannot be fetched from any source
            DataQualityError: If data quality is insufficient
        """
        operation_name = f"fetch_data_{request.symbol}_{request.timeframe}"

        if self._performance_tracker:
            with self._performance_tracker.track_operation(operation_name):
                return self._get_data_with_tracking(request)
        else:
            return self._get_data_internal(request)

    def _get_data_with_tracking(self, request: DataRequest) -> pd.DataFrame:
        """Internal method with performance tracking."""
        start_time = time.time()

        try:
            result = self._get_data_internal(request)

            if self._performance_tracker:
                fetch_time = time.time() - start_time
                self._performance_tracker.add_metric(
                    PerformanceMetricType.DATA_FETCH_TIME,
                    fetch_time,
                    "seconds",
                    context={"symbol": request.symbol, "timeframe": request.timeframe}
                )

            return result

        except Exception as e:
            if self._performance_tracker:
                self._performance_tracker.record_cache_miss("data")
            raise

    def _get_data_internal(self, request: DataRequest) -> pd.DataFrame:
        """Internal data fetching logic."""
        cache_key = self._generate_cache_key(request)

        # Try to get from cache first
        if DataSourceType.CACHED in request.source_preference:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Using cached data for {cache_key}")
                if self._performance_tracker:
                    self._performance_tracker.record_cache_hit("data")
                return cached_data

        # Try each data source in preference order
        last_error = None

        for source_type in request.source_preference:
            if source_type == DataSourceType.CACHED:
                continue  # Already tried above

            try:
                data = self._fetch_from_source(request, source_type)

                # Validate data quality
                quality_report = self._assess_data_quality(data, request)
                if quality_report.quality_level == DataQuality.INVALID:
                    logger.warning(f"Invalid data from {source_type}, trying next source")
                    continue

                # Cache the successful result
                self._cache_data(cache_key, data, source_type, quality_report, request)

                logger.info(f"Successfully fetched data from {source_type} for {request.symbol}")
                return data

            except Exception as e:
                logger.warning(f"Failed to fetch from {source_type}: {e}")
                last_error = e
                continue

        # If we get here, all sources failed
        if self._performance_tracker:
            self._performance_tracker.record_cache_miss("data")
        raise DataFetchError(f"Unable to fetch data for {request.symbol} from any source. Last error: {last_error}")

    def get_multi_timeframe_data_parallel(self,
                                         symbol: str,
                                         timeframes: List[str],
                                         exchange: Optional[str] = None,
                                         days_back: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple timeframes efficiently using parallel fetching.

        Args:
            symbol: Market symbol (e.g., "ETH-USD")
            timeframes: List of timeframes (e.g., ["1h", "4h", "1d"])
            exchange: Optional exchange name
            days_back: Number of days of historical data

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        if not self.enable_parallel_fetching or not self._fetch_executor:
            # Fall back to sequential fetching
            return self.get_multi_timeframe_data(symbol, timeframes, exchange, days_back)

        operation_name = f"parallel_fetch_{symbol}_{len(timeframes)}_timeframes"

        if self._performance_tracker:
            with self._performance_tracker.track_operation(operation_name):
                return self._get_multi_timeframe_parallel_internal(symbol, timeframes, exchange, days_back)
        else:
            return self._get_multi_timeframe_parallel_internal(symbol, timeframes, exchange, days_back)

    def _get_multi_timeframe_parallel_internal(self,
                                              symbol: str,
                                              timeframes: List[str],
                                              exchange: Optional[str],
                                              days_back: int) -> Dict[str, pd.DataFrame]:
        """Internal parallel multi-timeframe fetching."""
        results = {}

        # Create requests for all timeframes
        fetch_jobs = []
        for timeframe in timeframes:
            request = DataRequest(
                symbol=symbol,
                timeframe=timeframe,
                exchange=exchange,
                days_back=days_back
            )
            fetch_jobs.append((timeframe, request))

        # Submit parallel fetch jobs
        future_to_timeframe = {}
        for timeframe, request in fetch_jobs:
            future = self._fetch_executor.submit(self._get_data_internal, request)
            future_to_timeframe[future] = timeframe

        # Collect results as they complete
        for future in as_completed(future_to_timeframe):
            timeframe = future_to_timeframe[future]
            try:
                data = future.result()
                results[timeframe] = data
                logger.debug(f"Completed parallel fetch for {symbol} {timeframe}")
            except Exception as e:
                logger.error(f"Failed to fetch {symbol} {timeframe}: {e}")
                # Continue with other timeframes

        return results

    def get_multi_timeframe_data(self,
                                symbol: str,
                                timeframes: List[str],
                                exchange: Optional[str] = None,
                                days_back: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple timeframes efficiently.

        Args:
            symbol: Market symbol (e.g., "ETH-USD")
            timeframes: List of timeframes (e.g., ["1h", "4h", "1d"])
            exchange: Optional exchange name
            days_back: Number of days of historical data

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        results = {}

        # Create requests for all timeframes
        requests = []
        for timeframe in timeframes:
            request = DataRequest(
                symbol=symbol,
                timeframe=timeframe,
                exchange=exchange,
                days_back=days_back
            )
            requests.append((timeframe, request))

        # Fetch data for each timeframe
        for timeframe, request in requests:
            try:
                data = self.get_data(request)
                results[timeframe] = data
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol} {timeframe}: {e}")
                # Continue with other timeframes

        return results

    def cache_indicator_calculation(self,
                                   indicator_name: str,
                                   data: pd.DataFrame,
                                   parameters: Dict[str, Any],
                                   result: pd.DataFrame,
                                   computation_time: float) -> None:
        """
        Cache indicator calculation results for reuse across strategies.

        Args:
            indicator_name: Name of the indicator
            data: Input data used for calculation
            parameters: Indicator parameters
            result: Calculated indicator result
            computation_time: Time taken for calculation
        """
        if not self.enable_indicator_caching:
            return

        # Generate cache key based on data hash and parameters
        data_hash = self._generate_data_hash(data)
        param_hash = self._generate_parameter_hash(parameters)
        cache_key = f"indicator_{indicator_name}_{data_hash}_{param_hash}"

        # Create cache entry
        entry = IndicatorCacheEntry(
            indicator_name=indicator_name,
            data_hash=data_hash,
            parameters=parameters,
            result=result.copy(),
            timestamp=time.time(),
            computation_time=computation_time
        )

        with self._cache_lock:
            self._indicator_cache[cache_key] = entry

            # Evict oldest entries if cache is full
            if len(self._indicator_cache) > self.INDICATOR_CACHE_SIZE:
                oldest_key = min(self._indicator_cache.keys(),
                               key=lambda k: self._indicator_cache[k].timestamp)
                del self._indicator_cache[oldest_key]
                self._cache_stats.evictions += 1

        logger.debug(f"Cached indicator calculation: {indicator_name}")

    def get_cached_indicator_calculation(self,
                                       indicator_name: str,
                                       data: pd.DataFrame,
                                       parameters: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Retrieve cached indicator calculation if available.

        Args:
            indicator_name: Name of the indicator
            data: Input data for calculation
            parameters: Indicator parameters

        Returns:
            Cached result DataFrame or None if not found
        """
        if not self.enable_indicator_caching:
            return None

        # Generate cache key
        data_hash = self._generate_data_hash(data)
        param_hash = self._generate_parameter_hash(parameters)
        cache_key = f"indicator_{indicator_name}_{data_hash}_{param_hash}"

        with self._cache_lock:
            entry = self._indicator_cache.get(cache_key)
            if entry is not None:
                # Check if entry is still fresh (1 hour TTL for indicator calculations)
                age_hours = (time.time() - entry.timestamp) / 3600
                if age_hours < 1.0:
                    if self._performance_tracker:
                        self._performance_tracker.record_cache_hit("indicator")
                    logger.debug(f"Using cached indicator calculation: {indicator_name}")
                    return entry.result.copy()
                else:
                    # Remove stale entry
                    del self._indicator_cache[cache_key]

        if self._performance_tracker:
            self._performance_tracker.record_cache_miss("indicator")
        return None

    def cache_strategy_data(self,
                           strategy_data: MultiTimeframeDataSet,
                           strategy_name: str) -> None:
        """
        Cache strategy data efficiently for reuse across multiple strategies.

        Args:
            strategy_data: MultiTimeframeDataSet to cache
            strategy_name: Name of the strategy for cache key generation
        """
        operation_name = f"cache_strategy_data_{strategy_name}"

        if self._performance_tracker:
            with self._performance_tracker.track_operation(operation_name):
                self._cache_strategy_data_internal(strategy_data, strategy_name)
        else:
            self._cache_strategy_data_internal(strategy_data, strategy_name)

    def _cache_strategy_data_internal(self,
                                     strategy_data: MultiTimeframeDataSet,
                                     strategy_name: str) -> None:
        """Internal strategy data caching logic."""
        # Cache individual timeframe data in cross-strategy cache
        for timeframe, data in strategy_data.data.items():
            symbol = strategy_data.metadata.get('market', 'unknown')
            cross_cache_key = f"cross_strategy_{symbol}_{timeframe}"

            with self._cache_lock:
                self._cross_strategy_cache[cross_cache_key] = data.copy()

            # Also cache in regular data cache
            cache_key = f"strategy_{strategy_name}_{timeframe}_{symbol}"

            # Create a dummy request for caching
            dummy_request = DataRequest(
                symbol=symbol,
                timeframe=timeframe,
                exchange=strategy_data.metadata.get('exchange')
            )

            quality_report = strategy_data.quality_reports.get(timeframe)
            if quality_report is None:
                quality_report = self._assess_data_quality(data, dummy_request)

            self._cache_data(
                cache_key=cache_key,
                data=data,
                source=DataSourceType.CACHED,
                quality_report=quality_report,
                request=dummy_request
            )

        # Cache the full strategy dataset
        with self._cache_lock:
            self._strategy_data_cache[strategy_name] = strategy_data

            # Evict oldest strategy data if cache is full
            if len(self._strategy_data_cache) > self.STRATEGY_CACHE_SIZE:
                oldest_strategy = min(self._strategy_data_cache.keys(),
                                    key=lambda k: getattr(self._strategy_data_cache[k], 'cached_at', 0))
                del self._strategy_data_cache[oldest_strategy]
                self._cache_stats.evictions += 1

        logger.info(f"Cached strategy data for '{strategy_name}' with {len(strategy_data.data)} timeframes")

    def get_cross_strategy_data(self,
                               symbol: str,
                               timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get data that can be shared across multiple strategies for the same symbol/timeframe.

        Args:
            symbol: Market symbol
            timeframe: Data timeframe

        Returns:
            Cached DataFrame or None if not available
        """
        cross_cache_key = f"cross_strategy_{symbol}_{timeframe}"

        with self._cache_lock:
            data = self._cross_strategy_cache.get(cross_cache_key)
            if data is not None:
                if self._performance_tracker:
                    self._performance_tracker.record_cache_hit("cross_strategy")
                logger.debug(f"Using cross-strategy cached data for {symbol} {timeframe}")
                return data.copy()

        if self._performance_tracker:
            self._performance_tracker.record_cache_miss("cross_strategy")
        return None

    def optimize_cache_for_strategies(self, strategy_configs: List[StrategyConfig]) -> None:
        """
        Pre-warm caches based on strategy requirements for optimal performance.

        Args:
            strategy_configs: List of strategy configurations to optimize for
        """
        operation_name = f"optimize_cache_{len(strategy_configs)}_strategies"

        if self._performance_tracker:
            with self._performance_tracker.track_operation(operation_name):
                self._optimize_cache_internal(strategy_configs)
        else:
            self._optimize_cache_internal(strategy_configs)

    def _optimize_cache_internal(self, strategy_configs: List[StrategyConfig]) -> None:
        """Internal cache optimization logic."""
        # Identify common data requirements across strategies
        data_requirements = defaultdict(set)

        for config in strategy_configs:
            symbol = config.market
            timeframe = config.timeframe
            data_requirements[(symbol, timeframe)].add(config.name)

            # Also consider indicator timeframes
            for indicator_config in config.indicators:
                if hasattr(indicator_config, 'timeframe') and indicator_config.timeframe:
                    indicator_timeframe = indicator_config.timeframe
                    data_requirements[(symbol, indicator_timeframe)].add(f"{config.name}_indicator")

        # Pre-fetch data for requirements used by multiple strategies
        high_priority_data = [
            (symbol, timeframe) for (symbol, timeframe), strategies in data_requirements.items()
            if len(strategies) > 1
        ]

        logger.info(f"Pre-warming cache for {len(high_priority_data)} high-priority data sets")

        # Use parallel fetching for pre-warming if available
        if self.enable_parallel_fetching and self._fetch_executor:
            futures = []
            for symbol, timeframe in high_priority_data:
                request = DataRequest(symbol=symbol, timeframe=timeframe, days_back=30)
                future = self._fetch_executor.submit(self._get_data_internal, request)
                futures.append(future)

            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Failed to pre-warm cache: {e}")
        else:
            # Sequential pre-warming
            for symbol, timeframe in high_priority_data:
                try:
                    request = DataRequest(symbol=symbol, timeframe=timeframe, days_back=30)
                    self._get_data_internal(request)
                except Exception as e:
                    logger.warning(f"Failed to pre-warm cache for {symbol} {timeframe}: {e}")

    def get_enhanced_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics including performance metrics.

        Returns:
            Dictionary with detailed cache statistics
        """
        with self._cache_lock:
            # Base cache stats
            memory_entries = len(self._memory_cache)
            indicator_entries = len(self._indicator_cache)
            strategy_entries = len(self._strategy_data_cache)
            cross_strategy_entries = len(self._cross_strategy_cache)

            # Calculate memory usage estimates
            memory_usage_mb = 0.0
            for entry in self._memory_cache.values():
                # Rough estimate: DataFrame size in MB
                memory_usage_mb += entry.data.memory_usage(deep=True).sum() / 1024 / 1024

            # Disk cache stats
            disk_entries = 0
            disk_size_mb = 0.0
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.cache"):
                    disk_entries += 1
                    disk_size_mb += cache_file.stat().st_size / 1024 / 1024

            # Update cache stats
            self._cache_stats.total_entries = memory_entries + disk_entries
            self._cache_stats.memory_entries = memory_entries
            self._cache_stats.disk_entries = disk_entries
            self._cache_stats.total_size_mb = memory_usage_mb + disk_size_mb
            self._cache_stats.memory_usage_mb = memory_usage_mb

            # Calculate hit rate
            total_requests = self._cache_stats.hits + self._cache_stats.misses
            self._cache_stats.hit_rate = (self._cache_stats.hits / total_requests
                                        if total_requests > 0 else 0.0)

        return {
            'total_entries': self._cache_stats.total_entries,
            'memory_entries': memory_entries,
            'disk_entries': disk_entries,
            'indicator_cache_entries': indicator_entries,
            'strategy_cache_entries': strategy_entries,
            'cross_strategy_cache_entries': cross_strategy_entries,
            'cache_hits': self._cache_stats.hits,
            'cache_misses': self._cache_stats.misses,
            'hit_rate': self._cache_stats.hit_rate,
            'evictions': self._cache_stats.evictions,
            'total_size_mb': self._cache_stats.total_size_mb,
            'memory_usage_mb': memory_usage_mb,
            'disk_size_mb': disk_size_mb,
            'cache_efficiency': {
                'memory_hit_rate': memory_entries / (memory_entries + disk_entries) if (memory_entries + disk_entries) > 0 else 0,
                'indicator_reuse_rate': indicator_entries / max(1, indicator_entries + strategy_entries),
                'cross_strategy_sharing': cross_strategy_entries / max(1, strategy_entries)
            }
        }

    def _generate_data_hash(self, data: pd.DataFrame) -> str:
        """Generate hash for DataFrame to detect identical data."""
        # Use a subset of data for hash to avoid performance issues with large datasets
        if len(data) > 100:
            sample_data = data.iloc[::len(data)//100]  # Sample every nth row
        else:
            sample_data = data

        # Create hash from values and index
        hash_input = f"{sample_data.values.tobytes()}{str(sample_data.index.tolist())}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _generate_parameter_hash(self, parameters: Dict[str, Any]) -> str:
        """Generate hash for parameters dictionary."""
        # Sort parameters for consistent hashing
        sorted_params = json.dumps(parameters, sort_keys=True, default=str)
        return hashlib.md5(sorted_params.encode()).hexdigest()[:8]

    # Private methods for internal functionality

    def _initialize_data_sources(self) -> None:
        """Initialize available data sources."""
        if self.enable_real_data:
            try:
                # Initialize Hyperliquid connector
                self._connectors['hyperliquid'] = HyperliquidConnector()
                logger.debug("Initialized Hyperliquid connector")
            except Exception as e:
                logger.warning(f"Failed to initialize Hyperliquid connector: {e}")

    def _generate_cache_key(self, request: DataRequest) -> str:
        """Generate a cache key for the request."""
        key_parts = [
            request.symbol.replace('/', '_').replace('-', '_'),
            request.timeframe,
            request.exchange or 'default'
        ]

        if request.days_back:
            key_parts.append(f"days_{request.days_back}")
        elif request.start_date and request.end_date:
            key_parts.append(f"range_{request.start_date.date()}_{request.end_date.date()}")

        return "_".join(key_parts)

    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get data from cache if available and fresh."""
        with self._cache_lock:
            # Check memory cache first
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                if self._is_cache_entry_fresh(entry):
                    return entry.data.copy()
                else:
                    # Remove stale entry
                    del self._memory_cache[cache_key]

            # Check disk cache
            cache_file = self.cache_dir / f"{cache_key}.cache"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        entry = pickle.load(f)

                    if self._is_cache_entry_fresh(entry):
                        # Move to memory cache
                        self._memory_cache[cache_key] = entry
                        self._evict_memory_cache_if_needed()
                        return entry.data.copy()
                    else:
                        # Remove stale file
                        cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {e}")
                    cache_file.unlink(missing_ok=True)

        return None

    def _is_cache_entry_fresh(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still fresh."""
        age_hours = (time.time() - entry.timestamp) / 3600
        return age_hours < self.disk_cache_ttl_hours

    def _fetch_from_source(self, request: DataRequest, source_type: DataSourceType) -> pd.DataFrame:
        """Fetch data from the specified source type."""
        if source_type == DataSourceType.REAL_EXCHANGE:
            return self._fetch_real_data(request)
        elif source_type == DataSourceType.SYNTHETIC:
            return self._fetch_synthetic_data(request)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def _fetch_real_data(self, request: DataRequest) -> pd.DataFrame:
        """Fetch real data from exchange."""
        if not self.enable_real_data:
            raise DataFetchError("Real data fetching is disabled")

        exchange = request.exchange or 'hyperliquid'
        connector = self._connectors.get(exchange)

        if not connector:
            raise DataFetchError(f"No connector available for exchange: {exchange}")

        # Implement retry logic
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                if request.days_back:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=request.days_back)
                else:
                    start_date = request.start_date
                    end_date = request.end_date

                # Use the app's data manager for actual fetching
                app_data_manager = AppDataManager()
                data = app_data_manager.get_historical_data(
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    start_date=start_date,
                    end_date=end_date
                )

                return data

            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    delay = self.DEFAULT_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.retry_attempts} attempts failed")

        raise DataFetchError(f"Failed to fetch real data after {self.retry_attempts} attempts: {last_error}")

    def _fetch_synthetic_data(self, request: DataRequest) -> pd.DataFrame:
        """Fetch synthetic data."""
        days = request.days_back or 30
        return self.generate_synthetic_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            days=days,
            pattern="trend",  # Default pattern
            seed=42  # Deterministic for testing
        )

    def _assess_data_quality(self, data: pd.DataFrame, request: Optional[DataRequest] = None, symbol: str = "unknown") -> DataQualityReport:
        """Assess the quality of market data."""
        total_records = len(data)

        if total_records == 0:
            return DataQualityReport(
                quality_level=DataQuality.INVALID,
                total_records=0,
                missing_records=0,
                duplicate_records=0,
                data_gaps=[],
                timestamp_issues=["No data available"],
                price_anomalies=[],
                recommendations=["Fetch data from alternative source"]
            )

        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            return DataQualityReport(
                quality_level=DataQuality.INVALID,
                total_records=total_records,
                missing_records=0,
                duplicate_records=0,
                data_gaps=[],
                timestamp_issues=[f"Missing columns: {missing_columns}"],
                price_anomalies=[],
                recommendations=[f"Ensure data includes columns: {required_columns}"]
            )

        # Check for missing values
        missing_records = data[required_columns].isnull().sum().sum()
        missing_percentage = (missing_records / (total_records * len(required_columns))) * 100

        # Check for duplicates
        duplicate_records = data.index.duplicated().sum()

        # Check for data gaps
        data_gaps = self._find_data_gaps(data)

        # Check timestamp issues
        timestamp_issues = []
        if not isinstance(data.index, pd.DatetimeIndex):
            timestamp_issues.append("Index is not datetime-based")
        elif data.index.duplicated().any():
            timestamp_issues.append("Duplicate timestamps found")
        elif not data.index.is_monotonic_increasing:
            timestamp_issues.append("Timestamps are not in chronological order")

        # Check price anomalies
        price_anomalies = self._detect_price_anomalies(data)

        # Determine quality level
        if missing_percentage > 50 or len(timestamp_issues) > 0:
            quality_level = DataQuality.INVALID
        elif missing_percentage > 20 or duplicate_records > total_records * 0.1:
            quality_level = DataQuality.POOR
        elif missing_percentage > 5 or len(data_gaps) > 5:
            quality_level = DataQuality.ACCEPTABLE
        elif missing_percentage > 1 or len(price_anomalies) > 0:
            quality_level = DataQuality.GOOD
        else:
            quality_level = DataQuality.EXCELLENT

        # Generate recommendations
        recommendations = []
        if missing_percentage > 0:
            recommendations.append(f"Fill {missing_records} missing values")
        if duplicate_records > 0:
            recommendations.append(f"Remove {duplicate_records} duplicate records")
        if len(data_gaps) > 0:
            recommendations.append(f"Address {len(data_gaps)} data gaps")
        if len(price_anomalies) > 0:
            recommendations.append(f"Review {len(price_anomalies)} price anomalies")

        return DataQualityReport(
            quality_level=quality_level,
            total_records=total_records,
            missing_records=missing_records,
            duplicate_records=duplicate_records,
            data_gaps=data_gaps,
            timestamp_issues=timestamp_issues,
            price_anomalies=price_anomalies,
            recommendations=recommendations
        )

    def _find_data_gaps(self, data: pd.DataFrame) -> List[Tuple[datetime, datetime]]:
        """Find gaps in the data timeline."""
        if len(data) < 2:
            return []

        gaps = []
        time_diffs = data.index.to_series().diff()
        median_diff = time_diffs.median()

        # Consider a gap if the time difference is more than 2x the median
        gap_threshold = median_diff * 2

        for i, diff in enumerate(time_diffs):
            if pd.notna(diff) and diff > gap_threshold:
                gap_start = data.index[i-1]
                gap_end = data.index[i]
                gaps.append((gap_start, gap_end))

        return gaps

    def _detect_price_anomalies(self, data: pd.DataFrame) -> List[str]:
        """Detect price anomalies in the data."""
        anomalies = []

        if len(data) < 10:  # Need minimum data for analysis
            return anomalies

        # Check for impossible OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )

        if invalid_ohlc.any():
            anomalies.append(f"Invalid OHLC relationships in {invalid_ohlc.sum()} records")

        # Check for extreme price movements (>50% in single candle)
        if 'close' in data.columns:
            returns = data['close'].pct_change()
            extreme_moves = abs(returns) > 0.5
            if extreme_moves.any():
                anomalies.append(f"Extreme price movements (>50%) in {extreme_moves.sum()} records")

        # Check for zero or negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                invalid_prices = data[col] <= 0
                if invalid_prices.any():
                    anomalies.append(f"Zero or negative {col} prices in {invalid_prices.sum()} records")

        return anomalies

    def _cache_data(self,
                   cache_key: str,
                   data: pd.DataFrame,
                   source: DataSourceType,
                   quality_report: DataQualityReport,
                   request: DataRequest) -> None:
        """Cache data for future use."""
        entry = CacheEntry(
            data=data.copy(),
            timestamp=time.time(),
            source=source,
            quality_report=quality_report,
            metadata={
                'symbol': request.symbol,
                'timeframe': request.timeframe,
                'exchange': request.exchange,
                'cache_key': cache_key
            }
        )

        with self._cache_lock:
            # Add to memory cache
            self._memory_cache[cache_key] = entry
            self._evict_memory_cache_if_needed()

            # Save to disk cache
            try:
                cache_file = self.cache_dir / f"{cache_key}.cache"
                with open(cache_file, 'wb') as f:
                    pickle.dump(entry, f)
            except Exception as e:
                logger.warning(f"Failed to save cache file: {e}")

    def _evict_memory_cache_if_needed(self) -> None:
        """Evict oldest entries if memory cache is full."""
        while len(self._memory_cache) > self.memory_cache_size:
            # Remove oldest entry
            oldest_key = min(self._memory_cache.keys(),
                           key=lambda k: self._memory_cache[k].timestamp)
            del self._memory_cache[oldest_key]

    def _get_candles_per_day(self, timeframe: str) -> float:
        """Get number of candles per day for a timeframe."""
        timeframe_minutes = self._timeframe_to_minutes(timeframe)
        return 1440 / timeframe_minutes  # 1440 minutes in a day

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        timeframe = timeframe.lower()

        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        elif timeframe == '1min':
            return 1
        elif timeframe == '1hour':
            return 60
        elif timeframe == '1day':
            return 1440
        else:
            # Default to 1 hour if can't parse
            logger.warning(f"Unknown timeframe {timeframe}, defaulting to 60 minutes")
            return 60

    # Strategy-specific data management methods

    def get_strategy_data(self, request: StrategyDataRequest) -> MultiTimeframeDataSet:
        """
        Get data for a strategy with all required timeframes.

        Args:
            request: StrategyDataRequest containing strategy config and options

        Returns:
            MultiTimeframeDataSet with data for all required timeframes

        Raises:
            DataFetchError: If any required data cannot be fetched
        """
        strategy_config = request.strategy_config

        # Determine all required timeframes
        required_timeframes = self._get_strategy_timeframes(strategy_config)
        primary_timeframe = strategy_config.timeframe

        logger.info(f"Fetching data for strategy '{strategy_config.name}' - timeframes: {required_timeframes}")

        # Prepare data collection
        data = {}
        quality_reports = {}

        # Fetch data for each timeframe
        for timeframe in required_timeframes:
            # Create data request for this timeframe
            data_request = DataRequest(
                symbol=strategy_config.market,
                timeframe=timeframe,
                exchange=strategy_config.exchange,
                start_date=request.start_date,
                end_date=request.end_date,
                days_back=request.days_back,
                source_preference=request.source_preference
            )

            # Handle scenario-specific data if specified
            if request.scenario_type and request.scenario_type != MarketScenarioType.REAL_DATA:
                scenario_data = self.generate_scenario_data(
                    scenario_type=request.scenario_type,
                    symbol=strategy_config.market,
                    timeframe=timeframe,
                    days_back=request.days_back or 30,
                    include_volume=request.include_volume
                )
                data[timeframe] = scenario_data.data
                quality_reports[timeframe] = self._assess_data_quality(scenario_data.data, data_request)
            else:
                # Fetch real or cached data
                timeframe_data = self.get_data(data_request)
                data[timeframe] = timeframe_data
                quality_reports[timeframe] = self._assess_data_quality(timeframe_data, data_request)

        # Validate data completeness
        self._validate_strategy_data_completeness(data, strategy_config)

        return MultiTimeframeDataSet(
            primary_timeframe=primary_timeframe,
            data=data,
            metadata={
                'strategy_name': strategy_config.name,
                'market': strategy_config.market,
                'exchange': strategy_config.exchange,
                'scenario_type': request.scenario_type.value if request.scenario_type else None,
                'data_source': request.source_preference[0].value
            },
            quality_reports=quality_reports
        )

    def generate_scenario_data(self,
                              scenario_type: MarketScenarioType,
                              symbol: str,
                              timeframe: str,
                              days_back: int = 30,
                              include_volume: bool = True) -> ScenarioDataSet:
        """
        Generate synthetic data for specific market scenarios.

        Args:
            scenario_type: Type of market scenario to generate
            symbol: Trading symbol (e.g., 'ETH-USD')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            days_back: Number of days of data to generate
            include_volume: Whether to include volume data

        Returns:
            ScenarioDataSet containing the generated scenario data
        """
        logger.info(f"Generating {scenario_type.value} scenario data for {symbol} {timeframe}")

        # Calculate number of candles needed
        candles_per_day = self._get_candles_per_day(timeframe)
        num_candles = int(days_back * candles_per_day)

        # Generate base price action
        base_price = 2000.0  # Starting price

        if scenario_type == MarketScenarioType.BULL_MARKET:
            data = self._generate_bull_market_data(base_price, num_candles, timeframe, include_volume)
            characteristics = {
                'trend': 'bullish',
                'volatility': 'moderate',
                'up_days_percentage': 0.70,
                'expected_return': 0.25
            }
        elif scenario_type == MarketScenarioType.BEAR_MARKET:
            data = self._generate_bear_market_data(base_price, num_candles, timeframe, include_volume)
            characteristics = {
                'trend': 'bearish',
                'volatility': 'moderate',
                'down_days_percentage': 0.70,
                'expected_return': -0.20
            }
        elif scenario_type == MarketScenarioType.SIDEWAYS_RANGING:
            data = self._generate_sideways_market_data(base_price, num_candles, timeframe, include_volume)
            characteristics = {
                'trend': 'sideways',
                'volatility': 'low',
                'range_percentage': 0.10,
                'expected_return': 0.02
            }
        elif scenario_type == MarketScenarioType.HIGH_VOLATILITY:
            data = self._generate_high_volatility_data(base_price, num_candles, timeframe, include_volume)
            characteristics = {
                'trend': 'random',
                'volatility': 'high',
                'daily_range_percentage': 0.20,
                'expected_return': 0.05
            }
        elif scenario_type == MarketScenarioType.LOW_VOLATILITY:
            data = self._generate_low_volatility_data(base_price, num_candles, timeframe, include_volume)
            characteristics = {
                'trend': 'stable',
                'volatility': 'low',
                'daily_range_percentage': 0.015,
                'expected_return': 0.01
            }
        elif scenario_type == MarketScenarioType.CHOPPY_WHIPSAW:
            data = self._generate_choppy_market_data(base_price, num_candles, timeframe, include_volume)
            characteristics = {
                'trend': 'choppy',
                'volatility': 'moderate',
                'direction_changes': 'frequent',
                'expected_return': -0.05
            }
        elif scenario_type == MarketScenarioType.GAP_HEAVY:
            data = self._generate_gap_heavy_data(base_price, num_candles, timeframe, include_volume)
            characteristics = {
                'trend': 'gapping',
                'volatility': 'high',
                'gap_frequency': 'high',
                'expected_return': 0.08
            }
        else:
            raise ValueError(f"Unsupported scenario type: {scenario_type}")

        return ScenarioDataSet(
            scenario_type=scenario_type,
            symbol=symbol,
            timeframe=timeframe,
            data=data,
            characteristics=characteristics,
            generated_at=datetime.now()
        )

    def _get_strategy_timeframes(self, strategy_config: StrategyConfig) -> Set[str]:
        """
        Get all timeframes required by a strategy and its indicators.

        Args:
            strategy_config: Strategy configuration

        Returns:
            Set of required timeframes
        """
        timeframes = {strategy_config.timeframe}

        # Since indicators is List[str] in the actual StrategyConfig,
        # we can only return the strategy's primary timeframe for now.
        # In a full implementation, we would need to look up indicator
        # configurations from the config manager to get their timeframes.

        # For now, just use the strategy's primary timeframe
        return timeframes

    def _validate_strategy_data_completeness(self,
                                           data: Dict[str, pd.DataFrame],
                                           strategy_config: StrategyConfig) -> None:
        """
        Validate that all required data for the strategy is available.

        Args:
            data: Dictionary of timeframe -> DataFrame
            strategy_config: Strategy configuration

        Raises:
            DataFetchError: If required data is missing or insufficient
        """
        required_timeframes = self._get_strategy_timeframes(strategy_config)

        for timeframe in required_timeframes:
            if timeframe not in data:
                raise DataFetchError(f"Missing data for required timeframe {timeframe}")

            df = data[timeframe]
            if df.empty:
                raise DataFetchError(f"Empty data for timeframe {timeframe}")

            # Check minimum data requirements (at least 100 candles for reliable indicators)
            min_candles = 100
            if len(df) < min_candles:
                logger.warning(f"Limited data for {timeframe}: {len(df)} candles (recommended: {min_candles}+)")

    # Market scenario data generation methods

    def _generate_bull_market_data(self, start_price: float, num_candles: int,
                                  timeframe: str, include_volume: bool) -> pd.DataFrame:
        """Generate bullish trending market data."""
        np.random.seed(42)  # For reproducible results

        # Parameters for bull market
        trend_strength = 0.0003  # 0.03% per candle average drift upward
        volatility = 0.02  # 2% volatility
        up_bias = 0.6  # 60% chance of up moves

        prices = [start_price]
        volumes = [1000000] if include_volume else []

        for i in range(num_candles - 1):
            # Add trend component and random walk
            trend_component = trend_strength
            random_component = np.random.normal(0, volatility)

            # Apply upward bias
            if np.random.random() < up_bias:
                random_component = abs(random_component)
            else:
                random_component = -abs(random_component) * 0.7  # Smaller down moves

            price_change = trend_component + random_component
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 1.0))  # Prevent negative prices

            if include_volume:
                # Volume tends to increase on up moves in bull markets
                base_volume = 1000000
                volume_multiplier = 1.0 + random_component * 2
                volumes.append(max(int(base_volume * volume_multiplier), 100000))

        return self._create_ohlcv_dataframe(prices, timeframe, include_volume, volumes)

    def _generate_bear_market_data(self, start_price: float, num_candles: int,
                                  timeframe: str, include_volume: bool) -> pd.DataFrame:
        """Generate bearish trending market data."""
        np.random.seed(43)

        trend_strength = -0.0002  # -0.02% per candle average drift downward
        volatility = 0.025  # 2.5% volatility (slightly higher in bear markets)
        down_bias = 0.65  # 65% chance of down moves

        prices = [start_price]
        volumes = [1000000] if include_volume else []

        for i in range(num_candles - 1):
            trend_component = trend_strength
            random_component = np.random.normal(0, volatility)

            if np.random.random() < down_bias:
                random_component = -abs(random_component)
            else:
                random_component = abs(random_component) * 0.6  # Smaller up moves

            price_change = trend_component + random_component
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 1.0))

            if include_volume:
                base_volume = 1200000  # Higher volume in bear markets
                volume_multiplier = 1.0 + abs(random_component) * 1.5
                volumes.append(max(int(base_volume * volume_multiplier), 100000))

        return self._create_ohlcv_dataframe(prices, timeframe, include_volume, volumes)

    def _generate_sideways_market_data(self, start_price: float, num_candles: int,
                                      timeframe: str, include_volume: bool) -> pd.DataFrame:
        """Generate sideways/ranging market data."""
        np.random.seed(44)

        range_center = start_price
        range_width = start_price * 0.05  # 5% range
        volatility = 0.01  # 1% volatility

        prices = [start_price]
        volumes = [800000] if include_volume else []

        for i in range(num_candles - 1):
            current_price = prices[-1]

            # Mean reversion toward range center
            distance_from_center = (current_price - range_center) / range_width
            mean_reversion = -distance_from_center * 0.0001

            random_component = np.random.normal(0, volatility)
            price_change = mean_reversion + random_component

            new_price = current_price * (1 + price_change)
            new_price = max(new_price, range_center - range_width)
            new_price = min(new_price, range_center + range_width)
            prices.append(new_price)

            if include_volume:
                base_volume = 800000  # Lower volume in sideways markets
                volume_multiplier = 1.0 + abs(random_component) * 0.8
                volumes.append(max(int(base_volume * volume_multiplier), 100000))

        return self._create_ohlcv_dataframe(prices, timeframe, include_volume, volumes)

    def _generate_high_volatility_data(self, start_price: float, num_candles: int,
                                      timeframe: str, include_volume: bool) -> pd.DataFrame:
        """Generate high volatility market data."""
        np.random.seed(45)

        volatility = 0.05  # 5% volatility

        prices = [start_price]
        volumes = [1500000] if include_volume else []

        for i in range(num_candles - 1):
            random_component = np.random.normal(0, volatility)
            price_change = random_component

            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 1.0))

            if include_volume:
                base_volume = 1500000  # High volume with high volatility
                volume_multiplier = 1.0 + abs(random_component) * 3
                volumes.append(max(int(base_volume * volume_multiplier), 100000))

        return self._create_ohlcv_dataframe(prices, timeframe, include_volume, volumes)

    def _generate_low_volatility_data(self, start_price: float, num_candles: int,
                                     timeframe: str, include_volume: bool) -> pd.DataFrame:
        """Generate low volatility market data."""
        np.random.seed(46)

        volatility = 0.005  # 0.5% volatility
        drift = 0.00005  # Very small upward drift

        prices = [start_price]
        volumes = [600000] if include_volume else []

        for i in range(num_candles - 1):
            random_component = np.random.normal(0, volatility)
            price_change = drift + random_component

            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 1.0))

            if include_volume:
                base_volume = 600000  # Low volume with low volatility
                volume_multiplier = 1.0 + abs(random_component) * 0.5
                volumes.append(max(int(base_volume * volume_multiplier), 100000))

        return self._create_ohlcv_dataframe(prices, timeframe, include_volume, volumes)

    def _generate_choppy_market_data(self, start_price: float, num_candles: int,
                                    timeframe: str, include_volume: bool) -> pd.DataFrame:
        """Generate choppy/whipsaw market data with frequent direction changes."""
        np.random.seed(47)

        volatility = 0.03  # 3% volatility
        direction_change_freq = 0.3  # 30% chance of direction change each candle

        prices = [start_price]
        volumes = [900000] if include_volume else []
        current_direction = 1  # 1 for up, -1 for down

        for i in range(num_candles - 1):
            # Potentially change direction
            if np.random.random() < direction_change_freq:
                current_direction *= -1

            # Generate move in current direction with some noise
            base_move = current_direction * np.random.uniform(0.005, 0.015)
            noise = np.random.normal(0, volatility * 0.5)
            price_change = base_move + noise

            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 1.0))

            if include_volume:
                base_volume = 900000
                volume_multiplier = 1.0 + abs(price_change) * 2
                volumes.append(max(int(base_volume * volume_multiplier), 100000))

        return self._create_ohlcv_dataframe(prices, timeframe, include_volume, volumes)

    def _generate_gap_heavy_data(self, start_price: float, num_candles: int,
                                timeframe: str, include_volume: bool) -> pd.DataFrame:
        """Generate market data with frequent price gaps."""
        np.random.seed(48)

        volatility = 0.02  # 2% base volatility
        gap_frequency = 0.05  # 5% chance of gap each candle
        gap_size_range = (0.02, 0.08)  # 2-8% gaps

        prices = [start_price]
        volumes = [1100000] if include_volume else []

        for i in range(num_candles - 1):
            # Check for gap
            if np.random.random() < gap_frequency:
                # Generate gap
                gap_size = np.random.uniform(*gap_size_range)
                gap_direction = 1 if np.random.random() < 0.5 else -1
                gap_change = gap_direction * gap_size

                # Normal volatility on top of gap
                normal_change = np.random.normal(0, volatility * 0.5)
                total_change = gap_change + normal_change
            else:
                # Normal price movement
                total_change = np.random.normal(0, volatility)

            new_price = prices[-1] * (1 + total_change)
            prices.append(max(new_price, 1.0))

            if include_volume:
                base_volume = 1100000
                # Higher volume on gaps
                volume_multiplier = 1.0 + abs(total_change) * 4
                volumes.append(max(int(base_volume * volume_multiplier), 100000))

        return self._create_ohlcv_dataframe(prices, timeframe, include_volume, volumes)

    def _create_ohlcv_dataframe(self, close_prices: List[float], timeframe: str,
                               include_volume: bool, volumes: Optional[List[int]] = None) -> pd.DataFrame:
        """Create OHLCV DataFrame from close prices."""
        timeframe_minutes = self._timeframe_to_minutes(timeframe)

        # Create timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=timeframe_minutes * len(close_prices))
        timestamps = pd.date_range(start=start_time, end=end_time, periods=len(close_prices))

        # Generate OHLC from close prices
        data = []
        for i, close in enumerate(close_prices):
            if i == 0:
                open_price = close
                high = close
                low = close
            else:
                # Generate realistic OHLC based on previous close and current close
                prev_close = close_prices[i-1]
                open_price = prev_close

                # High and low around the range between open and close
                if close > open_price:
                    # Up candle
                    high = close * (1 + np.random.uniform(0, 0.005))
                    low = min(open_price, close) * (1 - np.random.uniform(0, 0.003))
                else:
                    # Down candle
                    high = max(open_price, close) * (1 + np.random.uniform(0, 0.003))
                    low = close * (1 - np.random.uniform(0, 0.005))

            candle_data = {
                'open': open_price,
                'high': high,
                'low': low,
                'close': close
            }

            if include_volume and volumes:
                candle_data['volume'] = volumes[i]

            data.append(candle_data)

        df = pd.DataFrame(data, index=timestamps)
        return df
