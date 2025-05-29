import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from app.connectors.base_connector import BaseConnector
from app.core.symbol_converter import convert_symbol_for_exchange
from app.core.trading_engine import TradingEngine
from app.indicators.base_indicator import (BaseIndicator, Signal,
                                           SignalDirection)
from app.indicators.macd_indicator import MACDIndicator
from app.metrics import (INDICATOR_MACD, MARKET_ETH_USD, TIMEFRAME_1M,
                         record_mvp_signal_latency, update_candle_data,
                         update_macd_indicator, update_mvp_signal_state)
from app.metrics.registry import (publish_historical_time_series,
                                  set_historical_data_mode,
                                  verify_historical_data)

logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Strategy manager that runs indicators and forwards signals to the trading engine.

    This class is responsible for:
    1. Managing indicator instances
    2. Running indicators at regular intervals on market data
    3. Processing signals and forwarding them to the trading engine
    """

    def __init__(
        self,
        trading_engine: TradingEngine,
        indicators: Dict[str, BaseIndicator] = None,
        strategies: List[Dict[str, Any]] = None,
        data_window_size: int = 100,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize the strategy manager.

        Args:
            trading_engine: Trading engine instance to forward signals to
            indicators: Dictionary of indicator instances
            strategies: List of strategy configurations
            data_window_size: Size of the price data window to maintain
            config: Application configuration dictionary
        """
        self.trading_engine = trading_engine
        self.indicators = indicators or {}
        self.strategies = strategies or []
        self.data_window_size = data_window_size
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.historical_data_fetched: Dict[Tuple[str, str], bool] = {}
        self.config = config or {}

        # Build strategy mappings
        self.strategy_indicators: Dict[str, List[str]] = {}
        self._build_strategy_mappings()

        # Check if publishing historical data to metrics is enabled
        self.publish_historical = self.config.get("metrics_publish_historical", False)
        if self.publish_historical:
            logger.info("Historical metrics publishing is enabled - will publish all candles to metrics")

        logger.info(
            f"Strategy manager initialized with {len(self.indicators)} indicators and {len(self.strategies)} strategies"
        )

    def _build_strategy_mappings(self) -> None:
        """
        Build strategy-indicator mappings from strategy configurations.

        This method processes the strategies list and creates a mapping dictionary
        that links each strategy name to its list of indicators. This mapping is used
        during strategy-driven execution to determine which indicators to run for each strategy.

        The method validates that each strategy has a name and indicators defined,
        logging warnings for any invalid configurations and skipping them.

        Updates:
            self.strategy_indicators: Dictionary mapping strategy names to indicator lists

        Side Effects:
            - Clears existing strategy_indicators mapping
            - Logs debug information for each strategy mapping created
            - Logs warnings for strategies with missing names or indicators
        """
        self.strategy_indicators.clear()

        for strategy in self.strategies:
            # Handle both StrategyConfig objects and dictionary configurations
            if hasattr(strategy, 'name'):
                # StrategyConfig object
                strategy_name = strategy.name
                indicators = strategy.indicators
            else:
                # Dictionary configuration (legacy support)
                strategy_name = strategy.get("name")
                indicators = strategy.get("indicators", [])

            if not strategy_name:
                logger.warning("Strategy missing name field, skipping")
                continue

            if not indicators:
                logger.warning(f"Strategy '{strategy_name}' has no indicators defined")
                continue

            self.strategy_indicators[strategy_name] = indicators
            logger.debug(f"Mapped strategy '{strategy_name}' to indicators: {indicators}")

        logger.info(f"Built strategy mappings for {len(self.strategy_indicators)} strategies")

    def add_indicator(self, name: str, indicator: BaseIndicator) -> None:
        """
        Add an indicator to the strategy.

        Args:
            name: Name to identify the indicator
            indicator: Indicator instance
        """
        self.indicators[name] = indicator
        logger.info(f"Added indicator: {indicator}")

    def remove_indicator(self, name: str) -> bool:
        """
        Remove an indicator from the strategy.

        Args:
            name: Name of the indicator to remove

        Returns:
            bool: True if removed, False if not found
        """
        if name in self.indicators:
            indicator = self.indicators.pop(name)
            logger.info(f"Removed indicator: {indicator}")
            return True
        return False

    def update_price_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Update price data for a symbol and potentially update candle metrics.

        Args:
            symbol: Market symbol
            data: New price data (expected to contain OHLCV)
        """
        # --- Candle Metric Update ---
        # Check if the update is for our target market/timeframe
        # Assuming the 'data' DataFrame represents 1m candles for ETH-USD
        # This check needs refinement based on how data is actually passed.
        logger.info(f"Checking candle update for: {symbol}, matches ETH-USD? {symbol == MARKET_ETH_USD}")

        if symbol == MARKET_ETH_USD: # This needs to be more robust
             # Check if the DataFrame has the required columns
            logger.info(f"Data is empty? {data.empty}, columns: {data.columns.tolist()}")

            if not data.empty and all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                latest_candle = data.iloc[-1]
                logger.info(f"Latest candle for {symbol}: {latest_candle.to_dict()}")
                try:
                    # Assuming timeframe is implicitly 1m here
                    update_candle_data(market=symbol, timeframe="1m", field="open", value=float(latest_candle['open']))
                    update_candle_data(market=symbol, timeframe="1m", field="high", value=float(latest_candle['high']))
                    update_candle_data(market=symbol, timeframe="1m", field="low", value=float(latest_candle['low']))
                    update_candle_data(market=symbol, timeframe="1m", field="close", value=float(latest_candle['close']))
                    logger.info(f"Updated candle metrics for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to update candle metrics for {symbol}: {e}", exc_info=True)
            else:
                logger.warning(f"Skipping candle metric update for {symbol}: Data empty or missing OHLC columns.")

        # --- End Candle Metric Update ---

        if symbol not in self.price_data:
            self.price_data[symbol] = data.tail(self.data_window_size)
        else:
            # Append new data and trim to window size
            # Ensure timestamps are unique if concatenating
            self.price_data[symbol] = pd.concat([self.price_data[symbol], data]).drop_duplicates(subset=['timestamp'], keep='last')
            self.price_data[symbol] = self.price_data[symbol].tail(
                self.data_window_size
            )

    def _fetch_current_market_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch current market data from the exchange connector.

        Args:
            symbol: Market symbol to fetch data for

        Returns:
            DataFrame with current market data or a dummy DataFrame if fetching fails
        """
        try:
            # Get reference to the main connector from trading engine
            connector = self.trading_engine.main_connector

            if not connector:
                logger.error("No main connector available to fetch market data")
                return self._create_dummy_data(symbol)

            # Try to get current ticker data
            ticker = connector.get_ticker(symbol)

            if not ticker or "last_price" not in ticker or not ticker["last_price"]:
                logger.warning(
                    f"Could not get valid ticker data for {symbol}, using default values"
                )
                return self._create_dummy_data(symbol)

            # Create a DataFrame with the ticker data
            last_price = float(ticker["last_price"])
            timestamp = int(time.time() * 1000)

            logger.info(f"Fetched current price for {symbol}: {last_price}")

            # Use a simulated price if we can't get real data
            if last_price <= 0.0:
                logger.warning(
                    f"Invalid price {last_price} for {symbol}, using simulated price of 1.0"
                )
                last_price = 1.0

            df = pd.DataFrame(
                {
                    "timestamp": [timestamp],
                    "open": [last_price],
                    "high": [last_price],
                    "low": [last_price],
                    "close": [last_price],
                    "volume": [0.0],
                    "symbol": [symbol],
                }
            )

            return df

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}", exc_info=True)
            return self._create_dummy_data(symbol)

    def _create_dummy_data(self, symbol: str) -> pd.DataFrame:
        """
        Create dummy data for testing when real market data is unavailable.

        Args:
            symbol: Market symbol

        Returns:
            DataFrame with dummy data
        """
        # For testing purposes, use a simulated price of 1.0
        simulated_price = 1.0
        timestamp = int(time.time() * 1000)

        logger.warning(f"Using simulated price data for {symbol}: {simulated_price}")

        return pd.DataFrame(
            {
                "timestamp": [timestamp],
                "open": [simulated_price],
                "high": [simulated_price],
                "low": [simulated_price],
                "close": [simulated_price],
                "volume": [1000.0],
                "symbol": [symbol],
            }
        )

    def _fetch_historical_data(self, symbol: str, interval: str, limit: int, periods: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical candle data for a symbol.

        Args:
            symbol: Symbol to fetch data for
            interval: Time interval (e.g., '1m', '5m', '1h')
            limit: Number of candles to fetch
            periods: Minimum number of periods needed for indicators

        Returns:
            DataFrame with historical price data
        """
        try:
            required_candles = max(limit, periods if periods else 50)
            logger.info(f"Fetching {required_candles} historical candles for {symbol} ({interval})...")

            # Get historical data from connector
            connector = self.trading_engine.main_connector
            candles = connector.get_historical_candles(
                symbol=symbol,
                interval=interval,
                limit=required_candles
            )

            if not candles:
                logger.warning(f"No historical data received for {symbol}")
                return pd.DataFrame()

            # Sort candles by timestamp (ascending)
            sorted_candles = sorted(candles, key=lambda x: x['timestamp'])

            # Publish historical data to metrics if enabled
            self._publish_historical_data_to_metrics(symbol, interval, sorted_candles)

            # Convert to DataFrame
            df = pd.DataFrame(candles)

            # Convert all relevant columns to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Ensure required columns exist
            if 'timestamp' not in df.columns:
                if 'time' in df.columns:
                    df['timestamp'] = df['time']
                else:
                    logger.error(f"Missing timestamp column in candle data for {symbol}")
                    return pd.DataFrame()

            if 'symbol' not in df.columns:
                df['symbol'] = symbol

            return df.tail(required_candles)

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def _publish_historical_data_to_metrics(self, symbol: str, interval: str, candles: List[Dict]) -> None:
        """
        Publish historical data to metrics for visualization and persistence.

        This method is essential for data persistence across container restarts and provides
        historical data to monitoring systems like Grafana. It publishes both raw OHLC candle
        data and calculated indicator values (like MACD) to Prometheus metrics.

        The method handles:
        1. Publishing OHLC candle data as time series
        2. Calculating and publishing MACD indicator values if applicable
        3. Updating standard metrics with the most recent values
        4. Verifying that historical data was properly published

        Args:
            symbol: Market symbol (e.g., "ETH-USD")
            interval: Time interval (e.g., "1m", "1h", "4h")
            candles: List of candle dictionaries with timestamp, open, high, low, close, volume

        Side Effects:
            - Publishes historical time series data to Prometheus
            - Updates MACD metrics if MACD indicators are found for the symbol
            - Updates standard candle and indicator metrics
            - Logs detailed information about publishing progress
            - Verifies data availability in Prometheus

        Note:
            This method only runs if self.publish_historical is True (controlled by
            the "metrics_publish_historical" configuration setting).
        """
        if not self.publish_historical:
            logger.info(f"Historical data publishing is disabled. Not publishing {len(candles)} candles for {symbol}")
            return

        try:
            # Sort candles by timestamp (ascending) to ensure correct order
            sorted_candles = sorted(candles, key=lambda x: x['timestamp'])

            logger.info(f"Publishing {len(sorted_candles)} historical candles to metrics for {symbol} ({interval})")

            # NEW APPROACH: Use our time series publishing system that properly handles historical data

            # First handle the candle data (OHLC)
            for field in ['open', 'high', 'low', 'close']:
                # Format data into the required structure for the time series publisher
                data_points = [
                    {"timestamp": int(candle['timestamp']), "value": float(candle[field])}
                    for candle in sorted_candles
                ]

                # Publish the time series data
                success = publish_historical_time_series(
                    market=symbol,
                    timeframe=interval,
                    field=field,
                    data_points=data_points,
                    gauge_name="spark_stacker_historical_candle",
                    gauge_description=f"Historical {field} price data"
                )

                if success:
                    logger.info(f"Successfully published historical {field} data for {symbol}/{interval}")
                else:
                    logger.warning(f"Failed to publish historical {field} data for {symbol}/{interval}")

            # Now handle MACD indicators if available
            macd_indicators = [
                (name, indicator) for name, indicator in self.indicators.items()
                if isinstance(indicator, MACDIndicator) and
                getattr(indicator, 'symbol', None) == symbol
            ]

            if macd_indicators:
                logger.info(f"Found {len(macd_indicators)} MACD indicators for {symbol}, calculating historical values")

                # Convert to DataFrame for MACD calculation
                df = pd.DataFrame(sorted_candles)

                # Convert all relevant columns to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Ensure timestamp column exists
                if 'timestamp' not in df.columns and 'time' in df.columns:
                    df['timestamp'] = df['time']

                try:
                    # Calculate MACD for each indicator and publish
                    for name, indicator in macd_indicators:
                        processed_df = indicator.calculate(df)

                        if 'macd' in processed_df.columns and 'macd_signal' in processed_df.columns:
                            # Publish MACD line
                            macd_points = [
                                {"timestamp": int(row['timestamp']), "value": float(row['macd'])}
                                for _, row in processed_df.iterrows() if not pd.isna(row['macd'])
                            ]
                            publish_historical_time_series(
                                market=symbol,
                                timeframe=interval,
                                field="macd_line",
                                data_points=macd_points,
                                gauge_name="spark_stacker_historical_macd",
                                gauge_description="Historical MACD line values"
                            )

                            # Publish MACD signal line
                            signal_points = [
                                {"timestamp": int(row['timestamp']), "value": float(row['macd_signal'])}
                                for _, row in processed_df.iterrows() if not pd.isna(row['macd_signal'])
                            ]
                            publish_historical_time_series(
                                market=symbol,
                                timeframe=interval,
                                field="signal_line",
                                data_points=signal_points,
                                gauge_name="spark_stacker_historical_macd",
                                gauge_description="Historical MACD signal line values"
                            )

                            # Publish MACD histogram
                            histogram_points = [
                                {"timestamp": int(row['timestamp']), "value": float(row['macd_histogram'])}
                                for _, row in processed_df.iterrows() if not pd.isna(row['macd_histogram'])
                            ]
                            publish_historical_time_series(
                                market=symbol,
                                timeframe=interval,
                                field="histogram",
                                data_points=histogram_points,
                                gauge_name="spark_stacker_historical_macd",
                                gauge_description="Historical MACD histogram values"
                            )

                            logger.info(f"Successfully published historical MACD data for {symbol}/{interval}")
                except Exception as e:
                    logger.error(f"Failed to calculate or publish historical MACD: {e}", exc_info=True)

            # Also publish the most recent values to the standard metrics for real-time display
            try:
                # Get the most recent candle
                latest_candle = sorted_candles[-1]

                # Update standard metrics with the latest candle data
                update_candle_data(market=symbol, timeframe=interval, field="open", value=float(latest_candle['open']))
                update_candle_data(market=symbol, timeframe=interval, field="high", value=float(latest_candle['high']))
                update_candle_data(market=symbol, timeframe=interval, field="low", value=float(latest_candle['low']))
                update_candle_data(market=symbol, timeframe=interval, field="close", value=float(latest_candle['close']))

                # If we have MACD data, update those metrics too
                if macd_indicators and 'macd' in processed_df.columns:
                    latest_macd = processed_df.iloc[-1]
                    update_macd_indicator(market=symbol, timeframe=interval, component="macd_line", value=float(latest_macd['macd']))
                    update_macd_indicator(market=symbol, timeframe=interval, component="signal_line", value=float(latest_macd['macd_signal']))
                    update_macd_indicator(market=symbol, timeframe=interval, component="histogram", value=float(latest_macd['macd_histogram']))

                logger.info(f"Updated standard metrics with latest data for {symbol}/{interval}")
            except Exception as e:
                logger.error(f"Failed to update standard metrics with latest data: {e}", exc_info=True)

            # Verify historical data was properly published
            try:
                if verify_historical_data(market=symbol, timeframe=interval):
                    logger.info(f"✅ VERIFIED: Historical data for {symbol}/{interval} is properly available in Prometheus")
                else:
                    logger.warning(f"❌ WARNING: Could not verify historical data for {symbol}/{interval} in Prometheus")
            except Exception as e:
                logger.error(f"Error verifying historical data: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error while publishing historical data to metrics: {e}", exc_info=True)

    def _prepare_indicator_data(self, market_symbol: str, timeframe: str, exchange: str, required_periods: int = 50) -> pd.DataFrame:
        """
        Prepare indicator data for a specific market symbol and timeframe.

        Args:
            market_symbol: Market symbol in standard format (e.g., "ETH-USD")
            timeframe: Timeframe for the data (e.g., "1h", "4h")
            exchange: Exchange name for symbol conversion
            required_periods: Minimum number of periods needed

        Returns:
            DataFrame with prepared price data
        """
        try:
            # Use market + timeframe for cache keys
            cache_key = f"{market_symbol}_{timeframe}"
            cache_tuple_key = (market_symbol, timeframe)

            # Check if we've already fetched historical data for this market/timeframe
            first_fetch_done = self.historical_data_fetched.get(cache_tuple_key, False)

            # Get existing cached data if available
            current_data = self.price_data.get(cache_key, pd.DataFrame()).copy()

            # Convert market symbol for exchange-specific API calls
            exchange_symbol = convert_symbol_for_exchange(market_symbol, exchange)
            logger.debug(f"Converted market symbol '{market_symbol}' to '{exchange_symbol}' for exchange '{exchange}'")

            # Fetch historical data if this is the first time or we don't have enough data
            if not first_fetch_done or len(current_data) < required_periods:
                logger.info(f"Fetching historical data for {market_symbol} ({timeframe}) - need {required_periods}, have {len(current_data)}")

                historical_data = self._fetch_historical_data(
                    symbol=exchange_symbol,  # Use exchange-specific symbol
                    interval=timeframe,
                    limit=self.data_window_size,
                    periods=required_periods
                )

                if not historical_data.empty:
                    # Update cache with historical data
                    self.price_data[cache_key] = historical_data
                    current_data = historical_data.copy()
                    self.historical_data_fetched[cache_tuple_key] = True
                    logger.info(f"Updated cache for {cache_key} with {len(historical_data)} periods")
                else:
                    logger.warning(f"Failed to fetch historical data for {market_symbol} ({timeframe})")

            # If we still don't have enough data, try to fetch current market data as fallback
            if current_data.empty:
                logger.info(f"No cached data available for {market_symbol} ({timeframe}), fetching current market data")
                current_data = self._fetch_current_market_data(exchange_symbol)  # Use exchange-specific symbol

            # Verify we have sufficient data
            if len(current_data) < required_periods:
                logger.warning(f"Insufficient data for {market_symbol} ({timeframe}): need {required_periods}, have {len(current_data)}")

            return current_data

        except Exception as e:
            logger.error(f"Error preparing indicator data for {market_symbol} ({timeframe}): {e}", exc_info=True)
            return pd.DataFrame()

    def run_strategy_indicators(self, strategy_config: Dict[str, Any], market: str, indicator_names: List[str]) -> List[Signal]:
        """
        Run indicators for a specific strategy on a market.

        Args:
            strategy_config: Strategy configuration dictionary
            market: Market symbol in standard format (e.g., "ETH-USD")
            indicator_names: List of indicator names to run

        Returns:
            List of signals with strategy metadata
        """
        signals = []
        # Extract key strategy parameters with fallbacks
        strategy_name = strategy_config.get("name", "unknown")
        strategy_timeframe = strategy_config.get("timeframe", "1h")  # Strategy dictates timeframe for all indicators
        exchange = strategy_config.get("exchange", "hyperliquid")

        logger.info(f"Running strategy '{strategy_name}' indicators for {market} on {strategy_timeframe} timeframe")

        # Pre-flight check: Validate all indicators exist before processing
        # This prevents partial processing if some indicators are missing
        missing_indicators = []
        for indicator_name in indicator_names:
            if indicator_name not in self.indicators:
                missing_indicators.append(indicator_name)

        if missing_indicators:
            logger.error(f"Strategy '{strategy_name}' references missing indicators: {missing_indicators}")
            return signals  # Return empty signals list - strategy cannot run

        # Calculate minimum data requirements across all indicators
        # Each indicator may need different amounts of historical data
        required_periods = 50  # Conservative default minimum
        for indicator_name in indicator_names:
            indicator = self.indicators[indicator_name]
            indicator_periods = getattr(indicator, 'required_periods', 50)
            # Take the maximum requirement to ensure all indicators have sufficient data
            required_periods = max(required_periods, indicator_periods)

        # Prepare market data using strategy timeframe (not indicator timeframes)
        # This is the key architectural change: strategy controls timeframe
        market_data = self._prepare_indicator_data(
            market_symbol=market,           # Standard format like "ETH-USD"
            timeframe=strategy_timeframe,   # Strategy's timeframe overrides indicator defaults
            exchange=exchange,              # For symbol conversion in API calls
            required_periods=required_periods
        )

        # Abort if no data available - indicators cannot run without data
        if market_data.empty:
            logger.warning(f"No data available for strategy '{strategy_name}' on market {market}")
            return signals

        # Process each indicator with strategy context
        for indicator_name in indicator_names:
            try:
                indicator = self.indicators[indicator_name]

                # Final data sufficiency check per indicator
                indicator_required_periods = getattr(indicator, 'required_periods', 50)
                if len(market_data) < indicator_required_periods:
                    logger.warning(f"Insufficient data for indicator '{indicator_name}': need {indicator_required_periods}, have {len(market_data)}")
                    continue  # Skip this indicator but continue with others

                # Key change: Pass strategy timeframe to indicator
                # The indicator should use strategy timeframe instead of its default
                logger.debug(f"Processing indicator '{indicator_name}' with strategy timeframe '{strategy_timeframe}'")

                # Process indicator with strategy timeframe override
                # Second parameter (strategy_timeframe) tells indicator to use this timeframe
                processed_data, signal = indicator.process(market_data, strategy_timeframe)

                # Enhance signal with strategy context if signal was generated
                if signal:
                    # Add strategy metadata to signal for downstream processing
                    signal.strategy_name = strategy_name    # Which strategy generated this
                    signal.market = market                  # Standard market format
                    signal.exchange = exchange              # Target exchange for routing
                    signal.timeframe = strategy_timeframe   # Timeframe used for analysis

                    signals.append(signal)
                    logger.info(f"Signal generated by indicator '{indicator_name}' for strategy '{strategy_name}': {signal}")

            except Exception as e:
                # Log error but continue processing other indicators
                # One indicator failure shouldn't stop the entire strategy
                logger.error(f"Error running indicator '{indicator_name}' for strategy '{strategy_name}': {e}", exc_info=True)

        logger.debug(f"Strategy '{strategy_name}' generated {len(signals)} signals for {market}")
        return signals

    def run_indicators(self, symbol: str) -> List[Signal]:
        """
        Run all indicators for a symbol and collect signals.
        Ensures sufficient historical data is loaded before running.
        Update relevant Prometheus metrics.

        Args:
            symbol: Market symbol to run indicators for

        Returns:
            List of signals generated by indicators
        """
        signals = []
        generated_signal_state = 0 # Default to NEUTRAL

        # --- Prepare Data ---
        # Use a copy of existing data if available, otherwise start empty
        current_data = self.price_data.get(symbol, pd.DataFrame()).copy()

        # Check data for each indicator *before* running it
        for name, indicator in self.indicators.items():
            # --- Get indicator requirements ---
            try:
                # Get indicator-specific timeframe
                required_periods = getattr(indicator, 'required_periods', 50)
                interval = indicator.get_effective_timeframe()  # Use the new method from BaseIndicator
                cache_key = (symbol, interval)
                first_fetch_done = self.historical_data_fetched.get(cache_key, False)

                # Check if this is the target MACD indicator for metrics
                is_target_macd = (
                    isinstance(indicator, MACDIndicator) and
                    symbol == MARKET_ETH_USD and
                    getattr(indicator, 'symbol', None) == MARKET_ETH_USD
                )

                # Add debug logging for the target MACD check
                if isinstance(indicator, MACDIndicator):
                    logger.info(f"MACD check details: indicator={name}, is_instance={isinstance(indicator, MACDIndicator)}, symbol_match={symbol == MARKET_ETH_USD}, indicator_symbol={getattr(indicator, 'symbol', None)}, target={MARKET_ETH_USD}, timeframe={interval}")

                if is_target_macd:
                    logger.info(f"Found target MACD indicator: {name} for {symbol} on {interval} timeframe")

                # If this is the first time running this indicator for this symbol/interval
                if not first_fetch_done:
                    logger.info(f"Indicator {name} requires {required_periods} periods for {symbol}({interval}), cache has {len(current_data)}. Fetching historical data...")

                    # Fetch historical data if needed
                    historical_data = self._fetch_historical_data(
                        symbol=symbol,
                        interval=interval,
                        limit=self.data_window_size,
                        periods=required_periods
                    )

                    if not historical_data.empty:
                        # Update cache with historical data for this specific timeframe
                        cache_key_str = f"{symbol}_{interval}"
                        self.price_data[cache_key_str] = historical_data
                        current_data = historical_data.copy()
                        self.historical_data_fetched[cache_key] = True
                        logger.info(f"Updated cache for {cache_key_str} with {len(historical_data)} periods (after historical fetch).")
                    else:
                        logger.warning(f"Failed to fetch historical data for {symbol} ({interval})")

                # Use the timeframe-specific cached data if available
                cache_key_str = f"{symbol}_{interval}"
                if cache_key_str in self.price_data and not self.price_data[cache_key_str].empty:
                    current_data = self.price_data[cache_key_str].copy()
                elif not current_data.empty:
                    # If we don't have timeframe-specific data, use the general symbol data
                    logger.debug(f"Using general data for {symbol} since no timeframe-specific data available for {interval}")
                else:
                    # Fetch current market data as fallback
                    logger.info(f"No cached data available for {symbol} ({interval}), fetching current market data")
                    current_data = self._fetch_current_market_data(symbol)

                # --- END Prepare Data ---

                # Verify we have sufficient data
                if len(current_data) < required_periods:
                    logger.warning(f"Insufficient data for indicator {name}: need {required_periods}, have {len(current_data)}")
                    continue

                # Run the indicator
                try:
                    processed_data, signal = indicator.process(current_data)

                    # --- Update signal-related Prometheus metrics for MVP strategy ---
                    if is_target_macd:
                        if signal and signal.direction in [SignalDirection.BUY, SignalDirection.SELL]:
                            generated_signal_state = 1 if signal.direction == SignalDirection.BUY else -1
                        else:
                            generated_signal_state = 0

                        # Update MACD indicator metrics using the indicator's timeframe
                        if 'macd' in processed_data.columns and not processed_data['macd'].empty:
                            latest_values = processed_data.iloc[-1]
                            if not pd.isna(latest_values['macd']):
                                update_macd_indicator(market=symbol, timeframe=interval, component="macd_line", value=float(latest_values['macd']))
                                update_macd_indicator(market=symbol, timeframe=interval, component="signal_line", value=float(latest_values['macd_signal']))
                                update_macd_indicator(market=symbol, timeframe=interval, component="histogram", value=float(latest_values['macd_histogram']))
                                logger.debug(f"Updated MACD metrics for {symbol} on {interval} timeframe")

                    if signal:
                        signals.append(signal)
                        logger.info(f"Signal generated by {name}: {signal}")

                    # --- Record Signal Metrics ---
                    # (existing signal metrics code...)

                except Exception as e:
                    logger.error(f"Error running indicator {name}: {e}", exc_info=True)
                    if is_target_macd:
                        # Set signal state to neutral on error using the indicator's timeframe
                        update_mvp_signal_state(market=symbol, timeframe=interval, state=0) # Set to Neutral on error

            except Exception as e:
                logger.error(f"Error processing indicator {name}: {e}", exc_info=True)

        # --- Update MVP Signal State ---
        # Find the first MACD indicator to determine which timeframe to use for MVP metrics
        macd_timeframe = "1m"  # Default fallback
        for name, indicator in self.indicators.items():
            if isinstance(indicator, MACDIndicator) and symbol == MARKET_ETH_USD:
                macd_timeframe = indicator.get_effective_timeframe()
                break

        update_mvp_signal_state(market=symbol, timeframe=macd_timeframe, state=generated_signal_state)

        # --- Fetch Current Market Data ---
        # Only fetch if we have indicators to run
        if self.indicators:
            current_candle_data = self._fetch_current_market_data(symbol)
            if not current_candle_data.empty:
                latest_candle = current_candle_data.iloc[-1]

                # Update candle metrics using the first available timeframe or default to 1m
                primary_timeframe = "1m"  # Default for live candle updates
                if self.indicators:
                    # Use the timeframe of the first indicator as primary
                    first_indicator = next(iter(self.indicators.values()))
                    primary_timeframe = first_indicator.get_effective_timeframe()

                update_candle_data(market=symbol, timeframe=primary_timeframe, field="open", value=float(latest_candle['open']))
                update_candle_data(market=symbol, timeframe=primary_timeframe, field="high", value=float(latest_candle['high']))
                update_candle_data(market=symbol, timeframe=primary_timeframe, field="low", value=float(latest_candle['low']))
                update_candle_data(market=symbol, timeframe=primary_timeframe, field="close", value=float(latest_candle['close']))
                logger.debug(f"Updated candle metrics for {symbol} on {primary_timeframe} timeframe")

        return signals

    async def run_cycle(self, symbols: List[str] = None) -> int:
        """
        Run a full cycle using strategy-driven execution.

        Args:
            symbols: List of symbols to run for, or None to use strategies' markets

        Returns:
            Number of signals generated and processed
        """
        if not self.strategies:
            logger.warning("No strategies configured, skipping strategy cycle")
            return 0

        signal_count = 0

        logger.info(f"Running strategy-driven cycle for {len(self.strategies)} strategies")

        for strategy in self.strategies:
            try:
                # Handle both StrategyConfig objects and dictionary configurations
                if hasattr(strategy, 'name'):
                    # StrategyConfig object
                    strategy_name = strategy.name
                    enabled = strategy.enabled
                    market = strategy.market
                    exchange = strategy.exchange
                    strategy_dict = strategy.to_dict()  # Convert to dict for run_strategy_indicators
                else:
                    # Dictionary configuration (legacy support)
                    strategy_name = strategy.get("name", "unknown")
                    enabled = strategy.get("enabled", True)
                    market = strategy.get("market")
                    exchange = strategy.get("exchange")
                    strategy_dict = strategy

                # Validate strategy configuration
                if not enabled:
                    logger.debug(f"Strategy '{strategy_name}' is disabled, skipping")
                    continue

                if not market:
                    logger.error(f"Strategy '{strategy_name}' missing market configuration")
                    continue

                if not exchange:
                    logger.error(f"Strategy '{strategy_name}' missing exchange configuration")
                    continue

                # Validate market format (must contain "-")
                if "-" not in market:
                    logger.error(f"Strategy '{strategy_name}' has invalid market format '{market}' - must contain '-' (e.g., 'ETH-USD')")
                    continue

                # Optional symbol filtering
                if symbols and market not in symbols:
                    logger.debug(f"Strategy '{strategy_name}' market '{market}' not in specified symbols, skipping")
                    continue

                # Get indicators for this strategy
                indicator_names = self.strategy_indicators.get(strategy_name, [])
                if not indicator_names:
                    logger.warning(f"Strategy '{strategy_name}' has no indicators defined")
                    continue

                logger.info(f"Running strategy '{strategy_name}' for market '{market}' on exchange '{exchange}'")

                # Run strategy indicators
                signals = self.run_strategy_indicators(strategy_dict, market, indicator_names)

                # Process any signals generated
                for signal in signals:
                    try:
                        success = await self.trading_engine.process_signal(signal)
                        if success:
                            signal_count += 1
                            logger.info(f"Processed signal from strategy '{strategy_name}' for {market}")
                        else:
                            logger.warning(f"Failed to process signal from strategy '{strategy_name}' for {market}")
                    except Exception as e:
                        logger.error(f"Error processing signal from strategy '{strategy_name}': {e}", exc_info=True)

            except Exception as e:
                logger.error(f"Error in strategy cycle for strategy '{getattr(strategy, 'name', strategy.get('name', 'unknown')) if hasattr(strategy, 'name') else strategy.get('name', 'unknown')}': {e}", exc_info=True)

        logger.info(f"Strategy cycle completed: processed {signal_count} signals from {len(self.strategies)} strategies")
        return signal_count
