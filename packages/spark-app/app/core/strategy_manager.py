import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from connectors.base_connector import BaseConnector
from core.trading_engine import TradingEngine
from indicators.base_indicator import BaseIndicator, Signal, SignalDirection
from indicators.macd_indicator import MACDIndicator
from metrics import (INDICATOR_MACD, MARKET_ETH_USD, TIMEFRAME_1M,
                     record_mvp_signal_latency, update_candle_data,
                     update_macd_indicator, update_mvp_signal_state)
from metrics.registry import (publish_historical_time_series,
                              set_historical_data_mode, verify_historical_data)

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
        data_window_size: int = 100,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize the strategy manager.

        Args:
            trading_engine: Trading engine instance to forward signals to
            indicators: Dictionary of indicator instances
            data_window_size: Size of the price data window to maintain
            config: Application configuration dictionary
        """
        self.trading_engine = trading_engine
        self.indicators = indicators or {}
        self.data_window_size = data_window_size
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.historical_data_fetched: Dict[Tuple[str, str], bool] = {}
        self.config = config or {}

        # Check if publishing historical data to metrics is enabled
        self.publish_historical = self.config.get("metrics_publish_historical", False)
        if self.publish_historical:
            logger.info("Historical metrics publishing is enabled - will publish all candles to metrics")

        logger.info(
            f"Strategy manager initialized with {len(self.indicators)} indicators"
        )

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
        Publish historical data to metrics for visualization.
        This is essential for data persistence across container restarts.

        Args:
            symbol: Market symbol
            interval: Time interval
            candles: List of candle dictionaries
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
            # --- Placeholder: Get indicator requirements ---
            # These need to be implemented in the BaseIndicator/specific indicators
            try:
                # Assume indicators have 'required_periods' and 'interval' attributes
                # Default to 50 periods and '1m' if not specified
                required_periods = getattr(indicator, 'required_periods', 50)
                interval = getattr(indicator, 'interval', '1m')
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
                    logger.info(f"MACD check details: indicator={name}, is_instance={isinstance(indicator, MACDIndicator)}, symbol_match={symbol == MARKET_ETH_USD}, indicator_symbol={getattr(indicator, 'symbol', None)}, target={MARKET_ETH_USD}")

                if is_target_macd:
                    logger.info(f"Found target MACD indicator: {name} for {symbol}")

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
                        # Update cache with historical data
                        self.price_data[symbol] = historical_data
                        current_data = historical_data.copy()
                        self.historical_data_fetched[cache_key] = True
                        logger.info(f"Updated cache for {symbol} with {len(historical_data)} periods (after historical fetch).")
                    else:
                        logger.warning(f"Failed to fetch historical data for {symbol} ({interval})")

            except Exception as e:
                logger.error(f"Could not get requirements for indicator {name}: {e}. Skipping data check.")
                continue

        # --- Fetch Latest Tick ---
        logger.debug(f"Fetching latest tick for {symbol} to append...")
        latest_tick_data = self._fetch_current_market_data(symbol)

        if not latest_tick_data.empty:
            # Use concat to append and remove duplicates based on timestamp
            combined_data = pd.concat([current_data, latest_tick_data]).drop_duplicates(subset=['timestamp'], keep='last')
            combined_data = combined_data.sort_values(by='timestamp').reset_index(drop=True)
            current_data = combined_data.tail(self.data_window_size) # Ensure window size
            self.price_data[symbol] = current_data # Update cache with latest tick included
            logger.debug(f"Appended latest tick. Cache size for {symbol}: {len(current_data)}")
        else:
            logger.warning(f"Failed to fetch latest tick for {symbol}. Indicators will run on existing cached data.")

        # --- End Prepare Data ---

        # Check if data is still empty after all attempts
        if self.price_data.get(symbol, pd.DataFrame()).empty:
            logger.warning(f"Price data for {symbol} is empty after fetch attempts. Cannot run indicators.")
            # Update signal state to neutral if applicable
            if symbol == MARKET_ETH_USD: # Assuming this check is still relevant for MVP
                try:
                    update_mvp_signal_state(market=symbol, timeframe="1m", state=0)
                except Exception as metric_e:
                    logger.error(f"Failed to update NEUTRAL signal state metric: {metric_e}")
            return signals # Return empty list

        # Use the final prepared data from cache for running indicators
        data_to_process = self.price_data[symbol].copy()

        # Run each indicator
        for name, indicator in self.indicators.items():
            try:
                # Check if this is the target MACD indicator for metrics
                is_target_macd = (
                    isinstance(indicator, MACDIndicator) and
                    symbol == MARKET_ETH_USD and
                    getattr(indicator, 'symbol', None) == MARKET_ETH_USD
                )

                # Process data through indicator
                processed_data = indicator.calculate(data_to_process)
                signal = indicator.generate_signal(processed_data)

                # --- Metrics Update for MACD MVP ---
                if is_target_macd:
                    logger.debug(f"Processing target MACD indicator for {symbol}")
                    # Update MACD component values
                    if not processed_data.empty and all(col in processed_data.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
                        latest_values = processed_data.iloc[-1]
                        try:
                            update_macd_indicator(market=symbol, timeframe="1m", component="macd_line", value=float(latest_values['macd']))
                            update_macd_indicator(market=symbol, timeframe="1m", component="signal_line", value=float(latest_values['macd_signal']))
                            update_macd_indicator(market=symbol, timeframe="1m", component="histogram", value=float(latest_values['macd_histogram']))
                            logger.debug(f"Updated MACD value metrics for {symbol}")
                        except Exception as e:
                            logger.error(f"Failed to update MACD value metrics for {symbol}: {e}", exc_info=True)
                    else:
                        logger.warning(f"MACD columns missing or data empty in processed_data for {symbol}. Cannot update value metrics.")

                    # Determine and update signal state metric
                    if signal:
                        if signal.direction == SignalDirection.BUY:
                            generated_signal_state = 1
                        elif signal.direction == SignalDirection.SELL:
                            generated_signal_state = -1
                        else: # Handle NEUTRAL case
                            generated_signal_state = 0
                    else:
                        # Check if enough data was available to even generate a signal
                        required_periods_met = len(data_to_process) >= getattr(indicator, 'required_periods', 50)
                        if required_periods_met:
                            logger.debug(f"MACD for {symbol} ran with enough data but generated NO signal. State: NEUTRAL.")
                            generated_signal_state = 0 # No signal means NEUTRAL state if enough data
                        else:
                            logger.debug(f"MACD for {symbol} ran with INSUFFICIENT data. State: NEUTRAL.")
                            generated_signal_state = 0 # Still NEUTRAL if not enough data

                    # Update the metric state outside the inner try/except for values
                    try:
                        update_mvp_signal_state(market=symbol, timeframe="1m", state=generated_signal_state)
                        logger.debug(f"Updated MACD MVP signal state metric to {generated_signal_state} for {symbol}")
                    except Exception as e:
                        logger.error(f"Failed to update signal state metric for {symbol}: {e}", exc_info=True)
                # --- End Metrics Update ---

                # Add signal if one was generated
                if signal:
                    logger.info(f"Indicator {name} for {symbol} generated signal: {signal}")
                    signals.append(signal)

            except Exception as e:
                logger.error(f"Error running indicator {name} for {symbol}: {e}", exc_info=True)
                # If target MACD fails during processing, ensure state is set to NEUTRAL
                if is_target_macd:
                    try:
                        update_mvp_signal_state(market=symbol, timeframe="1m", state=0) # Set to Neutral on error
                        logger.warning(f"Setting signal state to NEUTRAL due to error during MACD processing for {symbol}")
                    except Exception as metric_e:
                        logger.error(f"Failed to update NEUTRAL signal state metric after error: {metric_e}")

        # --- Handle Case Where Target MACD Indicator Is Not Configured At All ---
        # Check if the target symbol was processed but no MACD indicator was found
        target_symbol_processed = symbol == MARKET_ETH_USD
        target_macd_exists = any(
            isinstance(ind, MACDIndicator) and
            getattr(ind, 'symbol', None) == MARKET_ETH_USD
            for ind in self.indicators.values()
        )

        if target_symbol_processed and not target_macd_exists:
            try:
                update_mvp_signal_state(market=symbol, timeframe="1m", state=0) # Ensure Neutral if MACD isn't configured
                logger.debug(f"Target MACD indicator not configured for {symbol}. Setting signal state metric to NEUTRAL.")
            except Exception as metric_e:
                logger.error(f"Failed to update NEUTRAL signal state metric when MACD not configured: {metric_e}")

        # --- Update Candle Data Directly ---
        if symbol == MARKET_ETH_USD and not data_to_process.empty:
            try:
                latest_candle = data_to_process.iloc[-1]
                logger.info(f"Directly updating candle data metrics from run_indicators for {symbol}: {latest_candle[['open', 'high', 'low', 'close']].to_dict()}")

                update_candle_data(market=symbol, timeframe="1m", field="open", value=float(latest_candle['open']))
                update_candle_data(market=symbol, timeframe="1m", field="high", value=float(latest_candle['high']))
                update_candle_data(market=symbol, timeframe="1m", field="low", value=float(latest_candle['low']))
                update_candle_data(market=symbol, timeframe="1m", field="close", value=float(latest_candle['close']))

                logger.info(f"Candle data metrics updated for {symbol}")
            except Exception as e:
                logger.error(f"Failed to update candle metrics directly: {e}", exc_info=True)

        return signals

    async def run_cycle(self, symbols: List[str] = None) -> int:
        """
        Run a full cycle of the strategy for all or specified symbols.

        Args:
            symbols: List of symbols to run for, or None for all

        Returns:
            Number of signals generated and processed
        """
        if not self.indicators:
            logger.warning("No indicators registered, skipping strategy cycle")
            return 0

        signal_count = 0
        run_symbols = symbols or list(self.price_data.keys())

        # If no symbols specified and no price data, use unique symbols from indicators
        if not run_symbols:
            indicator_symbols = set()
            for indicator in self.indicators.values():
                # Extract symbol from indicator if available
                if hasattr(indicator, "symbol"):
                    indicator_symbols.add(indicator.symbol)

            run_symbols = list(indicator_symbols)

        logger.info(f"Running strategy cycle for {len(run_symbols)} symbols")

        for symbol in run_symbols:
            try:
                # Run indicators for this symbol
                signals = self.run_indicators(symbol)

                # Process any signals generated
                for signal in signals:
                    success = await self.trading_engine.process_signal(signal)
                    if success:
                        signal_count += 1
                        logger.info(f"Processed signal for {symbol}")
                    else:
                        logger.warning(f"Failed to process signal for {symbol}")
            except Exception as e:
                logger.error(
                    f"Error in strategy cycle for {symbol}: {e}", exc_info=True
                )

        return signal_count
