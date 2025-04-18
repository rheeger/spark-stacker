import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Union

import requests
from prometheus_client import Gauge, start_http_server

# Import the custom registry
from .metrics import custom_registry

logger = logging.getLogger(__name__)

# Default metrics port
DEFAULT_METRICS_PORT = 9000
DEFAULT_METRICS_HOST = "0.0.0.0"  # Bind to all interfaces

# Server instance reference
_server_thread: Optional[threading.Thread] = None
_is_running = False
_standard_scrape_interval = 15.0  # Default Prometheus scrape interval in seconds

# Cache for historical time series data
# Format: {(market, timeframe, field): [(timestamp, value), ...]}
_time_series_cache: Dict[tuple, List[tuple]] = {}

# Gauge for each time series that includes the timestamp in the label
_time_series_gauges: Dict[str, Gauge] = {}


def register_time_series_gauge(name: str, description: str, label_names: List[str]) -> Gauge:
    """
    Register a gauge for time series data with timestamp as a label.

    Args:
        name: Name of the gauge
        description: Description of the gauge
        label_names: Names of the labels

    Returns:
        The registered gauge
    """
    if name not in _time_series_gauges:
        gauge = Gauge(
            name,
            description,
            label_names,
            registry=custom_registry
        )
        _time_series_gauges[name] = gauge
        logger.info(f"Registered time series gauge: {name}")
    return _time_series_gauges[name]


def publish_historical_time_series(
    market: str,
    timeframe: str,
    field: str,
    data_points: List[Dict[str, Union[int, float]]],
    gauge_name: str = "spark_stacker_historical_series",
    gauge_description: str = "Historical time series data"
) -> bool:
    """
    Publish a series of historical time-indexed data points to Prometheus.

    This creates a special gauge that includes the timestamp in the label,
    which allows Prometheus to store multiple values for the same metric.

    Args:
        market: Market symbol (e.g., "ETH-USD")
        timeframe: Time interval (e.g., "1m")
        field: Data field (e.g., "close", "macd_line")
        data_points: List of {timestamp: int, value: float} dictionaries
        gauge_name: Base name for the gauge
        gauge_description: Description for the gauge

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create a unique name for this time series
        unique_name = f"{gauge_name}_{market.replace('-', '_')}_{timeframe}_{field}"

        # Store in cache for easy retrieval
        cache_key = (market, timeframe, field)
        _time_series_cache[cache_key] = [(point["timestamp"], point["value"]) for point in data_points]

        # Register the gauge if it doesn't exist
        gauge = register_time_series_gauge(
            unique_name,
            f"{gauge_description} for {market} {timeframe} {field}",
            ["timestamp"]
        )

        # Clear any existing values for this gauge
        for label in gauge._metrics:
            del gauge._metrics[label]

        # Publish each data point with timestamp as a label
        for point in data_points:
            timestamp = str(point["timestamp"])
            value = float(point["value"])
            gauge.labels(timestamp=timestamp).set(value)

        logger.info(f"Published {len(data_points)} historical points for {market}/{timeframe}/{field}")
        return True

    except Exception as e:
        logger.error(f"Error publishing historical time series: {e}")
        return False


def get_historical_time_series(
    market: str,
    timeframe: str,
    field: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> List[Dict[str, Union[int, float]]]:
    """
    Get historical time series data from the cache.

    Args:
        market: Market symbol (e.g., "ETH-USD")
        timeframe: Time interval (e.g., "1m")
        field: Data field (e.g., "close", "macd_line")
        start_time: Optional start time in milliseconds
        end_time: Optional end time in milliseconds

    Returns:
        List of {timestamp: int, value: float} dictionaries
    """
    cache_key = (market, timeframe, field)
    if cache_key not in _time_series_cache:
        return []

    data = _time_series_cache[cache_key]
    if start_time is not None:
        data = [point for point in data if point[0] >= start_time]
    if end_time is not None:
        data = [point for point in data if point[0] <= end_time]

    return [{"timestamp": point[0], "value": point[1]} for point in data]


def start_metrics_server(
    port: int = DEFAULT_METRICS_PORT, host: str = DEFAULT_METRICS_HOST
) -> None:
    """
    Start the Prometheus metrics server on the specified port.

    Args:
        port: The port to expose metrics on (default: 9000)
        host: The host interface to bind to (default: 0.0.0.0)
    """
    global _server_thread, _is_running

    if _is_running:
        logger.warning("Metrics server is already running")
        return

    def _run_server() -> None:
        global _is_running
        try:
            logger.info(f"Starting metrics server on {host}:{port}")
            # Use the custom registry instead of the default one
            start_http_server(port, addr=host, registry=custom_registry)
            _is_running = True
            logger.info(f"Metrics server started on {host}:{port}")

            # Keep the thread alive
            while _is_running:
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error in metrics server: {str(e)}")
            _is_running = False

    _server_thread = threading.Thread(target=_run_server, daemon=True)
    _server_thread.start()

    # Give the server a moment to start
    time.sleep(0.5)

    if not _is_running:
        logger.error("Failed to start metrics server")


def stop_metrics_server() -> None:
    """Stop the Prometheus metrics server."""
    global _is_running

    if not _is_running:
        logger.warning("Metrics server is not running")
        return

    logger.info("Stopping metrics server")
    _is_running = False

    if _server_thread:
        _server_thread.join(timeout=5)
        logger.info("Metrics server stopped")


def set_historical_data_mode(enabled: bool = True) -> None:
    """
    Set the metrics server to historical data mode.

    In historical data mode, we use a different approach to publish metrics
    that ensures Prometheus can capture multiple updates in a short time window.

    Args:
        enabled: Whether historical data mode is enabled
    """
    logger.info(f"Setting historical data mode: {enabled}")

    # This function is a placeholder for now - it would normally interact
    # with Prometheus configuration or provide a way to signal to the
    # metrics publishing code that we're in historical mode.
    #
    # For now, our strategy is to add delays in the code that publishes
    # historical data to ensure Prometheus can scrape each update.


def verify_historical_data(market: str = "ETH-USD", timeframe: str = "1m") -> bool:
    """
    Verify that historical data is being properly published to Prometheus.

    This function checks if the historical time series data is available
    in our custom historical time series gauges.

    Args:
        market: Market symbol (default: ETH-USD)
        timeframe: Timeframe (default: 1m)

    Returns:
        True if historical data is available, False otherwise
    """
    logger.info(f"Verifying historical data availability for {market} ({timeframe})")

    try:
        # Check the historical time series cache
        expected_fields = {"open", "high", "low", "close"}
        available_fields = set()

        for cache_key in _time_series_cache.keys():
            cache_market, cache_timeframe, field = cache_key
            if cache_market == market and cache_timeframe == timeframe:
                available_fields.add(field)
                data_points = _time_series_cache[cache_key]
                logger.info(f"Found {len(data_points)} historical points for {field}")

        # Check if all expected fields are available
        missing_fields = expected_fields - available_fields

        if missing_fields:
            logger.warning(f"Missing fields in historical data: {missing_fields}")
            return False

        # Check if we have a reasonable number of data points
        for field in expected_fields:
            cache_key = (market, timeframe, field)
            if cache_key in _time_series_cache:
                if len(_time_series_cache[cache_key]) < 5:
                    logger.warning(f"Too few data points for {field}: {len(_time_series_cache[cache_key])}")
                    return False

        # Also check if the time series gauges were created
        gauge_name_pattern = f"spark_stacker_historical_candle_{market.replace('-', '_')}_{timeframe}"

        gauge_count = 0
        for name in _time_series_gauges:
            if gauge_name_pattern in name:
                gauge_count += 1

        if gauge_count < len(expected_fields):
            logger.warning(f"Expected {len(expected_fields)} gauges, but found only {gauge_count}")
            return False

        logger.info(f"✅ Verified historical data for {market}/{timeframe}: {len(available_fields)} fields with data")
        return True

    except Exception as e:
        logger.error(f"Error verifying historical data: {e}")
        return False


def start_historical_data_api(port: int = 9001):
    """
    Start a simple Flask API server to serve historical time series data.

    This creates an API endpoint that Grafana can query directly to get
    historical time series data with proper timestamps.

    Args:
        port: Port to run the API server on (default: 9001)
    """
    try:
        import flask
        from flask import Flask, jsonify, request

        app = Flask("HistoricalDataAPI")

        @app.route("/api/v1/historical", methods=["GET"])
        def get_historical_data():
            """API endpoint to get historical time series data"""
            market = request.args.get("market", "ETH-USD")
            timeframe = request.args.get("timeframe", "1m")
            field = request.args.get("field", "close")
            start_time = request.args.get("start")
            end_time = request.args.get("end")

            if start_time:
                start_time = int(start_time)
            if end_time:
                end_time = int(end_time)

            cache_key = (market, timeframe, field)
            if cache_key not in _time_series_cache:
                return jsonify({"status": "error", "message": "No data found"}), 404

            data = _time_series_cache[cache_key]
            if start_time:
                data = [point for point in data if point[0] >= start_time]
            if end_time:
                data = [point for point in data if point[0] <= end_time]

            # Format data for Grafana
            result = [
                {"timestamp": point[0], "value": point[1]}
                for point in data
            ]

            return jsonify({
                "status": "success",
                "data": result,
                "meta": {
                    "market": market,
                    "timeframe": timeframe,
                    "field": field,
                    "count": len(result)
                }
            })

        @app.route("/metrics/list", methods=["GET"])
        def list_available_metrics():
            """List all available metrics"""
            metrics = []
            for cache_key in _time_series_cache:
                market, timeframe, field = cache_key
                metrics.append({
                    "market": market,
                    "timeframe": timeframe,
                    "field": field,
                    "count": len(_time_series_cache[cache_key])
                })

            return jsonify({
                "status": "success",
                "metrics": metrics
            })

        # Start the API server in a separate thread
        import threading

        def run_api_server():
            app.run(host="0.0.0.0", port=port)

        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()

        logger.info(f"Started historical data API server on port {port}")
        return True

    except ImportError:
        logger.error("Flask is required to run the historical data API server")
        return False
    except Exception as e:
        logger.error(f"Failed to start historical data API server: {e}")
        return False
