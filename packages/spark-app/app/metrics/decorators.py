import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, Union

from .prometheus_exporter import exporter

logger = logging.getLogger(__name__)

def track_api_latency(exchange: str, endpoint: str):
    """
    Decorator to track API request latency and increment request counters.

    Args:
        exchange: The exchange being called
        endpoint: The endpoint being called

    Returns:
        Decorator function that tracks API metrics
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            status = "success"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                latency = time.time() - start_time

                # Record the API request if the exporter is available
                if exporter:
                    exporter.record_api_request(exchange, endpoint, latency, status)
                else:
                    logger.debug(
                        f"API request to {exchange}/{endpoint} took {latency:.3f}s (status: {status})"
                    )
        return wrapper
    return decorator

def track_order_execution(exchange: str, market: str, order_type: str):
    """
    Decorator to track order execution time.

    Args:
        exchange: The exchange where the order is being placed
        market: The market for the order
        order_type: The type of order (market, limit, etc.)

    Returns:
        Decorator function that tracks order execution time
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time

                # Record the order execution if the exporter is available
                if exporter:
                    exporter.record_order_execution(exchange, market, order_type, execution_time)
                else:
                    logger.debug(
                        f"Order execution on {exchange}/{market} ({order_type}) took {execution_time:.3f}s"
                    )
        return wrapper
    return decorator

def update_rate_limit(exchange: str, endpoint: str, remaining: Optional[int] = None):
    """
    Update the rate limit metrics for an exchange and endpoint.

    Args:
        exchange: The exchange name
        endpoint: The API endpoint
        remaining: Number of requests remaining
    """
    if exporter and remaining is not None:
        exporter.update_rate_limit(exchange, endpoint, remaining)

def update_margin_ratio(exchange: str, market: str, ratio: float):
    """
    Update the margin ratio for a market.

    Args:
        exchange: The exchange name
        market: The market symbol
        ratio: Current margin ratio (0-1)
    """
    if exporter:
        exporter.update_margin_ratio(exchange, market, ratio)
