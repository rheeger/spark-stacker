import logging
import time
from typing import Any, Dict, Optional

from prometheus_client import (Counter, Gauge, Histogram, Summary,
                               start_http_server)

logger = logging.getLogger(__name__)

class PrometheusExporter:
    """Prometheus metrics exporter for Spark Stacker"""

    def __init__(self, port: int = 9000):
        """Initialize the metrics exporter

        Args:
            port: Port to expose metrics on
        """
        self.port = port

        # API metrics
        self.api_requests = Counter(
            'spark_stacker_api_requests_total',
            'Number of API requests',
            ['exchange', 'endpoint', 'status']
        )

        self.api_latency = Histogram(
            'spark_stacker_api_latency_seconds',
            'API request latency in seconds',
            ['exchange', 'endpoint'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
        )

        # Order execution metrics
        self.order_execution_time = Histogram(
            'spark_stacker_order_execution_seconds',
            'Order execution time in seconds',
            ['exchange', 'market', 'order_type'],
            buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0)
        )

        # Exchange rate limits
        self.rate_limit_remaining = Gauge(
            'spark_stacker_rate_limit_remaining',
            'Number of API requests remaining before rate limit',
            ['exchange', 'endpoint']
        )

        # Risk metrics
        self.margin_ratio = Gauge(
            'spark_stacker_margin_ratio',
            'Current margin ratio for positions',
            ['exchange', 'market']
        )

        self.liquidation_price = Gauge(
            'spark_stacker_liquidation_price',
            'Liquidation price for positions',
            ['exchange', 'market', 'position_side']
        )

        # Portfolio metrics
        self.capital_utilization = Gauge(
            'spark_stacker_capital_utilization_percent',
            'Percentage of capital currently in use',
            ['exchange']
        )

        self.max_drawdown = Gauge(
            'spark_stacker_max_drawdown_percent',
            'Maximum drawdown percentage',
            ['exchange']
        )

        self.pnl_percent = Gauge(
            'spark_stacker_pnl_percent',
            'Profit and loss percentage',
            ['exchange', 'market', 'timeframe']
        )

        # Start time for uptime tracking
        self.start_time = time.time()

    def start(self):
        """Start the metrics server"""
        start_http_server(self.port)
        logger.info(f"Started Prometheus metrics server on port {self.port}")

    def record_api_request(self, exchange: str, endpoint: str, latency: float, status: str = "success"):
        """Record an API request with its latency

        Args:
            exchange: Exchange name
            endpoint: API endpoint
            latency: Request latency in seconds
            status: Request status (success/error)
        """
        self.api_requests.labels(exchange=exchange, endpoint=endpoint, status=status).inc()
        self.api_latency.labels(exchange=exchange, endpoint=endpoint).observe(latency)

    def record_order_execution(self, exchange: str, market: str, order_type: str, execution_time: float):
        """Record order execution time

        Args:
            exchange: Exchange name
            market: Market symbol
            order_type: Type of order (market, limit, etc.)
            execution_time: Time to execute the order in seconds
        """
        self.order_execution_time.labels(
            exchange=exchange,
            market=market,
            order_type=order_type
        ).observe(execution_time)

    def update_rate_limit(self, exchange: str, endpoint: str, remaining: int):
        """Update rate limit metrics

        Args:
            exchange: Exchange name
            endpoint: API endpoint
            remaining: Number of requests remaining
        """
        self.rate_limit_remaining.labels(exchange=exchange, endpoint=endpoint).set(remaining)

    def update_margin_ratio(self, exchange: str, market: str, ratio: float):
        """Update margin ratio for a market

        Args:
            exchange: Exchange name
            market: Market symbol
            ratio: Current margin ratio (0-1)
        """
        self.margin_ratio.labels(exchange=exchange, market=market).set(ratio)

    def update_liquidation_price(self, exchange: str, market: str, position_side: str, price: float):
        """Update liquidation price for a position

        Args:
            exchange: Exchange name
            market: Market symbol
            position_side: Position side (long/short)
            price: Liquidation price
        """
        self.liquidation_price.labels(
            exchange=exchange,
            market=market,
            position_side=position_side
        ).set(price)

    def update_capital_utilization(self, exchange: str, utilization: float):
        """Update capital utilization percentage

        Args:
            exchange: Exchange name
            utilization: Capital utilization percentage (0-100)
        """
        self.capital_utilization.labels(exchange=exchange).set(utilization)

    def update_max_drawdown(self, exchange: str, drawdown: float):
        """Update maximum drawdown percentage

        Args:
            exchange: Exchange name
            drawdown: Maximum drawdown percentage (positive value)
        """
        self.max_drawdown.labels(exchange=exchange).set(drawdown)

    def update_pnl(self, exchange: str, market: str, timeframe: str, pnl: float):
        """Update profit and loss percentage

        Args:
            exchange: Exchange name
            market: Market symbol
            timeframe: Time period (e.g., daily, weekly)
            pnl: Profit and loss percentage
        """
        self.pnl_percent.labels(exchange=exchange, market=market, timeframe=timeframe).set(pnl)

# Global instance to be used throughout the application
exporter = None

def initialize_metrics(port: int = 9000):
    """Initialize and start the metrics exporter

    Args:
        port: Port to expose metrics on
    """
    global exporter
    exporter = PrometheusExporter(port=port)
    exporter.start()
    return exporter
