#!/usr/bin/env python3
"""
Mock Metrics Exporter for Spark Stacker

This script generates mock metrics for the Spark Stacker application
until real metrics are implemented. It exposes a Prometheus endpoint
with synthetic trading metrics.
"""

import logging
import math
import random
import time
from typing import Dict, List, Union

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
EXCHANGES = ["binance", "coinbase", "ftx", "kraken"]
MARKETS = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]
ENDPOINTS = ["order", "market", "position", "account", "trade"]


class SparkStackerMetrics:
    """Generate and export mock metrics for Spark Stacker"""

    def __init__(self, port: int = 8000):
        """Initialize the metrics exporter.

        Args:
            port: Port to expose metrics on
        """
        self.port = port

        # Initialize metrics
        self.uptime = Gauge(
            "spark_stacker_uptime_seconds",
            "Application uptime in seconds"
        )

        self.trades = Counter(
            "spark_stacker_trades_total",
            "Total number of trades",
            ["exchange", "market"]
        )

        self.active_positions = Gauge(
            "spark_stacker_active_positions",
            "Number of active positions",
            ["exchange", "market"]
        )

        self.api_requests = Counter(
            "spark_stacker_api_requests_total",
            "Number of API requests made",
            ["exchange", "endpoint"]
        )

        self.api_latency = Histogram(
            "spark_stacker_api_latency_seconds",
            "API request latency in seconds",
            ["endpoint"]
        )

        self.margin_ratio = Gauge(
            "spark_stacker_margin_ratio",
            "Current margin ratio",
            ["market"]
        )

        self.pnl_percent = Gauge(
            "spark_stacker_pnl_percent",
            "PnL as a percentage",
            ["market"]
        )

        # Keep track of start time
        self.start_time = time.time()

        # Register update methods
        self.update_methods = [
            self.update_uptime,
            self.update_trades,
            self.update_positions,
            self.update_api_metrics,
            self.update_margin_ratio,
            self.update_pnl,
        ]

    def start(self):
        """Start the metrics server"""
        start_http_server(self.port)
        logger.info(f"Metrics server started on port {self.port}")

        try:
            while True:
                self.update_all_metrics()
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Stopping metrics server")

    def update_all_metrics(self):
        """Update all metrics with new values"""
        for method in self.update_methods:
            try:
                method()
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")

    def update_uptime(self):
        """Update the uptime metric"""
        uptime_seconds = time.time() - self.start_time
        self.uptime.set(uptime_seconds)

    def update_trades(self):
        """Simulate new trades occurring"""
        for exchange in EXCHANGES:
            for market in MARKETS:
                # Simulate some markets being more active than others
                if random.random() < 0.3:
                    trade_count = random.randint(1, 5)
                    self.trades.labels(exchange=exchange, market=market).inc(trade_count)

    def update_positions(self):
        """Update active positions"""
        for exchange in EXCHANGES:
            for market in MARKETS:
                # Randomly update position counts
                if random.random() < 0.1:
                    position_count = random.randint(0, 10)
                    self.active_positions.labels(exchange=exchange, market=market).set(position_count)

    def update_api_metrics(self):
        """Update API request and latency metrics"""
        for exchange in EXCHANGES:
            for endpoint in ENDPOINTS:
                # Simulate API calls
                if random.random() < 0.4:
                    request_count = random.randint(1, 10)
                    self.api_requests.labels(exchange=exchange, endpoint=endpoint).inc(request_count)

                    # Simulate API latency
                    latency = random.uniform(0.01, 0.5)
                    self.api_latency.labels(endpoint=endpoint).observe(latency)

    def update_margin_ratio(self):
        """Update margin ratios for all markets"""
        for market in MARKETS:
            # Simulate margin ratios between 0.1 and 0.9
            ratio = random.uniform(0.1, 0.9)
            self.margin_ratio.labels(market=market).set(ratio)

    def update_pnl(self):
        """Update PnL percentage for all markets"""
        for market in MARKETS:
            # Simulate PnL between -10% and +10%
            pnl = random.uniform(-0.1, 0.1)
            self.pnl_percent.labels(market=market).set(pnl)


if __name__ == "__main__":
    logger.info("Starting Spark Stacker mock metrics exporter")
    metrics = SparkStackerMetrics()
    metrics.start()
