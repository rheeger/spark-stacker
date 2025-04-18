#!/usr/bin/env python3
"""
Log-based Metrics Exporter for Spark Stacker

This script parses Spark Stacker log files and exports metrics for Prometheus.
It monitors the connection status, errors, and other information found in the logs.
"""

import argparse
import glob
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_LOG_DIR = "/logs"
DEFAULT_PORT = 9001
LOG_CHECK_INTERVAL = 5  # seconds - check more frequently
MAIN_LOG_PATTERN = "spark_stacker.log"
CONFIG_FILE_PATH = "/config/config.json"

# Error patterns
CONNECTION_ERROR_PATTERNS = {
    "RemoteDisconnected": r"RemoteDisconnected\('Remote end closed connection without response'\)",
    "NameResolutionError": r"NameResolutionError.*Name or service not known",
    "ConnectionTimeout": r"Max retries exceeded with url",
    "ConnectionRefused": r"Connection refused",
    "Other": r"Failed to get positions: .*"
}


class SparkStackerLogMetrics:
    """Parse Spark Stacker logs and export metrics"""

    def __init__(self, log_dir: str, port: int = DEFAULT_PORT):
        """Initialize the metrics exporter.

        Args:
            log_dir: Directory containing Spark Stacker logs
            port: Port to expose metrics on
        """
        self.log_dir = log_dir
        self.port = port
        self.config = self._load_config(CONFIG_FILE_PATH)
        self.enabled_exchanges = self._get_enabled_exchanges()

        # Print log directory contents at startup
        self._print_log_directory_contents()

        # Keep track of last read position
        self.file_positions = {}

        # Exchange and service status
        self.exchange_status = Gauge(
            "spark_stacker_exchange_connection_status",
            "Connection status of exchanges (1=up, 0=down)",
            ["exchange"]
        )

        self.service_status = Gauge(
            "spark_stacker_service_status",
            "Status of services (1=up, 0=down)",
            ["service"]
        )

        # Connection errors
        self.connection_errors = Counter(
            "spark_stacker_connection_errors_total",
            "Connection errors by type",
            ["exchange", "error_type"]
        )

        # Websocket metrics
        self.websocket_ping_latency = Gauge(
            "spark_stacker_websocket_ping_latency_ms",
            "Latency between websocket ping and pong in milliseconds",
            ["exchange"]
        )

        # Balance metrics
        self.account_balance = Gauge(
            "spark_stacker_account_balance",
            "Account balance by currency",
            ["exchange", "currency"]
        )

        # Last connection status
        self.last_connection_status = Gauge(
            "spark_stacker_last_connection_status",
            "Last connection status (1=success, 0=failure)",
            ["exchange"]
        )

        # Exchange market counts
        self.market_count = Gauge(
            "spark_stacker_market_count",
            "Number of markets available on each exchange",
            ["exchange"]
        )

        # System build info
        self.build_info = Gauge(
            "spark_stacker_build_info",
            "Build information",
            ["build_id", "timestamp"]
        )

        # System configuration
        self.system_config = Gauge(
            "spark_stacker_system_config",
            "System configuration",
            ["dry_run", "hedging_enabled"]
        )

        # Add new metrics for order history tracking
        self.order_count = Counter(
            "spark_stacker_orders_total",
            "Total number of orders by type and side",
            ["exchange", "side", "type", "status"],
        )

        self.order_value = Gauge(
            "spark_stacker_order_value",
            "Value of orders in USD equivalent",
            ["exchange", "side"],
        )

        self.last_order_price = Gauge(
            "spark_stacker_last_order_price",
            "Price of the last order executed",
            ["exchange", "symbol"],
        )

        # Initialize exchange status as unknown (0.5) for enabled exchanges
        for exchange in self.enabled_exchanges:
            self.exchange_status.labels(exchange=exchange).set(0.5)
            self.last_connection_status.labels(exchange=exchange).set(0.5)

        # Initialize service status
        self.service_status.labels(service="metrics_server").set(0.5)
        self.service_status.labels(service="webhook_server").set(0.5)

        # Set a default build info from log
        self.build_info.labels(build_id="placeholder", timestamp="placeholder").set(1)
        logger.info("Set placeholder build info. Will update when real build info is found in logs.")

        # Find latest log directory if any
        self.current_log_dir = self._find_latest_log_dir()

        logger.info(f"Initialized metrics for log directory: {self.log_dir}")
        if self.current_log_dir:
            logger.info(f"Using latest log directory: {self.current_log_dir}")
        else:
            logger.warning(f"No log directories found in {self.log_dir}")
            logger.info("Will continue checking for new log directories")

    def _load_config(self, config_path: str) -> Optional[Dict]:
        """Load configuration from JSON file."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    logger.info(f"Successfully loaded configuration from {config_path}")
                    return config_data
            else:
                logger.error(f"Configuration file not found at {config_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            return None

    def _get_enabled_exchanges(self) -> List[str]:
        """Get the list of enabled exchange names from the configuration."""
        enabled = []
        if self.config and 'exchanges' in self.config:
            for exchange_config in self.config['exchanges']:
                if exchange_config.get('enabled', False):
                    enabled.append(exchange_config.get('name'))
            logger.info(f"Enabled exchanges based on config: {enabled}")
        else:
            logger.warning("Could not determine enabled exchanges from config. Falling back to empty list.")
        return enabled

    def _print_log_directory_contents(self):
        """Print the contents of the log directory at startup for debugging."""
        try:
            logger.info(f"Log directory contents ({self.log_dir}):")
            if os.path.exists(self.log_dir):
                for root, dirs, files in os.walk(self.log_dir):
                    level = root.replace(self.log_dir, '').count(os.sep)
                    indent = ' ' * 4 * level
                    logger.info(f"{indent}{os.path.basename(root)}/")
                    sub_indent = ' ' * 4 * (level + 1)
                    for f in files:
                        logger.info(f"{sub_indent}{f}")
            else:
                logger.error(f"Log directory {self.log_dir} does not exist!")
        except Exception as e:
            logger.error(f"Error printing log directory contents: {e}")

    def _find_latest_log_dir(self) -> Optional[str]:
        """Find the latest log directory based on timestamp in name."""
        # Check if log_dir is a directory containing date-named subdirectories
        try:
            # First try listing direct log subdirectories with full path
            log_dirs = glob.glob(os.path.join(self.log_dir, "*"))
            logger.info(f"Checking for log directories in {self.log_dir}")

            if not log_dirs:
                logger.warning(f"No directories found directly in {self.log_dir}")

                # Try looking for the main log file directly
                main_log_path = os.path.join(self.log_dir, MAIN_LOG_PATTERN)
                if os.path.exists(main_log_path):
                    logger.info(f"Found main log file directly at {main_log_path}")
                    return self.log_dir

                # Also check for connector log directories directly in the log_dir
                for connector in self.enabled_exchanges:
                    connector_dir = os.path.join(self.log_dir, connector)
                    if os.path.exists(connector_dir) and os.path.isdir(connector_dir):
                        logger.info(f"Found connector directory: {connector_dir}")
                        return self.log_dir

                return None

            # Filter to only get directories
            log_dirs = [d for d in log_dirs if os.path.isdir(d)]

            if not log_dirs:
                logger.warning(f"No subdirectories found in {self.log_dir}")
                return None

            # Sort by directory creation time (most recent first)
            log_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)

            # Log the directories we found
            logger.info(f"Found {len(log_dirs)} log directories. Most recent: {log_dirs[0]}")

            return log_dirs[0]
        except Exception as e:
            logger.error(f"Error finding log directories: {e}")
            return None

    def start(self):
        """Start the metrics server and begin log processing."""
        start_http_server(self.port)
        logger.info(f"Metrics server started on port {self.port}")

        try:
            while True:
                # Check if we need to update to a newer log directory
                latest_dir = self._find_latest_log_dir()
                if latest_dir != self.current_log_dir:
                    logger.info(f"Switching to new log directory: {latest_dir}")
                    self.current_log_dir = latest_dir
                    self.file_positions = {}  # Reset file positions

                if self.current_log_dir:
                    try:
                        self.process_logs()
                    except Exception as e:
                        logger.error(f"Error processing logs: {e}")
                else:
                    logger.warning("No valid log directory found. Waiting...")

                time.sleep(LOG_CHECK_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Stopping metrics server")

    def process_logs(self):
        """Process all relevant log files."""
        # Process main log file
        try:
            main_log_path = os.path.join(self.current_log_dir, MAIN_LOG_PATTERN)
            if os.path.exists(main_log_path):
                logger.debug(f"Processing main log: {main_log_path}")
                self._process_main_log(main_log_path)
            else:
                logger.debug(f"Main log file not found at {main_log_path}")
        except Exception as e:
            logger.error(f"Error processing main log: {e}")

        # Process connector logs for enabled exchanges
        try:
            for connector in self.enabled_exchanges:
                connector_dir = os.path.join(self.current_log_dir, connector)
                if os.path.exists(connector_dir):
                    self._process_connector_logs(connector, connector_dir)
        except Exception as e:
            logger.error(f"Error processing connector logs: {e}")

    def _process_main_log(self, log_path: str):
        """Process the main spark_stacker.log file."""
        new_lines = self._read_new_lines(log_path)
        if not new_lines:
            return

        # Track exchanges with recent failures
        exchanges_with_errors = set()

        # Pattern matching for logs
        build_id_pattern = r"Container build ID: (\w+)"
        build_timestamp_pattern = r"Build timestamp: ([\d-]+)"
        trading_config_pattern = r"Trading engine started \(dry_run=(\w+), hedging_enabled=(\w+)\)"
        webhook_server_pattern = r"Webhook server started at"
        metrics_server_pattern = r"Metrics server started on"
        market_count_pattern = r"Available markets on (\w+): (\d+) markets found"
        balance_pattern = r"Account balances on (\w+): (\d+) non-zero balances found"
        balance_detail_pattern = r"  (\w+): ([\d.e-]+)"
        connection_error_pattern = r"Failed to get positions: (.*)"
        websocket_ping_pattern = r"Websocket sending ping"
        websocket_pong_pattern = r"Websocket received pong"

        last_ping_time = {}
        exchange_in_context = None

        line_count = 0
        for line in new_lines:
            line_count += 1
            try:
                # Build information
                build_id_match = re.search(build_id_pattern, line)
                if build_id_match:
                    build_id = build_id_match.group(1)
                    logger.info(f"Found build ID: {build_id}")
                    # Store build ID for later use with timestamp
                    self._last_build_id = build_id

                    # Look for timestamp in the same line
                    timestamp_match = re.search(build_timestamp_pattern, line)
                    if timestamp_match:
                        timestamp = timestamp_match.group(1)
                        logger.info(f"Found build timestamp: {timestamp}")
                        # Remove any existing build info metrics
                        self.build_info._metrics.clear()
                        logger.info(f"Updating build info metric: build_id={build_id}, timestamp={timestamp}")
                        self.build_info.labels(build_id=build_id, timestamp=timestamp).set(1)
                    else:
                        logger.warning(f"Build ID found but no timestamp in line: {line}")

                # Also check for timestamp in other lines if we have a build ID already
                timestamp_match = re.search(build_timestamp_pattern, line)
                if timestamp_match and hasattr(self, '_last_build_id'):
                    timestamp = timestamp_match.group(1)
                    logger.info(f"Found build timestamp separately: {timestamp}")
                    # Remove any existing build info metrics
                    self.build_info._metrics.clear()
                    logger.info(f"Updating build info metric: build_id={self._last_build_id}, timestamp={timestamp}")
                    self.build_info.labels(build_id=self._last_build_id, timestamp=timestamp).set(1)

                # Trading configuration
                trading_config_match = re.search(trading_config_pattern, line)
                if trading_config_match:
                    dry_run = trading_config_match.group(1)
                    hedging_enabled = trading_config_match.group(2)
                    logger.info(f"Trading config: dry_run={dry_run}, hedging_enabled={hedging_enabled}")
                    self.system_config.labels(dry_run=dry_run, hedging_enabled=hedging_enabled).set(1)

                # Service status
                if re.search(webhook_server_pattern, line):
                    logger.info("Webhook server is up")
                    self.service_status.labels(service="webhook_server").set(1)

                if re.search(metrics_server_pattern, line):
                    logger.info("Metrics server is up")
                    self.service_status.labels(service="metrics_server").set(1)

                # Track which exchange is in context based on mentions
                for exchange in self.enabled_exchanges:
                    if f"for {exchange}" in line.lower() or f"exchange type: {exchange}" in line.lower():
                        exchange_in_context = exchange
                        logger.debug(f"Exchange context set to {exchange}")

                # Market counts
                market_count_match = re.search(market_count_pattern, line)
                if market_count_match:
                    exchange = market_count_match.group(1)
                    count = int(market_count_match.group(2))
                    logger.info(f"Exchange {exchange} has {count} markets")
                    self.market_count.labels(exchange=exchange).set(count)

                # Account balances
                balance_match = re.search(balance_pattern, line)
                if balance_match:
                    exchange = balance_match.group(1)
                    # Set connection status to UP when balances are found
                    self.exchange_status.labels(exchange=exchange).set(1)
                    self.last_connection_status.labels(exchange=exchange).set(1)
                    exchange_in_context = exchange  # Set context for balance details

                # Balance details
                balance_detail_match = re.search(balance_detail_pattern, line)
                if balance_detail_match and exchange_in_context:
                    currency = balance_detail_match.group(1)
                    amount = float(balance_detail_match.group(2))
                    logger.info(f"Balance for {exchange_in_context}/{currency}: {amount}")
                    self.account_balance.labels(exchange=exchange_in_context, currency=currency).set(amount)

                # Connection errors
                for exchange in self.enabled_exchanges:
                    if "Failed to get positions" in line and exchange in line.lower():
                        exchanges_with_errors.add(exchange)
                        self.exchange_status.labels(exchange=exchange).set(0)
                        self.last_connection_status.labels(exchange=exchange).set(0)

                        # Categorize error type
                        error_type = "Other"
                        for error_name, pattern in CONNECTION_ERROR_PATTERNS.items():
                            if re.search(pattern, line):
                                error_type = error_name
                                break

                        self.connection_errors.labels(exchange=exchange, error_type=error_type).inc()
                        logger.info(f"Connection error for {exchange}: {error_type}")

                # Websocket ping/pong timing
                if "Websocket sending ping" in line:
                    if exchange_in_context:
                        last_ping_time[exchange_in_context] = datetime.now()
                        logger.debug(f"Ping sent for {exchange_in_context}")

                if "Websocket received pong" in line:
                    if exchange_in_context and exchange_in_context in last_ping_time:
                        latency = (datetime.now() - last_ping_time[exchange_in_context]).total_seconds() * 1000
                        self.websocket_ping_latency.labels(exchange=exchange_in_context).set(latency)
                        logger.debug(f"Pong received for {exchange_in_context}, latency: {latency}ms")
            except Exception as e:
                logger.error(f"Error processing line {line_count}: {e}")
                logger.error(f"Line content: {line}")

        logger.info(f"Processed {line_count} lines from main log")

        # Update exchange status if no errors were found in this batch of logs
        for exchange in self.enabled_exchanges:
            if exchange not in exchanges_with_errors and "on_open" in str(new_lines):
                # If we see 'on_open' and no errors, set status to UP
                if any(f"Connected to {exchange.capitalize()}" in line for line in new_lines):
                    self.exchange_status.labels(exchange=exchange).set(1)
                    self.last_connection_status.labels(exchange=exchange).set(1)
                    logger.info(f"Exchange {exchange} connection status set to UP")

    def _process_connector_logs(self, connector: str, connector_dir: str):
        """Process logs for a specific connector."""
        # Process balance logs
        balance_log = os.path.join(connector_dir, "balance.log")
        if os.path.exists(balance_log):
            new_lines = self._read_new_lines(balance_log)
            logger.debug(f"Read {len(new_lines)} new lines from {balance_log}")
            # Process balances if found in the logs
            if new_lines:
                self.exchange_status.labels(exchange=connector).set(1)

                # Parse balance data directly from the log file
                balance_pattern = r"Hyperliquid balances: ({.*})"
                currency_value_pattern = r"'([^']+)': ([\d.e-]+)"

                for line in new_lines:
                    # Look for the line with all balances as a dictionary
                    balance_match = re.search(balance_pattern, line)
                    if balance_match:
                        # Extract the dictionary content
                        balance_dict_str = balance_match.group(1)

                        # Find all currency-value pairs in the dictionary
                        for currency_match in re.finditer(currency_value_pattern, balance_dict_str):
                            currency = currency_match.group(1)
                            amount = float(currency_match.group(2))
                            logger.info(f"Setting balance metric for {connector}/{currency}: {amount}")
                            self.account_balance.labels(exchange=connector, currency=currency).set(amount)

        # Process orders logs
        orders_log = os.path.join(connector_dir, "orders.log")
        if os.path.exists(orders_log):
            self._process_orders_log(connector, orders_log)

        # Process markets logs
        markets_log = os.path.join(connector_dir, "markets.log")
        if os.path.exists(markets_log):
            new_lines = self._read_new_lines(markets_log)
            logger.debug(f"Read {len(new_lines)} new lines from {markets_log}")
            if new_lines:
                # Mark exchange as up if we have market data
                self.exchange_status.labels(exchange=connector).set(1)

    def _process_orders_log(self, connector: str, orders_log: str):
        """Process the orders log file for a connector."""
        new_lines = self._read_new_lines(orders_log)
        if not new_lines:
            return

        logger.debug(f"Processing {len(new_lines)} new lines from {orders_log}")

        # Patterns to match in orders.log
        order_placed_pattern = r"Placing order: ({.*})"
        order_response_pattern = r"Order placement response: ({.*})"
        order_filled_pattern = r"Order (\d+) filled immediately at ([\d.]+)"
        order_status_pattern = r"Found open order (\d+): ({.*})"

        for line in new_lines:
            try:
                # Match order placement
                order_placed_match = re.search(order_placed_pattern, line)
                if order_placed_match:
                    order_data_str = order_placed_match.group(1)
                    try:
                        order_data = json.loads(order_data_str)
                        symbol = order_data.get("coin", "unknown")
                        side = "buy" if order_data.get("is_buy", False) else "sell"
                        size = float(order_data.get("sz", 0))
                        price = float(order_data.get("limit_px", 0))
                        order_type = "limit" if "limit" in str(order_data.get("order_type", {})) else "market"

                        # Record order metrics
                        self.order_count.labels(
                            exchange=connector,
                            side=side,
                            type=order_type,
                            status="placed"
                        ).inc()

                        # Record order value (size * price)
                        if price > 0 and size > 0:
                            self.order_value.labels(
                                exchange=connector,
                                side=side
                            ).set(size * price)

                        logger.info(f"Recorded order placement: {connector}/{symbol} {side} {size} @ {price}")
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Error parsing order data: {e}, data: {order_data_str}")

                # Match order filled
                order_filled_match = re.search(order_filled_pattern, line)
                if order_filled_match:
                    order_id = order_filled_match.group(1)
                    fill_price = float(order_filled_match.group(2))

                    # Extract symbol from context if possible
                    symbol_match = re.search(r"([\w-]+)-USD", line)
                    symbol = symbol_match.group(1) if symbol_match else "unknown"

                    # Record fill price
                    self.last_order_price.labels(
                        exchange=connector,
                        symbol=symbol
                    ).set(fill_price)

                    # Determine side from context
                    side = "buy" if "BUY order" in line else "sell" if "SELL order" in line else "unknown"
                    if side == "unknown":
                        side = "buy" if "buy" in line.lower() else "sell" if "sell" in line.lower() else "unknown"

                    # Update order count for filled orders
                    self.order_count.labels(
                        exchange=connector,
                        side=side,
                        type="any",  # Type is unknown at this point
                        status="filled"
                    ).inc()

                    logger.info(f"Recorded order fill: {connector}/{symbol} order {order_id} filled at {fill_price}")

                # Match order status updates
                order_status_match = re.search(order_status_pattern, line)
                if order_status_match:
                    order_id = order_status_match.group(1)
                    order_details_str = order_status_match.group(2)
                    try:
                        order_details = json.loads(order_details_str)
                        symbol = order_details.get("symbol", "unknown")
                        side = order_details.get("side", "unknown").lower()
                        status = order_details.get("status", "unknown").lower()

                        # Record order status update
                        self.order_count.labels(
                            exchange=connector,
                            side=side,
                            type=order_details.get("type", "unknown").lower(),
                            status=status
                        ).inc()

                        logger.info(f"Recorded order status: {connector}/{symbol} order {order_id} is {status}")
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Error parsing order details: {e}, data: {order_details_str}")

            except Exception as e:
                logger.error(f"Error processing order log line: {e}, line: {line}")

    def _read_new_lines(self, file_path: str) -> List[str]:
        """Read new lines from a file since last check."""
        if file_path not in self.file_positions:
            self.file_positions[file_path] = 0

        last_position = self.file_positions[file_path]
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return []

            with open(file_path, 'r') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                self.file_positions[file_path] = f.tell()

            if new_lines:
                logger.debug(f"Read {len(new_lines)} new lines from {file_path}")
            return new_lines
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark Stacker Log Metrics Exporter")
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help=f"Directory containing Spark Stacker logs (default: {DEFAULT_LOG_DIR})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to expose metrics on (default: {DEFAULT_PORT})"
    )

    args = parser.parse_args()

    logger.info(f"Starting Spark Stacker log metrics exporter")
    logger.info(f"Log directory: {args.log_dir}")
    logger.info(f"Metrics port: {args.port}")

    metrics = SparkStackerLogMetrics(args.log_dir, args.port)
    metrics.start()
