# Spark Stacker Logging System

## Overview

Spark Stacker uses a unified logging system that captures detailed information about application behavior, connector activities, and system events. This document explains the logging architecture and provides guidelines for accessing and interpreting logs.

## Log Directory Structure

All logs are now stored centrally in the `packages/spark-app/_logs` directory. Each application run creates a timestamped and uniquely identified subdirectory with the following structure:

```
packages/spark-app/_logs/YYYY-MM-DD-HH-MM-SS_XXXXXXXX/
├── spark_stacker.log         # Main application log
├── coinbase/                 # Connector specific directory
│   ├── balance.log           # Balance operations log
│   ├── markets.log           # Market data operations log
│   └── orders.log            # Order operations log
├── hyperliquid/              # Another connector
└── ...                       # Other connectors
```

## Log Types

### Main Application Log (`spark_stacker.log`)

Contains general application information including:

- Startup and shutdown events
- Configuration information
- Exchange connections
- Trading engine activity
- Error and warning messages

### Connector Logs

Each connector (exchange) has its own set of logs:

- **Balance Log (`balance.log`)**: Records account balance data and updates
- **Markets Log (`markets.log`)**: Contains market data, tickers, and price information
- **Orders Log (`orders.log`)**: Records order submissions, updates, and cancellations

## Log Format

By default, logs use the following format:

```
YYYY-MM-DD HH:MM:SS,ms - logger.name - LEVEL - Message
```

When structured logging is enabled, logs are output in JSON format for easier machine parsing.

## Accessing Logs

### Via Local Filesystem

Logs can be accessed directly in the `packages/spark-app/_logs` directory.

### Via Docker

If running in Docker, logs are mapped to the container at `/app/_logs` and can be accessed using Docker commands:

```bash
# View the main log file from the most recent run
docker exec spark-stacker cat $(ls -t /app/_logs | head -1)/spark_stacker.log

# Tail the logs in real-time
docker exec spark-stacker tail -f $(ls -t /app/_logs | head -1)/spark_stacker.log
```

## Monitoring Integration

The logging system is integrated with the monitoring stack:

1. **Loki/Promtail**: All logs are ingested into Loki for centralized viewing in Grafana
2. **Log Metrics Exporter**: Extracts metrics from logs (connection status, errors, etc.)

## Log Cleanup

To clean up all log files:

```bash
# From packages/spark-app directory
make clean-logs
```

This removes all existing logs and ensures a clean start for the new logging system.

## Compatibility Notes

For backward compatibility with monitoring tools, symbolic links are automatically created in the root `/logs` directory pointing to the corresponding log directories in `packages/spark-app/_logs`.

## Configuring Logging

Logging can be configured in the application configuration file:

```json
{
  "logging": {
    "log_level": "INFO",
    "connector_log_level": "DEBUG",
    "log_to_file": true,
    "enable_console": true,
    "structured": true,
    "show_market_details": false,
    "show_zero_balances": false
  }
}
```

### Configuration Options

- **log_level**: Overall logging level (DEBUG, INFO, WARNING, ERROR)
- **connector_log_level**: Logging level for connector operations
- **log_to_file**: Whether to save logs to disk
- **enable_console**: Whether to output logs to console
- **structured**: Use structured (JSON) logging format
- **show_market_details**: Whether to log detailed market data
- **show_zero_balances**: Whether to log zero-value balances
