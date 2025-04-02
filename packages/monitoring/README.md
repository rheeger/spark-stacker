# Spark Stacker Monitoring System

This directory contains the monitoring stack for the Spark Stacker trading application, including:

- Grafana dashboards for visualizing performance metrics
- Prometheus for metrics collection
- Loki for log aggregation
- Log metrics exporter for extracting metrics from log files

## Key Features

- Real-time connection status monitoring for all exchanges (Hyperliquid, Coinbase)
- Connection error tracking with categorization and visualization
- Exchange account balance monitoring
- Websocket ping/pong latency metrics
- Overall system status and configuration display
- Advanced visualization of market metrics

## Dashboard Components

The main dashboard provides:

1. **Connection Status Panel** - Shows the current status of all exchange connections
2. **Service Status Panel** - Displays webhook and metrics server status
3. **Websocket Health Panel** - Monitors websocket ping/pong latency
4. **Error Tracking Panel** - Displays connection errors by type and exchange
5. **Account Balances** - Shows current account balances by exchange and currency
6. **System Information** - Provides build info, configuration, and market counts

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Access to the Spark Stacker log files

### Running the Monitoring Stack

From the `packages/monitoring` directory:

```bash
cd docker
docker-compose up -d
```

Access the dashboard at http://localhost:3000 (default credentials: admin/admin)

## Log Metrics Exporter

The custom log metrics exporter (`packages/monitoring/exporters/log-metrics.py`) parses the Spark Stacker log files and exposes metrics for Prometheus to scrape. Key metrics include:

- `spark_stacker_exchange_connection_status` - Connection status (1=up, 0=down)
- `spark_stacker_connection_errors_total` - Count of connection errors by type
- `spark_stacker_websocket_ping_latency_ms` - Websocket ping/pong latency
- `spark_stacker_account_balance` - Account balances by currency
- `spark_stacker_market_count` - Available markets on each exchange

## Troubleshooting

If metrics aren't appearing in the dashboard:

1. Check that the log metrics exporter is running: `docker-compose ps log-metrics`
2. Verify that Prometheus can scrape metrics: `curl http://localhost:9001/metrics`
3. Ensure the log files directory is properly mounted in the Docker container

For log data issues, check the Loki logs using the Explore tab in Grafana.
