# Spark Stacker Monitoring

This package contains the monitoring and control interface for the Spark Stacker trading
application.

## Components

- **Prometheus & Grafana**: For metric collection, visualization, and alerting
- **Loki & Promtail**: For log aggregation and analysis
- **React Frontend**: For the control interface and dashboard views
- **Backend APIs**: For interacting with the trading application

## Getting Started

### Starting the Monitoring Stack

```bash
# Start the monitoring stack
make start-monitoring

# Access Grafana at http://localhost:3000
# Default credentials: admin/admin

# View Prometheus at http://localhost:9090
```

### Development

```bash
# Install dependencies
make install

# Build the frontend
make build

# Start the frontend in development mode
make start
```

### Stopping the Monitoring Stack

```bash
# Stop the monitoring stack
make stop-monitoring

# Remove all data volumes
make clean-monitoring
```

## Directory Structure

- `docker/`: Docker configuration for the monitoring stack
- `frontend/`: React-based control interface
- `apis/`: Backend APIs for the control interface
- `dashboards/`: Grafana dashboard definitions
- `exporters/`: Custom metric exporters

## Metrics

Prometheus scrapes metrics from the trading application on `host.docker.internal:8000/metrics`. The
following metrics are collected:

- `spark_stacker_uptime_seconds`: Application uptime
- `spark_stacker_trades_total`: Trade count
- `spark_stacker_active_positions`: Current positions
- `spark_stacker_api_requests_total`: API calls
- `spark_stacker_api_latency_seconds`: API latency
- `spark_stacker_margin_ratio`: Position margin ratios
- `spark_stacker_pnl_percent`: PnL metrics
