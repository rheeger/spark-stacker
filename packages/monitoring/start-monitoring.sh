#!/bin/bash
# Start the Spark Stacker monitoring stack with the new dashboard and log metrics exporter

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Spark Stacker monitoring stack..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running or not installed."
    exit 1
fi

# Ensure we're in the right directory
cd docker

# Build and start the containers
docker-compose --project-name spark-stacker down
docker-compose --project-name spark-stacker up -d --build

echo ""
echo "Monitoring stack is now running!"
echo ""
echo "Access Grafana dashboard at: http://localhost:3000"
echo "Default credentials: admin/admin"
echo ""
echo "Prometheus metrics available at: http://localhost:9090"
echo "Log metrics exporter available at: http://localhost:9001/metrics"
echo ""
echo "To stop the monitoring stack, run: docker-compose --project-name spark-stacker down"
