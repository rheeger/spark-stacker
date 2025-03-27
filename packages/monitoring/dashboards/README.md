# Spark Stacker Grafana Dashboards

This directory contains the Grafana dashboard JSON files for the Spark Stacker monitoring system.

## Available Dashboards

- `spark-stacker-overview.json`: The main overview dashboard showing container status and key metrics
- `home.json`: Home dashboard that redirects to the main overview

## How to Use

1. Start the monitoring stack:
   ```bash
   make monitoring-start
   ```

2. Access Grafana at http://localhost:3000
   - Default credentials: admin/admin
   - The main overview dashboard should load automatically

## Updating Dashboards

### Method 1: Edit in Grafana UI and Export

1. Make changes to dashboards using Grafana UI
2. Click the save icon and select "Export" → "Save to file"
3. Save the JSON file to this directory
4. Update dashboards in the running container:
   ```bash
   make update-dashboards
   ```

### Method 2: Edit JSON Files Directly

1. Edit the dashboard JSON files in this directory
2. Update dashboards in the running container:
   ```bash
   make update-dashboards
   ```

## Dashboard Structure

- **Main Overview**: High-level view of the system with container status and key metrics
  - Container Status: Up/Down status of all containers
  - CPU & Memory Usage: Per-container resource utilization
  - Trading Activity: Trade volume by exchange
  - Margin Ratio: Visual representation of current margin ratios
  - PnL Percentage: Gauge showing profit/loss percentages
  - API Latency: Response times for API endpoints
  - Recent Logs: Latest application logs

## Adding New Dashboards

1. Create a new JSON file in this directory
2. Follow Grafana dashboard JSON format
3. Use a unique `uid` in the JSON to avoid conflicts
4. Add appropriate tags for categorization
5. Run `make update-dashboards` to deploy the dashboard
