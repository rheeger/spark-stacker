# Spark Stacker

![Quick CI](https://github.com/yourusername/spark-stacker/actions/workflows/ci-quick.yml/badge.svg)

Spark Stacker is an advanced on-chain perpetual trading system with integrated hedging capabilities, organized as an NX monorepo. It's designed to interact with multiple exchanges, implement technical indicators, and execute trading strategies with risk management.

## Features

- **Multi-Exchange Support**: Supports Hyperliquid and Coinbase with planned expansion to Synthetix
- **Technical Indicators**: RSI implementation with framework for custom indicators
- **Risk Management**: Advanced position sizing, leverage control, and hedging
- **Trading Webhooks**: Support for TradingView alerts integration
- **Dry Run Mode**: Test strategies without deploying capital
- **Structured Logging**: Comprehensive logging for monitoring and debugging
- **Containerization**: Docker support for easy deployment
- **Monitoring**: Grafana dashboards for real-time system and trading performance monitoring
- **Control Interface**: Web-based UI for managing strategies and positions

## Project Structure

The project is organized as an NX monorepo with multiple packages:

```tree
spark-stacker/
├── packages/
│   ├── spark-app/         # Python trading application
│   │   ├── app/           # Main application code
│   │   ├── docker/        # Docker configuration files
│   │   ├── scripts/       # Utility scripts
│   │   └── tests/         # Test files
│   ├── monitoring/        # Monitoring and dashboard application
│   │   ├── apis/          # API endpoints
│   │   ├── dashboards/    # Dashboard components
│   │   ├── docker/        # Docker setup for monitoring stack
│   │   ├── exporters/     # Custom Prometheus exporters
│   │   └── frontend/      # Web UI
│   └── shared/            # Shared configuration and types
│       ├── docs/          # Project documentation
│       ├── examples/      # Example scripts and implementations
│       ├── .env           # Environment variables
│       ├── .env.example   # Example environment file
│       ├── config.json    # Application configuration
│       ├── .prettierrc    # Prettier configuration
│       └── .markdownlint.json # Markdown linting rules
├── node_modules/          # Node.js dependencies (managed by Yarn)
├── nx.json                # NX configuration
├── package.json           # Root package definition
├── tsconfig.json          # TypeScript configuration
└── yarn.lock              # Yarn lock file
```

## Architecture

Spark Stacker is built with a modular architecture:

- **Connectors**: Exchange-specific implementations with a common interface
- **Indicators**: Technical indicators for signal generation
- **Risk Management**: Handles position sizing and risk control
- **Trading Engine**: Core component that coordinates all operations
- **Webhook Server**: Receives external signals via HTTP
- **Monitoring System**: Grafana dashboards for performance tracking and analysis

## Prerequisites

- Python 3.11 or higher
- Node.js 18.x or higher
- Yarn 1.22.x or higher
- Docker and Docker Compose (for containerized deployment)

## Setup and Installation

### Configuration

All shared configuration files are located in the `packages/shared` directory:

1. Copy `.env.example` to `.env` and set your environment variables
2. Configure your trading settings in `config.json`
3. Set `dry_run` to `true` for testing without executing trades

Example configuration:

```json
{
  "log_level": "INFO",
  "webhook_enabled": false,
  "webhook_port": 8080,
  "webhook_host": "0.0.0.0",
  "max_parallel_trades": 1,
  "polling_interval": 60,
  "dry_run": true,
  "exchanges": [
    {
      "name": "hyperliquid",
      "wallet_address": "YOUR_WALLET_ADDRESS",
      "private_key": "YOUR_PRIVATE_KEY",
      "testnet": true,
      "use_as_main": true,
      "use_as_hedge": true
    }
  ],
  "strategies": [
    {
      "name": "rsi_eth_strategy",
      "market": "ETH",
      "enabled": true,
      "main_leverage": 5.0,
      "hedge_leverage": 2.0,
      "hedge_ratio": 0.2,
      "stop_loss_pct": 10.0,
      "take_profit_pct": 20.0,
      "max_position_size": 100.0
    }
  ],
  "indicators": [
    {
      "name": "eth_rsi",
      "type": "rsi",
      "enabled": true,
      "parameters": {
        "period": 14,
        "overbought": 70,
        "oversold": 30,
        "signal_period": 1
      }
    }
  ]
}
```

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/spark-stacker.git
cd spark-stacker

# Install Node dependencies
yarn install

# Set up Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r packages/spark-app/requirements.txt

# Run the trading application
cd packages/spark-app
python app/main.py
```

### Docker Deployment

Docker files for the trading application are located in `packages/spark-app/docker/`:

```bash
# From the root directory
docker-compose -f packages/spark-app/docker/docker-compose.yml up
```

## Development Workflow

### Working with NX Monorepo

```bash
# Build all packages
yarn build

# Run tests
yarn test

# Serve a specific application
yarn nx serve monitoring
```

### Running the Trading Application

```bash
# From the root directory
cd packages/spark-app
python app/main.py
```

### Starting the Monitoring Stack

```bash
# From the root directory
cd packages/monitoring
make start-monitoring

# Access Grafana at http://localhost:3000
# Default credentials: admin/admin
```

## Monitoring & Control Interface

Spark Stacker includes a comprehensive monitoring and control system based on Grafana and Prometheus:

### Key Features

- **Real-time Dashboards**: Monitor system health, trading performance, and exchange connectivity
- **Position Management**: View, modify, and close positions through a web interface
- **Strategy Control**: Enable/disable strategies and adjust parameters in real-time
- **Performance Analysis**: Track historical performance with detailed metrics
- **Alerting**: Receive notifications for critical events (liquidation risks, errors, etc.)

### Available Dashboards

- **System Health**: Resource utilization, container status, and application health
- **Trading Performance**: Active positions, P&L metrics, and trade history
- **Exchange Integration**: API call rates, order execution metrics, and connection status
- **Risk Management**: Margin utilization, liquidation risks, and position sizing metrics

## Metrics

The application exposes Prometheus metrics on port 8000. The following metrics are available:

- `spark_stacker_uptime_seconds`: Application uptime
- `spark_stacker_trades_total`: Trade count
- `spark_stacker_active_positions`: Current positions
- `spark_stacker_api_requests_total`: API calls
- And more...

## Extending the Platform

### Adding a New Indicator

1. Create a new file in `packages/spark-app/app/indicators/` directory
2. Implement your indicator by extending the `BaseIndicator` class
3. Register your indicator in the `IndicatorFactory`

### Adding a New Exchange Connector

1. Create a new file in `packages/spark-app/app/connectors/` directory
2. Implement the connector by extending the `BaseConnector` class
3. Register your connector in the `ConnectorFactory`

### Adding a New Dashboard

1. Create a new dashboard in Grafana
2. Export the dashboard to JSON
3. Save the JSON file in `packages/monitoring/dashboards/`

## Documentation

For detailed documentation on each package, see the README.md file in each package directory:

- [Trading Application Documentation](./packages/spark-app/README.md)
- [Monitoring System Documentation](./packages/monitoring/README.md)

## Development Roadmap

See [ROADMAP.md](packages/shared/docs/roadmap.md) for the project development roadmap and progress tracking.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This software is provided for educational and informational purposes only. Trading cryptocurrencies involves substantial risk of loss and is not suitable for every investor. The past performance of any trading strategy is not necessarily indicative of future results. Only risk capital should be used for trading and only those with sufficient experience should trade.
