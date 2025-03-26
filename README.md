# Spark Stacker

Spark Stacker is an advanced on-chain perpetual trading system with integrated hedging capabilities. It's designed to interact with multiple exchanges, implement technical indicators, and execute trading strategies with risk management.

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

## Architecture

Spark Stacker is built with a modular architecture:

- **Connectors**: Exchange-specific implementations with a common interface
- **Indicators**: Technical indicators for signal generation
- **Risk Management**: Handles position sizing and risk control
- **Trading Engine**: Core component that coordinates all operations
- **Webhook Server**: Receives external signals via HTTP
- **Monitoring System**: Grafana dashboards for performance tracking and analysis

## Setup

### Prerequisites

- Python 3.11 or higher
- Docker (optional, for containerized deployment)
- Node.js 18+ (for the monitoring and control interface)

### Configuration

1. Copy `config.json` to create your own configuration file
2. Fill in your exchange credentials and strategy parameters
3. Set `dry_run` to `true` for testing without executing trades

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

### Installation

#### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/spark-stacker.git
cd spark-stacker

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app/main.py
```

#### Docker Installation

```bash
# Build the Docker image
docker build -t spark-stacker .

# Run the container
docker run -p 8080:8080 --name spark-stacker -v $(pwd)/config.json:/app/config.json spark-stacker
```

## Monitoring & Control Interface

Spark Stacker includes a comprehensive monitoring and control system based on Grafana and Prometheus:

### Key Features

- **Real-time Dashboards**: Monitor system health, trading performance, and exchange connectivity
- **Position Management**: View, modify, and close positions through a web interface
- **Strategy Control**: Enable/disable strategies and adjust parameters in real-time
- **Performance Analysis**: Track historical performance with detailed metrics
- **Alerting**: Receive notifications for critical events (liquidation risks, errors, etc.)

### Monitoring Setup

```bash
# Start the monitoring stack
cd packages/monitoring
docker-compose up -d

# Access the Grafana interface
# Default credentials: admin/admin
open http://localhost:3000
```

### Available Dashboards

- **System Health**: Resource utilization, container status, and application health
- **Trading Performance**: Active positions, P&L metrics, and trade history
- **Exchange Integration**: API call rates, order execution metrics, and connection status
- **Risk Management**: Margin utilization, liquidation risks, and position sizing metrics

For more information, see the [Grafana Monitoring System](docs/grafana-monitoring-plan.md) documentation.

## Adding New Components

### Adding a New Indicator

1. Create a new file in `app/indicators/` directory
2. Implement your indicator by extending the `BaseIndicator` class
3. Register your indicator in the `IndicatorFactory`

### Adding a New Exchange Connector

1. Create a new file in `app/connectors/` directory
2. Implement the connector by extending the `BaseConnector` class
3. Register your connector in the `ConnectorFactory`

### Adding a New Dashboard

1. Create a new dashboard in Grafana
2. Export the dashboard to JSON
3. Save the JSON file in `packages/monitoring/dashboards/`

## Development Roadmap

See [ROADMAP.md](docs/roadmap.md) for the project development roadmap and progress tracking.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This software is provided for educational and informational purposes only. Trading cryptocurrencies involves substantial risk of loss and is not suitable for every investor. The past performance of any trading strategy is not necessarily indicative of future results. Only risk capital should be used for trading and only those with sufficient experience should trade.
