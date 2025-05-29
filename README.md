# Spark Stacker

![Quick CI](https://github.com/rheeger/spark-stacker/actions/workflows/ci-quick.yml/badge.svg)

Spark Stacker is an advanced on-chain perpetual trading system with integrated hedging capabilities, organized as an NX monorepo. It's designed to interact with multiple exchanges, implement technical indicators, and execute trading strategies with risk management.

## Features

- **Multi-Exchange Support**: Supports Hyperliquid and Coinbase with planned expansion to Synthetix
- **Strategy-Driven Architecture**: Clear separation between strategies (WHAT to trade) and indicators (HOW to analyze)
- **Multi-Timeframe Support**: Different indicators can operate on different timeframes within the same strategy
- **Technical Indicators**: RSI, MACD implementations with framework for custom indicators
- **Strategy-Specific Position Sizing**: Each strategy can have its own position sizing method and parameters
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

Spark Stacker is built with a modular, strategy-driven architecture:

### Core Components

- **Strategies**: Define WHAT to trade (market, exchange, indicators to use, risk parameters)
- **Indicators**: Define HOW to analyze data (algorithm type, timeframe, parameters)
- **Connectors**: Exchange-specific implementations with a common interface
- **Risk Management**: Handles position sizing and risk control with strategy-specific configurations
- **Trading Engine**: Core component that coordinates all operations
- **Webhook Server**: Receives external signals via HTTP
- **Monitoring System**: Grafana dashboards for performance tracking and analysis

### Strategy-Indicator Relationship

The system follows a clear separation of concerns:

1. **Strategies** specify:

   - Which market to trade ("ETH-USD", "BTC-USD")
   - Which exchange to use ("hyperliquid", "coinbase")
   - Which indicators to consult for signals
   - Risk management parameters
   - Position sizing method (optional, inherits from global if not specified)

2. **Indicators** specify:

   - The algorithm to use ("rsi", "macd")
   - The timeframe for data analysis ("1h", "4h", "1d")
   - Algorithm-specific parameters (period, thresholds, etc.)

3. **Signal Flow**:

   ```
   Strategy → Indicators → Signals → Trading Engine → Exchange
   ```

This separation allows:

- Multiple strategies to share the same indicators
- Strategies to use different timeframes for different indicators
- Easy addition of new strategies without modifying indicators
- Strategy-specific position sizing and risk management

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

#### Basic Configuration Example

```json
{
  "log_level": "INFO",
  "webhook_enabled": false,
  "dry_run": true,
  "polling_interval": 60,
  "max_parallel_trades": 1,

  "position_sizing": {
    "method": "fixed_usd",
    "fixed_usd_amount": 100.0,
    "equity_percentage": 0.05,
    "risk_per_trade_pct": 0.02,
    "max_position_size_usd": 1000.0
  },

  "exchanges": [
    {
      "name": "hyperliquid",
      "exchange_type": "hyperliquid",
      "wallet_address": "${WALLET_ADDRESS}",
      "private_key": "${PRIVATE_KEY}",
      "testnet": true,
      "enabled": true,
      "use_as_main": true
    }
  ],

  "strategies": [
    {
      "name": "eth_momentum_strategy",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "enabled": true,
      "timeframe": "4h",
      "indicators": ["rsi_4h", "macd_1h"],
      "main_leverage": 5.0,
      "stop_loss_pct": 10.0,
      "take_profit_pct": 20.0,
      "max_position_size": 100.0
    }
  ],

  "indicators": [
    {
      "name": "rsi_4h",
      "type": "rsi",
      "enabled": true,
      "timeframe": "4h",
      "parameters": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
      }
    },
    {
      "name": "macd_1h",
      "type": "macd",
      "enabled": true,
      "timeframe": "1h",
      "parameters": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
      }
    }
  ]
}
```

#### Strategy-Specific Position Sizing

Each strategy can override the global position sizing configuration:

```json
{
  "position_sizing": {
    "method": "fixed_usd",
    "fixed_usd_amount": 100.0
  },

  "strategies": [
    {
      "name": "conservative_strategy",
      "market": "BTC-USD",
      "exchange": "hyperliquid",
      "indicators": ["rsi_4h"]
      // Uses global position sizing (fixed_usd, $100)
    },
    {
      "name": "aggressive_strategy",
      "market": "ETH-USD",
      "exchange": "hyperliquid",
      "indicators": ["macd_1h"],
      // Strategy-specific position sizing
      "position_sizing": {
        "method": "risk_based",
        "risk_per_trade_pct": 0.05,
        "default_stop_loss_pct": 0.03,
        "max_position_size_usd": 2000.0
      }
    },
    {
      "name": "percent_equity_strategy",
      "market": "SOL-USD",
      "exchange": "hyperliquid",
      "indicators": ["rsi_1h"],
      // Another position sizing method
      "position_sizing": {
        "method": "percent_equity",
        "equity_percentage": 0.1,
        "max_position_size_usd": 5000.0
      }
    }
  ]
}
```

### Key Configuration Principles

#### Strategy-Indicator Relationship

- **Strategies define WHAT to trade**: Market symbol, exchange, which indicators to use, risk parameters
- **Indicators define HOW to analyze**: Algorithm type, timeframe, and parameters
- **Use full market symbols**: "ETH-USD", "BTC-USD", not just "ETH"
- **Connect strategies to indicators**: Use the `indicators` array in strategy config
- **Timeframe independence**: Strategies can use indicators with different timeframes

#### Position Sizing Configuration

- **Global defaults**: Set in the root `position_sizing` object
- **Strategy overrides**: Add `position_sizing` object to individual strategies
- **Inheritance**: Strategies inherit global settings and can override specific parameters
- **Validation**: Each strategy's position sizing configuration is validated at startup

#### Symbol Conversion

- **Standard format**: Always use "SYMBOL-USD" format in configuration ("ETH-USD", "BTC-USD")
- **Exchange conversion**: System automatically converts to exchange-specific format:
  - Hyperliquid: "ETH-USD" → "ETH"
  - Coinbase: "ETH-USD" → "ETH-USD" (unchanged)

### Position Sizing Methods

| Method           | Description                           | Key Parameters                                      |
| ---------------- | ------------------------------------- | --------------------------------------------------- |
| `fixed_usd`      | Fixed dollar amount per trade         | `fixed_usd_amount`                                  |
| `percent_equity` | Percentage of account equity          | `equity_percentage`                                 |
| `risk_based`     | Based on stop loss and risk tolerance | `risk_per_trade_pct`, `default_stop_loss_pct`       |
| `kelly`          | Kelly criterion optimization          | `kelly_win_rate`, `kelly_avg_win`, `kelly_avg_loss` |

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/spark-stacker.git
cd spark-stacker

# Install Node dependencies
yarn install

# Set up Python virtual environment
cd packages/spark-app
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the trading application
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

## Troubleshooting

### Common Configuration Errors

#### "Market [INDICATOR-NAME] not found" Error

**Problem**: Indicator names being treated as market symbols

```
ERROR: Market RSI-4H not found
```

**Solution**: Ensure proper strategy-indicator configuration:

```json
// ❌ Incorrect - this will cause the error
"strategies": [
  {
    "market": "RSI-4H",  // Wrong! This is an indicator name
    "indicators": ["rsi_4h"]
  }
]

// ✅ Correct configuration
"strategies": [
  {
    "market": "ETH-USD",  // Market symbol
    "exchange": "hyperliquid",
    "indicators": ["rsi_4h"]  // Indicator references
  }
]
```

#### Invalid Market Symbol Format

**Problem**: Using single symbols instead of pairs

```json
// ❌ Incorrect
"market": "ETH"

// ✅ Correct
"market": "ETH-USD"
```

#### Missing Exchange Field

**Problem**: Strategy doesn't specify which exchange to use

```json
// ❌ Missing exchange
"strategies": [
  {
    "name": "my_strategy",
    "market": "ETH-USD",
    "indicators": ["rsi_4h"]
  }
]

// ✅ Include exchange
"strategies": [
  {
    "name": "my_strategy",
    "market": "ETH-USD",
    "exchange": "hyperliquid",
    "indicators": ["rsi_4h"]
  }
]
```

#### Indicator Not Found

**Problem**: Strategy references indicator that doesn't exist

```json
// Strategy references "rsi_daily" but it's not defined
"strategies": [
  {
    "indicators": ["rsi_daily"]  // ❌ Not defined below
  }
],
"indicators": [
  {
    "name": "rsi_4h",  // ✅ Different name
    "type": "rsi"
  }
]
```

#### Invalid Position Sizing Configuration

**Problem**: Strategy-specific position sizing with invalid parameters

```json
// ❌ Invalid configuration
"strategies": [
  {
    "position_sizing": {
      "method": "risk_based",
      // Missing required parameters for risk_based method
    }
  }
]

// ✅ Complete configuration
"strategies": [
  {
    "position_sizing": {
      "method": "risk_based",
      "risk_per_trade_pct": 0.02,
      "default_stop_loss_pct": 0.05,
      "max_position_size_usd": 1000.0
    }
  }
]
```

### Testing Configuration

Validate your configuration without trading:

```bash
cd packages/spark-app

# Test configuration loading
.venv/bin/python -c "
import json
from app.core.strategy_config import StrategyConfigLoader

with open('../shared/config.json') as f:
    config = json.load(f)

strategies = StrategyConfigLoader.load_strategies(config['strategies'])
print('✅ Configuration loaded successfully')
print(f'Loaded {len(strategies)} strategies')
"

# Test strategy-indicator relationships
.venv/bin/python -c "
from app.main import _validate_strategy_indicators
import json

with open('../shared/config.json') as f:
    config = json.load(f)

_validate_strategy_indicators(config['strategies'], config['indicators'])
print('✅ Strategy-indicator validation passed')
"

# Run in dry-run mode
python app/main.py --dry-run
```

### Debugging Tips

1. **Enable debug logging**: Set `log_level: "DEBUG"` in config.json
2. **Check symbol conversion**: Verify exchange-specific symbol formats
3. **Validate timeframes**: Ensure indicator timeframes are supported
4. **Test indicators individually**: Use indicator testing scripts
5. **Check exchange connectivity**: Verify API credentials and network access
