# Phase 4: Monitoring & Control Interface (🟡 IN PROGRESS)

## Dependencies

- Phase 3: Integration & Dry Run (🟡 In Progress)
  - Can start basic monitoring during Phase 3
  - Full implementation requires Phase 3 completion

## Parallel Work Opportunities

- ✅ Basic monitoring setup can begin during Phase 3
- 🔲 Core control interface development can start in parallel
- 🔲 Advanced features can be developed incrementally

## Goals & Objectives

- 🟡 Provide real-time visibility into trading system performance metrics
- ✅ Enable monitoring of Docker container health and resource utilization
- 🟡 Create dashboards for strategy performance analysis and comparison
- 🔲 Implement alerting for critical system events and trade outcomes
- 🔲 Develop a control interface for managing trading strategies and positions
- 🔲 Ensure secure access to monitoring and control capabilities

## NX Monorepo Setup (CRITICAL PATH)

- ✅ Install NX CLI tools
  - ✅ Set up global NX installation
  - ✅ Create project initialization scripts
- ✅ Configure NX workspace structure

  - ✅ Set up directory layout as follows:

    ```tree
    spark-stacker/
    ├── packages/
    │   ├── spark-app/            # Python trading application
    │   │   ├── app/              # Main application code
    │   │   ├── docker/           # Docker configuration files
    │   │   ├── scripts/          # Utility scripts
    │   │   └── tests/            # Test files
    │   ├── monitoring/           # Monitoring and dashboard application
    │   │   ├── apis/             # API endpoints
    │   │   ├── dashboards/       # Dashboard components
    │   │   ├── docker/           # Docker setup for monitoring stack
    │   │   ├── exporters/        # Custom Prometheus exporters
    │   │   └── frontend/         # Web UI
    │   └── shared/               # Shared configuration and types
    │       ├── docs/             # Project documentation
    │       ├── examples/         # Example scripts and implementations
    │       ├── .env              # Environment variables
    │       ├── .env.example      # Example environment file
    │       ├── config.json       # Application configuration
    │       ├── .prettierrc       # Prettier configuration
    │       └── .markdownlint.json # Markdown linting rules
    ```

  - ✅ Define project boundaries
  - ✅ Configure dependency graph

- ✅ Create package.json and nx.json configuration
  - ✅ Define workspace defaults
  - ✅ Configure task runners
  - ✅ Set up caching options
- ✅ Set up TypeScript configuration
  - ✅ Configure compiler options
  - ✅ Set up project references
  - ✅ Define type definitions
- ✅ Create directory structure for monitoring packages
  - 🔲 Set up frontend application
  - 🔲 Configure backend API services
  - ✅ Create shared library packages

## Testing Implementation

### Unit Testing Requirements

- 🔲 NX configuration and setup tests
  - 🔲 Test workspace configuration validity
  - 🔲 Verify project references and dependencies
  - 🔲 Validate build pipeline configuration
- 🔲 Frontend component tests
  - 🔲 Test React component rendering
  - 🔲 Verify state management
  - 🔲 Validate user interaction handling
  - 🔲 Test responsive design behavior
- 🔲 Backend API tests
  - 🔲 Test API endpoint functionality
  - 🔲 Verify request validation
  - 🔲 Test error handling
  - 🔲 Validate authentication and authorization
- 🔲 Metric collection tests
  - 🔲 Test Prometheus client integration
  - 🔲 Verify metric registration
  - 🔲 Validate metric value accuracy
  - 🔲 Test counter, gauge, and histogram behavior
- 🔲 Dashboard configuration tests
  - 🔲 Validate Grafana dashboard JSON
  - 🔲 Test dashboard provisioning
  - 🔲 Verify data source configuration

### Integration Testing Requirements

- 🔲 Frontend-backend integration tests
  - 🔲 Test API data fetching from frontend
  - 🔲 Verify form submission and data processing
  - 🔲 Validate real-time updates via WebSockets
- 🔲 Monitoring stack integration tests
  - 🔲 Test Prometheus-Grafana integration
  - 🔲 Verify log collection and visualization
  - 🔲 Test alert configuration and triggering
- 🔲 Python app-monitoring integration tests
  - 🔲 Validate metric export from trading application
  - 🔲 Test log collection from application containers
  - 🔲 Verify end-to-end metric visualization

### End-to-End Testing Requirements

- 🔲 User interface workflow tests
  - 🔲 Test strategy configuration workflow
  - 🔲 Verify position management process
  - 🔲 Validate dashboard navigation and filtering
- 🔲 Alert notification tests
  - 🔲 Test email alert delivery
  - 🔲 Verify Slack/Discord notifications
  - 🔲 Validate escalation policies
- 🔲 System control tests
  - 🔲 Test strategy activation/deactivation
  - 🔲 Verify system status monitoring
  - 🔲 Validate configuration changes

### Performance Testing Requirements

- 🔲 Frontend performance tests
  - 🔲 Measure component rendering time
  - 🔲 Test large dataset handling
  - 🔲 Verify UI responsiveness under load
- 🔲 Backend API performance tests
  - 🔲 Measure request handling latency
  - 🔲 Test concurrent request handling
  - 🔲 Verify database query performance
- 🔲 Monitoring system performance tests
  - 🔲 Measure metric collection overhead
  - 🔲 Test high-cardinality metric handling
  - 🔲 Verify dashboard rendering performance

## Prometheus & Grafana Setup (CRITICAL PATH)

- ✅ Create docker-compose.yml for monitoring stack
  - ✅ Configure Prometheus service
  - ✅ Set up Grafana service
  - ✅ Add Loki for log aggregation
  - ✅ Configure Node Exporter for host metrics
  - ✅ Add cAdvisor for container metrics
- 🟡 Configure Prometheus data collection
  - ✅ Set up scrape configs for services
  - 🔲 Configure retention policies
  - 🔲 Set up recording rules for common queries
  - 🔲 Configure alert rules
- ✅ Set up Grafana dashboards
  - ✅ Configure data sources (Prometheus, Loki)
  - ✅ Create consolidated home dashboard
  - ✅ Configure dashboard provisioning
- ✅ Configure Loki for log aggregation
  - ✅ Set up Promtail for log collection
  - 🔲 Configure log parsing rules
  - 🔲 Set up log retention policies
- 🔲 Set up alerting rules and notification channels
  - 🔲 Configure email notifications
  - 🔲 Set up Slack/Discord integration
  - 🔲 Create PagerDuty/OpsGenie integration
  - 🔲 Define escalation policies

## Metrics Collection (CRITICAL PATH)

- 🔲 Integrate Prometheus client in Python app
  - 🔲 Add client library to requirements
  - 🔲 Create metrics registry
  - 🔲 Implement metrics endpoint
- 🔲 Define core application metrics
  - 🔲 System uptime and health metrics
    - 🔲 `spark_stacker_uptime_seconds`: Application uptime
    - 🔲 `spark_stacker_trades_total{result="success|failure", exchange="name", side="buy|sell"}`:
      Trade count
    - 🔲 `spark_stacker_active_positions{exchange="name", market="symbol", side="long|short"}`:
      Current positions
    - 🔲 `spark_stacker_signal_count{indicator="name", signal="buy|sell|neutral"}`: Signal
      generation
  - 🔲 Trading operation counters
  - 🔲 Signal generation metrics
  - 🔲 Performance timing metrics
- 🔲 Add metrics for exchange connectors
  - 🔲 API call counts and latencies
    - 🔲 `spark_stacker_api_requests_total{exchange="name", endpoint="path", method="GET|POST"}`:
      API calls
    - 🔲 `spark_stacker_api_latency_seconds{exchange="name", endpoint="path"}`: API latency
    - 🔲 `spark_stacker_order_execution_seconds{exchange="name", order_type="market|limit"}`: Order
      execution time
    - 🔲 `spark_stacker_rate_limit_remaining{exchange="name", endpoint="path"}`: API rate limits
  - 🔲 Order execution metrics
  - 🔲 Error rates by exchange
  - 🔲 Position tracking metrics
- 🔲 Create risk management metrics
  - 🔲 Liquidation risk indicators
    - 🔲 `spark_stacker_margin_ratio{exchange="name", position_id="id"}`: Position margin ratios
    - 🔲 `spark_stacker_liquidation_price{exchange="name", position_id="id"}`: Liquidation prices
    - 🔲 `spark_stacker_capital_utilization_percent`: Percentage of capital in use
    - 🔲 `spark_stacker_max_drawdown_percent{timeframe="1h|1d|1w"}`: Maximum drawdown by timeframe
  - 🔲 Margin utilization metrics
  - 🔲 Stop-loss/take-profit triggers
  - 🔲 Portfolio exposure metrics
- 🔲 Set up trading performance metrics
  - 🔲 P&L tracking by strategy
    - 🔲 `spark_stacker_pnl_percent{strategy="name", position_type="main|hedge"}`: PnL metrics
  - 🔲 Win/loss ratios
  - 🔲 Drawdown measurements
  - 🔲 Strategy correlation metrics

## Log Collection Implementation (CRITICAL PATH)

- 🔲 Define structured log format (JSON)

  ```json
  {
    "timestamp": "2023-03-15T12:34:56.789Z",
    "level": "INFO",
    "category": "trading",
    "message": "Trade executed",
    "data": {
      "exchange": "hyperliquid",
      "market": "ETH-USD",
      "side": "BUY",
      "size": 1.0,
      "price": 3000.0,
      "leverage": 5.0
    },
    "trace_id": "abc123"
  }
  ```

- 🔲 Implement log categories:
  - 🔲 `app`: General application logs
  - 🔲 `trading`: Trading-specific logs
  - 🔲 `connector`: Exchange connector logs
  - 🔲 `risk`: Risk management logs
  - 🔲 `security`: Authentication and security logs
- 🔲 Configure log levels with appropriate usage:
  - 🔲 `DEBUG`: Detailed debugging information
  - 🔲 `INFO`: General operational information
  - 🔲 `WARNING`: Warnings and non-critical issues
  - 🔲 `ERROR`: Error conditions requiring attention
  - 🔲 `CRITICAL`: Critical conditions requiring immediate action

## Dashboard Development (CRITICAL PATH)

- ✅ Create system health dashboard
  - ✅ CPU, memory, disk, and network panels
  - ✅ Application uptime and restart tracking
  - ✅ Error rate visualization
  - ✅ Log volume analysis
- 🟡 Build trading performance dashboard
  - ✅ Active positions display (placeholder)
  - ✅ P&L charts by strategy and time (placeholder)
  - 🔲 Trade history visualization
  - 🔲 Strategy comparison panels
  - 🔲 Win/loss ratios and average trade metrics
  - 🔲 Comparative strategy performance
- 🟡 Develop exchange integration dashboard
  - ✅ API call rates and latency (placeholder)
  - 🔲 Rate limit utilization
  - 🔲 Error tracking by endpoint
  - 🔲 Order execution success rates
  - 🔲 Connectivity status by exchange
- 🟡 Create risk management dashboard
  - ✅ Margin health visualization (placeholder)
  - 🔲 Liquidation risk indicators
  - 🔲 Position sizing analysis
  - 🔲 Hedge effectiveness metrics
  - 🔲 Stop-loss trigger frequency
- 🔲 Build control panel dashboard
  - 🔲 Strategy activation controls
  - 🔲 Position management interface
  - 🔲 Parameter adjustment panels
  - 🔲 System control widgets
  - 🔲 Strategy parameter adjustment interface
  - 🔲 Position entry/exit controls
  - 🔲 Backtest execution and analysis

## Control Interface (CRITICAL PATH)

- 🔲 Design RESTful API specification
  - 🔲 Define endpoint structure
  - 🔲 Document request/response formats
  - 🔲 Create OpenAPI/Swagger spec
  - 🔲 Define security requirements
  - 🔲 Design endpoints for:
    - 🔲 Strategy activation/deactivation
    - 🔲 Position management (view, modify, close)
    - 🔲 System status and configuration
    - 🔲 Backtest execution and results retrieval
- 🔲 Implement backend API endpoints
  - 🔲 Strategy management endpoints
  - 🔲 Position control endpoints
  - 🔲 System configuration endpoints
  - 🔲 Authentication endpoints
- 🔲 Create React-based frontend
  - 🔲 Set up component library
  - 🔲 Implement layout and navigation
  - 🔲 Create strategy management views
  - 🔲 Build position management interface
  - 🔲 Implement features for:
    - 🔲 View and edit strategy parameters
    - 🔲 Enable/disable strategies
    - 🔲 Schedule strategy execution
    - 🔲 Compare strategy performance
    - 🔲 View positions with real-time updates
    - 🔲 Manually close or modify positions
    - 🔲 Implement stop-loss/take-profit adjustments
    - 🔲 View position history and performance
- 🔲 Add authentication and authorization
  - 🔲 Implement JWT authentication
  - 🔲 Create role-based access control (admin, trader, viewer)
  - 🔲 Add secure session management
  - 🔲 Implement audit logging
- 🔲 Implement WebSocket for real-time updates
  - 🔲 Create WebSocket server
  - 🔲 Implement client-side connection
  - 🔲 Add real-time updates for positions
  - 🔲 Create notification system

## Testing & Optimization

- 🟡 Conduct load testing on monitoring infrastructure
  - ✅ Test metric collection impact on trading system
  - 🔲 Benchmark monitoring stack resource usage
  - 🔲 Optimize metric storage and retention
  - 🔲 Test system under high load conditions
- 🔲 Security testing
  - 🔲 Conduct authentication vulnerability assessment
  - 🔲 Test API endpoint security
  - 🔲 Review network security configuration
  - 🔲 Audit Docker container security
- 🔲 Implement monitoring for the monitoring system
  - 🔲 Set up meta-monitoring for Prometheus
  - 🔲 Configure Grafana availability monitoring
  - 🔲 Implement log monitoring for the monitoring stack
  - 🔲 Create alerts for monitoring system failures

## Documentation & Deployment

- 🔲 Create user documentation
  - 🔲 Dashboard usage guide
  - 🔲 Alert interpretation documentation
  - 🔲 Control interface user manual
  - 🔲 API documentation
- 🔲 Document deployment procedures
  - 🔲 Monitoring stack deployment guide
  - 🔲 Configuration management
  - 🔲 Backup and recovery procedures
  - 🔲 Scaling guidelines

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03-15T12:35:00.123Z",
      "level": "INFO",
      "category": "trading",
      "strategy": "macd_eth_usd",
      "message": "Position opened",
      "data": {
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "side": "BUY",
        "size": 0.0003,
        "usd_value": 1.0,
        "price": 3333.33,
        "leverage": 10.0
      }
    }
    ```

- 🔲 Create log parsing rules for Loki
  - 🔲 Extract structured data for querying
  - 🔲 Create derived fields for quick linking

### Control Interface Components

- 🔲 Implement strategy control panel

  - 🔲 MACD parameter adjustment interface
  - 🔲 Strategy enable/disable toggle
  - 🔲 Position size control
  - 🔲 Manual position close button
  - 🔲 Stop-loss/take-profit adjustment sliders

- 🔲 Add backtest comparison feature
  - 🔲 Run backtest with current parameters
  - 🔲 Compare live performance to backtest expectations
  - 🔲 Parameter optimization suggestions

## Next Steps (Prioritized)

1. Set up NX monorepo structure (CRITICAL PATH)

   - Initialize NX workspace
   - Configure TypeScript
   - Set up package structure

2. Implement basic monitoring (CRITICAL PATH)

   - Set up Prometheus and Grafana
   - Configure basic metrics collection
   - Create essential dashboards

3. Develop core control interface (CRITICAL PATH)

   - Create basic API endpoints
   - Implement essential frontend features
   - Add authentication

4. Add advanced features incrementally

   - Additional dashboards
   - Advanced control features
   - Extended monitoring capabilities

5. Develop comprehensive testing strategy
   - Implement unit tests for frontend and backend
   - Create integration tests for monitoring stack
   - Build end-to-end tests for user workflows

## Technical Requirements

### Hardware Requirements

- ✅ Ensure minimum requirements are met:
  - ✅ 4GB RAM
  - ✅ 2 CPU cores
  - ✅ 20GB disk space
- 🟡 Recommended configuration:
  - 🟡 8GB RAM
  - 🟡 4 CPU cores
  - 🟡 100GB SSD

### Software Requirements

- ✅ Verify required software versions:
  - ✅ Docker Engine 20.10.x or higher
  - ✅ Docker Compose 2.x or higher
  - ✅ Node.js 18.x or higher
  - ✅ NX 16.x or higher
  - ✅ Grafana 9.x or higher (using 10.1.0)
  - ✅ Prometheus 2.40.x or higher (using 2.46.0)
  - ✅ Loki 2.7.x or higher (using 2.9.0)

### Network Requirements

- ✅ Configure required inbound ports:
  - ✅ 3000 (Grafana UI)
  - ✅ 9090 (Prometheus, optional)
  - ✅ 3100 (Loki, optional)
  - 🔲 8080 (Control API)
- ✅ Set up internal networking between containers
- 🔲 Configure optional external access via reverse proxy with TLS

## Security Implementation

- 🔲 Implement authentication & access control
  - 🔲 Secure authentication for all components
  - 🔲 Role-based access control for dashboards and controls
  - 🔲 API token-based authentication for programmatic access
- 🔲 Configure data protection
  - 🔲 Encryption of sensitive data in transit and at rest
  - 🔲 Secure storage of API keys and credentials
  - 🔲 Data retention policies and cleanup
- 🔲 Set up network security
  - 🔲 Firewall rules to restrict access
  - 🔲 TLS for all external connections
  - 🔲 Internal network isolation where possible
- 🔲 Implement vulnerability management
  - 🔲 Regular updates of all components
  - 🔲 Security scanning of container images
  - 🔲 Dependency auditing and updates

## Progress Tracking

| Component      | Task                | Status         | Assigned To | Target Date | Notes                                                                                                                         |
| -------------- | ------------------- | -------------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure | NX Setup            | ✅ Completed   |             |             |                                                                                                                               |
| Infrastructure | Docker Config       | ✅ Completed   |             |             | Successfully set up Docker Compose for monitoring stack with Prometheus, Grafana, Loki, Promtail, Node Exporter, and cAdvisor |
| Monitoring     | Core Metrics        | 🟡 In Progress |             |             | Basic metrics are being collected, but custom metrics still need implementation                                               |
| Monitoring     | Log Collection      | ✅ Completed   |             |             | Loki and Promtail configured for log collection                                                                               |
| Monitoring     | Dashboards          | ✅ Completed   |             |             | Created consolidated home dashboard with system metrics and placeholder panels for application metrics                        |
| Control        | API Development     | Not Started    |             |             |                                                                                                                               |
| Control        | Frontend            | Not Started    |             |             |                                                                                                                               |
| Control        | Authentication      | Not Started    |             |             |                                                                                                                               |
| Security       | Auth Implementation | Not Started    |             |             |                                                                                                                               |
| Security       | Encryption          | Not Started    |             |             |                                                                                                                               |
| Testing        | Performance Testing | 🟡 In Progress |             |             | Initial testing of monitoring stack performed                                                                                 |
| Testing        | Security Testing    | Not Started    |             |             |                                                                                                                               |
| Documentation  | User Docs           | Not Started    |             |             |                                                                                                                               |
| Documentation  | Deployment Guide    | Not Started    |             |             |                                                                                                                               |

## Current Implementation Status

Phase 4 is currently in progress, with significant advances in the monitoring infrastructure setup.
The following components have been successfully implemented:

### Monitoring Stack Implementation

1. ✅ **Docker Compose Configuration**: A comprehensive docker-compose.yml has been created and
   tested for the monitoring stack, including:

   - Prometheus (v2.46.0) for metrics collection
   - Grafana (v10.1.0) for visualization dashboards
   - Loki (v2.9.0) for log aggregation
   - Promtail for log collection
   - Node Exporter for host metrics
   - cAdvisor for container metrics

2. ✅ **Container Network**: All components are successfully connected through a dedicated
   monitoring network.

3. ✅ **Volume Management**: Persistent volumes have been configured for Prometheus, Grafana, and
   Loki data.

4. ✅ **Loki Configuration**:

   - Fixed permission issues with the Loki WAL directory
   - Implemented proper initialization for data directories
   - Created environment file for Loki configuration

5. ✅ **Service Accessibility**:

   - Grafana UI is accessible on port 3000
   - Prometheus UI is accessible on port 9090
   - Loki API is accessible on port 3100
   - cAdvisor metrics are available on port 8090

6. ✅ **Dashboard Implementation**:
   - Created consolidated home dashboard with system metrics
   - Added placeholder panels for application-specific metrics
   - Configured dashboard provisioning
   - Removed duplicate dashboards for simpler maintenance

## MACD Strategy Monitoring Implementation (MVP)

The following monitoring enhancements are required to support the MACD ETH-USD strategy on
Hyperliquid with 1-minute timeframe and $1.00 maximum positions.

### Core Metrics Implementation

- 🔲 Define and implement MACD strategy-specific metrics

  - 🔲
    `spark_stacker_strategy_active{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD"}`:
    Boolean indicating if strategy is active
  - 🔲
    `spark_stacker_strategy_position{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge", side="long|short"}`:
    Current position size
  - 🔲
    `spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", exchange="hyperliquid", market="ETH-USD", type="main|hedge"}`:
    Current P&L percentage
  - 🔲 `spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd", signal="buy|sell"}`:
    Counter for generated signals
  - 🔲
    `spark_stacker_strategy_trade_executed_total{strategy="macd_eth_usd", result="success|failure"}`:
    Counter for executed trades

- 🔲 Implement indicator value metrics for visualization

  - 🔲 `spark_stacker_macd_value{strategy="macd_eth_usd", component="macd|signal|histogram"}`:
    Current MACD indicator values
  - 🔲 `spark_stacker_macd_crossover_total{strategy="macd_eth_usd", direction="bullish|bearish"}`:
    Counter for MACD crossovers

- 🔲 Add time-series metrics for performance tracking
  - 🔲
    `spark_stacker_strategy_execution_seconds{strategy="macd_eth_usd", phase="signal_generation|position_sizing|order_execution"}`:
    Timing metrics for strategy execution phases
  - 🔲 `spark_stacker_strategy_trades_total{strategy="macd_eth_usd", outcome="win|loss"}`: Counter
    for trade outcomes

### Dashboard Implementation

- 🔲 Create dedicated MACD Strategy Dashboard

  - 🔲 Strategy Overview Panel

    - 🔲 Strategy status (active/inactive)
    - 🔲 Current positions (main and hedge)
    - 🔲 Current P&L
    - 🔲 Win/loss ratio
    - 🔲 MACD parameter display (8-21-5)

  - 🔲 MACD Indicator Visualization

    - 🔲 Time-series chart of MACD, signal line, and histogram
    - 🔲 Visual indicators for buy/sell signals
    - 🔲 Crossover event markers
    - 🔲 Current values prominently displayed

  - 🔲 Position History Panel

    - 🔲 Table of recent trades with entry/exit prices
    - 🔲 P&L visualization per trade
    - 🔲 Position duration statistics
    - 🔲 Histogram of trade outcomes

  - 🔲 Performance Metrics Panel

    - 🔲 Trade success rate
    - 🔲 Average P&L per trade
    - 🔲 Maximum drawdown
    - 🔲 Sharpe ratio (if available)
    - 🔲 Strategy execution timing

  - 🔲 Hyperliquid Connection Panel
    - 🔲 API latency for ETH-USD market data
    - 🔲 Order execution success rate
    - 🔲 WebSocket connection status
    - 🔲 Recent error count

- 🔲 Update Home Dashboard with MACD Strategy Status
  - 🔲 Add MACD strategy card to strategies panel
  - 🔲 Include current position and P&L in overview

### Alert Configuration

- 🔲 Create strategy-specific alerts

  - 🔲 Configure alerts for signal generation

    ```yaml
    - alert: MACDSignalGenerated
      expr: increase(spark_stacker_strategy_signal_generated_total{strategy="macd_eth_usd"}[5m]) > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy generated a new {{ $labels.signal }} signal'
    ```

  - 🔲 Set up position monitoring alerts

    ```yaml
    - alert: MACDPositionOpened
      expr: spark_stacker_strategy_position{strategy="macd_eth_usd", type="main"} > 0
      labels:
        severity: info
      annotations:
        summary: 'MACD strategy opened a {{ $labels.side }} position'
        details: 'Position size: {{ $value }}'
    ```

  - 🔲 Configure performance alerts

    ```yaml
    - alert: MACDStrategyLoss
      expr: spark_stacker_strategy_pnl_percent{strategy="macd_eth_usd", type="main"} < -3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: 'MACD strategy experiencing sustained loss'
        details: 'Current P&L: {{ $value }}%'
    ```

### Logging Enhancements

- 🔲 Implement structured logging for MACD strategy

  - 🔲 Log signal generation events

    ```json
    {
      "timestamp": "2023-03-15T12:34:56.789Z",
      "level": "INFO",
      "category": "strategy",
      "strategy": "macd_eth_usd",
      "message": "MACD signal generated",
      "data": {
        "signal": "BUY",
        "macd_value": 0.25,
        "signal_value": 0.15,
        "histogram": 0.1,
        "confidence": 0.85
      }
    }
    ```

  - 🔲 Log position events

    ```json
    {
      "timestamp": "2023-03
    ```
