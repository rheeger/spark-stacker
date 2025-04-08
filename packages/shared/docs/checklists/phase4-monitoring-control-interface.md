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
- 🔲 Prepare GCP Kubernetes deployment for continuous monitoring and operation

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
- 🔲 Google Cloud Platform monitoring deployment
  - 🔲 Configure monitoring stack for GKE deployment
  - 🔲 Setup Cloud Load Balancer for monitoring interfaces
  - 🔲 Configure persistent disk storage for monitoring data
  - 🔲 Setup IAM roles for secure access to monitoring

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
- 🔲 Add high-frequency monitoring for 1-minute timeframes
  - 🔲 Add metrics specific to short-interval trading
    - 🔲
      `spark_stacker_candle_data{exchange="name", market="symbol", timeframe="1m", type="open|close|high|low"}`:
      Price data
    - 🔲
      `spark_stacker_indicator_value{indicator="MACD", component="main|signal|histogram", market="ETH-USD"}`:
      Indicator values
    - 🔲 `spark_stacker_signal_latency_ms{exchange="name", market="symbol"}`: Signal to execution
      latency
  - 🔲 Implement efficient storage for high-frequency metrics
  - 🔲 Configure appropriate aggregation rules for 1-minute data
  - 🔲 Monitor real-time position state during short timeframe testing

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
  - 🔲 1-minute timeframe performance visualization
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
- 🔲 Create 1-minute Hyperliquid trading dashboard
  - 🔲 Real-time MACD value visualization
  - 🔲 Signal detection display for short timeframes
  - 🔲 Position lifecycle tracking for 1-minute strategy
  - 🔲 Short-term P&L visualization
  - 🔲 Live indicator value charts
  - 🔲 Current price stream with signal overlay
  - 🔲 Active position monitoring with real-time updates

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
  - 🔲 Validate GKE security configuration
- 🔲 Implement monitoring for the monitoring system
  - 🔲 Set up meta-monitoring for Prometheus
  - 🔲 Configure Grafana availability monitoring
  - 🔲 Implement log monitoring for the monitoring stack
  - 🔲 Create alerts for monitoring system failures
- 🔲 Optimize for high-frequency monitoring
  - 🔲 Test Prometheus performance with 1-minute metrics
  - 🔲 Optimize metric cardinality for short timeframes
  - 🔲 Configure appropriate retention policies for frequency tiers
  - 🔲 Implement efficient visualization for high-volume data
  - 🔲 Test monitoring performance during active trading periods

## Documentation & Deployment

- 🔲 Create user documentation
  - 🔲 Dashboard usage guide
  - 🔲 Alert interpretation documentation
  - 🔲 Control interface user manual
  - 🔲 API documentation
  - 🔲 De-minimus trading monitoring guide
- 🔲 Document deployment procedures
  - 🔲 Monitoring stack deployment guide
  - 🔲 Configuration management
  - 🔲 Backup and recovery procedures
  - 🔲 Scaling guidelines
  - 🔲 Google Cloud Platform deployment guide
  - 🔲 Kubernetes configuration documentation
  - 🔲 Cloud security best practices

## Google Cloud Platform Integration (CRITICAL PATH)

- 🔲 Define GKE cluster architecture
  - 🔲 Configure appropriate node pool types
  - 🔲 Set up auto-scaling configuration
  - 🔲 Define resource quotas and limits
- 🔲 Create Kubernetes deployment manifests
  - 🔲 Trading application deployment
  - 🔲 Monitoring stack deployment
  - 🔲 Database services configuration
  - 🔲 Network policy definitions
- 🔲 Configure persistent storage
  - 🔲 Set up Persistent Volumes for databases
  - 🔲 Configure Persistent Volumes for monitoring data
  - 🔲 Set up backup policies for critical data
- 🔲 Implement security best practices
  - 🔲 Configure IAM roles and service accounts
  - 🔲 Set up Secret Manager for sensitive credentials
  - 🔲 Implement network security policies
  - 🔲 Configure VPC and firewall rules
  - 🔲 Secure storage of exchange API keys
  - 🔲 Implement least privilege principle for service accounts
- 🔲 Set up CI/CD pipeline for GCP deployment
  - 🔲 Create Cloud Build configuration
  - 🔲 Set up Artifact Registry for container images
  - 🔲 Configure deployment automation
  - 🔲 Implement testing in deployment pipeline

## De-Minimus Production Monitoring Support

- 🔲 Configure real-time monitoring for 1-minute trading
  - 🔲 Create high-resolution dashboards for active observation
  - 🔲 Set up alerts for trade execution anomalies
  - 🔲 Implement position tracking visualizations
- 🔲 Implement metrics for $1.00 position trading
  - 🔲 Configure precision handling for small position sizes
  - 🔲 Add specialized metrics for micro-positions
  - 🔲 Create P&L tracking optimized for small trades
- 🔲 Enhance Hyperliquid-specific monitoring
  - 🔲 Add exchange-specific metrics for Hyperliquid
  - 🔲 Implement monitoring for funding rates and market conditions
  - 🔲 Create visualizations for trade execution quality

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

The focus of the next development phase is to enhance monitoring capabilities to support de-minimus
production trading with 1-minute timeframes and prepare for GCP Kubernetes deployment for continuous
operation.
