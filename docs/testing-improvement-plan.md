# Testing Improvement Plan

## Coverage Analysis (as of March 2025)

Current overall test coverage: **18%** (253/1407 statements covered)

| Module | Coverage | Missing Statements | Total Statements | Priority |
|--------|----------|-------------------|-----------------|----------|
| app/main.py | 0% | 128/128 | 128 | High |
| app/webhook/webhook_server.py | 4% | 150/157 | 157 | High |
| app/connectors/hyperliquid_connector.py | 9% | 262/287 | 287 | Critical |
| app/core/trading_engine.py | 10% | 251/280 | 280 | Critical |
| app/risk_management/risk_manager.py | 10% | 126/140 | 140 | High |
| app/indicators/rsi_indicator.py | 21% | 42/53 | 53 | Medium |
| app/connectors/connector_factory.py | 27% | 41/56 | 56 | Medium |
| app/indicators/indicator_factory.py | 34% | 29/44 | 44 | Medium |
| app/utils/config.py | 44% | 92/164 | 164 | Medium |
| app/indicators/base_indicator.py | 56% | 17/39 | 39 | Low |
| app/connectors/base_connector.py | 73% | 16/59 | 59 | Low |

## Implementation Checklist

### Phase 1: Critical Components (Target: 40% overall coverage)

#### HyperliquidConnector (9% → 50%)
- [ ] Create mock responses for API calls
  - [ ] Create mock order responses
  - [ ] Create mock market data responses
  - [ ] Create mock account data responses
- [ ] Test core trading functionality
  - [ ] Test place_order for market orders
  - [ ] Test place_order for limit orders
  - [ ] Test cancel_order functionality
  - [ ] Test close_position functionality
  - [ ] Test get_positions functionality
- [ ] Test error handling and edge cases
  - [ ] Test behavior with invalid API credentials
  - [ ] Test behavior with network failures
  - [ ] Test handling of API rate limits
  - [ ] Test behavior with unusual market conditions
- [ ] Test connection and reconnection logic
  - [ ] Test connect/disconnect methods
  - [ ] Test auto-reconnection after failures

#### Trading Engine (10% → 50%)
- [ ] Test signal processing logic
  - [ ] Test process_signal with various signal types
  - [ ] Test signal validation and filtering
  - [ ] Test signal queueing and priorities
- [ ] Test trade execution logic
  - [ ] Test execute_trade for long positions
  - [ ] Test execute_trade for short positions
  - [ ] Test execute_hedged_trade functionality
- [ ] Test position management
  - [ ] Test tracking of active trades
  - [ ] Test trade history logging
  - [ ] Test position closure logic
- [ ] Test engine lifecycle
  - [ ] Test start/stop functionality
  - [ ] Test pause/resume functionality
  - [ ] Test error recovery mechanisms

### Phase 2: High Priority Components (Target: 60% overall coverage)

#### main.py (0% → 40%)
- [ ] Test application initialization
  - [ ] Test load_config functionality
  - [ ] Test connector initialization
  - [ ] Test indicator initialization
  - [ ] Test risk manager initialization
  - [ ] Test trading engine initialization
- [ ] Test command line argument processing
  - [ ] Test --config argument
  - [ ] Test --dry-run argument
  - [ ] Test --verbose argument
- [ ] Test startup sequence
  - [ ] Test normal startup flow
  - [ ] Test startup with missing configuration
  - [ ] Test startup with invalid configuration

#### Webhook Server (4% → 60%)
- [ ] Test server initialization and configuration
  - [ ] Test with different host/port combinations
  - [ ] Test with and without authentication token
- [ ] Test endpoint functionality
  - [ ] Test health check endpoint ('/')
  - [ ] Test primary webhook endpoint ('/webhook')
  - [ ] Test TradingView endpoint ('/webhook/tradingview')
- [ ] Test signal parsing and validation
  - [ ] Test parse_signal with valid data
  - [ ] Test parse_signal with invalid data
  - [ ] Test parse_tradingview_alert with various formats
- [ ] Test authentication and security
  - [ ] Test with valid token
  - [ ] Test with invalid token
  - [ ] Test with missing token
- [ ] Test server lifecycle
  - [ ] Test start/stop methods
  - [ ] Test handling of concurrent requests

#### Risk Manager (10% → 70%)
- [ ] Test position size calculation
  - [ ] Test calculate_position_size with various inputs
  - [ ] Test leverage boundary conditions
  - [ ] Test balance percentage constraints
- [ ] Test risk limit enforcement
  - [ ] Test validate_trade for valid trades
  - [ ] Test validate_trade for trades exceeding risk limits
  - [ ] Test validate_trade for trades exceeding margin requirements
- [ ] Test position management
  - [ ] Test update_positions method
  - [ ] Test position tracking accuracy
- [ ] Test hedging calculations
  - [ ] Test calculate_hedge_parameters with various inputs
  - [ ] Test hedge ratio constraints
- [ ] Test drawdown protection
  - [ ] Test max drawdown enforcement
  - [ ] Test should_close_position logic

### Phase 3: Medium Priority Components (Target: 80% overall coverage)

#### RSI Indicator (21% → 80%)
- [ ] Test calculation correctness
  - [ ] Test with known input-output pairs
  - [ ] Test with edge case data (all increasing, all decreasing, flat)
  - [ ] Test with missing data points
- [ ] Test parameter variations
  - [ ] Test with different period values
  - [ ] Test with different overbought/oversold thresholds
  - [ ] Test with different signal period values
- [ ] Test signal generation
  - [ ] Test oversold to neutral transitions
  - [ ] Test overbought to neutral transitions
  - [ ] Test signal validation logic

#### Connector Factory (27% → 80%)
- [ ] Test connector creation
  - [ ] Test create_connector with valid parameters
  - [ ] Test create_connector with invalid parameters
  - [ ] Test creation of all supported connector types
- [ ] Test connector registration
  - [ ] Test register_connector with valid connector classes
  - [ ] Test register_connector with invalid classes
- [ ] Test configuration-based creation
  - [ ] Test create_connectors_from_config with valid configurations
  - [ ] Test with missing required fields
  - [ ] Test with disabled connectors
  - [ ] Test with duplicate connector names

#### Indicator Factory (34% → 80%)
- [ ] Test indicator creation
  - [ ] Test create_indicator with valid parameters
  - [ ] Test create_indicator with invalid parameters
  - [ ] Test creation of all supported indicator types
- [ ] Test indicator registration
  - [ ] Test register_indicator with valid indicator classes
  - [ ] Test register_indicator with invalid classes
- [ ] Test configuration-based creation
  - [ ] Test create_indicators_from_config with valid configurations
  - [ ] Test with missing required fields
  - [ ] Test with disabled indicators
  - [ ] Test with invalid indicator types

#### Config Utils (44% → 80%)
- [ ] Test config loading
  - [ ] Test loading from valid JSON files
  - [ ] Test loading from invalid JSON files
  - [ ] Test loading from non-existent files
- [ ] Test config validation
  - [ ] Test with valid configurations
  - [ ] Test with missing required fields
  - [ ] Test with invalid data types
  - [ ] Test with out-of-range values
- [ ] Test default values
  - [ ] Test that defaults are applied correctly
  - [ ] Test that explicit values override defaults
- [ ] Test environment variable integration
  - [ ] Test that environment variables override defaults
  - [ ] Test priority between file and environment variables

### Phase 4: Low Priority Components (Target: 85%+ overall coverage)

#### Base Indicator (56% → 90%)
- [ ] Test abstract method enforcement
  - [ ] Test that concrete classes must implement abstract methods
- [ ] Test Signal class
  - [ ] Test Signal initialization with various inputs
  - [ ] Test SignalDirection enum values
  - [ ] Test to_dict and from_dict methods
- [ ] Test process method
  - [ ] Test integration of calculate and generate_signal methods
  - [ ] Test with various input data types and formats

#### Base Connector (73% → 90%)
- [ ] Test abstract method enforcement
  - [ ] Test that concrete classes must implement abstract methods
- [ ] Test utility methods
  - [ ] Test calculate_margin_requirement method
  - [ ] Test enum definitions (OrderSide, OrderType, OrderStatus)
- [ ] Test initialization
  - [ ] Test with various name and exchange_type combinations

## Recommended Testing Practices

1. **Use Mocks and Fixtures**
   - Create reusable fixtures for common test scenarios
   - Mock external dependencies (APIs, databases, etc.)
   - Use context managers for setup/teardown

2. **Implement Parametrized Tests**
   - Use pytest's parametrize to test multiple scenarios efficiently
   - Test boundary conditions and edge cases
   - Include both valid and invalid inputs

3. **Integration Testing**
   - Test how components work together
   - Create realistic workflows that mimic production usage
   - Test error propagation between components

4. **Running the Tests**
   - Use `pytest --cov=app` to generate coverage reports
   - Use `pytest --cov=app --cov-report=html` for detailed HTML reports
   - Aim to run tests frequently during development

## Continuous Integration

- [ ] Set up GitHub Actions or similar CI pipeline
- [ ] Configure automatic test runs on each pull request
- [ ] Set minimum coverage thresholds for new code
- [ ] Generate and publish coverage reports

## Measuring Progress

Track coverage improvements over time:

| Date | Overall Coverage | Notes |
|------|------------------|-------|
| March 2025 | 18% | Initial measurement |
| | | |
| | | |

---

This testing improvement plan should be updated as progress is made or as new components are added to the system. 