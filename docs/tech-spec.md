# Technical Specification: On-Chain Perpetual Trading System

## System Overview

The **On-Chain Perpetual Trading System** is a Python-based trading application designed to execute high-leverage positions on decentralized perpetual futures exchanges while implementing sophisticated hedging strategies to protect principal. The system receives signals from technical indicators, executes primary and hedge trades, and actively manages risk throughout the trade lifecycle.

## System Architecture

The system follows a modular architecture with clearly separated components for flexibility and maintainability:

### 1. **Exchange Connector Layer**

- **Supported Exchanges:**
  - **Synthetix Perps (Optimism/Base)** - Smart contract-based DEX with deep liquidity
  - **Hyperliquid DEX** - High-performance on-chain orderbook exchange

- **Integration Methods:**
  - **Synthetix:**
    - Web3.py for Ethereum interactions
    - Synthetix TypeScript/JavaScript SDK (optional wrapper)
    - Subgraph/GraphQL for market data retrieval
  - **Hyperliquid:**
    - Hyperliquid Python SDK for API access
    - HTTP API with ECDSA signature authentication
    - WebSocket API for real-time data

- **Exchange Adaptor Interface:**
  - Generic methods (place_order, get_market_data, get_position)
  - Exchange-specific implementations handling signature requirements

### 2. **Indicator & Signal Module**

- **Indicator Sources:**
  - **Internal Calculation:**
    - TA-Lib / Pandas-TA integration (150+ technical indicators)
    - Market data ingestion via exchange APIs
    - Real-time and historical indicator computation
  - **External Signal Reception:**
    - TradingView webhook receiver
    - Pine Script alert parsing and signal extraction
    - Standardized signal format conversion

- **Signal Processing:**
  - Signal standardization (direction, asset, confidence)
  - Signal filtering and combination logic
  - Signal timing optimization

### 3. **Trade Execution Module**

- **Order Flow Pipeline:**
  1. Signal validation and trade preparation
  2. Main position order construction
  3. Order submission and confirmation
  4. Hedge position calculation and execution
  5. Combined position monitoring

- **Order Type Management:**
  - Market orders for immediate execution
  - Limit orders to minimize slippage
  - Smart order routing based on market conditions

- **Hedge Implementation:**
  - Configurable hedge ratio (typically 20-50% of main position)
  - Leverage optimization for both positions
  - Cross-exchange hedging when appropriate

### 4. **Risk Management System**

- **Position Risk Controls:**
  - Leverage limitations by exchange and asset
  - Position sizing rules (percentage of capital)
  - Stop-loss and take-profit placement

- **Margin Monitoring:**
  - Real-time margin ratio tracking
  - Maintenance margin threshold alerts
  - Liquidation prevention actions

- **Liquidation Mitigation:**
  - Partial deleveraging procedures
  - Hedge position adjustment in volatile conditions
  - Principal protection mechanisms

### 5. **Backtesting & Simulation Engine**

- **Historical Data Handling:**
  - Price candle retrieval and storage
  - Indicator calculation on historical series
  - Realistic fee and funding rate simulation

- **Simulation Components:**
  - Trade execution simulation with slippage models
  - P&L tracking throughout simulated periods
  - Performance metrics calculation (win rate, Sharpe ratio)

### 6. **Monitoring & Logging System**

- **Live Position Tracking:**
  - Current position status and P&L
  - Margin health and liquidation risk
  - Combined hedge effectiveness metrics

- **Logging Infrastructure:**
  - Structured logging with detailed context
  - Critical event alerting
  - Transaction recording for audit purposes

## Technical Implementation Details

### Programming Languages & Frameworks

- **Primary Language:** Python 3.9+
- **Key Libraries:**
  - **Web3.py:** For Ethereum/Optimism blockchain interactions
  - **TA-Lib / Pandas-TA:** For technical indicator calculations
  - **Synthetix SDK / ABIs:** For Synthetix contract interaction
  - **Hyperliquid Python SDK:** For Hyperliquid API integration
  - **Flask / FastAPI:** For webhook reception from TradingView
  - **NumPy / Pandas:** For data manipulation and analysis
  - **SQLite / PostgreSQL:** For trade logging and performance tracking

### Exchange-Specific Integration Details

#### Synthetix Integration

- **Contract Interaction:**
  - Synthetix Perps V2/V3 smart contracts via Web3
  - Key functions: `modifyPosition()` for trade execution
  - Oracle-based pricing with ~8 second update intervals
  - Support for up to 50× leverage on major assets

- **Data Retrieval:**
  - The Graph subgraph queries for market data
  - Current price, funding rates, and open interest monitoring
  - Position status and margin management

#### Hyperliquid Integration

- **API Authentication:**
  - ECDSA signature generation for order authentication
  - API key management and security practices
  - Request signing as per Hyperliquid documentation

- **Order Execution:**
  - REST API endpoints for order placement
  - WebSocket connections for real-time order status
  - Market data streaming for order book depth

### Hedging Implementation

The core hedging strategy will be implemented as follows:

```python
# Pseudocode for hedge execution
def execute_hedged_trade(signal, asset, confidence):
    # Calculate position sizes
    available_margin = get_available_margin()
    main_position_margin = available_margin * MAIN_POSITION_RATIO  # e.g. 0.8
    hedge_position_margin = available_margin * HEDGE_POSITION_RATIO  # e.g. 0.2
    
    # Calculate leverage based on confidence and risk settings
    main_leverage = calculate_dynamic_leverage(confidence, MAX_LEVERAGE)
    hedge_leverage = HEDGE_LEVERAGE_RATIO * main_leverage  # e.g. 0.5 * main_leverage
    
    # Execute main position in signal direction
    main_order = {
        "asset": asset,
        "side": signal.direction,  # LONG or SHORT
        "margin": main_position_margin,
        "leverage": main_leverage,
        "order_type": "MARKET"
    }
    main_order_id = exchange.place_order(main_order)
    
    # Confirm main order execution
    main_position = wait_for_order_execution(main_order_id)
    if not main_position:
        log.error("Main position failed to execute")
        return None
    
    # Execute hedge in opposite direction
    hedge_order = {
        "asset": asset,
        "side": "SHORT" if signal.direction == "LONG" else "LONG",
        "margin": hedge_position_margin,
        "leverage": hedge_leverage,
        "order_type": "MARKET"
    }
    hedge_order_id = hedge_exchange.place_order(hedge_order)
    
    # Confirm hedge execution
    hedge_position = wait_for_order_execution(hedge_order_id)
    
    # Return combined position for monitoring
    return {
        "main_position": main_position,
        "hedge_position": hedge_position,
        "net_exposure": calculate_net_exposure(main_position, hedge_position),
        "timestamp": current_time()
    }
```

### Risk Management Implementation

```python
# Pseudocode for risk monitoring
def monitor_position_health(combined_position):
    # Track margin ratios for both positions
    main_margin_ratio = get_margin_ratio(combined_position["main_position"])
    hedge_margin_ratio = get_margin_ratio(combined_position["hedge_position"])
    
    # Check for liquidation risk on main position
    if main_margin_ratio < LIQUIDATION_WARNING_THRESHOLD:
        if hedge_margin_ratio > HEDGE_PROFIT_THRESHOLD:
            # Close profitable hedge to free up margin
            close_position(combined_position["hedge_position"])
            log.info("Closed hedge position to protect main position")
            return True
        else:
            # Close both positions if both are at risk
            close_position(combined_position["main_position"])
            close_position(combined_position["hedge_position"])
            log.warning("Closed both positions due to liquidation risk")
            return False
    
    # Check for stop-loss conditions
    net_pnl_percent = calculate_net_pnl_percent(combined_position)
    if net_pnl_percent < STOP_LOSS_THRESHOLD:
        # Close both positions
        close_position(combined_position["main_position"])
        close_position(combined_position["hedge_position"])
        log.info(f"Stop-loss triggered at {net_pnl_percent}%")
        return False
    
    return True  # Position is healthy
```

## Performance & Latency Considerations

- **Execution Speed Optimization:**
  - Websocket connections for real-time data to minimize latency
  - Asynchronous request handling where appropriate
  - Prioritization of critical transactions (e.g., stop-loss execution)

- **Network Reliability:**
  - Connection retry mechanisms with exponential backoff
  - Redundant endpoints for critical exchanges
  - Transaction status verification

- **Transaction Timing:**
  - Account for Synthetix oracle delay (~8 seconds between updates)
  - Optimize order placement timing to account for network conditions
  - Monitor execution slippage compared to expected price

## Testing Methodology

### 1. Unit Testing

- **Exchange Connector Testing:**
  - Mock exchange API responses for deterministic testing
  - Verify correct order formation and parameter handling
  - Test authentication and signature generation
  - Validate error handling and retry logic

- **Indicator Testing:**
  - Validate indicator calculation against known expected values
  - Test with edge cases (insufficient data, extreme values)
  - Verify signal generation logic for buy/sell conditions
  - Test indicator parameter validation

- **Risk Management Testing:**
  - Verify position sizing calculations
  - Test leverage constraints and margin requirements
  - Validate stop-loss and take-profit logic
  - Test liquidation prevention mechanisms

- **Trading Engine Testing:**
  - Verify signal processing pipeline
  - Test trade execution flow
  - Validate state management and transitions
  - Test active position monitoring

### 2. Integration Testing

- End-to-end signal to order flow
- Cross-exchange hedge coordination
- Error handling and recovery
- Webhook signal reception and processing

### 3. Simulation Testing

- Backtesting on historical data
- Stress testing with extreme market scenarios
- Performance evaluation against benchmark strategies
- Parameter optimization through simulations

### 4. Live Testing

- Testnet deployment for exchange integration verification
- Paper trading with live price feeds
- Small-scale live trading before full deployment
- A/B testing of strategy variations

## Testing Infrastructure

- **Testing Framework:** Pytest with fixtures for common test components
- **Test Automation:** File watcher for continuous test execution during development
- **Coverage Reporting:** Test coverage analysis to ensure critical paths are tested
- **Mocking:** Mock objects for external dependencies to enable deterministic testing
- **CI/CD Integration:** Automated testing on code changes

## Development Roadmap

### Phase 1: Design & Prototyping
- Requirement finalization
- Development environment setup
- Basic prototype with dummy indicators
- Initial exchange connectivity tests

### Phase 2: Core Development & Backtesting
- Indicator module implementation
- Hedging logic development
- Backtesting framework construction
- Strategy parameter optimization

### Phase 3: Integration & Dry Run
- Full system integration
- Paper trading on testnet
- End-to-end testing
- Performance and security review

### Phase 4: Deployment & Live Trading
- Initial deployment with minimal capital
- Performance monitoring and comparison to backtests
- Gradual capital scaling
- Ongoing optimization and improvement

## Security Considerations

- **Private Key Management:**
  - Secure storage of signing keys and API credentials
  - Environment variable-based configuration
  - Hardware security module integration (future)

- **Trade Validation:**
  - Double-check order parameters before submission
  - Rate limiting to prevent excessive trading
  - Maximum order size limitations

- **System Access:**
  - Authentication for dashboard access
  - Role-based permissions for strategy modifications
  - Audit logging for all system interactions

This technical specification provides a comprehensive overview of the system architecture, implementation details, and development approach for the On-Chain Perpetual Trading System.
