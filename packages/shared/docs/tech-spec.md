# Technical Specification: On-Chain Perpetual Trading System

## System Overview

The **On-Chain Perpetual Trading System** is a Python-based trading application designed to execute
high-leverage positions on decentralized perpetual futures exchanges while implementing
sophisticated hedging strategies to protect principal. The system receives signals from technical
indicators, executes primary and hedge trades, and actively manages risk throughout the trade
lifecycle.

## System Architecture

The system follows a modular architecture with clearly separated components for flexibility and
maintainability:

### 1. **Exchange Connector Layer**

- **Supported Exchanges:**

  - **Synthetix Perps (Optimism/Base)** - Smart contract-based DEX with deep liquidity
  - **Hyperliquid DEX** - High-performance on-chain orderbook exchange
  - **Coinbase Exchange** - Major centralized exchange with robust API and high liquidity

- **Integration Methods:**

  - **Synthetix:**
    - Web3.py for Ethereum interactions
    - Synthetix TypeScript/JavaScript SDK (optional wrapper)
    - Subgraph/GraphQL for market data retrieval
  - **Hyperliquid:**
    - Hyperliquid Python SDK for API access
    - HTTP API with ECDSA signature authentication
    - WebSocket API for real-time data
  - **Coinbase:**
    - Official Coinbase Pro Python client
    - REST API with HMAC authentication
    - WebSocket API for real-time market data
    - FIX API for high-throughput trading (optional)

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

#### Coinbase Integration

- **API Authentication:**

  - HMAC-SHA256 based request signing
  - API key, secret, and passphrase management
  - Rate limit management and backoff strategies

- **Order Execution:**

  - REST API endpoints for order execution and management
  - WebSocket feeds for real-time market data
  - Support for market, limit, and stop orders
  - Position management via the Advanced Trade API

- **Data Integration:**
  - Market data polling for indicators
  - Candle data retrieval for technical analysis
  - Account balance and position monitoring

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

## MVP MACD Strategy Implementation

As a critical proof-of-concept for validating system functionality, we'll implement a specific MACD
strategy on Hyperliquid's ETH-USD market with minimal position sizes. This section details the
technical implementation of this MVP strategy.

### Strategy Class Implementation

```python
from app.indicators.macd_indicator import MACDIndicator
from app.strategies.base_strategy import BaseStrategy
from app.core.types import Market, Timeframe, Signal, SignalDirection

class MACDStrategy(BaseStrategy):
    """
    MACD strategy for 1-minute ETH-USD on Hyperliquid with custom parameters.
    Uses Fast(8), Slow(21), Signal(5) for increased sensitivity on short timeframes.
    """

    def __init__(self, market: Market, exchange: str = "hyperliquid"):
        super().__init__(
            name="MACD_ETH_USD_1m",
            market=market,
            exchange=exchange,
            timeframe=Timeframe.ONE_MINUTE
        )
        # Initialize MACD indicator with custom parameters
        self.macd = MACDIndicator(
            name="MACD_8_21_5",
            params={
                "fast_period": 8,
                "slow_period": 21,
                "signal_period": 5
            }
        )
        # Initialize position management parameters
        self.max_position_size = 1.0  # $1.00 max position
        self.leverage = 10.0
        self.hedge_ratio = 0.2  # 20% hedge
        self.stop_loss_percent = -5.0
        self.take_profit_percent = 10.0
        self.max_position_duration = 60 * 24  # 24 hours (in minutes)

    async def process_market_data(self, market_data):
        """Process incoming market data and generate signals."""
        # Apply MACD indicator to market data
        processed_data, signal = self.macd.process(market_data)

        # Log current indicator values for monitoring
        self._log_indicator_values(processed_data)

        # Return signal if one was generated
        return signal

    def _log_indicator_values(self, processed_data):
        """Log current MACD values for monitoring systems."""
        if len(processed_data) == 0:
            return

        last_row = processed_data.iloc[-1]
        self.logger.info(
            "MACD indicator values updated",
            extra={
                "strategy": self.name,
                "macd_value": float(last_row["macd"]),
                "signal_value": float(last_row["macd_signal"]),
                "histogram": float(last_row["macd_histogram"]),
                "timestamp": int(last_row.name.timestamp() * 1000)
            }
        )

        # Export metrics for monitoring
        if hasattr(self, "metrics_client"):
            self.metrics_client.gauge(
                "spark_stacker_macd_value",
                float(last_row["macd"]),
                {"strategy": self.name, "component": "macd"}
            )
            self.metrics_client.gauge(
                "spark_stacker_macd_value",
                float(last_row["macd_signal"]),
                {"strategy": self.name, "component": "signal"}
            )
            self.metrics_client.gauge(
                "spark_stacker_macd_value",
                float(last_row["macd_histogram"]),
                {"strategy": self.name, "component": "histogram"}
            )

    def calculate_position_size(self, signal, current_price):
        """Calculate position size based on strategy parameters."""
        # For MVP, we use fixed position size of $1.00
        notional_size = self.max_position_size

        # Convert notional size to asset amount
        asset_amount = notional_size / current_price

        # Calculate hedge position
        hedge_notional = notional_size * self.hedge_ratio
        hedge_amount = hedge_notional / current_price

        return {
            "main_position": {
                "amount": asset_amount,
                "notional": notional_size,
                "leverage": self.leverage,
                "direction": signal.direction
            },
            "hedge_position": {
                "amount": hedge_amount,
                "notional": hedge_notional,
                "leverage": self.leverage * 0.5,  # Half the leverage for hedge
                "direction": SignalDirection.SELL if signal.direction == SignalDirection.BUY else SignalDirection.BUY
            }
        }
```

### Market Data Collection

For 1-minute timeframe data from Hyperliquid, we'll implement an optimized collector with WebSocket
support for real-time updates:

```python
class HyperliquidMinuteDataCollector:
    """Specialized collector for 1-minute data from Hyperliquid."""

    def __init__(self, symbol="ETH-USD", websocket_manager=None):
        self.symbol = symbol
        self.websocket_manager = websocket_manager
        self.candle_cache = {}
        self.current_candle = None

    async def setup(self):
        """Initialize WebSocket connection and historical data."""
        # Subscribe to trades for building 1-minute candles
        await self.websocket_manager.subscribe_trades(self.symbol, self._process_trade)

        # Get initial historical data
        await self.fetch_historical_data()

    async def fetch_historical_data(self, lookback_periods=100):
        """Fetch initial historical 1-minute candles."""
        # Get historical 1-minute candles from REST API
        # Implementation depends on Hyperliquid API
        pass

    def _process_trade(self, trade_data):
        """Process incoming trade to build 1-minute candles in real-time."""
        # Update current candle with trade data
        # When minute changes, finalize candle and begin new one
        pass

    async def get_current_data(self, periods=30):
        """Get the most recent n periods of 1-minute data."""
        # Return DataFrame with OHLCV data for the requested periods
        pass
```

### Configuration

Strategy configuration in the system's `config.yml`:

```yaml
strategies:
  macd_eth_usd:
    name: 'MACD ETH-USD 1m'
    type: 'MACD'
    class_name: 'MACDStrategy'
    enabled: true
    exchange: 'hyperliquid'
    market: 'ETH-USD'
    timeframe: '1m'
    parameters:
      fast_period: 8
      slow_period: 21
      signal_period: 5
    risk_parameters:
      max_position_size: 1.00
      leverage: 10
      stop_loss_percent: -5.0
      take_profit_percent: 10.0
      hedge_ratio: 0.2
      max_position_duration_minutes: 1440 # 24 hours
```

### Monitoring Integration

To enable real-time monitoring of the strategy, we'll add metrics export:

```python
# In strategy initialization
self.metrics_client.gauge(
    "spark_stacker_strategy_active",
    1,
    {"strategy": "macd_eth_usd", "exchange": "hyperliquid", "market": "ETH-USD"}
)

# On signal generation
self.metrics_client.counter(
    "spark_stacker_strategy_signal_generated_total",
    1,
    {"strategy": "macd_eth_usd", "signal": signal.direction.name.lower()}
)

# On trade execution
self.metrics_client.counter(
    "spark_stacker_strategy_trade_executed_total",
    1,
    {"strategy": "macd_eth_usd", "result": "success" if success else "failure"}
)

# On position update
self.metrics_client.gauge(
    "spark_stacker_strategy_position",
    position_size,
    {
        "strategy": "macd_eth_usd",
        "exchange": "hyperliquid",
        "market": "ETH-USD",
        "type": "main",
        "side": position_direction.name.lower()
    }
)
```

### Performance Optimizations for 1-Minute Data

For high-frequency trading with 1-minute candles, several optimizations are necessary:

1. **Cache Management**:

   - Implement rolling cache for historical data
   - Store preprocessed indicator values to avoid recalculation
   - Use TTL-based caching for API responses

2. **Connection Resilience**:

   - Implement heartbeat for WebSocket connections
   - Automatic reconnection with exponential backoff
   - Duplicate connection paths for critical data

3. **Trade Execution**:
   - Optimize order placement for minimal latency
   - Implement retry logic with timeout controls
   - Add circuit breakers for error conditions

This MVP implementation provides a comprehensive test of all critical system components while
minimizing financial risk through small position sizes.

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

- **Position Management Testing:**
  - Test position opening and closing functionality
  - Validate spot market position handling
  - Verify proper tracking of open positions
  - Test error handling during position operations

### 2. Integration Testing

- End-to-end signal to order flow
- Cross-exchange hedge coordination
- Error handling and recovery
- Webhook signal reception and processing
- Real market data integration across components
- MACD indicator testing with real market data visualization
- Position closing with realistic exchange conditions

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
- **Real Market Data Cache:**
  - Local storage of market data for exchange APIs
  - Automatic refresh when data is older than 24 hours
  - Smart caching to prevent redundant API calls
  - Offline testing capability with cached data
- **Testing Scripts:**
  - Virtual environment management for consistent execution
  - Market data refresh automation
  - Visualization tools for signal analysis and debugging
- **Synthetic Data Generation:**
  - Fallback mechanism when real market data is unavailable
  - Configurable parameters for realistic test scenarios
  - Reproducible test conditions for reliable results

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

This technical specification provides a comprehensive overview of the system architecture,
implementation details, and development approach for the On-Chain Perpetual Trading System.
