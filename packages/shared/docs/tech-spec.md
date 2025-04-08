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
  - **Prometheus Client:** For metrics collection and monitoring
  - **Kubernetes Python Client:** For GKE integration and management

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
import asyncio
import websockets
import json
from datetime import datetime, timedelta
import pandas as pd
from app.core.types import Timeframe

class HyperliquidMarketDataCollector:
    """
    Specialized collector for 1-minute candle data from Hyperliquid.
    Uses WebSocket for real-time data and REST API for historical data.
    """
    def __init__(self, client, market):
        self.client = client
        self.market = market
        self.ws_endpoint = "wss://api.hyperliquid.xyz/ws"
        self.ws_connection = None
        self.candle_buffer = {}  # Buffer for current candles
        self.candle_history = pd.DataFrame()  # Historical candle data
        self.timeframe = Timeframe.ONE_MINUTE

    async def connect(self):
        """Establish WebSocket connection and subscribe to market data."""
        self.ws_connection = await websockets.connect(self.ws_endpoint)
        subscribe_msg = {
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "market": self.market
            }
        }
        await self.ws_connection.send(json.dumps(subscribe_msg))

        # Start background task to process incoming messages
        asyncio.create_task(self._process_messages())

        # Initialize with recent historical data
        await self._load_initial_history()

    async def _load_initial_history(self):
        """Load initial historical data to prime the collector."""
        # Get candles for the past hour (60 1-minute candles)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)

        # Convert to UNIX timestamps in milliseconds
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)

        # Fetch historical candles from REST API
        candles = await self.client.get_candles(
            self.market,
            Timeframe.ONE_MINUTE,
            start_time=start_ts,
            end_time=end_ts
        )

        # Convert to DataFrame with datetime index
        self.candle_history = pd.DataFrame(candles)
        self.candle_history['timestamp'] = pd.to_datetime(self.candle_history['timestamp'], unit='ms')
        self.candle_history.set_index('timestamp', inplace=True)

        self.logger.info(
            f"Loaded {len(self.candle_history)} initial candles for {self.market}"
        )

    async def _process_messages(self):
        """Process incoming WebSocket messages and update candle data."""
        try:
            while True:
                message = await self.ws_connection.recv()
                data = json.loads(message)

                if "trades" in data:
                    # Process trade data to build 1-minute candles
                    for trade in data["trades"]:
                        await self._process_trade(trade)

                elif "error" in data:
                    self.logger.error(f"WebSocket error: {data['error']}")
        except Exception as e:
            self.logger.error(f"WebSocket processing error: {str(e)}")
            # Attempt to reconnect
            await self.connect()

    async def _process_trade(self, trade):
        """Process a single trade and update the current candle."""
        # Extract trade details
        timestamp = trade["timestamp"]
        price = float(trade["price"])
        size = float(trade["size"])

        # Determine which 1-minute candle this belongs to
        candle_time = datetime.fromtimestamp(timestamp / 1000)
        # Truncate to the start of the minute
        candle_time = candle_time.replace(second=0, microsecond=0)
        candle_key = candle_time.isoformat()

        # Update or create candle
        if candle_key not in self.candle_buffer:
            self.candle_buffer[candle_key] = {
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': size
            }
        else:
            candle = self.candle_buffer[candle_key]
            candle['high'] = max(candle['high'], price)
            candle['low'] = min(candle['low'], price)
            candle['close'] = price
            candle['volume'] += size

        # Check if we need to finalize any candles (older than 1 minute)
        current_time = datetime.now()
        for time_key in list(self.candle_buffer.keys()):
            candle_time = datetime.fromisoformat(time_key)
            if current_time - candle_time > timedelta(minutes=1, seconds=10):
                # This candle is complete, add to history
                self._finalize_candle(time_key, self.candle_buffer[time_key])
                del self.candle_buffer[time_key]

    def _finalize_candle(self, time_key, candle_data):
        """Add a completed candle to the candle history."""
        # Convert to DataFrame row and add to history
        candle_time = datetime.fromisoformat(time_key)
        df_row = pd.DataFrame([candle_data], index=[candle_time])
        self.candle_history = pd.concat([self.candle_history, df_row])

        # Notify any listeners that new candle data is available
        self._notify_candle_update(candle_time, candle_data)

    def _notify_candle_update(self, candle_time, candle_data):
        """Notify listeners that a new candle is available."""
        # Implement observer pattern to notify strategy of new data
        if hasattr(self, "on_candle_update") and callable(self.on_candle_update):
            self.on_candle_update(candle_time, candle_data)

        # Also publish metrics
        if hasattr(self, "metrics_client"):
            self.metrics_client.gauge(
                "spark_stacker_candle_price",
                candle_data['close'],
                {"market": self.market, "timeframe": "1m", "type": "close"}
            )
            self.metrics_client.gauge(
                "spark_stacker_candle_volume",
                candle_data['volume'],
                {"market": self.market, "timeframe": "1m"}
            )

    async def get_latest_candles(self, count=100):
        """Get the most recent candles from history."""
        return self.candle_history.tail(count)

    async def close(self):
        """Close WebSocket connection."""
        if self.ws_connection:
            await self.ws_connection.close()
```

### De-Minimus Production Testing Implementation

Instead of using testnet environments, we'll implement de-minimus trading on production exchanges
with minimal capital:

```python
from decimal import Decimal

class DeMinimusTradeExecutor:
    """
    Specialized executor for de-minimus trading on production exchanges.
    Enforces strict position size limits and implements enhanced monitoring.
    """
    def __init__(self, connector, max_position_size=1.0, metrics_client=None):
        self.connector = connector
        self.max_position_size = Decimal(str(max_position_size))
        self.metrics_client = metrics_client
        self.logger = logging.getLogger(__name__)
        self.active_positions = {}

    async def execute_order(self, market, side, size, leverage, order_type="MARKET"):
        """Execute an order with strict size enforcement."""
        # Convert size to Decimal for precise comparison
        size_decimal = Decimal(str(size))

        # Enforce maximum position size
        if size_decimal > self.max_position_size:
            self.logger.warning(
                f"Order size {size} exceeds maximum allowed size {self.max_position_size}. "
                f"Size will be capped."
            )
            size = float(self.max_position_size)

        # Log the de-minimus trade attempt
        self.logger.info(
            f"Executing de-minimus {side} order for {size} {market} with {leverage}x leverage"
        )

        # Record metrics before execution
        if self.metrics_client:
            self.metrics_client.counter(
                "spark_stacker_deminimus_trade_attempt",
                1,
                {"market": market, "side": side, "order_type": order_type}
            )

        # Execute the order on the production exchange
        try:
            order_result = await self.connector.place_order(
                market=market,
                side=side,
                size=size,
                leverage=leverage,
                order_type=order_type
            )

            # Record successful execution
            if self.metrics_client:
                self.metrics_client.counter(
                    "spark_stacker_deminimus_trade_success",
                    1,
                    {"market": market, "side": side}
                )

            # Track the position
            if "position_id" in order_result:
                self.active_positions[order_result["position_id"]] = {
                    "market": market,
                    "side": side,
                    "size": size,
                    "leverage": leverage,
                    "entry_price": order_result.get("price", 0),
                    "entry_time": datetime.now().isoformat()
                }

            return order_result

        except Exception as e:
            # Record failure
            if self.metrics_client:
                self.metrics_client.counter(
                    "spark_stacker_deminimus_trade_failure",
                    1,
                    {"market": market, "side": side, "error": str(e)[:50]}
                )
            self.logger.error(f"De-minimus trade execution failed: {str(e)}")
            raise

    async def close_position(self, position_id):
        """Close a position and record results."""
        if position_id not in self.active_positions:
            self.logger.warning(f"Position {position_id} not found in active positions")
            return False

        position = self.active_positions[position_id]

        try:
            # Close the position
            result = await self.connector.close_position(
                market=position["market"],
                position_id=position_id
            )

            # Calculate metrics if price information is available
            if "exit_price" in result and position.get("entry_price"):
                entry_price = Decimal(str(position["entry_price"]))
                exit_price = Decimal(str(result["exit_price"]))
                side = position["side"]

                # Calculate PnL percentage
                if side.upper() == "BUY" or side.upper() == "LONG":
                    pnl_pct = ((exit_price / entry_price) - 1) * 100 * Decimal(str(position["leverage"]))
                else:
                    pnl_pct = ((entry_price / exit_price) - 1) * 100 * Decimal(str(position["leverage"]))

                # Record metrics
                if self.metrics_client:
                    self.metrics_client.gauge(
                        "spark_stacker_deminimus_trade_pnl",
                        float(pnl_pct),
                        {"market": position["market"], "side": position["side"]}
                    )

                self.logger.info(
                    f"Closed de-minimus position with PnL: {float(pnl_pct)}%"
                )

            # Remove from active positions
            del self.active_positions[position_id]
            return result

        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {str(e)}")
            if self.metrics_client:
                self.metrics_client.counter(
                    "spark_stacker_deminimus_position_close_failure",
                    1,
                    {"market": position["market"], "error": str(e)[:50]}
                )
            raise
```

### Google Cloud Platform Deployment

To ensure persistent operation, the system will be deployed to Google Kubernetes Engine (GKE):

```python
from kubernetes import client, config
from google.cloud import secretmanager

class GCPDeploymentManager:
    """
    Manages deployment of the trading application to Google Kubernetes Engine.
    Handles Kubernetes configuration, secret management, and deployment monitoring.
    """
    def __init__(self, project_id, cluster_name, namespace="spark-stacker"):
        self.project_id = project_id
        self.cluster_name = cluster_name
        self.namespace = namespace
        self.logger = logging.getLogger(__name__)
        self.secret_client = secretmanager.SecretManagerServiceClient()

        # Load GKE credentials
        try:
            config.load_kube_config()
            self.k8s_client = client.CoreV1Api()
            self.k8s_apps = client.AppsV1Api()
        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes client: {str(e)}")
            raise

    def create_secret_from_sm(self, secret_name, k8s_secret_name):
        """Create Kubernetes secret from Google Secret Manager."""
        try:
            # Get secret from Secret Manager
            secret_path = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
            response = self.secret_client.access_secret_version(request={"name": secret_path})
            secret_value = response.payload.data.decode("UTF-8")

            # Create K8s secret
            secret = client.V1Secret(
                metadata=client.V1ObjectMeta(
                    name=k8s_secret_name,
                    namespace=self.namespace
                ),
                string_data={"value": secret_value}
            )

            # Check if secret already exists
            try:
                self.k8s_client.read_namespaced_secret(
                    name=k8s_secret_name,
                    namespace=self.namespace
                )
                # Update existing secret
                self.k8s_client.replace_namespaced_secret(
                    name=k8s_secret_name,
                    namespace=self.namespace,
                    body=secret
                )
                self.logger.info(f"Updated Kubernetes secret {k8s_secret_name}")
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    # Create new secret
                    self.k8s_client.create_namespaced_secret(
                        namespace=self.namespace,
                        body=secret
                    )
                    self.logger.info(f"Created Kubernetes secret {k8s_secret_name}")
                else:
                    raise

            return True

        except Exception as e:
            self.logger.error(f"Failed to create secret {k8s_secret_name}: {str(e)}")
            return False

    def deploy_trading_app(self, image, replicas=1):
        """Deploy the trading application to GKE."""
        try:
            # Create deployment configuration
            container = client.V1Container(
                name="trading-app",
                image=image,
                resources=client.V1ResourceRequirements(
                    requests={"cpu": "100m", "memory": "512Mi"},
                    limits={"cpu": "500m", "memory": "1Gi"}
                ),
                env=[
                    client.V1EnvVar(
                        name="ENVIRONMENT",
                        value="production"
                    ),
                    client.V1EnvVar(
                        name="HYPERLIQUID_API_KEY",
                        value_from=client.V1EnvVarSource(
                            secret_key_ref=client.V1SecretKeySelector(
                                name="hyperliquid-credentials",
                                key="value"
                            )
                        )
                    )
                ],
                liveness_probe=client.V1Probe(
                    http_get=client.V1HTTPGetAction(
                        path="/health",
                        port=8080
                    ),
                    initial_delay_seconds=30,
                    period_seconds=30
                )
            )

            # Create pod template
            template = client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"app": "trading-app"}),
                spec=client.V1PodSpec(containers=[container])
            )

            # Create deployment spec
            spec = client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(
                    match_labels={"app": "trading-app"}
                ),
                template=template
            )

            # Create deployment
            deployment = client.V1Deployment(
                api_version="apps/v1",
                kind="Deployment",
                metadata=client.V1ObjectMeta(name="trading-app"),
                spec=spec
            )

            # Apply deployment
            try:
                self.k8s_apps.read_namespaced_deployment(
                    name="trading-app",
                    namespace=self.namespace
                )
                # Update existing deployment
                self.k8s_apps.replace_namespaced_deployment(
                    name="trading-app",
                    namespace=self.namespace,
                    body=deployment
                )
                self.logger.info("Updated trading application deployment")
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    # Create new deployment
                    self.k8s_apps.create_namespaced_deployment(
                        namespace=self.namespace,
                        body=deployment
                    )
                    self.logger.info("Created trading application deployment")
                else:
                    raise

            return True

        except Exception as e:
            self.logger.error(f"Failed to deploy trading application: {str(e)}")
            return False

    def deploy_monitoring_stack(self):
        """Deploy Prometheus, Grafana, and other monitoring components."""
        # Implementation for deploying monitoring stack to GKE
        # This would create deployments for Prometheus, Grafana, etc.
        pass

    def check_deployment_status(self, deployment_name="trading-app"):
        """Check the status of a deployment."""
        try:
            status = self.k8s_apps.read_namespaced_deployment_status(
                name=deployment_name,
                namespace=self.namespace
            )

            ready_replicas = status.status.ready_replicas or 0
            total_replicas = status.status.replicas or 0

            self.logger.info(
                f"Deployment status: {ready_replicas}/{total_replicas} replicas ready"
            )

            return {
                "name": deployment_name,
                "ready_replicas": ready_replicas,
                "total_replicas": total_replicas,
                "is_ready": ready_replicas == total_replicas and total_replicas > 0
            }

        except Exception as e:
            self.logger.error(f"Failed to check deployment status: {str(e)}")
            return {
                "name": deployment_name,
                "error": str(e),
                "is_ready": False
            }
```

### 1-Minute Trading Implementation

The system is optimized for 1-minute trading to allow observation during development sessions:

```python
class OneMinuteMACD:
    """
    Optimized MACD strategy for 1-minute timeframes.
    Implements specialized logic for high-frequency signal detection.
    """
    def __init__(self, connector, metrics_client=None):
        self.connector = connector
        self.metrics_client = metrics_client
        self.logger = logging.getLogger(__name__)
        self.signal_buffer = []  # Buffer to filter out noise

    def initialize(self):
        """Set up data collection and signal processing."""
        # Set up market data collector for 1-minute candles
        self.collector = HyperliquidMarketDataCollector(
            self.connector,
            "ETH-USD"
        )

        # Set up MACD indicator with fast parameters
        self.macd = MACDIndicator(
            name="MACD_1m",
            params={
                "fast_period": 8,
                "slow_period": 21,
                "signal_period": 5
            }
        )

        # Set up de-minimus trade executor
        self.executor = DeMinimusTradeExecutor(
            self.connector,
            max_position_size=1.0,
            metrics_client=self.metrics_client
        )

        # Register for candle updates
        self.collector.on_candle_update = self.on_candle_update

    async def start(self):
        """Start the strategy execution."""
        await self.collector.connect()
        # Initialize with historical data
        initial_candles = await self.collector.get_latest_candles(30)
        if not initial_candles.empty:
            # Calculate initial MACD values
            self.macd.process(initial_candles)
            self.logger.info("Initialized 1-minute MACD with historical data")

    async def on_candle_update(self, candle_time, candle_data):
        """Process new candle data and generate signals."""
        # Get latest candles including the new one
        latest_candles = await self.collector.get_latest_candles(30)

        # Calculate MACD values and check for signals
        processed_data, signal = self.macd.process(latest_candles)

        # Log for monitoring
        self._log_indicator_values(processed_data.iloc[-1])

        # If we got a signal, validate and act
        if signal:
            # Add to signal buffer for noise filtering
            self.signal_buffer.append({
                "time": candle_time,
                "direction": signal.direction,
                "strength": signal.strength
            })

            # Only keep the most recent 5 signals
            if len(self.signal_buffer) > 5:
                self.signal_buffer.pop(0)

            # Check if this is a valid signal (not noise)
            if self._validate_signal(signal):
                await self._execute_signal(signal, latest_candles.iloc[-1])

    def _validate_signal(self, signal):
        """Additional validation to filter out noise in 1-minute data."""
        # If we have at least 3 signals, check consistency
        if len(self.signal_buffer) >= 3:
            # Check the most recent 3 signals
            recent_signals = self.signal_buffer[-3:]

            # If direction changes frequently, it's likely noise
            directions = [s["direction"] for s in recent_signals]
            if len(set(directions)) > 1:
                self.logger.info("Signal rejected: inconsistent direction in recent signals")
                return False

        # Check signal strength - weak signals might be noise
        if signal.strength < 0.3:
            self.logger.info(f"Signal rejected: strength {signal.strength} below threshold")
            return False

        return True

    async def _execute_signal(self, signal, current_candle):
        """Execute a validated signal."""
        # Get current price
        current_price = current_candle["close"]

        # Calculate position size
        strategy = MACDStrategy(market="ETH-USD")
        position_sizing = strategy.calculate_position_size(signal, current_price)

        # Execute main position
        main_pos = position_sizing["main_position"]
        try:
            main_order = await self.executor.execute_order(
                market="ETH-USD",
                side=main_pos["direction"],
                size=main_pos["notional"],
                leverage=main_pos["leverage"]
            )

            # Execute hedge position
            hedge_pos = position_sizing["hedge_position"]
            hedge_order = await self.executor.execute_order(
                market="ETH-USD",
                side=hedge_pos["direction"],
                size=hedge_pos["notional"],
                leverage=hedge_pos["leverage"]
            )

            self.logger.info(
                f"Executed 1-minute MACD signal: {signal.direction} with "
                f"${main_pos['notional']} main position and "
                f"${hedge_pos['notional']} hedge position"
            )

            # Track position for management
            self._track_combined_position(main_order, hedge_order)

        except Exception as e:
            self.logger.error(f"Failed to execute 1-minute MACD signal: {str(e)}")

    def _track_combined_position(self, main_order, hedge_order):
        """Track the combined position for management."""
        # Implementation for position tracking and management
        pass

    def _log_indicator_values(self, last_row):
        """Log current indicator values for monitoring."""
        # Similar to the implementation in MACDStrategy class
        pass
```

This technical implementation provides the foundation for the de-minimus real-money testing approach
using 1-minute timeframes, with plans for deployment to Google Cloud Platform for persistent
operation.

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
