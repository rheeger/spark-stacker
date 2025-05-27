# Spark Stacker Developer Guide

This guide provides in-depth technical information for developers working on the Spark Stacker
trading system.

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Spark Stacker System                     │
├─────────────────────────────────────────────────────────────┤
│  Trading Engine                                             │
│  ├── Indicator Engine (RSI, MACD, Bollinger Bands, etc.)   │
│  ├── Signal Processor (Entry/Exit Logic)                   │
│  ├── Risk Manager (Position Sizing, Stop Losses)          │
│  └── Portfolio Manager (Multi-Strategy Coordination)       │
├─────────────────────────────────────────────────────────────┤
│  Exchange Connectors                                        │
│  ├── Hyperliquid Connector                                 │
│  ├── Coinbase Connector                                    │
│  └── Kraken Connector                                      │
├─────────────────────────────────────────────────────────────┤
│  Data Management                                            │
│  ├── Market Data Feed                                      │
│  ├── Historical Data Storage                               │
│  └── Real-time Price Streams                               │
├─────────────────────────────────────────────────────────────┤
│  Backtesting Framework                                      │
│  ├── BacktestEngine                                        │
│  ├── SimulationEngine                                      │
│  ├── IndicatorBacktestManager                              │
│  └── Performance Analytics                                 │
├─────────────────────────────────────────────────────────────┤
│  Monitoring & Observability                                │
│  ├── Metrics Collection (Prometheus)                       │
│  ├── Dashboards (Grafana)                                  │
│  └── Alerting System                                       │
└─────────────────────────────────────────────────────────────┘
```

### Package Structure

The monorepo is organized into distinct packages:

- **`packages/spark-app/`**: Core trading application

  - `app/indicators/`: Technical indicator implementations
  - `app/connectors/`: Exchange API integrations
  - `app/backtesting/`: Backtesting framework
  - `app/core/`: Core trading engine components
  - `app/utils/`: Utility functions and helpers

- **`packages/monitoring/`**: Observability infrastructure

  - Grafana dashboards
  - Prometheus configurations
  - Alert rules and notification setups

- **`packages/shared/`**: Shared resources
  - Documentation and tutorials
  - Examples and common utilities

## 🔧 Indicator Development

### BaseIndicator Architecture

All indicators inherit from `BaseIndicator` and must implement two key methods:

```python
from app.indicators.base_indicator import BaseIndicator

class BaseIndicator:
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """Initialize indicator with parameters"""
        self.name = name
        self.params = params or {}

    @abc.abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicator values - adds columns to the DataFrame"""
        pass

    @abc.abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate trading signals based on indicator values"""
        pass

    def process(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Signal]]:
        """Main entry point - combines calculation and signal generation"""
        processed_data = self.calculate(data)
        signal = self.generate_signal(processed_data)
        return processed_data, signal
```

### Currently Implemented Indicators

**Available via IndicatorFactory:**

- `rsi` - RSI Indicator (Relative Strength Index)
- `macd` - MACD Indicator (Moving Average Convergence Divergence)
- `bollinger` - Bollinger Bands Indicator
- `ma` - Moving Average Indicator (SMA/EMA with crossover detection)
- `adaptive_supertrend` - Adaptive SuperTrend Indicator
- `adaptive_trend_finder` - Adaptive Trend Finder (regression-based)
- `ultimate_ma` - Ultimate Moving Average (8 different MA types)

### IndicatorFactory Pattern

The system uses a factory pattern for creating indicators:

```python
from app.indicators.indicator_factory import IndicatorFactory

# Register new indicators automatically on import
IndicatorFactory.register_defaults()

# Create indicator instances
rsi = IndicatorFactory.create_indicator(
    name="rsi_14",
    indicator_type="rsi",
    params={"period": 14, "overbought": 70, "oversold": 30}
)

# Create multiple indicators from config
indicators = IndicatorFactory.create_indicators_from_config([
    {"name": "rsi_eth", "type": "rsi", "enabled": True, "parameters": {"period": 14}},
    {"name": "macd_eth", "type": "macd", "enabled": True, "parameters": {"fast_period": 12}}
])
```

### Signal Generation Patterns

#### Real Signal Generation Examples

From RSI Indicator:

```python
def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
    latest = data.iloc[-1]

    # Oversold condition - potential buy signal
    if latest["rsi"] < self.oversold_threshold:
        return Signal(
            direction=SignalDirection.BUY,
            symbol=symbol,
            confidence=min(1.0, (self.oversold_threshold - latest["rsi"]) / 30),
            indicator=self.name,
            params={"rsi": latest["rsi"], "trigger": "oversold"}
        )

    # Overbought condition - potential sell signal
    elif latest["rsi"] > self.overbought_threshold:
        return Signal(
            direction=SignalDirection.SELL,
            symbol=symbol,
            confidence=min(1.0, (latest["rsi"] - self.overbought_threshold) / 30),
            indicator=self.name,
            params={"rsi": latest["rsi"], "trigger": "overbought"}
        )
```

From Moving Average Indicator:

```python
def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
    latest = data.iloc[-1]

    # Golden Cross - fast MA crosses above slow MA
    if latest.get("ma_crosses_above", False):
        ma_ratio = latest["ma_ratio"]
        confidence = min(1.0, 0.6 + (ma_ratio - 1) * 10)

        return Signal(
            direction=SignalDirection.BUY,
            symbol=symbol,
            indicator=self.name,
            confidence=confidence,
            params={
                "fast_ma": latest["fast_ma"],
                "slow_ma": latest["slow_ma"],
                "trigger": "golden_cross"
            }
        )
```

## 🔄 Backtesting Framework

### BacktestEngine Architecture

The real BacktestEngine implementation:

```python
from app.backtesting.backtest_engine import BacktestEngine
from app.backtesting.data_manager import DataManager

# Initialize with data manager
engine = BacktestEngine(
    data_manager=data_manager,
    initial_balance={"USD": 10000.0},
    maker_fee=0.0001,  # 0.01%
    taker_fee=0.0005,  # 0.05%
    slippage_model="random"
)

# Run backtest
result = engine.run_backtest(
    strategy_func=strategy_function,
    symbol="ETH-USD",
    interval="1h",
    start_date="2024-01-01",
    end_date="2024-06-01",
    data_source_name="csv",
    strategy_params={"fast_period": 10, "slow_period": 30},
    leverage=1.0,
    indicators=[rsi_indicator, macd_indicator]
)
```

### SimulationEngine Integration

The BacktestEngine uses SimulationEngine for realistic trade execution:

```python
# SimulationEngine handles:
# - Order execution with fees and slippage
# - Position management
# - Balance tracking
# - Trade history recording

simulation_engine = SimulationEngine(
    initial_balance={"USD": 10000.0},
    maker_fee=0.0001,
    taker_fee=0.0005,
    slippage_model="random"
)
```

### Performance Metrics Calculation

Real metrics calculated by the system:

```python
# From BacktestEngine._calculate_metrics()
metrics = {
    "total_trades": len(trades),
    "winning_trades": len(winning_trades),
    "losing_trades": len(losing_trades),
    "win_rate": winning_trades / total_trades,
    "avg_profit": average_profit_per_winning_trade,
    "avg_loss": average_loss_per_losing_trade,
    "max_profit": maximum_single_trade_profit,
    "max_loss": maximum_single_trade_loss,
    "profit_factor": gross_profit / gross_loss,
    "total_return": (final_balance - initial_balance) / initial_balance,
    "annualized_return": annualized_return_calculation,
    "max_drawdown": maximum_drawdown_percentage,
    "max_drawdown_abs": maximum_drawdown_absolute,
    "sharpe_ratio": annualized_sharpe_ratio,
    "sortino_ratio": annualized_sortino_ratio,
    "calmar_ratio": annualized_return / max_drawdown
}
```

### Advanced Backtesting Features

#### Walk-Forward Analysis (Actually Implemented)

```python
# Real implementation in BacktestEngine
results = engine.walk_forward_analysis(
    strategy_func=strategy_function,
    symbol="ETH-USD",
    interval="1h",
    start_date="2024-01-01",
    end_date="2024-12-01",
    data_source_name="csv",
    param_grid={
        "fast_period": [5, 10, 15],
        "slow_period": [20, 30, 40],
        "position_size": [0.1, 0.2, 0.3]
    },
    train_size=6,  # months
    test_size=2,   # months
    metric_to_optimize="sharpe_ratio"
)
```

#### Genetic Algorithm Optimization (Actually Implemented)

```python
# Real implementation in BacktestEngine
best_params, best_result = engine.genetic_optimize(
    strategy_func=strategy_function,
    symbol="ETH-USD",
    interval="1h",
    start_date="2024-01-01",
    end_date="2024-06-01",
    data_source_name="csv",
    param_space={
        "fast_period": [5, 10, 15, 20],
        "slow_period": [25, 30, 35, 40],
        "position_size": (0.1, 0.5, 0.1)  # (min, max, step)
    },
    population_size=20,
    generations=10,
    metric_to_optimize="sharpe_ratio"
)
```

### IndicatorBacktestManager

For testing multiple indicators:

```python
from app.backtesting.indicator_backtest_manager import IndicatorBacktestManager

manager = IndicatorBacktestManager(backtest_engine)

# Add indicators to test
manager.add_indicator(rsi_indicator)
manager.add_indicator(macd_indicator)

# Run backtests for all indicators
results = manager.backtest_all_indicators(
    symbol="ETH-USD",
    interval="1h",
    start_date="2024-01-01",
    end_date="2024-06-01",
    data_source_name="csv"
)

# Compare performance
comparison = manager.compare_indicators(metric_name="sharpe_ratio")
```

## 🔌 Exchange Connector Development

### BaseConnector Interface

All exchange connectors inherit from BaseConnector:

```python
from app.connectors.base_connector import BaseConnector, MarketType

class BaseConnector(abc.ABC):
    def __init__(self, name: str, exchange_type: str, market_types: List[MarketType] = None):
        self.name = name
        self.exchange_type = exchange_type
        self.market_types = market_types or [MarketType.SPOT]
        self._is_connected = False

    # Required methods to implement:
    @abc.abstractmethod
    def connect(self) -> bool: pass

    @abc.abstractmethod
    def disconnect(self) -> bool: pass

    @abc.abstractmethod
    def get_markets(self) -> List[Dict[str, Any]]: pass

    @abc.abstractmethod
    def get_ticker(self, symbol: str) -> Dict[str, Any]: pass

    @abc.abstractmethod
    def get_account_balance(self) -> Dict[str, float]: pass

    @abc.abstractmethod
    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                   size: float, price: Optional[float] = None) -> Dict[str, Any]: pass
```

### ConnectorFactory Pattern

Real connector creation using ConnectorFactory:

```python
from app.connectors.connector_factory import ConnectorFactory

# Create individual connectors
hyperliquid = ConnectorFactory.create_connector(
    exchange_type="hyperliquid",
    name="hyperliquid_main",
    wallet_address="0x...",
    private_key="your_private_key",
    testnet=True
)

kraken = ConnectorFactory.create_connector(
    exchange_type="kraken",
    name="kraken_spot",
    api_key="your_api_key",
    api_secret="your_api_secret",
    testnet=True
)

# Create from configuration
connectors = ConnectorFactory.create_connectors_from_config([
    {
        "name": "hyperliquid",
        "exchange_type": "hyperliquid",
        "enabled": True,
        "wallet_address": "${HYPERLIQUID_WALLET_ADDRESS}",
        "private_key": "${HYPERLIQUID_PRIVATE_KEY}",
        "testnet": True
    }
])
```

### Currently Implemented Connectors

**Available Connectors:**

- `hyperliquid` - HyperliquidConnector (perpetual futures, on-chain)
- `coinbase` - CoinbaseConnector (spot trading)
- `kraken` - KrakenConnector (spot and perpetual futures)

### Real Connection Management

From HyperliquidConnector:

```python
@retry_api_call(max_tries=3, backoff_factor=2)
def connect(self) -> bool:
    try:
        # Initialize info client for market data
        self.info = Info(base_url=self.api_url, skip_ws=True)

        # Test connection
        user_state = self.info.user_state(self.wallet_address)

        # Initialize exchange client for trading if private key provided
        if self.private_key:
            wallet = Account.from_key(self.private_key)
            self.exchange = Exchange(wallet=wallet, base_url=self.api_url)

        self._is_connected = True
        return True

    except Exception as e:
        logger.error(f"Connection failed: {e}")
        self._is_connected = False
        return False
```

## 📊 Data Management

### DataManager and DataSource Pattern

Real data management implementation:

```python
from app.backtesting.data_manager import DataManager
from app.backtesting.data_manager import CSVDataSource, ExchangeDataSource

# CSV data source
csv_source = CSVDataSource(data_directory="data/historical")
data_manager = DataManager()
data_manager.register_source("csv", csv_source)

# Exchange data source
exchange_source = ExchangeDataSource(connector=hyperliquid_connector)
data_manager.register_source("hyperliquid", exchange_source)

# Get historical data
data = data_manager.get_data(
    source_name="csv",
    symbol="ETH-USD",
    interval="1h",
    start_time=start_timestamp,
    end_time=end_timestamp
)
```

## 📈 Monitoring and Metrics

### Real Prometheus Metrics

From app/metrics/metrics.py:

```python
from prometheus_client import Counter, Gauge, Histogram

# Actual metrics collected:
trades_total = Counter(
    "spark_stacker_trades_total",
    "Total number of trades executed",
    ["result", "exchange", "side"]
)

active_positions = Gauge(
    "spark_stacker_active_positions",
    "Number of currently active positions",
    ["exchange", "market", "side"]
)

api_latency_seconds = Histogram(
    "spark_stacker_api_latency_seconds",
    "API request latency in seconds",
    ["exchange", "endpoint"]
)

pnl_percent = Gauge(
    "spark_stacker_pnl_percent",
    "Profit and loss percentage",
    ["strategy", "position_type"]
)
```

### Log Processing

Real log metrics exporter (monitoring/exporters/log-metrics.py):

```python
class SparkStackerLogMetrics:
    def _process_main_log(self, log_path: str):
        # Parse actual log patterns from spark_stacker.log:
        # - Account balances
        # - Trade executions
        # - Connection status
        # - API errors

        balance_pattern = r"Account balance for (\w+):"
        trade_pattern = r"Trade executed: .*"
        error_pattern = r"Failed to get positions.*"
```

## 🧪 Testing Patterns

### Real Test Structure

The actual test organization:

```
tests/
├── _helpers/          # Test helper utilities
├── fixtures/          # Test data fixtures
├── unit/             # Unit tests
│   ├── test_indicators/
│   ├── test_connectors/
│   └── test_backtesting/
└── integration/      # Integration tests
    ├── test_backtest_integration/
    └── test_connector_integration/
```

### Actual Test Examples

From tests/indicators/unit/test_rsi_indicator.py:

```python
class TestRSIIndicator:
    @pytest.fixture
    def rsi_indicator(self):
        return RSIIndicator("test_rsi", {"period": 14, "overbought": 70, "oversold": 30})

    def test_rsi_calculation_accuracy(self, rsi_indicator, sample_price_data):
        result = rsi_indicator.calculate(sample_price_data)

        # Verify RSI bounds
        assert (result['rsi'] >= 0).all()
        assert (result['rsi'] <= 100).all()

        # Verify no NaN after warmup period
        assert not result['rsi'].iloc[14:].isna().any()

    def test_signal_generation(self, rsi_indicator):
        # Test oversold signal generation
        oversold_data = create_oversold_test_data()
        signal = rsi_indicator.generate_signal(oversold_data)

        assert signal is not None
        assert signal.direction == SignalDirection.BUY
```

## 🔄 Development Workflow

### Adding New Indicators

1. Create indicator class inheriting from BaseIndicator
2. Implement calculate() and generate_signal() methods
3. Add to IndicatorFactory.register_defaults()
4. Write comprehensive tests
5. Update configuration examples

### Adding New Connectors

1. Create connector class inheriting from BaseConnector
2. Implement all required abstract methods
3. Register in ConnectorFactory.\_connector_registry
4. Add configuration handling
5. Write integration tests

### Running Tests

```bash
# Quick test run (recommended before commits)
cd packages/spark-app
.venv/bin/python -m pytest -m "not slow" --cov=app

# Full test suite
.venv/bin/python -m pytest --cov=app

# Specific test categories
.venv/bin/python -m pytest tests/indicators/
.venv/bin/python -m pytest tests/backtesting/
.venv/bin/python -m pytest tests/connectors/
```

This developer guide focuses exclusively on implemented patterns and real code examples from the
Spark Stacker codebase. All examples are taken from actual working implementations.
