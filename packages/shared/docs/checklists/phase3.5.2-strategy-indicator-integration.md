# Phase 3.5.2: Strategy-Indicator Integration Fixes

**Objective**: Fix the fundamental architectural issues between strategies and indicators to
eliminate "Market RSI-4H not found" errors and establish proper strategy-driven execution flow.

**Priority**: 🚨 **CRITICAL** - Blocking live trading functionality **Estimated Time**: 2-3 days
**Dependencies**: Phase 3.5.1 completion

## 📋 **Overview**

This phase addresses the core architectural problems identified in the Spark Stacker system:

- Strategies and indicators are disconnected
- Indicator names ("RSI-4H") being treated as market symbols
- Legacy symbol parsing causing failures
- No strategy-driven execution flow

## 🏗️ **1. Core Architecture Fixes**

### 1.1 Strategy Manager Overhaul

- [ ] **Update StrategyManager constructor** (`packages/spark-app/app/core/strategy_manager.py`)

  - [ ] Add `strategies: List[Dict[str, Any]]` parameter
  - [ ] Add `strategy_indicators: Dict[str, List[str]]` mapping
  - [ ] Create `_build_strategy_mappings()` method
  - [ ] Update initialization to accept strategy configs

- [ ] **Replace run_cycle() method** (`packages/spark-app/app/core/strategy_manager.py`)

  - [ ] Remove indicator-driven execution
  - [ ] Implement strategy-driven execution loop
  - [ ] Add strategy validation (enabled, market, exchange)
  - [ ] Add proper error handling per strategy
  - [ ] Add logging for strategy execution flow

- [ ] **Create run_strategy_indicators() method**
      (`packages/spark-app/app/core/strategy_manager.py`)

  - [ ] Accept strategy config, market, and indicator names
  - [ ] Validate indicators exist for strategy
  - [ ] Get indicator-specific timeframes
  - [ ] Prepare data per market/timeframe combination
  - [ ] Add strategy context to generated signals
  - [ ] Return list of signals with strategy metadata

- [ ] **Update \_prepare_indicator_data() method**
      (`packages/spark-app/app/core/strategy_manager.py`)
  - [ ] Accept market symbol (e.g., "ETH-USD") instead of indicator name
  - [ ] Use market + timeframe for cache keys
  - [ ] Fetch historical data using proper market symbols
  - [ ] Add error handling for data fetching failures

### 1.2 Main Application Integration

- [ ] **Remove legacy symbol parsing** (`packages/spark-app/app/main.py`)

  - [ ] Delete regex symbol parsing block (lines ~561-581)
  - [ ] Remove symbol assignment to indicators
  - [ ] Clean up related comments and logging

- [ ] **Add strategy initialization** (`packages/spark-app/app/main.py`)

  - [ ] Load strategies from config
  - [ ] Create `_validate_strategy_indicators()` function
  - [ ] Validate strategy-indicator relationships
  - [ ] Validate market symbol formats (must contain "-")
  - [ ] Validate exchange fields are present
  - [ ] Pass strategies to StrategyManager constructor

- [ ] **Update async_main() function** (`packages/spark-app/app/main.py`)
  - [ ] Add strategy loading after indicator loading
  - [ ] Add strategy-indicator validation call
  - [ ] Update StrategyManager initialization with strategies
  - [ ] Add comprehensive error handling

### 1.3 Signal Enhancement

- [ ] **Expand Signal class** (`packages/spark-app/app/indicators/base_indicator.py`)
  - [ ] Add `strategy_name: Optional[str]` field
  - [ ] Add `market: Optional[str]` field
  - [ ] Add `exchange: Optional[str]` field
  - [ ] Update `__str__()` method to include strategy context
  - [ ] Ensure backward compatibility

### 1.4 Trading Engine Updates

- [ ] **Update process_signal() method** (`packages/spark-app/app/core/trading_engine.py`)

  - [ ] Use signal's exchange context for connector routing
  - [ ] Use signal's market field instead of symbol
  - [ ] Add error handling for missing signal context
  - [ ] Add logging for signal processing flow

- [ ] **Create symbol conversion utilities** (`packages/spark-app/app/core/trading_engine.py`)

  - [ ] Add `_convert_symbol_for_exchange()` method
  - [ ] Handle Hyperliquid format ("ETH" from "ETH-USD")
  - [ ] Handle Coinbase format ("ETH-USD" unchanged)
  - [ ] Add support for additional exchanges

- [ ] **Enhance connector selection** (`packages/spark-app/app/core/trading_engine.py`)
  - [ ] Add `_get_connector_by_name()` method
  - [ ] Route to correct connector based on signal's exchange
  - [ ] Add fallback to main connector
  - [ ] Add error handling for unknown exchanges

## 🏗️ **2. Configuration Schema & Validation**

### 2.1 Strategy Configuration Schema

- [ ] **Create StrategyConfig class** (`packages/spark-app/app/core/strategy_config.py` - NEW FILE)

  - [ ] Define StrategyConfig dataclass with all required fields
  - [ ] Add validation in `__post_init__()`
  - [ ] Validate market format (must contain "-")
  - [ ] Validate exchange is specified
  - [ ] Validate indicators list is not empty

- [ ] **Create StrategyConfigLoader class** (`packages/spark-app/app/core/strategy_config.py`)
  - [ ] Add `load_strategies()` static method
  - [ ] Add comprehensive error handling
  - [ ] Add `validate_indicators()` static method
  - [ ] Verify all strategy indicators exist
  - [ ] Add detailed logging for validation steps

### 2.2 Configuration Validation

- [ ] **Validate current config.json** (`packages/shared/config.json`)
  - [ ] Ensure all strategies have proper market format ("ETH-USD")
  - [ ] Ensure all strategies specify exchange
  - [ ] Ensure all strategies list their indicators
  - [ ] Ensure all referenced indicators exist in config
  - [ ] Test configuration loading without errors

## 🧪 **3. Testing Infrastructure**

### 3.1 Unit Tests - Strategy Manager

- [ ] **Create test_strategy_manager_integration.py**
      (`packages/spark-app/tests/unit/core/test_strategy_manager_integration.py` - NEW FILE)
  - [ ] Test StrategyManager with strategy configs
  - [ ] Test `_build_strategy_mappings()` method
  - [ ] Test strategy-driven `run_cycle()` execution
  - [ ] Test `run_strategy_indicators()` method
  - [ ] Test signal generation with strategy context
  - [ ] Test error handling for missing indicators
  - [ ] Test error handling for invalid strategies

### 3.2 Unit Tests - Signal Enhancement

- [ ] **Update test_base_indicator.py**
      (`packages/spark-app/tests/unit/indicators/test_base_indicator.py`)
  - [ ] Test Signal creation with strategy context
  - [ ] Test Signal string representation
  - [ ] Test backward compatibility with existing signals
  - [ ] Test signal metadata preservation

### 3.3 Unit Tests - Trading Engine

- [ ] **Update test_trading_engine.py**
      (`packages/spark-app/tests/unit/core/test_trading_engine.py`)
  - [ ] Test `process_signal()` with strategy context
  - [ ] Test `_convert_symbol_for_exchange()` method
  - [ ] Test `_get_connector_by_name()` method
  - [ ] Test connector routing based on signal exchange
  - [ ] Test fallback behavior for unknown exchanges

### 3.4 Unit Tests - Configuration

- [ ] **Create test_strategy_config.py**
      (`packages/spark-app/tests/unit/core/test_strategy_config.py` - NEW FILE)
  - [ ] Test StrategyConfig validation
  - [ ] Test invalid market format handling
  - [ ] Test missing exchange handling
  - [ ] Test empty indicators list handling
  - [ ] Test StrategyConfigLoader functionality
  - [ ] Test strategy-indicator relationship validation

### 3.5 Integration Tests

- [ ] **Create test_strategy_indicator_integration.py**
      (`packages/spark-app/tests/integration/test_strategy_indicator_integration.py` - NEW FILE)
  - [ ] Test complete strategy execution flow
  - [ ] Test strategy → indicator → signal → trading pipeline
  - [ ] Test multi-strategy execution
  - [ ] Test multi-timeframe support
  - [ ] Test error propagation through the pipeline
  - [ ] Test configuration loading and validation

### 3.6 Fixture Updates

- [ ] **Update test fixtures** (`packages/spark-app/tests/_fixtures/`)
  - [ ] Create strategy configuration fixtures
  - [ ] Create integrated strategy-indicator fixtures
  - [ ] Update existing fixtures to include strategy context
  - [ ] Add multi-timeframe test data fixtures

## 🔍 **4. Validation & Testing**

### 4.1 Configuration Validation Tests

- [ ] **Test configuration loading**

  ```bash
  cd packages/spark-app
  .venv/bin/python -c "
  import json
  from app.core.strategy_config import StrategyConfigLoader
  from app.indicators.indicator_factory import IndicatorFactory

  with open('../shared/config.json') as f:
      config = json.load(f)

  strategies = StrategyConfigLoader.load_strategies(config['strategies'])
  indicators = IndicatorFactory.create_indicators_from_config(config['indicators'])
  StrategyConfigLoader.validate_indicators(strategies, indicators)
  print('✅ Configuration validation passed')
  "
  ```

- [ ] **Test strategy-indicator relationship validation**

  ```bash
  cd packages/spark-app
  .venv/bin/python -c "
  from app.main import _validate_strategy_indicators
  import json

  with open('../shared/config.json') as f:
      config = json.load(f)

  _validate_strategy_indicators(config['strategies'], config['indicators'])
  print('✅ Strategy-indicator validation passed')
  "
  ```

### 4.2 Unit Test Execution

- [ ] **Run all unit tests with coverage**

  ```bash
  cd packages/spark-app
  .venv/bin/python -m pytest tests/unit/ --cov=app --cov-report=html
  ```

- [ ] **Verify coverage for new code**
  - [ ] StrategyManager changes: >90% coverage
  - [ ] Signal enhancements: >95% coverage
  - [ ] Trading engine updates: >90% coverage
  - [ ] Strategy configuration: >95% coverage

### 4.3 Integration Test Execution

- [ ] **Run integration tests**

  ```bash
  cd packages/spark-app
  .venv/bin/python -m pytest tests/integration/test_strategy_indicator_integration.py -v
  ```

- [ ] **Test end-to-end strategy execution**

  ```bash
  cd packages/spark-app
  .venv/bin/python app/main.py --dry-run --strategy=eth_multi_timeframe_strategy
  ```

### 4.4 Performance Testing

- [ ] **Test multi-strategy performance**
  - [ ] Measure strategy execution time
  - [ ] Verify no memory leaks with multiple strategies
  - [ ] Test concurrent strategy execution
  - [ ] Validate data caching efficiency

## 📚 **5. Documentation Updates**

### 5.1 Code Documentation

- [ ] **Update docstrings** for all modified methods

  - [ ] StrategyManager class and methods
  - [ ] Signal class enhancements
  - [ ] Trading engine updates
  - [ ] Strategy configuration classes

- [ ] **Add inline comments** for complex logic
  - [ ] Strategy mapping logic
  - [ ] Symbol conversion logic
  - [ ] Signal routing logic
  - [ ] Error handling flows

### 5.2 Architecture Documentation

- [ ] **Update architectural-fixes.md** with implementation status
- [ ] **Update configuration.md** with new strategy schema
- [ ] **Update userguide.md** with strategy setup instructions
- [ ] **Create strategy-development.md** guide for adding new strategies

### 5.3 README Updates

- [ ] **Update main README.md** with strategy-indicator relationship explanation
- [ ] **Update packages/spark-app/README.md** with configuration examples
- [ ] **Add troubleshooting section** for common strategy configuration errors

## 🚀 **6. Deployment Preparation**

### 6.1 Migration Testing

- [ ] **Test configuration migration**
  - [ ] Backup existing config.json
  - [ ] Test new configuration loading
  - [ ] Verify backward compatibility where possible
  - [ ] Test rollback procedures

### 6.2 Performance Validation

- [ ] **Benchmark strategy execution**
  - [ ] Measure time per strategy cycle
  - [ ] Validate memory usage patterns
  - [ ] Test with multiple concurrent strategies
  - [ ] Verify no performance regression

### 6.3 Error Handling Validation

- [ ] **Test error scenarios**
  - [ ] Missing indicator configurations
  - [ ] Invalid market symbols
  - [ ] Exchange connection failures
  - [ ] Data fetching failures
  - [ ] Signal processing failures

## ✅ **7. Final Validation**

### 7.1 End-to-End Testing

- [ ] **Complete strategy execution test**

  ```bash
  cd packages/spark-app
  .venv/bin/python app/main.py --dry-run --verbose
  ```

- [ ] **Verify no "Market RSI-4H not found" errors**
- [ ] **Confirm proper signal generation with strategy context**
- [ ] **Validate exchange-specific symbol conversion**
- [ ] **Test multi-timeframe strategy execution**

### 7.2 Regression Testing

- [ ] **Run full test suite**

  ```bash
  cd packages/spark-app
  .venv/bin/python -m pytest tests/ --cov=app
  ```

- [ ] **Verify no existing functionality broken**
- [ ] **Confirm all indicators still work independently**
- [ ] **Validate trading engine compatibility**

### 7.3 Configuration Validation

- [ ] **Test strategy configuration scenarios**
  - [ ] Single strategy with single indicator
  - [ ] Single strategy with multiple indicators
  - [ ] Multiple strategies with shared indicators
  - [ ] Multiple strategies with different exchanges
  - [ ] Multiple strategies with different timeframes

## 🎯 **Success Criteria**

Upon completion of this phase:

✅ **No more "Market RSI-4H not found" errors** ✅ **Clear strategy → indicator → signal → trade
flow** ✅ **Proper symbol handling ("ETH-USD" → "ETH" for Hyperliquid)** ✅ **Multi-timeframe
support (RSI on 4h, MACD on 1h)** ✅ **Strategy-driven execution instead of indicator-driven** ✅
**Easy addition of new strategies and indicators** ✅ **Comprehensive test coverage (>90%)** ✅
**Clear documentation and examples**

## 🔄 **Next Phase**

After completion, proceed to **Phase 4: Monitoring & Control Interface** with a properly integrated
strategy-indicator architecture that supports:

- Real-time strategy performance monitoring
- Dynamic strategy enabling/disabling
- Strategy-specific risk management
- Multi-exchange strategy execution
- Strategy backtesting and optimization

---

**Last Updated**: 2024-12-28 **Status**: 🟡 **READY TO START** **Assigned**: TBD
