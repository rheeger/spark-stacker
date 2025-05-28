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

### 1.1 Exchange Symbol Conversion Utilities

- [x] **Create symbol conversion utilities** (`packages/spark-app/app/core/symbol_converter.py` -
      NEW FILE)

  - [x] Add `convert_symbol_for_exchange()` function
  - [x] Handle Hyperliquid format ("ETH" from "ETH-USD")
  - [x] Handle Coinbase format ("ETH-USD" unchanged)
  - [x] Add support for additional exchanges
  - [x] Add validation for symbol formats
  - [x] Add error handling for unknown exchanges
  - [x] Create reverse conversion utilities (exchange → standard format)

### 1.2 Strategy Manager Overhaul

- [x] **Update StrategyManager constructor** (`packages/spark-app/app/core/strategy_manager.py`)

  - [x] Add `strategies: List[Dict[str, Any]]` parameter
  - [x] Add `strategy_indicators: Dict[str, List[str]]` mapping
  - [x] Create `_build_strategy_mappings()` method
  - [x] Update initialization to accept strategy configs

- [x] **Replace run_cycle() method** (`packages/spark-app/app/core/strategy_manager.py`)

  - [x] Remove indicator-driven execution
  - [x] Implement strategy-driven execution loop
  - [x] Add strategy validation (enabled, market, exchange)
  - [x] Add proper error handling per strategy
  - [x] Add logging for strategy execution flow

- [x] **Create run_strategy_indicators() method**
      (`packages/spark-app/app/core/strategy_manager.py`)

  - [x] Accept strategy config, market, and indicator names
  - [x] Validate indicators exist for strategy
  - [x] **Use strategy timeframe for all indicators** (strategy dictates timeframe)
  - [x] Pass strategy timeframe to indicators during processing
  - [x] Prepare data per market/strategy-timeframe combination
  - [x] Add strategy context to generated signals (including timeframe)
  - [x] Return list of signals with strategy metadata

- [x] **Update \_prepare_indicator_data() method**
      (`packages/spark-app/app/core/strategy_manager.py`)
  - [x] Accept market symbol (e.g., "ETH-USD") instead of indicator name
  - [x] **Use symbol conversion utilities** to convert market symbol for exchange
  - [x] Use market + timeframe for cache keys
  - [x] Fetch historical data using exchange-specific symbols
  - [x] Add error handling for data fetching failures
  - [x] Add logging for symbol conversion process

### 1.3 Main Application Integration

- [x] **Remove legacy symbol parsing** (`packages/spark-app/app/main.py`)

  - [x] Delete regex symbol parsing block (lines ~561-581)
  - [x] Remove symbol assignment to indicators
  - [x] Clean up related comments and logging

- [x] **Add strategy initialization** (`packages/spark-app/app/main.py`)

  - [x] Load strategies from config
  - [x] Create `_validate_strategy_indicators()` function
  - [x] Validate strategy-indicator relationships
  - [x] Validate market symbol formats (must contain "-")
  - [x] Validate exchange fields are present
  - [x] Pass strategies to StrategyManager constructor

- [x] **Update async_main() function** (`packages/spark-app/app/main.py`)
  - [x] Add strategy loading after indicator loading
  - [x] Add strategy-indicator validation call
  - [x] Update StrategyManager initialization with strategies
  - [x] Add comprehensive error handling

### 1.4 Indicator Timeframe Integration

- [x] **Update BaseIndicator class** (`packages/spark-app/app/indicators/base_indicator.py`)
  - [x] **Remove fixed timeframe from indicators** - indicators should not have hardcoded timeframes
  - [x] **Update process() method** to accept timeframe parameter from strategy
  - [x] **Add get_effective_timeframe() override** to use strategy-provided timeframe when available
  - [x] Update indicator data preparation to use strategy timeframe
  - [x] Add validation for supported timeframes per indicator type

### 1.5 Signal Enhancement

- [x] **Expand Signal class** (`packages/spark-app/app/indicators/base_indicator.py`)
  - [x] Add `strategy_name: Optional[str]` field
  - [x] Add `market: Optional[str]` field
  - [x] Add `exchange: Optional[str]` field
  - [x] **Add `timeframe: Optional[str]` field** to capture strategy timeframe
  - [x] Update `__str__()` method to include strategy context
  - [x] Ensure backward compatibility

### 1.6 Trading Engine Updates

- [x] **Update process_signal() method** (`packages/spark-app/app/core/trading_engine.py`)

  - [x] Use signal's exchange context for connector routing
  - [x] Use signal's market field instead of symbol
  - [x] Add error handling for missing signal context
  - [x] Add logging for signal processing flow

- [x] **Import and use symbol conversion utilities**
      (`packages/spark-app/app/core/trading_engine.py`)

  - [x] Import `convert_symbol_for_exchange()` from symbol_converter
  - [x] Use symbol conversion utilities instead of local methods
  - [x] Update `_execute_trade()` to use standardized symbol conversion
  - [x] Remove duplicate symbol conversion code

- [x] **Enhance connector selection** (`packages/spark-app/app/core/trading_engine.py`)
  - [x] Add `_get_connector_by_name()` method
  - [x] Route to correct connector based on signal's exchange
  - [x] Add fallback to main connector
  - [x] Add error handling for unknown exchanges

## 🏗️ **2. Configuration Schema & Validation**

### 2.1 Strategy Configuration Schema

- [x] **Create StrategyConfig class** (`packages/spark-app/app/core/strategy_config.py` - NEW FILE)

  - [x] Define StrategyConfig dataclass with all required fields
  - [x] Add validation in `__post_init__()`
  - [x] Validate market format (must contain "-")
  - [x] Validate exchange is specified
  - [x] Validate indicators list is not empty
  - [x] **Add timeframe field** (`timeframe: str = "1h"`) - strategy dictates timeframe for all
        indicators
  - [x] Add position sizing configuration field (`position_sizing: Optional[Dict[str, Any]]`)
  - [x] Add risk management fields (stop_loss_pct, take_profit_pct, max_position_size_usd)
  - [x] **Validate timeframe format** (e.g., "1m", "5m", "1h", "4h", "1d")

- [x] **Create StrategyConfigLoader class** (`packages/spark-app/app/core/strategy_config.py`)
  - [x] Add `load_strategies()` static method
  - [x] Add comprehensive error handling
  - [x] Add `validate_indicators()` static method
  - [x] Verify all strategy indicators exist
  - [x] Add detailed logging for validation steps
  - [x] Validate position sizing configurations per strategy

### 2.2 Strategy-Specific Position Sizing Integration

- [x] **Update RiskManager for strategy context**
      (`packages/spark-app/app/risk_management/risk_manager.py`)

  - [x] Add `strategy_position_sizers: Dict[str, PositionSizer]` attribute
  - [x] Add `_create_strategy_position_sizers()` method
  - [x] Update `calculate_position_size()` to accept strategy name parameter
  - [x] Add strategy context routing to appropriate position sizer
  - [x] Update `from_config()` to handle strategy-specific position sizing configs
  - [x] Add fallback to default position sizer for strategies without specific config

- [x] **Create strategy position sizer factory**
      (`packages/spark-app/app/risk_management/risk_manager.py`)

  - [x] Add `_create_position_sizer_for_strategy()` method
  - [x] Merge strategy-specific config with global defaults
  - [x] Handle inheritance of global position sizing parameters
  - [x] Add validation for strategy position sizing configuration
  - [x] Log position sizer creation per strategy

- [x] **Update TradingEngine for strategy context**
      (`packages/spark-app/app/core/trading_engine.py`)
  - [x] Pass strategy name to risk manager methods
  - [x] Update `_execute_trade()` to include strategy context
  - [x] Update hedge parameter calculation with strategy context
  - [x] Add strategy-specific risk logging

### 2.3 Configuration Validation

- [x] **Validate current config.json** (`packages/shared/config.json`)
  - [x] Ensure all strategies have proper market format ("ETH-USD")
  - [x] Ensure all strategies specify exchange
  - [x] Ensure all strategies list their indicators
  - [x] Ensure all referenced indicators exist in config
  - [x] Test configuration loading without errors
  - [x] Validate strategy-specific position sizing configurations
  - [x] Test position sizer creation for each strategy

## 🧪 **3. Testing Infrastructure**

### 3.1 Unit Tests - Symbol Conversion

- [x] **Create test_symbol_converter.py**
      (`packages/spark-app/tests/connectors/unit/test_symbol_converter.py` - NEW FILE)
  - [x] Test `convert_symbol_for_exchange()` with Hyperliquid format
  - [x] Test `convert_symbol_for_exchange()` with Coinbase format
  - [x] Test symbol conversion with unknown exchanges
  - [x] Test reverse symbol conversion utilities
  - [x] Test symbol validation
  - [x] Test error handling for invalid symbols

### 3.2 Unit Tests - Strategy Manager

- [x] **Create test_strategy_manager_integration.py**
      (`packages/spark-app/tests/core/integration/test_strategy_manager_integration.py` - NEW FILE)
  - [x] Test StrategyManager with strategy configs
  - [x] Test `_build_strategy_mappings()` method
  - [x] Test strategy-driven `run_cycle()` execution
  - [x] Test `run_strategy_indicators()` method
  - [x] Test signal generation with strategy context
  - [x] Test error handling for missing indicators
  - [x] Test error handling for invalid strategies
  - [x] Test symbol conversion in `_prepare_indicator_data()` method

### 3.3 Unit Tests - Signal Enhancement

- [x] **Update test_base_indicator.py**
      (`packages/spark-app/tests/indicators/unit/test_base_indicator.py`)
  - [x] Test Signal creation with strategy context
  - [x] Test Signal string representation
  - [x] Test backward compatibility with existing signals
  - [x] Test signal metadata preservation

### 3.4 Unit Tests - Trading Engine

- [x] **Update test_trading_engine.py**
      (`packages/spark-app/tests/backtesting/unit/test_trading_engine.py`)
  - [x] Test `process_signal()` with strategy context
  - [x] Test symbol conversion utilities integration
  - [x] Test `_get_connector_by_name()` method
  - [x] Test connector routing based on signal exchange
  - [x] Test fallback behavior for unknown exchanges

### 3.5 Unit Tests - Configuration

- [x] **Create test_strategy_config.py**
      (`packages/spark-app/tests/core/unit/test_strategy_config.py` - NEW FILE)
  - [x] Test StrategyConfig validation
  - [x] Test invalid market format handling
  - [x] Test missing exchange handling
  - [x] Test empty indicators list handling
  - [x] Test StrategyConfigLoader functionality
  - [x] Test strategy-indicator relationship validation
  - [x] Test strategy-specific position sizing configuration validation
  - [x] Test position sizing inheritance from global config

### 3.6 Unit Tests - Strategy-Specific Position Sizing

- [x] **Update test_risk_manager_integration.py**
      (`packages/spark-app/tests/risk_management/unit/test_risk_manager_integration.py`)

  - [x] Test RiskManager creation with strategy-specific position sizing
  - [x] Test `calculate_position_size()` with strategy context
  - [x] Test strategy position sizer routing
  - [x] Test fallback to default position sizer
  - [x] Test multiple strategies with different position sizing methods
  - [x] Test position sizer factory method

- [x] **Create test_strategy_position_sizing.py**
      (`packages/spark-app/tests/risk_management/unit/test_strategy_position_sizing.py` - NEW FILE)

  - [x] Test strategy-specific fixed USD position sizing
  - [x] Test strategy-specific risk-based position sizing
  - [x] Test strategy-specific percent equity position sizing
  - [x] Test position sizing config inheritance
  - [x] Test invalid strategy position sizing configs
  - [x] Test strategy position sizer creation and validation

- [x] **Update test_trading_engine.py**
      (`packages/spark-app/tests/backtesting/unit/test_trading_engine.py`)
  - [x] Test strategy context passing to risk manager
  - [x] Test position size calculation with strategy names
  - [x] Test hedge parameter calculation with strategy context
  - [x] Test trade execution with strategy-specific position sizing

### 3.7 Integration Tests

- [x] **Create test_strategy_indicator_integration.py**
      (`packages/spark-app/tests/core/integration/test_strategy_indicator_integration.py` - NEW
      FILE)
  - [x] Test complete strategy execution flow
  - [x] Test strategy → indicator → signal → trading pipeline
  - [x] Test multi-strategy execution
  - [x] Test multi-timeframe support
  - [x] Test error propagation through the pipeline
  - [x] Test configuration loading and validation
  - [x] Test strategy-specific position sizing in full pipeline
  - [x] Test multiple strategies with different position sizing methods
  - [x] Test symbol conversion integration across the pipeline

### 3.8 Fixture Updates

- [x] **Update test fixtures** (`packages/spark-app/tests/_fixtures/`)
  - [x] Create strategy configuration fixtures
  - [x] Create integrated strategy-indicator fixtures
  - [x] Update existing fixtures to include strategy context
  - [x] Add multi-timeframe test data fixtures
  - [x] Add strategy-specific position sizing fixture data
  - [x] Create fixtures for multiple strategies with different position sizing
  - [x] Add symbol conversion test fixtures for different exchanges

## 🔍 **4. Validation & Testing**

### 4.1 Configuration Validation Tests

- [x] **Test configuration loading**

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

- [x] **Test strategy-indicator relationship validation**

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

- [x] **Test strategy-specific position sizing validation**

  ```bash
  cd packages/spark-app
  .venv/bin/python -c "
  import json
  from app.risk_management.risk_manager import RiskManager
  from app.core.strategy_config import StrategyConfigLoader

  with open('../shared/config.json') as f:
      config = json.load(f)

  # Test RiskManager creation with strategy configs
  strategies = StrategyConfigLoader.load_strategies(config['strategies'])
  risk_manager = RiskManager.from_config(config)

  # Test position sizer creation for each strategy
  for strategy in strategies:
      print(f'Testing position sizing for strategy: {strategy.name}')
      position_sizer = risk_manager._create_position_sizer_for_strategy(strategy)
      print(f'✅ Created position sizer for {strategy.name}: {position_sizer.config.method.value}')

  print('✅ Strategy-specific position sizing validation passed')
  "
  ```

### 4.2 Unit Test Execution

- [x] **Run all unit tests with coverage**

  ```bash
  cd packages/spark-app
  .venv/bin/python -m pytest tests/*/unit/ --cov=app --cov-report=html
  ```

- [x] **Verify coverage for new code**
  - [x] StrategyManager changes: >90% coverage
  - [x] Signal enhancements: >95% coverage
  - [x] Trading engine updates: >90% coverage
  - [x] Strategy configuration: >95% coverage

### 4.3 Integration Test Execution

- [x] **Run integration tests**

  ```bash
  cd packages/spark-app
  .venv/bin/python -m pytest tests/integration/test_strategy_indicator_integration.py -v
  ```

- [x] **Test end-to-end strategy execution**

  ```bash
  cd packages/spark-app
  .venv/bin/python app/main.py --dry-run --strategy=eth_multi_timeframe_strategy
  ```

### 4.4 Performance Testing

- [x] **Test multi-strategy performance**
  - [x] Measure strategy execution time
  - [x] Verify no memory leaks with multiple strategies
  - [x] Test concurrent strategy execution
  - [x] Validate data caching efficiency

## 📚 **5. Documentation Updates**

### 5.1 Code Documentation

- [x] **Update docstrings** for all modified methods

  - [x] Symbol conversion utilities
  - [x] StrategyManager class and methods
  - [x] Signal class enhancements
  - [x] Trading engine updates
  - [x] Strategy configuration classes
  - [x] RiskManager strategy-specific methods
  - [x] Position sizing strategy integration

- [x] **Add inline comments** for complex logic
  - [x] Symbol conversion logic and exchange mappings
  - [x] Strategy mapping logic
  - [x] Signal routing logic
  - [x] Error handling flows
  - [x] Strategy-specific position sizer creation
  - [x] Position sizing inheritance logic

### 5.2 Architecture Documentation

- [x] **Update architectural-fixes.md** with implementation status, move to Retro folder with
      updated name
- [x] **Remove the IMMEDIATE-FIXES-SUMMARY.md file**
- [x] **Update configuration.md** with new strategy schema
- [x] **Update userguide.md** with strategy setup instructions
- [x] **Create strategy-development.md** guide for adding new strategies
- [x] **Create symbol-conversion.md** guide for exchange symbol handling
- [x] **Create strategy-position-sizing.md** guide for configuring position sizing per strategy
- [x] **Update risk-management.md** with strategy-specific position sizing documentation

### 5.3 README Updates

- [ ] **Update main README.md** with strategy-indicator relationship explanation
- [ ] **Update packages/spark-app/README.md** with configuration examples
- [ ] **Add troubleshooting section** for common strategy configuration errors
- [ ] **Add section on strategy-specific position sizing** with configuration examples
- [ ] **Update configuration examples** to show position sizing per strategy

## 🚀 **6. Deployment Preparation**

### 6.1 Migration Testing

- [x] **Test configuration migration**
  - [x] Backup existing config.json
  - [x] Test new configuration loading

### 6.2 Performance Validation

- [ ] **Benchmark strategy execution**
  - [ ] Measure time per strategy cycle
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
  - [ ] Multiple strategies with different position sizing methods
  - [ ] Strategy with custom position sizing vs global defaults
  - [ ] Strategy position sizing inheritance and override scenarios

## 🎯 **Success Criteria**

Upon completion of this phase:

✅ **No more "Market RSI-4H not found" errors** ✅ **Clear strategy → indicator → signal → trade
flow** ✅ **Proper symbol handling ("ETH-USD" → "ETH" for Hyperliquid)** ✅ **Multi-timeframe
support (RSI on 4h, MACD on 1h)** ✅ **Strategy-driven execution instead of indicator-driven** ✅
**Strategy-specific position sizing (each strategy can have different sizing methods)** ✅
**Position sizing inheritance from global config with strategy-specific overrides** ✅ **Risk
management integration with strategy context** ✅ **Easy addition of new strategies and indicators**
✅ **Comprehensive test coverage (>90%)** ✅ **Clear documentation and examples**

## 🔄 **Next Phase**

After completion, proceed to **Phase 4: Monitoring & Control Interface** with a properly integrated
strategy-indicator architecture that supports:

- Real-time strategy performance monitoring
- Dynamic strategy enabling/disabling
- Strategy-specific risk management and position sizing
- Multi-exchange strategy execution with different position sizing per exchange
- Strategy backtesting and optimization with configurable position sizing
- Strategy-specific position sizing monitoring and adjustment

---

**Last Updated**: 2024-12-28 **Status**: 🟡 **READY TO START** **Assigned**: TBD
