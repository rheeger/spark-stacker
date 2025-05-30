# Phase 3.5.3: Backtesting CLI Improvements

**Objective**: Update the CLI backtesting tool to work with the new strategy-driven architecture,
enabling full strategy backtesting with proper configuration, position sizing, and comprehensive
reporting.

**Priority**: 🟡 **HIGH** - Enhances development and validation workflow **Estimated Time**: 3-4
days **Dependencies**: Phase 3.5.2 completion (Strategy-Indicator Integration)

## 📋 **Overview**

The current CLI was built before the strategy-indicator integration work and focuses on individual
indicators. This phase updates the CLI to:

- Work with strategy configurations from config.json
- Respect strategy timeframes, position sizing, and risk parameters
- Generate comprehensive strategy reports
- Maintain backward compatibility with indicator-only testing
- Improve validation and error handling

## 🏗️ **1. Core CLI Architecture Updates**

### 1.1 Configuration Integration

- [x] **Add config.json loading capabilities** (`packages/spark-app/tests/_utils/cli.py`)

  - [x] Add `--config` option to specify config file path
  - [x] Add `load_config()` function to parse config.json
  - [x] Add `validate_config()` function for configuration validation
  - [x] Add environment variable expansion (${VARIABLE} support)
  - [x] Add fallback to default config path (`../shared/config.json`)
  - [x] Add error handling for missing or invalid config files

- [x] **Create strategy discovery utilities** (`packages/spark-app/tests/_utils/cli.py`)
  - [x] Add `list_strategies()` function to show available strategies from config
  - [x] Add `get_strategy_config()` function to extract specific strategy config
  - [x] Add `validate_strategy_config()` function for strategy validation
  - [x] Add strategy filtering by exchange, market, or enabled status
  - [x] Add detailed strategy information display (indicators, timeframe, position sizing)

### 1.2 Strategy-Based Commands

- [x] **Add strategy backtesting command** (`packages/spark-app/tests/_utils/cli.py`)

  - [x] Create `@cli.command("strategy")` for strategy-specific backtesting
  - [x] Add `--strategy-name` parameter to select from config
  - [x] **Run multi-scenario testing by default** (all 7 synthetic scenarios + real data)
  - [x] Add `--days` parameter to set testing duration for all scenarios
  - [x] Add `--scenarios` parameter to select specific scenarios (e.g., "bull,bear,real")
  - [x] Add `--scenario-only` flag to run single scenario instead of full suite
  - [x] Add `--override-timeframe` option to temporarily change timeframe
  - [x] Add `--override-market` option for different market testing
  - [x] Add `--override-position-size` option for testing different sizing
  - [x] Add `--use-real-data` flag for live data vs synthetic data (legacy compatibility)
  - [x] Add validation that strategy exists in config
  - [x] Add `--export-data` flag to save scenario data for external analysis

- [x] **Add strategy comparison command** (`packages/spark-app/tests/_utils/cli.py`)
  - [x] Create `@cli.command("compare-strategies")` for multi-strategy comparison
  - [x] **Run all strategies through multi-scenario testing for fair comparison**
  - [x] Add `--strategy-names` parameter for comma-separated strategy list
  - [x] Add `--all-strategies` flag to compare all enabled strategies
  - [x] Add `--same-market` flag to filter strategies by market
  - [x] Add `--same-exchange` flag to filter strategies by exchange
  - [x] Add strategy performance ranking and comparison metrics
  - [x] **Include cross-scenario robustness scoring in comparison**
  - [x] Add `--scenarios` parameter to limit comparison to specific scenarios

### 1.3 Enhanced List Commands

- [x] **Update list-indicators command** (`packages/spark-app/tests/_utils/cli.py`)

  - [x] Show which strategies use each indicator
  - [x] Display indicator parameters from config
  - [x] Show timeframe and market context per indicator
  - [x] Add filtering by strategy or market

- [x] **Add list-strategies command** (`packages/spark-app/tests/_utils/cli.py`)
  - [x] Show all strategies from config with status (enabled/disabled)
  - [x] Display strategy details (market, exchange, timeframe, indicators)
  - [x] Show position sizing method and parameters per strategy
  - [x] Add filtering and sorting options (by market, exchange, performance)

### 1.4 CLI Modularization and Refactoring ⚠️ **INTERRUPT - CRITICAL REFACTOR NEEDED**

**Problem**: The `cli.py` file is becoming a monolithic file with too much logic, making it
difficult to maintain, test, and extend. Before implementing the remaining features, we need to
modularize the CLI architecture.

**Objective**: Break down the CLI into focused, single-responsibility modules with clear separation
of concerns.

#### 1.4.1 CLI Architecture Redesign

- [x] **Create modular CLI directory structure** (`packages/spark-app/tests/_utils/cli/` - NEW DIR)

  ```
  cli/
  ├── __init__.py                   # CLI package initialization
  ├── main.py                       # Main CLI entry point (refactored from cli.py)
  ├── commands/                     # Command handler modules
  │   ├── __init__.py
  │   ├── strategy_commands.py      # Strategy backtesting commands
  │   ├── indicator_commands.py     # Legacy indicator commands
  │   ├── comparison_commands.py    # Strategy/indicator comparison commands
  │   ├── list_commands.py          # List strategies/indicators commands
  │   └── utility_commands.py       # Config validation, migration commands
  ├── core/                         # Core business logic modules
  │   ├── __init__.py
  │   ├── config_manager.py         # Configuration loading and validation
  │   ├── data_manager.py           # Data fetching and caching
  │   ├── backtest_orchestrator.py  # Coordinates backtesting workflow
  │   └── scenario_manager.py       # Multi-scenario testing coordination
  ├── managers/                     # Specialized manager classes
  │   ├── __init__.py
  │   ├── strategy_backtest_manager.py    # Strategy-specific backtesting
  │   ├── indicator_backtest_manager.py   # Legacy indicator backtesting
  │   ├── scenario_backtest_manager.py    # Multi-scenario execution
  │   └── comparison_manager.py           # Strategy comparison logic
  ├── reporting/                    # Report generation modules
  │   ├── __init__.py
  │   ├── strategy_reporter.py      # Strategy-specific reporting
  │   ├── comparison_reporter.py    # Strategy comparison reports
  │   ├── scenario_reporter.py      # Multi-scenario reporting
  │   └── interactive_reporter.py   # Interactive trade selection features
  ├── validation/                   # Validation and error handling
  │   ├── __init__.py
  │   ├── config_validator.py       # Configuration validation
  │   ├── strategy_validator.py     # Strategy-specific validation
  │   └── data_validator.py         # Data quality validation
  └── utils/                        # Utility functions and helpers
      ├── __init__.py
      ├── cli_helpers.py            # CLI utility functions
      ├── output_formatters.py      # Console output formatting
      └── progress_trackers.py      # Progress tracking utilities
  ```

- [x] **Migrate main CLI entry point** (`packages/spark-app/tests/_utils/cli/main.py` - NEW FILE)

- [x] **Create backward compatibility shim** (`packages/spark-app/tests/_utils/cli.py` - UPDATE)
  - [x] Replace existing monolithic code with import from `cli.main`
  - [x] Maintain all existing command signatures and behavior
  - [x] Add deprecation notice about file location change
  - [x] Provide migration path documentation

#### 1.4.2 Command Handler Modules

- [x] **Create strategy command handlers**
      (`packages/spark-app/tests/_utils/cli/commands/strategy_commands.py` - NEW FILE)

  - [x] Move `@cli.command("strategy")` implementation
  - [x] Move `@cli.command("compare-strategies")` implementation
  - [x] Add strategy-specific parameter validation
  - [x] Import and use appropriate manager classes
  - [x] Handle strategy command error cases
  - [x] Add comprehensive logging for strategy commands

- [x] **Create indicator command handlers**
      (`packages/spark-app/tests/_utils/cli/commands/indicator_commands.py` - NEW FILE)

  - [x] Move existing indicator commands (`demo`, `real-data`, `compare`, `compare-popular`)
  - [x] Add deprecation warnings with strategy migration suggestions
  - [x] Maintain full backward compatibility
  - [x] Add `--suggest-strategy` functionality
  - [x] Import and use IndicatorBacktestManager

- [x] **Create comparison command handlers**
      (`packages/spark-app/tests/_utils/cli/commands/comparison_commands.py` - NEW FILE)

  - [x] Consolidate all comparison logic (strategy and indicator)
  - [x] Add unified comparison interface
  - [x] Handle cross-type comparisons (strategy vs indicator)
  - [x] Add comparison result caching
  - [x] Implement parallel comparison execution

- [x] **Create list command handlers**
      (`packages/spark-app/tests/_utils/cli/commands/list_commands.py` - NEW FILE)

  - [x] Move `list-strategies` and `list-indicators` commands
  - [x] Add advanced filtering and sorting logic
  - [x] Add formatted output with tables and colors
  - [x] Add export functionality for lists
  - [x] Add interactive selection for subsequent commands

- [x] **Create utility command handlers**
      (`packages/spark-app/tests/_utils/cli/commands/utility_commands.py` - NEW FILE)
  - [x] Add `validate-config` command for configuration checking
  - [x] Add `migrate-config` command for config file updates
  - [x] Add `diagnose` command for troubleshooting
  - [x] Add `clean-cache` command for clearing cached data
  - [x] Add `export-examples` command for generating example configs

#### 1.4.3 Core Business Logic Modules ✅ **COMPLETED**

- [x] **Create configuration manager**
      (`packages/spark-app/tests/_utils/cli/core/config_manager.py` - NEW FILE)

  - [x] Centralize all configuration loading logic
  - [x] Add configuration caching and reload functionality
  - [x] Handle environment variable expansion
  - [x] Add configuration merging (global + strategy overrides)
  - [x] Add configuration versioning and migration support
  - [x] Provide configuration validation and repair utilities

- [x] **Create data manager** (`packages/spark-app/tests/_utils/cli/core/data_manager.py` - NEW
      FILE)

  - [x] Centralize all data fetching logic (real and synthetic)
  - [x] Add intelligent caching across multiple runs
  - [x] Handle multi-timeframe data requirements
  - [x] Add data quality validation and cleanup
  - [x] Implement data source failover and retry logic
  - [x] Add data export and import functionality

- [x] **Create backtest orchestrator**
      (`packages/spark-app/tests/_utils/cli/core/backtest_orchestrator.py` - NEW FILE)

  - [x] Coordinate overall backtesting workflow
  - [x] Handle resource allocation and cleanup
  - [x] Manage parallel execution of multiple backtests
  - [x] Add progress tracking and user updates
  - [x] Handle interruption and graceful shutdown
  - [x] Coordinate between different manager types

- [x] **Create scenario manager** (`packages/spark-app/tests/_utils/cli/core/scenario_manager.py` -
      NEW FILE)
  - [x] Centralize multi-scenario testing logic
  - [x] Coordinate scenario data generation
  - [x] Handle scenario execution scheduling
  - [x] Add scenario result aggregation
  - [x] Manage scenario-specific configuration overrides
  - [x] Add scenario performance analysis and ranking

#### 1.4.4 Specialized Manager Classes

- [x] **Refactor strategy backtest manager**
      (`packages/spark-app/tests/_utils/cli/managers/strategy_backtest_manager.py` - MOVE FROM
      SECTION 2.1)

  - [x] Move from `tests/_utils/strategy_backtest_manager.py`
  - [x] Add integration with new core modules
  - [x] Add enhanced error handling and recovery
  - [x] Add strategy-specific optimization features
  - [x] Add strategy performance caching
  - [x] Add strategy result comparison utilities

- [x] **Refactor indicator backtest manager**
      (`packages/spark-app/tests/_utils/cli/managers/indicator_backtest_manager.py` - MOVE EXISTING)

  - [x] Move existing IndicatorBacktestManager logic
  - [x] Add integration with new architecture
  - [x] Maintain compatibility with existing functionality
  - [x] Add enhanced reporting features
  - [x] Add indicator performance caching

- [x] **Create scenario backtest manager**
      (`packages/spark-app/tests/_utils/cli/managers/scenario_backtest_manager.py` - MOVE FROM
      SECTION 2.4)

  - [x] Move from `tests/_utils/scenario_backtest_manager.py`
  - [x] Add integration with core scenario manager
  - [x] Add parallel scenario execution
  - [x] Add scenario result aggregation
  - [x] Add scenario-specific performance metrics

- [x] **Create comparison manager**
      (`packages/spark-app/tests/_utils/cli/managers/comparison_manager.py` - NEW FILE)
  - [x] Centralize all comparison logic
  - [x] Handle strategy-to-strategy comparisons
  - [x] Handle indicator-to-indicator comparisons
  - [x] Handle cross-type comparisons
  - [x] Add statistical comparison features
  - [x] Add comparison result export functionality

#### 1.4.5 Reporting Module Architecture

- [x] **Create strategy reporter**
      (`packages/spark-app/tests/_utils/cli/reporting/strategy_reporter.py` - NEW FILE)

  - [x] Centralize strategy-specific reporting logic
  - [x] Add comprehensive strategy report generation
  - [x] Handle strategy configuration display
  - [x] Add strategy performance breakdown
  - [x] Add strategy optimization suggestions
  - [x] Add export functionality for strategy results

- [x] **Create comparison reporter**
      (`packages/spark-app/tests/_utils/cli/reporting/comparison_reporter.py` - NEW FILE)

  - [x] Handle all types of comparison reporting
  - [x] Add side-by-side comparison displays
  - [x] Add ranking and scoring displays
  - [x] Add statistical significance testing
  - [x] Add comparison visualization generation
  - [x] Add comparison result export

- [x] **Create scenario reporter**
      (`packages/spark-app/tests/_utils/cli/reporting/scenario_reporter.py` - NEW FILE)

  - [x] Handle multi-scenario reporting
  - [x] Add scenario performance comparison tables
  - [x] Add scenario robustness analysis
  - [x] Add scenario-specific visualizations
  - [x] Add scenario correlation analysis
  - [x] Add scenario optimization recommendations

- [x] **Create interactive reporter**
      (`packages/spark-app/tests/_utils/cli/reporting/interactive_reporter.py` - NEW FILE)
  - [x] Centralize interactive report generation
  - [x] Add trade selection and highlighting features
  - [x] Add JavaScript component generation
  - [x] Add responsive design features
  - [x] Add accessibility features
  - [x] Add interactive chart configuration

#### 1.4.6 Validation Module Architecture

- [x] **Create config validator**
      (`packages/spark-app/tests/_utils/cli/validation/config_validator.py` - MOVE FROM SECTION 4.1)

  - [x] Move from `tests/_utils/config_validation.py`
  - [x] Add comprehensive configuration validation
  - [x] Add configuration repair suggestions
  - [x] Add configuration compatibility checking
  - [x] Add configuration performance analysis
  - [x] Add configuration optimization recommendations

- [x] **Create strategy validator**
      (`packages/spark-app/tests/_utils/cli/validation/strategy_validator.py` - NEW FILE)

  - [x] Add strategy-specific validation logic
  - [x] Validate strategy-indicator compatibility
  - [x] Validate strategy timeframe consistency
  - [x] Validate strategy position sizing
  - [x] Add strategy feasibility analysis
  - [x] Add strategy risk assessment

- [x] **Create data validator**
      (`packages/spark-app/tests/_utils/cli/validation/data_validator.py` - NEW FILE)
  - [x] Add data quality validation
  - [x] Add data completeness checking
  - [x] Add data consistency validation
  - [x] Add data format validation
  - [x] Add data source reliability assessment
  - [x] Add data preprocessing validation

#### 1.4.7 Migration and Integration

- [ ] **Update all import statements throughout the codebase**

  - [ ] Update test files to use new module structure
  - [ ] Update documentation references
  - [ ] Update example scripts and tutorials
  - [ ] Add import compatibility layer for transition period
  - [ ] Add migration warnings and guidance

- [ ] **Add comprehensive module testing** (`packages/spark-app/tests/_utils/cli/test_modules/` -
      NEW DIR)

  - [ ] Create unit tests for each new module
  - [ ] Add integration tests for module interactions
  - [ ] Add performance tests for new architecture
  - [ ] Add backward compatibility tests
  - [ ] Add error handling and edge case tests

- [ ] **Update CLI documentation** (`packages/spark-app/tests/_utils/cli/README.md` - NEW FILE)
  - [ ] Document new modular architecture
  - [ ] Add module interaction diagrams
  - [ ] Document extension points for new features
  - [ ] Add troubleshooting guide for common issues
  - [ ] Document best practices for adding new commands

#### 1.4.8 Benefits of Modular Architecture

**Maintainability**:

- Single-responsibility modules are easier to understand and modify
- Clear separation of concerns reduces coupling
- Individual modules can be tested and debugged independently

**Extensibility**:

- New commands can be added without modifying existing code
- New report types can be added through the reporting module system
- New validation rules can be added through the validation system

**Performance**:

- Lazy loading of modules reduces startup time
- Caching can be implemented at the module level
- Parallel execution can be optimized per module

**Testing**:

- Each module can have focused unit tests
- Integration testing becomes more systematic
- Mocking and stubbing becomes easier with clear interfaces

## 🏗️ **2. Strategy Backtesting Engine Integration**

### 2.1 Strategy Configuration Processing

- [ ] **Enhanced StrategyBacktestManager** (now in `cli/managers/strategy_backtest_manager.py`)

  - [ ] Initialize with StrategyConfig object from config.json
  - [ ] Load and validate all strategy indicators from config
  - [ ] Set up position sizing based on strategy configuration
  - [ ] Configure data sources based on strategy market and exchange
  - [ ] Handle strategy-specific timeframe and data requirements
  - [ ] Add comprehensive error handling and validation
  - [ ] **Integrate with new ConfigManager and DataManager modules**
  - [ ] **Use new validation modules for comprehensive strategy validation**

- [ ] **Integrate with modular architecture** (updated in `cli/commands/strategy_commands.py`)
  - [ ] Update strategy commands to use new StrategyBacktestManager location
  - [ ] Use ConfigManager for configuration handling
  - [ ] Use DataManager for data operations
  - [ ] Use StrategyValidator for validation
  - [ ] Add factory method in BacktestOrchestrator to choose appropriate manager
  - [ ] Ensure consistent data handling through DataManager module

### 2.2 Position Sizing Integration

- [ ] **Add strategy position sizing support** (in `cli/core/config_manager.py` and
      `cli/managers/strategy_backtest_manager.py`)

  - [ ] Load position sizing config from strategy configuration through ConfigManager
  - [ ] Support strategy-specific position sizing overrides
  - [ ] Handle position sizing inheritance from global config
  - [ ] Add validation for position sizing parameters through StrategyValidator
  - [ ] Include position sizing details in backtest reports via StrategyReporter
  - [ ] Add CLI options to override position sizing for testing in strategy commands

- [ ] **Enhanced position sizing validation utilities** (now in
      `cli/validation/strategy_validator.py`)
  - [ ] Add `validate_position_sizing_config()` function
  - [ ] Add `merge_position_sizing_config()` for strategy inheritance
  - [ ] Add `calculate_effective_position_size()` for report generation
  - [ ] Add position sizing comparison utilities for strategy comparison
  - [ ] Add position sizing impact analysis for different configurations
  - [ ] **Integrate with ConfigManager for configuration access**

### 2.3 Data Management Updates

- [ ] **Enhanced data fetching for strategies** (now in `cli/core/data_manager.py`)

  - [ ] Use strategy's market and exchange for data fetching
  - [ ] Respect strategy's timeframe for data requirements
  - [ ] Add multi-timeframe data support for strategies with mixed indicator timeframes
  - [ ] Cache data efficiently for multiple strategies on same market
  - [ ] Add data validation specific to strategy requirements through DataValidator
  - [ ] **Centralize all data operations in DataManager module**

- [ ] **Update synthetic data generation** (now in `cli/core/data_manager.py`)
  - [ ] Generate data appropriate for strategy's market and timeframe
  - [ ] Create market scenario data for strategy testing (trending, sideways, volatile)
  - [ ] Add support for multi-timeframe synthetic data generation
  - [ ] Include volume and other market data required by strategy indicators
  - [ ] **Integrate with ScenarioManager for scenario-specific data generation**

### 2.4 Multi-Scenario Testing Framework

- [ ] **Enhanced market scenario generator** (now in `cli/core/scenario_manager.py`)

  - [ ] Add **bull market scenario** (consistent uptrend with 60-80% up days)
  - [ ] Add **bear market scenario** (consistent downtrend with 60-80% down days)
  - [ ] Add **sideways/range-bound scenario** (oscillating within 5-10% range)
  - [ ] Add **high volatility scenario** (large daily swings, 15-25% moves)
  - [ ] Add **low volatility scenario** (minimal daily changes, <2% moves)
  - [ ] Add **choppy market scenario** (frequent direction changes, whipsaws)
  - [ ] Add **gap-heavy scenario** (frequent price gaps, simulating news events)
  - [ ] Ensure all scenarios generate data for the exact same timeframe duration
  - [ ] **Integrate with DataManager for centralized data operations**

- [ ] **Add real data parallel testing** (in `cli/core/data_manager.py`)

  - [ ] Fetch real market data for the same symbol and timeframe duration
  - [ ] Ensure real data covers the exact same time period as synthetic scenarios
  - [ ] Add fallback to alternative time periods if recent data insufficient
  - [ ] Cache real data to avoid repeated API calls during testing
  - [ ] Add data quality validation (sufficient candles, no gaps) through DataValidator
  - [ ] **Coordinate with ScenarioManager for scenario execution**

- [ ] **Enhanced scenario-based backtesting pipeline** (now in
      `cli/managers/scenario_backtest_manager.py`)
  - [ ] Run strategy against all synthetic market scenarios
  - [ ] Run strategy against real market data for same duration
  - [ ] Collect performance metrics for each scenario independently
  - [ ] Ensure consistent position sizing and risk parameters across scenarios
  - [ ] Add scenario labeling and metadata tracking
  - [ ] Implement parallel execution for faster scenario testing
  - [ ] **Integrate with BacktestOrchestrator for workflow coordination**
  - [ ] **Use ScenarioReporter for comprehensive scenario reporting**

## 🏗️ **3. Enhanced Reporting System**

### 3.1 Strategy Report Generation

- [ ] **Create comprehensive strategy reports** (now in `cli/reporting/strategy_reporter.py`)

  - [ ] Include full strategy configuration in report header
  - [ ] Show all strategy indicators and their individual performance
  - [ ] Display position sizing method and parameters used
  - [ ] Include risk management settings (stop loss, take profit, max position)
  - [ ] Add strategy-specific metadata (exchange, market, timeframe)
  - [ ] Show strategy vs individual indicator performance comparison
  - [ ] **Integrate with ConfigManager for configuration access**
  - [ ] **Use InteractiveReporter for trade selection features**

- [ ] **Add interactive trade analysis features** (now in `cli/reporting/interactive_reporter.py`)

  - [ ] Create interactive trade list with clickable trade entries
  - [ ] Add trade highlighting functionality when selected from list
  - [ ] Implement JavaScript-based chart interaction for trade selection
  - [ ] Add trade numbering/indexing for easy reference and navigation
  - [ ] Include trade details popup or sidebar when trade is selected
  - [ ] Add keyboard navigation (arrow keys) for trade selection
  - [ ] Implement trade filtering and search within the report
  - [ ] Add "zoom to trade" functionality to focus chart on selected trade timeframe
  - [ ] **Integrate with StrategyReporter for comprehensive strategy context**

- [ ] **Enhance chart interactivity** (in `cli/reporting/interactive_reporter.py`)

  - [ ] Add clickable trade markers on price charts
  - [ ] Implement hover tooltips for trade entry/exit points
  - [ ] Add trade annotation with PnL, duration, and strategy context
  - [ ] Create synchronized highlighting between trade list and chart
  - [ ] Add trade sequence visualization (connecting entry to exit with lines)
  - [ ] Implement chart zoom and pan controls for detailed trade analysis
  - [ ] Add toggle for showing/hiding different trade types (winning/losing)
  - [ ] **Coordinate with ScenarioReporter for scenario-specific trade highlighting**

- [ ] **Update HTML report templates** (`packages/spark-app/app/backtesting/reporting/`)
  - [ ] Add strategy configuration section to HTML templates
  - [ ] Display position sizing information and impact analysis
  - [ ] Show indicator breakdown within strategy context
  - [ ] Add strategy-specific charts and metrics
  - [ ] Include configuration vs actual performance comparison
  - [ ] Add strategy optimization suggestions based on results
  - [ ] **Integrate interactive JavaScript components from InteractiveReporter**
  - [ ] **Add responsive design for trade list and chart interaction**
  - [ ] **Include CSS styling for highlighted trades and selected states**

### 3.2 Strategy Comparison Reports

- [ ] **Create strategy comparison reporting** (now in `cli/reporting/comparison_reporter.py`)

  - [ ] Generate side-by-side strategy performance comparison
  - [ ] Include configuration differences (position sizing, indicators, timeframes)
  - [ ] Show relative performance metrics and rankings
  - [ ] Add risk-adjusted performance comparison (Sharpe ratio, max drawdown)
  - [ ] Include efficiency metrics (trades per day, win rate consistency)
  - [ ] Generate strategy combination analysis (portfolio effect)
  - [ ] **Integrate with ComparisonManager for comparison logic coordination**
  - [ ] **Use ConfigManager for configuration comparison features**

- [ ] **Add strategy ranking and analysis** (in `cli/reporting/comparison_reporter.py`)
  - [ ] Rank strategies by multiple criteria (return, Sharpe, drawdown)
  - [ ] Analyze strategy correlation and diversification benefits
  - [ ] Identify best performing strategies by market condition
  - [ ] Generate strategy allocation recommendations
  - [ ] Add sensitivity analysis for position sizing and timeframe changes
  - [ ] **Coordinate with ScenarioReporter for cross-scenario analysis**

### 3.3 Configuration Impact Analysis

- [ ] **Add configuration sensitivity analysis** (in `cli/reporting/strategy_reporter.py`)
  - [ ] Test strategy performance with different position sizing methods
  - [ ] Analyze impact of timeframe changes on strategy performance
  - [ ] Show effect of different indicator parameters on strategy results
  - [ ] Generate optimization suggestions for strategy configuration
  - [ ] Include risk parameter sensitivity (stop loss, take profit impact)
  - [ ] **Use ConfigManager for configuration variation testing**
  - [ ] **Integrate with StrategyValidator for feasibility analysis**

### 3.4 Interactive Trade Selection Technical Implementation

- [ ] **Create JavaScript modules for trade interaction** (coordinated by
      `cli/reporting/interactive_reporter.py`)
      (`packages/spark-app/app/backtesting/reporting/static/js/` - NEW DIR)

  - [ ] Add `trade-selector.js` for trade list interaction logic
  - [ ] Create `chart-highlighter.js` for chart marker highlighting
  - [ ] Add `trade-details.js` for popup/sidebar functionality
  - [ ] Create `trade-navigation.js` for keyboard and sequence navigation
  - [ ] Add `chart-zoom.js` for zoom-to-trade functionality
  - [ ] Create `trade-filter.js` for search and filtering capabilities
  - [ ] **Generate JavaScript configuration through InteractiveReporter**

- [ ] **Update chart generation for interactivity** (in `cli/reporting/interactive_reporter.py`)

  - [ ] Add unique IDs to all trade markers for JavaScript targeting
  - [ ] Include trade metadata in chart data attributes
  - [ ] Generate trade sequence data for connecting entry/exit points
  - [ ] Add chart configuration for zoom and pan capabilities
  - [ ] Create responsive chart sizing for different screen sizes
  - [ ] Add accessibility attributes for screen readers
  - [ ] **Coordinate with StrategyReporter for strategy-specific chart features**

- [ ] **Create HTML template components** (generated by `cli/reporting/interactive_reporter.py`)
      (`packages/spark-app/app/backtesting/reporting/templates/` - UPDATE)

  - [ ] Add interactive trade list component with click handlers
  - [ ] Create trade details sidebar/popup template
  - [ ] Add trade navigation controls (previous/next/filter)
  - [ ] Create responsive layout for chart and trade list
  - [ ] Add loading states for trade data and chart updates
  - [ ] Include error handling displays for JavaScript failures
  - [ ] **Integrate with modular reporting system templates**

- [ ] **Add CSS styling for interactive elements** (coordinated by
      `cli/reporting/interactive_reporter.py`)
      (`packages/spark-app/app/backtesting/reporting/static/css/` - UPDATE)
  - [ ] Style selected trade states and hover effects
  - [ ] Add highlighting styles for chart markers
  - [ ] Create responsive design for mobile/tablet viewing
  - [ ] Add animation styles for smooth transitions
  - [ ] Style trade details popup/sidebar
  - [ ] Add accessibility-friendly focus indicators
  - [ ] **Generate CSS through InteractiveReporter module system**

### 3.5 Multi-Scenario Performance Reporting

- [ ] **Create comprehensive scenario comparison reports** (now in
      `cli/reporting/scenario_reporter.py`)

  - [ ] Generate side-by-side performance comparison across all market scenarios
  - [ ] Include **Bull Market**, **Bear Market**, **Sideways**, **High Vol**, **Low Vol**,
        **Choppy**, **Gap-Heavy**, and **Real Data** results
  - [ ] Display unified metrics table with win rate, total return, Sharpe ratio, max drawdown for
        each scenario
  - [ ] Add scenario ranking by different performance criteria
  - [ ] Include statistical significance testing between scenarios
  - [ ] Generate scenario robustness score based on consistency across conditions
  - [ ] **Integrate with ScenarioManager for scenario coordination**
  - [ ] **Use ComparisonReporter for cross-scenario comparison features**

- [ ] **Add visual scenario performance comparison** (in `cli/reporting/scenario_reporter.py`)

  - [ ] Create radar/spider charts showing strategy performance across all scenarios
  - [ ] Add scenario performance heatmap (green=good, red=poor performance)
  - [ ] Generate overlay charts showing equity curves for all scenarios
  - [ ] Create trade distribution charts by scenario type
  - [ ] Add scenario-specific trade highlighting in interactive charts
  - [ ] Include market condition timeline showing when each scenario type occurred
  - [ ] **Coordinate with InteractiveReporter for scenario-specific interactivity**

- [ ] **Enhanced strategy robustness analysis** (in `cli/core/scenario_manager.py`)

  - [ ] Calculate **consistency score** (low variance across scenarios)
  - [ ] Compute **adaptability score** (performance in diverse conditions)
  - [ ] Generate **risk-adjusted robustness** (Sharpe ratio consistency)
  - [ ] Add **worst-case scenario analysis** (performance in hardest conditions)
  - [ ] Calculate **scenario correlation** (which scenarios strategy struggles with)
  - [ ] Generate **optimization recommendations** based on weak scenarios
  - [ ] **Integrate with StrategyValidator for robustness validation**
  - [ ] **Use ScenarioReporter for robustness report generation**

- [ ] **Update HTML templates for multi-scenario display** (coordinated by
      `cli/reporting/scenario_reporter.py`)
      (`packages/spark-app/app/backtesting/reporting/templates/`)
  - [ ] Add tabbed interface for switching between scenarios
  - [ ] Create scenario comparison dashboard with key metrics
  - [ ] Add expandable sections for detailed scenario analysis
  - [ ] Include scenario filtering and sorting controls
  - [ ] Add export functionality for scenario performance data
  - [ ] Create mobile-responsive design for scenario navigation
  - [ ] **Integrate with modular template system**

## 🏗️ **4. Validation and Error Handling**

### 4.1 Configuration Validation

- [ ] **Enhanced comprehensive config validation** (now in `cli/validation/config_validator.py`)

  - [ ] Validate all strategy configurations before backtesting
  - [ ] Check that all strategy indicators exist in config
  - [ ] Validate position sizing configurations
  - [ ] Ensure market symbols and exchanges are valid
  - [ ] Check timeframe compatibility between strategy and indicators
  - [ ] Add detailed error messages for configuration issues
  - [ ] **Integrate with ConfigManager for configuration access**
  - [ ] **Use StrategyValidator for strategy-specific validation**

- [ ] **Enhanced config validation utilities** (now in `cli/validation/config_validator.py`)
  - [ ] Add `validate_strategy_indicator_consistency()` function
  - [ ] Add `validate_position_sizing_config()` function
  - [ ] Add `validate_market_exchange_compatibility()` function
  - [ ] Add `validate_timeframe_consistency()` function
  - [ ] Add configuration repair suggestions for common issues
  - [ ] **Add integration with all other validation modules**

### 4.2 Enhanced Error Handling

- [ ] **Improved CLI error handling** (distributed across command modules in `cli/commands/`)

  - [ ] Add specific error handling for strategy configuration issues
  - [ ] Provide helpful error messages with fix suggestions
  - [ ] Add graceful degradation for partial configuration issues
  - [ ] Include error logging with sufficient context for debugging
  - [ ] Add retry mechanisms for data fetching failures
  - [ ] **Centralize error handling patterns across all command modules**
  - [ ] **Use validation modules for comprehensive error prevention**

- [ ] **Add pre-flight checks** (in `cli/core/backtest_orchestrator.py`)
  - [ ] Validate data availability for strategy requirements
  - [ ] Check exchange connectivity for real data commands
  - [ ] Verify indicator factory can create all required indicators
  - [ ] Test position sizer creation with strategy configuration
  - [ ] Validate output directory permissions and disk space
  - [ ] **Coordinate pre-flight checks across all manager modules**
  - [ ] **Use DataValidator for data availability checking**

## 🏗️ **5. Backward Compatibility and Migration**

### 5.1 Legacy Command Support

- [ ] **Maintain existing indicator commands** (now in `cli/commands/indicator_commands.py`)

  - [ ] Keep `demo`, `real-data`, `compare`, `compare-popular` commands working
  - [ ] Add deprecation warnings for indicator-only commands
  - [ ] Provide migration suggestions to strategy-based commands
  - [ ] Ensure existing scripts and workflows continue working
  - [ ] Add flag to disable deprecation warnings if needed
  - [ ] **Use backward compatibility shim in main cli.py file**
  - [ ] **Integrate with new IndicatorBacktestManager location**

- [ ] **Add migration utilities** (in `cli/commands/utility_commands.py`)
  - [ ] Add `--suggest-strategy` flag to indicator commands
  - [ ] Show which strategies use the specified indicator
  - [ ] Provide example strategy commands for equivalent functionality
  - [ ] Add automatic strategy creation suggestions for common patterns
  - [ ] **Use ConfigManager for strategy discovery and suggestions**

### 5.2 Configuration Migration Support

- [ ] **Add config file migration** (in `cli/commands/utility_commands.py`)
  - [ ] Add `--migrate-config` command to update old config files
  - [ ] Validate config file version and suggest updates
  - [ ] Add config file format conversion utilities
  - [ ] Provide config validation and repair suggestions
  - [ ] Generate example strategy configurations for common use cases
  - [ ] **Use ConfigValidator for migration validation**
  - [ ] **Integrate with ConfigManager for config file operations**

## 🏗️ **6. Performance and Optimization**

### 6.1 Caching and Performance

- [ ] **Optimize data caching for strategies** (in `cli/core/data_manager.py`)

  - [ ] Cache market data across multiple strategy tests
  - [ ] Reuse indicator calculations for strategies sharing indicators
  - [ ] Optimize multi-timeframe data handling
  - [ ] Add progress indicators for long-running strategy backtests
  - [ ] Implement parallel strategy execution for comparisons
  - [ ] **Coordinate caching across all manager modules**
  - [ ] **Use BacktestOrchestrator for resource optimization**

- [ ] **Add performance monitoring** (in `cli/utils/progress_trackers.py`)
  - [ ] Measure and report backtest execution time
  - [ ] Track memory usage for large strategy comparisons
  - [ ] Add performance benchmarks for strategy vs indicator testing
  - [ ] Include performance metrics in CLI output
  - [ ] Add performance optimization suggestions
  - [ ] **Integrate performance tracking across all modules**

### 6.2 Resource Management

- [ ] **Improve resource cleanup** (in `cli/core/backtest_orchestrator.py`)
  - [ ] Ensure proper cleanup after strategy backtests
  - [ ] Add timeout handling for long-running operations
  - [ ] Implement graceful shutdown for interrupted operations
  - [ ] Add disk space management for large report generation
  - [ ] Clean up temporary files and cached data appropriately
  - [ ] **Coordinate resource management across all manager modules**
  - [ ] **Use DataManager for centralized data cleanup**

## 🧪 **7. Testing Infrastructure**

### 7.1 CLI Testing Framework

- [ ] **Create modular CLI testing suite** (`packages/spark-app/tests/_utils/cli/test_modules/` -
      NEW DIR)

  - [ ] Test all new strategy commands with mock data
  - [ ] Test configuration loading and validation
  - [ ] Test strategy backtesting with various configurations
  - [ ] Test error handling and edge cases
  - [ ] Test backward compatibility with existing commands
  - [ ] Test report generation and file output
  - [ ] **Add unit tests for each CLI module individually**
  - [ ] **Add integration tests for module interactions**
  - [ ] **Test backward compatibility shim functionality**

- [ ] **Add enhanced integration tests**
      (`packages/spark-app/tests/backtesting/integration/test_cli_integration.py` - UPDATE)
  - [ ] Test CLI with real config.json file
  - [ ] Test end-to-end strategy backtesting workflow
  - [ ] Test strategy comparison functionality
  - [ ] Test configuration migration and validation
  - [ ] Test CLI performance with multiple strategies
  - [ ] **Test modular architecture integration**
  - [ ] **Test cross-module communication and data flow**

### 7.2 Test Data and Fixtures

- [ ] **Create modular CLI test fixtures** (`packages/spark-app/tests/_fixtures/cli_fixtures.py` -
      UPDATE)

  - [ ] Create test strategy configurations
  - [ ] Add test market data for various scenarios
  - [ ] Create mock backtesting results for testing
  - [ ] Add test position sizing configurations
  - [ ] Create test error scenarios and edge cases
  - [ ] **Add module-specific test fixtures**
  - [ ] **Create integration test scenarios for module combinations**

- [ ] **Add enhanced CLI test utilities** (`packages/spark-app/tests/_helpers/cli_test_helpers.py` -
      UPDATE)
  - [ ] Add CLI command testing utilities
  - [ ] Create mock data generation for CLI tests
  - [ ] Add report validation utilities
  - [ ] Create configuration testing helpers
  - [ ] Add performance testing utilities
  - [ ] **Add module testing utilities and mocks**
  - [ ] **Create test utilities for modular architecture validation**

### 7.3 Interactive Report Testing Framework

- [ ] **Create enhanced interactive report testing utilities**
      (`packages/spark-app/tests/_utils/interactive_report_test.py` - UPDATE)

  - [ ] Add automated browser testing for JavaScript functionality
  - [ ] Create test scenarios for trade selection and highlighting
  - [ ] Add performance testing for large trade datasets
  - [ ] Create cross-browser compatibility test suite
  - [ ] Add accessibility testing for interactive elements
  - [ ] Create visual regression testing for chart interactions
  - [ ] **Test integration with modular reporting system**
  - [ ] **Test InteractiveReporter module functionality**

- [ ] **Add JavaScript testing infrastructure**
      (`packages/spark-app/app/backtesting/reporting/static/js/tests/` - UPDATE)
  - [ ] Add unit tests for trade selection JavaScript functions
  - [ ] Create integration tests for chart-list synchronization
  - [ ] Add performance tests for DOM manipulation with large datasets
  - [ ] Create mock data generators for JavaScript testing
  - [ ] Add test utilities for simulating user interactions
  - [ ] **Test JavaScript generation from InteractiveReporter module**

## 🏗️ **8. Documentation and Examples**

### 8.1 CLI Documentation Updates

- [ ] **Update modular CLI help and documentation** (distributed across command modules)

  - [ ] Update docstring with new strategy commands
  - [ ] Add comprehensive help text for all new options
  - [ ] Include examples for common use cases
  - [ ] Add troubleshooting section for common issues
  - [ ] Document configuration requirements and format
  - [ ] **Document new modular architecture and command organization**
  - [ ] **Add module-specific help and examples**

- [ ] **Create comprehensive CLI user guide**
      (`packages/spark-app/tests/_utils/cli/CLI_USER_GUIDE.md` - NEW FILE)
  - [ ] Document all available commands and options
  - [ ] Provide step-by-step examples for strategy backtesting
  - [ ] Include configuration setup instructions
  - [ ] Add best practices for CLI usage
  - [ ] Document performance considerations and optimization tips
  - [ ] **Add interactive report usage guide with trade selection examples**
  - [ ] **Document keyboard shortcuts and navigation controls for reports**
  - [ ] **Include troubleshooting guide for JavaScript/browser compatibility issues**
  - [ ] **Add multi-scenario testing examples and interpretation guide**
  - [ ] **Document modular architecture and extension points**
  - [ ] **Add module-by-module documentation with examples**

### 8.2 Example Configurations

- [ ] **Create enhanced example strategy configurations**
      (`packages/spark-app/tests/_utils/cli/examples/` - NEW DIR)

  - [ ] Add simple single-indicator strategy example
  - [ ] Add complex multi-indicator strategy example
  - [ ] Add multi-timeframe strategy example
  - [ ] Add different position sizing strategy examples
  - [ ] Add multi-exchange strategy examples
  - [ ] **Add examples demonstrating modular CLI usage**
  - [ ] **Add configuration examples for different module features**

- [ ] **Add enhanced CLI usage examples**
      (`packages/spark-app/tests/_utils/cli/examples/cli_examples.sh` - NEW FILE)
  - [ ] Provide shell script examples for common CLI operations
  - [ ] Add batch processing examples for multiple strategies
  - [ ] Include report generation and analysis examples
  - [ ] Document integration with external tools
  - [ ] Add automation and scripting examples
  - [ ] **Add examples using modular command structure**
  - [ ] **Document module-specific command patterns**

### 8.3 Multi-Scenario CLI Output Example

- [ ] **Document expected CLI output format for multi-scenario testing with modular architecture**

## 🔍 **9. Validation and Testing**

### 9.1 Strategy Command Testing

- [ ] **Test strategy backtesting commands**

  ```bash
  cd packages/spark-app

  # Test strategy backtesting with config
  .venv/bin/python tests/_utils/cli.py strategy eth_multi_timeframe_strategy

  # Test with real data
  .venv/bin/python tests/_utils/cli.py strategy eth_multi_timeframe_strategy --use-real-data --days 7

  # Test with overrides
  .venv/bin/python tests/_utils/cli.py strategy eth_multi_timeframe_strategy --override-timeframe 1h
  ```

- [ ] **Test multi-scenario strategy backtesting**

  ```bash
  cd packages/spark-app

  # Test full multi-scenario suite (default behavior)
  .venv/bin/python tests/_utils/cli.py strategy eth_multi_timeframe_strategy --days 30

  # Test specific scenarios only
  .venv/bin/python tests/_utils/cli.py strategy eth_multi_timeframe_strategy --scenarios "bull,bear,real" --days 30

  # Test single scenario for quick testing
  .venv/bin/python tests/_utils/cli.py strategy eth_multi_timeframe_strategy --scenario-only bull --days 30

  # Test with data export for analysis
  .venv/bin/python tests/_utils/cli.py strategy eth_multi_timeframe_strategy --days 14 --export-data
  ```

- [ ] **Test strategy comparison commands**

  ```bash
  # Compare all strategies
  .venv/bin/python tests/_utils/cli.py compare-strategies --all-strategies

  # Compare specific strategies
  .venv/bin/python tests/_utils/cli.py compare-strategies --strategy-names "strategy1,strategy2"

  # Compare strategies on same market
  .venv/bin/python tests/_utils/cli.py compare-strategies --same-market ETH-USD

  # Compare with specific scenarios for faster testing
  .venv/bin/python tests/_utils/cli.py compare-strategies --all-strategies --scenarios "bull,bear,sideways" --days 14
  ```

### 9.2 Configuration Integration Testing

- [ ] **Test config.json integration**

  ```bash
  # Test with default config
  .venv/bin/python tests/_utils/cli.py list-strategies

  # Test with custom config
  .venv/bin/python tests/_utils/cli.py --config /path/to/config.json list-strategies

  # Test config validation
  .venv/bin/python tests/_utils/cli.py validate-config
  ```

### 9.3 Report Quality Validation

- [ ] **Validate enhanced reports**

  - [ ] Ensure strategy configuration appears in reports
  - [ ] Verify position sizing information is included
  - [ ] Check that all strategy indicators are shown
  - [ ] Validate strategy comparison report accuracy
  - [ ] Test report generation with various strategy configurations

- [ ] **Validate multi-scenario reporting**

  - [ ] Verify all 8 scenarios (7 synthetic + real data) appear in reports
  - [ ] Check scenario performance comparison table accuracy
  - [ ] Validate radar/spider charts display correctly for all scenarios
  - [ ] Test scenario heatmap color coding (green=good, red=poor)
  - [ ] Verify equity curve overlays show all scenarios distinctly
  - [ ] Check scenario robustness scoring calculations
  - [ ] Validate worst-case scenario analysis identifies correct weak points
  - [ ] Test tabbed interface switching between scenarios
  - [ ] Verify scenario filtering and sorting functionality
  - [ ] Check export functionality saves correct scenario data
  - [ ] Validate statistical significance testing between scenarios
  - [ ] Test scenario correlation analysis accuracy

- [ ] **Test interactive trade selection functionality**
  - [ ] Verify trade list displays all trades with proper numbering
  - [ ] Test clicking on trades highlights correct entry/exit markers on chart
  - [ ] Validate trade details popup/sidebar shows accurate information
  - [ ] Test keyboard navigation between trades works smoothly
  - [ ] Verify trade filtering and search functionality works correctly
  - [ ] Test "zoom to trade" functionality focuses on correct timeframe
  - [ ] Validate synchronized highlighting between list and chart
  - [ ] Test hover tooltips display correct trade information
  - [ ] Verify trade sequence visualization (entry-to-exit lines) renders properly
  - [ ] Test chart zoom and pan controls work with trade highlighting
  - [ ] Validate toggle for showing/hiding trade types functions correctly
  - [ ] Test interactive features work across different browsers (Chrome, Firefox, Safari)
  - [ ] Verify mobile responsiveness of interactive trade selection
  - [ ] Test performance with large numbers of trades (100+ trades)
  - [ ] Validate JavaScript error handling and graceful degradation
  - [ ] **Test scenario-specific trade highlighting works correctly**
  - [ ] **Verify trade selection works across all scenario tabs**

## 🔍 **10. Performance and Quality Assurance**

### 10.1 Performance Testing

- [ ] **Benchmark CLI performance**
  - [ ] Measure strategy backtesting time vs indicator backtesting
  - [ ] Test performance with large numbers of strategies
  - [ ] Validate memory usage with complex strategy configurations
  - [ ] Test parallel execution performance for strategy comparisons
  - [ ] Benchmark report generation time for complex strategies

### 10.2 Quality Validation

- [ ] **Validate backtest accuracy**
  - [ ] Compare strategy backtest results with manual calculations
  - [ ] Verify position sizing calculations match configuration
  - [ ] Validate indicator calculations within strategy context
  - [ ] Test edge cases and boundary conditions
  - [ ] Ensure consistent results across multiple runs

## ✅ **Success Criteria**

Upon completion of this phase:

✅ **CLI fully integrated with config.json strategy definitions** ✅ **Strategy backtesting commands
working with real strategy configurations** ✅ **Position sizing properly integrated from strategy
configuration** ✅ **Comprehensive strategy reports with full configuration context** ✅ **Strategy
comparison functionality with detailed analysis** ✅ **Multi-scenario testing across 7 synthetic
market conditions plus real data** ✅ **Strategy robustness analysis and scenario performance
comparison** ✅ **Interactive trade selection and highlighting in reports** ✅ **Backward
compatibility maintained for existing indicator commands** ✅ **Enhanced error handling and
validation for strategy configurations** ✅ **Performance optimizations for strategy backtesting**
✅ **Comprehensive documentation and examples** ✅ **Full test coverage for new CLI functionality
(>90%)**

## 🔄 **Next Phase**

After completion, the CLI will be fully aligned with the strategy-driven architecture and ready for:

- **Phase 4**: Monitoring & Control Interface integration
- **Phase 5**: Live trading preparation with validated strategies
- **Phase 6**: Advanced strategy optimization and portfolio management

---

**Status**: 🟡 **READY TO START** **Assigned**: TBD
