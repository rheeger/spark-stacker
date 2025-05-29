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

- [ ] **Add strategy backtesting command** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Create `@cli.command("strategy")` for strategy-specific backtesting
  - [ ] Add `--strategy-name` parameter to select from config
  - [ ] **Run multi-scenario testing by default** (all 7 synthetic scenarios + real data)
  - [ ] Add `--days` parameter to set testing duration for all scenarios
  - [ ] Add `--scenarios` parameter to select specific scenarios (e.g., "bull,bear,real")
  - [ ] Add `--scenario-only` flag to run single scenario instead of full suite
  - [ ] Add `--override-timeframe` option to temporarily change timeframe
  - [ ] Add `--override-market` option for different market testing
  - [ ] Add `--override-position-size` option for testing different sizing
  - [ ] Add `--use-real-data` flag for live data vs synthetic data (legacy compatibility)
  - [ ] Add validation that strategy exists in config
  - [ ] Add `--export-data` flag to save scenario data for external analysis

- [ ] **Add strategy comparison command** (`packages/spark-app/tests/_utils/cli.py`)
  - [ ] Create `@cli.command("compare-strategies")` for multi-strategy comparison
  - [ ] **Run all strategies through multi-scenario testing for fair comparison**
  - [ ] Add `--strategy-names` parameter for comma-separated strategy list
  - [ ] Add `--all-strategies` flag to compare all enabled strategies
  - [ ] Add `--same-market` flag to filter strategies by market
  - [ ] Add `--same-exchange` flag to filter strategies by exchange
  - [ ] Add strategy performance ranking and comparison metrics
  - [ ] **Include cross-scenario robustness scoring in comparison**
  - [ ] Add `--scenarios` parameter to limit comparison to specific scenarios

### 1.3 Enhanced List Commands

- [ ] **Update list-indicators command** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Show which strategies use each indicator
  - [ ] Display indicator parameters from config
  - [ ] Show timeframe and market context per indicator
  - [ ] Add filtering by strategy or market

- [ ] **Add list-strategies command** (`packages/spark-app/tests/_utils/cli.py`)
  - [ ] Show all strategies from config with status (enabled/disabled)
  - [ ] Display strategy details (market, exchange, timeframe, indicators)
  - [ ] Show position sizing method and parameters per strategy
  - [ ] Add filtering and sorting options (by market, exchange, performance)

## 🏗️ **2. Strategy Backtesting Engine Integration**

### 2.1 Strategy Configuration Processing

- [ ] **Create StrategyBacktestManager class**
      (`packages/spark-app/tests/_utils/strategy_backtest_manager.py` - NEW FILE)

  - [ ] Initialize with StrategyConfig object from config.json
  - [ ] Load and validate all strategy indicators from config
  - [ ] Set up position sizing based on strategy configuration
  - [ ] Configure data sources based on strategy market and exchange
  - [ ] Handle strategy-specific timeframe and data requirements
  - [ ] Add comprehensive error handling and validation

- [ ] **Integrate with existing IndicatorBacktestManager**
      (`packages/spark-app/tests/_utils/cli.py`)
  - [ ] Update CLI to use StrategyBacktestManager for strategy commands
  - [ ] Maintain IndicatorBacktestManager for indicator-only commands
  - [ ] Add factory method to choose appropriate manager based on command
  - [ ] Ensure consistent data handling between both managers

### 2.2 Position Sizing Integration

- [ ] **Add strategy position sizing support** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Load position sizing config from strategy configuration
  - [ ] Support strategy-specific position sizing overrides
  - [ ] Handle position sizing inheritance from global config
  - [ ] Add validation for position sizing parameters
  - [ ] Include position sizing details in backtest reports
  - [ ] Add CLI options to override position sizing for testing

- [ ] **Create position sizing validation utilities**
      (`packages/spark-app/tests/_utils/position_sizing_utils.py` - NEW FILE)
  - [ ] Add `validate_position_sizing_config()` function
  - [ ] Add `merge_position_sizing_config()` for strategy inheritance
  - [ ] Add `calculate_effective_position_size()` for report generation
  - [ ] Add position sizing comparison utilities for strategy comparison
  - [ ] Add position sizing impact analysis for different configurations

### 2.3 Data Management Updates

- [ ] **Enhance data fetching for strategies** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Use strategy's market and exchange for data fetching
  - [ ] Respect strategy's timeframe for data requirements
  - [ ] Add multi-timeframe data support for strategies with mixed indicator timeframes
  - [ ] Cache data efficiently for multiple strategies on same market
  - [ ] Add data validation specific to strategy requirements

- [ ] **Update synthetic data generation** (`packages/spark-app/tests/_utils/cli.py`)
  - [ ] Generate data appropriate for strategy's market and timeframe
  - [ ] Create market scenario data for strategy testing (trending, sideways, volatile)
  - [ ] Add support for multi-timeframe synthetic data generation
  - [ ] Include volume and other market data required by strategy indicators

### 2.4 Multi-Scenario Testing Framework

- [ ] **Create comprehensive market scenario generator**
      (`packages/spark-app/tests/_utils/market_scenario_generator.py` - NEW FILE)

  - [ ] Add **bull market scenario** (consistent uptrend with 60-80% up days)
  - [ ] Add **bear market scenario** (consistent downtrend with 60-80% down days)
  - [ ] Add **sideways/range-bound scenario** (oscillating within 5-10% range)
  - [ ] Add **high volatility scenario** (large daily swings, 15-25% moves)
  - [ ] Add **low volatility scenario** (minimal daily changes, <2% moves)
  - [ ] Add **choppy market scenario** (frequent direction changes, whipsaws)
  - [ ] Add **gap-heavy scenario** (frequent price gaps, simulating news events)
  - [ ] Ensure all scenarios generate data for the exact same timeframe duration

- [ ] **Add real data parallel testing** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Fetch real market data for the same symbol and timeframe duration
  - [ ] Ensure real data covers the exact same time period as synthetic scenarios
  - [ ] Add fallback to alternative time periods if recent data insufficient
  - [ ] Cache real data to avoid repeated API calls during testing
  - [ ] Add data quality validation (sufficient candles, no gaps)

- [ ] **Create scenario-based backtesting pipeline**
      (`packages/spark-app/tests/_utils/scenario_backtest_manager.py` - NEW FILE)
  - [ ] Run strategy against all synthetic market scenarios
  - [ ] Run strategy against real market data for same duration
  - [ ] Collect performance metrics for each scenario independently
  - [ ] Ensure consistent position sizing and risk parameters across scenarios
  - [ ] Add scenario labeling and metadata tracking
  - [ ] Implement parallel execution for faster scenario testing

## 🏗️ **3. Enhanced Reporting System**

### 3.1 Strategy Report Generation

- [ ] **Create comprehensive strategy reports** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Include full strategy configuration in report header
  - [ ] Show all strategy indicators and their individual performance
  - [ ] Display position sizing method and parameters used
  - [ ] Include risk management settings (stop loss, take profit, max position)
  - [ ] Add strategy-specific metadata (exchange, market, timeframe)
  - [ ] Show strategy vs individual indicator performance comparison

- [ ] **Add interactive trade analysis features** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Create interactive trade list with clickable trade entries
  - [ ] Add trade highlighting functionality when selected from list
  - [ ] Implement JavaScript-based chart interaction for trade selection
  - [ ] Add trade numbering/indexing for easy reference and navigation
  - [ ] Include trade details popup or sidebar when trade is selected
  - [ ] Add keyboard navigation (arrow keys) for trade selection
  - [ ] Implement trade filtering and search within the report
  - [ ] Add "zoom to trade" functionality to focus chart on selected trade timeframe

- [ ] **Enhance chart interactivity** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Add clickable trade markers on price charts
  - [ ] Implement hover tooltips for trade entry/exit points
  - [ ] Add trade annotation with PnL, duration, and strategy context
  - [ ] Create synchronized highlighting between trade list and chart
  - [ ] Add trade sequence visualization (connecting entry to exit with lines)
  - [ ] Implement chart zoom and pan controls for detailed trade analysis
  - [ ] Add toggle for showing/hiding different trade types (winning/losing)

- [ ] **Update HTML report templates** (`packages/spark-app/app/backtesting/reporting/`)
  - [ ] Add strategy configuration section to HTML templates
  - [ ] Display position sizing information and impact analysis
  - [ ] Show indicator breakdown within strategy context
  - [ ] Add strategy-specific charts and metrics
  - [ ] Include configuration vs actual performance comparison
  - [ ] Add strategy optimization suggestions based on results
  - [ ] **Integrate interactive JavaScript components for trade selection**
  - [ ] **Add responsive design for trade list and chart interaction**
  - [ ] **Include CSS styling for highlighted trades and selected states**

### 3.2 Strategy Comparison Reports

- [ ] **Create strategy comparison reporting** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Generate side-by-side strategy performance comparison
  - [ ] Include configuration differences (position sizing, indicators, timeframes)
  - [ ] Show relative performance metrics and rankings
  - [ ] Add risk-adjusted performance comparison (Sharpe ratio, max drawdown)
  - [ ] Include efficiency metrics (trades per day, win rate consistency)
  - [ ] Generate strategy combination analysis (portfolio effect)

- [ ] **Add strategy ranking and analysis** (`packages/spark-app/tests/_utils/cli.py`)
  - [ ] Rank strategies by multiple criteria (return, Sharpe, drawdown)
  - [ ] Analyze strategy correlation and diversification benefits
  - [ ] Identify best performing strategies by market condition
  - [ ] Generate strategy allocation recommendations
  - [ ] Add sensitivity analysis for position sizing and timeframe changes

### 3.3 Configuration Impact Analysis

- [ ] **Add configuration sensitivity analysis** (`packages/spark-app/tests/_utils/cli.py`)
  - [ ] Test strategy performance with different position sizing methods
  - [ ] Analyze impact of timeframe changes on strategy performance
  - [ ] Show effect of different indicator parameters on strategy results
  - [ ] Generate optimization suggestions for strategy configuration
  - [ ] Include risk parameter sensitivity (stop loss, take profit impact)

### 3.4 Interactive Trade Selection Technical Implementation

- [ ] **Create JavaScript modules for trade interaction**
      (`packages/spark-app/app/backtesting/reporting/static/js/` - NEW DIR)

  - [ ] Add `trade-selector.js` for trade list interaction logic
  - [ ] Create `chart-highlighter.js` for chart marker highlighting
  - [ ] Add `trade-details.js` for popup/sidebar functionality
  - [ ] Create `trade-navigation.js` for keyboard and sequence navigation
  - [ ] Add `chart-zoom.js` for zoom-to-trade functionality
  - [ ] Create `trade-filter.js` for search and filtering capabilities

- [ ] **Update chart generation for interactivity** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Add unique IDs to all trade markers for JavaScript targeting
  - [ ] Include trade metadata in chart data attributes
  - [ ] Generate trade sequence data for connecting entry/exit points
  - [ ] Add chart configuration for zoom and pan capabilities
  - [ ] Create responsive chart sizing for different screen sizes
  - [ ] Add accessibility attributes for screen readers

- [ ] **Create HTML template components**
      (`packages/spark-app/app/backtesting/reporting/templates/` - UPDATE)

  - [ ] Add interactive trade list component with click handlers
  - [ ] Create trade details sidebar/popup template
  - [ ] Add trade navigation controls (previous/next/filter)
  - [ ] Create responsive layout for chart and trade list
  - [ ] Add loading states for trade data and chart updates
  - [ ] Include error handling displays for JavaScript failures

- [ ] **Add CSS styling for interactive elements**
      (`packages/spark-app/app/backtesting/reporting/static/css/` - UPDATE)
  - [ ] Style selected trade states and hover effects
  - [ ] Add highlighting styles for chart markers
  - [ ] Create responsive design for mobile/tablet viewing
  - [ ] Add animation styles for smooth transitions
  - [ ] Style trade details popup/sidebar
  - [ ] Add accessibility-friendly focus indicators

### 3.5 Multi-Scenario Performance Reporting

- [ ] **Create comprehensive scenario comparison reports**
      (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Generate side-by-side performance comparison across all market scenarios
  - [ ] Include **Bull Market**, **Bear Market**, **Sideways**, **High Vol**, **Low Vol**,
        **Choppy**, **Gap-Heavy**, and **Real Data** results
  - [ ] Display unified metrics table with win rate, total return, Sharpe ratio, max drawdown for
        each scenario
  - [ ] Add scenario ranking by different performance criteria
  - [ ] Include statistical significance testing between scenarios
  - [ ] Generate scenario robustness score based on consistency across conditions

- [ ] **Add visual scenario performance comparison** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Create radar/spider charts showing strategy performance across all scenarios
  - [ ] Add scenario performance heatmap (green=good, red=poor performance)
  - [ ] Generate overlay charts showing equity curves for all scenarios
  - [ ] Create trade distribution charts by scenario type
  - [ ] Add scenario-specific trade highlighting in interactive charts
  - [ ] Include market condition timeline showing when each scenario type occurred

- [ ] **Generate strategy robustness analysis**
      (`packages/spark-app/tests/_utils/strategy_robustness_analyzer.py` - NEW FILE)

  - [ ] Calculate **consistency score** (low variance across scenarios)
  - [ ] Compute **adaptability score** (performance in diverse conditions)
  - [ ] Generate **risk-adjusted robustness** (Sharpe ratio consistency)
  - [ ] Add **worst-case scenario analysis** (performance in hardest conditions)
  - [ ] Calculate **scenario correlation** (which scenarios strategy struggles with)
  - [ ] Generate **optimization recommendations** based on weak scenarios

- [ ] **Update HTML templates for multi-scenario display**
      (`packages/spark-app/app/backtesting/reporting/templates/`)
  - [ ] Add tabbed interface for switching between scenarios
  - [ ] Create scenario comparison dashboard with key metrics
  - [ ] Add expandable sections for detailed scenario analysis
  - [ ] Include scenario filtering and sorting controls
  - [ ] Add export functionality for scenario performance data
  - [ ] Create mobile-responsive design for scenario navigation

## 🏗️ **4. Validation and Error Handling**

### 4.1 Configuration Validation

- [ ] **Add comprehensive config validation** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Validate all strategy configurations before backtesting
  - [ ] Check that all strategy indicators exist in config
  - [ ] Validate position sizing configurations
  - [ ] Ensure market symbols and exchanges are valid
  - [ ] Check timeframe compatibility between strategy and indicators
  - [ ] Add detailed error messages for configuration issues

- [ ] **Create config validation utilities**
      (`packages/spark-app/tests/_utils/config_validation.py` - NEW FILE)
  - [ ] Add `validate_strategy_indicator_consistency()` function
  - [ ] Add `validate_position_sizing_config()` function
  - [ ] Add `validate_market_exchange_compatibility()` function
  - [ ] Add `validate_timeframe_consistency()` function
  - [ ] Add configuration repair suggestions for common issues

### 4.2 Enhanced Error Handling

- [ ] **Improve CLI error handling** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Add specific error handling for strategy configuration issues
  - [ ] Provide helpful error messages with fix suggestions
  - [ ] Add graceful degradation for partial configuration issues
  - [ ] Include error logging with sufficient context for debugging
  - [ ] Add retry mechanisms for data fetching failures

- [ ] **Add pre-flight checks** (`packages/spark-app/tests/_utils/cli.py`)
  - [ ] Validate data availability for strategy requirements
  - [ ] Check exchange connectivity for real data commands
  - [ ] Verify indicator factory can create all required indicators
  - [ ] Test position sizer creation with strategy configuration
  - [ ] Validate output directory permissions and disk space

## 🏗️ **5. Backward Compatibility and Migration**

### 5.1 Legacy Command Support

- [ ] **Maintain existing indicator commands** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Keep `demo`, `real-data`, `compare`, `compare-popular` commands working
  - [ ] Add deprecation warnings for indicator-only commands
  - [ ] Provide migration suggestions to strategy-based commands
  - [ ] Ensure existing scripts and workflows continue working
  - [ ] Add flag to disable deprecation warnings if needed

- [ ] **Add migration utilities** (`packages/spark-app/tests/_utils/cli.py`)
  - [ ] Add `--suggest-strategy` flag to indicator commands
  - [ ] Show which strategies use the specified indicator
  - [ ] Provide example strategy commands for equivalent functionality
  - [ ] Add automatic strategy creation suggestions for common patterns

### 5.2 Configuration Migration Support

- [ ] **Add config file migration** (`packages/spark-app/tests/_utils/cli.py`)
  - [ ] Add `--migrate-config` command to update old config files
  - [ ] Validate config file version and suggest updates
  - [ ] Add config file format conversion utilities
  - [ ] Provide config validation and repair suggestions
  - [ ] Generate example strategy configurations for common use cases

## 🏗️ **6. Performance and Optimization**

### 6.1 Caching and Performance

- [ ] **Optimize data caching for strategies** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Cache market data across multiple strategy tests
  - [ ] Reuse indicator calculations for strategies sharing indicators
  - [ ] Optimize multi-timeframe data handling
  - [ ] Add progress indicators for long-running strategy backtests
  - [ ] Implement parallel strategy execution for comparisons

- [ ] **Add performance monitoring** (`packages/spark-app/tests/_utils/cli.py`)
  - [ ] Measure and report backtest execution time
  - [ ] Track memory usage for large strategy comparisons
  - [ ] Add performance benchmarks for strategy vs indicator testing
  - [ ] Include performance metrics in CLI output
  - [ ] Add performance optimization suggestions

### 6.2 Resource Management

- [ ] **Improve resource cleanup** (`packages/spark-app/tests/_utils/cli.py`)
  - [ ] Ensure proper cleanup after strategy backtests
  - [ ] Add timeout handling for long-running operations
  - [ ] Implement graceful shutdown for interrupted operations
  - [ ] Add disk space management for large report generation
  - [ ] Clean up temporary files and cached data appropriately

## 🧪 **7. Testing Infrastructure**

### 7.1 CLI Testing Framework

- [ ] **Create CLI testing suite** (`packages/spark-app/tests/_utils/test_cli.py` - NEW FILE)

  - [ ] Test all new strategy commands with mock data
  - [ ] Test configuration loading and validation
  - [ ] Test strategy backtesting with various configurations
  - [ ] Test error handling and edge cases
  - [ ] Test backward compatibility with existing commands
  - [ ] Test report generation and file output

- [ ] **Add integration tests**
      (`packages/spark-app/tests/backtesting/integration/test_cli_integration.py` - NEW FILE)
  - [ ] Test CLI with real config.json file
  - [ ] Test end-to-end strategy backtesting workflow
  - [ ] Test strategy comparison functionality
  - [ ] Test configuration migration and validation
  - [ ] Test CLI performance with multiple strategies

### 7.2 Test Data and Fixtures

- [ ] **Create CLI test fixtures** (`packages/spark-app/tests/_fixtures/cli_fixtures.py` - NEW FILE)

  - [ ] Create test strategy configurations
  - [ ] Add test market data for various scenarios
  - [ ] Create mock backtesting results for testing
  - [ ] Add test position sizing configurations
  - [ ] Create test error scenarios and edge cases

- [ ] **Add CLI test utilities** (`packages/spark-app/tests/_helpers/cli_test_helpers.py` - NEW
      FILE)
  - [ ] Add CLI command testing utilities
  - [ ] Create mock data generation for CLI tests
  - [ ] Add report validation utilities
  - [ ] Create configuration testing helpers
  - [ ] Add performance testing utilities

### 7.3 Interactive Report Testing Framework

- [ ] **Create interactive report testing utilities**
      (`packages/spark-app/tests/_utils/interactive_report_test.py` - NEW FILE)

  - [ ] Add automated browser testing for JavaScript functionality
  - [ ] Create test scenarios for trade selection and highlighting
  - [ ] Add performance testing for large trade datasets
  - [ ] Create cross-browser compatibility test suite
  - [ ] Add accessibility testing for interactive elements
  - [ ] Create visual regression testing for chart interactions

- [ ] **Add JavaScript testing infrastructure**
      (`packages/spark-app/app/backtesting/reporting/static/js/tests/` - NEW DIR)
  - [ ] Add unit tests for trade selection JavaScript functions
  - [ ] Create integration tests for chart-list synchronization
  - [ ] Add performance tests for DOM manipulation with large datasets
  - [ ] Create mock data generators for JavaScript testing
  - [ ] Add test utilities for simulating user interactions

## 🏗️ **8. Documentation and Examples**

### 8.1 CLI Documentation Updates

- [ ] **Update CLI help and documentation** (`packages/spark-app/tests/_utils/cli.py`)

  - [ ] Update docstring with new strategy commands
  - [ ] Add comprehensive help text for all new options
  - [ ] Include examples for common use cases
  - [ ] Add troubleshooting section for common issues
  - [ ] Document configuration requirements and format

- [ ] **Create CLI user guide** (`packages/spark-app/tests/_utils/CLI_USER_GUIDE.md` - NEW FILE)
  - [ ] Document all available commands and options
  - [ ] Provide step-by-step examples for strategy backtesting
  - [ ] Include configuration setup instructions
  - [ ] Add best practices for CLI usage
  - [ ] Document performance considerations and optimization tips
  - [ ] **Add interactive report usage guide with trade selection examples**
  - [ ] **Document keyboard shortcuts and navigation controls for reports**
  - [ ] **Include troubleshooting guide for JavaScript/browser compatibility issues**
  - [ ] **Add multi-scenario testing examples and interpretation guide**

### 8.2 Example Configurations

- [ ] **Create example strategy configurations** (`packages/spark-app/tests/_utils/examples/` - NEW
      DIR)

  - [ ] Add simple single-indicator strategy example
  - [ ] Add complex multi-indicator strategy example
  - [ ] Add multi-timeframe strategy example
  - [ ] Add different position sizing strategy examples
  - [ ] Add multi-exchange strategy examples

- [ ] **Add CLI usage examples** (`packages/spark-app/tests/_utils/examples/cli_examples.sh` - NEW
      FILE)
  - [ ] Provide shell script examples for common CLI operations
  - [ ] Add batch processing examples for multiple strategies
  - [ ] Include report generation and analysis examples
  - [ ] Document integration with external tools
  - [ ] Add automation and scripting examples

### 8.3 Multi-Scenario CLI Output Example

- [ ] **Document expected CLI output format for multi-scenario testing**

  ```bash
  $ python cli.py strategy eth_multi_timeframe_strategy --days 30

  🚀 Running multi-scenario backtest for eth_multi_timeframe_strategy (30 days)
  📊 Testing across 8 market scenarios...

  [1/8] 📈 Bull Market Scenario    ✅ Complete (45 trades, +12.5% return)
  [2/8] 📉 Bear Market Scenario    ✅ Complete (32 trades, -3.2% return)
  [3/8] 📊 Sideways Scenario       ✅ Complete (28 trades, +2.1% return)
  [4/8] 🔥 High Volatility         ✅ Complete (67 trades, +8.9% return)
  [5/8] 😴 Low Volatility          ✅ Complete (15 trades, +1.4% return)
  [6/8] ⚡ Choppy Market          ✅ Complete (89 trades, -1.8% return)
  [7/8] 📈💥 Gap-Heavy Scenario    ✅ Complete (42 trades, +5.3% return)
  [8/8] 🌐 Real Market Data        ✅ Complete (38 trades, +4.7% return)

  📋 Scenario Performance Summary:
  ┌─────────────────┬────────┬─────────┬──────────┬────────────┬─────────┐
  │ Scenario        │ Trades │ Return  │ Win Rate │ Max DD     │ Sharpe  │
  ├─────────────────┼────────┼─────────┼──────────┼────────────┼─────────┤
  │ Bull Market     │   45   │ +12.5%  │  67.2%   │   -5.3%    │  1.43   │
  │ Bear Market     │   32   │  -3.2%  │  43.8%   │  -12.1%    │ -0.21   │
  │ Sideways        │   28   │  +2.1%  │  57.1%   │   -4.8%    │  0.34   │
  │ High Volatility │   67   │  +8.9%  │  58.2%   │   -8.7%    │  0.89   │
  │ Low Volatility  │   15   │  +1.4%  │  60.0%   │   -2.1%    │  0.52   │
  │ Choppy Market   │   89   │  -1.8%  │  44.9%   │   -9.4%    │ -0.15   │
  │ Gap-Heavy       │   42   │  +5.3%  │  61.9%   │   -6.2%    │  0.71   │
  │ Real Data       │   38   │  +4.7%  │  55.3%   │   -7.1%    │  0.68   │
  └─────────────────┴────────┴─────────┴──────────┴────────────┴─────────┘

  🎯 Strategy Robustness Analysis:
  • Consistency Score: 72/100 (Good - performs well in most conditions)
  • Adaptability Score: 68/100 (Fair - struggles in choppy/bear markets)
  • Risk-Adjusted Robustness: 75/100 (Good - maintains positive Sharpe in most scenarios)
  • Worst-Case Scenario: Bear Market (-3.2% return, -12.1% max drawdown)
  • Best Scenario: Bull Market (+12.5% return, 1.43 Sharpe ratio)

  📈 HTML Report: /path/to/report.html (opens automatically)
  💾 Scenario Data: /path/to/scenario_data.json (use --export-data flag)

  ✨ Strategy shows strong bull market performance but needs improvement for bear/choppy conditions
  ```

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
