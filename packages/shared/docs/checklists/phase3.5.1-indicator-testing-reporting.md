Run cd packages/spark-app ImportError while loading conftest
'/home/runner/work/spark-stacker/spark-stacker/packages/spark-app/tests/conftest.py'.
tests/conftest.py:77: in <module> from app.core.trading_engine import TradingEngine
app/core/trading_engine.py:21: in <module> from app.utils.config import AppConfig as Config E
ModuleNotFoundError: No module named 'app.utils.config' Initializing logging_setup.py app_dir:
/home/runner/work/spark-stacker/spark-stacker/packages/spark-app/app Logs directory path:
/home/runner/work/spark-stacker/spark-stacker/packages/spark-app/logs In test mode - using dummy log
directory: /tmp/test_logs Error: Process completed with exit code 4.# Phase 3.5.1: Simplified
Indicator Performance Reporting

## Overview

This phase focuses on building clear, digestible static HTML reports for our trading indicators
using the existing spark-app backtesting suite. We'll create Python-based templates that generate
static HTML reports with key performance metrics and visualizations. The goal is to provide simple,
accessible reports that highlight indicator performance without the complexity of a full interactive
application.

## Prerequisites

- Existing indicator test harness (already implemented)
- Backtesting engine capabilities (already implemented)
- Basic results storage in JSON/Markdown (already implemented)
- Functional exchange connectors (Hyperliquid, etc.)

## Tasks

### 1. Clean Up Current Implementation

- [x] **Remove unnecessary NX packages**
  - [x] Remove backtesting-ui React application package
  - [x] Remove backtesting-ui-e2e package
  - [x] Clean up dependencies in root package.json
  - [x] Update NX configuration to reflect removed packages

### 2. Set Up Python-Based Report Generation

- [x] **Install required libraries**

  - [x] Add to packages/spark-app/requirements.txt: `jinja2==3.1.2 plotly==5.14.1`
  - [x] Install with: `cd packages/spark-app && pip install -r requirements.txt`

- [x] **Create report template structure**

  - [x] Create directory: `packages/spark-app/app/backtesting/reporting/templates`
  - [x] Create base template: `packages/spark-app/app/backtesting/reporting/templates/base.html`
  - [x] Create CSS file: `packages/spark-app/app/backtesting/reporting/templates/static/style.css`

- [x] **Create report generator module**
  - [x] Create file: `packages/spark-app/app/backtesting/reporting/generator.py`
  - [x] Implement template loader with Jinja2
  - [x] Add function to save HTML reports to disk
  - [x] Add function to generate report filenames

### 3. Develop Core Report Components

- [x] **Implement essential visualization functions**

  - [x] Create file: `packages/spark-app/app/backtesting/reporting/visualizations.py`
  - [x] Add function: `generate_price_chart(df, trades, filename)` using plotly
  - [x] Add function: `generate_equity_curve(trades, filename)` using plotly
  - [x] Add function: `generate_drawdown_chart(equity_curve, filename)` using plotly

- [x] **Create metrics calculator**

  - [x] Create file: `packages/spark-app/app/backtesting/reporting/metrics.py`
  - [x] Add function: `calculate_performance_metrics(trades)` that returns dict with:
    - Win rate, profit factor, max drawdown, Sharpe ratio, total return

- [x] **Implement report data transformer**

  - [x] Create file: `packages/spark-app/app/backtesting/reporting/transformer.py`
  - [x] Add function: `transform_backtest_results(results)` to prepare data for templates
  - [x] Add function: `format_trade_list(trades)` to generate HTML table

- [x] **Create report generator script**
  - [x] Create file: `packages/spark-app/app/backtesting/reporting/generate_report.py`
  - [x] Implement CLI interface to generate reports from command line
  - [x] Add option to specify output directory

### 4. Back-Testing Suite Refactor (Prerequisite)

> **Why this block?** The remaining report-generation work is blocked by structural issues in the
> current back-testing/indicator validation layer. The following incremental checkpoints (adapted
> from the audit in `docs/retros/5-17-backtesting-suite-audit.md`) must be completed **before**
> resuming new reporting features. Each sub-section is a bite-sized PR/commit.

#### 4.0 Repository Hygiene & Baseline Clean-up

- [x] **4.0-A** Add `tests/test_results/` and `*.png` patterns to `.gitignore`
- [x] **4.0-B** Purge committed artefacts (`git rm -r tests/test_results`)
- [x] **4.0-C** Add `make clean-results` target that deletes local artefacts
- [x] **4.0-D** Generate baseline coverage report (`pytest --cov=app && coverage html`)
- [x] **4.0-E** Set up a **local quick-test** target (e.g., `make test-quick`) that runs
      `pytest -m "not slow" --cov=app` on Python 3.11 and prints a concise summary in the terminal.

#### 4.1 Foundational Fixtures

- [x] **4.1-A** Create `tests/conftest.py` with shared fixtures (`price_dataframe`, `temp_csv_dir`,
      `backtest_env`, `results_dir`) using deterministic seed
- [x] **4.1-B** Add typing stubs `tests/conftest.pyi` for IDE support
- [x] **4.1-C** Refactor `tests/unit/test_backtest_engine.py` to use new fixtures
- [x] **4.1-D** Ensure `pytest -q` passes and count of `TemporaryDirectory` usages drops (track via
      `git grep`)

#### 4.2 Single Synthetic-Data Generator

- [x] **4.2-A** Implement `tests/_helpers/data_factory.py::make_price_dataframe`
- [x] **4.2-B** Wire the `price_dataframe` fixture to the factory (`pattern="trend"`)
- [x] **4.2-C** Add deterministic output unit test `tests/backtesting/unit/test_data_factory.py`

#### 4.3 Directory Realignment

- [x] **4.3-A** Create `tests/backtesting/{unit,integration,simulation,regression}` and
      `tests/indicators/unit` directories (with `__init__.py`)
- [x] **4.3-B** Move `tests/unit/test_backtest_engine.py` to `tests/backtesting/unit/`; fix imports
- [x] **4.3-C** Update `pytest.ini` (`testpaths = tests`, declare `slow` marker)
- [x] **4.3-D** Migrate remaining test files folder-by-folder (one PR per folder)

#### 4.4 Artefact Stewardship

- [x] **4.4-A** Extend `IndicatorBacktestManager` & integration tests to accept `output_path`
      parameter (default tmp fixture) and return generated paths
- [x] **4.4-B** Refactor `tests/indicators/test_harness.py` to use `results_dir` fixture; remove
      hard-coded paths
- [x] **4.4-C** Add README section "Viewing local artefacts" explaining tmp paths

#### 4.5 Script ➜ CLI Consolidation

- [x] **4.5-A** Create `tests/_utils/cli.py` with `click` command `backtest`
- [x] **4.5-B** Add integration smoke test invoking CLI via `subprocess.run`
- [x] **4.5-C** Replace `run_eth_macd_backtest.py` with `cli demo-macd` sub-command
- [x] **4.5-D** Remove orphan scripts; update docs to point to CLI

#### 4.6 Automated Report Verification

- [x] **4.6-A** Use `backtest_env` to run one-trade backtest yielding result object
- [x] **4.6-B** Generate HTML report into `results_dir`
- [x] **4.6-C** Parse HTML via BeautifulSoup; assert key metrics present
- [x] **4.6-D** Verify linked chart files exist on disk

#### 4.7 Indicator Onboarding Template

- [x] **4.7-A** Implement `scripts/new_indicator.sh <Name>` scaffolder
- [x] **4.7-B** Auto-insert import into `IndicatorFactory.register_defaults()`
- [x] **4.7-C** Update CONTRIBUTING.md with onboarding steps

#### 4.8 Test-Suite Strategy

- [x] **4.8-A** Ensure the **quick-test run** (`pytest -m "not slow"`) completes in < 3 minutes on a
      typical dev laptop.
- [x] **4.8-B** Document an extended test target (`pytest -m slow --cov=app`) that generates an
      updated HTML coverage report in `_htmlcov/`.
- [x] **4.8-C** Add a note in the README reminding contributors to execute `make test-quick` before
      pushing changes.

---

_Once **all** 4.x checkpoints are ✅ the original feature-work can resume._

### 5. Add Comparative Analysis (BLOCKED — awaits 4.x)

_No action until Back-Testing Suite Refactor completes._

- [ ] **Create multi-indicator report template**
  - [ ] `packages/spark-app/app/backtesting/reporting/templates/comparison.html`
  - [ ] Design comparison table for key metrics
- [ ] **Implement comparison generator**
  - [ ] `generate_comparison_report(indicator_results, output_file)`
  - [ ] `create_metrics_table(indicator_results)`
  - [ ] Market condition classifier (bull/bear/sideways)

### 6. Documentation (BLOCKED — awaits 4.x)

- [ ] **Create minimal documentation**
  - [ ] `packages/spark-app/app/backtesting/reporting/README.md`
  - [ ] Document CLI usage / metrics / screenshots

## Validation Criteria

- [ ] **Report generation works correctly**

  - [ ] Reports generate without errors from backtesting results
  - [ ] Visualizations accurately represent the data
  - [ ] Reports display correctly in modern browsers
  - [ ] Generation process is documented and repeatable

- [ ] **Reports contain all essential information**
  - [ ] Key performance metrics are clearly presented
  - [ ] Charts are readable and properly labeled
  - [ ] Trade list provides useful filtering capabilities
  - [ ] Reports are usable without interactive features

## Expected Outcomes

1. [ ] Python-based static HTML report generation system
2. [ ] Clean, readable reports with essential metrics and visualizations
3. [ ] Ability to compare indicators using simple side-by-side reports
4. [ ] Documentation for generating and interpreting reports

## Deliverables

1. [ ] Python module for report generation in spark-app package
2. [ ] HTML/CSS templates for standard reports
3. [ ] Sample reports for all current indicators using real data
4. [ ] User guide for generating and interpreting reports

## Next Steps

After completing the reporting system:

1. 🔜 Use reports to select top 3 most predictable indicator strategies for live testing
2. 🔜 Consider automating regular report generation
3. 🔜 Evaluate the need for more advanced visualizations based on user feedback
