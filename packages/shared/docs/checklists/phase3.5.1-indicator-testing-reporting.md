# Phase 3.5.1: Simplified Indicator Performance Reporting

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

- [ ] **Install required libraries**

  - [ ] Add to packages/spark-app/requirements.txt: `jinja2==3.1.2 plotly==5.14.1`
  - [ ] Install with: `cd packages/spark-app && pip install -r requirements.txt`

- [ ] **Create report template structure**

  - [ ] Create directory: `packages/spark-app/app/backtesting/reporting/templates`
  - [ ] Create base template: `packages/spark-app/app/backtesting/reporting/templates/base.html`
  - [ ] Create CSS file: `packages/spark-app/app/backtesting/reporting/templates/static/style.css`

- [ ] **Create report generator module**
  - [ ] Create file: `packages/spark-app/app/backtesting/reporting/generator.py`
  - [ ] Implement template loader with Jinja2
  - [ ] Add function to save HTML reports to disk
  - [ ] Add function to generate report filenames

### 3. Develop Core Report Components

- [ ] **Implement essential visualization functions**

  - [ ] Create file: `packages/spark-app/app/backtesting/reporting/visualizations.py`
  - [ ] Add function: `generate_price_chart(df, trades, filename)` using plotly
  - [ ] Add function: `generate_equity_curve(trades, filename)` using plotly
  - [ ] Add function: `generate_drawdown_chart(equity_curve, filename)` using plotly

- [ ] **Create metrics calculator**

  - [ ] Create file: `packages/spark-app/app/backtesting/reporting/metrics.py`
  - [ ] Add function: `calculate_performance_metrics(trades)` that returns dict with:
    - Win rate, profit factor, max drawdown, Sharpe ratio, total return

- [ ] **Implement report data transformer**

  - [ ] Create file: `packages/spark-app/app/backtesting/reporting/transformer.py`
  - [ ] Add function: `transform_backtest_results(results)` to prepare data for templates
  - [ ] Add function: `format_trade_list(trades)` to generate HTML table

- [ ] **Create report generator script**
  - [ ] Create file: `packages/spark-app/app/backtesting/reporting/generate_report.py`
  - [ ] Implement CLI interface to generate reports from command line
  - [ ] Add option to specify output directory

### 4. Add Comparative Analysis

- [ ] **Create multi-indicator report template**

  - [ ] Create file: `packages/spark-app/app/backtesting/reporting/templates/comparison.html`
  - [ ] Design simple table to compare key metrics between indicators

- [ ] **Implement comparison generator**
  - [ ] Add function: `generate_comparison_report(indicator_results, output_file)`
  - [ ] Add function: `create_metrics_table(indicator_results)` for HTML table generation
  - [ ] Add simple market condition classifier (bull/bear/sideways) based on price trends

### 5. Documentation

- [ ] **Create minimal documentation**
  - [ ] Add file: `packages/spark-app/app/backtesting/reporting/README.md`
  - [ ] Document command to generate reports:
    ```
    python -m packages.spark-app.app.backtesting.reporting.generate_report \
      --indicator=RSI \
      --output-dir=reports
    ```
  - [ ] Document each metric with brief explanation
  - [ ] Include sample screenshots of reports

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
