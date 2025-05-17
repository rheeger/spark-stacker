# Phase 3.5.1: Indicator Backtesting and Performance Reporting

## Overview

This phase focuses on evaluating our existing trading indicators against recent Hyperliquid market
data to identify highly predictable trading signals. We will leverage our existing infrastructure to
backtest all indicators on 1-minute candles and generate comprehensive reports highlighting
predictability metrics rather than just returns. The goal is to identify indicators or combinations
that produce consistent, reliable signals suitable for high-leverage trading with tight stop losses.

## Prerequisites

- Functional exchange connectors to pull Hyperliquid historical data
- Existing indicators in the `app/indicators` directory
- Basic backtesting capabilities

## Tasks

### 1. Create Streamlined Backtesting Framework for Indicator Evaluation

- [ ] **Set up standardized testing environment**

  - [ ] Create a script to pull the last 1000 1-minute candles from Hyperliquid
  - [ ] Set up config to use production environment variables
  - [ ] Build a simple runner to test each indicator with identical market data
  - [ ] Implement parallel processing to evaluate all indicators efficiently

- [ ] **Implement practical signal evaluation metrics**
  - [ ] Add win rate calculation (% of profitable trades)
  - [ ] Calculate average win/loss ratio
  - [ ] Measure signal consistency (false positives vs. true positives)
  - [ ] Track maximum consecutive wins/losses
  - [ ] Add drawdown metrics for risk assessment
  - [ ] Implement stop-loss simulation to measure effectiveness

### 2. Develop Trade Predictability Analysis

- [ ] **Build entry/exit precision metrics**

  - [ ] Measure average price movement after signal before reversal
  - [ ] Calculate signal-to-noise ratio for each indicator
  - [ ] Track time-to-profit metrics (how quickly signals become profitable)
  - [ ] Implement "perfect hindsight" comparison (how close to optimal entry/exit)
  - [ ] Create false signal detection and statistics

- [ ] **Implement practical performance metrics**
  - [ ] Calculate risk-adjusted returns (Sharpe, Sortino ratios)
  - [ ] Measure average trade duration
  - [ ] Track maximum adverse excursion (MAE) to optimize stop-loss placement
  - [ ] Calculate position sizing effectiveness based on signal strength
  - [ ] Implement volatility-adjusted performance metrics

### 3. Create Visual HTML Reporting System

- [ ] **Build comprehensive HTML reports**

  - [ ] Create an enhanced version of the existing view_plots.html
  - [ ] Add interactive price charts with entry/exit points
  - [ ] Include performance metrics dashboard at the top
  - [ ] Show trade journal with all entries/exits and reasons
  - [ ] Add equity curve with drawdown visualization
  - [ ] Create comparison view for multiple indicators

- [ ] **Implement key predictability visualizations**
  - [ ] Add distribution chart of winning vs. losing trades
  - [ ] Create signal reliability heatmap (time of day, market conditions)
  - [ ] Show false positive/negative rate visualization
  - [ ] Add stop-loss effectiveness visualization
  - [ ] Include chart of consecutive wins/losses sequences

### 4. Develop Multi-Indicator Strategy Testing

- [ ] **Create simple indicator combination framework**

  - [ ] Implement basic signal confirmation logic (require multiple indicators to agree)
  - [ ] Add weighted signal approach based on individual indicator reliability
  - [ ] Create simple voting mechanism for entry/exit decisions
  - [ ] Test basic AND/OR logic combinations of top-performing indicators
  - [ ] Develop filter approach (one indicator triggers, another confirms)

- [ ] **Build comparative reporting for strategies**
  - [ ] Create head-to-head comparison view for different strategies
  - [ ] Add combined metrics dashboard for multi-indicator strategies
  - [ ] Implement correlation analysis between indicators
  - [ ] Show win rate improvement from combining indicators
  - [ ] Create predictability improvement visualization

## Validation Criteria

- [ ] **Practical backtest validation**

  - [ ] All indicators can be backtested against the same Hyperliquid dataset
  - [ ] Win rates and performance metrics are accurately calculated
  - [ ] HTML reports clearly show entry/exit points with reasons
  - [ ] Stop-loss simulation shows realistic results

- [ ] **Predictability validation**
  - [ ] Identified indicators show consistent signals with minimal false positives
  - [ ] Win/loss ratio and consecutive win/loss metrics show reliability
  - [ ] Combined indicator strategies show improved predictability over single indicators
  - [ ] Reports clearly highlight most predictable indicators/strategies

## Expected Outcomes

1. [ ] Complete backtests of all indicators against recent Hyperliquid 1-minute data
2. [ ] Clear identification of the most predictable indicators (not just highest return)
3. [ ] Detailed HTML reports showing performance and predictability metrics
4. [ ] Validated multi-indicator strategies with improved reliability
5. [ ] Identification of optimal stop-loss and position sizing for selected strategies

## Next Steps

After completing testing and reporting:

1. 🔜 Select top 3 most predictable indicator strategies for live testing
2. 🔜 Implement the selected strategies with proper risk management
3. 🔜 Set up real-time performance monitoring for deployed strategies
4. 🔜 Create periodic re-evaluation process to ensure continued effectiveness
