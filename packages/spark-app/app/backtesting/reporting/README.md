# Backtesting Reporting Module

The backtesting reporting module provides static HTML report generation for trading indicator performance analysis. It creates professional, easy-to-read reports with key performance metrics, visualizations, and comparison capabilities.

## Overview

This module generates two types of reports:

1. **Individual Indicator Reports** - Detailed analysis of a single indicator's performance
2. **Comparison Reports** - Side-by-side comparison of multiple indicators

## Features

- **Static HTML Reports** - Self-contained reports that work in any web browser
- **Performance Metrics** - Win rate, profit factor, Sharpe ratio, max drawdown, and more
- **Interactive Charts** - Price charts, equity curves, and drawdown analysis using Plotly
- **Market Condition Analysis** - Automatic classification of bull/bear/sideways market periods
- **Trade Filtering** - Filter trade lists by winners/losers
- **Responsive Design** - Reports look good on desktop and mobile devices

## Quick Start

### Generate a Single Indicator Report

```python
from app.backtesting.reporting.generator import generate_indicator_report

# Example indicator results from backtesting
indicator_results = {
    "indicator_name": "RSI",
    "market": "ETH/USD",
    "timeframe": "4h",
    "start_date": "2024-01-01",
    "end_date": "2024-12-01",
    "metrics": {
        "win_rate": 65.5,
        "profit_factor": 1.45,
        "total_return": 15.2,
        "max_drawdown": 8.3,
        "sharpe": 1.12,
        "total_trades": 24
    },
    "trades": [
        # List of trade dictionaries
    ]
}

# Chart paths (generated separately using visualizations.py)
charts = {
    "price_chart": "./charts/rsi_price_chart.html",
    "equity_curve": "./charts/rsi_equity_curve.html",
    "drawdown_chart": "./charts/rsi_drawdown.html"
}

# Generate the report
report_path = generate_indicator_report(
    indicator_results=indicator_results,
    charts=charts,
    output_dir="./reports",
    indicator_config={
        "type": "rsi",
        "parameters": {
            "period": 14,
            "overbought": 70,
            "oversold": 30
        }
    }
)

print(f"Report generated: {report_path}")
```

### Generate a Comparison Report

```python
from app.backtesting.reporting.generator import generate_comparison_report

# Results from multiple indicators
indicator_results = [
    {
        "indicator_name": "RSI",
        "market": "ETH/USD",
        "timeframe": "4h",
        "metrics": {"win_rate": 65.5, "total_return": 15.2, ...},
        "trades": [...]
    },
    {
        "indicator_name": "MACD",
        "market": "ETH/USD",
        "timeframe": "4h",
        "metrics": {"win_rate": 58.3, "total_return": 12.8, ...},
        "trades": [...]
    }
]

# Optional: Price data for market condition analysis
price_data = [3500, 3520, 3480, 3510, ...]  # List of closing prices

# Generate comparison report
report_path = generate_comparison_report(
    indicator_results=indicator_results,
    output_dir="./reports",
    market_price_data=price_data
)

print(f"Comparison report generated: {report_path}")
```

## Command Line Interface

The reporting module can be used via the CLI for quick report generation:

```bash
# Navigate to the spark-app directory
cd packages/spark-app

# Generate a demo report using the CLI
.venv/bin/python -m tests._utils.cli backtest --indicator rsi --generate-report

# Run a comparison across multiple indicators
.venv/bin/python -m tests._utils.cli compare-indicators --indicators rsi,macd,bb --output-dir ./reports
```

## Performance Metrics Explained

### Core Metrics

| Metric            | Description                         | Good Value | Calculation                                           |
| ----------------- | ----------------------------------- | ---------- | ----------------------------------------------------- |
| **Win Rate**      | Percentage of profitable trades     | > 60%      | (Winning Trades / Total Trades) × 100                 |
| **Profit Factor** | Ratio of gross profit to gross loss | > 1.5      | Gross Profit / Gross Loss                             |
| **Total Return**  | Overall percentage return           | > 10%      | ((Final Value - Initial Value) / Initial Value) × 100 |
| **Max Drawdown**  | Largest peak-to-trough decline      | < 15%      | Max((Peak - Trough) / Peak)                           |
| **Sharpe Ratio**  | Risk-adjusted return                | > 1.0      | (Return - Risk Free Rate) / Standard Deviation        |
| **Total Trades**  | Number of completed round trips     | > 20       | Count of buy-sell pairs                               |

### Advanced Metrics

- **Average Trade Return** - Mean profit/loss per trade
- **Best Trade** - Largest single trade profit
- **Worst Trade** - Largest single trade loss
- **Consecutive Wins** - Longest winning streak
- **Consecutive Losses** - Longest losing streak

## Market Condition Classification

The reporting module automatically classifies market conditions based on price movement:

### Classification Rules

- **Bull Market** - Periods with > 2% daily gains
- **Bear Market** - Periods with > 2% daily losses
- **Sideways Market** - All other periods

### Output

```python
{
    "description": "Strong upward trend with 1.23% volatility",
    "periods": [
        {"type": "bull", "label": "Bull Market", "percentage": 45.2},
        {"type": "sideways", "label": "Sideways Market", "percentage": 32.1},
        {"type": "bear", "label": "Bear Market", "percentage": 22.7}
    ],
    "primary_condition": "bull",
    "total_return": 0.152,
    "volatility": 0.0123
}
```

## File Structure

```
app/backtesting/reporting/
├── README.md                    # This documentation
├── generator.py                 # Main report generation logic
├── metrics.py                   # Performance metrics calculations
├── transformer.py               # Data transformation utilities
├── visualizations.py            # Chart generation with Plotly
├── generate_report.py           # CLI script for report generation
└── templates/
    ├── base.html               # Individual indicator report template
    ├── comparison.html         # Multi-indicator comparison template
    └── static/
        └── style.css           # CSS styling for reports
```

## Output Structure

Generated reports are saved with the following structure:

```
reports/
├── static/
│   └── style.css               # Copied CSS files
├── images/                     # Chart images (if using static images)
│   ├── rsi_price_chart.png
│   └── rsi_equity_curve.png
├── charts/                     # Interactive chart HTML files
│   ├── rsi_price_chart.html
│   └── rsi_equity_curve.html
├── rsi_eth-usd_4h_2024-12-17.html          # Individual reports
├── macd_eth-usd_4h_2024-12-17.html
└── indicator_comparison_2024-12-17.html     # Comparison reports
```

## Chart Types

### Price Chart with Trades

- Shows historical price data with candlesticks
- Overlays buy/sell signals from the indicator
- Includes volume data if available
- Interactive zoom and pan capabilities

### Equity Curve

- Displays account balance over time
- Shows the cumulative effect of all trades
- Highlights winning and losing periods
- Includes benchmark comparison if provided

### Drawdown Chart

- Visualizes portfolio drawdowns over time
- Shows both absolute and percentage drawdowns
- Highlights maximum drawdown periods
- Useful for risk assessment

## Configuration Options

### Report Generator Options

```python
generator = ReportGenerator(
    output_dir="./custom_reports",     # Custom output directory
)

# Custom filename generation
filename = generator.generate_report_filename(
    indicator_name="Custom RSI",
    market="BTC/USDT",
    timeframe="1h"
)
# Output: custom_rsi_btc-usdt_1h_2024-12-17.html
```

### Template Customization

The HTML templates use Jinja2 templating and can be customized:

- Modify `templates/base.html` for individual reports
- Modify `templates/comparison.html` for comparison reports
- Update `templates/static/style.css` for styling changes

## Error Handling

The module includes comprehensive error handling:

```python
try:
    report_path = generate_indicator_report(results, charts)
except ValueError as e:
    print(f"Invalid data provided: {e}")
except FileNotFoundError as e:
    print(f"Required files not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

### Data Preparation

1. **Ensure Complete Data** - Include all required fields in results dictionaries
2. **Validate Metrics** - Check that metrics are calculated correctly before reporting
3. **Generate Charts First** - Create all visualization files before generating reports
4. **Use Consistent Naming** - Follow the standard naming conventions for files

### Performance Optimization

1. **Batch Processing** - Generate multiple reports in a single session
2. **Reuse Charts** - Pre-generate charts and reuse across reports when possible
3. **Cache Results** - Store intermediate results to avoid recalculation

### Quality Assurance

1. **Validate HTML** - Check that generated reports display correctly
2. **Test Edge Cases** - Handle scenarios with no trades, missing data, etc.
3. **Cross-Browser Testing** - Verify reports work in different browsers

## Integration with Testing Framework

The reporting module integrates seamlessly with the testing framework:

```python
# In your test files
from app.backtesting.reporting.generator import generate_indicator_report

def test_rsi_performance(backtest_env, results_dir):
    # Run your backtest
    results = run_rsi_backtest(backtest_env)

    # Generate charts
    charts = generate_charts(results, output_dir=results_dir)

    # Generate report
    report_path = generate_indicator_report(
        indicator_results=results,
        charts=charts,
        output_dir=results_dir
    )

    # Verify report exists and contains expected content
    assert Path(report_path).exists()

    # Parse HTML and verify metrics
    with open(report_path) as f:
        content = f.read()
        assert "Win Rate" in content
        assert "Profit Factor" in content
```

## Troubleshooting

### Common Issues

**Issue**: Report generation fails with template not found

```
jinja2.exceptions.TemplateNotFound: base.html
```

**Solution**: Ensure you're running from the correct working directory and templates exist

**Issue**: Charts not displaying in reports

```
Charts show as broken links or missing iframes
```

**Solution**: Verify chart files exist and paths are correct relative to the HTML report

**Issue**: CSS styling not applied

```
Report displays but looks unstyled
```

**Solution**: Check that static files are copied to the output directory

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run report generation with detailed logs
report_path = generate_indicator_report(...)
```

## Examples

See the `tests/` directory for complete working examples:

- `tests/backtesting/integration/test_reporting.py` - Integration tests with report generation
- `tests/_utils/cli.py` - Command-line interface examples
- `tests/__test_results__/backtesting_reports/` - Example generated reports

## Future Enhancements

Planned improvements include:

- **PDF Export** - Generate PDF versions of reports
- **Email Integration** - Automatically send reports via email
- **Real-time Updates** - Live updating reports for ongoing backtests
- **Custom Templates** - User-defined report templates
- **Advanced Analytics** - More sophisticated market condition analysis

## Support

For issues or questions about the reporting module:

1. Check this documentation first
2. Review example code in the `tests/` directory
3. Check the git history for recent changes
4. Create an issue in the project repository

---

_Generated by Spark Stacker Backtesting Engine_
