"""
Data transformer module for backtesting reporting.

This module provides functions to transform raw backtest results into formats
suitable for report templates. It handles data preparation and formatting
for HTML reports.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def transform_backtest_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform raw backtest results into a format suitable for report templates.

    Args:
        results: Dictionary containing raw backtest results with at least:
            - indicator_name: Name of the indicator
            - market: Market/symbol being traded
            - timeframe: Timeframe of the backtest
            - start_date: Start date of the backtest
            - end_date: End date of the backtest
            - trades: List of trade dictionaries
            - metrics: Performance metrics (optional)

    Returns:
        Dictionary with transformed data ready for template rendering
    """
    if not isinstance(results, dict):
        logger.error("Results must be a dictionary")
        raise ValueError("Invalid results format: expected dictionary")

    transformed = {
        "indicator_name": results.get("indicator_name", "Unknown Indicator"),
        "market": results.get("market", "Unknown Market"),
        "timeframe": results.get("timeframe", "Unknown Timeframe"),
        "summary": generate_summary(results),
        "trades_table": format_trade_list(results.get("trades", [])),
        "metrics_table": format_metrics_table(results.get("metrics", {})),
    }

    # Format dates
    if "start_date" in results:
        transformed["start_date"] = format_date(results["start_date"])
    if "end_date" in results:
        transformed["end_date"] = format_date(results["end_date"])

    # Add chart paths if available
    if "charts" in results:
        transformed["charts"] = results["charts"]

    return transformed


def generate_summary(results: Dict[str, Any]) -> str:
    """
    Generate a text summary of the backtest results.

    Args:
        results: Dictionary containing backtest results

    Returns:
        HTML string with summary information
    """
    metrics = results.get("metrics", {})

    # Extract key metrics
    total_return = metrics.get("total_return", 0)
    win_rate = metrics.get("win_rate", 0)
    num_trades = metrics.get("num_trades", 0)
    market = results.get("market", "Unknown Market")

    # Build summary string
    summary = f"""
    <div class="summary-box">
        <h3>Summary</h3>
        <p>Backtest of <strong>{results.get('indicator_name', 'Unknown')}</strong> on
        <strong>{market}</strong> ({results.get('timeframe', 'Unknown')}).</p>

        <p>Performance: <span class="{get_return_class(total_return)}">{total_return:.2f}%</span>
        total return with a {win_rate:.1f}% win rate over {num_trades} trades.</p>
    </div>
    """

    return summary


def get_return_class(return_value: float) -> str:
    """
    Get CSS class based on return value.

    Args:
        return_value: Return percentage

    Returns:
        CSS class name
    """
    if return_value > 5:
        return "positive-strong"
    elif return_value > 0:
        return "positive"
    elif return_value < -5:
        return "negative-strong"
    elif return_value < 0:
        return "negative"
    else:
        return "neutral"


def format_date(date_value: Any) -> str:
    """
    Format a date value to a standardized string.

    Args:
        date_value: Date as string, datetime, or timestamp

    Returns:
        Formatted date string
    """
    if isinstance(date_value, str):
        try:
            # Try to parse string to datetime
            date_obj = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            return date_obj.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            # Return as is if parsing fails
            return date_value
    elif isinstance(date_value, datetime):
        return date_value.strftime("%Y-%m-%d %H:%M")
    elif isinstance(date_value, (int, float)):
        # Assume timestamp in seconds
        return datetime.fromtimestamp(date_value).strftime("%Y-%m-%d %H:%M")
    else:
        return str(date_value)


def format_trade_list(trades: List[Dict[str, Any]]) -> str:
    """
    Generate an HTML table from a list of trades.

    Args:
        trades: List of trade dictionaries

    Returns:
        HTML string with a formatted table of trades
    """
    if not trades:
        return "<p>No trades were executed during this backtest.</p>"

    # Start building the HTML table
    html = """
    <div class="table-responsive">
      <table class="trades-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Entry Time</th>
            <th>Exit Time</th>
            <th>Direction</th>
            <th>Entry Price</th>
            <th>Exit Price</th>
            <th>Size</th>
            <th>P&L</th>
            <th>Return</th>
          </tr>
        </thead>
        <tbody>
    """

    # Add table rows for each trade
    for i, trade in enumerate(trades, 1):
        # Extract trade data with defaults
        entry_time = format_date(trade.get("entry_time", ""))
        exit_time = format_date(trade.get("exit_time", ""))
        direction = trade.get("direction", "unknown").upper()
        entry_price = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        size = trade.get("size", 0)
        pnl = trade.get("profit_loss", 0)
        pnl_pct = trade.get("profit_loss_pct", 0) * 100 if "profit_loss_pct" in trade else 0

        # Determine CSS class based on P&L
        row_class = "positive-row" if pnl > 0 else "negative-row" if pnl < 0 else ""
        pnl_class = "positive" if pnl > 0 else "negative" if pnl < 0 else "neutral"

        # Add row to table
        html += f"""
        <tr class="{row_class}">
          <td>{i}</td>
          <td>{entry_time}</td>
          <td>{exit_time}</td>
          <td>{direction}</td>
          <td>{entry_price:.6f}</td>
          <td>{exit_price:.6f}</td>
          <td>{size:.4f}</td>
          <td class="{pnl_class}">{pnl:.6f}</td>
          <td class="{pnl_class}">{pnl_pct:.2f}%</td>
        </tr>
        """

    # Complete the table
    html += """
        </tbody>
      </table>
    </div>
    <div class="trade-filters">
      <button class="filter-btn" data-filter="all">All Trades</button>
      <button class="filter-btn" data-filter="long">Long Only</button>
      <button class="filter-btn" data-filter="short">Short Only</button>
      <button class="filter-btn" data-filter="win">Winning</button>
      <button class="filter-btn" data-filter="loss">Losing</button>
    </div>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
      const filterButtons = document.querySelectorAll('.filter-btn');
      const tradeRows = document.querySelectorAll('.trades-table tbody tr');

      filterButtons.forEach(button => {
        button.addEventListener('click', function() {
          const filter = this.getAttribute('data-filter');

          tradeRows.forEach(row => {
            const direction = row.children[3].textContent.trim().toLowerCase();
            const pnl = parseFloat(row.children[7].textContent);

            switch(filter) {
              case 'all':
                row.style.display = '';
                break;
              case 'long':
                row.style.display = direction === 'long' ? '' : 'none';
                break;
              case 'short':
                row.style.display = direction === 'short' ? '' : 'none';
                break;
              case 'win':
                row.style.display = pnl > 0 ? '' : 'none';
                break;
              case 'loss':
                row.style.display = pnl < 0 ? '' : 'none';
                break;
            }
          });

          // Update active button
          filterButtons.forEach(btn => btn.classList.remove('active'));
          this.classList.add('active');
        });
      });

      // Set 'All Trades' as default
      filterButtons[0].classList.add('active');
    });
    </script>
    """

    return html


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """
    Generate an HTML table from performance metrics.

    Args:
        metrics: Dictionary of performance metrics

    Returns:
        HTML string with a formatted table of metrics
    """
    if not metrics:
        return "<p>No performance metrics available.</p>"

    # Define metrics to include and their display names
    metric_definitions = [
        {"key": "total_return", "name": "Total Return", "format": "{:.2f}%", "description": "Total percentage return over the test period"},
        {"key": "win_rate", "name": "Win Rate", "format": "{:.2f}%", "description": "Percentage of trades that were profitable"},
        {"key": "profit_factor", "name": "Profit Factor", "format": "{:.2f}", "description": "Ratio of gross profit to gross loss"},
        {"key": "max_drawdown", "name": "Max Drawdown", "format": "{:.2f}%", "description": "Maximum percentage decline from peak to trough"},
        {"key": "sharpe_ratio", "name": "Sharpe Ratio", "format": "{:.2f}", "description": "Risk-adjusted return (higher is better)"},
        {"key": "avg_trade_return", "name": "Avg. Trade Return", "format": "{:.2f}%", "description": "Average percentage return per trade"},
        {"key": "num_trades", "name": "Number of Trades", "format": "{}", "description": "Total number of trades executed"}
    ]

    # Start building the HTML table
    html = """
    <div class="metrics-table-container">
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
    """

    # Add table rows for each metric
    for metric_def in metric_definitions:
        key = metric_def["key"]
        if key in metrics:
            value = metrics[key]
            formatted_value = metric_def["format"].format(value)

            # Apply color coding for certain metrics
            value_class = ""
            if key == "total_return" or key == "avg_trade_return":
                value_class = get_return_class(value)
            elif key == "win_rate":
                value_class = "positive" if value > 50 else "negative" if value < 50 else "neutral"
            elif key == "sharpe_ratio":
                value_class = "positive" if value > 1 else "negative" if value < 0 else "neutral"

            html += f"""
            <tr>
              <td>{metric_def["name"]}</td>
              <td class="{value_class}">{formatted_value}</td>
              <td><small>{metric_def["description"]}</small></td>
            </tr>
            """

    # Complete the table
    html += """
        </tbody>
      </table>
    </div>
    """

    return html
