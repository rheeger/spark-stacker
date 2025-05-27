"""
Report generation module for backtesting results.
Generates static HTML reports from backtesting data using Jinja2 templates.
"""

import datetime
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jinja2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates HTML reports from backtesting results using Jinja2 templates.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the report generator.

        Args:
            output_dir: The directory where reports will be saved. If None, uses
                        a default directory in the project's test_results folder.
        """
        # Set up template environment
        template_dir = Path(__file__).parent / "templates"
        static_dir = template_dir / "static"

        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Add custom filters
        self.template_env.filters['format_timestamp'] = self._format_timestamp
        self.template_env.filters['format_price'] = self._format_price
        self.template_env.filters['format_size'] = self._format_size
        self.template_env.filters['format_pnl'] = self._format_pnl

        # Set up output directory
        if output_dir is None:
            # Default to __test_results__/backtesting_reports directory
            project_root = Path(__file__).parent.parent.parent.parent
            self.output_dir = project_root / "tests" / "__test_results__" / "backtesting_reports"
        else:
            self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create a static directory in the output directory for CSS and other assets
        self.output_static_dir = self.output_dir / "static"
        self.output_static_dir.mkdir(exist_ok=True)

        # Copy static files (CSS, etc.) to the output directory
        self._copy_static_files(static_dir, self.output_static_dir)

        logger.info(f"Report generator initialized with output directory: {self.output_dir}")

    def _format_timestamp(self, timestamp) -> str:
        """Format a timestamp (milliseconds since epoch) as a human-readable date/time."""
        if not timestamp:
            return ""
        try:
            # Convert to seconds if in milliseconds
            if isinstance(timestamp, (int, float)) and timestamp > 1000000000000:
                timestamp = timestamp / 1000
            return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return str(timestamp)

    def _format_price(self, price) -> str:
        """Format a price value with appropriate precision."""
        if not price:
            return "0.00"
        try:
            return f"{float(price):.2f}"
        except (ValueError, TypeError):
            return str(price)

    def _format_size(self, size) -> str:
        """Format a size/amount value with appropriate precision."""
        if not size:
            return "0.0000"
        try:
            return f"{float(size):.4f}"
        except (ValueError, TypeError):
            return str(size)

    def _format_pnl(self, pnl) -> str:
        """Format a profit/loss value with appropriate precision and sign."""
        if not pnl:
            return "0.00"
        try:
            value = float(pnl)
            if value > 0:
                return f"+{value:.2f}"
            else:
                return f"{value:.2f}"
        except (ValueError, TypeError):
            return str(pnl)

    def _copy_static_files(self, source_dir: Path, target_dir: Path) -> None:
        """
        Copy static files (CSS, JS, etc.) to the output directory.

        Args:
            source_dir: The source directory containing static files
            target_dir: The target directory where files will be copied
        """
        if not source_dir.exists():
            logger.warning(f"Static files directory {source_dir} does not exist.")
            return

        # Copy all files from the static directory
        for file_path in source_dir.glob('*'):
            if file_path.is_file():
                shutil.copy2(file_path, target_dir)
                logger.debug(f"Copied static file: {file_path.name}")

    def generate_report_filename(
        self,
        indicator_name: str,
        market: str,
        timeframe: str
    ) -> str:
        """
        Generate a standardized filename for a report.

        Args:
            indicator_name: Name of the indicator
            market: Market/symbol being traded
            timeframe: Timeframe of the backtest

        Returns:
            A filename for the report in the format:
            indicator_market_timeframe_YYYY-MM-DD.html
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        safe_indicator = indicator_name.replace(" ", "_").lower()
        safe_market = market.replace("/", "-").lower()

        filename = f"{safe_indicator}_{safe_market}_{timeframe}_{timestamp}.html"
        return filename

    def generate_report(
        self,
        template_name: str,
        output_filename: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate an HTML report from a template and context data.

        Args:
            template_name: Name of the template file (e.g., "base.html")
            output_filename: Name of the output file
            context: Dictionary of context data to pass to the template

        Returns:
            Path to the generated report file
        """
        try:
            # Make sure the template exists
            template = self.template_env.get_template(template_name)

            # Add some standard context variables
            if 'generation_time' not in context:
                context['generation_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if 'current_year' not in context:
                context['current_year'] = datetime.datetime.now().year

            # Add static path to context
            context['static_path'] = './static'

            # Render the template with the provided context
            html_content = template.render(**context)

            # Save to output file
            output_path = self.output_dir / output_filename
            with open(output_path, 'w') as f:
                f.write(html_content)

            logger.info(f"Generated report: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise

    def save_chart_image(
        self,
        image_data: Union[str, bytes],
        filename: str
    ) -> str:
        """
        Save a chart image file to the output directory.

        Args:
            image_data: Image data as a string (file path) or bytes object
            filename: Desired filename for the saved image

        Returns:
            Relative path to the saved image from the report HTML file
        """
        # Create an images directory if it doesn't exist
        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        output_path = images_dir / filename

        # Handle different input types
        if isinstance(image_data, str) and os.path.isfile(image_data):
            # It's a file path, copy the file
            shutil.copy2(image_data, output_path)
        elif isinstance(image_data, bytes):
            # It's binary data, write it to a file
            with open(output_path, 'wb') as f:
                f.write(image_data)
        else:
            raise ValueError("image_data must be a file path or bytes object")

        # Return a relative path usable from the HTML file
        return f"./images/{filename}"


def generate_indicator_report(
    indicator_results: Dict[str, Any],
    charts: Dict[str, Any],
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    indicator_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a report for a single indicator's backtest results.

    Args:
        indicator_results: Results from the backtesting run
        charts: Dictionary of chart data (can be paths to pre-generated chart files)
        output_dir: Optional output directory override
        output_filename: Optional filename override
        indicator_config: Configuration parameters for the indicator

    Returns:
        Path to the generated report file
    """
    # Create report generator
    generator = ReportGenerator(output_dir)

    # Generate filename if not provided
    if output_filename is None:
        indicator_name = indicator_results.get("indicator_name", "unknown")
        market = indicator_results.get("market", "unknown")
        timeframe = indicator_results.get("timeframe", "unknown")
        output_filename = generator.generate_report_filename(
            indicator_name, market, timeframe)

    # Prepare context for the template
    context = {
        "indicator_name": indicator_results.get("indicator_name", "Unknown Indicator"),
        "market": indicator_results.get("market", "Unknown Market"),
        "timeframe": indicator_results.get("timeframe", "Unknown Timeframe"),
        "start_date": indicator_results.get("start_date", "Unknown Start Date"),
        "end_date": indicator_results.get("end_date", "Unknown End Date"),
        "metrics": indicator_results.get("metrics", {}),
        "trades": indicator_results.get("trades", []),
        "charts": charts,
        "indicator_config": indicator_config
    }

    # Generate the report
    report_path = generator.generate_report(
        template_name="base.html",
        output_filename=output_filename,
        context=context
    )

    return report_path


def classify_market_conditions(price_data: List[float]) -> Dict[str, Any]:
    """
    Classify market conditions based on price movement.

    Args:
        price_data: List of price values over time

    Returns:
        Dictionary containing market condition analysis
    """
    if len(price_data) < 2:
        return {
            "description": "Insufficient data for market condition analysis",
            "periods": []
        }

    import numpy as np

    # Calculate price changes
    prices = np.array(price_data)
    returns = np.diff(prices) / prices[:-1]

    # Calculate rolling statistics (using 20-period windows)
    window_size = min(20, len(returns) // 4)
    if window_size < 5:
        window_size = len(returns)

    # Simple trend classification
    total_return = (prices[-1] - prices[0]) / prices[0]
    volatility = np.std(returns)

    # Classify overall market
    if total_return > 0.1:  # 10% positive
        primary_condition = "bull"
        condition_description = "Strong upward trend"
    elif total_return < -0.1:  # 10% negative
        primary_condition = "bear"
        condition_description = "Strong downward trend"
    else:
        primary_condition = "sideways"
        condition_description = "Sideways/consolidating market"

    # Estimate periods (simplified)
    bull_periods = max(0, len([r for r in returns if r > 0.02]))  # 2% daily gains
    bear_periods = max(0, len([r for r in returns if r < -0.02]))  # 2% daily losses
    sideways_periods = len(returns) - bull_periods - bear_periods

    total_periods = len(returns)

    periods = []
    if bull_periods > 0:
        periods.append({
            "type": "bull",
            "label": "Bull Market",
            "percentage": round((bull_periods / total_periods) * 100, 1)
        })
    if bear_periods > 0:
        periods.append({
            "type": "bear",
            "label": "Bear Market",
            "percentage": round((bear_periods / total_periods) * 100, 1)
        })
    if sideways_periods > 0:
        periods.append({
            "type": "sideways",
            "label": "Sideways Market",
            "percentage": round((sideways_periods / total_periods) * 100, 1)
        })

    return {
        "description": f"{condition_description} with {volatility:.2%} volatility",
        "periods": periods,
        "primary_condition": primary_condition,
        "total_return": total_return,
        "volatility": volatility
    }


def create_metrics_table(indicator_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a structured metrics comparison table for multiple indicators.

    Args:
        indicator_results: List of indicator result dictionaries

    Returns:
        Dictionary containing structured comparison data
    """
    if not indicator_results:
        return {"comparison_metrics": [], "rankings": {}}

    # Define the metrics we want to compare
    metric_definitions = [
        {
            "key": "win_rate",
            "display_name": "Win Rate",
            "units": "%",
            "higher_is_better": True
        },
        {
            "key": "total_return",
            "display_name": "Total Return",
            "units": "%",
            "higher_is_better": True
        },
        {
            "key": "profit_factor",
            "display_name": "Profit Factor",
            "units": "",
            "higher_is_better": True
        },
        {
            "key": "sharpe",
            "display_name": "Sharpe Ratio",
            "units": "",
            "higher_is_better": True
        },
        {
            "key": "max_drawdown",
            "display_name": "Max Drawdown",
            "units": "%",
            "higher_is_better": False
        },
        {
            "key": "total_trades",
            "display_name": "Total Trades",
            "units": "",
            "higher_is_better": True
        }
    ]

    comparison_metrics = []

    # Process each metric
    for metric_def in metric_definitions:
        metric_key = metric_def["key"]

        # Extract values for this metric from all indicators
        values = {}
        for result in indicator_results:
            indicator_name = result.get("indicator_name", "Unknown")
            metrics = result.get("metrics", {})

            if metric_key in metrics:
                value = metrics[metric_key]
                # Format the value appropriately
                if metric_def["units"] == "%":
                    values[indicator_name] = f"{value:.1f}"
                elif metric_key == "profit_factor" or metric_key == "sharpe":
                    values[indicator_name] = f"{value:.2f}"
                else:
                    values[indicator_name] = str(int(value)) if isinstance(value, (int, float)) else str(value)
            else:
                values[indicator_name] = "N/A"

        # Find best and worst performers
        numeric_values = {}
        for name, val in values.items():
            try:
                numeric_values[name] = float(val) if val != "N/A" else None
            except ValueError:
                numeric_values[name] = None

        valid_values = {k: v for k, v in numeric_values.items() if v is not None}

        if valid_values:
            if metric_def["higher_is_better"]:
                best_indicator = max(valid_values.keys(), key=lambda k: valid_values[k])
                worst_indicator = min(valid_values.keys(), key=lambda k: valid_values[k])
                best_value = max(valid_values.values())
            else:
                best_indicator = min(valid_values.keys(), key=lambda k: valid_values[k])
                worst_indicator = max(valid_values.keys(), key=lambda k: valid_values[k])
                best_value = min(valid_values.values())

            # Format best value
            if metric_def["units"] == "%":
                best_value_formatted = f"{best_value:.1f}"
            elif metric_key == "profit_factor" or metric_key == "sharpe":
                best_value_formatted = f"{best_value:.2f}"
            else:
                best_value_formatted = str(int(best_value)) if isinstance(best_value, (int, float)) else str(best_value)
        else:
            best_indicator = None
            worst_indicator = None
            best_value_formatted = "N/A"

        comparison_metrics.append({
            "key": metric_key,
            "display_name": metric_def["display_name"],
            "units": metric_def["units"],
            "indicator_values": values,  # Changed from 'values' to avoid conflict with dict.values()
            "best_indicator": best_indicator,
            "worst_indicator": worst_indicator,
            "best_value": best_value_formatted
        })

    # Create rankings
    rankings = create_performance_rankings(indicator_results)

    return {
        "comparison_metrics": comparison_metrics,
        "rankings": rankings
    }


def create_performance_rankings(indicator_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create performance rankings for indicators across different metrics.

    Args:
        indicator_results: List of indicator result dictionaries

    Returns:
        Dictionary containing rankings by different criteria
    """
    rankings = {
        "by_return": [],
        "by_sharpe": [],
        "by_win_rate": []
    }

    # Extract and sort by total return
    return_data = []
    for result in indicator_results:
        name = result.get("indicator_name", "Unknown")
        metrics = result.get("metrics", {})
        total_return = metrics.get("total_return", 0)
        return_data.append({"name": name, "value": total_return})

    rankings["by_return"] = sorted(return_data, key=lambda x: x["value"], reverse=True)

    # Extract and sort by Sharpe ratio
    sharpe_data = []
    for result in indicator_results:
        name = result.get("indicator_name", "Unknown")
        metrics = result.get("metrics", {})
        sharpe = metrics.get("sharpe", 0)
        sharpe_data.append({"name": name, "value": sharpe})

    rankings["by_sharpe"] = sorted(sharpe_data, key=lambda x: x["value"], reverse=True)

    # Extract and sort by win rate
    win_rate_data = []
    for result in indicator_results:
        name = result.get("indicator_name", "Unknown")
        metrics = result.get("metrics", {})
        win_rate = metrics.get("win_rate", 0)
        win_rate_data.append({"name": name, "value": win_rate})

    rankings["by_win_rate"] = sorted(win_rate_data, key=lambda x: x["value"], reverse=True)

    return rankings


def generate_comparison_report(
    indicator_results: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    market_price_data: Optional[List[float]] = None
) -> str:
    """
    Generate a comparison report for multiple indicators.

    Args:
        indicator_results: List of dictionaries containing results for each indicator
        output_dir: Optional output directory override
        output_filename: Optional filename override
        market_price_data: Optional price data for market condition analysis

    Returns:
        Path to the generated report file
    """
    # Create report generator
    generator = ReportGenerator(output_dir)

    # Generate filename if not provided
    if output_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        output_filename = f"indicator_comparison_{timestamp}.html"

    # Process results and add rankings
    processed_indicators = []
    for i, result in enumerate(indicator_results):
        processed_result = result.copy()
        # Add ranking based on total return (simple ranking for now)
        processed_result["ranking"] = i + 1
        # Ensure 'name' field exists for template compatibility
        processed_result["name"] = processed_result.get("indicator_name", "Unknown")
        processed_indicators.append(processed_result)

    # Sort indicators by total return for ranking
    processed_indicators.sort(
        key=lambda x: x.get("metrics", {}).get("total_return", 0),
        reverse=True
    )

    # Update rankings
    for i, result in enumerate(processed_indicators):
        result["ranking"] = i + 1

    # Create metrics comparison table
    metrics_data = create_metrics_table(indicator_results)

    # Analyze market conditions if price data provided
    market_conditions = None
    if market_price_data:
        market_conditions = classify_market_conditions(market_price_data)

    # Extract common metadata
    if indicator_results:
        first_result = indicator_results[0]
        market = first_result.get("market", "Unknown Market")
        timeframe = first_result.get("timeframe", "Unknown Timeframe")
        start_date = first_result.get("start_date", "Unknown Start Date")
        end_date = first_result.get("end_date", "Unknown End Date")
    else:
        market = timeframe = start_date = end_date = "Unknown"

    # Prepare context for the template
    context = {
        "market": market,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "indicator_count": len(indicator_results),
        "indicators": processed_indicators,
        "comparison_metrics": metrics_data["comparison_metrics"],
        "rankings": metrics_data["rankings"],
        "market_conditions": market_conditions,
        "individual_reports": []  # Could be populated with links to individual reports
    }

    # Generate the report
    report_path = generator.generate_report(
        template_name="comparison.html",
        output_filename=output_filename,
        context=context
    )

    return report_path
