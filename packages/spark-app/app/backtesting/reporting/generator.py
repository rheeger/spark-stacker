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


def generate_comparison_report(
    indicator_results: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None
) -> str:
    """
    Generate a comparison report for multiple indicators.

    Args:
        indicator_results: List of dictionaries containing results for each indicator
        output_dir: Optional output directory override
        output_filename: Optional filename override

    Returns:
        Path to the generated report file
    """
    # Create report generator
    generator = ReportGenerator(output_dir)

    # Generate filename if not provided
    if output_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        output_filename = f"indicator_comparison_{timestamp}.html"

    # Prepare context for the template
    indicator_names = [res.get("indicator_name", "Unknown") for res in indicator_results]
    market = indicator_results[0].get("market", "Unknown Market") if indicator_results else "Unknown"

    context = {
        "title": "Indicator Comparison",
        "market": market,
        "indicators": indicator_results,
        "indicator_names": indicator_names
    }

    # Generate the report (Note: This will require a comparison.html template)
    report_path = generator.generate_report(
        template_name="comparison.html",  # Will be created in a later task
        output_filename=output_filename,
        context=context
    )

    return report_path
