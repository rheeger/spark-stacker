"""
Interactive Report Generation Module

This module centralizes interactive report generation including:
- Interactive trade selection and highlighting features
- JavaScript component generation
- Responsive design features
- Accessibility features
- Interactive chart configuration
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InteractiveReporter:
    """
    Handles interactive report generation and analysis.

    Centralizes interactive report generation including trade selection,
    chart interaction, and JavaScript component generation.
    """

    def __init__(self):
        """Initialize the InteractiveReporter."""
        self.chart_counter = 0
        self.trade_counter = 0

    def generate_interactive_report(
        self,
        report_data: Dict[str, Any],
        trades: List[Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate an interactive HTML report with trade selection features.

        Args:
            report_data: Base report data
            trades: List of trade data for interaction
            output_path: Optional path to save the report

        Returns:
            Dictionary containing the interactive report data
        """
        logger.info("Generating interactive report with trade selection features")

        try:
            interactive_report = {
                "base_report": report_data,
                "interactive_features": {
                    "trade_list": self._generate_trade_list(trades),
                    "chart_config": self._generate_chart_config(trades),
                    "javascript_components": self._generate_modular_javascript_components(trades),
                    "css_styling": self._generate_css_styling(),
                    "html_template": self._generate_html_template(report_data, trades)
                },
                "accessibility": self._generate_accessibility_features(),
                "responsive_design": self._generate_responsive_config()
            }

            if output_path:
                self._save_interactive_report(interactive_report, output_path)

            return interactive_report

        except Exception as e:
            logger.error(f"Error generating interactive report: {e}")
            raise

    def _generate_trade_list(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate interactive trade list configuration."""
        trade_list_items = []

        for i, trade in enumerate(trades):
            trade_id = f"trade_{i+1}"

            trade_item = {
                "id": trade_id,
                "index": i + 1,
                "entry_time": trade.get("entry_time", ""),
                "exit_time": trade.get("exit_time", ""),
                "pnl": trade.get("pnl", 0),
                "duration_minutes": trade.get("duration_minutes", 0),
                "entry_price": trade.get("entry_price", 0),
                "exit_price": trade.get("exit_price", 0),
                "position_size": trade.get("position_size", 0),
                "exit_reason": trade.get("exit_reason", "unknown"),
                "chart_marker_id": f"marker_{trade_id}",
                "css_class": "trade-profitable" if trade.get("pnl", 0) > 0 else "trade-loss"
            }

            trade_list_items.append(trade_item)

        return {
            "items": trade_list_items,
            "total_trades": len(trades),
            "profitable_trades": len([t for t in trades if t.get("pnl", 0) > 0]),
            "losing_trades": len([t for t in trades if t.get("pnl", 0) < 0]),
            "list_config": {
                "sortable": True,
                "filterable": True,
                "searchable": True,
                "pagination": len(trades) > 50
            }
        }

    def _generate_chart_config(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate chart configuration for interactive features."""
        chart_markers = []

        for i, trade in enumerate(trades):
            trade_id = f"trade_{i+1}"

            # Entry marker
            entry_marker = {
                "id": f"entry_{trade_id}",
                "type": "entry",
                "timestamp": trade.get("entry_time", ""),
                "price": trade.get("entry_price", 0),
                "trade_id": trade_id,
                "color": "#2ca02c" if trade.get("pnl", 0) > 0 else "#d62728",
                "symbol": "triangle-up",
                "size": 8,
                "clickable": True,
                "hover_info": {
                    "title": f"Trade {i+1} Entry",
                    "price": trade.get("entry_price", 0),
                    "time": trade.get("entry_time", ""),
                    "position_size": trade.get("position_size", 0)
                }
            }

            # Exit marker
            exit_marker = {
                "id": f"exit_{trade_id}",
                "type": "exit",
                "timestamp": trade.get("exit_time", ""),
                "price": trade.get("exit_price", 0),
                "trade_id": trade_id,
                "color": "#2ca02c" if trade.get("pnl", 0) > 0 else "#d62728",
                "symbol": "triangle-down",
                "size": 8,
                "clickable": True,
                "hover_info": {
                    "title": f"Trade {i+1} Exit",
                    "price": trade.get("exit_price", 0),
                    "time": trade.get("exit_time", ""),
                    "pnl": trade.get("pnl", 0),
                    "exit_reason": trade.get("exit_reason", "unknown")
                }
            }

            # Connection line
            connection_line = {
                "id": f"line_{trade_id}",
                "type": "connection",
                "trade_id": trade_id,
                "start_time": trade.get("entry_time", ""),
                "end_time": trade.get("exit_time", ""),
                "start_price": trade.get("entry_price", 0),
                "end_price": trade.get("exit_price", 0),
                "color": "#2ca02c" if trade.get("pnl", 0) > 0 else "#d62728",
                "opacity": 0.6,
                "width": 2,
                "clickable": True
            }

            chart_markers.extend([entry_marker, exit_marker, connection_line])

        return {
            "markers": chart_markers,
            "chart_settings": {
                "zoom_enabled": True,
                "pan_enabled": True,
                "crosshair_enabled": True,
                "tooltip_enabled": True,
                "selection_enabled": True
            },
            "interaction_config": {
                "click_to_select": True,
                "hover_to_highlight": True,
                "keyboard_navigation": True,
                "zoom_to_trade": True
            }
        }

    def _generate_modular_javascript_components(self, trades: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate JavaScript code for modular interactive features."""
        import json

        # Generate configuration for the modules
        module_config = {
            "tradeSelector": {
                "tradeItemSelector": ".trade-item",
                "selectedClass": "selected",
                "autoScroll": True,
                "smoothScroll": True
            },
            "chartHighlighter": {
                "highlightClass": "highlighted",
                "glowEffect": True,
                "glowColor": "#2196f3",
                "glowSize": "6px",
                "pulseAnimation": False,
                "animationDuration": "0.3s",
                "zIndexHighlight": 1000,
                "markerSelector": "[data-trade-id]"
            },
            "tradeDetails": {
                "displayMode": "sidebar",
                "position": "right",
                "width": "380px",
                "animationDuration": 300,
                "responsive": True
            },
            "tradeNavigation": {
                "enableKeyboardNavigation": True,
                "enableNavigationHistory": True,
                "wrapNavigation": True
            },
            "chartZoom": {
                "zoomEnabled": True,
                "autoZoomOnSelect": False,
                "zoomPadding": 0.2,
                "showZoomControls": True
            },
            "tradeFilter": {
                "enableSearch": True,
                "enableFilters": True,
                "searchFields": ["entry_time", "exit_time", "exit_reason", "pnl", "position_size"],
                "debounceDelay": 300
            }
        }

        # Format trade data for JavaScript
        trade_data = [self._format_trade_for_js(trade, i) for i, trade in enumerate(trades)]

        # Generate main coordination script
        coordination_script = f"""
        // Spark Stacker Interactive Report Coordination
        class InteractiveReportManager {{
            constructor() {{
                this.modules = {{}};
                this.tradeData = {json.dumps(trade_data)};
                this.config = {json.dumps(module_config, indent=2)};
                this.initialized = false;
            }}

            async initialize() {{
                if (this.initialized) return;

                try {{
                    const chartInstance = this.detectChart();
                    await this.initializeModules(chartInstance);
                    this.setupModuleCommunication();
                    this.setupGlobalAPI();
                    this.initialized = true;

                    console.log('Interactive Report initialized');
                    document.dispatchEvent(new CustomEvent('interactiveReportReady'));
                }} catch (error) {{
                    console.error('Failed to initialize Interactive Report:', error);
                }}
            }}

            detectChart() {{
                const selectors = ['.js-plotly-plot', '.chart-container', '#chart'];
                for (const selector of selectors) {{
                    const element = document.querySelector(selector);
                    if (element) return element;
                }}
                return null;
            }}

            async initializeModules(chartInstance) {{
                const dependencies = {{}};

                // Initialize modules in dependency order
                if (window.TradeSelector) {{
                    this.modules.tradeSelector = new window.TradeSelector(this.config.tradeSelector);
                    dependencies.tradeSelector = this.modules.tradeSelector;
                }}

                if (window.ChartHighlighter) {{
                    this.modules.chartHighlighter = new window.ChartHighlighter(this.config.chartHighlighter);
                    dependencies.chartHighlighter = this.modules.chartHighlighter;
                }}

                if (window.TradeDetails) {{
                    this.modules.tradeDetails = new window.TradeDetails(this.config.tradeDetails);
                    dependencies.tradeDetails = this.modules.tradeDetails;
                }}

                if (window.TradeNavigation) {{
                    this.modules.tradeNavigation = new window.TradeNavigation(this.config.tradeNavigation);
                    dependencies.tradeNavigation = this.modules.tradeNavigation;
                }}

                if (window.ChartZoom) {{
                    this.modules.chartZoom = new window.ChartZoom(this.config.chartZoom);
                    dependencies.chartZoom = this.modules.chartZoom;
                }}

                if (window.TradeFilter) {{
                    this.modules.tradeFilter = new window.TradeFilter(this.config.tradeFilter);
                    dependencies.tradeFilter = this.modules.tradeFilter;
                }}

                // Initialize all modules with dependencies
                for (const [name, module] of Object.entries(this.modules)) {{
                    if (module && typeof module.initialize === 'function') {{
                        try {{
                            if (name === 'chartHighlighter' || name === 'chartZoom') {{
                                module.initialize(chartInstance, dependencies);
                            }} else {{
                                module.initialize(this.tradeData, dependencies);
                            }}
                        }} catch (error) {{
                            console.error(`Failed to initialize ${{name}}:`, error);
                        }}
                    }}
                }}
            }}

            setupModuleCommunication() {{
                document.addEventListener('tradeSelected', (event) => {{
                    console.log('Trade selected:', event.detail.tradeId);
                }});

                document.addEventListener('tradesFiltered', (event) => {{
                    console.log('Trades filtered:', event.detail.filteredTrades.length);
                }});
            }}

            setupGlobalAPI() {{
                window.sparkStackerInteractive = {{
                    selectTrade: (tradeId) => this.modules.tradeSelector?.selectTrade(tradeId),
                    clearSelection: () => this.modules.tradeSelector?.clearSelection(),
                    applyFilter: (type, value) => this.modules.tradeFilter?.setFilter(type, value),
                    zoomToTrade: (tradeId) => this.modules.chartZoom?.zoomToTrade(tradeId),
                    getModule: (name) => this.modules[name],
                    debug: () => ({{ modules: this.modules, config: this.config }})
                }};
            }}
        }}

        // Auto-initialize
        document.addEventListener('DOMContentLoaded', function() {{
            window.interactiveReportManager = new InteractiveReportManager();
            setTimeout(() => window.interactiveReportManager.initialize(), 100);
        }});
        """

        return {
            "coordination_script": coordination_script,
            "module_config": json.dumps(module_config, indent=2),
            "trade_data": json.dumps(trade_data)
        }

    def generate_javascript_module_files(self, output_dir: Path) -> List[Path]:
        """Generate separate JavaScript module files in the specified directory."""
        js_dir = output_dir / "static" / "js"
        js_dir.mkdir(parents=True, exist_ok=True)

        created_files = []

        # Copy module files from templates to output directory
        module_files = [
            "trade-selector.js",
            "chart-highlighter.js",
            "trade-details.js",
            "trade-navigation.js",
            "chart-zoom.js",
            "trade-filter.js"
        ]

        template_dir = Path(__file__).parent.parent.parent.parent / "app" / "backtesting" / "reporting" / "static" / "js"

        for module_file in module_files:
            src_path = template_dir / module_file
            dst_path = js_dir / module_file

            if src_path.exists():
                import shutil
                shutil.copy2(src_path, dst_path)
                created_files.append(dst_path)
                logger.info(f"Copied JavaScript module: {module_file}")
            else:
                logger.warning(f"Module file not found: {src_path}")

        return created_files

    def _generate_css_styling(self) -> str:
        """Generate CSS styling for interactive elements."""
        return """
        /* Interactive Trade Report Styles */

        .interactive-report {
            display: grid;
            grid-template-columns: 300px 1fr 250px;
            grid-template-rows: auto 1fr;
            gap: 20px;
            height: 100vh;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .report-header {
            grid-column: 1 / -1;
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }

        .trade-list-panel {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .trade-list-header {
            padding: 15px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            font-weight: 600;
        }

        .trade-filters {
            padding: 10px 15px;
            border-bottom: 1px solid #dee2e6;
            background: #fafbfc;
        }

        .filter-group {
            margin-bottom: 10px;
        }

        .filter-group label {
            display: block;
            font-size: 12px;
            font-weight: 500;
            margin-bottom: 4px;
            color: #6c757d;
        }

        .filter-group select,
        .filter-group input {
            width: 100%;
            padding: 6px 8px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 13px;
        }

        .trade-list {
            flex: 1;
            overflow-y: auto;
            max-height: calc(100vh - 300px);
        }

        .trade-item {
            padding: 12px 15px;
            border-bottom: 1px solid #f1f3f4;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .trade-item:hover {
            background: #f8f9fa;
        }

        .trade-item.selected {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
        }

        .trade-item.trade-profitable {
            border-left: 3px solid #4caf50;
        }

        .trade-item.trade-loss {
            border-left: 3px solid #f44336;
        }

        .trade-number {
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 4px;
        }

        .trade-pnl {
            font-weight: 500;
            font-size: 13px;
        }

        .trade-pnl.profit {
            color: #4caf50;
        }

        .trade-pnl.loss {
            color: #f44336;
        }

        .trade-time {
            font-size: 11px;
            color: #6c757d;
            margin-top: 4px;
        }

        .chart-panel {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            position: relative;
        }

        .chart-controls {
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 100;
        }

        .chart-control {
            margin-left: 10px;
        }

        .chart-control input[type="checkbox"] {
            margin-right: 5px;
        }

        .chart-control label {
            font-size: 12px;
            color: #6c757d;
        }

        .main-chart {
            width: 100%;
            height: calc(100% - 60px);
            border: 1px solid #e9ecef;
            border-radius: 4px;
        }

        .chart-marker {
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .chart-marker:hover {
            transform: scale(1.2);
        }

        .chart-marker.highlighted {
            filter: drop-shadow(0 0 6px currentColor);
            z-index: 1000;
            transform: scale(1.3);
        }

        .trade-details-panel {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            overflow-y: auto;
        }

        .trade-details h3 {
            margin: 0 0 15px 0;
            font-size: 16px;
            font-weight: 600;
            color: #212529;
        }

        .trade-detail-grid {
            display: grid;
            gap: 12px;
        }

        .detail-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #f1f3f4;
        }

        .detail-item label {
            font-size: 12px;
            font-weight: 500;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .detail-item span {
            font-size: 13px;
            font-weight: 500;
            color: #212529;
        }

        .detail-item span.profit {
            color: #4caf50;
        }

        .detail-item span.loss {
            color: #f44336;
        }

        .filtered-count {
            padding: 8px 15px;
            font-size: 11px;
            color: #6c757d;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }

        /* Responsive Design */
        @media (max-width: 1200px) {
            .interactive-report {
                grid-template-columns: 250px 1fr 200px;
            }
        }

        @media (max-width: 768px) {
            .interactive-report {
                grid-template-columns: 1fr;
                grid-template-rows: auto auto 1fr auto;
                height: auto;
            }

            .trade-list-panel {
                order: 3;
                max-height: 300px;
            }

            .chart-panel {
                order: 2;
                height: 400px;
            }

            .trade-details-panel {
                order: 4;
                max-height: 200px;
            }
        }

        /* Accessibility */
        .trade-item:focus {
            outline: 2px solid #2196f3;
            outline-offset: -2px;
        }

        .chart-marker:focus {
            outline: 2px solid #2196f3;
            outline-offset: 2px;
        }

        /* Animation for smooth transitions */
        .trade-item,
        .chart-marker {
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Loading states */
        .loading {
            opacity: 0.6;
            pointer-events: none;
        }

        .loading::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 20px;
            height: 20px;
            margin: -10px 0 0 -10px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #2196f3;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        """

    def _generate_html_template(self, report_data: Dict[str, Any], trades: List[Dict[str, Any]]) -> str:
        """Generate HTML template for interactive report."""
        strategy_name = report_data.get("strategy_name", "Strategy")
        generated_at = report_data.get("generated_at", datetime.now().isoformat())

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Interactive Report - {strategy_name}</title>
            <style>
                {{CSS_STYLES}}
            </style>
        </head>
        <body>
            <div class="interactive-report">
                <header class="report-header">
                    <h1>Interactive Trading Report</h1>
                    <h2>{strategy_name}</h2>
                    <p>Generated: {generated_at}</p>
                    <div class="report-summary">
                        <span>Total Trades: {len(trades)}</span>
                        <span>Profitable: {len([t for t in trades if t.get('pnl', 0) > 0])}</span>
                        <span>Losing: {len([t for t in trades if t.get('pnl', 0) < 0])}</span>
                    </div>
                </header>

                <div class="trade-list-panel">
                    <div class="trade-list-header">
                        Trade List
                    </div>
                    <div class="trade-filters">
                        <div class="filter-group">
                            <label for="profit-filter">Profitability</label>
                            <select id="profit-filter">
                                <option value="all">All Trades</option>
                                <option value="profitable">Profitable Only</option>
                                <option value="losing">Losing Only</option>
                            </select>
                        </div>
                        <div class="filter-group">
                            <label for="duration-filter">Duration</label>
                            <select id="duration-filter">
                                <option value="all">All Durations</option>
                                <option value="short">Short (&lt; 4h)</option>
                                <option value="medium">Medium (4-24h)</option>
                                <option value="long">Long (&gt; 24h)</option>
                            </select>
                        </div>
                        <div class="filter-group">
                            <label for="trade-search">Search</label>
                            <input type="text" id="trade-search" placeholder="Search trades...">
                        </div>
                    </div>
                    <div class="trade-list" id="trade-list">
                        {{TRADE_LIST_ITEMS}}
                    </div>
                    <div class="filtered-count" id="filtered-count">
                        {len(trades)} of {len(trades)} trades
                    </div>
                </div>

                <div class="chart-panel">
                    <div class="chart-controls">
                        <div class="chart-control">
                            <input type="checkbox" id="zoom-to-trade-checkbox">
                            <label for="zoom-to-trade-checkbox">Zoom to Trade</label>
                        </div>
                        <div class="chart-control">
                            <input type="checkbox" id="show-connections-checkbox" checked>
                            <label for="show-connections-checkbox">Show Connections</label>
                        </div>
                    </div>
                    <div class="main-chart" id="main-chart">
                        <!-- Chart will be rendered here -->
                        <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #6c757d;">
                            <div>
                                <p>Chart Integration Placeholder</p>
                                <p>This would integrate with your charting library (e.g., Plotly, D3.js, Chart.js)</p>
                                <p>Trade markers and interactions would be rendered here</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="trade-details-panel">
                    <div id="trade-details">
                        <h3>Trade Details</h3>
                        <p>Select a trade from the list or chart to view details</p>
                    </div>
                </div>
            </div>

            <script>
                {{JAVASCRIPT_COMPONENTS}}
            </script>
        </body>
        </html>
        """

    def _generate_accessibility_features(self) -> Dict[str, Any]:
        """Generate accessibility configuration."""
        return {
            "aria_labels": {
                "trade_list": "List of trading transactions",
                "chart": "Interactive trading chart with trade markers",
                "trade_details": "Selected trade details panel",
                "filters": "Trade filtering controls"
            },
            "keyboard_navigation": {
                "enabled": True,
                "shortcuts": {
                    "arrow_up": "Navigate to previous trade",
                    "arrow_down": "Navigate to next trade",
                    "enter": "Select highlighted trade",
                    "escape": "Clear selection",
                    "tab": "Navigate between interface elements"
                }
            },
            "screen_reader": {
                "announcements": True,
                "live_regions": ["trade-details", "filtered-count"],
                "descriptions": {
                    "trade_selection": "Trade {index} selected. PnL: {pnl}. Duration: {duration}",
                    "filter_applied": "{count} trades match current filters"
                }
            },
            "high_contrast": {
                "supported": True,
                "css_class": "high-contrast-mode"
            }
        }

    def _generate_responsive_config(self) -> Dict[str, Any]:
        """Generate responsive design configuration."""
        return {
            "breakpoints": {
                "mobile": "768px",
                "tablet": "1024px",
                "desktop": "1200px"
            },
            "layout_changes": {
                "mobile": {
                    "grid_template": "1fr",
                    "panel_order": ["header", "chart", "trade_list", "details"],
                    "panel_heights": {
                        "chart": "400px",
                        "trade_list": "300px",
                        "details": "200px"
                    }
                },
                "tablet": {
                    "grid_template": "250px 1fr 200px",
                    "panel_order": ["header", "trade_list", "chart", "details"]
                }
            },
            "touch_optimization": {
                "enabled": True,
                "min_touch_target": "44px",
                "swipe_gestures": {
                    "trade_navigation": True,
                    "chart_pan": True
                }
            }
        }

    def _format_trade_for_js(self, trade: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Format trade data for JavaScript consumption."""
        return {
            "id": f"trade_{index + 1}",
            "index": index + 1,
            "entry_time": trade.get("entry_time", ""),
            "exit_time": trade.get("exit_time", ""),
            "pnl": trade.get("pnl", 0),
            "duration_minutes": trade.get("duration_minutes", 0),
            "entry_price": trade.get("entry_price", 0),
            "exit_price": trade.get("exit_price", 0),
            "position_size": trade.get("position_size", 0),
            "exit_reason": trade.get("exit_reason", "unknown")
        }

    def _save_interactive_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """Save interactive report to HTML file."""
        html_template = report["interactive_features"]["html_template"]
        css_styles = report["interactive_features"]["css_styling"]
        js_components = report["interactive_features"]["javascript_components"]

        # Combine all JavaScript components
        all_js = "\n\n".join([
            js_components["module_imports"],
            js_components["module_loader"],
            js_components["coordination_script"],
            js_components["module_config"],
            js_components["trade_data"]
        ])

        # Generate trade list items HTML
        trade_list_html = self._generate_trade_list_html(report["base_report"])

        # Replace placeholders in template
        final_html = html_template.replace("{{CSS_STYLES}}", css_styles)
        final_html = final_html.replace("{{JAVASCRIPT_COMPONENTS}}", all_js)
        final_html = final_html.replace("{{TRADE_LIST_ITEMS}}", trade_list_html)

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)

        logger.info(f"Interactive report saved to: {output_path}")

    def _generate_trade_list_html(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML for trade list items."""
        trades = report_data.get("trades", [])
        html_items = []

        for i, trade in enumerate(trades):
            trade_id = f"trade_{i + 1}"
            pnl = trade.get("pnl", 0)
            css_class = "trade-profitable" if pnl > 0 else "trade-loss"
            pnl_class = "profit" if pnl > 0 else "loss"

            duration_hours = trade.get("duration_minutes", 0) / 60

            item_html = f"""
            <div class="trade-item {css_class}" data-trade-id="{trade_id}" tabindex="0">
                <div class="trade-number">Trade {i + 1}</div>
                <div class="trade-pnl {pnl_class}">
                    {'+ ' if pnl > 0 else ''}${pnl:.2f}
                </div>
                <div class="trade-time">
                    {duration_hours:.1f}h • {trade.get('exit_reason', 'unknown')}
                </div>
            </div>
            """
            html_items.append(item_html)

        return "\n".join(html_items)

    def export_interactive_report(
        self,
        report_data: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """
        Export interactive report as standalone HTML file.

        Args:
            report_data: Complete interactive report data
            output_path: Path to save the HTML file

        Returns:
            Path to the exported file
        """
        self._save_interactive_report(report_data, output_path)
        return output_path
