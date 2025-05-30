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
        """Generate enhanced chart configuration for interactive features."""
        chart_markers = []
        trade_sequences = []

        for i, trade in enumerate(trades):
            trade_id = f"trade_{i+1}"

            # Generate unique IDs for all trade markers
            entry_marker_id = f"entry_marker_{trade_id}"
            exit_marker_id = f"exit_marker_{trade_id}"
            connection_line_id = f"connection_line_{trade_id}"

            # Entry marker with comprehensive metadata
            entry_marker = {
                "id": entry_marker_id,
                "unique_id": entry_marker_id,  # For JavaScript targeting
                "type": "entry",
                "timestamp": trade.get("entry_time", ""),
                "price": trade.get("entry_price", 0),
                "trade_id": trade_id,
                "trade_index": i + 1,
                "color": "#2ca02c" if trade.get("pnl", 0) > 0 else "#d62728",
                "symbol": "triangle-up",
                "size": 8,
                "clickable": True,
                # Trade metadata in data attributes
                "data_attributes": {
                    "data-trade-id": trade_id,
                    "data-trade-index": str(i + 1),
                    "data-trade-type": "entry",
                    "data-position-size": str(trade.get("position_size", 0)),
                    "data-entry-reason": trade.get("entry_reason", ""),
                    "data-strategy": trade.get("strategy", ""),
                    "data-pnl": str(trade.get("pnl", 0)),
                    "data-duration": str(trade.get("duration_hours", 0)),
                    # Accessibility attributes
                    "aria-label": f"Trade {i+1} entry at ${trade.get('entry_price', 0):.2f}",
                    "role": "button",
                    "tabindex": "0"
                },
                "hover_info": {
                    "title": f"Trade {i+1} Entry",
                    "price": trade.get("entry_price", 0),
                    "time": trade.get("entry_time", ""),
                    "position_size": trade.get("position_size", 0),
                    "entry_reason": trade.get("entry_reason", ""),
                    "strategy": trade.get("strategy", "")
                }
            }

            # Exit marker with comprehensive metadata
            exit_marker = {
                "id": exit_marker_id,
                "unique_id": exit_marker_id,  # For JavaScript targeting
                "type": "exit",
                "timestamp": trade.get("exit_time", ""),
                "price": trade.get("exit_price", 0),
                "trade_id": trade_id,
                "trade_index": i + 1,
                "color": "#2ca02c" if trade.get("pnl", 0) > 0 else "#d62728",
                "symbol": "triangle-down",
                "size": 8,
                "clickable": True,
                # Trade metadata in data attributes
                "data_attributes": {
                    "data-trade-id": trade_id,
                    "data-trade-index": str(i + 1),
                    "data-trade-type": "exit",
                    "data-pnl": str(trade.get("pnl", 0)),
                    "data-exit-reason": trade.get("exit_reason", ""),
                    "data-strategy": trade.get("strategy", ""),
                    "data-duration": str(trade.get("duration_hours", 0)),
                    "data-win-rate": str(trade.get("win_rate", 0)),
                    # Accessibility attributes
                    "aria-label": f"Trade {i+1} exit at ${trade.get('exit_price', 0):.2f}, PnL: ${trade.get('pnl', 0):.2f}",
                    "role": "button",
                    "tabindex": "0"
                },
                "hover_info": {
                    "title": f"Trade {i+1} Exit",
                    "price": trade.get("exit_price", 0),
                    "time": trade.get("exit_time", ""),
                    "pnl": trade.get("pnl", 0),
                    "exit_reason": trade.get("exit_reason", "unknown"),
                    "duration": trade.get("duration_hours", 0),
                    "strategy": trade.get("strategy", "")
                }
            }

            # Connection line with metadata
            connection_line = {
                "id": connection_line_id,
                "unique_id": connection_line_id,  # For JavaScript targeting
                "type": "connection",
                "trade_id": trade_id,
                "trade_index": i + 1,
                "start_time": trade.get("entry_time", ""),
                "end_time": trade.get("exit_time", ""),
                "start_price": trade.get("entry_price", 0),
                "end_price": trade.get("exit_price", 0),
                "color": "#2ca02c" if trade.get("pnl", 0) > 0 else "#d62728",
                "opacity": 0.6,
                "width": 2,
                "clickable": True,
                # Trade metadata in data attributes
                "data_attributes": {
                    "data-trade-id": trade_id,
                    "data-trade-index": str(i + 1),
                    "data-trade-type": "connection",
                    "data-pnl": str(trade.get("pnl", 0)),
                    "data-strategy": trade.get("strategy", ""),
                    # Accessibility attributes
                    "aria-label": f"Trade {i+1} connection line, PnL: ${trade.get('pnl', 0):.2f}",
                    "role": "button",
                    "tabindex": "0"
                }
            }

            chart_markers.extend([entry_marker, exit_marker, connection_line])

            # Generate trade sequence data for connecting entry/exit points
            trade_sequence = {
                "trade_id": trade_id,
                "trade_index": i + 1,
                "entry_marker_id": entry_marker_id,
                "exit_marker_id": exit_marker_id,
                "connection_line_id": connection_line_id,
                "entry_point": {
                    "time": trade.get("entry_time", ""),
                    "price": trade.get("entry_price", 0)
                },
                "exit_point": {
                    "time": trade.get("exit_time", ""),
                    "price": trade.get("exit_price", 0)
                },
                "metadata": {
                    "pnl": trade.get("pnl", 0),
                    "duration_hours": trade.get("duration_hours", 0),
                    "strategy": trade.get("strategy", ""),
                    "entry_reason": trade.get("entry_reason", ""),
                    "exit_reason": trade.get("exit_reason", ""),
                    "position_size": trade.get("position_size", 0)
                }
            }
            trade_sequences.append(trade_sequence)

        return {
            "markers": chart_markers,
            "trade_sequences": trade_sequences,  # New: trade sequence data
            "chart_settings": {
                "zoom_enabled": True,
                "pan_enabled": True,
                "crosshair_enabled": True,
                "tooltip_enabled": True,
                "selection_enabled": True,
                # Enhanced: responsive chart sizing
                "responsive_sizing": {
                    "mobile": {
                        "width": "100%",
                        "height": "300px",
                        "margin": {"l": 20, "r": 20, "t": 30, "b": 30}
                    },
                    "tablet": {
                        "width": "100%",
                        "height": "400px",
                        "margin": {"l": 40, "r": 40, "t": 40, "b": 40}
                    },
                    "desktop": {
                        "width": "100%",
                        "height": "500px",
                        "margin": {"l": 60, "r": 60, "t": 50, "b": 50}
                    }
                },
                # Enhanced: accessibility features
                "accessibility": {
                    "aria_live": "polite",
                    "aria_describedby": "chart-description",
                    "keyboard_navigation": True,
                    "screen_reader_support": True,
                    "high_contrast_mode": False,
                    "focus_indicators": True
                }
            },
            "interaction_config": {
                "click_to_select": True,
                "hover_to_highlight": True,
                "keyboard_navigation": True,
                "zoom_to_trade": True,
                # Enhanced: zoom and pan capabilities
                "zoom_config": {
                    "scroll_zoom": True,
                    "box_zoom": True,
                    "auto_zoom": True,
                    "reset_zoom": True,
                    "zoom_speed": 1.2,
                    "min_zoom": 0.1,
                    "max_zoom": 10.0
                },
                "pan_config": {
                    "drag_pan": True,
                    "touch_pan": True,
                    "momentum_pan": True,
                    "pan_speed": 1.0
                }
            },
            # New: strategy-specific chart features
            "strategy_features": {
                "show_strategy_indicators": True,
                "highlight_strategy_signals": True,
                "show_position_sizing": True,
                "display_risk_levels": True,
                "show_stop_loss": True,
                "show_take_profit": True
            },
            # New: chart layout configuration
            "layout_config": {
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
                "font": {"family": "Arial, sans-serif", "size": 12},
                "showlegend": True,
                "legend": {"x": 0.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.8)"},
                "hovermode": "closest",
                "dragmode": "zoom"
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
        """Generate enhanced CSS styling for interactive elements."""
        return """
        /* Interactive Trade Report Styles */

        /* Root Variables for Theme Consistency */
        :root {
            --primary-color: #007bff;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --info-color: #17a2b8;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
            --muted-color: #6c757d;
            --border-color: #dee2e6;
            --hover-color: #e9ecef;
            --focus-color: #80bdff;
            --transition-speed: 0.2s;
            --border-radius: 6px;
            --box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            --box-shadow-hover: 0 4px 8px rgba(0,0,0,0.15);
        }

        /* Base Layout */
        * {
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f6fa;
            color: var(--dark-color);
            line-height: 1.5;
        }

        .interactive-report {
            display: grid;
            grid-template-columns: 300px 1fr 350px;
            grid-template-rows: auto 1fr;
            grid-template-areas:
                "header header header"
                "trade-list chart trade-details";
            height: 100vh;
            gap: 1rem;
            padding: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        /* Header Styles */
        .report-header {
            grid-area: header;
            background: white;
            padding: 1.5rem;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            border-left: 4px solid var(--primary-color);
        }

        .report-header h1 {
            margin: 0 0 0.5rem 0;
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--dark-color);
        }

        .report-header h2 {
            margin: 0 0 0.5rem 0;
            font-size: 1.25rem;
            font-weight: 500;
            color: var(--primary-color);
        }

        .report-header p {
            margin: 0 0 1rem 0;
            color: var(--muted-color);
            font-size: 0.9rem;
        }

        .report-summary {
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
        }

        .report-summary span {
            background: var(--light-color);
            padding: 0.5rem 1rem;
            border-radius: var(--border-radius);
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--dark-color);
            border: 1px solid var(--border-color);
        }

        /* Trade List Panel */
        .trade-list-panel {
            grid-area: trade-list;
            background: white;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .trade-list-header {
            background: linear-gradient(135deg, var(--primary-color), #0056b3);
            color: white;
            padding: 1rem;
            font-size: 1.1rem;
            font-weight: 600;
            text-align: center;
        }

        .trade-filters {
            padding: 1rem;
            background: var(--light-color);
            border-bottom: 1px solid var(--border-color);
        }

        .filter-group {
            margin-bottom: 0.75rem;
        }

        .filter-group:last-child {
            margin-bottom: 0;
        }

        .filter-group label {
            display: block;
            margin-bottom: 0.25rem;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--muted-color);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .filter-group select,
        .filter-group input {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            font-size: 0.875rem;
            transition: all var(--transition-speed) ease;
            background: white;
        }

        .filter-group select:focus,
        .filter-group input:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }

        /* Trade List */
        .trade-list {
            flex: 1;
            overflow-y: auto;
            padding: 0.5rem;
        }

        /* Enhanced Trade Item Styles */
        .trade-item {
            background: white;
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            margin-bottom: 0.5rem;
            padding: 1rem;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            position: relative;
            overflow: hidden;
        }

        .trade-item::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background: var(--muted-color);
            transition: all var(--transition-speed) ease;
        }

        .trade-item.profit::before {
            background: var(--success-color);
        }

        .trade-item.loss::before {
            background: var(--danger-color);
        }

        .trade-item.neutral::before {
            background: var(--warning-color);
        }

        .trade-item:hover {
            transform: translateY(-2px);
            box-shadow: var(--box-shadow-hover);
            border-color: var(--primary-color);
        }

        .trade-item:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.25);
        }

        .trade-item.selected {
            background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
            border-color: var(--primary-color);
            box-shadow: var(--box-shadow-hover);
        }

        .trade-item.highlighted {
            animation: pulse 1s ease-in-out infinite alternate;
            border-color: var(--warning-color);
            box-shadow: 0 0 12px rgba(255, 193, 7, 0.4);
        }

        @keyframes pulse {
            from {
                box-shadow: 0 0 12px rgba(255, 193, 7, 0.4);
            }
            to {
                box-shadow: 0 0 20px rgba(255, 193, 7, 0.6);
            }
        }

        .trade-item-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .trade-number {
            font-weight: 700;
            color: var(--muted-color);
            font-size: 0.875rem;
        }

        .trade-pnl {
            font-weight: 700;
            font-size: 1rem;
        }

        .trade-pnl.profit {
            color: var(--success-color);
        }

        .trade-pnl.loss {
            color: var(--danger-color);
        }

        .trade-pnl.neutral {
            color: var(--warning-color);
        }

        .trade-item-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }

        .trade-detail {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
        }

        .trade-detail .label {
            color: var(--muted-color);
            font-weight: 500;
        }

        .trade-detail .value {
            color: var(--dark-color);
            font-weight: 600;
        }

        .trade-item-indicators {
            margin-top: 0.5rem;
        }

        .trade-progress-bar {
            height: 3px;
            background: var(--light-color);
            border-radius: 2px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            border-radius: 2px;
            transition: width var(--transition-speed) ease;
        }

        .progress-fill.profit {
            background: linear-gradient(90deg, var(--success-color), #20c997);
        }

        .progress-fill.loss {
            background: linear-gradient(90deg, var(--danger-color), #e74c3c);
        }

        .progress-fill.neutral {
            background: linear-gradient(90deg, var(--warning-color), #f39c12);
        }

        .filtered-count {
            padding: 0.75rem 1rem;
            font-size: 0.75rem;
            color: var(--muted-color);
            background: var(--light-color);
            border-top: 1px solid var(--border-color);
            text-align: center;
            font-weight: 500;
        }

        /* Chart Panel */
        .chart-panel {
            grid-area: chart;
            background: white;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            display: flex;
            flex-direction: column;
            position: relative;
        }

        .chart-controls {
            padding: 1rem;
            background: var(--light-color);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: center;
        }

        .chart-control {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .chart-control input[type="checkbox"] {
            width: 18px;
            height: 18px;
            accent-color: var(--primary-color);
        }

        .chart-control label {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--dark-color);
            cursor: pointer;
        }

        .main-chart {
            flex: 1;
            position: relative;
            background: white;
            border-radius: 0 0 var(--border-radius) var(--border-radius);
        }

        /* Enhanced Chart Marker Styles */
        .chart-marker {
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
        }

        .chart-marker:hover {
            transform: scale(1.2);
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
        }

        .chart-marker:focus {
            outline: 2px solid var(--focus-color);
            outline-offset: 2px;
        }

        .chart-marker.highlighted {
            animation: chartMarkerPulse 1.5s ease-in-out infinite;
            z-index: 1000;
            transform: scale(1.3);
        }

        @keyframes chartMarkerPulse {
            0%, 100% {
                filter: drop-shadow(0 0 8px currentColor);
            }
            50% {
                filter: drop-shadow(0 0 16px currentColor);
            }
        }

        .chart-marker.selected {
            transform: scale(1.4);
            filter: drop-shadow(0 0 12px var(--primary-color));
            z-index: 999;
        }

        /* Trade Details Panel */
        .trade-details-panel {
            grid-area: trade-details;
            background: white;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .trade-details-container {
            flex: 1;
            position: relative;
        }

        .trade-details-state {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            padding: 1.5rem;
            opacity: 0;
            visibility: hidden;
            transition: all var(--transition-speed) ease;
            overflow-y: auto;
        }

        .trade-details-state.active {
            opacity: 1;
            visibility: visible;
        }

        .trade-details-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--border-color);
        }

        .trade-details-header h3 {
            margin: 0;
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--dark-color);
        }

        .close-button {
            background: none;
            border: none;
            font-size: 1.5rem;
            color: var(--muted-color);
            cursor: pointer;
            padding: 0.25rem;
            border-radius: var(--border-radius);
            transition: all var(--transition-speed) ease;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .close-button:hover {
            background: var(--danger-color);
            color: white;
        }

        .trade-detail-grid {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .detail-section {
            background: var(--light-color);
            padding: 1rem;
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
        }

        .detail-section h4 {
            margin: 0 0 1rem 0;
            font-size: 1rem;
            font-weight: 600;
            color: var(--primary-color);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .detail-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(108, 117, 125, 0.2);
        }

        .detail-item:last-child {
            border-bottom: none;
        }

        .detail-item label {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--muted-color);
        }

        .detail-item span {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--dark-color);
        }

        .detail-item span.profit {
            color: var(--success-color);
        }

        .detail-item span.loss {
            color: var(--danger-color);
        }

        .pnl-value {
            font-size: 1.1rem !important;
            font-weight: 700 !important;
        }

        .trade-actions {
            margin-top: 1.5rem;
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        /* Navigation Controls */
        .navigation-controls {
            padding: 1rem;
            background: var(--light-color);
            border-top: 1px solid var(--border-color);
        }

        .nav-section {
            margin-bottom: 1rem;
        }

        .nav-section:last-child {
            margin-bottom: 0;
        }

        .nav-section h4 {
            margin: 0 0 0.5rem 0;
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--muted-color);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .nav-buttons {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .nav-button {
            background: white;
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            padding: 0.5rem;
            font-size: 1rem;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .nav-button:hover {
            background: var(--primary-color);
            color: white;
            transform: translateY(-1px);
        }

        .nav-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .nav-button:disabled:hover {
            background: white;
            color: var(--dark-color);
        }

        #current-trade-indicator {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--dark-color);
            padding: 0.5rem;
            background: white;
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
        }

        .quick-actions {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }

        .jump-to-trade {
            display: flex;
            gap: 0.5rem;
        }

        #jump-trade-input {
            flex: 1;
            padding: 0.5rem;
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            font-size: 0.875rem;
        }

        /* Button Styles */
        .action-button {
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: var(--border-radius);
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            text-align: center;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .action-button:hover {
            background: #0056b3;
            transform: translateY(-1px);
            box-shadow: var(--box-shadow-hover);
        }

        .action-button:active {
            transform: translateY(0);
        }

        .action-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .action-button.secondary {
            background: var(--muted-color);
        }

        .action-button.secondary:hover {
            background: #5a6268;
        }

        .action-button.success {
            background: var(--success-color);
        }

        .action-button.success:hover {
            background: #1e7e34;
        }

        .action-button.danger {
            background: var(--danger-color);
        }

        .action-button.danger:hover {
            background: #c82333;
        }

        /* Responsive Design */
        @media (max-width: 1400px) {
            .interactive-report {
                grid-template-columns: 280px 1fr 320px;
            }
        }

        @media (max-width: 1200px) {
            .interactive-report {
                grid-template-columns: 250px 1fr 280px;
            }

            .chart-controls {
                flex-direction: column;
                align-items: stretch;
            }
        }

        @media (max-width: 992px) {
            .interactive-report {
                grid-template-columns: 1fr;
                grid-template-rows: auto auto 1fr auto;
                grid-template-areas:
                    "header"
                    "chart"
                    "trade-list"
                    "trade-details";
                height: auto;
                min-height: 100vh;
            }

            .trade-list-panel,
            .trade-details-panel {
                max-height: 400px;
            }

            .chart-panel {
                height: 500px;
            }
        }

        @media (max-width: 768px) {
            .interactive-report {
                padding: 0.5rem;
                gap: 0.5rem;
            }

            .report-header {
                padding: 1rem;
            }

            .report-header h1 {
                font-size: 1.5rem;
            }

            .report-summary {
                flex-direction: column;
                gap: 0.5rem;
            }

            .trade-item {
                padding: 0.75rem;
            }

            .trade-item-details {
                grid-template-columns: 1fr;
                gap: 0.25rem;
            }

            .chart-panel {
                height: 400px;
            }

            .trade-details-panel {
                max-height: 300px;
            }

            .quick-actions {
                grid-template-columns: 1fr;
            }
        }

        /* Accessibility Features */
        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }

        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }

        /* Focus indicators for keyboard navigation */
        .trade-item:focus-visible,
        .action-button:focus-visible,
        .nav-button:focus-visible {
            outline: 3px solid var(--focus-color);
            outline-offset: 2px;
        }

        /* High contrast mode support */
        @media (prefers-contrast: high) {
            :root {
                --border-color: #000;
                --hover-color: #000;
                --box-shadow: 0 2px 4px rgba(0,0,0,0.5);
            }

            .trade-item {
                border-width: 2px;
            }

            .trade-item:hover {
                border-width: 3px;
            }
        }

        /* Print styles */
        @media print {
            .interactive-report {
                display: block;
                background: white;
            }

            .chart-controls,
            .navigation-controls,
            .trade-filters {
                display: none;
            }

            .trade-item {
                break-inside: avoid;
            }
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

    def _generate_interactive_html_components(self, trades: List[Dict[str, Any]], strategy_name: str = "Strategy") -> Dict[str, str]:
        """Generate enhanced HTML template components for interactive features."""

        # Generate interactive trade list with click handlers
        trade_list_items = self._generate_interactive_trade_list(trades)

        # Generate trade details sidebar/popup template
        trade_details_template = self._generate_trade_details_template()

        # Generate trade navigation controls
        navigation_controls = self._generate_navigation_controls()

        # Generate loading states template
        loading_states = self._generate_loading_states_template()

        # Generate error handling displays
        error_handling = self._generate_error_handling_template()

        return {
            "trade_list_items": trade_list_items,
            "trade_details_template": trade_details_template,
            "navigation_controls": navigation_controls,
            "loading_states": loading_states,
            "error_handling": error_handling
        }

    def _generate_interactive_trade_list(self, trades: List[Dict[str, Any]]) -> str:
        """Generate interactive trade list component with click handlers."""
        trade_items = []

        for i, trade in enumerate(trades):
            trade_id = f"trade_{i+1}"
            pnl = trade.get("pnl", 0)
            profit_class = "profit" if pnl > 0 else "loss" if pnl < 0 else "neutral"
            duration = trade.get("duration_hours", 0)

            trade_item = f"""
                <div class="trade-item {profit_class}"
                     data-trade-id="{trade_id}"
                     data-trade-index="{i+1}"
                     data-pnl="{pnl}"
                     data-duration="{duration}"
                     data-strategy="{trade.get('strategy', '')}"
                     data-entry-reason="{trade.get('entry_reason', '')}"
                     data-exit-reason="{trade.get('exit_reason', '')}"
                     tabindex="0"
                     role="button"
                     aria-label="Trade {i+1}, PnL: ${pnl:.2f}, Duration: {duration:.1f}h"
                     onclick="window.interactiveReport?.selectTrade('{trade_id}')"
                     onkeydown="window.interactiveReport?.handleTradeKeydown(event, '{trade_id}')">

                    <div class="trade-item-header">
                        <span class="trade-number">#{i+1}</span>
                        <span class="trade-pnl ${profit_class}">${pnl:.2f}</span>
                    </div>

                    <div class="trade-item-details">
                        <div class="trade-detail">
                            <span class="label">Entry:</span>
                            <span class="value">${trade.get('entry_price', 0):.2f}</span>
                        </div>
                        <div class="trade-detail">
                            <span class="label">Exit:</span>
                            <span class="value">${trade.get('exit_price', 0):.2f}</span>
                        </div>
                        <div class="trade-detail">
                            <span class="label">Duration:</span>
                            <span class="value">{duration:.1f}h</span>
                        </div>
                    </div>

                    <div class="trade-item-indicators">
                        <div class="trade-progress-bar">
                            <div class="progress-fill {profit_class}"
                                 style="width: {min(abs(pnl/100), 1) * 100}%"></div>
                        </div>
                    </div>
                </div>
            """
            trade_items.append(trade_item)

        return "\n".join(trade_items)

    def _generate_trade_details_template(self) -> str:
        """Generate trade details sidebar/popup template."""
        return """
            <div id="trade-details" class="trade-details-container">
                <!-- Default state -->
                <div id="trade-details-default" class="trade-details-state active">
                    <h3>Trade Details</h3>
                    <p>Select a trade from the list or chart to view details</p>
                </div>

                <!-- Loading state -->
                <div id="trade-details-loading" class="trade-details-state">
                    <h3>Loading Trade Details...</h3>
                    <div class="loading-spinner"></div>
                </div>

                <!-- Trade details content -->
                <div id="trade-details-content" class="trade-details-state">
                    <div class="trade-details-header">
                        <h3 id="trade-title">Trade #1</h3>
                        <button id="close-trade-details" class="close-button"
                                aria-label="Close trade details">×</button>
                    </div>

                    <div class="trade-detail-grid">
                        <div class="detail-section">
                            <h4>Entry Information</h4>
                            <div class="detail-item">
                                <label>Entry Price:</label>
                                <span id="entry-price">-</span>
                            </div>
                            <div class="detail-item">
                                <label>Entry Time:</label>
                                <span id="entry-time">-</span>
                            </div>
                            <div class="detail-item">
                                <label>Entry Reason:</label>
                                <span id="entry-reason">-</span>
                            </div>
                            <div class="detail-item">
                                <label>Position Size:</label>
                                <span id="position-size">-</span>
                            </div>
                        </div>

                        <div class="detail-section">
                            <h4>Exit Information</h4>
                            <div class="detail-item">
                                <label>Exit Price:</label>
                                <span id="exit-price">-</span>
                            </div>
                            <div class="detail-item">
                                <label>Exit Time:</label>
                                <span id="exit-time">-</span>
                            </div>
                            <div class="detail-item">
                                <label>Exit Reason:</label>
                                <span id="exit-reason">-</span>
                            </div>
                            <div class="detail-item">
                                <label>Duration:</label>
                                <span id="trade-duration">-</span>
                            </div>
                        </div>

                        <div class="detail-section">
                            <h4>Performance</h4>
                            <div class="detail-item">
                                <label>PnL:</label>
                                <span id="trade-pnl" class="pnl-value">-</span>
                            </div>
                            <div class="detail-item">
                                <label>Return %:</label>
                                <span id="trade-return">-</span>
                            </div>
                            <div class="detail-item">
                                <label>Strategy:</label>
                                <span id="trade-strategy">-</span>
                            </div>
                        </div>
                    </div>

                    <div class="trade-actions">
                        <button id="zoom-to-trade-btn" class="action-button">
                            📍 Zoom to Trade
                        </button>
                        <button id="highlight-trade-btn" class="action-button">
                            🔆 Highlight on Chart
                        </button>
                    </div>
                </div>

                <!-- Error state -->
                <div id="trade-details-error" class="trade-details-state">
                    <h3>Error Loading Trade</h3>
                    <p>Unable to load trade details. Please try again.</p>
                    <button id="retry-trade-details" class="action-button">Retry</button>
                </div>
            </div>
        """

    def _generate_navigation_controls(self) -> str:
        """Generate trade navigation controls (previous/next/filter)."""
        return """
            <div class="navigation-controls">
                <div class="nav-section">
                    <h4>Trade Navigation</h4>
                    <div class="nav-buttons">
                        <button id="first-trade-btn" class="nav-button"
                                aria-label="Go to first trade">⏮️</button>
                        <button id="prev-trade-btn" class="nav-button"
                                aria-label="Go to previous trade">⏪</button>
                        <span id="current-trade-indicator">No trade selected</span>
                        <button id="next-trade-btn" class="nav-button"
                                aria-label="Go to next trade">⏩</button>
                        <button id="last-trade-btn" class="nav-button"
                                aria-label="Go to last trade">⏭️</button>
                    </div>
                </div>

                <div class="nav-section">
                    <h4>Quick Actions</h4>
                    <div class="quick-actions">
                        <button id="show-all-trades-btn" class="action-button">
                            👁️ Show All
                        </button>
                        <button id="hide-losing-trades-btn" class="action-button">
                            🚫 Hide Losing
                        </button>
                        <button id="show-profitable-only-btn" class="action-button">
                            💰 Profitable Only
                        </button>
                        <button id="reset-filters-btn" class="action-button">
                            🔄 Reset Filters
                        </button>
                    </div>
                </div>

                <div class="nav-section">
                    <h4>Jump to Trade</h4>
                    <div class="jump-to-trade">
                        <input type="number" id="jump-trade-input"
                               placeholder="Trade #" min="1" max="{total_trades}">
                        <button id="jump-to-trade-btn" class="action-button">Go</button>
                    </div>
                </div>
            </div>
        """

    def _generate_loading_states_template(self) -> str:
        """Generate loading states for trade data and chart updates."""
        return """
            <!-- Global Loading Overlay -->
            <div id="global-loading" class="loading-overlay hidden">
                <div class="loading-content">
                    <div class="loading-spinner large"></div>
                    <h3>Loading Interactive Report...</h3>
                    <p id="loading-status">Initializing components...</p>
                </div>
            </div>

            <!-- Chart Loading State -->
            <div id="chart-loading" class="chart-loading-state hidden">
                <div class="loading-content">
                    <div class="loading-spinner"></div>
                    <p>Loading chart data...</p>
                </div>
            </div>

            <!-- Trade List Loading State -->
            <div id="trade-list-loading" class="trade-list-loading-state hidden">
                <div class="loading-content">
                    <div class="loading-spinner small"></div>
                    <p>Loading trades...</p>
                </div>
            </div>

            <style>
                .loading-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(255, 255, 255, 0.9);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 9999;
                    backdrop-filter: blur(2px);
                }

                .loading-content {
                    text-align: center;
                    padding: 2rem;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }

                .loading-spinner {
                    border: 3px solid #f3f3f3;
                    border-top: 3px solid #007bff;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                    margin: 0 auto 1rem;
                }

                .loading-spinner.large {
                    width: 60px;
                    height: 60px;
                    border-width: 4px;
                }

                .loading-spinner.small {
                    width: 24px;
                    height: 24px;
                    border-width: 2px;
                }

                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }

                .hidden {
                    display: none !important;
                }

                .chart-loading-state,
                .trade-list-loading-state {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: rgba(255, 255, 255, 0.95);
                    padding: 1rem;
                    border-radius: 6px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
            </style>
        """

    def _generate_error_handling_template(self) -> str:
        """Generate error handling displays for JavaScript failures."""
        return """
            <!-- Global Error Handler -->
            <div id="global-error" class="error-overlay hidden">
                <div class="error-content">
                    <h3>⚠️ Something went wrong</h3>
                    <p id="error-message">An unexpected error occurred</p>
                    <div class="error-actions">
                        <button id="retry-error-btn" class="error-button primary">
                            🔄 Retry
                        </button>
                        <button id="reload-error-btn" class="error-button secondary">
                            ♻️ Reload Page
                        </button>
                        <button id="dismiss-error-btn" class="error-button tertiary">
                            ✕ Dismiss
                        </button>
                    </div>
                    <details class="error-details">
                        <summary>Technical Details</summary>
                        <pre id="error-stack"></pre>
                    </details>
                </div>
            </div>

            <!-- Chart Error State -->
            <div id="chart-error" class="chart-error-state hidden">
                <div class="error-content">
                    <h4>📊 Chart Error</h4>
                    <p>Unable to load chart. Please check your browser compatibility.</p>
                    <button id="retry-chart-btn" class="error-button">Retry Chart</button>
                </div>
            </div>

            <!-- JavaScript Disabled Warning -->
            <noscript>
                <div class="noscript-warning">
                    <h3>JavaScript Required</h3>
                    <p>This interactive report requires JavaScript to function properly.
                       Please enable JavaScript in your browser and refresh the page.</p>
                </div>
            </noscript>

            <style>
                .error-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(220, 53, 69, 0.1);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 10000;
                    backdrop-filter: blur(2px);
                }

                .error-content {
                    background: white;
                    border: 2px solid #dc3545;
                    border-radius: 8px;
                    padding: 2rem;
                    max-width: 500px;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(220, 53, 69, 0.2);
                }

                .error-content h3 {
                    color: #dc3545;
                    margin: 0 0 1rem 0;
                }

                .error-actions {
                    margin: 1.5rem 0;
                    display: flex;
                    gap: 0.5rem;
                    justify-content: center;
                    flex-wrap: wrap;
                }

                .error-button {
                    padding: 0.5rem 1rem;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 14px;
                    transition: all 0.2s ease;
                }

                .error-button.primary {
                    background: #dc3545;
                    color: white;
                }

                .error-button.primary:hover {
                    background: #c82333;
                }

                .error-button.secondary {
                    background: #6c757d;
                    color: white;
                }

                .error-button.secondary:hover {
                    background: #5a6268;
                }

                .error-button.tertiary {
                    background: #f8f9fa;
                    color: #6c757d;
                    border: 1px solid #dee2e6;
                }

                .error-button.tertiary:hover {
                    background: #e2e6ea;
                }

                .error-details {
                    margin-top: 1rem;
                    text-align: left;
                }

                .error-details summary {
                    cursor: pointer;
                    font-weight: 500;
                    margin-bottom: 0.5rem;
                }

                .error-details pre {
                    background: #f8f9fa;
                    padding: 0.5rem;
                    border-radius: 4px;
                    font-size: 12px;
                    overflow-x: auto;
                    max-height: 150px;
                }

                .chart-error-state {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: white;
                    border: 1px solid #dc3545;
                    border-radius: 6px;
                    padding: 1rem;
                    text-align: center;
                    box-shadow: 0 2px 8px rgba(220, 53, 69, 0.1);
                }

                .noscript-warning {
                    background: #fff3cd;
                    border: 1px solid #ffeaa7;
                    color: #856404;
                    padding: 1rem;
                    margin: 1rem;
                    border-radius: 8px;
                    text-align: center;
                }
            </style>
        """
