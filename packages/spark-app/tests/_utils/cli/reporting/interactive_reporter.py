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
                    "javascript_components": self._generate_javascript_components(trades),
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

    def _generate_javascript_components(self, trades: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate JavaScript code for interactive features."""

        # Trade selector JavaScript
        trade_selector_js = """
        class TradeSelector {
            constructor(tradeData, chartInstance) {
                this.trades = tradeData;
                this.chart = chartInstance;
                this.selectedTrade = null;
                this.initializeEventListeners();
            }

            initializeEventListeners() {
                // Trade list click handlers
                document.querySelectorAll('.trade-item').forEach(item => {
                    item.addEventListener('click', (e) => {
                        const tradeId = e.currentTarget.dataset.tradeId;
                        this.selectTrade(tradeId);
                    });
                });

                // Chart marker click handlers
                this.chart.on('click', (event) => {
                    if (event.target && event.target.dataset.tradeId) {
                        this.selectTrade(event.target.dataset.tradeId);
                    }
                });

                // Keyboard navigation
                document.addEventListener('keydown', (e) => {
                    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
                        this.navigateTrades(e.key === 'ArrowUp' ? -1 : 1);
                        e.preventDefault();
                    }
                });
            }

            selectTrade(tradeId) {
                // Clear previous selection
                this.clearSelection();

                // Highlight trade in list
                const tradeItem = document.querySelector(`[data-trade-id="${tradeId}"]`);
                if (tradeItem) {
                    tradeItem.classList.add('selected');
                    tradeItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }

                // Highlight markers on chart
                this.highlightTradeMarkers(tradeId);

                // Show trade details
                this.showTradeDetails(tradeId);

                // Zoom to trade if enabled
                if (this.shouldZoomToTrade()) {
                    this.zoomToTrade(tradeId);
                }

                this.selectedTrade = tradeId;
            }

            clearSelection() {
                document.querySelectorAll('.trade-item.selected').forEach(item => {
                    item.classList.remove('selected');
                });

                document.querySelectorAll('.chart-marker.highlighted').forEach(marker => {
                    marker.classList.remove('highlighted');
                });
            }

            highlightTradeMarkers(tradeId) {
                const markers = document.querySelectorAll(`[data-trade-id="${tradeId}"]`);
                markers.forEach(marker => {
                    marker.classList.add('highlighted');
                });
            }

            showTradeDetails(tradeId) {
                const trade = this.trades.find(t => t.id === tradeId);
                if (trade) {
                    const detailsPanel = document.getElementById('trade-details');
                    detailsPanel.innerHTML = this.generateTradeDetailsHTML(trade);
                    detailsPanel.style.display = 'block';
                }
            }

            zoomToTrade(tradeId) {
                const trade = this.trades.find(t => t.id === tradeId);
                if (trade && this.chart) {
                    const startTime = new Date(trade.entry_time);
                    const endTime = new Date(trade.exit_time);
                    const padding = (endTime - startTime) * 0.2; // 20% padding

                    this.chart.zoomToTimeRange(
                        new Date(startTime.getTime() - padding),
                        new Date(endTime.getTime() + padding)
                    );
                }
            }

            navigateTrades(direction) {
                const currentIndex = this.selectedTrade ?
                    this.trades.findIndex(t => t.id === this.selectedTrade) : -1;
                const newIndex = Math.max(0, Math.min(this.trades.length - 1, currentIndex + direction));

                if (newIndex !== currentIndex && this.trades[newIndex]) {
                    this.selectTrade(this.trades[newIndex].id);
                }
            }

            generateTradeDetailsHTML(trade) {
                return `
                    <h3>Trade ${trade.index} Details</h3>
                    <div class="trade-detail-grid">
                        <div class="detail-item">
                            <label>Entry Time:</label>
                            <span>${trade.entry_time}</span>
                        </div>
                        <div class="detail-item">
                            <label>Exit Time:</label>
                            <span>${trade.exit_time}</span>
                        </div>
                        <div class="detail-item">
                            <label>Duration:</label>
                            <span>${Math.round(trade.duration_minutes / 60 * 100) / 100} hours</span>
                        </div>
                        <div class="detail-item">
                            <label>Entry Price:</label>
                            <span>$${trade.entry_price.toFixed(2)}</span>
                        </div>
                        <div class="detail-item">
                            <label>Exit Price:</label>
                            <span>$${trade.exit_price.toFixed(2)}</span>
                        </div>
                        <div class="detail-item">
                            <label>P&L:</label>
                            <span class="${trade.pnl >= 0 ? 'profit' : 'loss'}">
                                ${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}
                            </span>
                        </div>
                        <div class="detail-item">
                            <label>Position Size:</label>
                            <span>$${trade.position_size.toFixed(2)}</span>
                        </div>
                        <div class="detail-item">
                            <label>Exit Reason:</label>
                            <span>${trade.exit_reason}</span>
                        </div>
                    </div>
                `;
            }

            shouldZoomToTrade() {
                return document.getElementById('zoom-to-trade-checkbox')?.checked || false;
            }
        }
        """

        # Chart highlighter JavaScript
        chart_highlighter_js = """
        class ChartHighlighter {
            constructor(chartInstance) {
                this.chart = chartInstance;
                this.highlightedElements = new Set();
            }

            highlightTrade(tradeId) {
                this.clearHighlights();

                const elements = this.chart.querySelectorAll(`[data-trade-id="${tradeId}"]`);
                elements.forEach(element => {
                    element.classList.add('highlighted');
                    this.highlightedElements.add(element);
                });

                // Add glow effect
                this.addGlowEffect(elements);
            }

            clearHighlights() {
                this.highlightedElements.forEach(element => {
                    element.classList.remove('highlighted');
                    this.removeGlowEffect(element);
                });
                this.highlightedElements.clear();
            }

            addGlowEffect(elements) {
                elements.forEach(element => {
                    element.style.filter = 'drop-shadow(0 0 6px currentColor)';
                    element.style.zIndex = '1000';
                });
            }

            removeGlowEffect(element) {
                element.style.filter = '';
                element.style.zIndex = '';
            }
        }
        """

        # Trade filter JavaScript
        trade_filter_js = """
        class TradeFilter {
            constructor(trades) {
                this.allTrades = trades;
                this.filteredTrades = trades;
                this.filters = {
                    profitability: 'all', // 'all', 'profitable', 'losing'
                    duration: 'all',      // 'all', 'short', 'medium', 'long'
                    search: ''
                };
                this.initializeFilters();
            }

            initializeFilters() {
                // Profitability filter
                document.getElementById('profit-filter')?.addEventListener('change', (e) => {
                    this.filters.profitability = e.target.value;
                    this.applyFilters();
                });

                // Duration filter
                document.getElementById('duration-filter')?.addEventListener('change', (e) => {
                    this.filters.duration = e.target.value;
                    this.applyFilters();
                });

                // Search filter
                document.getElementById('trade-search')?.addEventListener('input', (e) => {
                    this.filters.search = e.target.value.toLowerCase();
                    this.applyFilters();
                });
            }

            applyFilters() {
                this.filteredTrades = this.allTrades.filter(trade => {
                    // Profitability filter
                    if (this.filters.profitability === 'profitable' && trade.pnl <= 0) return false;
                    if (this.filters.profitability === 'losing' && trade.pnl >= 0) return false;

                    // Duration filter
                    const durationHours = trade.duration_minutes / 60;
                    if (this.filters.duration === 'short' && durationHours > 4) return false;
                    if (this.filters.duration === 'medium' && (durationHours <= 4 || durationHours > 24)) return false;
                    if (this.filters.duration === 'long' && durationHours <= 24) return false;

                    // Search filter
                    if (this.filters.search) {
                        const searchText = `${trade.entry_time} ${trade.exit_time} ${trade.exit_reason}`.toLowerCase();
                        if (!searchText.includes(this.filters.search)) return false;
                    }

                    return true;
                });

                this.updateTradeList();
                this.updateChartVisibility();
            }

            updateTradeList() {
                const tradeList = document.getElementById('trade-list');
                if (!tradeList) return;

                // Hide all trades
                tradeList.querySelectorAll('.trade-item').forEach(item => {
                    item.style.display = 'none';
                });

                // Show filtered trades
                this.filteredTrades.forEach(trade => {
                    const item = document.querySelector(`[data-trade-id="${trade.id}"]`);
                    if (item) {
                        item.style.display = 'block';
                    }
                });

                // Update count
                const countElement = document.getElementById('filtered-count');
                if (countElement) {
                    countElement.textContent = `${this.filteredTrades.length} of ${this.allTrades.length} trades`;
                }
            }

            updateChartVisibility() {
                // Hide all chart markers
                document.querySelectorAll('.chart-marker').forEach(marker => {
                    marker.style.display = 'none';
                });

                // Show markers for filtered trades
                this.filteredTrades.forEach(trade => {
                    document.querySelectorAll(`[data-trade-id="${trade.id}"]`).forEach(marker => {
                        marker.style.display = 'block';
                    });
                });
            }
        }
        """

        # Main initialization JavaScript
        main_js = f"""
        document.addEventListener('DOMContentLoaded', function() {{
            // Initialize trade data
            const tradeData = {json.dumps([self._format_trade_for_js(trade, i) for i, trade in enumerate(trades)])};

            // Initialize chart (placeholder - would integrate with actual charting library)
            const chartElement = document.getElementById('main-chart');

            // Initialize components
            const tradeSelector = new TradeSelector(tradeData, chartElement);
            const chartHighlighter = new ChartHighlighter(chartElement);
            const tradeFilter = new TradeFilter(tradeData);

            // Global functions for external access
            window.sparkStackerInteractive = {{
                selectTrade: (tradeId) => tradeSelector.selectTrade(tradeId),
                clearSelection: () => tradeSelector.clearSelection(),
                filterTrades: (filters) => tradeFilter.applyFilters(filters),
                highlightTrade: (tradeId) => chartHighlighter.highlightTrade(tradeId)
            }};

            console.log('Spark Stacker Interactive Report initialized');
        }});
        """

        return {
            "trade_selector": trade_selector_js,
            "chart_highlighter": chart_highlighter_js,
            "trade_filter": trade_filter_js,
            "main_initialization": main_js
        }

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
            js_components["trade_selector"],
            js_components["chart_highlighter"],
            js_components["trade_filter"],
            js_components["main_initialization"]
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
