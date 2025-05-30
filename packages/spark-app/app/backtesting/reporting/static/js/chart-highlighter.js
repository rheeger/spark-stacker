/**
 * Chart Highlighter Module
 *
 * Handles chart marker highlighting and visual feedback including:
 * - Trade marker highlighting on charts
 * - Visual effects (glow, pulsing, etc.)
 * - Chart element state management
 * - Integration with chart libraries (Plotly, etc.)
 */

class ChartHighlighter {
  constructor(config = {}) {
    this.highlightedElements = new Set();
    this.chart = null;
    this.config = {
      highlightClass: 'highlighted',
      glowEffect: true,
      glowColor: '#2196f3',
      glowSize: '6px',
      pulseAnimation: false,
      animationDuration: '0.3s',
      zIndexHighlight: 1000,
      markerSelector: '[data-trade-id]',
      entryMarkerClass: 'entry-marker',
      exitMarkerClass: 'exit-marker',
      ...config,
    };

    this.animationFrameId = null;
    this.initialized = false;
  }

  /**
   * Initialize the chart highlighter with chart instance
   * @param {Object} chartInstance - Chart instance (Plotly, D3, etc.)
   * @param {Object} dependencies - Other module instances
   */
  initialize(chartInstance = null, dependencies = {}) {
    if (this.initialized) {
      console.warn('ChartHighlighter already initialized');
      return;
    }

    this.chart = chartInstance;
    this.tradeSelector = dependencies.tradeSelector;

    this.setupEventListeners();
    this.injectStyles();
    this.initialized = true;

    console.log('ChartHighlighter initialized');
  }

  /**
   * Set up event listeners for chart interaction
   */
  setupEventListeners() {
    // Listen for chart clicks if chart instance is available
    if (this.chart && typeof this.chart.on === 'function') {
      this.chart.on('plotly_click', (data) => {
        this.handleChartClick(data);
      });

      this.chart.on('plotly_hover', (data) => {
        this.handleChartHover(data);
      });

      this.chart.on('plotly_unhover', (data) => {
        this.handleChartUnhover(data);
      });
    }

    // Listen for direct DOM clicks on chart markers
    document.addEventListener('click', (event) => {
      if (event.target.matches(this.config.markerSelector)) {
        this.handleMarkerClick(event);
      }
    });

    // Listen for trade selection events from other modules
    document.addEventListener('tradeSelected', (event) => {
      this.highlightTrade(event.detail.tradeId);
    });

    document.addEventListener('tradeDeselected', () => {
      this.clearHighlights();
    });
  }

  /**
   * Inject CSS styles for highlighting effects
   */
  injectStyles() {
    const styleId = 'chart-highlighter-styles';

    // Remove existing styles
    const existingStyles = document.getElementById(styleId);
    if (existingStyles) {
      existingStyles.remove();
    }

    const styles = `
            .${this.config.highlightClass} {
                filter: drop-shadow(0 0 ${this.config.glowSize} ${
      this.config.glowColor
    }) !important;
                z-index: ${this.config.zIndexHighlight} !important;
                transition: all ${
                  this.config.animationDuration
                } ease !important;
                transform: scale(1.1) !important;
            }

            .${this.config.highlightClass}.${this.config.entryMarkerClass} {
                filter: drop-shadow(0 0 ${
                  this.config.glowSize
                } #4caf50) !important;
            }

            .${this.config.highlightClass}.${this.config.exitMarkerClass} {
                filter: drop-shadow(0 0 ${
                  this.config.glowSize
                } #f44336) !important;
            }

            ${
              this.config.pulseAnimation
                ? `
            .${this.config.highlightClass}.pulse {
                animation: chart-highlight-pulse 2s infinite;
            }

            @keyframes chart-highlight-pulse {
                0% { opacity: 1; transform: scale(1.1); }
                50% { opacity: 0.7; transform: scale(1.05); }
                100% { opacity: 1; transform: scale(1.1); }
            }
            `
                : ''
            }

            .chart-marker-tooltip {
                position: absolute;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none;
                z-index: ${this.config.zIndexHighlight + 1};
                opacity: 0;
                transition: opacity ${this.config.animationDuration} ease;
            }

            .chart-marker-tooltip.visible {
                opacity: 1;
            }

            .trade-sequence-line {
                stroke: ${this.config.glowColor};
                stroke-width: 2;
                stroke-dasharray: 5,5;
                opacity: 0.7;
                transition: all ${this.config.animationDuration} ease;
            }

            .trade-sequence-line.highlighted {
                stroke-width: 3;
                opacity: 1;
                stroke-dasharray: none;
            }
        `;

    const styleElement = document.createElement('style');
    styleElement.id = styleId;
    styleElement.textContent = styles;
    document.head.appendChild(styleElement);
  }

  /**
   * Highlight a specific trade on the chart
   * @param {string} tradeId - Trade identifier
   */
  highlightTrade(tradeId) {
    try {
      // Clear previous highlights
      this.clearHighlights();

      // Find all elements related to this trade
      const tradeElements = document.querySelectorAll(
        `[data-trade-id="${tradeId}"]`
      );

      if (tradeElements.length === 0) {
        console.warn(`No chart elements found for trade: ${tradeId}`);
        return;
      }

      // Highlight each element
      tradeElements.forEach((element) => {
        this.highlightElement(element);
        this.highlightedElements.add(element);
      });

      // Add trade sequence visualization if we have entry and exit markers
      this.addTradeSequenceVisualization(tradeId);

      // Update chart layout if needed (for Plotly charts)
      this.updateChartLayout(tradeId);

      console.log(`Trade highlighted on chart: ${tradeId}`);
    } catch (error) {
      console.error('Error highlighting trade on chart:', error);
    }
  }

  /**
   * Highlight a single element
   * @param {HTMLElement} element - Element to highlight
   */
  highlightElement(element) {
    element.classList.add(this.config.highlightClass);

    // Add specific marker type highlighting
    if (element.classList.contains(this.config.entryMarkerClass)) {
      element.classList.add(this.config.entryMarkerClass);
    } else if (element.classList.contains(this.config.exitMarkerClass)) {
      element.classList.add(this.config.exitMarkerClass);
    }

    // Add pulse animation if configured
    if (this.config.pulseAnimation) {
      element.classList.add('pulse');
    }

    // Bring to front
    element.style.zIndex = this.config.zIndexHighlight;
  }

  /**
   * Clear all current highlights
   */
  clearHighlights() {
    this.highlightedElements.forEach((element) => {
      this.unhighlightElement(element);
    });
    this.highlightedElements.clear();

    // Remove trade sequence lines
    this.removeTradeSequenceVisualization();

    // Hide tooltips
    this.hideAllTooltips();
  }

  /**
   * Remove highlighting from a single element
   * @param {HTMLElement} element - Element to unhighlight
   */
  unhighlightElement(element) {
    element.classList.remove(this.config.highlightClass);
    element.classList.remove('pulse');
    element.style.zIndex = '';
    element.style.filter = '';
    element.style.transform = '';
  }

  /**
   * Add visualization line connecting trade entry and exit
   * @param {string} tradeId - Trade identifier
   */
  addTradeSequenceVisualization(tradeId) {
    if (!this.config.showSequenceLines) {
      return;
    }

    const entryMarker = document.querySelector(
      `[data-trade-id="${tradeId}"].${this.config.entryMarkerClass}`
    );
    const exitMarker = document.querySelector(
      `[data-trade-id="${tradeId}"].${this.config.exitMarkerClass}`
    );

    if (entryMarker && exitMarker) {
      this.drawSequenceLine(entryMarker, exitMarker, tradeId);
    }
  }

  /**
   * Draw a line connecting entry and exit markers
   * @param {HTMLElement} entryMarker - Entry marker element
   * @param {HTMLElement} exitMarker - Exit marker element
   * @param {string} tradeId - Trade identifier
   */
  drawSequenceLine(entryMarker, exitMarker, tradeId) {
    const entryRect = entryMarker.getBoundingClientRect();
    const exitRect = exitMarker.getBoundingClientRect();
    const chartContainer = this.getChartContainer();

    if (!chartContainer) {
      return;
    }

    const containerRect = chartContainer.getBoundingClientRect();

    // Calculate relative positions
    const startX = entryRect.left + entryRect.width / 2 - containerRect.left;
    const startY = entryRect.top + entryRect.height / 2 - containerRect.top;
    const endX = exitRect.left + exitRect.width / 2 - containerRect.left;
    const endY = exitRect.top + exitRect.height / 2 - containerRect.top;

    // Create SVG line
    const svg = this.getOrCreateSequenceSVG();
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');

    line.setAttribute('x1', startX);
    line.setAttribute('y1', startY);
    line.setAttribute('x2', endX);
    line.setAttribute('y2', endY);
    line.setAttribute('class', 'trade-sequence-line highlighted');
    line.setAttribute('data-trade-id', tradeId);

    svg.appendChild(line);
  }

  /**
   * Get or create SVG overlay for sequence lines
   * @returns {SVGElement} SVG element
   */
  getOrCreateSequenceSVG() {
    let svg = document.getElementById('trade-sequence-svg');

    if (!svg) {
      const chartContainer = this.getChartContainer();
      if (!chartContainer) {
        return null;
      }

      svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.id = 'trade-sequence-svg';
      svg.style.position = 'absolute';
      svg.style.top = '0';
      svg.style.left = '0';
      svg.style.width = '100%';
      svg.style.height = '100%';
      svg.style.pointerEvents = 'none';
      svg.style.zIndex = this.config.zIndexHighlight - 1;

      chartContainer.style.position = 'relative';
      chartContainer.appendChild(svg);
    }

    return svg;
  }

  /**
   * Remove trade sequence visualization
   */
  removeTradeSequenceVisualization() {
    const svg = document.getElementById('trade-sequence-svg');
    if (svg) {
      svg.innerHTML = ''; // Clear all lines
    }
  }

  /**
   * Get the main chart container element
   * @returns {HTMLElement} Chart container element
   */
  getChartContainer() {
    // Try common chart container selectors
    const selectors = [
      '.chart-container',
      '.plotly-graph-div',
      '#chart',
      '.js-plotly-plot',
      '.chart',
    ];

    for (const selector of selectors) {
      const container = document.querySelector(selector);
      if (container) {
        return container;
      }
    }

    return null;
  }

  /**
   * Handle chart click events
   * @param {Object} data - Chart click data
   */
  handleChartClick(data) {
    if (data.points && data.points.length > 0) {
      const point = data.points[0];
      const tradeId = point.customdata?.tradeId;

      if (tradeId && this.tradeSelector) {
        this.tradeSelector.selectTrade(tradeId);
      }
    }
  }

  /**
   * Handle chart hover events
   * @param {Object} data - Chart hover data
   */
  handleChartHover(data) {
    if (data.points && data.points.length > 0) {
      const point = data.points[0];
      const tradeId = point.customdata?.tradeId;

      if (tradeId) {
        this.showTradeTooltip(point, tradeId);
      }
    }
  }

  /**
   * Handle chart unhover events
   * @param {Object} data - Chart unhover data
   */
  handleChartUnhover(data) {
    this.hideAllTooltips();
  }

  /**
   * Handle direct marker click events
   * @param {Event} event - Click event
   */
  handleMarkerClick(event) {
    const tradeId = event.target.dataset.tradeId;

    if (tradeId && this.tradeSelector) {
      this.tradeSelector.selectTrade(tradeId);
    }
  }

  /**
   * Show tooltip for a trade marker
   * @param {Object} point - Chart point data
   * @param {string} tradeId - Trade identifier
   */
  showTradeTooltip(point, tradeId) {
    const tooltip = this.getOrCreateTooltip();
    const tradeData = this.getTradeData(tradeId);

    if (!tradeData) {
      return;
    }

    tooltip.innerHTML = this.generateTooltipContent(tradeData);
    tooltip.classList.add('visible');

    // Position tooltip near the mouse/point
    this.positionTooltip(tooltip, point);
  }

  /**
   * Get or create tooltip element
   * @returns {HTMLElement} Tooltip element
   */
  getOrCreateTooltip() {
    let tooltip = document.getElementById('chart-marker-tooltip');

    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.id = 'chart-marker-tooltip';
      tooltip.className = 'chart-marker-tooltip';
      document.body.appendChild(tooltip);
    }

    return tooltip;
  }

  /**
   * Generate tooltip content for a trade
   * @param {Object} tradeData - Trade data
   * @returns {string} HTML content
   */
  generateTooltipContent(tradeData) {
    const pnl = tradeData.pnl || 0;
    const pnlClass = pnl >= 0 ? 'profit' : 'loss';
    const pnlSign = pnl >= 0 ? '+' : '';

    return `
            <div class="tooltip-header">Trade ${tradeData.index || 'N/A'}</div>
            <div class="tooltip-content">
                <div>P&L: <span class="${pnlClass}">${pnlSign}$${pnl.toFixed(
      2
    )}</span></div>
                <div>Entry: $${(tradeData.entry_price || 0).toFixed(2)}</div>
                <div>Exit: $${(tradeData.exit_price || 0).toFixed(2)}</div>
                <div>Duration: ${
                  Math.round(((tradeData.duration_minutes || 0) / 60) * 10) / 10
                }h</div>
            </div>
        `;
  }

  /**
   * Position tooltip relative to chart point
   * @param {HTMLElement} tooltip - Tooltip element
   * @param {Object} point - Chart point data
   */
  positionTooltip(tooltip, point) {
    const chartContainer = this.getChartContainer();
    if (!chartContainer) {
      return;
    }

    const containerRect = chartContainer.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();

    let x = containerRect.left + (point.x || 0);
    let y = containerRect.top + (point.y || 0);

    // Adjust position to keep tooltip in viewport
    if (x + tooltipRect.width > window.innerWidth) {
      x = window.innerWidth - tooltipRect.width - 10;
    }

    if (y - tooltipRect.height < 0) {
      y += 30; // Show below point
    } else {
      y -= tooltipRect.height + 10; // Show above point
    }

    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
  }

  /**
   * Hide all tooltips
   */
  hideAllTooltips() {
    const tooltip = document.getElementById('chart-marker-tooltip');
    if (tooltip) {
      tooltip.classList.remove('visible');
    }
  }

  /**
   * Update chart layout for highlighting (Plotly specific)
   * @param {string} tradeId - Trade identifier
   */
  updateChartLayout(tradeId) {
    if (!this.chart || typeof this.chart.relayout !== 'function') {
      return;
    }

    // This could be used to update chart annotations or layout
    // when a trade is highlighted
  }

  /**
   * Get trade data by ID
   * @param {string} tradeId - Trade identifier
   * @returns {Object|null} Trade data
   */
  getTradeData(tradeId) {
    if (this.tradeSelector) {
      return this.tradeSelector.getTradeData(tradeId);
    }
    return null;
  }

  /**
   * Update configuration
   * @param {Object} newConfig - New configuration options
   */
  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };

    // Re-inject styles if style-related config changed
    if (this.initialized) {
      this.injectStyles();
    }
  }

  /**
   * Clean up resources
   */
  destroy() {
    // Clear highlights
    this.clearHighlights();

    // Remove event listeners
    if (this.chart && typeof this.chart.removeAllListeners === 'function') {
      this.chart.removeAllListeners();
    }

    // Remove injected styles
    const styles = document.getElementById('chart-highlighter-styles');
    if (styles) {
      styles.remove();
    }

    // Remove SVG overlay
    const svg = document.getElementById('trade-sequence-svg');
    if (svg) {
      svg.remove();
    }

    // Remove tooltips
    const tooltip = document.getElementById('chart-marker-tooltip');
    if (tooltip) {
      tooltip.remove();
    }

    // Cancel any pending animations
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }

    // Reset state
    this.chart = null;
    this.highlightedElements.clear();
    this.initialized = false;

    console.log('ChartHighlighter destroyed');
  }

  /**
   * Get module status information
   * @returns {Object} Status information
   */
  getStatus() {
    return {
      initialized: this.initialized,
      hasChart: !!this.chart,
      highlightedCount: this.highlightedElements.size,
      hasTradeSelector: !!this.tradeSelector,
      config: { ...this.config },
    };
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ChartHighlighter;
}

// Global registration for direct script inclusion
if (typeof window !== 'undefined') {
  window.ChartHighlighter = ChartHighlighter;
}
