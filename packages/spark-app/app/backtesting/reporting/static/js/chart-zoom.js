/**
 * Chart Zoom Module
 *
 * Handles zoom-to-trade functionality including:
 * - Automatic zooming to selected trades
 * - Chart interaction and navigation
 * - Zoom state management
 * - Integration with chart libraries
 */

class ChartZoom {
  constructor(config = {}) {
    this.chart = null;
    this.isEnabled = true;
    this.currentZoomState = null;
    this.zoomHistory = [];
    this.historyIndex = -1;
    this.config = {
      zoomEnabled: true,
      autoZoomOnSelect: false,
      zoomPadding: 0.2, // 20% padding around trade timeframe
      animationDuration: 500,
      minZoomDuration: 60000, // 1 minute minimum zoom window (ms)
      maxZoomDuration: 86400000, // 24 hours maximum zoom window (ms)
      zoomToTradeDelay: 100, // Delay before zooming (ms)
      preserveZoomHistory: true,
      maxHistorySize: 20,
      snapToCandles: true,
      showZoomControls: true,
      ...config,
    };

    this.zoomTimer = null;
    this.initialized = false;

    // Bind methods to maintain context
    this.handleTradeSelection = this.handleTradeSelection.bind(this);
    this.handleZoomChange = this.handleZoomChange.bind(this);
  }

  /**
   * Initialize the chart zoom functionality
   * @param {Object} chartInstance - Chart instance (Plotly, D3, etc.)
   * @param {Object} dependencies - Other module instances
   */
  initialize(chartInstance = null, dependencies = {}) {
    if (this.initialized) {
      console.warn('ChartZoom already initialized');
      return;
    }

    this.chart = chartInstance;
    this.tradeSelector = dependencies.tradeSelector;
    this.tradeNavigation = dependencies.tradeNavigation;

    this.setupEventListeners();

    if (this.config.showZoomControls) {
      this.createZoomControls();
    }

    this.detectChartType();
    this.initialized = true;

    console.log('ChartZoom initialized');
  }

  /**
   * Detect chart library type and setup appropriate methods
   */
  detectChartType() {
    if (this.chart) {
      // Plotly chart detection
      if (this.chart.layout && typeof this.chart.relayout === 'function') {
        this.chartType = 'plotly';
        this.setupPlotlyIntegration();
      }
      // D3 chart detection
      else if (this.chart.select && typeof this.chart.select === 'function') {
        this.chartType = 'd3';
        this.setupD3Integration();
      }
      // Generic chart detection
      else {
        this.chartType = 'generic';
        this.setupGenericIntegration();
      }
    } else {
      // Try to find chart automatically
      this.autoDetectChart();
    }
  }

  /**
   * Auto-detect chart from DOM
   */
  autoDetectChart() {
    // Look for Plotly charts
    const plotlyChart = document.querySelector('.js-plotly-plot');
    if (plotlyChart && plotlyChart.data) {
      this.chart = plotlyChart;
      this.chartType = 'plotly';
      this.setupPlotlyIntegration();
      return;
    }

    // Look for common chart containers
    const chartContainer = document.querySelector(
      '.chart-container, #chart, .chart'
    );
    if (chartContainer) {
      this.chart = chartContainer;
      this.chartType = 'generic';
      this.setupGenericIntegration();
    }
  }

  /**
   * Set up Plotly chart integration
   */
  setupPlotlyIntegration() {
    if (this.chart.on) {
      this.chart.on('plotly_relayout', this.handleZoomChange);
    }
  }

  /**
   * Set up D3 chart integration
   */
  setupD3Integration() {
    // D3-specific setup would go here
    console.log('D3 chart integration not fully implemented');
  }

  /**
   * Set up generic chart integration
   */
  setupGenericIntegration() {
    // Generic chart integration using DOM events
    console.log('Generic chart integration active');
  }

  /**
   * Set up event listeners
   */
  setupEventListeners() {
    // Listen for trade selection events
    document.addEventListener('tradeSelected', this.handleTradeSelection);

    // Listen for zoom toggle events
    document.addEventListener('zoomToggled', (event) => {
      this.setEnabled(event.detail.enabled);
    });

    // Listen for manual zoom requests
    document.addEventListener('zoomToTrade', (event) => {
      this.zoomToTrade(event.detail.tradeId, event.detail.tradeData);
    });
  }

  /**
   * Create zoom control UI
   */
  createZoomControls() {
    const existingControls = document.getElementById('chart-zoom-controls');
    if (existingControls) {
      existingControls.remove();
    }

    const controlsContainer = document.createElement('div');
    controlsContainer.id = 'chart-zoom-controls';
    controlsContainer.className = 'chart-zoom-controls';
    controlsContainer.innerHTML = this.getZoomControlsHTML();

    // Find appropriate place to insert controls
    const chartContainer = this.getChartContainer();
    if (chartContainer) {
      chartContainer.style.position = 'relative';
      chartContainer.appendChild(controlsContainer);
    } else {
      document.body.appendChild(controlsContainer);
    }

    this.setupZoomControlEvents();
  }

  /**
   * Get zoom controls HTML
   * @returns {string} HTML content
   */
  getZoomControlsHTML() {
    return `
            <div class="zoom-controls-header">
                <span class="zoom-controls-title">Chart Zoom</span>
                <label class="zoom-toggle">
                    <input type="checkbox" id="auto-zoom-checkbox" ${
                      this.config.autoZoomOnSelect ? 'checked' : ''
                    }>
                    <span>Auto-zoom to trades</span>
                </label>
            </div>

            <div class="zoom-buttons">
                <button id="zoom-in" class="zoom-button" title="Zoom In" aria-label="Zoom in">+</button>
                <button id="zoom-out" class="zoom-button" title="Zoom Out" aria-label="Zoom out">−</button>
                <button id="zoom-fit" class="zoom-button" title="Fit All Data" aria-label="Fit all data">⌐</button>
                <button id="zoom-reset" class="zoom-button" title="Reset Zoom" aria-label="Reset zoom">⟲</button>
            </div>

            <div class="zoom-history">
                <button id="zoom-back" class="zoom-button small" title="Previous Zoom"
                        aria-label="Go to previous zoom level">↶</button>
                <button id="zoom-forward" class="zoom-button small" title="Next Zoom"
                        aria-label="Go to next zoom level">↷</button>
            </div>

            <div class="zoom-presets">
                <label for="zoom-preset-select">Quick zoom:</label>
                <select id="zoom-preset-select">
                    <option value="">Select timeframe</option>
                    <option value="1h">Last Hour</option>
                    <option value="4h">Last 4 Hours</option>
                    <option value="1d">Last Day</option>
                    <option value="1w">Last Week</option>
                    <option value="1m">Last Month</option>
                </select>
            </div>
        `;
  }

  /**
   * Set up zoom control event listeners
   */
  setupZoomControlEvents() {
    // Auto-zoom toggle
    const autoZoomCheckbox = document.getElementById('auto-zoom-checkbox');
    autoZoomCheckbox?.addEventListener('change', (e) => {
      this.config.autoZoomOnSelect = e.target.checked;
    });

    // Zoom buttons
    document
      .getElementById('zoom-in')
      ?.addEventListener('click', () => this.zoomIn());
    document
      .getElementById('zoom-out')
      ?.addEventListener('click', () => this.zoomOut());
    document
      .getElementById('zoom-fit')
      ?.addEventListener('click', () => this.fitAllData());
    document
      .getElementById('zoom-reset')
      ?.addEventListener('click', () => this.resetZoom());

    // History navigation
    document
      .getElementById('zoom-back')
      ?.addEventListener('click', () => this.goToPreviousZoom());
    document
      .getElementById('zoom-forward')
      ?.addEventListener('click', () => this.goToNextZoom());

    // Preset selection
    const presetSelect = document.getElementById('zoom-preset-select');
    presetSelect?.addEventListener('change', (e) => {
      if (e.target.value) {
        this.zoomToPreset(e.target.value);
        e.target.value = ''; // Reset selection
      }
    });
  }

  /**
   * Handle trade selection for auto-zoom
   * @param {CustomEvent} event - Trade selection event
   */
  handleTradeSelection(event) {
    if (this.config.autoZoomOnSelect && this.isEnabled) {
      const { tradeId, tradeData } = event.detail;

      // Add delay to allow trade highlighting to complete
      clearTimeout(this.zoomTimer);
      this.zoomTimer = setTimeout(() => {
        this.zoomToTrade(tradeId, tradeData);
      }, this.config.zoomToTradeDelay);
    }
  }

  /**
   * Handle zoom change events (for history tracking)
   * @param {Object} eventData - Zoom change data
   */
  handleZoomChange(eventData) {
    if (this.config.preserveZoomHistory) {
      this.saveZoomState(eventData);
    }
  }

  /**
   * Zoom to a specific trade
   * @param {string} tradeId - Trade identifier
   * @param {Object} tradeData - Trade data object
   */
  zoomToTrade(tradeId, tradeData = null) {
    if (!this.isEnabled || !this.chart) {
      return;
    }

    try {
      // Get trade data if not provided
      if (!tradeData && this.tradeSelector) {
        tradeData = this.tradeSelector.getTradeData(tradeId);
      }

      if (!tradeData) {
        console.warn(`No trade data found for zoom: ${tradeId}`);
        return;
      }

      const zoomRange = this.calculateZoomRange(tradeData);
      this.applyZoom(zoomRange);

      console.log(`Zoomed to trade: ${tradeId}`);
    } catch (error) {
      console.error('Error zooming to trade:', error);
    }
  }

  /**
   * Calculate zoom range for a trade
   * @param {Object} tradeData - Trade data
   * @returns {Object} Zoom range with start and end times
   */
  calculateZoomRange(tradeData) {
    const entryTime = new Date(tradeData.entry_time);
    const exitTime = new Date(tradeData.exit_time);

    // Calculate trade duration
    const tradeDuration = exitTime.getTime() - entryTime.getTime();

    // Apply padding
    const padding = Math.max(
      tradeDuration * this.config.zoomPadding,
      this.config.minZoomDuration * this.config.zoomPadding
    );

    // Calculate zoom window
    let zoomStart = new Date(entryTime.getTime() - padding);
    let zoomEnd = new Date(exitTime.getTime() + padding);

    // Ensure minimum zoom duration
    const zoomDuration = zoomEnd.getTime() - zoomStart.getTime();
    if (zoomDuration < this.config.minZoomDuration) {
      const additionalPadding =
        (this.config.minZoomDuration - zoomDuration) / 2;
      zoomStart = new Date(zoomStart.getTime() - additionalPadding);
      zoomEnd = new Date(zoomEnd.getTime() + additionalPadding);
    }

    // Ensure maximum zoom duration
    const finalDuration = zoomEnd.getTime() - zoomStart.getTime();
    if (finalDuration > this.config.maxZoomDuration) {
      const center = (zoomStart.getTime() + zoomEnd.getTime()) / 2;
      const halfMax = this.config.maxZoomDuration / 2;
      zoomStart = new Date(center - halfMax);
      zoomEnd = new Date(center + halfMax);
    }

    return {
      start: zoomStart,
      end: zoomEnd,
      center: new Date((zoomStart.getTime() + zoomEnd.getTime()) / 2),
      duration: zoomEnd.getTime() - zoomStart.getTime(),
    };
  }

  /**
   * Apply zoom to chart
   * @param {Object} zoomRange - Zoom range object
   */
  applyZoom(zoomRange) {
    if (!this.chart) {
      return;
    }

    switch (this.chartType) {
      case 'plotly':
        this.applyPlotlyZoom(zoomRange);
        break;
      case 'd3':
        this.applyD3Zoom(zoomRange);
        break;
      case 'generic':
        this.applyGenericZoom(zoomRange);
        break;
    }
  }

  /**
   * Apply zoom to Plotly chart
   * @param {Object} zoomRange - Zoom range object
   */
  applyPlotlyZoom(zoomRange) {
    if (typeof this.chart.relayout === 'function') {
      const update = {
        'xaxis.range': [zoomRange.start, zoomRange.end],
      };

      this.chart.relayout(update);
    }
  }

  /**
   * Apply zoom to D3 chart
   * @param {Object} zoomRange - Zoom range object
   */
  applyD3Zoom(zoomRange) {
    // D3-specific zoom implementation
    console.log('D3 zoom not fully implemented', zoomRange);
  }

  /**
   * Apply zoom to generic chart
   * @param {Object} zoomRange - Zoom range object
   */
  applyGenericZoom(zoomRange) {
    // Generic zoom implementation using DOM manipulation
    console.log('Generic zoom applied', zoomRange);

    // Fire custom event for external handling
    document.dispatchEvent(
      new CustomEvent('chartZoomApplied', {
        detail: { zoomRange },
      })
    );
  }

  /**
   * Zoom in on current view
   */
  zoomIn() {
    if (!this.chart) {
      return;
    }

    const currentRange = this.getCurrentZoomRange();
    if (currentRange) {
      const center =
        (currentRange.start.getTime() + currentRange.end.getTime()) / 2;
      const newDuration =
        (currentRange.end.getTime() - currentRange.start.getTime()) * 0.6;

      const zoomRange = {
        start: new Date(center - newDuration / 2),
        end: new Date(center + newDuration / 2),
      };

      this.applyZoom(zoomRange);
    }
  }

  /**
   * Zoom out from current view
   */
  zoomOut() {
    if (!this.chart) {
      return;
    }

    const currentRange = this.getCurrentZoomRange();
    if (currentRange) {
      const center =
        (currentRange.start.getTime() + currentRange.end.getTime()) / 2;
      const newDuration =
        (currentRange.end.getTime() - currentRange.start.getTime()) * 1.6;

      const zoomRange = {
        start: new Date(center - newDuration / 2),
        end: new Date(center + newDuration / 2),
      };

      this.applyZoom(zoomRange);
    }
  }

  /**
   * Fit all data in view
   */
  fitAllData() {
    if (!this.chart) {
      return;
    }

    switch (this.chartType) {
      case 'plotly':
        if (typeof this.chart.relayout === 'function') {
          this.chart.relayout({
            'xaxis.autorange': true,
            'yaxis.autorange': true,
          });
        }
        break;
      default:
        // Fire custom event for external handling
        document.dispatchEvent(new CustomEvent('chartFitAllData'));
        break;
    }
  }

  /**
   * Reset zoom to default state
   */
  resetZoom() {
    this.fitAllData();

    // Clear zoom history
    this.zoomHistory = [];
    this.historyIndex = -1;
    this.updateZoomHistoryButtons();
  }

  /**
   * Zoom to preset timeframe
   * @param {string} preset - Preset identifier ('1h', '4h', '1d', etc.)
   */
  zoomToPreset(preset) {
    const now = new Date();
    let duration;

    switch (preset) {
      case '1h':
        duration = 60 * 60 * 1000; // 1 hour
        break;
      case '4h':
        duration = 4 * 60 * 60 * 1000; // 4 hours
        break;
      case '1d':
        duration = 24 * 60 * 60 * 1000; // 1 day
        break;
      case '1w':
        duration = 7 * 24 * 60 * 60 * 1000; // 1 week
        break;
      case '1m':
        duration = 30 * 24 * 60 * 60 * 1000; // 30 days
        break;
      default:
        return;
    }

    const zoomRange = {
      start: new Date(now.getTime() - duration),
      end: now,
    };

    this.applyZoom(zoomRange);
  }

  /**
   * Get current zoom range
   * @returns {Object|null} Current zoom range or null
   */
  getCurrentZoomRange() {
    if (!this.chart) {
      return null;
    }

    switch (this.chartType) {
      case 'plotly':
        if (
          this.chart.layout &&
          this.chart.layout.xaxis &&
          this.chart.layout.xaxis.range
        ) {
          const range = this.chart.layout.xaxis.range;
          return {
            start: new Date(range[0]),
            end: new Date(range[1]),
          };
        }
        break;
    }

    return null;
  }

  /**
   * Save current zoom state to history
   * @param {Object} zoomData - Zoom state data
   */
  saveZoomState(zoomData) {
    if (!this.config.preserveZoomHistory) {
      return;
    }

    const currentRange = this.getCurrentZoomRange();
    if (!currentRange) {
      return;
    }

    // Remove any forward history if we're not at the end
    if (this.historyIndex < this.zoomHistory.length - 1) {
      this.zoomHistory.splice(this.historyIndex + 1);
    }

    // Add new state
    this.zoomHistory.push({
      range: currentRange,
      timestamp: new Date(),
      data: zoomData,
    });

    this.historyIndex = this.zoomHistory.length - 1;

    // Limit history size
    if (this.zoomHistory.length > this.config.maxHistorySize) {
      this.zoomHistory.shift();
      this.historyIndex--;
    }

    this.updateZoomHistoryButtons();
  }

  /**
   * Go to previous zoom state
   */
  goToPreviousZoom() {
    if (this.historyIndex <= 0) {
      return;
    }

    this.historyIndex--;
    const zoomState = this.zoomHistory[this.historyIndex];

    if (zoomState) {
      this.applyZoom(zoomState.range);
      this.updateZoomHistoryButtons();
    }
  }

  /**
   * Go to next zoom state
   */
  goToNextZoom() {
    if (this.historyIndex >= this.zoomHistory.length - 1) {
      return;
    }

    this.historyIndex++;
    const zoomState = this.zoomHistory[this.historyIndex];

    if (zoomState) {
      this.applyZoom(zoomState.range);
      this.updateZoomHistoryButtons();
    }
  }

  /**
   * Update zoom history button states
   */
  updateZoomHistoryButtons() {
    const backButton = document.getElementById('zoom-back');
    const forwardButton = document.getElementById('zoom-forward');

    if (backButton) {
      backButton.disabled = this.historyIndex <= 0;
    }

    if (forwardButton) {
      forwardButton.disabled = this.historyIndex >= this.zoomHistory.length - 1;
    }
  }

  /**
   * Get chart container element
   * @returns {HTMLElement|null} Chart container
   */
  getChartContainer() {
    if (this.chart && this.chart.nodeType === Node.ELEMENT_NODE) {
      return this.chart;
    }

    // Try common selectors
    const selectors = [
      '.chart-container',
      '.plotly-graph-div',
      '#chart',
      '.js-plotly-plot',
      '.chart',
    ];

    for (const selector of selectors) {
      const element = document.querySelector(selector);
      if (element) {
        return element;
      }
    }

    return null;
  }

  /**
   * Check if zoom is enabled
   * @returns {boolean} True if enabled
   */
  isZoomEnabled() {
    return this.isEnabled && this.config.zoomEnabled;
  }

  /**
   * Enable or disable zoom functionality
   * @param {boolean} enabled - Whether to enable zoom
   */
  setEnabled(enabled) {
    this.isEnabled = enabled;

    // Update UI state
    const autoZoomCheckbox = document.getElementById('auto-zoom-checkbox');
    if (autoZoomCheckbox) {
      autoZoomCheckbox.disabled = !enabled;
    }

    // Update zoom controls
    const zoomControls = document.getElementById('chart-zoom-controls');
    if (zoomControls) {
      zoomControls.classList.toggle('disabled', !enabled);
    }
  }

  /**
   * Update configuration
   * @param {Object} newConfig - New configuration options
   */
  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Get current zoom state
   * @returns {Object} Current zoom state
   */
  getZoomState() {
    return {
      enabled: this.isEnabled,
      autoZoomOnSelect: this.config.autoZoomOnSelect,
      currentRange: this.getCurrentZoomRange(),
      historyLength: this.zoomHistory.length,
      historyIndex: this.historyIndex,
      chartType: this.chartType,
    };
  }

  /**
   * Clean up resources
   */
  destroy() {
    // Clear timers
    if (this.zoomTimer) {
      clearTimeout(this.zoomTimer);
    }

    // Remove event listeners
    document.removeEventListener('tradeSelected', this.handleTradeSelection);

    if (this.chart && this.chart.removeListener) {
      this.chart.removeListener('plotly_relayout', this.handleZoomChange);
    }

    // Remove UI
    const zoomControls = document.getElementById('chart-zoom-controls');
    if (zoomControls) {
      zoomControls.remove();
    }

    // Reset state
    this.chart = null;
    this.currentZoomState = null;
    this.zoomHistory = [];
    this.historyIndex = -1;
    this.initialized = false;

    console.log('ChartZoom destroyed');
  }

  /**
   * Get module status information
   * @returns {Object} Status information
   */
  getStatus() {
    return {
      initialized: this.initialized,
      enabled: this.isEnabled,
      hasChart: !!this.chart,
      chartType: this.chartType,
      autoZoomOnSelect: this.config.autoZoomOnSelect,
      zoomHistoryLength: this.zoomHistory.length,
      currentRange: this.getCurrentZoomRange(),
    };
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ChartZoom;
}

// Global registration for direct script inclusion
if (typeof window !== 'undefined') {
  window.ChartZoom = ChartZoom;
}
