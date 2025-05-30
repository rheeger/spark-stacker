/**
 * Trade Details Module
 *
 * Handles trade details popup/sidebar functionality including:
 * - Trade information display
 * - Popup and sidebar management
 * - Responsive design features
 * - Animation and transitions
 */

class TradeDetails {
  constructor(config = {}) {
    this.currentTrade = null;
    this.isVisible = false;
    this.config = {
      displayMode: 'sidebar', // 'sidebar', 'popup', 'inline'
      position: 'right', // 'left', 'right', 'top', 'bottom'
      width: '380px',
      height: 'auto',
      animationDuration: 300,
      closeOnClickOutside: true,
      closeOnEscape: true,
      showCloseButton: true,
      autoHide: false,
      autoHideDelay: 5000,
      responsive: true,
      mobileBreakpoint: 768,
      ...config,
    };

    this.container = null;
    this.autoHideTimer = null;
    this.resizeObserver = null;
    this.initialized = false;

    // Bind methods to maintain context
    this.handleClickOutside = this.handleClickOutside.bind(this);
    this.handleEscapeKey = this.handleEscapeKey.bind(this);
    this.handleResize = this.handleResize.bind(this);
  }

  /**
   * Initialize the trade details component
   * @param {Object} dependencies - Other module instances
   */
  initialize(dependencies = {}) {
    if (this.initialized) {
      console.warn('TradeDetails already initialized');
      return;
    }

    this.tradeSelector = dependencies.tradeSelector;
    this.chartHighlighter = dependencies.chartHighlighter;

    this.createContainer();
    this.setupEventListeners();
    this.setupResponsive();
    this.initialized = true;

    console.log('TradeDetails initialized');
  }

  /**
   * Create the main container element
   */
  createContainer() {
    // Remove existing container if it exists
    const existing = document.getElementById('trade-details-container');
    if (existing) {
      existing.remove();
    }

    this.container = document.createElement('div');
    this.container.id = 'trade-details-container';
    this.container.className = `trade-details trade-details-${this.config.displayMode}`;

    // Apply initial styling
    this.applyContainerStyles();

    // Create content structure
    this.container.innerHTML = this.getContainerHTML();

    // Add to document
    document.body.appendChild(this.container);

    // Setup close button if enabled
    if (this.config.showCloseButton) {
      this.setupCloseButton();
    }
  }

  /**
   * Apply CSS styles to the container
   */
  applyContainerStyles() {
    const styles = {
      position: 'fixed',
      zIndex: '1000',
      background: 'white',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
      borderRadius: '8px',
      border: '1px solid #ddd',
      display: 'none',
      transition: `all ${this.config.animationDuration}ms ease`,
      overflow: 'hidden',
    };

    // Position-specific styles
    switch (this.config.displayMode) {
      case 'sidebar':
        if (this.config.position === 'right') {
          styles.right = '-' + this.config.width;
          styles.top = '0';
          styles.width = this.config.width;
          styles.height = '100vh';
        } else if (this.config.position === 'left') {
          styles.left = '-' + this.config.width;
          styles.top = '0';
          styles.width = this.config.width;
          styles.height = '100vh';
        }
        break;

      case 'popup':
        styles.top = '50%';
        styles.left = '50%';
        styles.transform = 'translate(-50%, -50%) scale(0.9)';
        styles.width = this.config.width;
        styles.height = this.config.height;
        styles.maxWidth = '90vw';
        styles.maxHeight = '90vh';
        break;

      case 'inline':
        styles.position = 'relative';
        styles.width = '100%';
        styles.height = 'auto';
        break;
    }

    Object.assign(this.container.style, styles);
  }

  /**
   * Get the HTML structure for the container
   * @returns {string} HTML content
   */
  getContainerHTML() {
    return `
            <div class="trade-details-header">
                <h3 class="trade-details-title">Trade Details</h3>
                ${
                  this.config.showCloseButton
                    ? '<button class="trade-details-close" aria-label="Close">&times;</button>'
                    : ''
                }
            </div>
            <div class="trade-details-content">
                <div class="trade-details-placeholder">
                    Select a trade to view details
                </div>
            </div>
        `;
  }

  /**
   * Set up event listeners
   */
  setupEventListeners() {
    // Listen for trade selection events
    document.addEventListener('tradeSelected', (event) => {
      this.showTradeDetails(event.detail.tradeId, event.detail.tradeData);
    });

    document.addEventListener('tradeDeselected', () => {
      if (this.config.autoHide) {
        this.hideDetails();
      }
    });

    // Global event listeners
    if (this.config.closeOnClickOutside) {
      document.addEventListener('click', this.handleClickOutside);
    }

    if (this.config.closeOnEscape) {
      document.addEventListener('keydown', this.handleEscapeKey);
    }

    // Window resize handling
    window.addEventListener('resize', this.handleResize);
  }

  /**
   * Setup responsive behavior
   */
  setupResponsive() {
    if (!this.config.responsive) {
      return;
    }

    // Create resize observer to watch for size changes
    if (window.ResizeObserver) {
      this.resizeObserver = new ResizeObserver(this.handleResize);
      this.resizeObserver.observe(document.body);
    }

    // Initial responsive check
    this.handleResize();
  }

  /**
   * Setup close button functionality
   */
  setupCloseButton() {
    const closeButton = this.container.querySelector('.trade-details-close');
    if (closeButton) {
      closeButton.addEventListener('click', () => {
        this.hideDetails();
      });
    }
  }

  /**
   * Show trade details
   * @param {string} tradeId - Trade identifier
   * @param {Object} tradeData - Trade data object
   */
  showTradeDetails(tradeId, tradeData = null) {
    try {
      // Get trade data if not provided
      if (!tradeData && this.tradeSelector) {
        tradeData = this.tradeSelector.getTradeData(tradeId);
      }

      if (!tradeData) {
        console.warn(`No trade data found for ID: ${tradeId}`);
        return;
      }

      this.currentTrade = { id: tradeId, ...tradeData };

      // Update content
      this.updateContent(this.currentTrade);

      // Show the container
      this.show();

      // Setup auto-hide if configured
      if (this.config.autoHide) {
        this.setupAutoHide();
      }

      console.log(`Trade details shown for: ${tradeId}`);
    } catch (error) {
      console.error('Error showing trade details:', error);
    }
  }

  /**
   * Update the content with trade data
   * @param {Object} tradeData - Trade data
   */
  updateContent(tradeData) {
    const contentElement = this.container.querySelector(
      '.trade-details-content'
    );
    if (!contentElement) {
      return;
    }

    contentElement.innerHTML = this.generateTradeDetailsHTML(tradeData);

    // Update title
    const titleElement = this.container.querySelector('.trade-details-title');
    if (titleElement) {
      titleElement.textContent = `Trade ${
        tradeData.index || tradeData.id || 'N/A'
      } Details`;
    }
  }

  /**
   * Generate HTML for trade details
   * @param {Object} tradeData - Trade data
   * @returns {string} HTML content
   */
  generateTradeDetailsHTML(tradeData) {
    const pnl = tradeData.pnl || 0;
    const pnlClass = pnl >= 0 ? 'profit' : 'loss';
    const pnlSign = pnl >= 0 ? '+' : '';
    const returnPercent = this.calculateReturnPercent(tradeData);
    const durationHours = (tradeData.duration_minutes || 0) / 60;

    return `
            <div class="trade-summary">
                <div class="trade-summary-pnl ${pnlClass}">
                    <span class="pnl-amount">${pnlSign}$${Math.abs(pnl).toFixed(
      2
    )}</span>
                    <span class="pnl-percent">(${
                      returnPercent >= 0 ? '+' : ''
                    }${returnPercent.toFixed(2)}%)</span>
                </div>
                <div class="trade-summary-status ${pnlClass}">
                    ${pnl >= 0 ? 'Profitable' : 'Loss'} Trade
                </div>
            </div>

            <div class="trade-details-grid">
                <div class="detail-section">
                    <h4>Trade Information</h4>
                    <div class="detail-items">
                        <div class="detail-item">
                            <label>Trade Number:</label>
                            <span>#${tradeData.index || 'N/A'}</span>
                        </div>
                        <div class="detail-item">
                            <label>Entry Time:</label>
                            <span>${this.formatDateTime(
                              tradeData.entry_time
                            )}</span>
                        </div>
                        <div class="detail-item">
                            <label>Exit Time:</label>
                            <span>${this.formatDateTime(
                              tradeData.exit_time
                            )}</span>
                        </div>
                        <div class="detail-item">
                            <label>Duration:</label>
                            <span>${this.formatDuration(durationHours)}</span>
                        </div>
                        <div class="detail-item">
                            <label>Exit Reason:</label>
                            <span class="exit-reason">${
                              tradeData.exit_reason || 'Unknown'
                            }</span>
                        </div>
                    </div>
                </div>

                <div class="detail-section">
                    <h4>Price Information</h4>
                    <div class="detail-items">
                        <div class="detail-item">
                            <label>Entry Price:</label>
                            <span class="price">$${(
                              tradeData.entry_price || 0
                            ).toFixed(4)}</span>
                        </div>
                        <div class="detail-item">
                            <label>Exit Price:</label>
                            <span class="price">$${(
                              tradeData.exit_price || 0
                            ).toFixed(4)}</span>
                        </div>
                        <div class="detail-item">
                            <label>Price Change:</label>
                            <span class="price-change ${pnlClass}">
                                ${pnlSign}$${Math.abs(
      (tradeData.exit_price || 0) - (tradeData.entry_price || 0)
    ).toFixed(4)}
                            </span>
                        </div>
                        <div class="detail-item">
                            <label>Price Change %:</label>
                            <span class="price-change-percent ${pnlClass}">
                                ${
                                  returnPercent >= 0 ? '+' : ''
                                }${returnPercent.toFixed(2)}%
                            </span>
                        </div>
                    </div>
                </div>

                <div class="detail-section">
                    <h4>Position Information</h4>
                    <div class="detail-items">
                        <div class="detail-item">
                            <label>Position Size:</label>
                            <span class="position-size">$${(
                              tradeData.position_size || 0
                            ).toFixed(2)}</span>
                        </div>
                        <div class="detail-item">
                            <label>Quantity:</label>
                            <span class="quantity">${(
                              (tradeData.position_size || 0) /
                              (tradeData.entry_price || 1)
                            ).toFixed(6)}</span>
                        </div>
                        <div class="detail-item">
                            <label>Direction:</label>
                            <span class="direction">${
                              tradeData.side || 'Long'
                            }</span>
                        </div>
                    </div>
                </div>

                ${
                  tradeData.stop_loss || tradeData.take_profit
                    ? `
                <div class="detail-section">
                    <h4>Risk Management</h4>
                    <div class="detail-items">
                        ${
                          tradeData.stop_loss
                            ? `
                        <div class="detail-item">
                            <label>Stop Loss:</label>
                            <span class="stop-loss">$${tradeData.stop_loss.toFixed(
                              4
                            )}</span>
                        </div>
                        `
                            : ''
                        }
                        ${
                          tradeData.take_profit
                            ? `
                        <div class="detail-item">
                            <label>Take Profit:</label>
                            <span class="take-profit">$${tradeData.take_profit.toFixed(
                              4
                            )}</span>
                        </div>
                        `
                            : ''
                        }
                    </div>
                </div>
                `
                    : ''
                }

                ${
                  tradeData.indicators &&
                  Object.keys(tradeData.indicators).length > 0
                    ? `
                <div class="detail-section">
                    <h4>Indicator Signals</h4>
                    <div class="detail-items">
                        ${this.generateIndicatorSignalsHTML(
                          tradeData.indicators
                        )}
                    </div>
                </div>
                `
                    : ''
                }
            </div>

            <div class="trade-actions">
                <button class="action-button" onclick="window.tradeDetails?.highlightOnChart('${
                  tradeData.id
                }')">
                    Highlight on Chart
                </button>
                <button class="action-button" onclick="window.tradeDetails?.zoomToTrade('${
                  tradeData.id
                }')">
                    Zoom to Trade
                </button>
                <button class="action-button secondary" onclick="window.tradeDetails?.exportTradeData('${
                  tradeData.id
                }')">
                    Export Data
                </button>
            </div>
        `;
  }

  /**
   * Generate HTML for indicator signals
   * @param {Object} indicators - Indicator data
   * @returns {string} HTML content
   */
  generateIndicatorSignalsHTML(indicators) {
    return Object.entries(indicators)
      .map(
        ([name, data]) => `
            <div class="detail-item">
                <label>${name}:</label>
                <span class="indicator-value">${JSON.stringify(data)}</span>
            </div>
        `
      )
      .join('');
  }

  /**
   * Calculate return percentage
   * @param {Object} tradeData - Trade data
   * @returns {number} Return percentage
   */
  calculateReturnPercent(tradeData) {
    const entryPrice = tradeData.entry_price || 0;
    const exitPrice = tradeData.exit_price || 0;

    if (entryPrice === 0) {
      return 0;
    }

    return ((exitPrice - entryPrice) / entryPrice) * 100;
  }

  /**
   * Format date and time
   * @param {string|Date} dateTime - Date time value
   * @returns {string} Formatted date time
   */
  formatDateTime(dateTime) {
    if (!dateTime) {
      return 'N/A';
    }

    try {
      const date = new Date(dateTime);
      return date.toLocaleString();
    } catch (error) {
      return dateTime.toString();
    }
  }

  /**
   * Format duration in hours
   * @param {number} hours - Duration in hours
   * @returns {string} Formatted duration
   */
  formatDuration(hours) {
    if (hours < 1) {
      return `${Math.round(hours * 60)} minutes`;
    } else if (hours < 24) {
      return `${hours.toFixed(1)} hours`;
    } else {
      const days = Math.floor(hours / 24);
      const remainingHours = hours % 24;
      return `${days} days, ${remainingHours.toFixed(1)} hours`;
    }
  }

  /**
   * Show the details container
   */
  show() {
    if (this.isVisible) {
      return;
    }

    this.container.style.display = 'block';

    // Trigger animation
    requestAnimationFrame(() => {
      this.container.classList.add('visible');

      if (this.config.displayMode === 'sidebar') {
        if (this.config.position === 'right') {
          this.container.style.right = '0';
        } else if (this.config.position === 'left') {
          this.container.style.left = '0';
        }
      } else if (this.config.displayMode === 'popup') {
        this.container.style.transform = 'translate(-50%, -50%) scale(1)';
      }
    });

    this.isVisible = true;

    // Add backdrop for mobile
    if (this.isMobile()) {
      this.addBackdrop();
    }
  }

  /**
   * Hide the details container
   */
  hideDetails() {
    if (!this.isVisible) {
      return;
    }

    this.container.classList.remove('visible');

    if (this.config.displayMode === 'sidebar') {
      if (this.config.position === 'right') {
        this.container.style.right = '-' + this.config.width;
      } else if (this.config.position === 'left') {
        this.container.style.left = '-' + this.config.width;
      }
    } else if (this.config.displayMode === 'popup') {
      this.container.style.transform = 'translate(-50%, -50%) scale(0.9)';
    }

    setTimeout(() => {
      this.container.style.display = 'none';
    }, this.config.animationDuration);

    this.isVisible = false;
    this.currentTrade = null;

    // Remove backdrop
    this.removeBackdrop();

    // Clear auto-hide timer
    this.clearAutoHide();
  }

  /**
   * Add backdrop for mobile
   */
  addBackdrop() {
    let backdrop = document.getElementById('trade-details-backdrop');

    if (!backdrop) {
      backdrop = document.createElement('div');
      backdrop.id = 'trade-details-backdrop';
      backdrop.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                z-index: 999;
                opacity: 0;
                transition: opacity ${this.config.animationDuration}ms ease;
            `;

      backdrop.addEventListener('click', () => {
        this.hideDetails();
      });

      document.body.appendChild(backdrop);
    }

    requestAnimationFrame(() => {
      backdrop.style.opacity = '1';
    });
  }

  /**
   * Remove backdrop
   */
  removeBackdrop() {
    const backdrop = document.getElementById('trade-details-backdrop');
    if (backdrop) {
      backdrop.style.opacity = '0';
      setTimeout(() => {
        backdrop.remove();
      }, this.config.animationDuration);
    }
  }

  /**
   * Setup auto-hide functionality
   */
  setupAutoHide() {
    this.clearAutoHide();

    if (this.config.autoHideDelay > 0) {
      this.autoHideTimer = setTimeout(() => {
        this.hideDetails();
      }, this.config.autoHideDelay);
    }
  }

  /**
   * Clear auto-hide timer
   */
  clearAutoHide() {
    if (this.autoHideTimer) {
      clearTimeout(this.autoHideTimer);
      this.autoHideTimer = null;
    }
  }

  /**
   * Handle click outside to close
   * @param {Event} event - Click event
   */
  handleClickOutside(event) {
    if (!this.isVisible || !this.container) {
      return;
    }

    if (!this.container.contains(event.target)) {
      this.hideDetails();
    }
  }

  /**
   * Handle escape key to close
   * @param {KeyboardEvent} event - Keyboard event
   */
  handleEscapeKey(event) {
    if (event.key === 'Escape' && this.isVisible) {
      this.hideDetails();
    }
  }

  /**
   * Handle window resize
   */
  handleResize() {
    if (!this.config.responsive) {
      return;
    }

    const isMobile = this.isMobile();

    if (isMobile && this.config.displayMode === 'sidebar') {
      // Switch to popup mode on mobile
      this.container.className = this.container.className.replace(
        'trade-details-sidebar',
        'trade-details-popup'
      );
      this.applyContainerStyles();
    } else if (
      !isMobile &&
      this.container.classList.contains('trade-details-popup')
    ) {
      // Switch back to sidebar mode on desktop
      this.container.className = this.container.className.replace(
        'trade-details-popup',
        'trade-details-sidebar'
      );
      this.applyContainerStyles();
    }
  }

  /**
   * Check if device is mobile
   * @returns {boolean} True if mobile
   */
  isMobile() {
    return window.innerWidth <= this.config.mobileBreakpoint;
  }

  /**
   * Highlight trade on chart (action button handler)
   * @param {string} tradeId - Trade identifier
   */
  highlightOnChart(tradeId) {
    if (this.chartHighlighter) {
      this.chartHighlighter.highlightTrade(tradeId);
    }
  }

  /**
   * Zoom to trade (action button handler)
   * @param {string} tradeId - Trade identifier
   */
  zoomToTrade(tradeId) {
    // This would coordinate with chart zoom module
    console.log(`Zoom to trade: ${tradeId}`);
  }

  /**
   * Export trade data (action button handler)
   * @param {string} tradeId - Trade identifier
   */
  exportTradeData(tradeId) {
    if (!this.currentTrade) {
      return;
    }

    const dataStr = JSON.stringify(this.currentTrade, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `trade_${tradeId}_data.json`;
    link.click();

    URL.revokeObjectURL(url);
  }

  /**
   * Update configuration
   * @param {Object} newConfig - New configuration options
   */
  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };

    if (this.initialized) {
      this.applyContainerStyles();
    }
  }

  /**
   * Get current trade data
   * @returns {Object|null} Current trade data
   */
  getCurrentTrade() {
    return this.currentTrade;
  }

  /**
   * Check if details are currently visible
   * @returns {boolean} True if visible
   */
  isDetailsVisible() {
    return this.isVisible;
  }

  /**
   * Clean up resources
   */
  destroy() {
    // Hide details
    this.hideDetails();

    // Remove event listeners
    document.removeEventListener('click', this.handleClickOutside);
    document.removeEventListener('keydown', this.handleEscapeKey);
    window.removeEventListener('resize', this.handleResize);

    // Disconnect resize observer
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }

    // Remove container
    if (this.container) {
      this.container.remove();
    }

    // Remove backdrop
    this.removeBackdrop();

    // Clear timers
    this.clearAutoHide();

    // Reset state
    this.container = null;
    this.currentTrade = null;
    this.isVisible = false;
    this.initialized = false;

    console.log('TradeDetails destroyed');
  }

  /**
   * Get module status information
   * @returns {Object} Status information
   */
  getStatus() {
    return {
      initialized: this.initialized,
      isVisible: this.isVisible,
      currentTrade: this.currentTrade?.id || null,
      displayMode: this.config.displayMode,
      hasTradeSelector: !!this.tradeSelector,
      hasChartHighlighter: !!this.chartHighlighter,
    };
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TradeDetails;
}

// Global registration for direct script inclusion
if (typeof window !== 'undefined') {
  window.TradeDetails = TradeDetails;
}
