/**
 * Trade Selector Module
 *
 * Handles trade list interaction logic including:
 * - Trade selection and highlighting
 * - Event management for trade clicks
 * - Trade state management
 * - Integration with other modules
 */

class TradeSelector {
  constructor(config = {}) {
    this.selectedTrade = null;
    this.trades = [];
    this.config = {
      tradeItemSelector: '.trade-item',
      selectedClass: 'selected',
      autoScroll: true,
      smoothScroll: true,
      ...config,
    };

    this.eventListeners = [];
    this.initialized = false;

    // Bind methods to maintain context
    this.handleTradeClick = this.handleTradeClick.bind(this);
    this.handleKeyboardNavigation = this.handleKeyboardNavigation.bind(this);
  }

  /**
   * Initialize the trade selector with trade data and event listeners
   * @param {Array} trades - Array of trade objects
   * @param {Object} dependencies - Other module instances for coordination
   */
  initialize(trades = [], dependencies = {}) {
    if (this.initialized) {
      console.warn('TradeSelector already initialized');
      return;
    }

    this.trades = trades;
    this.chartHighlighter = dependencies.chartHighlighter;
    this.tradeDetails = dependencies.tradeDetails;
    this.chartZoom = dependencies.chartZoom;

    this.setupEventListeners();
    this.initialized = true;

    console.log(`TradeSelector initialized with ${trades.length} trades`);
  }

  /**
   * Set up event listeners for trade interaction
   */
  setupEventListeners() {
    // Trade item click handlers
    const tradeItems = document.querySelectorAll(this.config.tradeItemSelector);
    tradeItems.forEach((item) => {
      item.addEventListener('click', this.handleTradeClick);
      item.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          this.handleTradeClick(e);
        }
      });

      // Store reference for cleanup
      this.eventListeners.push({
        element: item,
        event: 'click',
        handler: this.handleTradeClick,
      });
    });

    // Keyboard navigation
    document.addEventListener('keydown', this.handleKeyboardNavigation);
    this.eventListeners.push({
      element: document,
      event: 'keydown',
      handler: this.handleKeyboardNavigation,
    });
  }

  /**
   * Handle trade item click events
   * @param {Event} event - Click event
   */
  handleTradeClick(event) {
    const tradeItem = event.currentTarget;
    const tradeId = tradeItem.dataset.tradeId;

    if (!tradeId) {
      console.warn('Trade item missing data-trade-id attribute');
      return;
    }

    this.selectTrade(tradeId, tradeItem);
  }

  /**
   * Handle keyboard navigation for trade selection
   * @param {KeyboardEvent} event - Keyboard event
   */
  handleKeyboardNavigation(event) {
    if (!this.isNavigationKey(event.key)) {
      return;
    }

    event.preventDefault();

    const visibleTrades = this.getVisibleTradeItems();
    if (visibleTrades.length === 0) {
      return;
    }

    let currentIndex = this.getCurrentTradeIndex(visibleTrades);

    switch (event.key) {
      case 'ArrowUp':
        currentIndex =
          currentIndex > 0 ? currentIndex - 1 : visibleTrades.length - 1;
        break;
      case 'ArrowDown':
        currentIndex =
          currentIndex < visibleTrades.length - 1 ? currentIndex + 1 : 0;
        break;
      case 'Home':
        currentIndex = 0;
        break;
      case 'End':
        currentIndex = visibleTrades.length - 1;
        break;
      case 'Escape':
        this.clearSelection();
        return;
    }

    const targetTrade = visibleTrades[currentIndex];
    const tradeId = targetTrade.dataset.tradeId;
    this.selectTrade(tradeId, targetTrade);
  }

  /**
   * Select a trade and coordinate with other modules
   * @param {string} tradeId - Trade identifier
   * @param {HTMLElement} tradeElement - Trade DOM element (optional)
   */
  selectTrade(tradeId, tradeElement = null) {
    try {
      // Clear previous selection
      this.clearSelection();

      // Find trade element if not provided
      if (!tradeElement) {
        tradeElement = document.querySelector(`[data-trade-id="${tradeId}"]`);
      }

      if (!tradeElement) {
        console.warn(`Trade element not found for ID: ${tradeId}`);
        return;
      }

      // Highlight trade in list
      tradeElement.classList.add(this.config.selectedClass);
      tradeElement.setAttribute('aria-selected', 'true');

      // Scroll to trade if configured
      if (this.config.autoScroll) {
        this.scrollToTrade(tradeElement);
      }

      // Update selected trade
      this.selectedTrade = tradeId;

      // Coordinate with other modules
      this.notifyOtherModules(tradeId);

      // Fire custom event for external listeners
      this.fireTradeSelectedEvent(tradeId);

      console.log(`Trade selected: ${tradeId}`);
    } catch (error) {
      console.error('Error selecting trade:', error);
    }
  }

  /**
   * Clear current trade selection
   */
  clearSelection() {
    // Remove selection from all trade items
    document
      .querySelectorAll(
        `${this.config.tradeItemSelector}.${this.config.selectedClass}`
      )
      .forEach((item) => {
        item.classList.remove(this.config.selectedClass);
        item.setAttribute('aria-selected', 'false');
      });

    // Clear chart highlighting if available
    if (this.chartHighlighter) {
      this.chartHighlighter.clearHighlights();
    }

    this.selectedTrade = null;

    // Fire custom event
    this.fireTradeDeselectedEvent();
  }

  /**
   * Get the currently selected trade ID
   * @returns {string|null} Selected trade ID or null
   */
  getSelectedTrade() {
    return this.selectedTrade;
  }

  /**
   * Get trade data by ID
   * @param {string} tradeId - Trade identifier
   * @returns {Object|null} Trade data object or null
   */
  getTradeData(tradeId) {
    return this.trades.find((trade) => trade.id === tradeId) || null;
  }

  /**
   * Navigate to next/previous trade
   * @param {number} direction - Direction to navigate (1 for next, -1 for previous)
   */
  navigateTrades(direction) {
    const visibleTrades = this.getVisibleTradeItems();
    if (visibleTrades.length === 0) {
      return;
    }

    const currentIndex = this.getCurrentTradeIndex(visibleTrades);
    const newIndex = Math.max(
      0,
      Math.min(visibleTrades.length - 1, currentIndex + direction)
    );

    if (newIndex !== currentIndex) {
      const targetTrade = visibleTrades[newIndex];
      const tradeId = targetTrade.dataset.tradeId;
      this.selectTrade(tradeId, targetTrade);
    }
  }

  /**
   * Check if a key is used for navigation
   * @param {string} key - Key name
   * @returns {boolean} True if navigation key
   */
  isNavigationKey(key) {
    return ['ArrowUp', 'ArrowDown', 'Home', 'End', 'Escape'].includes(key);
  }

  /**
   * Get visible trade items (not filtered out)
   * @returns {Array} Array of visible trade elements
   */
  getVisibleTradeItems() {
    return Array.from(
      document.querySelectorAll(this.config.tradeItemSelector)
    ).filter((item) => item.style.display !== 'none' && !item.hidden);
  }

  /**
   * Get current trade index in visible trades
   * @param {Array} visibleTrades - Array of visible trade elements
   * @returns {number} Current index or -1 if none selected
   */
  getCurrentTradeIndex(visibleTrades) {
    if (!this.selectedTrade) {
      return -1;
    }

    return visibleTrades.findIndex(
      (item) => item.dataset.tradeId === this.selectedTrade
    );
  }

  /**
   * Scroll to a trade element
   * @param {HTMLElement} tradeElement - Trade element to scroll to
   */
  scrollToTrade(tradeElement) {
    const scrollOptions = {
      behavior: this.config.smoothScroll ? 'smooth' : 'auto',
      block: 'center',
      inline: 'nearest',
    };

    tradeElement.scrollIntoView(scrollOptions);
  }

  /**
   * Notify other modules about trade selection
   * @param {string} tradeId - Selected trade ID
   */
  notifyOtherModules(tradeId) {
    const tradeData = this.getTradeData(tradeId);

    // Highlight chart markers
    if (this.chartHighlighter) {
      this.chartHighlighter.highlightTrade(tradeId);
    }

    // Show trade details
    if (this.tradeDetails) {
      this.tradeDetails.showTradeDetails(tradeId, tradeData);
    }

    // Zoom to trade if enabled
    if (this.chartZoom && this.chartZoom.isZoomEnabled()) {
      this.chartZoom.zoomToTrade(tradeId, tradeData);
    }
  }

  /**
   * Fire custom event for trade selection
   * @param {string} tradeId - Selected trade ID
   */
  fireTradeSelectedEvent(tradeId) {
    const event = new CustomEvent('tradeSelected', {
      detail: {
        tradeId: tradeId,
        tradeData: this.getTradeData(tradeId),
      },
    });
    document.dispatchEvent(event);
  }

  /**
   * Fire custom event for trade deselection
   */
  fireTradeDeselectedEvent() {
    const event = new CustomEvent('tradeDeselected');
    document.dispatchEvent(event);
  }

  /**
   * Update configuration
   * @param {Object} newConfig - New configuration options
   */
  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Clean up event listeners and resources
   */
  destroy() {
    // Remove event listeners
    this.eventListeners.forEach(({ element, event, handler }) => {
      element.removeEventListener(event, handler);
    });
    this.eventListeners = [];

    // Clear selection
    this.clearSelection();

    // Reset state
    this.selectedTrade = null;
    this.trades = [];
    this.initialized = false;

    console.log('TradeSelector destroyed');
  }

  /**
   * Get module status information
   * @returns {Object} Status information
   */
  getStatus() {
    return {
      initialized: this.initialized,
      selectedTrade: this.selectedTrade,
      tradeCount: this.trades.length,
      visibleTradeCount: this.getVisibleTradeItems().length,
      hasChartHighlighter: !!this.chartHighlighter,
      hasTradeDetails: !!this.tradeDetails,
      hasChartZoom: !!this.chartZoom,
    };
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TradeSelector;
}

// Global registration for direct script inclusion
if (typeof window !== 'undefined') {
  window.TradeSelector = TradeSelector;
}
