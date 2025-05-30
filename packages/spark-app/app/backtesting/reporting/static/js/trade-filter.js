/**
 * Trade Filter Module
 *
 * Handles search and filtering capabilities including:
 * - Text-based search across trade data
 * - Multiple filter criteria (profit/loss, duration, etc.)
 * - Advanced filtering combinations
 * - Real-time filtering and search
 */

class TradeFilter {
  constructor(config = {}) {
    this.trades = [];
    this.filteredTrades = [];
    this.activeFilters = new Map();
    this.searchQuery = '';
    this.config = {
      enableSearch: true,
      enableFilters: true,
      searchFields: [
        'entry_time',
        'exit_time',
        'exit_reason',
        'pnl',
        'position_size',
      ],
      debounceDelay: 300,
      caseSensitiveSearch: false,
      highlightMatches: true,
      ...config,
    };

    this.searchTimer = null;
    this.initialized = false;

    // Bind methods to maintain context
    this.handleSearchInput = this.handleSearchInput.bind(this);
    this.handleFilterChange = this.handleFilterChange.bind(this);
  }

  /**
   * Initialize the trade filter
   * @param {Array} trades - Array of trade objects
   * @param {Object} dependencies - Other module instances
   */
  initialize(trades = [], dependencies = {}) {
    if (this.initialized) {
      console.warn('TradeFilter already initialized');
      return;
    }

    this.trades = trades;
    this.filteredTrades = [...trades];
    this.tradeSelector = dependencies.tradeSelector;

    this.createFilterUI();
    this.setupEventListeners();
    this.initialized = true;

    console.log(`TradeFilter initialized with ${trades.length} trades`);
  }

  /**
   * Create filter UI components
   */
  createFilterUI() {
    const existingFilter = document.getElementById('trade-filter');
    if (existingFilter) {
      existingFilter.remove();
    }

    const filterContainer = document.createElement('div');
    filterContainer.id = 'trade-filter';
    filterContainer.className = 'trade-filter';
    filterContainer.innerHTML = this.getFilterHTML();

    // Find appropriate place to insert filter
    const tradeList =
      document.querySelector('.trade-list-container') ||
      document.querySelector('.trade-analysis-container') ||
      document.body;

    if (tradeList && tradeList !== document.body) {
      tradeList.insertBefore(filterContainer, tradeList.firstChild);
    } else {
      document.body.appendChild(filterContainer);
    }

    this.setupFilterControls();
  }

  /**
   * Get filter HTML structure
   * @returns {string} HTML content
   */
  getFilterHTML() {
    return `
            <div class="trade-filter-header">
                <span class="trade-filter-title">Trade Filters</span>
            </div>

            ${
              this.config.enableSearch
                ? `
            <div class="trade-search">
                <input id="trade-search-input" type="text" placeholder="Search trades..."
                       aria-label="Search trades">
                <button id="search-clear" class="search-clear">×</button>
            </div>
            `
                : ''
            }

            <div class="trade-filters">
                <div class="filter-group">
                    <h4>Profit/Loss</h4>
                    <label><input type="checkbox" id="filter-profitable" data-filter="profitable"> Profitable</label>
                    <label><input type="checkbox" id="filter-losses" data-filter="losses"> Losses</label>
                </div>

                <div class="filter-group">
                    <h4>Duration</h4>
                    <input type="number" id="duration-min" placeholder="Min hours" data-filter="duration-min">
                    <input type="number" id="duration-max" placeholder="Max hours" data-filter="duration-max">
                </div>
            </div>

            <div class="filter-actions">
                <button id="apply-filters" class="filter-button">Apply</button>
                <button id="clear-filters" class="filter-button">Clear</button>
            </div>

            <div class="filter-stats">
                <span>Showing <span id="filtered-count">${
                  this.filteredTrades.length
                }</span> of <span id="total-count">${
      this.trades.length
    }</span> trades</span>
            </div>
        `;
  }

  /**
   * Set up filter control event listeners
   */
  setupFilterControls() {
    // Search input
    const searchInput = document.getElementById('trade-search-input');
    searchInput?.addEventListener('input', this.handleSearchInput);

    // Search clear button
    document.getElementById('search-clear')?.addEventListener('click', () => {
      this.clearSearch();
    });

    // Filter controls
    const filterElements = document.querySelectorAll('[data-filter]');
    filterElements.forEach((element) => {
      element.addEventListener('change', this.handleFilterChange);
    });

    // Action buttons
    document.getElementById('apply-filters')?.addEventListener('click', () => {
      this.applyFilters();
    });

    document.getElementById('clear-filters')?.addEventListener('click', () => {
      this.clearAllFilters();
    });

    // Initial filters application
    setTimeout(() => this.applyFilters(), 100);
  }

  /**
   * Set up general event listeners
   */
  setupEventListeners() {
    // Listen for trade updates
    document.addEventListener('tradesUpdated', (event) => {
      this.updateTrades(event.detail.trades);
    });
  }

  /**
   * Handle search input with debouncing
   * @param {Event} event - Input event
   */
  handleSearchInput(event) {
    clearTimeout(this.searchTimer);

    this.searchTimer = setTimeout(() => {
      this.searchQuery = event.target.value.trim();
      this.applyFilters();
    }, this.config.debounceDelay);
  }

  /**
   * Handle filter control changes
   * @param {Event} event - Change event
   */
  handleFilterChange(event) {
    const filterType = event.target.dataset.filter;
    const value =
      event.target.type === 'checkbox'
        ? event.target.checked
        : event.target.value;

    if (filterType) {
      this.setFilter(filterType, value);
    }
  }

  /**
   * Set a specific filter
   * @param {string} filterType - Type of filter
   * @param {any} value - Filter value
   */
  setFilter(filterType, value) {
    if (value === '' || value === false) {
      this.activeFilters.delete(filterType);
    } else {
      this.activeFilters.set(filterType, value);
    }

    // Auto-apply for immediate feedback
    clearTimeout(this.searchTimer);
    this.searchTimer = setTimeout(() => {
      this.applyFilters();
    }, this.config.debounceDelay);
  }

  /**
   * Apply all active filters and search
   */
  applyFilters() {
    let filtered = [...this.trades];

    // Apply search filter
    if (this.searchQuery && this.config.enableSearch) {
      filtered = this.applySearchFilter(filtered);
    }

    // Apply individual filters
    for (const [filterType, value] of this.activeFilters) {
      filtered = this.applyIndividualFilter(filtered, filterType, value);
    }

    this.filteredTrades = filtered;
    this.updateFilteredTradeDisplay();
    this.updateFilterStats();

    // Notify other modules
    document.dispatchEvent(
      new CustomEvent('tradesFiltered', {
        detail: {
          filteredTrades: this.filteredTrades,
          totalTrades: this.trades.length,
        },
      })
    );

    console.log(
      `Filtered ${this.filteredTrades.length} trades from ${this.trades.length} total`
    );
  }

  /**
   * Apply search filter to trades
   * @param {Array} trades - Trades to filter
   * @returns {Array} Filtered trades
   */
  applySearchFilter(trades) {
    if (!this.searchQuery) {
      return trades;
    }

    const searchPattern = new RegExp(
      this.searchQuery,
      this.config.caseSensitiveSearch ? 'g' : 'gi'
    );

    return trades.filter((trade) => {
      return this.config.searchFields.some((field) => {
        const fieldValue = trade[field];
        if (fieldValue == null) return false;

        const stringValue = String(fieldValue);
        return searchPattern.test(stringValue);
      });
    });
  }

  /**
   * Apply individual filter to trades
   * @param {Array} trades - Trades to filter
   * @param {string} filterType - Type of filter
   * @param {any} value - Filter value
   * @returns {Array} Filtered trades
   */
  applyIndividualFilter(trades, filterType, value) {
    switch (filterType) {
      case 'profitable':
        return value ? trades.filter((trade) => (trade.pnl || 0) > 0) : trades;

      case 'losses':
        return value ? trades.filter((trade) => (trade.pnl || 0) < 0) : trades;

      case 'duration-min':
        return trades.filter(
          (trade) => (trade.duration_minutes || 0) / 60 >= parseFloat(value)
        );

      case 'duration-max':
        return trades.filter(
          (trade) => (trade.duration_minutes || 0) / 60 <= parseFloat(value)
        );

      default:
        return trades;
    }
  }

  /**
   * Update filtered trade display
   */
  updateFilteredTradeDisplay() {
    // Hide/show trade items based on filtering
    const tradeItems = document.querySelectorAll('.trade-item');
    const filteredIds = new Set(this.filteredTrades.map((trade) => trade.id));

    tradeItems.forEach((item) => {
      const tradeId = item.dataset.tradeId;
      const isVisible = filteredIds.has(tradeId);

      item.style.display = isVisible ? '' : 'none';
      item.hidden = !isVisible;
    });
  }

  /**
   * Update filter statistics
   */
  updateFilterStats() {
    const filteredCountElement = document.getElementById('filtered-count');
    const totalCountElement = document.getElementById('total-count');

    if (filteredCountElement) {
      filteredCountElement.textContent = this.filteredTrades.length;
    }

    if (totalCountElement) {
      totalCountElement.textContent = this.trades.length;
    }
  }

  /**
   * Clear search
   */
  clearSearch() {
    const searchInput = document.getElementById('trade-search-input');
    if (searchInput) {
      searchInput.value = '';
      this.searchQuery = '';
      this.applyFilters();
    }
  }

  /**
   * Clear all filters
   */
  clearAllFilters() {
    // Clear active filters
    this.activeFilters.clear();
    this.searchQuery = '';

    // Reset UI
    const searchInput = document.getElementById('trade-search-input');
    if (searchInput) {
      searchInput.value = '';
    }

    const filterElements = document.querySelectorAll('[data-filter]');
    filterElements.forEach((element) => {
      if (element.type === 'checkbox') {
        element.checked = false;
      } else {
        element.value = '';
      }
    });

    this.applyFilters();
  }

  /**
   * Update trades list
   * @param {Array} trades - New trades array
   */
  updateTrades(trades) {
    this.trades = trades;
    this.applyFilters();
  }

  /**
   * Get filtered trades
   * @returns {Array} Current filtered trades array
   */
  getFilteredTrades() {
    return [...this.filteredTrades];
  }

  /**
   * Update configuration
   * @param {Object} newConfig - New configuration options
   */
  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Clean up resources
   */
  destroy() {
    // Clear timers
    if (this.searchTimer) {
      clearTimeout(this.searchTimer);
    }

    // Remove UI
    const filterElement = document.getElementById('trade-filter');
    if (filterElement) {
      filterElement.remove();
    }

    // Reset state
    this.trades = [];
    this.filteredTrades = [];
    this.activeFilters.clear();
    this.searchQuery = '';
    this.initialized = false;

    console.log('TradeFilter destroyed');
  }

  /**
   * Get module status information
   * @returns {Object} Status information
   */
  getStatus() {
    return {
      initialized: this.initialized,
      totalTrades: this.trades.length,
      filteredTrades: this.filteredTrades.length,
      activeFiltersCount: this.activeFilters.size,
      hasSearchQuery: !!this.searchQuery,
    };
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TradeFilter;
}

// Global registration for direct script inclusion
if (typeof window !== 'undefined') {
  window.TradeFilter = TradeFilter;
}
