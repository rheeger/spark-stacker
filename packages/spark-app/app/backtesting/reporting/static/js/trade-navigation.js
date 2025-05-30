/**
 * Trade Navigation Module
 *
 * Handles keyboard and sequence navigation including:
 * - Keyboard shortcuts for trade navigation
 * - Sequential trade browsing
 * - Navigation state management
 * - Accessibility support
 */

class TradeNavigation {
  constructor(config = {}) {
    this.currentIndex = -1;
    this.trades = [];
    this.filteredTrades = [];
    this.navigationHistory = [];
    this.historyIndex = -1;
    this.config = {
      enableKeyboardNavigation: true,
      enableSequentialNavigation: true,
      enableNavigationHistory: true,
      wrapNavigation: true,
      animateNavigation: true,
      keyboardShortcuts: {
        nextTrade: ['ArrowDown', 'j'],
        previousTrade: ['ArrowUp', 'k'],
        firstTrade: ['Home', 'g g'],
        lastTrade: ['End', 'G'],
        goBack: ['Backspace', 'h'],
        goForward: ['Shift+Backspace', 'l'],
        jumpToTrade: ['g'],
        toggleFavorite: ['f'],
        escape: ['Escape'],
      },
      announceNavigation: true,
      ...config,
    };

    this.keySequence = [];
    this.keySequenceTimeout = null;
    this.initialized = false;

    // Bind methods to maintain context
    this.handleKeydown = this.handleKeydown.bind(this);
    this.handleKeySequence = this.handleKeySequence.bind(this);
  }

  /**
   * Initialize the trade navigation
   * @param {Array} trades - Array of trade objects
   * @param {Object} dependencies - Other module instances
   */
  initialize(trades = [], dependencies = {}) {
    if (this.initialized) {
      console.warn('TradeNavigation already initialized');
      return;
    }

    this.trades = trades;
    this.filteredTrades = [...trades];
    this.tradeSelector = dependencies.tradeSelector;
    this.tradeFilter = dependencies.tradeFilter;

    if (this.config.enableKeyboardNavigation) {
      this.setupKeyboardNavigation();
    }

    this.createNavigationUI();
    this.setupEventListeners();
    this.initialized = true;

    console.log(`TradeNavigation initialized with ${trades.length} trades`);
  }

  /**
   * Set up keyboard navigation
   */
  setupKeyboardNavigation() {
    document.addEventListener('keydown', this.handleKeydown);

    // Create keyboard shortcut help
    this.createKeyboardShortcutHelp();
  }

  /**
   * Create navigation UI controls
   */
  createNavigationUI() {
    const existingNav = document.getElementById('trade-navigation');
    if (existingNav) {
      existingNav.remove();
    }

    const navContainer = document.createElement('div');
    navContainer.id = 'trade-navigation';
    navContainer.className = 'trade-navigation';
    navContainer.innerHTML = this.getNavigationHTML();

    // Find appropriate place to insert navigation
    const tradeList =
      document.querySelector('.trade-list-container') ||
      document.querySelector('.trade-analysis-container') ||
      document.body;

    if (tradeList && tradeList !== document.body) {
      tradeList.insertBefore(navContainer, tradeList.firstChild);
    } else {
      document.body.appendChild(navContainer);
    }

    this.setupNavigationControls();
  }

  /**
   * Get navigation HTML structure
   * @returns {string} HTML content
   */
  getNavigationHTML() {
    return `
            <div class="trade-nav-header">
                <span class="trade-nav-title">Trade Navigation</span>
                <div class="trade-nav-position">
                    <span id="current-trade-index">-</span> of <span id="total-trades">${this.filteredTrades.length}</span>
                </div>
            </div>

            <div class="trade-nav-controls">
                <button id="nav-first" class="nav-button" title="First Trade (Home)" aria-label="Go to first trade">
                    <span>⏮</span>
                </button>
                <button id="nav-prev" class="nav-button" title="Previous Trade (↑)" aria-label="Go to previous trade">
                    <span>⏪</span>
                </button>
                <button id="nav-next" class="nav-button" title="Next Trade (↓)" aria-label="Go to next trade">
                    <span>⏩</span>
                </button>
                <button id="nav-last" class="nav-button" title="Last Trade (End)" aria-label="Go to last trade">
                    <span>⏭</span>
                </button>
            </div>

            <div class="trade-nav-extras">
                <div class="trade-jump">
                    <label for="trade-jump-input">Jump to:</label>
                    <input id="trade-jump-input" type="number" min="1" max="${this.filteredTrades.length}"
                           placeholder="#" title="Enter trade number">
                    <button id="trade-jump-go" class="nav-button small">Go</button>
                </div>

                <div class="trade-nav-history">
                    <button id="nav-back" class="nav-button small" title="Go Back (Backspace)"
                            aria-label="Go back in navigation history">↶</button>
                    <button id="nav-forward" class="nav-button small" title="Go Forward (Shift+Backspace)"
                            aria-label="Go forward in navigation history">↷</button>
                </div>

                <button id="nav-help" class="nav-button small" title="Keyboard Shortcuts"
                        aria-label="Show keyboard shortcuts">?</button>
            </div>

            <div class="trade-nav-status" id="nav-status" aria-live="polite"></div>
        `;
  }

  /**
   * Set up navigation control event listeners
   */
  setupNavigationControls() {
    // Button controls
    document
      .getElementById('nav-first')
      ?.addEventListener('click', () => this.goToFirst());
    document
      .getElementById('nav-prev')
      ?.addEventListener('click', () => this.goToPrevious());
    document
      .getElementById('nav-next')
      ?.addEventListener('click', () => this.goToNext());
    document
      .getElementById('nav-last')
      ?.addEventListener('click', () => this.goToLast());
    document
      .getElementById('nav-back')
      ?.addEventListener('click', () => this.goBack());
    document
      .getElementById('nav-forward')
      ?.addEventListener('click', () => this.goForward());
    document
      .getElementById('nav-help')
      ?.addEventListener('click', () => this.showKeyboardHelp());

    // Jump to trade
    const jumpInput = document.getElementById('trade-jump-input');
    const jumpButton = document.getElementById('trade-jump-go');

    jumpButton?.addEventListener('click', () => this.jumpToTrade());
    jumpInput?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        this.jumpToTrade();
      }
    });
  }

  /**
   * Set up general event listeners
   */
  setupEventListeners() {
    // Listen for trade selection events
    document.addEventListener('tradeSelected', (event) => {
      this.updateCurrentIndex(event.detail.tradeId);
      this.addToHistory(event.detail.tradeId);
      this.updateUI();
    });

    document.addEventListener('tradeDeselected', () => {
      this.currentIndex = -1;
      this.updateUI();
    });

    // Listen for filter changes
    document.addEventListener('tradesFiltered', (event) => {
      this.updateFilteredTrades(event.detail.filteredTrades);
    });

    // Listen for trade list updates
    document.addEventListener('tradesUpdated', (event) => {
      this.updateTrades(event.detail.trades);
    });
  }

  /**
   * Handle keyboard events
   * @param {KeyboardEvent} event - Keyboard event
   */
  handleKeydown(event) {
    if (!this.config.enableKeyboardNavigation) {
      return;
    }

    // Skip if user is typing in an input field
    if (this.isInputActive()) {
      return;
    }

    const key = this.normalizeKey(event);

    // Handle key sequences (like vim-style 'gg')
    if (this.handleKeySequence(key, event)) {
      return;
    }

    // Handle single key shortcuts
    if (this.handleSingleKeyShortcut(key, event)) {
      event.preventDefault();
    }
  }

  /**
   * Handle key sequences for complex shortcuts
   * @param {string} key - Normalized key
   * @param {KeyboardEvent} event - Original keyboard event
   * @returns {boolean} True if key sequence was handled
   */
  handleKeySequence(key, event) {
    const shortcuts = this.config.keyboardShortcuts;

    // Add key to sequence
    this.keySequence.push(key);

    // Reset sequence timeout
    clearTimeout(this.keySequenceTimeout);
    this.keySequenceTimeout = setTimeout(() => {
      this.keySequence = [];
    }, 1000);

    // Check for sequence matches
    const sequence = this.keySequence.join(' ');

    // Handle 'gg' (go to first)
    if (sequence === 'g g' && shortcuts.firstTrade.includes('g g')) {
      this.goToFirst();
      this.keySequence = [];
      event.preventDefault();
      return true;
    }

    // If we have 'g' and another key is pressed, handle jump commands
    if (this.keySequence.length === 2 && this.keySequence[0] === 'g') {
      const secondKey = this.keySequence[1];

      if (secondKey.match(/^\d$/)) {
        // Start number input for jump
        this.startNumberInput(secondKey);
        this.keySequence = [];
        event.preventDefault();
        return true;
      }

      // Reset if not a valid sequence
      this.keySequence = [];
    }

    return false;
  }

  /**
   * Handle single key shortcuts
   * @param {string} key - Normalized key
   * @param {KeyboardEvent} event - Keyboard event
   * @returns {boolean} True if shortcut was handled
   */
  handleSingleKeyShortcut(key, event) {
    const shortcuts = this.config.keyboardShortcuts;

    if (shortcuts.nextTrade.includes(key)) {
      this.goToNext();
      return true;
    }

    if (shortcuts.previousTrade.includes(key)) {
      this.goToPrevious();
      return true;
    }

    if (shortcuts.firstTrade.includes(key)) {
      this.goToFirst();
      return true;
    }

    if (shortcuts.lastTrade.includes(key)) {
      this.goToLast();
      return true;
    }

    if (shortcuts.goBack.includes(key)) {
      this.goBack();
      return true;
    }

    if (shortcuts.goForward.includes(key)) {
      this.goForward();
      return true;
    }

    if (shortcuts.escape.includes(key)) {
      this.clearSelection();
      return true;
    }

    if (shortcuts.toggleFavorite.includes(key)) {
      this.toggleFavorite();
      return true;
    }

    return false;
  }

  /**
   * Normalize key representation
   * @param {KeyboardEvent} event - Keyboard event
   * @returns {string} Normalized key string
   */
  normalizeKey(event) {
    let key = event.key;

    if (event.shiftKey && event.key !== 'Shift') {
      key = `Shift+${key}`;
    }

    if (event.ctrlKey && event.key !== 'Control') {
      key = `Ctrl+${key}`;
    }

    if (event.altKey && event.key !== 'Alt') {
      key = `Alt+${key}`;
    }

    return key;
  }

  /**
   * Check if an input field is currently active
   * @returns {boolean} True if input is active
   */
  isInputActive() {
    const activeElement = document.activeElement;
    return (
      activeElement &&
      (activeElement.tagName === 'INPUT' ||
        activeElement.tagName === 'TEXTAREA' ||
        activeElement.isContentEditable)
    );
  }

  /**
   * Navigate to next trade
   */
  goToNext() {
    if (this.filteredTrades.length === 0) {
      return;
    }

    let newIndex = this.currentIndex + 1;

    if (newIndex >= this.filteredTrades.length) {
      newIndex = this.config.wrapNavigation
        ? 0
        : this.filteredTrades.length - 1;
    }

    this.navigateToIndex(newIndex);
    this.announceNavigation('Next trade');
  }

  /**
   * Navigate to previous trade
   */
  goToPrevious() {
    if (this.filteredTrades.length === 0) {
      return;
    }

    let newIndex = this.currentIndex - 1;

    if (newIndex < 0) {
      newIndex = this.config.wrapNavigation
        ? this.filteredTrades.length - 1
        : 0;
    }

    this.navigateToIndex(newIndex);
    this.announceNavigation('Previous trade');
  }

  /**
   * Navigate to first trade
   */
  goToFirst() {
    if (this.filteredTrades.length === 0) {
      return;
    }

    this.navigateToIndex(0);
    this.announceNavigation('First trade');
  }

  /**
   * Navigate to last trade
   */
  goToLast() {
    if (this.filteredTrades.length === 0) {
      return;
    }

    this.navigateToIndex(this.filteredTrades.length - 1);
    this.announceNavigation('Last trade');
  }

  /**
   * Navigate to specific index
   * @param {number} index - Target index
   */
  navigateToIndex(index) {
    if (index < 0 || index >= this.filteredTrades.length) {
      return;
    }

    this.currentIndex = index;
    const trade = this.filteredTrades[index];

    if (trade && this.tradeSelector) {
      this.tradeSelector.selectTrade(trade.id);
    }
  }

  /**
   * Jump to specific trade number
   */
  jumpToTrade() {
    const input = document.getElementById('trade-jump-input');
    if (!input) {
      return;
    }

    const tradeNumber = parseInt(input.value);
    if (
      isNaN(tradeNumber) ||
      tradeNumber < 1 ||
      tradeNumber > this.filteredTrades.length
    ) {
      this.showStatus('Invalid trade number', 'error');
      return;
    }

    const index = tradeNumber - 1; // Convert to 0-based index
    this.navigateToIndex(index);
    this.announceNavigation(`Jumped to trade ${tradeNumber}`);

    // Clear input
    input.value = '';
  }

  /**
   * Start number input for vim-style navigation
   * @param {string} firstDigit - First digit entered
   */
  startNumberInput(firstDigit) {
    const input = document.getElementById('trade-jump-input');
    if (input) {
      input.value = firstDigit;
      input.focus();
      input.select();
    }
  }

  /**
   * Go back in navigation history
   */
  goBack() {
    if (!this.config.enableNavigationHistory || this.historyIndex <= 0) {
      return;
    }

    this.historyIndex--;
    const tradeId = this.navigationHistory[this.historyIndex];

    if (tradeId && this.tradeSelector) {
      // Temporarily disable history recording
      const tempDisable = this.config.enableNavigationHistory;
      this.config.enableNavigationHistory = false;

      this.tradeSelector.selectTrade(tradeId);

      this.config.enableNavigationHistory = tempDisable;
      this.announceNavigation('Went back');
    }
  }

  /**
   * Go forward in navigation history
   */
  goForward() {
    if (
      !this.config.enableNavigationHistory ||
      this.historyIndex >= this.navigationHistory.length - 1
    ) {
      return;
    }

    this.historyIndex++;
    const tradeId = this.navigationHistory[this.historyIndex];

    if (tradeId && this.tradeSelector) {
      // Temporarily disable history recording
      const tempDisable = this.config.enableNavigationHistory;
      this.config.enableNavigationHistory = false;

      this.tradeSelector.selectTrade(tradeId);

      this.config.enableNavigationHistory = tempDisable;
      this.announceNavigation('Went forward');
    }
  }

  /**
   * Clear current selection
   */
  clearSelection() {
    if (this.tradeSelector) {
      this.tradeSelector.clearSelection();
    }
    this.announceNavigation('Selection cleared');
  }

  /**
   * Toggle favorite status of current trade
   */
  toggleFavorite() {
    if (this.currentIndex >= 0 && this.filteredTrades[this.currentIndex]) {
      const trade = this.filteredTrades[this.currentIndex];

      // This would integrate with a favorites system
      trade.isFavorite = !trade.isFavorite;

      const status = trade.isFavorite ? 'added to' : 'removed from';
      this.announceNavigation(`Trade ${status} favorites`);

      // Fire custom event
      document.dispatchEvent(
        new CustomEvent('tradeFavoriteToggled', {
          detail: { tradeId: trade.id, isFavorite: trade.isFavorite },
        })
      );
    }
  }

  /**
   * Add trade to navigation history
   * @param {string} tradeId - Trade identifier
   */
  addToHistory(tradeId) {
    if (!this.config.enableNavigationHistory) {
      return;
    }

    // Remove any forward history if we're not at the end
    if (this.historyIndex < this.navigationHistory.length - 1) {
      this.navigationHistory.splice(this.historyIndex + 1);
    }

    // Add new entry
    this.navigationHistory.push(tradeId);
    this.historyIndex = this.navigationHistory.length - 1;

    // Limit history size
    const maxHistory = 50;
    if (this.navigationHistory.length > maxHistory) {
      this.navigationHistory.shift();
      this.historyIndex--;
    }
  }

  /**
   * Update current index based on trade ID
   * @param {string} tradeId - Trade identifier
   */
  updateCurrentIndex(tradeId) {
    this.currentIndex = this.filteredTrades.findIndex(
      (trade) => trade.id === tradeId
    );
  }

  /**
   * Update filtered trades list
   * @param {Array} filteredTrades - New filtered trades array
   */
  updateFilteredTrades(filteredTrades) {
    this.filteredTrades = filteredTrades;

    // Update current index if current trade is still visible
    if (this.currentIndex >= 0) {
      const currentTrade = this.trades[this.currentIndex];
      if (currentTrade) {
        this.currentIndex = filteredTrades.findIndex(
          (trade) => trade.id === currentTrade.id
        );
      }
    }

    this.updateUI();
  }

  /**
   * Update trades list
   * @param {Array} trades - New trades array
   */
  updateTrades(trades) {
    this.trades = trades;
    this.filteredTrades = [...trades];
    this.currentIndex = -1;
    this.updateUI();
  }

  /**
   * Update navigation UI
   */
  updateUI() {
    const currentIndexElement = document.getElementById('current-trade-index');
    const totalTradesElement = document.getElementById('total-trades');

    if (currentIndexElement) {
      currentIndexElement.textContent =
        this.currentIndex >= 0 ? this.currentIndex + 1 : '-';
    }

    if (totalTradesElement) {
      totalTradesElement.textContent = this.filteredTrades.length;
    }

    // Update button states
    this.updateButtonStates();

    // Update jump input max value
    const jumpInput = document.getElementById('trade-jump-input');
    if (jumpInput) {
      jumpInput.max = this.filteredTrades.length;
    }
  }

  /**
   * Update navigation button states
   */
  updateButtonStates() {
    const isFirst = this.currentIndex <= 0;
    const isLast = this.currentIndex >= this.filteredTrades.length - 1;
    const hasSelection = this.currentIndex >= 0;
    const canGoBack = this.historyIndex > 0;
    const canGoForward = this.historyIndex < this.navigationHistory.length - 1;

    document
      .getElementById('nav-first')
      ?.toggleAttribute('disabled', isFirst || !hasSelection);
    document
      .getElementById('nav-prev')
      ?.toggleAttribute('disabled', isFirst || !hasSelection);
    document
      .getElementById('nav-next')
      ?.toggleAttribute('disabled', isLast || !hasSelection);
    document
      .getElementById('nav-last')
      ?.toggleAttribute('disabled', isLast || !hasSelection);
    document
      .getElementById('nav-back')
      ?.toggleAttribute('disabled', !canGoBack);
    document
      .getElementById('nav-forward')
      ?.toggleAttribute('disabled', !canGoForward);
  }

  /**
   * Show status message
   * @param {string} message - Status message
   * @param {string} type - Message type ('info', 'error', 'success')
   */
  showStatus(message, type = 'info') {
    const statusElement = document.getElementById('nav-status');
    if (!statusElement) {
      return;
    }

    statusElement.textContent = message;
    statusElement.className = `trade-nav-status ${type}`;

    // Auto-hide after 3 seconds
    setTimeout(() => {
      statusElement.textContent = '';
      statusElement.className = 'trade-nav-status';
    }, 3000);
  }

  /**
   * Announce navigation for accessibility
   * @param {string} action - Action description
   */
  announceNavigation(action) {
    if (!this.config.announceNavigation) {
      return;
    }

    const currentTrade =
      this.currentIndex >= 0 ? this.filteredTrades[this.currentIndex] : null;
    const tradeInfo = currentTrade
      ? ` - Trade ${this.currentIndex + 1} of ${this.filteredTrades.length}`
      : '';

    this.showStatus(`${action}${tradeInfo}`);
  }

  /**
   * Create keyboard shortcut help
   */
  createKeyboardShortcutHelp() {
    // This would create a help overlay or modal
    // Implementation would be added based on UI requirements
  }

  /**
   * Show keyboard shortcuts help
   */
  showKeyboardHelp() {
    // Implementation would show keyboard shortcuts overlay
    console.log('Keyboard shortcuts:', this.config.keyboardShortcuts);
  }

  /**
   * Update configuration
   * @param {Object} newConfig - New configuration options
   */
  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Get current navigation state
   * @returns {Object} Navigation state
   */
  getNavigationState() {
    return {
      currentIndex: this.currentIndex,
      totalTrades: this.filteredTrades.length,
      historyLength: this.navigationHistory.length,
      historyIndex: this.historyIndex,
      canGoBack: this.historyIndex > 0,
      canGoForward: this.historyIndex < this.navigationHistory.length - 1,
    };
  }

  /**
   * Clean up resources
   */
  destroy() {
    // Remove event listeners
    document.removeEventListener('keydown', this.handleKeydown);

    // Clear timers
    if (this.keySequenceTimeout) {
      clearTimeout(this.keySequenceTimeout);
    }

    // Remove UI
    const navElement = document.getElementById('trade-navigation');
    if (navElement) {
      navElement.remove();
    }

    // Reset state
    this.currentIndex = -1;
    this.trades = [];
    this.filteredTrades = [];
    this.navigationHistory = [];
    this.historyIndex = -1;
    this.keySequence = [];
    this.initialized = false;

    console.log('TradeNavigation destroyed');
  }

  /**
   * Get module status information
   * @returns {Object} Status information
   */
  getStatus() {
    return {
      initialized: this.initialized,
      currentIndex: this.currentIndex,
      totalTrades: this.trades.length,
      filteredTrades: this.filteredTrades.length,
      navigationHistoryLength: this.navigationHistory.length,
      keyboardNavigationEnabled: this.config.enableKeyboardNavigation,
      hasTradeSelector: !!this.tradeSelector,
    };
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TradeNavigation;
}

// Global registration for direct script inclusion
if (typeof window !== 'undefined') {
  window.TradeNavigation = TradeNavigation;
}
