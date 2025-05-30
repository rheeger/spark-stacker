/**
 * Multi-Scenario Analysis JavaScript Module
 * Provides interactive functionality for multi-scenario backtesting reports
 */

// Global state management
let currentScenario = null;
const currentFilters = {
  scenario: 'all',
  sortBy: 'return',
  primaryMetric: 'return',
};

/**
 * Initialize multi-scenario functionality
 */
function initializeMultiScenario() {
  console.log('Initializing multi-scenario analysis...');

  // Set up initial state
  const firstTab = document.querySelector('.scenario-tab.active');
  if (firstTab) {
    currentScenario = firstTab.textContent
      .trim()
      .toLowerCase()
      .replace(/\s+/g, '_');
  }

  // Initialize expandable sections
  initializeExpandableSections();

  // Initialize accessibility features
  initializeAccessibility();

  console.log('Multi-scenario analysis initialized');
}

/**
 * Show specific scenario tab
 */
function showScenario(scenarioName) {
  // Update tab states
  const tabs = document.querySelectorAll('.scenario-tab');
  const contents = document.querySelectorAll('.scenario-content');

  tabs.forEach((tab) => {
    tab.classList.remove('active');
    tab.setAttribute('aria-selected', 'false');
  });

  contents.forEach((content) => {
    content.classList.remove('active');
  });

  // Activate selected tab and content
  const targetTab = document.querySelector(
    `[aria-controls="${scenarioName}-content"]`
  );
  const targetContent = document.getElementById(`${scenarioName}-content`);

  if (targetTab && targetContent) {
    targetTab.classList.add('active');
    targetTab.setAttribute('aria-selected', 'true');
    targetContent.classList.add('active');

    currentScenario = scenarioName;

    // Generate charts for this scenario if needed
    if (scenarioName === 'comparison') {
      generateComparisonCharts();
    } else {
      generateScenarioCharts(scenarioName);
    }
  }
}

/**
 * Toggle expandable sections
 */
function toggleSection(sectionId) {
  const content = document.getElementById(sectionId);
  const header = content.previousElementSibling;
  const icon = header.querySelector('.expand-icon');

  if (content.classList.contains('expanded')) {
    content.classList.remove('expanded');
    icon.classList.remove('expanded');
  } else {
    content.classList.add('expanded');
    icon.classList.add('expanded');
  }
}

/**
 * Initialize expandable sections
 */
function initializeExpandableSections() {
  const sections = document.querySelectorAll('.expandable-section');

  sections.forEach((section) => {
    const header = section.querySelector('.section-header');
    const content = section.querySelector('.section-content');

    if (header && content) {
      header.addEventListener('click', () => {
        toggleSection(content.id);
      });

      // Add keyboard support
      header.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          toggleSection(content.id);
        }
      });

      header.setAttribute('tabindex', '0');
      header.setAttribute('role', 'button');
    }
  });
}

/**
 * Filter scenarios
 */
function filterScenarios() {
  const filterValue = document.getElementById('scenario-filter').value;
  currentFilters.scenario = filterValue;

  const tabs = document.querySelectorAll(
    '.scenario-tab:not([aria-controls="comparison-content"])'
  );

  tabs.forEach((tab) => {
    const tabName = tab.textContent.trim().toLowerCase();
    let show = true;

    switch (filterValue) {
      case 'profitable':
        show = isProfitableScenario(tabName);
        break;
      case 'unprofitable':
        show = !isProfitableScenario(tabName);
        break;
      case 'high-vol':
        show = isHighVolScenario(tabName);
        break;
      case 'trending':
        show = isTrendingScenario(tabName);
        break;
      default:
        show = true;
    }

    tab.style.display = show ? 'block' : 'none';
  });

  updateDashboardMetrics();
}

/**
 * Sort scenarios
 */
function sortScenarios() {
  const sortBy = document.getElementById('sort-by').value;
  currentFilters.sortBy = sortBy;

  const tabContainer = document.querySelector('.scenario-tabs');
  const tabs = Array.from(
    tabContainer.querySelectorAll(
      '.scenario-tab:not([aria-controls="comparison-content"])'
    )
  );
  const comparisonTab = tabContainer.querySelector(
    '[aria-controls="comparison-content"]'
  );

  tabs.sort((a, b) => {
    const aValue = getScenarioMetric(a.textContent.trim(), sortBy);
    const bValue = getScenarioMetric(b.textContent.trim(), sortBy);

    // Sort in descending order (best first) except for drawdown
    if (sortBy === 'drawdown') {
      return aValue - bValue; // Lower drawdown is better
    } else {
      return bValue - aValue; // Higher values are better
    }
  });

  // Clear and rebuild tab container
  tabContainer.innerHTML = '';
  tabs.forEach((tab) => tabContainer.appendChild(tab));
  if (comparisonTab) {
    tabContainer.appendChild(comparisonTab);
  }
}

/**
 * Update primary metric display
 */
function updatePrimaryMetric() {
  const metric = document.getElementById('metric-display').value;
  currentFilters.primaryMetric = metric;

  updateDashboardMetrics();
  updateHeatmap();
}

/**
 * Update heatmap visualization
 */
function updateHeatmap() {
  const metric = document.getElementById('heatmap-metric').value;
  const colorScale = document.getElementById('color-scale').value;

  if (
    window.visualizationData &&
    window.visualizationData.performance_heatmap
  ) {
    const heatmapData = window.visualizationData.performance_heatmap;
    generateHeatmapChart(heatmapData, metric, colorScale);
  }
}

/**
 * Export scenario data
 */
function exportScenarioData(format) {
  if (!window.scenarioData) {
    alert('No data available for export');
    return;
  }

  const data = prepareExportData();

  switch (format) {
    case 'json':
      downloadJSON(data, 'scenario_analysis.json');
      break;
    case 'csv':
      downloadCSV(data, 'scenario_analysis.csv');
      break;
    case 'pdf':
      generatePDFReport();
      break;
    default:
      console.warn('Unsupported export format:', format);
  }
}

/**
 * Print report
 */
function printReport() {
  window.print();
}

/**
 * Highlight specific trade
 */
function highlightTrade(scenarioName, tradeIndex) {
  console.log(`Highlighting trade ${tradeIndex} in scenario ${scenarioName}`);

  // Remove existing highlights
  document.querySelectorAll('.trade-item.highlighted').forEach((item) => {
    item.classList.remove('highlighted');
  });

  // Highlight selected trade
  const tradeItem = document.querySelector(
    `#${scenarioName}-content .trade-item:nth-child(${tradeIndex + 2})`
  );
  if (tradeItem) {
    tradeItem.classList.add('highlighted');
  }

  // Update chart highlighting if available
  if (window.visualizationData) {
    updateTradeHighlighting(scenarioName, tradeIndex);
  }
}

/**
 * Generate charts for all scenarios
 */
function generateCharts() {
  console.log('Generating charts...');

  if (!window.visualizationData) {
    console.warn('No visualization data available');
    return;
  }

  // Generate main dashboard charts
  generateRobustnessChart();
  generateHeatmapChart();

  // Generate scenario-specific charts
  Object.keys(window.scenarioData || {}).forEach((scenarioName) => {
    generateScenarioCharts(scenarioName);
  });

  // Generate comparison charts
  generateComparisonCharts();

  console.log('Charts generated successfully');
}

/**
 * Generate robustness analysis chart
 */
function generateRobustnessChart() {
  const container = document.getElementById('robustness-chart');
  if (!container || !window.robustnessData) return;

  const data = [
    {
      type: 'bar',
      x: ['Consistency', 'Adaptability', 'Risk-Adjusted'],
      y: [
        window.robustnessData.consistency_score || 0,
        window.robustnessData.adaptability_score || 0,
        window.robustnessData.risk_adjusted_robustness || 0,
      ],
      marker: {
        color: ['#007bff', '#28a745', '#ffc107'],
      },
    },
  ];

  const layout = {
    title: 'Robustness Scores',
    yaxis: { title: 'Score (0-100)', range: [0, 100] },
    showlegend: false,
    responsive: true,
  };

  Plotly.newPlot(container, data, layout, { responsive: true });
}

/**
 * Generate scenario performance heatmap
 */
function generateHeatmapChart(
  heatmapData,
  metric = 'return',
  colorScale = 'RdYlGn'
) {
  const container = document.getElementById('scenario-heatmap');
  if (!container) return;

  // Use provided data or default from window
  const data =
    heatmapData ||
    (window.visualizationData && window.visualizationData.performance_heatmap);
  if (!data) return;

  const heatmapTrace = {
    type: 'heatmap',
    z: data.data.map((row) => row.values),
    x: data.scenarios,
    y: data.data.map((row) => row.metric_label),
    colorscale: colorScale,
    showscale: true,
  };

  const layout = {
    title: `Scenario Performance Heatmap - ${metric}`,
    xaxis: { title: 'Scenarios' },
    yaxis: { title: 'Metrics' },
    responsive: true,
  };

  Plotly.newPlot(container, [heatmapTrace], layout, { responsive: true });
}

/**
 * Generate charts for specific scenario
 */
function generateScenarioCharts(scenarioName) {
  generateEquityChart(scenarioName);
  generateTradeDistributionChart(scenarioName);
  generateTradeHighlightChart(scenarioName);
}

/**
 * Generate equity curve chart for scenario
 */
function generateEquityChart(scenarioName) {
  const container = document.getElementById(`${scenarioName}-equity-chart`);
  if (!container) return;

  const scenarioData = window.scenarioData && window.scenarioData[scenarioName];
  if (!scenarioData || !scenarioData.equity_curve) return;

  const trace = {
    type: 'scatter',
    mode: 'lines',
    x: scenarioData.equity_curve.map((_, i) => i),
    y: scenarioData.equity_curve,
    name: 'Equity',
    line: { color: '#007bff' },
  };

  const layout = {
    title: `${scenarioName} Equity Curve`,
    xaxis: { title: 'Time' },
    yaxis: { title: 'Portfolio Value' },
    responsive: true,
  };

  Plotly.newPlot(container, [trace], layout, { responsive: true });
}

/**
 * Generate trade distribution chart
 */
function generateTradeDistributionChart(scenarioName) {
  const container = document.getElementById(`${scenarioName}-trade-chart`);
  if (!container) return;

  const scenarioData = window.scenarioData && window.scenarioData[scenarioName];
  if (!scenarioData || !scenarioData.trades) return;

  const profits = scenarioData.trades.map((trade) => trade.pnl || 0);

  const trace = {
    type: 'histogram',
    x: profits,
    nbinsx: 20,
    marker: { color: '#28a745' },
  };

  const layout = {
    title: `${scenarioName} Trade P&L Distribution`,
    xaxis: { title: 'Profit/Loss (%)' },
    yaxis: { title: 'Frequency' },
    responsive: true,
  };

  Plotly.newPlot(container, [trace], layout, { responsive: true });
}

/**
 * Generate trade highlighting chart
 */
function generateTradeHighlightChart(scenarioName) {
  const container = document.getElementById(`${scenarioName}-trade-highlights`);
  if (!container) return;

  // Placeholder for trade highlighting chart
  // This would integrate with the main price chart when available
  container.innerHTML =
    '<p>Trade highlighting chart will be generated when price data is available</p>';
}

/**
 * Generate comparison charts
 */
function generateComparisonCharts() {
  generateRadarChart();
  generateEquityOverlayChart();
}

/**
 * Generate radar chart comparison
 */
function generateRadarChart() {
  const container = document.getElementById('radar-comparison-chart');
  if (!container || !window.visualizationData) return;

  const radarData = window.visualizationData.radar_chart;
  if (!radarData) return;

  const traces = radarData.data.map((scenario) => ({
    type: 'scatterpolar',
    r: scenario.values,
    theta: radarData.dimension_labels,
    fill: 'toself',
    name: scenario.scenario_label,
    line: { color: scenario.color },
  }));

  const layout = {
    polar: {
      radialaxis: { visible: true, range: [0, 100] },
    },
    title: 'Multi-Scenario Performance Radar',
    responsive: true,
  };

  Plotly.newPlot(container, traces, layout, { responsive: true });
}

/**
 * Generate equity curves overlay
 */
function generateEquityOverlayChart() {
  const container = document.getElementById('equity-overlay-chart');
  if (!container || !window.visualizationData) return;

  const overlayData = window.visualizationData.equity_curves_overlay;
  if (!overlayData) return;

  const traces = overlayData.curves.map((curve) => ({
    type: 'scatter',
    mode: 'lines',
    x: curve.data.map((point) => point.timestamp),
    y: curve.data.map((point) => point.percentage_gain),
    name: curve.scenario_label,
    line: { color: curve.color },
  }));

  const layout = {
    title: 'Equity Curves Overlay - All Scenarios',
    xaxis: { title: 'Time' },
    yaxis: { title: 'Percentage Gain/Loss (%)' },
    responsive: true,
  };

  Plotly.newPlot(container, traces, layout, { responsive: true });
}

// Helper functions

function isProfitableScenario(scenarioName) {
  const data = window.scenarioData && window.scenarioData[scenarioName];
  return data && data.total_return_pct > 0;
}

function isHighVolScenario(scenarioName) {
  return (
    scenarioName.includes('high') ||
    scenarioName.includes('volatile') ||
    scenarioName.includes('choppy')
  );
}

function isTrendingScenario(scenarioName) {
  return scenarioName.includes('bull') || scenarioName.includes('bear');
}

function getScenarioMetric(scenarioName, metric) {
  const data = window.scenarioData && window.scenarioData[scenarioName];
  if (!data) return 0;

  switch (metric) {
    case 'return':
      return data.total_return_pct || 0;
    case 'sharpe':
      return data.sharpe_ratio || 0;
    case 'drawdown':
      return Math.abs(data.max_drawdown_pct || 0);
    case 'win-rate':
      return (data.win_rate || 0) * 100;
    default:
      return 0;
  }
}

function updateDashboardMetrics() {
  // Update dashboard based on current filters
  const visibleScenarios = getVisibleScenarios();
  const metric = currentFilters.primaryMetric;

  // Update overall performance metric
  const avgValue = calculateAverageMetric(visibleScenarios, metric);
  const overallElement = document.getElementById('overall-return');
  if (overallElement) {
    overallElement.textContent = `${avgValue.toFixed(2)}%`;
  }
}

function getVisibleScenarios() {
  const tabs = document.querySelectorAll(
    '.scenario-tab:not([style*="display: none"])'
  );
  return Array.from(tabs).map((tab) => tab.textContent.trim());
}

function calculateAverageMetric(scenarios, metric) {
  if (scenarios.length === 0) return 0;

  const values = scenarios.map((scenario) =>
    getScenarioMetric(scenario, metric)
  );
  return values.reduce((sum, val) => sum + val, 0) / values.length;
}

function prepareExportData() {
  return {
    strategy_name: document.querySelector('h2').textContent,
    generated_at: new Date().toISOString(),
    scenarios: window.scenarioData || {},
    robustness_analysis: window.robustnessData || {},
    visualization_data: window.visualizationData || {},
  };
}

function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadCSV(data, filename) {
  // Convert scenario data to CSV format
  const scenarios = data.scenarios || {};
  const headers = [
    'Scenario',
    'Total Return (%)',
    'Win Rate (%)',
    'Sharpe Ratio',
    'Max Drawdown (%)',
  ];

  let csv = headers.join(',') + '\n';

  Object.entries(scenarios).forEach(([name, scenario]) => {
    const row = [
      name,
      scenario.total_return_pct || 0,
      (scenario.win_rate || 0) * 100,
      scenario.sharpe_ratio || 'N/A',
      scenario.max_drawdown_pct || 'N/A',
    ];
    csv += row.join(',') + '\n';
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function generatePDFReport() {
  // This would integrate with a PDF generation library
  alert(
    'PDF export functionality would be implemented with a PDF library like jsPDF'
  );
}

function updateTradeHighlighting(scenarioName, tradeIndex) {
  // This would update trade highlighting on charts
  console.log(
    `Updating trade highlighting for ${scenarioName}, trade ${tradeIndex}`
  );
}

function initializeAccessibility() {
  // Add ARIA labels and keyboard navigation
  const tabs = document.querySelectorAll('.scenario-tab');
  tabs.forEach((tab, index) => {
    tab.setAttribute('tabindex', index === 0 ? '0' : '-1');

    tab.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        const direction = e.key === 'ArrowRight' ? 1 : -1;
        const currentIndex = Array.from(tabs).indexOf(tab);
        const nextIndex =
          (currentIndex + direction + tabs.length) % tabs.length;
        const nextTab = tabs[nextIndex];

        tab.setAttribute('tabindex', '-1');
        nextTab.setAttribute('tabindex', '0');
        nextTab.focus();
        nextTab.click();
      }
    });
  });
}
