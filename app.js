// public/app.js

// Utility functions
function formatCurrency(amount) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(amount);
}

function formatDate(dateString) {
  return new Date(dateString).toLocaleString();
}

function formatTimeAgo(dateString) {
  const now = new Date();
  const date = new Date(dateString);
  const diffInMinutes = Math.floor((now - date) / (1000 * 60));
  if (diffInMinutes < 1) return 'Just now';
  if (diffInMinutes < 60) return `${diffInMinutes} min ago`;
  if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)} hours ago`;
  return `${Math.floor(diffInMinutes / 1440)} days ago`;
}

function getInventoryStatus(level) {
  if (level < 15) return 'low';
  if (level < 30) return 'medium';
  return 'high';
}

// State
let products = [];
let metrics = {};
let pricingHistory = [];
let integrations = [];
let currentUser = { name: "Alex Chen", company: "TechStore Pro" };
let filteredProducts = [];
let charts = {};
let onboardingStep = 1;

// On load
window.onload = () => {
  // Attach page nav
  document.querySelectorAll('.nav-item').forEach(item => {
    item.onclick = e => {
      e.preventDefault();
      switchPage(item.dataset.page);
    }
  });

  // Attach login
  document.getElementById('login-form').onsubmit = handleLogin;

  // Onboarding buttons
  document.getElementById('prev-btn').onclick = prevOnboardingStep;
  document.getElementById('next-btn').onclick = nextOnboardingStep;
  document.getElementById('finish-btn').onclick = finishOnboarding;

  // Initial data load
  loadAllData();
}

// -----------------------
// Auth & Onboarding
// -----------------------

function showLogin() {
  document.getElementById('login-modal').classList.remove('hidden');
}

function hideLogin() {
  document.getElementById('login-modal').classList.add('hidden');
}

function handleLogin(e) {
  e.preventDefault();
  hideLogin();
  document.getElementById('landing-page').classList.remove('active');
  document.getElementById('main-app').classList.add('active');
  setTimeout(showOnboarding, 400);
}

function showOnboarding() {
  document.getElementById('onboarding-modal').classList.remove('hidden');
  onboardingStep = 1;
  updateOnboardingDisplay();
}
function hideOnboarding() {
  document.getElementById('onboarding-modal').classList.add('hidden');
}
function updateOnboardingDisplay() {
  document.querySelectorAll('.onboarding-step').forEach(step => step.classList.add('hidden'));
  document.querySelector(`[data-step="${onboardingStep}"]`).classList.remove('hidden');
  document.getElementById('current-step').textContent = onboardingStep;
  document.querySelector('.progress-fill').style.width = `${(onboardingStep / 4) * 100}%`;
  document.getElementById('prev-btn').disabled = onboardingStep === 1;
  if (onboardingStep === 4) {
    document.getElementById('next-btn').classList.add('hidden');
    document.getElementById('finish-btn').classList.remove('hidden');
  } else {
    document.getElementById('next-btn').classList.remove('hidden');
    document.getElementById('finish-btn').classList.add('hidden');
  }
}
function nextOnboardingStep() {
  if (onboardingStep < 4) onboardingStep++;
  updateOnboardingDisplay();
}
function prevOnboardingStep() {
  if (onboardingStep > 1) onboardingStep--;
  updateOnboardingDisplay();
}
function finishOnboarding() { hideOnboarding(); }

// -----------------------
// Page navigation
// -----------------------

function switchPage(pageName) {
  document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
  document.querySelector(`[data-page="${pageName}"]`).classList.add('active');
  document.querySelectorAll('.content-page').forEach(page => page.classList.remove('active'));
  document.getElementById(`${pageName}-page`).classList.add('active');
  if (pageName === 'dashboard') setTimeout(initializeDashboard, 100);
  if (pageName === 'products') setTimeout(initializeProducts, 100);
  if (pageName === 'analytics') setTimeout(initializeAnalytics, 100);
  if (pageName === 'integrations') setTimeout(initializeIntegrations, 100);
}

async function loadAllData() {
  [products, metrics, pricingHistory, integrations] = await Promise.all([
    fetch('/api/products').then(r=>r.json()),
    fetch('/api/metrics').then(r=>r.json()),
    fetch('/api/pricing-history').then(r=>r.json()),
    fetch('/api/integrations').then(r=>r.json()),
  ]);
  filteredProducts = [...products];
  initializeDashboard();
}

function refreshData() { loadAllData(); }

// -----------------------
// Dashboard
// -----------------------
function initializeDashboard() {
  updateMetrics();
  renderRecentActivity();
  setTimeout(initializeCharts, 300);
}

function updateMetrics() {
  document.getElementById('total-revenue').textContent = formatCurrency(metrics.totalRevenue);
  document.getElementById('active-products').textContent = metrics.activeProducts;
  document.getElementById('price-changes').textContent = metrics.priceChangesToday;
  document.getElementById('profit-margin').textContent = `${metrics.averageProfitMargin}%`;
}

function renderRecentActivity() {
  const activityContainer = document.getElementById('recent-activity');
  activityContainer.innerHTML = '';
  pricingHistory.forEach(activity => {
    const div = document.createElement('div');
    div.className = 'activity-item';
    div.innerHTML = `
      <div class="activity-info"><h4>${activity.productName}</h4><p>${activity.reason}</p></div>
      <div class="activity-meta">
        <div class="activity-impact">${activity.impact}</div>
        <div class="activity-time">${formatTimeAgo(activity.timestamp)}</div>
      </div>`;
    activityContainer.appendChild(div);
  });
}

function initializeCharts() {
  // Revenue chart
  const revenueCtx = document.getElementById('revenue-chart');
  if (revenueCtx) {
    if (charts.revenue) charts.revenue.destroy();
    charts.revenue = new Chart(revenueCtx, {
      type: 'line',
      data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        datasets: [{
          label: 'Revenue',
          data: [980000, 1050000, 1120000, 1180000, 1220000, metrics.totalRevenue],
          borderColor: '#1FB8CD',
          backgroundColor: 'rgba(31, 184, 205, 0.1)',
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: { y: { beginAtZero: false, ticks: { callback: formatCurrency } } },
        plugins: { legend: { display: false } }
      }
    });
  }
  // Price impact chart
  const priceImpactCtx = document.getElementById('price-impact-chart');
  if (priceImpactCtx) {
    if (charts.priceImpact) charts.priceImpact.destroy();
    charts.priceImpact = new Chart(priceImpactCtx, {
      type: 'bar',
      data: {
        labels: ['Electronics', 'Fashion', 'Home', 'Sports', 'Books'],
        datasets: [{
          label: 'Revenue Impact',
          data: [25000, 18000, 12000, 8000, 5000],
          backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F']
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: { y: { beginAtZero: true, ticks: { callback: formatCurrency } } },
        plugins: { legend: { display: false } }
      }
    });
  }
}

// -----------------------
// Products Page
// -----------------------

function initializeProducts() {
  filteredProducts = [...products];
  renderProductsTable();
  setupProductFilters();
}

function renderProductsTable() {
  const tableBody = document.getElementById('products-table-body');
  tableBody.innerHTML = '';
  filteredProducts.forEach(product => {
    const row = document.createElement('tr');
    const inventoryStatus = getInventoryStatus(product.inventoryLevel);
    const priceChange = ((product.suggestedPrice - product.currentPrice) / product.currentPrice * 100).toFixed(1);
    const changeClass = priceChange > 0 ? 'positive' : 'negative';
    row.innerHTML = `
      <td><div class="product-name">${product.name}</div>
          <div class="product-category">${product.category}</div></td>
      <td>${product.category}</td>
      <td class="price-current">${formatCurrency(product.currentPrice)}</td>
      <td class="price-suggested">
        ${formatCurrency(product.suggestedPrice)}
        <span class="metric-trend ${changeClass}" style="font-size: 11px; margin-left: 4px;">
          ${priceChange > 0 ? '+' : ''}${priceChange}%
        </span>
      </td>
      <td>
        <div class="demand-score">
          <span>${product.demandScore}</span>
          <div class="demand-bar"><div class="demand-fill" style="width: ${product.demandScore}%"></div></div>
        </div>
      </td>
      <td class="inventory-level ${inventoryStatus}">${product.inventoryLevel}</td>
      <td>
        <button class="btn btn--primary btn--sm" onclick="showPriceModal('${product.id}')">
          Adjust Price
        </button>
      </td>
    `;
    tableBody.appendChild(row);
  });
}

function setupProductFilters() {
  const searchInput = document.getElementById('product-search');
  const categoryFilter = document.getElementById('category-filter');
  searchInput.oninput = filterProducts;
  categoryFilter.onchange = filterProducts;
}

function filterProducts() {
  const searchInput = document.getElementById('product-search');
  const categoryFilter = document.getElementById('category-filter');
  const searchTerm = searchInput.value.toLowerCase();
  const selectedCategory = categoryFilter.value;
  filteredProducts = products.filter(product => {
    const matchesSearch = product.name.toLowerCase().includes(searchTerm) || 
                         product.category.toLowerCase().includes(searchTerm);
    const matchesCategory = !selectedCategory || product.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });
  renderProductsTable();
}

// --- Price Modal ---
function showPriceModal(productId) {
  const product = products.find(p => p.id === productId);
  if (!product) return;
  const modal = document.createElement('div');
  modal.className = 'modal';
  modal.innerHTML = `
    <div class="modal-backdrop" onclick="this.parentNode.remove()"></div>
    <div class="modal-content" id="price-modal-content">
      <h3>Adjust Price for ${product.name}</h3>
      <div class="price-comparison" style="display:flex;gap:16px;">
        <div>
          <div class="price-label">Current</div>
          <div class="price-amount">${formatCurrency(product.currentPrice)}</div>
        </div>
        <div>
          <div class="price-label">Suggested</div>
          <div class="price-amount">${formatCurrency(product.suggestedPrice)}</div>
        </div>
      </div>
      <div class="form-group">
        <label>Custom Price</label>
        <input type="number" class="form-control" id="custom-price" value="${product.suggestedPrice}" min="0" step="0.01">
      </div>
      <div class="form-group">
        <label>Reason for Change</label>
        <select class="form-control" id="price-reason">
          <option>AI Recommendation</option>
          <option>Competitor Analysis</option>
          <option>Inventory Management</option>
          <option>Demand Surge</option>
          <option>Manual Override</option>
        </select>
      </div>
      <div style="margin-top:16px;display:flex;gap:8px;">
        <button class="btn btn--primary" id="confirm-price-btn">Confirm</button>
        <button class="btn btn--outline" onclick="this.closest('.modal').remove()">Cancel</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);

  document.getElementById('confirm-price-btn').onclick = async () => {
    const newPrice = parseFloat(document.getElementById('custom-price').value);
    const reason = document.getElementById('price-reason').value;
    await fetch(`/api/products/${product.id}/price`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ newPrice, reason })
    });
    modal.remove();
    loadAllData(); // Refresh all data
  };
}

// -----------------------
// Analytics Page
// -----------------------
function initializeAnalytics() {
  setTimeout(initializeAnalyticsCharts, 100);
}
function initializeAnalyticsCharts() {
  // Competitor analysis chart
  const competitorCtx = document.getElementById('competitor-chart');
  if (competitorCtx) {
    if (charts.competitor) charts.competitor.destroy();
    charts.competitor = new Chart(competitorCtx, {
      type: 'radar',
      data: {
        labels: ['Price', 'Quality', 'Selection', 'Service', 'Delivery', 'Brand'],
        datasets: [{
          label: 'Our Store',
          data: [85, 92, 88, 90, 87, 85],
          borderColor: '#1FB8CD',
          backgroundColor: 'rgba(31, 184, 205, 0.2)',
          pointBackgroundColor: '#1FB8CD'
        }, {
          label: 'Competitor Average',
          data: [78, 85, 85, 82, 88, 90],
          borderColor: '#B4413C',
          backgroundColor: 'rgba(180, 65, 60, 0.2)',
          pointBackgroundColor: '#B4413C'
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: { r: { beginAtZero: true, max: 100 } }
      }
    });
  }
  // Demand correlation chart
  const demandCtx = document.getElementById('demand-chart');
  if (demandCtx) {
    if (charts.demand) charts.demand.destroy();
    const scatterData = products.map(product => ({ x: product.currentPrice, y: product.demandScore }));
    charts.demand = new Chart(demandCtx, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Price vs Demand',
          data: scatterData,
          backgroundColor: '#1FB8CD',
          borderColor: '#1FB8CD'
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'Price ($)' }, ticks: { callback: formatCurrency } },
          y: { title: { display: true, text: 'Demand Score' }, beginAtZero: true, max: 100 }
        },
        plugins: { legend: { display: false } }
      }
    });
  }
}

// -----------------------
// Integrations Page
// -----------------------
function initializeIntegrations() {
  const grid = document.querySelector('.integrations-grid');
  if (!grid) return;
  grid.innerHTML = '';
  integrations.forEach(integration => {
    const card = document.createElement('div');
    card.className = 'integration-card';
    card.innerHTML = `
      <div class="integration-header">
        <div class="integration-info">
          <h3>${integration.name}</h3>
          <div class="status status--success">${integration.status.charAt(0).toUpperCase() + integration.status.slice(1)}</div>
        </div>
      </div>
      <div class="integration-stats">
        <div class="stat"><span class="stat-label">Products</span><span class="stat-value">${integration.productsCount}</span></div>
        <div class="stat"><span class="stat-label">Last Sync</span><span class="stat-value">${formatTimeAgo(integration.lastSync)}</span></div>
      </div>
      <button class="btn btn--outline btn--sm">Configure</button>
    `;
    grid.appendChild(card);
  });
}
