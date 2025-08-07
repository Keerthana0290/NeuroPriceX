const express = require('express');
const cors = require('cors');
const path = require('path');
const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// --- In-memory appData (migrated from browser JS) ---
let appData = {
  users: [
    {
      id: "demo-user",
      email: "demo@pricing-engine.com",
      name: "Alex Chen",
      company: "TechStore Pro",
      role: "Pricing Manager"
    }
  ],
  products: [
    {
      id: "prod-001",
      name: "Wireless Headphones Pro",
      category: "Electronics",
      currentPrice: 199.99,
      suggestedPrice: 219.99,
      demandScore: 85,
      inventoryLevel: 45,
      competitorPrices: [189.99, 209.99, 199.00],
      profitMargin: 35.2,
      lastUpdated: "2024-01-15T10:30:00Z"
    },
    {
      id: "prod-002", 
      name: "Smart Home Hub",
      category: "Electronics",
      currentPrice: 149.99,
      suggestedPrice: 139.99,
      demandScore: 92,
      inventoryLevel: 12,
      competitorPrices: [144.99, 155.00, 149.99],
      profitMargin: 28.7,
      lastUpdated: "2024-01-15T11:15:00Z"
    },
    {
      id: "prod-003",
      name: "Running Shoes Elite",
      category: "Fashion",
      currentPrice: 129.99,
      suggestedPrice: 134.99,
      demandScore: 78,
      inventoryLevel: 67,
      competitorPrices: [125.99, 139.99, 132.00],
      profitMargin: 42.1,
      lastUpdated: "2024-01-15T09:45:00Z"
    },
    {
      id: "prod-004",
      name: "Coffee Maker Deluxe",
      category: "Home",
      currentPrice: 89.99,
      suggestedPrice: 94.99,
      demandScore: 73,
      inventoryLevel: 23,
      competitorPrices: [87.99, 92.99, 95.00],
      profitMargin: 38.5,
      lastUpdated: "2024-01-15T08:20:00Z"
    },
    {
      id: "prod-005",
      name: "Gaming Keyboard RGB",
      category: "Electronics", 
      currentPrice: 79.99,
      suggestedPrice: 84.99,
      demandScore: 88,
      inventoryLevel: 34,
      competitorPrices: [77.99, 82.99, 79.00],
      profitMargin: 45.3,
      lastUpdated: "2024-01-15T12:00:00Z"
    },
    {
      id: "prod-006",
      name: "Leather Jacket Premium",
      category: "Fashion",
      currentPrice: 299.99,
      suggestedPrice: 319.99,
      demandScore: 71,
      inventoryLevel: 8,
      competitorPrices: [289.99, 315.00, 299.00],
      profitMargin: 52.7,
      lastUpdated: "2024-01-15T13:30:00Z"
    }
  ],
  metrics: {
    totalRevenue: 1250000,
    revenueGrowth: 12.5,
    activeProducts: 156,
    priceChangesToday: 23,
    averageProfitMargin: 34.8,
    competitorAnalysis: {
      priceAdvantage: 8.2,
      marketPosition: "Competitive"
    }
  },
  integrations: [
    {
      name: "Shopify Store",
      status: "connected",
      lastSync: "2024-01-15T11:30:00Z",
      productsCount: 89
    },
    {
      name: "WooCommerce Store", 
      status: "connected",
      lastSync: "2024-01-15T11:25:00Z",
      productsCount: 67
    }
  ],
  pricingHistory: [
    {
      timestamp: "2024-01-15T10:00:00Z",
      productId: "prod-001",
      productName: "Wireless Headphones Pro",
      oldPrice: 199.99,
      newPrice: 219.99,
      reason: "High demand detected",
      impact: "+$2,450 projected revenue"
    },
    {
      timestamp: "2024-01-15T09:30:00Z", 
      productId: "prod-002",
      productName: "Smart Home Hub",
      oldPrice: 149.99,
      newPrice: 139.99,
      reason: "Low inventory alert",
      impact: "+15% conversion rate"
    },
    {
      timestamp: "2024-01-15T08:45:00Z",
      productId: "prod-003", 
      productName: "Running Shoes Elite",
      oldPrice: 124.99,
      newPrice: 129.99,
      reason: "Competitor price increase",
      impact: "+$1,890 projected revenue"
    }
  ]
};

// --- API Endpoints ---

app.get('/api/products', (req, res) => {
  res.json(appData.products);
});

app.get('/api/metrics', (req, res) => {
  res.json(appData.metrics);
});

app.get('/api/pricing-history', (req, res) => {
  res.json(appData.pricingHistory);
});

app.get('/api/integrations', (req, res) => {
  res.json(appData.integrations);
});

// Adjust product price
app.post('/api/products/:id/price', (req, res) => {
  const { id } = req.params;
  const { newPrice, reason } = req.body;
  const prod = appData.products.find(p => p.id === id);
  if (!prod) return res.status(404).json({ error: 'Product not found' });
  const oldPrice = prod.currentPrice;
  prod.currentPrice = newPrice;
  prod.lastUpdated = new Date().toISOString();

  // Add to pricing history
  appData.pricingHistory.unshift({
    timestamp: new Date().toISOString(),
    productId: prod.id,
    productName: prod.name,
    oldPrice,
    newPrice,
    reason: reason || "Manual override",
    impact: `New price set by manager`
  });

  res.json({ success: true, product: prod });
});

// --- Fallback: Serve index.html for all unknown routes (for SPA routing) ---
app.get('*', (req, res) => {
  res.sendFile(path.resolve(__dirname, 'public', 'index.html'));
});

// --- Start server ---
app.listen(PORT, () => {
  console.log(`Dynamic Pricing Engine backend running at http://localhost:${PORT}`);
});
