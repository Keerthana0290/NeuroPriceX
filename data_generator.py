import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any

class DataGenerator:
    def __init__(self, seed: int = 42):
        self.fake = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Product categories and their characteristics
        self.categories = {
            'Electronics': {
                'price_range': (50, 2000),
                'margin_range': (0.15, 0.35),
                'seasonality': {'11': 1.3, '12': 1.4, '1': 0.8},
                'elasticity': -1.2
            },
            'Clothing': {
                'price_range': (15, 300),
                'margin_range': (0.45, 0.65),
                'seasonality': {'3': 1.2, '6': 1.1, '9': 1.3, '12': 1.2},
                'elasticity': -1.8
            },
            'Home & Garden': {
                'price_range': (10, 500),
                'margin_range': (0.35, 0.55),
                'seasonality': {'3': 1.3, '4': 1.4, '5': 1.3},
                'elasticity': -1.0
            },
            'Sports': {
                'price_range': (20, 800),
                'margin_range': (0.25, 0.45),
                'seasonality': {'1': 1.2, '5': 1.3, '6': 1.4},
                'elasticity': -1.5
            },
            'Books': {
                'price_range': (5, 100),
                'margin_range': (0.20, 0.40),
                'seasonality': {'9': 1.3, '1': 1.2},
                'elasticity': -0.8
            }
        }
        
        self.competitor_names = [
            'MegaStore', 'PriceMax', 'ValueMart', 'QuickShop', 'BestDeals',
            'SuperSaver', 'EliteStore', 'FastTrade', 'EconoMart', 'PrimeBuy'
        ]
    
    def generate_product_data(self, num_products: int = 100) -> List[Dict]:
        """Generate realistic product data with pricing information"""
        products = []
        
        for i in range(num_products):
            category = random.choice(list(self.categories.keys()))
            category_info = self.categories[category]
            
            # Generate basic product info
            product_name = self._generate_product_name(category)
            
            # Price and cost information
            current_price = round(random.uniform(*category_info['price_range']), 2)
            margin = random.uniform(*category_info['margin_range'])
            cost = round(current_price * (1 - margin), 2)
            
            # Inventory and demand
            inventory_level = random.randint(0, 1000)
            optimal_inventory = random.randint(100, 500)
            historical_demand = random.randint(50, 500)
            daily_demand = random.randint(5, 50)
            
            # Product attributes
            quality_score = round(random.uniform(6, 10), 1)
            brand_strength = random.choice(['Strong', 'Medium', 'Weak'])
            
            # Historical sales data
            sales_history = self._generate_sales_history(current_price, historical_demand)
            
            product = {
                'product_id': f'PROD_{i+1:04d}',
                'product_name': product_name,
                'category': category,
                'current_price': current_price,
                'cost': cost,
                'inventory_level': inventory_level,
                'optimal_inventory': optimal_inventory,
                'historical_demand': historical_demand,
                'daily_demand': daily_demand,
                'quality_score': quality_score,
                'brand_strength': brand_strength,
                'sales_history': sales_history,
                'elasticity': category_info['elasticity'] + random.uniform(-0.3, 0.3),
                'created_date': self.fake.date_between(start_date='-2y', end_date='today'),
                'last_updated': datetime.now()
            }
            
            products.append(product)
        
        return products
    
    def generate_market_data(self) -> Dict[str, Any]:
        """Generate market and competitor data"""
        # Economic indicators
        economic_indicators = {
            'inflation_rate': round(random.uniform(2, 8), 2),
            'unemployment_rate': round(random.uniform(3, 12), 2),
            'consumer_confidence': round(random.uniform(70, 130), 1),
            'gdp_growth': round(random.uniform(-2, 6), 2)
        }
        
        # Market trends
        demand_trend = random.uniform(0.8, 1.3)  # Overall demand multiplier
        market_volatility = random.uniform(0.05, 0.25)  # Price volatility factor
        
        # Competitor pricing
        competitor_prices = {}
        base_price = random.uniform(20, 100)
        
        for competitor in random.sample(self.competitor_names, 5):
            # Competitors price within +/- 20% of base price
            price_variation = random.uniform(0.8, 1.2)
            competitor_prices[competitor] = round(base_price * price_variation, 2)
        
        # Seasonal factors
        current_month = datetime.now().month
        seasonal_multipliers = {
            'Electronics': self._get_seasonal_factor('Electronics', current_month),
            'Clothing': self._get_seasonal_factor('Clothing', current_month),
            'Home & Garden': self._get_seasonal_factor('Home & Garden', current_month),
            'Sports': self._get_seasonal_factor('Sports', current_month),
            'Books': self._get_seasonal_factor('Books', current_month)
        }
        
        return {
            'timestamp': datetime.now(),
            'economic_indicators': economic_indicators,
            'demand_trend': demand_trend,
            'market_volatility': market_volatility,
            'competitor_prices': competitor_prices,
            'seasonal_multipliers': seasonal_multipliers,
            'market_events': self._generate_market_events()
        }
    
    def generate_historical_pricing_data(self, product_id: str, days: int = 365) -> pd.DataFrame:
        """Generate historical pricing and performance data"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate base price trend
        base_price = random.uniform(20, 100)
        price_trend = np.cumsum(np.random.normal(0, 0.02, days)) + base_price
        
        # Add seasonality and noise
        seasonal_component = np.sin(np.arange(days) * 2 * np.pi / 365) * 0.1
        noise = np.random.normal(0, 0.05, days)
        
        prices = price_trend * (1 + seasonal_component + noise)
        prices = np.maximum(prices, base_price * 0.5)  # Ensure prices don't go too low
        
        # Generate corresponding demand (inversely correlated with price)
        base_demand = random.randint(50, 200)
        elasticity = -1.5
        
        demand = []
        for i, price in enumerate(prices):
            if i == 0:
                demand.append(base_demand)
            else:
                price_change = (price - prices[i-1]) / prices[i-1]
                demand_change = price_change * elasticity
                new_demand = demand[i-1] * (1 + demand_change)
                # Add randomness and ensure demand stays positive
                new_demand *= random.uniform(0.8, 1.2)
                demand.append(max(1, int(new_demand)))
        
        # Calculate revenue and other metrics
        revenue = [p * d for p, d in zip(prices, demand)]
        
        # Generate competitor prices (correlated with our prices but with variation)
        competitor_price = prices * random.uniform(0.9, 1.1) + np.random.normal(0, 2, days)
        
        df = pd.DataFrame({
            'date': dates,
            'product_id': product_id,
            'price': np.round(prices, 2),
            'demand': demand,
            'revenue': np.round(revenue, 2),
            'competitor_avg_price': np.round(competitor_price, 2),
            'inventory_level': np.random.randint(50, 500, days),
            'conversion_rate': np.random.uniform(0.02, 0.08, days),
            'margin': np.random.uniform(0.15, 0.45, days)
        })
        
        return df
    
    def generate_customer_segments(self) -> List[Dict]:
        """Generate customer segment data"""
        segments = [
            {
                'segment_name': 'Price Sensitive',
                'size_percentage': 35,
                'price_elasticity': -2.1,
                'avg_order_value': 45,
                'conversion_rate': 0.045,
                'characteristics': ['High price sensitivity', 'Deal seekers', 'Compare prices']
            },
            {
                'segment_name': 'Value Seekers',
                'size_percentage': 28,
                'price_elasticity': -1.3,
                'avg_order_value': 78,
                'conversion_rate': 0.062,
                'characteristics': ['Quality conscious', 'Research before buying', 'Brand loyal']
            },
            {
                'segment_name': 'Premium Buyers',
                'size_percentage': 18,
                'price_elasticity': -0.6,
                'avg_order_value': 156,
                'conversion_rate': 0.089,
                'characteristics': ['Premium products', 'Low price sensitivity', 'Brand conscious']
            },
            {
                'segment_name': 'Convenience Focused',
                'size_percentage': 19,
                'price_elasticity': -1.0,
                'avg_order_value': 62,
                'conversion_rate': 0.071,
                'characteristics': ['Fast shipping', 'Easy returns', 'Time-sensitive']
            }
        ]
        
        return segments
    
    def generate_ab_test_data(self, test_name: str, days: int = 30) -> Dict[str, Any]:
        """Generate A/B test performance data"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Control group (Strategy A)
        control_conversion = np.random.normal(0.045, 0.008, days)
        control_revenue = np.random.normal(2500, 400, days)
        control_aov = control_revenue / (control_conversion * 1000)  # Assuming 1000 visitors per day
        
        # Test group (Strategy B) - slightly better performance
        test_conversion = np.random.normal(0.051, 0.009, days)
        test_revenue = np.random.normal(2680, 420, days)
        test_aov = test_revenue / (test_conversion * 1000)
        
        # Calculate statistical significance
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(control_conversion, test_conversion)
        
        return {
            'test_name': test_name,
            'start_date': dates[0],
            'end_date': dates[-1],
            'control_data': {
                'conversion_rate': np.mean(control_conversion),
                'revenue': np.sum(control_revenue),
                'avg_order_value': np.mean(control_aov)
            },
            'test_data': {
                'conversion_rate': np.mean(test_conversion),
                'revenue': np.sum(test_revenue),
                'avg_order_value': np.mean(test_aov)
            },
            'statistical_significance': {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            },
            'daily_data': pd.DataFrame({
                'date': dates,
                'control_conversion': control_conversion,
                'control_revenue': control_revenue,
                'test_conversion': test_conversion,
                'test_revenue': test_revenue
            })
        }
    
    def _generate_product_name(self, category: str) -> str:
        """Generate realistic product names by category"""
        prefixes = {
            'Electronics': ['Smart', 'Pro', 'Ultra', 'Digital', 'Wireless', 'HD', '4K'],
            'Clothing': ['Premium', 'Comfort', 'Classic', 'Modern', 'Luxury', 'Casual'],
            'Home & Garden': ['Deluxe', 'Professional', 'Eco', 'Durable', 'Premium'],
            'Sports': ['Pro', 'Athletic', 'Performance', 'Elite', 'Training'],
            'Books': ['Complete', 'Essential', 'Advanced', 'Beginner\'s', 'Ultimate']
        }
        
        products = {
            'Electronics': ['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Speaker', 'Camera', 'Monitor'],
            'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Dress', 'Shoes', 'Sweater', 'Shorts'],
            'Home & Garden': ['Vacuum', 'Blender', 'Chair', 'Lamp', 'Plant Pot', 'Tool Set'],
            'Sports': ['Running Shoes', 'Yoga Mat', 'Dumbbells', 'Tennis Racket', 'Basketball'],
            'Books': ['Guide to Python', 'Marketing Handbook', 'Cookbook', 'Mystery Novel', 'Biography']
        }
        
        prefix = random.choice(prefixes[category])
        product = random.choice(products[category])
        
        return f"{prefix} {product}"
    
    def _generate_sales_history(self, current_price: float, base_demand: int) -> List[Dict]:
        """Generate historical sales data"""
        history = []
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        for date in dates:
            # Add some price variation over time
            price_variation = random.uniform(0.9, 1.1)
            price = round(current_price * price_variation, 2)
            
            # Demand inversely correlated with price
            demand_variation = random.uniform(0.7, 1.3)
            demand = int(base_demand * demand_variation * (current_price / price))
            
            history.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': price,
                'units_sold': demand,
                'revenue': round(price * demand, 2)
            })
        
        return history
    
    def _get_seasonal_factor(self, category: str, month: int) -> float:
        """Get seasonal multiplier for category and month"""
        seasonality = self.categories[category]['seasonality']
        return seasonality.get(str(month), 1.0)
    
    def _generate_market_events(self) -> List[Dict]:
        """Generate random market events that could affect pricing"""
        events = [
            {'event': 'Black Friday Sale', 'impact': 'High demand, competitive pricing'},
            {'event': 'Supply Chain Disruption', 'impact': 'Increased costs, potential price increases'},
            {'event': 'New Competitor Launch', 'impact': 'Price pressure, market share risk'},
            {'event': 'Economic Uncertainty', 'impact': 'Reduced consumer spending'},
            {'event': 'Seasonal Trend Start', 'impact': 'Category-specific demand changes'},
            {'event': 'Inventory Clearance', 'impact': 'Aggressive discounting opportunity'}
        ]
        
        # Return 1-3 random events
        return random.sample(events, random.randint(1, 3))
