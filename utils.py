import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Union, Tuple
import re

def format_currency(amount: float, currency: str = '$') -> str:
    """Format number as currency string"""
    if amount >= 1_000_000:
        return f"{currency}{amount/1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"{currency}{amount/1_000:.1f}K"
    else:
        return f"{currency}{amount:.2f}"

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0 if new_value == 0 else float('inf')
    return ((new_value - old_value) / old_value) * 100

def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format number as percentage string"""
    return f"{value:.{decimal_places}f}%"

def calculate_elasticity(price_changes: List[float], demand_changes: List[float]) -> float:
    """Calculate price elasticity of demand"""
    if len(price_changes) != len(demand_changes) or len(price_changes) == 0:
        return -1.5  # Default elasticity
    
    elasticities = []
    for p_change, d_change in zip(price_changes, demand_changes):
        if p_change != 0:
            elasticity = d_change / p_change
            if -5 <= elasticity <= 0:  # Filter reasonable values
                elasticities.append(elasticity)
    
    return float(np.mean(elasticities)) if elasticities else -1.5

def validate_price_constraints(price: float, min_price: float, max_price: float) -> float:
    """Validate and constrain price within bounds"""
    return max(min_price, min(price, max_price))

def calculate_profit_margin(price: float, cost: float) -> float:
    """Calculate profit margin percentage"""
    if price <= 0:
        return 0.0
    return ((price - cost) / price) * 100

def generate_color_palette(n_colors: int) -> List[str]:
    """Generate a color palette for visualizations"""
    base_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
        '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43'
    ]
    
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]
    
    # Generate additional colors if needed
    colors = base_colors.copy()
    while len(colors) < n_colors:
        # Generate random hex colors
        import random
        color = f"#{random.randint(0, 0xFFFFFF):06x}"
        colors.append(color)
    
    return colors[:n_colors]

def clean_product_name(name: str) -> str:
    """Clean and standardize product names"""
    # Remove special characters and extra spaces
    cleaned = re.sub(r'[^\w\s-]', '', name)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def calculate_seasonal_factor(month: int, category: str) -> float:
    """Calculate seasonal pricing factor"""
    seasonal_patterns = {
        'Electronics': {11: 1.3, 12: 1.4, 1: 0.8, 2: 0.9},
        'Clothing': {3: 1.2, 6: 1.1, 9: 1.3, 12: 1.2},
        'Home & Garden': {3: 1.3, 4: 1.4, 5: 1.3, 10: 1.1},
        'Sports': {1: 1.2, 5: 1.3, 6: 1.4, 12: 1.1},
        'Books': {9: 1.3, 1: 1.2, 8: 1.1}
    }
    
    return seasonal_patterns.get(category, {}).get(month, 1.0)

def detect_outliers(data: List[float], method: str = 'iqr') -> List[bool]:
    """Detect outliers in data using specified method"""
    if not data:
        return []
    
    data_array = np.array(data)
    
    if method == 'iqr':
        Q1 = np.percentile(data_array, 25)
        Q3 = np.percentile(data_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return [(x < lower_bound or x > upper_bound) for x in data_array]
    
    elif method == 'zscore':
        mean = np.mean(data_array)
        std = np.std(data_array)
        z_scores = np.abs((data_array - mean) / std)
        return [z > 3 for z in z_scores]
    
    else:
        return [False] * len(data)

def smooth_price_changes(current_price: float, target_price: float, 
                        max_change_pct: float = 10.0) -> float:
    """Smooth price changes to avoid large jumps"""
    max_change = current_price * (max_change_pct / 100)
    price_diff = target_price - current_price
    
    if abs(price_diff) <= max_change:
        return target_price
    
    # Limit the change to max_change
    if price_diff > 0:
        return current_price + max_change
    else:
        return current_price - max_change

def calculate_inventory_days(inventory_level: int, daily_demand: float) -> float:
    """Calculate days of inventory remaining"""
    if daily_demand <= 0:
        return float('inf')
    return inventory_level / daily_demand

def generate_price_recommendations_summary(recommendations: List[Dict]) -> Dict[str, Any]:
    """Generate summary statistics for price recommendations"""
    if not recommendations:
        return {}
    
    price_changes = [r['price_change'] for r in recommendations]
    margins = [r['profit_margin'] for r in recommendations]
    revenues = [r['recommended_price'] * r['demand_forecast'] for r in recommendations]
    
    return {
        'total_products': len(recommendations),
        'price_increases': sum(1 for p in price_changes if p > 0),
        'price_decreases': sum(1 for p in price_changes if p < 0),
        'no_changes': sum(1 for p in price_changes if abs(p) < 0.01),
        'avg_price_change': np.mean(price_changes),
        'avg_margin': np.mean(margins),
        'total_projected_revenue': sum(revenues),
        'max_price_increase': max(price_changes) if price_changes else 0,
        'max_price_decrease': min(price_changes) if price_changes else 0
    }

def validate_pricing_rules(price: float, cost: float, min_margin: float, 
                          max_discount: float, original_price: float) -> Dict[str, bool]:
    """Validate pricing against business rules"""
    
    # Calculate current margin
    current_margin = calculate_profit_margin(price, cost)
    
    # Calculate discount from original price
    discount_pct = ((original_price - price) / original_price) * 100 if original_price > 0 else 0
    
    return {
        'meets_min_margin': current_margin >= min_margin,
        'within_max_discount': discount_pct <= max_discount,
        'price_above_cost': price > cost,
        'reasonable_price': 0.5 * original_price <= price <= 2.0 * original_price
    }

def calculate_competitive_index(our_price: float, competitor_prices: List[float]) -> float:
    """Calculate competitive positioning index (0-100)"""
    if not competitor_prices:
        return 50.0  # Neutral if no competitor data
    
    # Sort prices to find our position
    all_prices = competitor_prices + [our_price]
    sorted_prices = sorted(all_prices)
    
    our_rank = sorted_prices.index(our_price) + 1
    total_prices = len(sorted_prices)
    
    # Convert to 0-100 scale (100 = cheapest, 0 = most expensive)
    competitive_index = ((total_prices - our_rank) / (total_prices - 1)) * 100
    
    return competitive_index

def format_duration(seconds: int) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"

def calculate_revenue_impact(current_prices: List[float], recommended_prices: List[float], 
                           demands: List[int]) -> Dict[str, float]:
    """Calculate revenue impact of price recommendations"""
    
    current_revenue = sum(p * d for p, d in zip(current_prices, demands))
    projected_revenue = sum(p * d for p, d in zip(recommended_prices, demands))
    
    revenue_change = projected_revenue - current_revenue
    revenue_change_pct = (revenue_change / current_revenue) * 100 if current_revenue > 0 else 0
    
    return {
        'current_revenue': current_revenue,
        'projected_revenue': projected_revenue,
        'revenue_change': revenue_change,
        'revenue_change_pct': revenue_change_pct
    }

def get_price_trend(price_history: List[Tuple[str, float]], days: int = 30) -> str:
    """Determine price trend from historical data"""
    if len(price_history) < 2:
        return "Insufficient data"
    
    # Get recent prices
    recent_prices = [p[1] for p in price_history[-days:]]
    
    if len(recent_prices) < 2:
        return "Stable"
    
    # Calculate trend using linear regression
    x = np.arange(len(recent_prices))
    slope = np.polyfit(x, recent_prices, 1)[0]
    
    if slope > 0.1:
        return "Increasing"
    elif slope < -0.1:
        return "Decreasing"
    else:
        return "Stable"

class PricingMetrics:
    """Class to calculate and track pricing performance metrics"""
    
    @staticmethod
    def calculate_price_volatility(prices: List[float]) -> float:
        """Calculate price volatility (coefficient of variation)"""
        if len(prices) < 2:
            return 0.0
        
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        return float(std_price / mean_price) if mean_price > 0 else 0.0
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for pricing strategy returns"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = np.array(returns) - risk_free_rate
        return float(np.mean(excess_returns) / np.std(excess_returns)) if np.std(excess_returns) > 0 else 0.0
    
    @staticmethod
    def calculate_hit_ratio(actual_results: List[bool]) -> float:
        """Calculate hit ratio (percentage of successful price changes)"""
        if not actual_results:
            return 0.0
        
        return sum(actual_results) / len(actual_results)
