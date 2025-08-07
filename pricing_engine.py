import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import math

class DynamicPricingEngine:
    def __init__(self):
        self.pricing_strategies = {
            'Revenue Maximization': self._revenue_maximization,
            'Profit Maximization': self._profit_maximization,
            'Market Share': self._market_share_strategy,
            'Competitive': self._competitive_strategy
        }
        
        self.base_elasticity = -1.5  # Base price elasticity of demand
        
    def calculate_optimal_prices(self, products_data: List[Dict], market_data: Dict, 
                               strategy: str, min_margin: float, max_discount: float) -> List[Dict]:
        """Calculate optimal prices for all products based on strategy and constraints"""
        
        recommendations = []
        
        for product in products_data:
            # Calculate base pricing factors
            demand_factor = self._calculate_demand_factor(product, market_data)
            inventory_factor = self._calculate_inventory_factor(product)
            competitive_factor = self._calculate_competitive_factor(product, market_data)
            seasonal_factor = self._calculate_seasonal_factor(product)
            
            # Apply pricing strategy
            strategy_multiplier = self.pricing_strategies[strategy](
                product, demand_factor, inventory_factor, competitive_factor, seasonal_factor
            )
            
            # Calculate recommended price
            base_price = product['current_price']
            recommended_price = base_price * strategy_multiplier
            
            # Apply constraints
            min_price = product['cost'] * (1 + min_margin / 100)
            max_price_with_discount = base_price * (1 - max_discount / 100)
            
            recommended_price = max(min_price, min(recommended_price, base_price * 1.5))
            recommended_price = max(recommended_price, max_price_with_discount)
            
            # Calculate metrics
            price_change = ((recommended_price - base_price) / base_price) * 100
            profit_margin = ((recommended_price - product['cost']) / recommended_price) * 100
            
            # Demand forecast using price elasticity
            demand_forecast = self._forecast_demand(product, recommended_price)
            
            # Competitive score
            competitive_score = self._calculate_competitive_score(
                recommended_price, product, market_data
            )
            
            recommendations.append({
                'product_name': product['product_name'],
                'current_price': base_price,
                'recommended_price': recommended_price,
                'price_change': price_change,
                'profit_margin': profit_margin,
                'demand_forecast': demand_forecast,
                'inventory_level': product['inventory_level'],
                'competitive_score': competitive_score,
                'strategy_factors': {
                    'demand': demand_factor,
                    'inventory': inventory_factor,
                    'competitive': competitive_factor,
                    'seasonal': seasonal_factor
                }
            })
        
        return recommendations
    
    def _calculate_demand_factor(self, product: Dict, market_data: Dict) -> float:
        """Calculate demand-based pricing factor"""
        historical_demand = product.get('historical_demand', 100)
        current_trend = market_data.get('demand_trend', 1.0)
        
        # Normalize demand (higher demand = higher prices)
        demand_factor = (historical_demand / 100) * current_trend
        return np.clip(demand_factor, 0.5, 2.0)
    
    def _calculate_inventory_factor(self, product: Dict) -> float:
        """Calculate inventory-based pricing factor"""
        inventory_level = product['inventory_level']
        optimal_inventory = product.get('optimal_inventory', 200)
        
        # Low inventory = higher prices, high inventory = lower prices
        inventory_ratio = inventory_level / optimal_inventory
        
        if inventory_ratio < 0.2:  # Very low inventory
            return 1.3
        elif inventory_ratio < 0.5:  # Low inventory
            return 1.1
        elif inventory_ratio > 2.0:  # High inventory
            return 0.8
        else:  # Normal inventory
            return 1.0
    
    def _calculate_competitive_factor(self, product: Dict, market_data: Dict) -> float:
        """Calculate competition-based pricing factor"""
        our_price = product['current_price']
        competitor_prices = market_data.get('competitor_prices', {})
        
        if not competitor_prices:
            return 1.0
        
        avg_competitor_price = np.mean(list(competitor_prices.values()))
        
        # If we're cheaper, we can potentially increase price
        # If we're more expensive, we might need to decrease
        price_ratio = our_price / avg_competitor_price
        
        if price_ratio < 0.9:  # We're significantly cheaper
            return 1.1
        elif price_ratio > 1.1:  # We're significantly more expensive
            return 0.9
        else:  # We're competitively priced
            return 1.0
    
    def _calculate_seasonal_factor(self, product: Dict) -> float:
        """Calculate seasonal pricing factor"""
        current_month = datetime.now().month
        category = product.get('category', 'general')
        
        # Seasonal factors by category and month
        seasonal_patterns = {
            'Electronics': {11: 1.2, 12: 1.3, 1: 0.9},  # Black Friday, Christmas, post-holiday
            'Clothing': {3: 1.1, 6: 1.1, 9: 1.1, 12: 1.2},  # Season changes
            'Home & Garden': {3: 1.2, 4: 1.3, 5: 1.2},  # Spring season
            'Sports': {1: 1.1, 5: 1.2, 6: 1.3},  # New Year, summer season
            'Books': {9: 1.2, 1: 1.1}  # Back to school, New Year
        }
        
        return seasonal_patterns.get(category, {}).get(current_month, 1.0)
    
    def _revenue_maximization(self, product: Dict, demand_factor: float, 
                            inventory_factor: float, competitive_factor: float, 
                            seasonal_factor: float) -> float:
        """Revenue maximization strategy"""
        # Weighted combination favoring demand and competitive factors
        multiplier = (
            demand_factor * 0.4 +
            competitive_factor * 0.3 +
            seasonal_factor * 0.2 +
            inventory_factor * 0.1
        )
        return np.clip(multiplier, 0.8, 1.4)
    
    def _profit_maximization(self, product: Dict, demand_factor: float, 
                           inventory_factor: float, competitive_factor: float, 
                           seasonal_factor: float) -> float:
        """Profit maximization strategy"""
        # Focus on maintaining margins while considering demand
        cost_ratio = product['cost'] / product['current_price']
        
        # Higher margins for low-cost ratio products
        margin_factor = 1.2 - cost_ratio
        
        multiplier = (
            demand_factor * 0.3 +
            margin_factor * 0.4 +
            inventory_factor * 0.2 +
            seasonal_factor * 0.1
        )
        return np.clip(multiplier, 0.9, 1.3)
    
    def _market_share_strategy(self, product: Dict, demand_factor: float, 
                             inventory_factor: float, competitive_factor: float, 
                             seasonal_factor: float) -> float:
        """Market share growth strategy (competitive pricing)"""
        # Aggressive pricing to gain market share
        multiplier = (
            competitive_factor * 0.5 +
            demand_factor * 0.2 +
            inventory_factor * 0.2 +
            seasonal_factor * 0.1
        )
        # Bias towards lower prices
        return np.clip(multiplier * 0.95, 0.7, 1.2)
    
    def _competitive_strategy(self, product: Dict, demand_factor: float, 
                            inventory_factor: float, competitive_factor: float, 
                            seasonal_factor: float) -> float:
        """Competitive matching strategy"""
        # Match competitor pricing with slight adjustments
        multiplier = (
            competitive_factor * 0.6 +
            demand_factor * 0.2 +
            seasonal_factor * 0.1 +
            inventory_factor * 0.1
        )
        return np.clip(multiplier, 0.9, 1.1)
    
    def _forecast_demand(self, product: Dict, new_price: float) -> int:
        """Forecast demand based on price elasticity"""
        base_demand = product.get('historical_demand', 100)
        current_price = product['current_price']
        
        # Price elasticity calculation
        price_change_ratio = new_price / current_price
        demand_change_ratio = price_change_ratio ** self.base_elasticity
        
        forecasted_demand = base_demand * demand_change_ratio
        
        # Add some randomness and seasonality
        seasonal_adjustment = np.random.normal(1.0, 0.1)
        forecasted_demand *= seasonal_adjustment
        
        return max(1, int(forecasted_demand))
    
    def _calculate_competitive_score(self, our_price: float, product: Dict, 
                                   market_data: Dict) -> float:
        """Calculate competitive positioning score (1-10)"""
        competitor_prices = market_data.get('competitor_prices', {})
        
        if not competitor_prices:
            return 7.0  # Neutral score if no competitor data
        
        prices = list(competitor_prices.values()) + [our_price]
        our_rank = sorted(prices).index(our_price) + 1
        total_competitors = len(prices)
        
        # Score: 10 = cheapest, 1 = most expensive
        score = 11 - (our_rank / total_competitors * 10)
        
        # Adjust for value proposition
        product_quality = product.get('quality_score', 7)  # 1-10 scale
        if score < 5 and product_quality > 8:  # High price but high quality
            score += 2
        
        return np.clip(score, 1, 10)
    
    def calculate_price_elasticity(self, product: Dict, price_history: List[Tuple[float, int]]) -> float:
        """Calculate price elasticity from historical data"""
        if len(price_history) < 2:
            return self.base_elasticity
        
        prices = [p[0] for p in price_history]
        demands = [p[1] for p in price_history]
        
        # Calculate percentage changes
        price_changes = []
        demand_changes = []
        
        for i in range(1, len(prices)):
            price_pct_change = (prices[i] - prices[i-1]) / prices[i-1]
            demand_pct_change = (demands[i] - demands[i-1]) / demands[i-1]
            
            if price_pct_change != 0:  # Avoid division by zero
                price_changes.append(price_pct_change)
                demand_changes.append(demand_pct_change)
        
        if not price_changes:
            return self.base_elasticity
        
        # Calculate elasticity as ratio of demand change to price change
        elasticities = [d/p for p, d in zip(price_changes, demand_changes) if p != 0]
        
        if elasticities:
            return float(np.mean(elasticities))
        else:
            return self.base_elasticity
    
    def optimize_inventory_pricing(self, product: Dict, target_inventory_days: int = 30) -> float:
        """Optimize pricing to achieve target inventory turnover"""
        current_inventory = product['inventory_level']
        daily_demand = product.get('daily_demand', 10)
        current_price = product['current_price']
        
        # Calculate days of inventory at current demand
        current_inventory_days = current_inventory / daily_demand if daily_demand > 0 else float('inf')
        
        # If we have too much inventory, decrease price to increase demand
        # If we have too little inventory, increase price to decrease demand
        
        if current_inventory_days > target_inventory_days * 1.5:  # Too much inventory
            # Aggressive price reduction
            price_reduction = min(0.3, (current_inventory_days - target_inventory_days) / target_inventory_days * 0.2)
            return current_price * (1 - price_reduction)
        
        elif current_inventory_days < target_inventory_days * 0.5:  # Too little inventory
            # Price increase to slow demand
            price_increase = min(0.2, (target_inventory_days - current_inventory_days) / target_inventory_days * 0.15)
            return current_price * (1 + price_increase)
        
        else:  # Inventory is within acceptable range
            return current_price
