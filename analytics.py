import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import math

class PricingAnalytics:
    def __init__(self):
        self.performance_metrics = {}
        self.elasticity_cache = {}
    
    def generate_analytics_data(self) -> Dict[str, Any]:
        """Generate comprehensive analytics data"""
        return {
            'revenue_metrics': self._generate_revenue_metrics(),
            'pricing_performance': self._generate_pricing_performance(),
            'competitive_analysis': self._generate_competitive_analysis(),
            'customer_behavior': self._generate_customer_behavior(),
            'inventory_insights': self._generate_inventory_insights(),
            'market_trends': self._generate_market_trends()
        }
    
    def calculate_price_elasticity(self, price_history: List[Tuple[float, int]], 
                                 product_name: str = '') -> float:
        """Calculate price elasticity of demand from historical data"""
        if len(price_history) < 2:
            return -1.5  # Default elasticity
        
        # Cache check
        cache_key = f"{product_name}_{len(price_history)}" if product_name else str(len(price_history))
        if cache_key in self.elasticity_cache:
            return self.elasticity_cache[cache_key]
        
        prices = [p[0] for p in price_history]
        demands = [p[1] for p in price_history]
        
        # Calculate percentage changes
        price_changes = []
        demand_changes = []
        
        for i in range(1, len(prices)):
            if prices[i-1] != 0 and demands[i-1] != 0:
                price_pct_change = (prices[i] - prices[i-1]) / prices[i-1]
                demand_pct_change = (demands[i] - demands[i-1]) / demands[i-1]
                
                if abs(price_pct_change) > 0.001:  # Avoid very small changes
                    price_changes.append(price_pct_change)
                    demand_changes.append(demand_pct_change)
        
        if not price_changes:
            return -1.5
        
        # Calculate elasticity as average ratio of demand change to price change
        elasticities = []
        for i in range(len(price_changes)):
            if price_changes[i] != 0:
                elasticity = demand_changes[i] / price_changes[i]
                # Filter out extreme values
                if -5 <= elasticity <= 0:
                    elasticities.append(elasticity)
        
        if elasticities:
            result = float(np.mean(elasticities))
            self.elasticity_cache[cache_key] = result
            return result
        else:
            return -1.5
    
    def analyze_pricing_performance(self, current_data: List[Dict], 
                                  historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze pricing strategy performance"""
        
        # Revenue impact analysis
        current_revenue = sum(p['recommended_price'] * p['demand_forecast'] for p in current_data)
        baseline_revenue = sum(p['current_price'] * p['demand_forecast'] for p in current_data)
        revenue_lift = ((current_revenue - baseline_revenue) / baseline_revenue) * 100
        
        # Price change distribution
        price_changes = [p['price_change'] for p in current_data]
        
        # Margin analysis
        margins = [p['profit_margin'] for p in current_data]
        
        # Competitive positioning
        competitive_scores = [p['competitive_score'] for p in current_data]
        
        return {
            'revenue_impact': {
                'current_revenue': current_revenue,
                'baseline_revenue': baseline_revenue,
                'revenue_lift_pct': revenue_lift,
                'revenue_lift_amount': current_revenue - baseline_revenue
            },
            'price_changes': {
                'avg_change': np.mean(price_changes),
                'median_change': np.median(price_changes),
                'std_change': np.std(price_changes),
                'increases_count': sum(1 for c in price_changes if c > 0),
                'decreases_count': sum(1 for c in price_changes if c < 0),
                'no_change_count': sum(1 for c in price_changes if abs(c) < 0.01)
            },
            'margin_analysis': {
                'avg_margin': np.mean(margins),
                'min_margin': np.min(margins),
                'max_margin': np.max(margins),
                'margin_distribution': np.histogram(margins, bins=10)
            },
            'competitive_position': {
                'avg_score': np.mean(competitive_scores),
                'competitive_advantage': sum(1 for s in competitive_scores if s > 7),
                'competitive_disadvantage': sum(1 for s in competitive_scores if s < 4)
            }
        }
    
    def generate_price_optimization_report(self, products_data: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive price optimization report"""
        
        # Identify optimization opportunities
        opportunities = []
        
        for product in products_data:
            # High inventory, low price increase opportunity
            if (product['inventory_level'] > product.get('optimal_inventory', 200) * 1.5 and 
                product['price_change'] > -10):
                opportunities.append({
                    'product': product['product_name'],
                    'type': 'inventory_clearance',
                    'current_price': product['current_price'],
                    'suggested_action': 'Increase discount for inventory clearance',
                    'potential_impact': 'High'
                })
            
            # Low inventory, price increase opportunity
            elif (product['inventory_level'] < product.get('optimal_inventory', 200) * 0.5 and 
                  product['price_change'] < 15):
                opportunities.append({
                    'product': product['product_name'],
                    'type': 'scarcity_pricing',
                    'current_price': product['current_price'],
                    'suggested_action': 'Implement scarcity pricing',
                    'potential_impact': 'Medium'
                })
            
            # High competitive score, price increase opportunity
            elif product['competitive_score'] > 8 and product['price_change'] < 10:
                opportunities.append({
                    'product': product['product_name'],
                    'type': 'competitive_advantage',
                    'current_price': product['current_price'],
                    'suggested_action': 'Leverage competitive advantage with higher pricing',
                    'potential_impact': 'Medium'
                })
        
        # Calculate portfolio-level metrics
        total_revenue = sum(p['recommended_price'] * p['demand_forecast'] for p in products_data)
        avg_margin = np.mean([p['profit_margin'] for p in products_data])
        
        return {
            'executive_summary': {
                'total_products_analyzed': len(products_data),
                'optimization_opportunities': len(opportunities),
                'projected_total_revenue': total_revenue,
                'average_profit_margin': avg_margin,
                'price_changes_recommended': sum(1 for p in products_data if abs(p['price_change']) > 1)
            },
            'opportunities': opportunities,
            'risk_assessment': self._assess_pricing_risks(products_data),
            'recommendations': self._generate_strategic_recommendations(products_data)
        }
    
    def create_elasticity_analysis_chart(self, products_data: List[Dict]) -> go.Figure:
        """Create price elasticity analysis visualization"""
        
        # Simulate elasticity data
        elasticities = []
        revenues = []
        margins = []
        categories = []
        
        for product in products_data:
            # Simulate elasticity based on category and other factors
            base_elasticity = -1.5
            if 'Electronics' in product.get('category', ''):
                elasticity = base_elasticity * np.random.uniform(0.7, 1.0)
            elif 'Clothing' in product.get('category', ''):
                elasticity = base_elasticity * np.random.uniform(1.1, 1.4)
            else:
                elasticity = base_elasticity * np.random.uniform(0.8, 1.2)
            
            elasticities.append(elasticity)
            revenues.append(product['recommended_price'] * product['demand_forecast'])
            margins.append(product['profit_margin'])
            categories.append(product.get('category', 'Unknown'))
        
        fig = px.scatter(
            x=elasticities,
            y=revenues,
            size=margins,
            color=categories,
            title="Price Elasticity vs Revenue Analysis",
            labels={
                'x': 'Price Elasticity',
                'y': 'Projected Revenue ($)',
                'size': 'Profit Margin (%)',
                'color': 'Category'
            },
            hover_data={'x': ':.2f', 'y': ':$,.0f'}
        )
        
        # Add reference lines
        fig.add_vline(x=-1.0, line_dash="dash", line_color="gray", 
                     annotation_text="Unitary Elasticity")
        
        fig.update_layout(
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_margin_optimization_chart(self, products_data: List[Dict]) -> go.Figure:
        """Create margin optimization analysis chart"""
        
        # Extract data for visualization
        margins = [p['profit_margin'] for p in products_data]
        revenues = [p['recommended_price'] * p['demand_forecast'] for p in products_data]
        price_changes = [p['price_change'] for p in products_data]
        product_names = [p['product_name'] for p in products_data]
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Margin vs Revenue', 'Price Change Distribution', 
                          'Margin Distribution', 'Revenue vs Price Change'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Margin vs Revenue scatter
        fig.add_trace(
            go.Scatter(x=margins, y=revenues, mode='markers',
                      name='Products', text=product_names,
                      hovertemplate='%{text}<br>Margin: %{x:.1f}%<br>Revenue: $%{y:,.0f}'),
            row=1, col=1
        )
        
        # Price change histogram
        fig.add_trace(
            go.Histogram(x=price_changes, nbinsx=20, name='Price Changes'),
            row=1, col=2
        )
        
        # Margin distribution
        fig.add_trace(
            go.Histogram(x=margins, nbinsx=15, name='Margins'),
            row=2, col=1
        )
        
        # Revenue vs Price Change
        fig.add_trace(
            go.Scatter(x=price_changes, y=revenues, mode='markers',
                      name='Revenue vs Change', text=product_names,
                      hovertemplate='%{text}<br>Price Change: %{x:.1f}%<br>Revenue: $%{y:,.0f}'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Margin and Pricing Optimization Analysis")
        
        return fig
    
    def _generate_revenue_metrics(self) -> Dict[str, Any]:
        """Generate revenue-related metrics"""
        # Simulate revenue data
        daily_revenue = np.random.normal(25000, 5000, 30)
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        return {
            'daily_revenue': list(daily_revenue),
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'total_revenue': np.sum(daily_revenue),
            'avg_daily_revenue': np.mean(daily_revenue),
            'revenue_growth': np.random.uniform(5, 15),  # % growth
            'revenue_volatility': np.std(daily_revenue) / np.mean(daily_revenue)
        }
    
    def _generate_pricing_performance(self) -> Dict[str, Any]:
        """Generate pricing performance metrics"""
        return {
            'price_optimization_score': np.random.uniform(75, 95),
            'avg_margin_improvement': np.random.uniform(2, 8),
            'successful_price_changes': np.random.randint(15, 25),
            'total_price_changes': np.random.randint(30, 40),
            'revenue_per_price_change': np.random.uniform(500, 2000)
        }
    
    def _generate_competitive_analysis(self) -> Dict[str, Any]:
        """Generate competitive analysis data"""
        competitors = ['Competitor A', 'Competitor B', 'Competitor C', 'Competitor D']
        
        return {
            'market_position': np.random.choice(['Leader', 'Challenger', 'Follower']),
            'price_advantage_pct': np.random.uniform(-5, 10),
            'win_rate': np.random.uniform(0.6, 0.8),
            'competitor_response_time': np.random.uniform(2, 8),  # hours
            'competitive_gaps': {
                comp: np.random.uniform(-15, 15) for comp in competitors
            }
        }
    
    def _generate_customer_behavior(self) -> Dict[str, Any]:
        """Generate customer behavior analytics"""
        return {
            'price_sensitivity_score': np.random.uniform(6, 9),
            'conversion_rate_by_price_change': {
                'decrease_10_pct': np.random.uniform(0.06, 0.08),
                'no_change': np.random.uniform(0.045, 0.055),
                'increase_10_pct': np.random.uniform(0.025, 0.035)
            },
            'customer_acquisition_cost': np.random.uniform(25, 45),
            'customer_lifetime_value': np.random.uniform(200, 500),
            'repeat_purchase_rate': np.random.uniform(0.3, 0.7)
        }
    
    def _generate_inventory_insights(self) -> Dict[str, Any]:
        """Generate inventory-related insights"""
        return {
            'inventory_turnover': np.random.uniform(4, 12),
            'stockout_risk_products': np.random.randint(5, 15),
            'overstock_products': np.random.randint(8, 20),
            'optimal_inventory_achievement': np.random.uniform(75, 90),
            'inventory_carrying_cost_pct': np.random.uniform(15, 25)
        }
    
    def _generate_market_trends(self) -> Dict[str, Any]:
        """Generate market trend data"""
        return {
            'market_growth_rate': np.random.uniform(3, 12),
            'seasonal_factor': np.random.uniform(0.8, 1.3),
            'economic_impact_score': np.random.uniform(-2, 2),
            'demand_volatility': np.random.uniform(0.1, 0.4),
            'new_entrants_threat': np.random.choice(['Low', 'Medium', 'High'])
        }
    
    def _assess_pricing_risks(self, products_data: List[Dict]) -> Dict[str, Any]:
        """Assess risks associated with pricing strategies"""
        
        high_risk_products = []
        medium_risk_products = []
        
        for product in products_data:
            risk_score = 0
            risk_factors = []
            
            # Large price increases are risky
            if product['price_change'] > 20:
                risk_score += 3
                risk_factors.append('Large price increase')
            
            # Low inventory with price increases
            if (product['inventory_level'] < 50 and product['price_change'] > 0):
                risk_score += 2
                risk_factors.append('Low inventory + price increase')
            
            # Low competitive score with price increases
            if (product['competitive_score'] < 5 and product['price_change'] > 0):
                risk_score += 2
                risk_factors.append('Poor competitive position')
            
            # Very low margins
            if product['profit_margin'] < 10:
                risk_score += 1
                risk_factors.append('Low profit margin')
            
            if risk_score >= 4:
                high_risk_products.append({
                    'product': product['product_name'],
                    'risk_score': risk_score,
                    'risk_factors': risk_factors
                })
            elif risk_score >= 2:
                medium_risk_products.append({
                    'product': product['product_name'],
                    'risk_score': risk_score,
                    'risk_factors': risk_factors
                })
        
        return {
            'overall_risk_level': 'High' if len(high_risk_products) > 5 else 'Medium' if len(medium_risk_products) > 10 else 'Low',
            'high_risk_products': high_risk_products,
            'medium_risk_products': medium_risk_products,
            'risk_mitigation_suggestions': [
                'Implement gradual price changes for high-risk products',
                'Monitor competitor responses closely',
                'Set up automatic rollback triggers',
                'Increase marketing for repriced products'
            ]
        }
    
    def _generate_strategic_recommendations(self, products_data: List[Dict]) -> List[str]:
        """Generate strategic pricing recommendations"""
        
        recommendations = []
        
        # Analyze overall patterns
        avg_margin = np.mean([p['profit_margin'] for p in products_data])
        avg_competitive_score = np.mean([p['competitive_score'] for p in products_data])
        price_increase_count = sum(1 for p in products_data if p['price_change'] > 0)
        
        if avg_margin < 20:
            recommendations.append(
                "Consider focusing on higher-margin products and gradually phase out low-margin items"
            )
        
        if avg_competitive_score < 6:
            recommendations.append(
                "Improve competitive positioning through value-added services or product differentiation"
            )
        
        if price_increase_count > len(products_data) * 0.7:
            recommendations.append(
                "High number of price increases detected - monitor customer response carefully"
            )
        
        # Category-specific recommendations
        categories = set(p.get('category', 'Unknown') for p in products_data)
        for category in categories:
            category_products = [p for p in products_data if p.get('category') == category]
            if len(category_products) > 5:  # Only analyze if sufficient products
                category_margin = np.mean([p['profit_margin'] for p in category_products])
                if category_margin < 15:
                    recommendations.append(
                        f"Consider repricing strategy for {category} category - margins below target"
                    )
        
        return recommendations[:5]  # Return top 5 recommendations
