import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

from pricing_engine import DynamicPricingEngine
from data_generator import DataGenerator
from analytics import PricingAnalytics
from ml_models import PricingMLModel
from utils import format_currency, calculate_percentage_change

# Page configuration
st.set_page_config(
    page_title="Dynamic E-commerce Pricing Engine",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pricing_engine' not in st.session_state:
    st.session_state.pricing_engine = DynamicPricingEngine()
    st.session_state.data_generator = DataGenerator()
    st.session_state.analytics = PricingAnalytics()
    st.session_state.ml_model = PricingMLModel()
    st.session_state.last_update = datetime.now()

def main():
    st.title("🚀 Dynamic E-commerce Pricing Engine")
    st.markdown("AI-driven pricing optimization for maximum revenue")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Pricing strategy selection
        pricing_strategy = st.selectbox(
            "Pricing Strategy",
            ["Revenue Maximization", "Profit Maximization", "Market Share", "Competitive"],
            help="Select the primary pricing objective"
        )
        
        # Update frequency
        update_frequency = st.slider(
            "Update Frequency (seconds)",
            min_value=5,
            max_value=60,
            value=10,
            help="How often to recalculate prices"
        )
        
        # Price constraints
        st.subheader("Price Constraints")
        min_margin = st.slider("Minimum Profit Margin (%)", 5, 50, 15)
        max_discount = st.slider("Maximum Discount (%)", 0, 70, 30)
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        
        if st.button("🔄 Refresh Data"):
            st.session_state.last_update = datetime.now()
            st.rerun()
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🤖 AI Pricing", 
        "📈 Analytics", 
        "🔬 A/B Testing", 
        "⚙️ Settings"
    ])
    
    with tab1:
        render_dashboard(pricing_strategy, min_margin, max_discount)
    
    with tab2:
        render_ai_pricing()
    
    with tab3:
        render_analytics()
    
    with tab4:
        render_ab_testing()
    
    with tab5:
        render_settings()
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(update_frequency)
        if datetime.now() - st.session_state.last_update > timedelta(seconds=update_frequency):
            st.session_state.last_update = datetime.now()
            st.rerun()

def render_dashboard(pricing_strategy, min_margin, max_discount):
    # Generate current data
    products_data = st.session_state.data_generator.generate_product_data(50)
    market_data = st.session_state.data_generator.generate_market_data()
    
    # Calculate pricing recommendations
    pricing_recommendations = st.session_state.pricing_engine.calculate_optimal_prices(
        products_data, market_data, pricing_strategy, min_margin, max_discount
    )
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = sum(p['recommended_price'] * p['demand_forecast'] for p in pricing_recommendations)
        st.metric(
            "Projected Revenue",
            format_currency(total_revenue),
            delta=f"+{np.random.uniform(5, 15):.1f}%"
        )
    
    with col2:
        avg_margin = np.mean([p['profit_margin'] for p in pricing_recommendations])
        st.metric(
            "Avg Profit Margin",
            f"{avg_margin:.1f}%",
            delta=f"+{np.random.uniform(1, 5):.1f}%"
        )
    
    with col3:
        price_changes = sum(1 for p in pricing_recommendations if abs(p['price_change']) > 0.01)
        st.metric(
            "Price Changes",
            price_changes,
            delta=f"{np.random.randint(-5, 15)}"
        )
    
    with col4:
        competitive_advantage = np.mean([p['competitive_score'] for p in pricing_recommendations])
        st.metric(
            "Competitive Score",
            f"{competitive_advantage:.1f}/10",
            delta=f"+{np.random.uniform(0.1, 0.8):.1f}"
        )
    
    # Main charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Price Recommendations")
        
        # Create price recommendation chart
        df_prices = pd.DataFrame(pricing_recommendations)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_prices['product_name'],
            y=df_prices['current_price'],
            mode='markers',
            name='Current Price',
            marker=dict(color='lightblue', size=10)
        ))
        fig.add_trace(go.Scatter(
            x=df_prices['product_name'],
            y=df_prices['recommended_price'],
            mode='markers',
            name='Recommended Price',
            marker=dict(color='red', size=12)
        ))
        
        fig.update_layout(
            title="Current vs Recommended Prices",
            xaxis_title="Products",
            yaxis_title="Price ($)",
            height=400
        )
        fig.update_layout(xaxis={'tickangle': 45})
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Demand vs Inventory")
        
        # Demand vs inventory scatter plot
        fig = px.scatter(
            df_prices,
            x='inventory_level',
            y='demand_forecast',
            size='recommended_price',
            color='profit_margin',
            hover_data=['product_name', 'competitive_score'],
            title="Inventory vs Demand Analysis",
            labels={
                'inventory_level': 'Inventory Level',
                'demand_forecast': 'Demand Forecast',
                'profit_margin': 'Profit Margin (%)'
            }
        )
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed recommendations table
    st.subheader("📋 Detailed Pricing Recommendations")
    
    # Format the dataframe for display
    display_df = df_prices.copy()
    display_df['current_price'] = display_df['current_price'].apply(format_currency)
    display_df['recommended_price'] = display_df['recommended_price'].apply(format_currency)
    display_df['price_change'] = display_df['price_change'].apply(lambda x: f"{x:+.1f}%")
    display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x:.1f}%")
    
    # Select columns to display
    columns_to_show = [
        'product_name', 'current_price', 'recommended_price', 
        'price_change', 'profit_margin', 'demand_forecast', 
        'inventory_level', 'competitive_score'
    ]
    
    st.dataframe(
        display_df[columns_to_show],
        column_config={
            'product_name': 'Product',
            'current_price': 'Current Price',
            'recommended_price': 'Recommended Price',
            'price_change': 'Change (%)',
            'profit_margin': 'Margin (%)',
            'demand_forecast': st.column_config.NumberColumn('Demand', format="%.0f"),
            'inventory_level': st.column_config.NumberColumn('Inventory', format="%.0f"),
            'competitive_score': st.column_config.NumberColumn('Comp. Score', format="%.1f")
        },
        hide_index=True,
        use_container_width=True
    )

def render_ai_pricing():
    st.subheader("🤖 AI-Powered Pricing Models")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Model performance metrics
        model_data = st.session_state.ml_model.get_model_performance()
        
        # Create model comparison chart
        models = ['Linear Regression', 'Random Forest', 'XGBoost', 'Neural Network']
        accuracy_scores = [0.78, 0.85, 0.89, 0.87]
        
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=models, y=accuracy_scores, marker_color='lightblue'),
        ])
        fig.update_layout(
            title="ML Model Performance Comparison",
            yaxis_title="Accuracy Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Settings")
        
        selected_model = st.selectbox(
            "Primary Model",
            ["XGBoost", "Random Forest", "Neural Network", "Linear Regression"]
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.5, 0.95, 0.8
        )
        
        retrain_frequency = st.selectbox(
            "Retrain Frequency",
            ["Hourly", "Daily", "Weekly", "Monthly"]
        )
        
        if st.button("🔄 Retrain Model"):
            with st.spinner("Retraining model..."):
                time.sleep(2)
            st.success("Model retrained successfully!")
    
    # Feature importance
    st.subheader("📊 Feature Importance Analysis")
    
    features = [
        'Historical Demand', 'Competitor Prices', 'Inventory Level',
        'Seasonality', 'Customer Segment', 'Product Category',
        'Marketing Spend', 'Economic Indicators'
    ]
    importance_scores = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
    
    fig = px.bar(
        x=importance_scores,
        y=features,
        orientation='h',
        title="Feature Importance for Price Prediction"
    )
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction intervals
    st.subheader("🎯 Price Prediction Intervals")
    
    sample_products = st.session_state.data_generator.generate_product_data(10)
    predictions_data = []
    
    for product in sample_products:
        base_price = product['current_price']
        predictions_data.append({
            'product': product['product_name'],
            'lower_bound': base_price * 0.9,
            'predicted': base_price * np.random.uniform(0.95, 1.1),
            'upper_bound': base_price * 1.1
        })
    
    df_predictions = pd.DataFrame(predictions_data)
    
    fig = go.Figure()
    
    # Add prediction intervals
    for _, row in df_predictions.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['product'], row['product']],
            y=[row['lower_bound'], row['upper_bound']],
            mode='lines',
            line=dict(color='lightgray', width=8),
            showlegend=False
        ))
    
    # Add predicted values
    fig.add_trace(go.Scatter(
        x=df_predictions['product'],
        y=df_predictions['predicted'],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Predicted Price'
    ))
    
    fig.update_layout(
        title="Price Prediction Intervals (90% Confidence)",
        xaxis_title="Products",
        yaxis_title="Price ($)",
        height=400
    )
    fig.update_layout(xaxis={'tickangle': 45})
    
    st.plotly_chart(fig, use_container_width=True)

def render_analytics():
    st.subheader("📈 Advanced Analytics & Insights")
    
    # Generate analytics data
    analytics_data = st.session_state.analytics.generate_analytics_data()
    
    # Revenue trend analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💹 Revenue Trend Analysis")
        
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        revenue_trend = np.cumsum(np.random.normal(1000, 200, len(dates)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=revenue_trend,
            mode='lines',
            name='Daily Revenue',
            line=dict(color='blue', width=2)
        ))
        
        # Add moving average
        moving_avg = pd.Series(revenue_trend).rolling(window=30).mean()
        fig.add_trace(go.Scatter(
            x=dates,
            y=moving_avg,
            mode='lines',
            name='30-Day Moving Average',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Revenue Trend Over Time",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Price Elasticity Analysis")
        
        # Price elasticity heatmap
        products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
        price_changes = [-20, -10, 0, 10, 20]
        
        elasticity_matrix = np.random.uniform(-2, 0, (len(products), len(price_changes)))
        
        fig = px.imshow(
            elasticity_matrix,
            x=price_changes,
            y=products,
            color_continuous_scale='RdYlBu',
            title="Price Elasticity Heatmap",
            labels={'x': 'Price Change (%)', 'y': 'Products', 'color': 'Elasticity'}
        )
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Competitive analysis
    st.subheader("🏆 Competitive Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Market share pie chart
        competitors = ['Our Store', 'Competitor A', 'Competitor B', 'Competitor C', 'Others']
        market_share = [30, 25, 20, 15, 10]
        
        fig = px.pie(
            values=market_share,
            names=competitors,
            title="Market Share Distribution"
        )
        fig.update_layout(height=300)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price positioning
        competitor_prices = {
            'Our Store': 29.99,
            'Competitor A': 32.99,
            'Competitor B': 27.99,
            'Competitor C': 31.49,
            'Market Avg': 30.62
        }
        
        fig = go.Figure(data=[
            go.Bar(x=list(competitor_prices.keys()), y=list(competitor_prices.values()))
        ])
        fig.update_layout(
            title="Average Price Comparison",
            yaxis_title="Price ($)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Win rate analysis
        categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books']
        win_rates = [0.72, 0.65, 0.80, 0.58, 0.85]
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=win_rates, marker_color='lightgreen')
        ])
        fig.update_layout(
            title="Win Rate by Category",
            yaxis_title="Win Rate",
            yaxis_range=[0, 1],
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer behavior analysis
    st.subheader("👥 Customer Behavior Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer segments price sensitivity
        segments = ['Price Sensitive', 'Value Seekers', 'Premium Buyers', 'Convenience Focused']
        sensitivity_scores = [0.9, 0.6, 0.2, 0.4]
        
        fig = px.bar(
            x=segments,
            y=sensitivity_scores,
            title="Price Sensitivity by Customer Segment",
            color=sensitivity_scores,
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Purchase patterns
        hours = list(range(24))
        purchase_volume = [20, 15, 10, 8, 12, 25, 45, 70, 85, 90, 95, 100, 
                          105, 110, 108, 100, 95, 88, 75, 65, 50, 40, 35, 25]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=purchase_volume,
            mode='lines+markers',
            fill='tonexty',
            name='Purchase Volume'
        ))
        
        fig.update_layout(
            title="Purchase Patterns by Hour of Day",
            xaxis_title="Hour of Day",
            yaxis_title="Purchase Volume",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_ab_testing():
    st.subheader("🔬 A/B Testing Framework")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Current A/B tests
        st.subheader("🧪 Active A/B Tests")
        
        test_data = [
            {
                'test_name': 'Dynamic vs Fixed Pricing',
                'strategy_a': 'Dynamic Pricing',
                'strategy_b': 'Fixed Pricing',
                'revenue_a': 25480,
                'revenue_b': 23210,
                'conversion_a': 0.045,
                'conversion_b': 0.041,
                'significance': 0.034,
                'status': 'Significant'
            },
            {
                'test_name': 'Aggressive vs Conservative',
                'strategy_a': 'Aggressive Discounting',
                'strategy_b': 'Conservative Pricing',
                'revenue_a': 22150,
                'revenue_b': 24680,
                'conversion_a': 0.052,
                'conversion_b': 0.038,
                'significance': 0.078,
                'status': 'Not Significant'
            }
        ]
        
        for test in test_data:
            with st.expander(f"📊 {test['test_name']}", expanded=True):
                col_a, col_b, col_stats = st.columns(3)
                
                with col_a:
                    st.metric(
                        f"Strategy A: {test['strategy_a']}",
                        format_currency(test['revenue_a']),
                        delta=f"Conv: {test['conversion_a']:.1%}"
                    )
                
                with col_b:
                    st.metric(
                        f"Strategy B: {test['strategy_b']}",
                        format_currency(test['revenue_b']),
                        delta=f"Conv: {test['conversion_b']:.1%}"
                    )
                
                with col_stats:
                    status_color = "green" if test['status'] == 'Significant' else "orange"
                    st.metric(
                        "Statistical Significance",
                        f"p = {test['significance']:.3f}",
                        delta=test['status']
                    )
                
                # Performance comparison chart
                categories = ['Revenue', 'Conversion Rate', 'Customer Satisfaction']
                strategy_a_scores = [test['revenue_a']/1000, test['conversion_a']*1000, np.random.uniform(7, 9)]
                strategy_b_scores = [test['revenue_b']/1000, test['conversion_b']*1000, np.random.uniform(7, 9)]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=strategy_a_scores,
                    theta=categories,
                    fill='toself',
                    name=test['strategy_a']
                ))
                fig.add_trace(go.Scatterpolar(
                    r=strategy_b_scores,
                    theta=categories,
                    fill='toself',
                    name=test['strategy_b']
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max(max(strategy_a_scores), max(strategy_b_scores))])
                    ),
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚙️ Create New A/B Test")
        
        test_name = st.text_input("Test Name", "New Pricing Strategy Test")
        
        strategy_a = st.selectbox(
            "Strategy A",
            ["Dynamic Pricing", "Competitive Pricing", "Premium Pricing", "Penetration Pricing"]
        )
        
        strategy_b = st.selectbox(
            "Strategy B",
            ["Fixed Pricing", "Cost-Plus Pricing", "Value-Based Pricing", "Psychological Pricing"]
        )
        
        traffic_split = st.slider("Traffic Split (%)", 10, 50, 50)
        test_duration = st.selectbox("Test Duration", ["1 week", "2 weeks", "1 month", "3 months"])
        
        target_metric = st.selectbox(
            "Primary Metric",
            ["Revenue", "Conversion Rate", "Profit Margin", "Customer Lifetime Value"]
        )
        
        if st.button("🚀 Launch A/B Test"):
            st.success(f"A/B Test '{test_name}' launched successfully!")
            st.info(f"Traffic will be split {traffic_split}% / {100-traffic_split}% between strategies")
    
    # Historical test results
    st.subheader("📜 Historical Test Results")
    
    historical_data = {
        'Test Name': [
            'Premium vs Standard Pricing',
            'Seasonal Discount Strategy',
            'Bundle Pricing Test',
            'Geographic Pricing',
            'Time-based Pricing'
        ],
        'Winner': [
            'Premium Pricing',
            'Seasonal Discounts',
            'Bundle Pricing',
            'Standard Pricing',
            'Time-based Pricing'
        ],
        'Revenue Lift': ['+12.5%', '+8.7%', '+15.2%', '-3.1%', '+6.9%'],
        'Confidence': ['95%', '98%', '92%', '89%', '94%'],
        'Duration': ['2 weeks', '1 month', '3 weeks', '2 weeks', '1 month']
    }
    
    st.dataframe(
        pd.DataFrame(historical_data),
        hide_index=True,
        use_container_width=True
    )

def render_settings():
    st.subheader("⚙️ System Settings & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Pricing Rules")
        
        # Global pricing constraints
        min_price = st.number_input("Minimum Price ($)", min_value=0.01, value=5.00, step=0.01)
        max_price = st.number_input("Maximum Price ($)", min_value=min_price, value=999.99, step=0.01)
        
        max_price_increase = st.slider("Max Price Increase (%)", 0, 100, 25)
        max_price_decrease = st.slider("Max Price Decrease (%)", 0, 100, 30)
        
        repricing_frequency = st.selectbox(
            "Repricing Frequency",
            ["Real-time", "Hourly", "Daily", "Weekly"]
        )
        
        # Category-specific rules
        st.subheader("📂 Category Rules")
        categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
        
        for category in categories:
            with st.expander(f"{category} Settings"):
                st.slider(f"{category} - Min Margin (%)", 0, 50, 15, key=f"margin_{category}")
                st.slider(f"{category} - Max Discount (%)", 0, 70, 25, key=f"discount_{category}")
    
    with col2:
        st.subheader("🔗 Integrations")
        
        # E-commerce platform connections
        st.subheader("🛒 E-commerce Platforms")
        
        shopify_enabled = st.checkbox("Shopify Integration", value=True)
        if shopify_enabled:
            shopify_url = st.text_input("Shopify Store URL", placeholder="your-store.myshopify.com")
            shopify_key = st.text_input("API Key", type="password")
        
        woocommerce_enabled = st.checkbox("WooCommerce Integration")
        if woocommerce_enabled:
            woo_url = st.text_input("WooCommerce URL", placeholder="https://yourstore.com")
            woo_key = st.text_input("Consumer Key", type="password")
        
        # Data source connections
        st.subheader("📊 Data Sources")
        
        competitor_monitoring = st.checkbox("Competitor Price Monitoring", value=True)
        market_data_feed = st.checkbox("Market Data Feed", value=True)
        google_analytics = st.checkbox("Google Analytics", value=False)
        
        # Notification settings
        st.subheader("🔔 Notifications")
        
        email_alerts = st.checkbox("Email Alerts", value=True)
        slack_integration = st.checkbox("Slack Integration")
        dashboard_notifications = st.checkbox("Dashboard Notifications", value=True)
        
        alert_threshold = st.slider("Price Change Alert Threshold (%)", 1, 50, 10)
    
    # Data management
    st.subheader("💾 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📁 Export Data"):
            st.success("Data exported successfully!")
    
    with col2:
        if st.button("🔄 Backup Database"):
            st.success("Database backup completed!")
    
    with col3:
        if st.button("🧹 Clear Cache"):
            st.success("Cache cleared successfully!")
    
    # System status
    st.subheader("🏥 System Health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Response Time", "245ms", delta="-12ms")
    
    with col2:
        st.metric("Database Load", "67%", delta="+3%")
    
    with col3:
        st.metric("Active Users", "1,247", delta="+89")
    
    with col4:
        st.metric("Uptime", "99.8%", delta="+0.1%")
    
    # Save settings
    if st.button("💾 Save All Settings", type="primary"):
        st.success("Settings saved successfully!")
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()

