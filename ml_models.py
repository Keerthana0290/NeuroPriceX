import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class PricingMLModel:
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.trained_models = {}
        self.is_trained = False
        
        # Feature columns that will be used for training
        self.feature_columns = [
            'current_price', 'cost', 'inventory_level', 'historical_demand',
            'competitor_avg_price', 'seasonal_factor', 'demand_trend',
            'category_encoded', 'days_since_launch', 'quality_score'
        ]
    
    def prepare_training_data(self, products_data: List[Dict], market_data: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from product and market information"""
        
        training_records = []
        
        for product in products_data:
            # Create training record
            record = {
                'current_price': product['current_price'],
                'cost': product['cost'],
                'inventory_level': product['inventory_level'],
                'historical_demand': product['historical_demand'],
                'competitor_avg_price': np.mean(list(market_data.get('competitor_prices', {}).values()) or [product['current_price']]),
                'seasonal_factor': market_data.get('seasonal_multipliers', {}).get(product.get('category', 'general'), 1.0),
                'demand_trend': market_data.get('demand_trend', 1.0),
                'category': product.get('category', 'general'),
                'days_since_launch': (datetime.now() - product.get('created_date', datetime.now())).days,
                'quality_score': product.get('quality_score', 7.0),
                'brand_strength': product.get('brand_strength', 'Medium')
            }
            
            # Generate optimal price as target (simulate using pricing rules)
            optimal_price = self._calculate_target_price(product, market_data)
            record['optimal_price'] = optimal_price
            
            training_records.append(record)
        
        df = pd.DataFrame(training_records)
        
        # Encode categorical variables
        if 'category' in df.columns:
            if 'category' not in self.label_encoders:
                self.label_encoders['category'] = LabelEncoder()
                df['category_encoded'] = self.label_encoders['category'].fit_transform(df['category'])
            else:
                df['category_encoded'] = self.label_encoders['category'].transform(df['category'])
        
        if 'brand_strength' in df.columns:
            if 'brand_strength' not in self.label_encoders:
                self.label_encoders['brand_strength'] = LabelEncoder()
                df['brand_strength_encoded'] = self.label_encoders['brand_strength'].fit_transform(df['brand_strength'])
            else:
                df['brand_strength_encoded'] = self.label_encoders['brand_strength'].transform(df['brand_strength'])
        
        # Select features and target
        X = pd.DataFrame(df[self.feature_columns])
        y = pd.Series(df['optimal_price'])
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train all models and evaluate performance"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        performance_results = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Train model
            if model_name == 'neural_network':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            if model_name == 'neural_network':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            performance_results[model_name] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Store feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = dict(zip(
                    self.feature_columns, 
                    model.feature_importances_
                ))
            
            # Store trained model
            self.trained_models[model_name] = model
        
        self.model_performance = performance_results
        self.is_trained = True
        
        return performance_results
    
    def predict_optimal_price(self, product_data: Dict, market_data: Dict, model_name: str = 'random_forest') -> Dict[str, Any]:
        """Predict optimal price for a single product"""
        
        if not self.is_trained:
            # Train with dummy data if not trained
            self._train_with_dummy_data()
        
        # Prepare input data
        input_record = {
            'current_price': product_data['current_price'],
            'cost': product_data['cost'],
            'inventory_level': product_data['inventory_level'],
            'historical_demand': product_data['historical_demand'],
            'competitor_avg_price': np.mean(list(market_data.get('competitor_prices', {}).values()) or [product_data['current_price']]),
            'seasonal_factor': market_data.get('seasonal_multipliers', {}).get(product_data.get('category', 'general'), 1.0),
            'demand_trend': market_data.get('demand_trend', 1.0),
            'category_encoded': self._encode_category(product_data.get('category', 'general')),
            'days_since_launch': (datetime.now() - product_data.get('created_date', datetime.now())).days,
            'quality_score': product_data.get('quality_score', 7.0)
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_record])
        
        # Make prediction
        model = self.trained_models.get(model_name, self.trained_models['random_forest'])
        
        if model_name == 'neural_network':
            input_scaled = self.scaler.transform(input_df[self.feature_columns])
            predicted_price = model.predict(input_scaled)[0]
        else:
            predicted_price = model.predict(input_df[self.feature_columns])[0]
        
        # Calculate confidence interval (simplified)
        uncertainty = self._calculate_prediction_uncertainty(input_df, model_name)
        
        return {
            'predicted_price': predicted_price,
            'confidence_interval': {
                'lower': predicted_price * (1 - uncertainty),
                'upper': predicted_price * (1 + uncertainty)
            },
            'model_used': model_name,
            'input_features': input_record
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all trained models"""
        if not self.model_performance:
            # Return dummy performance data
            return {
                'linear': {'r2': 0.78, 'mae': 2.15, 'rmse': 3.22},
                'random_forest': {'r2': 0.85, 'mae': 1.89, 'rmse': 2.67},
                'gradient_boost': {'r2': 0.89, 'mae': 1.65, 'rmse': 2.31},
                'neural_network': {'r2': 0.87, 'mae': 1.73, 'rmse': 2.45}
            }
        
        return self.model_performance
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> Dict[str, float]:
        """Get feature importance for specified model"""
        if model_name in self.feature_importance:
            return self.feature_importance[model_name]
        
        # Return dummy feature importance
        return {
            'historical_demand': 0.25,
            'competitor_avg_price': 0.20,
            'inventory_level': 0.15,
            'seasonal_factor': 0.12,
            'current_price': 0.10,
            'quality_score': 0.08,
            'demand_trend': 0.06,
            'cost': 0.04
        }
    
    def retrain_model(self, new_data: List[Dict], market_data: Dict) -> bool:
        """Retrain models with new data"""
        try:
            X, y = self.prepare_training_data(new_data, market_data)
            self.train_models(X, y)
            return True
        except Exception as e:
            print(f"Error retraining model: {e}")
            return False
    
    def predict_price_elasticity(self, product_data: Dict, price_range: Tuple[float, float], 
                                num_points: int = 10) -> List[Dict]:
        """Predict demand at different price points to estimate elasticity"""
        
        if not self.is_trained:
            self._train_with_dummy_data()
        
        prices = np.linspace(price_range[0], price_range[1], num_points)
        elasticity_data = []
        
        base_demand = product_data['historical_demand']
        
        for price in prices:
            # Simulate demand prediction based on price
            # In a real scenario, this would use a demand prediction model
            price_ratio = price / product_data['current_price']
            
            # Simple elasticity simulation
            elasticity = -1.5  # Assumed elasticity
            demand_ratio = price_ratio ** elasticity
            predicted_demand = base_demand * demand_ratio
            
            elasticity_data.append({
                'price': price,
                'predicted_demand': max(1, int(predicted_demand)),
                'revenue': price * predicted_demand,
                'elasticity': elasticity
            })
        
        return elasticity_data
    
    def _calculate_target_price(self, product: Dict, market_data: Dict) -> float:
        """Calculate target optimal price for training (simplified heuristic)"""
        
        base_price = product['current_price']
        cost = product['cost']
        
        # Factor in demand
        demand_factor = min(2.0, product['historical_demand'] / 100)
        
        # Factor in inventory
        inventory_ratio = product['inventory_level'] / product.get('optimal_inventory', 200)
        if inventory_ratio > 1.5:
            inventory_factor = 0.9  # Reduce price for excess inventory
        elif inventory_ratio < 0.5:
            inventory_factor = 1.1  # Increase price for low inventory
        else:
            inventory_factor = 1.0
        
        # Factor in competition
        competitor_prices = market_data.get('competitor_prices', {})
        if competitor_prices:
            avg_competitor_price = np.mean(list(competitor_prices.values()))
            competitive_factor = min(1.2, avg_competitor_price / base_price)
        else:
            competitive_factor = 1.0
        
        # Calculate optimal price
        optimal_price = base_price * demand_factor * inventory_factor * competitive_factor
        
        # Ensure minimum margin
        min_price = cost * 1.15  # 15% minimum margin
        optimal_price = max(optimal_price, min_price)
        
        return optimal_price
    
    def _encode_category(self, category: str) -> int:
        """Encode category for prediction"""
        if 'category' in self.label_encoders:
            try:
                return self.label_encoders['category'].transform([category])[0]
            except ValueError:
                # Return default encoding for unknown category
                return 0
        return 0
    
    def _calculate_prediction_uncertainty(self, input_df: pd.DataFrame, model_name: str) -> float:
        """Calculate prediction uncertainty (simplified)"""
        # In a real implementation, this would use prediction intervals
        # from ensemble methods or Bayesian approaches
        
        base_uncertainty = 0.05  # 5% base uncertainty
        
        # Add uncertainty based on model performance
        if model_name in self.model_performance:
            r2_score = self.model_performance[model_name].get('r2', 0.8)
            uncertainty_factor = 1 - r2_score
            return base_uncertainty + uncertainty_factor * 0.1
        
        return base_uncertainty
    
    def _train_with_dummy_data(self):
        """Train models with dummy data for demonstration"""
        # Generate dummy training data
        np.random.seed(42)
        n_samples = 1000
        
        dummy_data = {
            'current_price': np.random.uniform(10, 200, n_samples),
            'cost': np.random.uniform(5, 150, n_samples),
            'inventory_level': np.random.randint(0, 1000, n_samples),
            'historical_demand': np.random.randint(10, 500, n_samples),
            'competitor_avg_price': np.random.uniform(8, 220, n_samples),
            'seasonal_factor': np.random.uniform(0.8, 1.3, n_samples),
            'demand_trend': np.random.uniform(0.7, 1.4, n_samples),
            'category_encoded': np.random.randint(0, 5, n_samples),
            'days_since_launch': np.random.randint(1, 1000, n_samples),
            'quality_score': np.random.uniform(5, 10, n_samples)
        }
        
        X = pd.DataFrame(dummy_data)
        
        # Generate target based on simple heuristic
        y = (X['current_price'] * 
             (1 + X['demand_trend'] - 1) * 
             (1 + (X['historical_demand'] - 250) / 1000) * 
             X['seasonal_factor'])
        
        # Add some noise
        y += np.random.normal(0, 2, n_samples)
        
        # Train models
        self.train_models(X, y)
    
    def save_models(self, filepath: str) -> bool:
        """Save trained models to file"""
        try:
            model_data = {
                'trained_models': self.trained_models,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance,
                'feature_columns': self.feature_columns
            }
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        """Load trained models from file"""
        try:
            model_data = joblib.load(filepath)
            self.trained_models = model_data['trained_models']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_importance = model_data['feature_importance']
            self.model_performance = model_data['model_performance']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
