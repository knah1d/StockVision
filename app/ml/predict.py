"""
Stock Price Prediction Module

This module provides functionality to use the pre-trained machine learning models
for stock price prediction and trend analysis.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class StockPredictor:
    """
    A class for predicting stock prices using pre-trained ML models.
    """
    
    def __init__(self, models_dir: str = "/home/nahid/Desktop/StockVision/models"):
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.model_names = [
            'decision_tree',
            'gradient_boosting',
            'k-nearest_neighbors',
            'linear_regression',
            'random_forest',
            'ridge_regression'
        ]
        
        # Load preprocessing components
        self._load_preprocessing_components()
        
        # Load all available models
        self._load_models()
    
    def _load_preprocessing_components(self) -> bool:
        """Load scaler and feature names."""
        try:
            scaler_path = os.path.join(self.models_dir, "feature_scaler.pkl")
            features_path = os.path.join(self.models_dir, "feature_names.pkl")
            
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(features_path)
            
            logger.info(f"✅ Loaded preprocessing components with {len(self.feature_names)} features")
            return True
        except FileNotFoundError as e:
            logger.error(f"❌ Preprocessing components not found: {e}")
            return False
    
    def _load_models(self) -> None:
        """Load all available trained models."""
        for model_name in self.model_names:
            model_path = os.path.join(self.models_dir, f"{model_name}_model.pkl")
            try:
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"✅ Loaded {model_name} model")
            except FileNotFoundError:
                logger.warning(f"❌ Model not found: {model_path}")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and features for stock prediction.
        
        Args:
            data (pd.DataFrame): Historical stock data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        df_features = data.copy()
        
        # Ensure proper sorting for time series features
        df_features = df_features.sort_values(['trading_code', 'date'])
        
        # Technical indicators for each stock
        for code in df_features['trading_code'].unique():
            mask = df_features['trading_code'] == code
            stock_data = df_features[mask].copy()
            
            if len(stock_data) > 1:
                # Price change features
                df_features.loc[mask, 'price_change'] = stock_data['closing_price'].pct_change()
                df_features.loc[mask, 'price_change_abs'] = stock_data['closing_price'].diff()
                
                # Moving averages
                df_features.loc[mask, 'ma_5'] = stock_data['closing_price'].rolling(5).mean()
                df_features.loc[mask, 'ma_10'] = stock_data['closing_price'].rolling(10).mean()
                df_features.loc[mask, 'ma_20'] = stock_data['closing_price'].rolling(20).mean()
                
                # Volatility (rolling standard deviation)
                df_features.loc[mask, 'volatility_5'] = stock_data['closing_price'].rolling(5).std()
                df_features.loc[mask, 'volatility_10'] = stock_data['closing_price'].rolling(10).std()
                
                # Price position relative to high/low
                high_low_diff = stock_data['high'] - stock_data['low']
                high_low_diff = high_low_diff.replace(0, np.nan)  # Avoid division by zero
                df_features.loc[mask, 'price_position'] = (
                    (stock_data['closing_price'] - stock_data['low']) / high_low_diff
                )
                
                # Lagged features (previous day values)
                df_features.loc[mask, 'prev_close'] = stock_data['closing_price'].shift(1)
                df_features.loc[mask, 'prev_volume'] = stock_data['volume'].shift(1)
                df_features.loc[mask, 'prev_high'] = stock_data['high'].shift(1)
                df_features.loc[mask, 'prev_low'] = stock_data['low'].shift(1)
        
        # Additional features
        df_features['high_low_pct'] = (
            (df_features['high'] - df_features['low']) / df_features['low'] * 100
        )
        df_features['open_close_pct'] = (
            (df_features['closing_price'] - df_features['opening_price']) / 
            df_features['opening_price'] * 100
        )
        
        # Date-based features
        df_features['month'] = df_features['date'].dt.month
        df_features['day_of_week'] = df_features['date'].dt.dayofweek
        df_features['quarter'] = df_features['date'].dt.quarter
        
        return df_features
    
    def prepare_prediction_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for prediction by selecting features and scaling.
        
        Args:
            data (pd.DataFrame): Data with features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Scaled features and actual prices
        """
        if self.feature_names is None or self.scaler is None:
            raise ValueError("Preprocessing components not loaded")
        
        # Select only the required features
        try:
            X = data[self.feature_names].copy()
        except KeyError as e:
            missing_features = set(self.feature_names) - set(data.columns)
            raise ValueError(f"Missing features: {missing_features}")
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get actual prices for comparison
        y_actual = data['closing_price'].values
        
        return X_scaled, y_actual
    
    def predict_single_stock(self, ticker: str, data: pd.DataFrame, model_name: str = 'random_forest') -> Dict:
        """
        Predict stock prices for a single ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            data (pd.DataFrame): Historical data for the ticker
            model_name (str): Name of the model to use
            
        Returns:
            Dict: Prediction results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available. Available models: {list(self.models.keys())}")
        
        try:
            # Create features
            features_data = self.create_features(data)
            
            # Prepare data for prediction
            X_scaled, y_actual = self.prepare_prediction_data(features_data)
            
            # Make predictions
            model = self.models[model_name]
            predictions = model.predict(X_scaled)
            
            # Calculate metrics
            mse = np.mean((predictions - y_actual) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - y_actual))
            
            # Calculate prediction accuracy (within 5% tolerance)
            tolerance = 0.05
            accurate_predictions = np.abs((predictions - y_actual) / y_actual) <= tolerance
            accuracy = np.mean(accurate_predictions) * 100
            
            # Get latest prediction and trend
            latest_prediction = predictions[-1]
            latest_actual = y_actual[-1]
            
            # Determine trend (compare last 5 predictions)
            if len(predictions) >= 5:
                recent_trend = np.mean(predictions[-5:]) - np.mean(predictions[-10:-5]) if len(predictions) >= 10 else 0
                trend = "Upward" if recent_trend > 0 else "Downward" if recent_trend < 0 else "Stable"
            else:
                trend = "Insufficient data"
            
            return {
                'ticker': ticker,
                'model_used': model_name,
                'latest_actual_price': float(latest_actual),
                'latest_predicted_price': float(latest_prediction),
                'prediction_accuracy': f"{accuracy:.2f}%",
                'rmse': float(rmse),
                'mae': float(mae),
                'trend': trend,
                'predictions': predictions.tolist(),
                'actual_prices': y_actual.tolist(),
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'prediction_confidence': min(accuracy / 100, 1.0),
                'total_data_points': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error predicting for {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'predictions': [],
                'actual_prices': []
            }
    
    def predict_multiple_stocks(self, tickers: List[str], data: pd.DataFrame, 
                              model_name: str = 'random_forest') -> Dict:
        """
        Predict stock prices for multiple tickers.
        
        Args:
            tickers (List[str]): List of stock ticker symbols
            data (pd.DataFrame): Historical data containing all tickers
            model_name (str): Name of the model to use
            
        Returns:
            Dict: Prediction results for all tickers
        """
        results = {}
        
        for ticker in tickers:
            ticker_data = data[data['trading_code'] == ticker].copy()
            if len(ticker_data) > 0:
                results[ticker] = self.predict_single_stock(ticker, ticker_data, model_name)
            else:
                results[ticker] = {
                    'ticker': ticker,
                    'error': 'No data available',
                    'predictions': [],
                    'actual_prices': []
                }
        
        return results
    
    def compare_models(self, ticker: str, data: pd.DataFrame) -> Dict:
        """
        Compare predictions from all available models for a single ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            data (pd.DataFrame): Historical data for the ticker
            
        Returns:
            Dict: Comparison results
        """
        comparison_results = {}
        
        for model_name in self.models.keys():
            try:
                result = self.predict_single_stock(ticker, data, model_name)
                comparison_results[model_name] = {
                    'rmse': result['rmse'],
                    'mae': result['mae'],
                    'accuracy': result['prediction_accuracy'],
                    'latest_prediction': result['latest_predicted_price'],
                    'trend': result['trend']
                }
            except Exception as e:
                comparison_results[model_name] = {'error': str(e)}
        
        # Find best model based on RMSE
        best_model = min(
            comparison_results.keys(),
            key=lambda x: comparison_results[x].get('rmse', float('inf'))
        )
        
        return {
            'ticker': ticker,
            'comparison': comparison_results,
            'best_model': best_model,
            'best_model_rmse': comparison_results[best_model].get('rmse', 'N/A')
        }
    
    def predict_future_prices(self, ticker: str, data: pd.DataFrame, 
                            days_ahead: int = 5, model_name: str = 'random_forest') -> Dict:
        """
        Predict future stock prices (experimental - requires careful validation).
        
        Args:
            ticker (str): Stock ticker symbol
            data (pd.DataFrame): Historical data for the ticker
            days_ahead (int): Number of days to predict ahead
            model_name (str): Name of the model to use
            
        Returns:
            Dict: Future price predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available")
        
        try:
            # Get the most recent data point
            ticker_data = data[data['trading_code'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date')
            
            if len(ticker_data) < 30:  # Need sufficient history
                return {'error': 'Insufficient historical data for future prediction'}
            
            # Use the last known values to simulate future data points
            last_record = ticker_data.iloc[-1].copy()
            future_predictions = []
            
            for day in range(1, days_ahead + 1):
                # Create a future date
                future_date = last_record['date'] + timedelta(days=day)
                
                # Create a new record based on the last known record
                # Note: This is a simplified approach and should be enhanced with more sophisticated methods
                future_record = last_record.copy()
                future_record['date'] = future_date
                
                # Create features for this single record
                temp_data = pd.concat([ticker_data.tail(50), pd.DataFrame([future_record])], ignore_index=True)
                features_data = self.create_features(temp_data)
                
                # Get the last row (our future prediction point)
                prediction_row = features_data.iloc[-1:].copy()
                
                # Prepare for prediction
                try:
                    X_scaled, _ = self.prepare_prediction_data(prediction_row)
                    
                    # Make prediction
                    model = self.models[model_name]
                    predicted_price = model.predict(X_scaled)[0]
                    
                    future_predictions.append({
                        'date': future_date.strftime('%Y-%m-%d'),
                        'predicted_price': float(predicted_price),
                        'days_ahead': day
                    })
                    
                    # Update last_record with predicted price for next iteration
                    last_record['closing_price'] = predicted_price
                    last_record['opening_price'] = predicted_price
                    last_record['high'] = predicted_price * 1.02  # Rough estimate
                    last_record['low'] = predicted_price * 0.98   # Rough estimate
                    
                except Exception as e:
                    future_predictions.append({
                        'date': future_date.strftime('%Y-%m-%d'),
                        'error': f'Prediction failed: {str(e)}',
                        'days_ahead': day
                    })
            
            return {
                'ticker': ticker,
                'model_used': model_name,
                'future_predictions': future_predictions,
                'last_known_price': float(ticker_data.iloc[-1]['closing_price']),
                'last_known_date': ticker_data.iloc[-1]['date'].strftime('%Y-%m-%d'),
                'warning': 'Future predictions are experimental and should be used with caution'
            }
            
        except Exception as e:
            return {
                'ticker': ticker,
                'error': f'Future prediction failed: {str(e)}'
            }
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models and features."""
        return {
            'available_models': list(self.models.keys()),
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'features': self.feature_names if self.feature_names else [],
            'scaler_loaded': self.scaler is not None
        }

# Example usage functions
def predict_stock_price(ticker: str, data_path: str = "/home/nahid/Desktop/StockVision/data/processed/all_data.csv",
                       model_name: str = 'random_forest') -> Dict:
    """
    Convenience function to predict stock price for a single ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        data_path (str): Path to the CSV data file
        model_name (str): Name of the model to use
        
    Returns:
        Dict: Prediction results
    """
    try:
        # Load data
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Initialize predictor
        predictor = StockPredictor()
        
        # Filter data for the ticker
        ticker_data = df[df['trading_code'] == ticker].copy()
        
        if len(ticker_data) == 0:
            return {'error': f'No data found for ticker: {ticker}'}
        
        # Make prediction
        result = predictor.predict_single_stock(ticker, ticker_data, model_name)
        return result
        
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}

def compare_stock_models(ticker: str, data_path: str = "/home/nahid/Desktop/StockVision/data/processed/all_data.csv") -> Dict:
    """
    Convenience function to compare all models for a single ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        data_path (str): Path to the CSV data file
        
    Returns:
        Dict: Model comparison results
    """
    try:
        # Load data
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Initialize predictor
        predictor = StockPredictor()
        
        # Filter data for the ticker
        ticker_data = df[df['trading_code'] == ticker].copy()
        
        if len(ticker_data) == 0:
            return {'error': f'No data found for ticker: {ticker}'}
        
        # Compare models
        result = predictor.compare_models(ticker, ticker_data)
        return result
        
    except Exception as e:
        return {'error': f'Model comparison failed: {str(e)}'}

if __name__ == "__main__":
    # Example usage
    print("Stock Prediction Module")
    print("=" * 50)
    
    # Initialize predictor
    predictor = StockPredictor()
    
    # Print model info
    info = predictor.get_model_info()
    print(f"Available models: {info['available_models']}")
    print(f"Number of features: {info['feature_count']}")
    
    # Example prediction (uncomment to test)
    # result = predict_stock_price('AAPL')
    # print(result)
