#!/usr/bin/env python3
"""
Stock Prediction Test Script

This script demonstrates how to use the stock prediction functionality
directly without the API.
"""

import sys
import os
sys.path.append('/home/nahid/Desktop/StockVision')

from backend.services.prediction_service import PredictionService
import json

def test_prediction_service():
    """Test the prediction service functionality"""
    print("🚀 Stock Prediction Test")
    print("=" * 50)
    
    # Initialize the prediction service
    print("Initializing prediction service...")
    predictor = PredictionService()
    
    # Check service health
    print("\n📊 Service Health Check:")
    health = predictor.get_health_status()
    print(f"Status: {health['status']}")
    print(f"Models loaded: {health['models_loaded']}")
    print(f"Scaler loaded: {health['preprocessing_components']['scaler']}")
    print(f"Feature names loaded: {health['preprocessing_components']['feature_names']}")
    
    # Get model info
    print("\n🤖 Available Models:")
    model_info = predictor.get_model_info()
    print(f"Models: {model_info['available_models']}")
    print(f"Feature count: {model_info['feature_count']}")
    
    # Test with a sample ticker (if models are loaded)
    if model_info['available_models']:
        print("\n📈 Testing Prediction:")
        
        # Try to predict for a sample ticker
        # Note: Replace 'AAPL' with actual ticker from your dataset
        sample_tickers = ['ACI', 'BEXIMCO', 'GP', 'SQURPHARMA', 'BATBC']  # Common BD stock tickers
        
        for ticker in sample_tickers:
            try:
                print(f"\nPredicting for {ticker}...")
                result = predictor.predict_ticker(ticker, days=90, model_name='linear_regression')
                
                if 'error' not in result:
                    print(f"✅ Success!")
                    print(f"  Current Price: ৳{result['ticker_info']['current_price']:.2f}")
                    print(f"  Predicted Price: ৳{result['ticker_info']['predicted_price']:.2f}")
                    print(f"  Change: {result['ticker_info']['price_difference_pct']:.2f}%")
                    print(f"  Trend: {result['trend_analysis']['trend']}")
                    print(f"  Accuracy: {result['prediction_metrics']['accuracy']}")
                    break  # Success with first ticker
                else:
                    print(f"❌ Error: {result['error']}")
            except Exception as e:
                print(f"❌ Exception: {str(e)}")
        
        # Test model comparison
        if 'error' not in result:
            print(f"\n🔍 Comparing Models for {ticker}:")
            comparison = predictor.compare_models_for_ticker(ticker, days=90)
            
            if 'error' not in comparison:
                print(f"Best model: {comparison['best_model']}")
                print(f"Best RMSE: {comparison['best_model_rmse']:.2f}")
                
                for model_name, metrics in comparison['comparison'].items():
                    if 'error' not in metrics:
                        print(f"  {model_name}: RMSE={metrics['rmse']:.2f}, Accuracy={metrics['accuracy']}")
            
            # Test recommendations
            print(f"\n💡 Investment Recommendations for {ticker}:")
            recommendation = predictor.get_prediction_recommendations(ticker, days=90)
            
            if 'error' not in recommendation:
                print(f"Recommendation: {recommendation['recommendation']}")
                print(f"Reason: {recommendation['reason']}")
                print(f"Confidence: {recommendation['confidence_level']}")
                print(f"Risk Level: {recommendation['risk_level']}")
    else:
        print("\n❌ No models loaded. Please check:")
        print("1. Model files exist in /home/nahid/Desktop/StockVision/models/")
        print("2. Feature scaler and feature names are available")
        print("3. Required dependencies are installed (pandas, numpy, scikit-learn, joblib)")

def test_direct_prediction():
    """Test prediction using direct function calls"""
    print("\n" + "=" * 50)
    print("🧪 Direct Function Test")
    print("=" * 50)
    
    try:
        # Import the direct prediction functions
        from app.ml.predict import predict_stock_price, compare_stock_models
        
        # Test prediction
        print("Testing direct prediction function...")
        result = predict_stock_price('ACI', model_name='linear_regression')
        
        if 'error' not in result:
            print("✅ Direct prediction successful!")
            print(f"Ticker: {result['ticker']}")
            print(f"Latest predicted price: ৳{result['latest_predicted_price']:.2f}")
            print(f"Accuracy: {result['prediction_accuracy']}")
        else:
            print(f"❌ Direct prediction failed: {result['error']}")
        
        # Test model comparison
        print("\nTesting model comparison function...")
        comparison = compare_stock_models('ACI', model_names=['linear_regression'])
        
        if 'error' not in comparison:
            print("✅ Model comparison successful!")
            print(f"Best model: {comparison['best_model']}")
        else:
            print(f"❌ Model comparison failed: {comparison['error']}")
            
    except Exception as e:
        print(f"❌ Exception in direct function test: {str(e)}")

if __name__ == "__main__":
    print("StockVision ML Prediction System Test")
    print("====================================")
    
    # Test backend service
    test_prediction_service()
    
    # Test direct functions
    test_direct_prediction()
    
    print("\n🎉 Test completed!")
    print("\nTo use the prediction system:")
    print("1. Start the FastAPI server: python backend/main.py")
    print("2. Access predictions at: http://localhost:8001/api/v1/predict/{ticker}")
    print("3. View API docs at: http://localhost:8001/docs")
