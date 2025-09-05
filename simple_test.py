#!/usr/bin/env python3
"""
Simple Prediction Test

A simple test to verify the ML prediction system is working.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import joblib
        from sklearn.ensemble import RandomForestRegressor
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_models():
    """Test if models can be loaded"""
    print("Testing model loading...")
    try:
        import joblib
        models_dir = "/home/nahid/Desktop/StockVision/models"
        
        # Test loading scaler and features
        scaler = joblib.load(os.path.join(models_dir, "feature_scaler.pkl"))
        feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))
        print(f"✅ Preprocessing components loaded ({len(feature_names)} features)")
        
        # Test loading a model
        model = joblib.load(os.path.join(models_dir, "random_forest_model.pkl"))
        print("✅ Random Forest model loaded")
        
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_data():
    """Test if data can be loaded"""
    print("Testing data loading...")
    try:
        import pandas as pd
        
        data_path = "/home/nahid/Desktop/StockVision/data/processed/all_data.csv"
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Sample tickers: {list(df['trading_code'].unique()[:5])}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_backend_service():
    """Test the backend prediction service"""
    print("Testing backend prediction service...")
    try:
        # Add the project path
        sys.path.append('/home/nahid/Desktop/StockVision')
        
        from backend.services.prediction_service import PredictionService
        
        # Initialize service
        predictor = PredictionService()
        
        # Check health
        health = predictor.get_health_status()
        print(f"✅ Service status: {health['status']}")
        print(f"   Models loaded: {health['models_loaded']}")
        
        # Test a simple prediction
        sample_tickers = ['ACI', 'BEXIMCO', 'GP', 'SQURPHARMA', 'BATBC']
        
        for ticker in sample_tickers:
            try:
                result = predictor.predict_ticker(ticker, days=30)
                if 'error' not in result:
                    print(f"✅ Prediction successful for {ticker}")
                    print(f"   Current: ৳{result['ticker_info']['current_price']:.2f}")
                    print(f"   Predicted: ৳{result['ticker_info']['predicted_price']:.2f}")
                    print(f"   Change: {result['ticker_info']['price_difference_pct']:.2f}%")
                    return True
                else:
                    print(f"⚠️ Prediction failed for {ticker}: {result['error']}")
            except Exception as e:
                print(f"⚠️ Error with {ticker}: {str(e)}")
        
        print("❌ No successful predictions")
        return False
        
    except Exception as e:
        print(f"❌ Backend service error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 StockVision ML Prediction System - Quick Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Test", test_models),
        ("Data Test", test_data),
        ("Backend Service Test", test_backend_service)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        result = test_func()
        results.append(result)
        print()
    
    # Summary
    print("=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    overall = all(results)
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall else '❌ SOME TESTS FAILED'}")
    
    if overall:
        print("\n🚀 Ready to start the prediction system!")
        print("   Backend: source venv/bin/activate && python backend/main.py")
        print("   Frontend: cd stockvision-app && npm start")
    else:
        print("\n🛠️ Please fix the failing tests before proceeding.")

if __name__ == "__main__":
    main()
