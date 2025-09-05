#!/usr/bin/env python3
"""
Model Checker Script

This script checks if the required ML models and data files exist
and can be loaded properly.
"""

import os
import sys

def check_models():
    """Check if model files exist"""
    models_dir = "/home/nahid/Desktop/StockVision/models"
    
    print("🔍 Checking Model Files")
    print("=" * 50)
    
    required_files = [
        'linear_regression_model.pkl',
        'feature_scaler.pkl',
        'feature_names.pkl'
    ]
    
    all_exist = True
    
    for filename in required_files:
        filepath = os.path.join(models_dir, filename)
        exists = os.path.exists(filepath)
        size = os.path.getsize(filepath) if exists else 0
        
        status = "✅" if exists else "❌"
        size_str = f"({size/1024:.1f} KB)" if exists else "(missing)"
        
        print(f"{status} {filename} {size_str}")
        
        if not exists:
            all_exist = False
    
    return all_exist

def check_data():
    """Check if data files exist"""
    data_dir = "/home/nahid/Desktop/StockVision/data/processed"
    
    print("\n📊 Checking Data Files")
    print("=" * 50)
    
    required_files = [
        'all_data.csv',
        'securities.csv'
    ]
    
    all_exist = True
    
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        exists = os.path.exists(filepath)
        size = os.path.getsize(filepath) if exists else 0
        
        status = "✅" if exists else "❌"
        size_str = f"({size/1024/1024:.1f} MB)" if exists else "(missing)"
        
        print(f"{status} {filename} {size_str}")
        
        if not exists:
            all_exist = False
    
    return all_exist

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n📦 Checking Dependencies")
    print("=" * 50)
    
    required_packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
        'fastapi',
        'uvicorn'
    ]
    
    all_installed = True
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (not installed)")
            all_installed = False
    
    return all_installed

def test_model_loading():
    """Test if models can be loaded"""
    print("\n🧪 Testing Model Loading")
    print("=" * 50)
    
    try:
        import joblib
        import pandas as pd
        import numpy as np
        
        models_dir = "/home/nahid/Desktop/StockVision/models"
        
        # Test loading scaler and feature names
        scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
        features_path = os.path.join(models_dir, "feature_names.pkl")
        
        if os.path.exists(scaler_path) and os.path.exists(features_path):
            scaler = joblib.load(scaler_path)
            feature_names = joblib.load(features_path)
            print(f"✅ Scaler loaded (type: {type(scaler).__name__})")
            print(f"✅ Feature names loaded ({len(feature_names)} features)")
            print(f"   Sample features: {feature_names[:5]}")
        else:
            print("❌ Preprocessing components missing")
            return False
        
        # Test loading one model
        lr_path = os.path.join(models_dir, "linear_regression_model.pkl")
        if os.path.exists(lr_path):
            model = joblib.load(lr_path)
            print(f"✅ Linear Regression model loaded (type: {type(model).__name__})")
            
            # Test if model can make a dummy prediction
            dummy_data = np.random.random((1, len(feature_names)))
            dummy_data_scaled = scaler.transform(dummy_data)
            prediction = model.predict(dummy_data_scaled)
            print(f"✅ Model prediction test successful (output: {prediction[0]:.2f})")
            
            return True
        else:
            print("❌ Linear Regression model missing")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return False

def test_data_loading():
    """Test if data can be loaded"""
    print("\n📈 Testing Data Loading")
    print("=" * 50)
    
    try:
        import pandas as pd
        
        data_path = "/home/nahid/Desktop/StockVision/data/processed/all_data.csv"
        
        if not os.path.exists(data_path):
            print("❌ Data file not found")
            return False
        
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Check required columns
        required_columns = ['trading_code', 'date', 'closing_price', 'opening_price', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        else:
            print("✅ All required columns present")
        
        # Check data types
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Date conversion successful")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Check for sample tickers
        sample_tickers = df['trading_code'].unique()[:10]
        print(f"   Sample tickers: {list(sample_tickers)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return False

def main():
    """Main function to run all checks"""
    print("StockVision System Check")
    print("=" * 50)
    
    models_ok = check_models()
    data_ok = check_data()
    deps_ok = check_dependencies()
    
    if models_ok and deps_ok:
        model_loading_ok = test_model_loading()
    else:
        model_loading_ok = False
    
    if data_ok and deps_ok:
        data_loading_ok = test_data_loading()
    else:
        data_loading_ok = False
    
    # Summary
    print("\n📋 Summary")
    print("=" * 50)
    print(f"Models: {'✅ Ready' if models_ok else '❌ Missing files'}")
    print(f"Data: {'✅ Ready' if data_ok else '❌ Missing files'}")
    print(f"Dependencies: {'✅ Installed' if deps_ok else '❌ Missing packages'}")
    print(f"Model Loading: {'✅ Working' if model_loading_ok else '❌ Failed'}")
    print(f"Data Loading: {'✅ Working' if data_loading_ok else '❌ Failed'}")
    
    overall_status = all([models_ok, data_ok, deps_ok, model_loading_ok, data_loading_ok])
    
    print(f"\n🎯 Overall Status: {'✅ READY' if overall_status else '❌ NOT READY'}")
    
    if not overall_status:
        print("\n🛠️ Next Steps:")
        if not deps_ok:
            print("1. Install missing dependencies: pip install pandas numpy scikit-learn joblib fastapi uvicorn")
        if not models_ok:
            print("2. Train models by running the training notebook")
        if not data_ok:
            print("3. Process raw data to create the required CSV files")
        if not model_loading_ok:
            print("4. Check model compatibility and file integrity")
        if not data_loading_ok:
            print("5. Verify data file format and column names")
    else:
        print("\n🚀 You can now run the prediction system!")
        print("   python test_predictions.py")
        print("   python backend/main.py")

if __name__ == "__main__":
    main()
