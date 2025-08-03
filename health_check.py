"""
Health check script for the wine classification app.
Can be used for monitoring and debugging deployment issues.
"""

import os
import pickle
import pandas as pd
from sklearn.datasets import load_wine

def check_model_file():
    """Check if model file exists and can be loaded."""
    model_path = "random_forest_model.pkl"
    
    if not os.path.exists(model_path):
        return False, "Model file not found"
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return True, f"Model loaded successfully (size: {os.path.getsize(model_path)} bytes)"
    except Exception as e:
        return False, f"Error loading model: {e}"

def check_dependencies():
    """Check if all required packages are available."""
    try:
        import streamlit
        import sklearn
        import pandas
        import numpy
        return True, "All dependencies available"
    except ImportError as e:
        return False, f"Missing dependency: {e}"

def check_dataset():
    """Check if wine dataset can be loaded."""
    try:
        data = load_wine()
        return True, f"Dataset loaded: {data.data.shape[0]} samples, {data.data.shape[1]} features"
    except Exception as e:
        return False, f"Error loading dataset: {e}"

def run_health_check():
    """Run all health checks."""
    print("Running Health Check...")
    print("=" * 50)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Dataset", check_dataset),
        ("Model File", check_model_file),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            passed, message = check_func()
            status = "PASS" if passed else "FAIL"
            print(f"{check_name}: {status} - {message}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"{check_name}: ERROR - {e}")
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("All health checks passed! App should work correctly.")
    else:
        print("Some health checks failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    run_health_check()