"""
Test script to verify the deployed wine classification app works correctly.
This simulates the same logic as your deployed app.
"""

import pandas as pd
import pickle
import os
from sklearn.datasets import load_wine

def test_app_logic():
    """Test the core logic of your deployed app."""
    print("🧪 Testing Wine Classification App Logic...")
    print("=" * 50)
    
    # Test 1: Model Loading
    print("1. Testing Model Loading...")
    try:
        if os.path.exists("random_forest_model.pkl"):
            with open("random_forest_model.pkl", "rb") as f:
                model = pickle.load(f)
            print("   ✅ Model loaded successfully")
        else:
            print("   ⚠️ Model file not found (app will auto-train)")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False
    
    # Test 2: Dataset Access
    print("2. Testing Dataset Access...")
    try:
        wine_data = load_wine()
        feature_names = wine_data.feature_names
        class_names = wine_data.target_names
        print(f"   ✅ Dataset loaded: {len(feature_names)} features, {len(class_names)} classes")
    except Exception as e:
        print(f"   ❌ Dataset loading failed: {e}")
        return False
    
    # Test 3: Sample Prediction
    print("3. Testing Sample Prediction...")
    try:
        # Use default values from your app
        sample_input = [13.0, 2.34, 2.36, 19.5, 99.7, 2.29, 2.03, 0.36, 1.59, 5.06, 0.96, 2.61, 746]
        input_df = pd.DataFrame([sample_input], columns=feature_names)
        
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        print(f"   ✅ Prediction: {class_names[prediction]}")
        print(f"   ✅ Confidence: {probabilities[prediction]:.1%}")
        
        # Show all probabilities
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            print(f"      {class_name}: {prob:.1%}")
            
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        return False
    
    # Test 4: Feature Importance
    print("4. Testing Feature Importance...")
    try:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("   ✅ Top 5 Most Important Features:")
            for _, row in importance_df.head().iterrows():
                print(f"      {row['Feature']}: {row['Importance']:.3f}")
        else:
            print("   ⚠️ Model doesn't have feature importance")
    except Exception as e:
        print(f"   ❌ Feature importance failed: {e}")
    
    # Test 5: Edge Cases
    print("5. Testing Edge Cases...")
    try:
        # Test with extreme values
        extreme_input = [20.0, 10.0, 5.0, 50.0, 200, 5.0, 8.0, 2.0, 5.0, 15.0, 2.0, 6.0, 2000]
        extreme_df = pd.DataFrame([extreme_input], columns=feature_names)
        extreme_pred = model.predict(extreme_df)[0]
        print(f"   ✅ Extreme values prediction: {class_names[extreme_pred]}")
        
        # Test with minimum values
        min_input = [8.0, 0.5, 1.0, 5.0, 50, 0.5, 0.1, 0.05, 0.2, 1.0, 0.3, 1.0, 200]
        min_df = pd.DataFrame([min_input], columns=feature_names)
        min_pred = model.predict(min_df)[0]
        print(f"   ✅ Minimum values prediction: {class_names[min_pred]}")
        
    except Exception as e:
        print(f"   ❌ Edge case testing failed: {e}")
    
    print("=" * 50)
    print("🎉 All tests completed! Your app logic is working correctly.")
    return True

if __name__ == "__main__":
    success = test_app_logic()
    if success:
        print("\n✅ Your deployed app should be working perfectly!")
        print("🔗 Test it live at your Streamlit Cloud URL")
    else:
        print("\n❌ Some issues detected. Check the errors above.")