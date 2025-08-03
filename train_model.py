
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import pickle
import os

def train_wine_classifier():
    """Train and save a wine classification model."""
    print("🍷 Loading wine dataset...")
    
    # Load dataset
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {data.target_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("🔄 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    print(f"✅ Training accuracy: {train_accuracy:.3f}")
    print(f"✅ Test accuracy: {test_accuracy:.3f}")
    
    # Detailed classification report
    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': data.feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 5 Most Important Features:")
    for _, row in feature_importance.head().iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Save model
    model_path = "random_forest_model.pkl"
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"💾 Model saved to {model_path}")
        print(f"📁 File size: {os.path.getsize(model_path) / 1024:.1f} KB")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return None
    
    return model

if __name__ == "__main__":
    model = train_wine_classifier()
    if model:
        print("🎉 Model training completed successfully!")
