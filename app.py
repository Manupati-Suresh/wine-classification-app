
import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Wine Classification",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Wine Type Classification")
st.markdown("Classify wine types based on chemical analysis using machine learning.")

@st.cache_resource
def load_or_train_model():
    """Load existing model or train a new one if not found."""
    model_path = "random_forest_model.pkl"
    
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.warning(f"⚠️ Error loading model: {e}. Training new model...")
    
    # Train new model if loading fails
    with st.spinner("🔄 Training model... This may take a moment."):
        data = load_wine()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        try:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            st.success("✅ Model trained and saved successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not save model: {e}")
        
        return model

# Load model
model = load_or_train_model()

# Get feature names from the wine dataset
wine_data = load_wine()
feature_names = wine_data.feature_names
class_names = wine_data.target_names

# Feature ranges for validation (approximate from wine dataset)
feature_ranges = {
    'alcohol': (11.0, 15.0),
    'malic_acid': (0.7, 5.8),
    'ash': (1.4, 3.2),
    'alcalinity_of_ash': (10.0, 30.0),
    'magnesium': (70, 162),
    'total_phenols': (0.98, 3.88),
    'flavanoids': (0.34, 5.08),
    'nonflavanoid_phenols': (0.13, 0.66),
    'proanthocyanins': (0.41, 3.58),
    'color_intensity': (1.28, 13.0),
    'hue': (0.48, 1.71),
    'od280/od315_of_diluted_wines': (1.27, 4.0),
    'proline': (278, 1680)
}

# Default values (approximate means from dataset)
defaults = [13.0, 2.34, 2.36, 19.5, 99.7, 2.29, 2.03, 0.36, 1.59, 5.06, 0.96, 2.61, 746]

st.subheader("🧪 Enter Wine Chemical Analysis Values")
st.markdown("*Adjust the sliders or input fields to match your wine sample's chemical analysis.*")

input_vals = []
cols = st.columns(2)

for i, (feat, default) in enumerate(zip(feature_names, defaults)):
    col = cols[i % 2]
    
    with col:
        # Clean feature name for display
        display_name = feat.replace('_', ' ').title()
        
        # Get range for this feature
        min_val, max_val = feature_ranges.get(feat, (0.0, 100.0))
        
        # Create input with validation
        val = st.number_input(
            f"{display_name}:",
            min_value=float(min_val * 0.5),  # Allow some flexibility
            max_value=float(max_val * 1.5),
            value=float(default),
            step=0.01,
            help=f"Typical range: {min_val:.2f} - {max_val:.2f}",
            key=f"input_{feat}"
        )
        
        # Validation warning
        if val < min_val or val > max_val:
            st.warning(f"⚠️ Value outside typical range ({min_val:.2f} - {max_val:.2f})")
        
        input_vals.append(val)

# Create input dataframe
input_df = pd.DataFrame([input_vals], columns=feature_names)

# Prediction section
st.subheader("🔮 Prediction")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🍇 Predict Wine Class", type="primary", use_container_width=True)

if predict_button:
    try:
        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        # Display results
        st.success(f"🍷 **Predicted Wine Class: {class_names[prediction]}**")
        
        # Show confidence scores
        st.subheader("📊 Confidence Scores")
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            st.metric(
                label=f"Class {i+1}: {class_name}",
                value=f"{prob:.1%}",
                delta=None
            )
            st.progress(prob)
        
        # Feature importance (top 5)
        if hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Most Important Features")
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)
            
            for _, row in importance_df.iterrows():
                st.write(f"• **{row['Feature'].replace('_', ' ').title()}**: {row['Importance']:.3f}")
                
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("Please check your input values and try again.")

# Information section
with st.expander("ℹ️ About This App"):
    st.markdown("""
    This app uses a Random Forest classifier trained on the Wine Recognition dataset 
    to predict wine types based on chemical analysis results.
    
    **Wine Classes:**
    - **Class 1**: Wine from cultivar 1
    - **Class 2**: Wine from cultivar 2  
    - **Class 3**: Wine from cultivar 3
    
    **Dataset**: 178 wine samples with 13 chemical features each.
    
    **Model**: Random Forest with 100 trees, achieving high accuracy on this well-separated dataset.
    """)
