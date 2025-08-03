
import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

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

# Sample wine profiles for quick testing
sample_wines = {
    "🍷 Typical Red Wine": [13.2, 2.1, 2.4, 18.5, 95, 2.3, 2.1, 0.35, 1.8, 5.2, 1.0, 2.8, 720],
    "🍾 Light White Wine": [11.5, 1.8, 2.1, 15.0, 85, 1.9, 1.2, 0.25, 1.2, 3.5, 1.2, 3.2, 450],
    "🥂 Full-bodied Red": [14.8, 2.8, 2.8, 22.0, 110, 3.2, 3.5, 0.45, 2.5, 8.5, 0.8, 2.2, 1200]
}

# Wine class descriptions
wine_descriptions = {
    'class_0': "🍷 **Cultivar 1**: Typically full-bodied wines with high color intensity and robust flavor profile. Often characterized by higher alcohol content and pronounced tannins.",
    'class_1': "🍾 **Cultivar 2**: Medium-bodied wines with well-balanced characteristics. These wines often show moderate color intensity with harmonious flavor compounds.",
    'class_2': "🥂 **Cultivar 3**: Light to medium-bodied wines with distinctive flavor profiles. Often featuring unique aromatic compounds and elegant structure."
}

# Initialize session state for prediction history
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

st.subheader("🧪 Enter Wine Chemical Analysis Values")

# Quick sample buttons
st.markdown("**🎯 Quick Test Samples:**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🍷 Red Wine", help="Load typical red wine values"):
        st.session_state.sample_values = sample_wines["🍷 Typical Red Wine"]
        st.rerun()

with col2:
    if st.button("🍾 White Wine", help="Load light white wine values"):
        st.session_state.sample_values = sample_wines["🍾 Light White Wine"]
        st.rerun()

with col3:
    if st.button("🥂 Full-bodied", help="Load full-bodied wine values"):
        st.session_state.sample_values = sample_wines["🥂 Full-bodied Red"]
        st.rerun()

with col4:
    if st.button("🔄 Reset", help="Reset to default values"):
        if 'sample_values' in st.session_state:
            del st.session_state.sample_values
        st.rerun()

st.markdown("*Adjust the values below or use quick samples above to test different wine profiles.*")

input_vals = []
cols = st.columns(2)

for i, (feat, default) in enumerate(zip(feature_names, defaults)):
    col = cols[i % 2]
    
    with col:
        # Clean feature name for display
        display_name = feat.replace('_', ' ').title()
        
        # Get range for this feature
        min_val, max_val = feature_ranges.get(feat, (0.0, 100.0))
        
        # Use sample values if available
        current_value = default
        if 'sample_values' in st.session_state:
            current_value = st.session_state.sample_values[i]
        
        # Create input with validation
        val = st.number_input(
            f"{display_name}:",
            min_value=float(min_val * 0.5),  # Allow some flexibility
            max_value=float(max_val * 1.5),
            value=float(current_value),
            step=0.01,
            help=f"Typical range: {min_val:.2f} - {max_val:.2f}",
            key=f"input_{feat}_{current_value}"  # Include value in key to force update
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
        with st.spinner('🔬 Analyzing wine composition...'):
            # Make prediction
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            
            # Store prediction in history
            st.session_state.predictions.append({
                'timestamp': datetime.now(),
                'prediction': class_names[prediction],
                'confidence': probabilities[prediction],
                'all_probabilities': probabilities.tolist()
            })
        
        # Display results
        st.success(f"🍷 **Predicted Wine Class: {class_names[prediction]}**")
        
        # Show wine description
        st.info(wine_descriptions[class_names[prediction]])
        
        # Show confidence scores
        st.subheader("📊 Confidence Scores")
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(prob, text=f"{class_name}: {prob:.1%}")
            with col2:
                if i == prediction:
                    st.markdown("**✅ Predicted**")
                else:
                    st.markdown(f"{prob:.1%}")
        
        # Feature importance (top 5)
        if hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Most Important Features for Classification")
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': input_vals,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)
            
            for _, row in importance_df.iterrows():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{row['Feature'].replace('_', ' ').title()}**")
                with col2:
                    st.write(f"Value: {row['Value']:.2f}")
                with col3:
                    st.write(f"Importance: {row['Importance']:.3f}")
        
        # Download results
        st.subheader("📥 Export Analysis")
        results_data = {
            'Feature': feature_names,
            'Value': input_vals,
            'Importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else [0] * len(feature_names),
            'Prediction': [class_names[prediction]] * len(feature_names),
            'Confidence': [f"{probabilities[prediction]:.1%}"] * len(feature_names)
        }
        results_df = pd.DataFrame(results_data)
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Wine Analysis Report",
            data=csv,
            file_name=f"wine_analysis_{class_names[prediction]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download detailed analysis including all features, values, and model insights"
        )
                
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("Please check your input values and try again.")

# Prediction History
if st.session_state.predictions:
    with st.expander("📈 Recent Predictions History", expanded=False):
        st.markdown("**Last 5 Predictions:**")
        recent_predictions = st.session_state.predictions[-5:]
        
        for i, pred in enumerate(reversed(recent_predictions)):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            with col1:
                st.write(f"🕐 {pred['timestamp'].strftime('%H:%M:%S')}")
            with col2:
                st.write(f"🍷 {pred['prediction']}")
            with col3:
                st.write(f"📊 {pred['confidence']:.1%}")
            with col4:
                if st.button("🗑️", key=f"delete_{len(recent_predictions)-i}", help="Remove this prediction"):
                    st.session_state.predictions.remove(pred)
                    st.rerun()
        
        if len(st.session_state.predictions) > 1:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Show Statistics"):
                    predictions_df = pd.DataFrame(st.session_state.predictions)
                    st.write("**Prediction Summary:**")
                    st.write(predictions_df['prediction'].value_counts())
                    st.write(f"**Average Confidence:** {predictions_df['confidence'].mean():.1%}")
            with col2:
                if st.button("🗑️ Clear All History"):
                    st.session_state.predictions = []
                    st.rerun()

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
