# 🍷 Wine Classification App

A Streamlit web application that classifies wine types based on chemical analysis using machine learning.

## Features

- **Interactive UI**: Easy-to-use sliders and input fields for wine chemical properties
- **Real-time Prediction**: Instant classification with confidence scores
- **Robust Model Loading**: Automatically trains model if not found
- **Input Validation**: Warns users about values outside typical ranges
- **Feature Importance**: Shows which chemical properties are most important for classification

## Dataset

Uses the Wine Recognition dataset with 178 wine samples from 3 different cultivars, each with 13 chemical features:

- Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium
- Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins
- Color intensity, Hue, OD280/OD315 of diluted wines, Proline

## Deployment

### Streamlit Cloud
1. Push this repository to GitHub
2. Connect your GitHub account to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from your repository

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model Performance

The Random Forest classifier achieves high accuracy on this well-separated dataset, making it ideal for wine classification tasks.