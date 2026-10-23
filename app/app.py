from __future__ import annotations
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

def load_and_predict(X: ArrayLike, filename: str = "linear_regression_model.joblib") -> ArrayLike:
    model = load(filename)
    y = model.predict(X)
    return y

def create_streamlit_app():
    st.title("House Price Prediction Model")
    st.write("Enter the housing parameters below to predict the price:")
    area = st.number_input("Area (sq ft):", min_value=1000, max_value=20000, value=5000, step=100)
    bedrooms = st.slider("Bedrooms:", min_value=1, max_value=10, value=3)
    bathrooms = st.slider("Bathrooms:", min_value=1, max_value=5, value=1)
    stories = st.slider("Stories:", min_value=1, max_value=4, value=2)

    if st.button("Predict value"):
        input_features = np.array([[area, bedrooms, bathrooms, stories]])
        
        prediction = load_and_predict(input_features)
        predicted_price = prediction[0]
        
        st.success(f"Predicted Price: ${predicted_price:,.2f}")
        
        visualize_difference(input_features, predicted_price)

def visualize_difference(input_features: np.ndarray, prediction: float):
    X_filename = "X.joblib"
    y_filename = "y.joblib"

    X = load(X_filename)
    y = load(y_filename)

    area_feature = input_features[0][0]
    
    closest_idx = _index_of_closest(X['area'].values if hasattr(X, 'columns') else X[:, 0], area_feature)
    actual_target = float(y.iloc[closest_idx] if hasattr(y, 'iloc') else y[closest_idx])

    difference = actual_target - prediction

    fig = plt.figure(figsize=(8, 5))

    areas = X['area'] if hasattr(X, 'columns') else X[:, 0]
    plt.scatter(areas, y, color='grey', alpha=0.5, label='Dataset')

    closest_area = areas.iloc[closest_idx] if hasattr(areas, 'iloc') else areas[closest_idx]
    plt.scatter(closest_area, actual_target, color='blue', s=100, label='Actual Target (Similar Area)')

    plt.scatter(area_feature, prediction, color='red', s=100, label='Predicted Target')

    plt.legend()
    plt.title("Prediction vs Actual Target (Based on Area)")
    plt.xlabel('Area (sq ft)')
    plt.ylabel('Price')
    plt.grid(True)

    plt.plot([area_feature, closest_area], [prediction, actual_target], 'k--')

    mid_x = (area_feature + closest_area) / 2
    mid_y = (prediction + actual_target) / 2
    plt.annotate(f'Difference = ${difference:,.2f}', 
                 xy=(mid_x, mid_y), 
                 xytext=(10, 10), 
                 textcoords='offset points',
                 fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8))

    st.pyplot(fig)

def _index_of_closest(X: ArrayLike, k: float) -> int:
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx

if __name__ == '__main__':
    create_streamlit_app()