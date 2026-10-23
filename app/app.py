import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from joblib import load

def load_model_and_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, "linear_regression_model.joblib")
    X_path = os.path.join(current_dir, "X.joblib")
    y_path = os.path.join(current_dir, "y.joblib")

    model = load(model_path)
    X = load(X_path)
    y = load(y_path)
    
    return model, X, y

def create_streamlit_app():
    st.set_page_config(
        page_title="House Price Prediction",
        page_icon="🏠",
        layout="wide"
    )

    st.title("🏠 House Price Prediction Model")
    st.markdown(
        """
        This application predicts **house prices** based on physical parameters like area, 
        number of bedrooms, bathrooms, and stories using a trained Linear Regression model.
        """
    )

    try:
        model, X, y = load_model_and_data()
    except FileNotFoundError:
        st.error("Model or data files not found. Please run `python model.py` first!")
        return

    st.sidebar.header("Input Features")

    area_val = st.sidebar.slider(
        "Area (sq ft)",
        min_value=1500,
        max_value=17000,
        value=5000,
        step=100
    )

    bedrooms_val = st.sidebar.slider(
        "Bedrooms",
        min_value=1,
        max_value=6,
        value=3
    )
    
    bathrooms_val = st.sidebar.slider(
        "Bathrooms",
        min_value=1,
        max_value=4,
        value=1
    )
    
    stories_val = st.sidebar.slider(
        "Stories",
        min_value=1,
        max_value=4,
        value=2
    )

    if st.sidebar.button("Calculate Prediction", type="primary"):
        input_features = np.array([[area_val, bedrooms_val, bathrooms_val, stories_val]])

        pred_price = float(model.predict(input_features)[0])

        price_per_sqft = pred_price / area_val if area_val > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted House Price", f"${pred_price:,.2f}")
        col2.metric("Input Area", f"{area_val} sq ft")
        col3.metric("Price per sq ft", f"${price_per_sqft:,.2f}")

        fig, ax = plt.subplots(figsize=(10, 5))

        areas = X['area'].values if hasattr(X, 'columns') else X[:, 0]
        targets = y.values if hasattr(y, 'name') else y

        ax.scatter(
            areas,
            targets,
            color="gray",
            alpha=0.6,
            label="Historical Data (Dataset)"
        )

        area_range = np.linspace(areas.min(), areas.max() * 1.1, 150)
        
        line_input = np.column_stack((
            area_range,
            np.full_like(area_range, bedrooms_val),
            np.full_like(area_range, bathrooms_val),
            np.full_like(area_range, stories_val)
        ))
        
        predicted_line = model.predict(line_input)

        ax.plot(
            area_range,
            predicted_line,
            color="green",
            linewidth=2,
            label=f"Linear Trend (Bedrooms={bedrooms_val}, Bath={bathrooms_val})"
        )

        ax.scatter(
            [area_val],
            [pred_price],
            color="red",
            s=120,
            zorder=5,
            label=f"Your Prediction: ${pred_price:,.0f}"
        )

        ax.axvline(area_val, color="red", linestyle=":", alpha=0.5)
        ax.axhline(pred_price, color="red", linestyle=":", alpha=0.5)

        ax.set_title("Pricing Model: Area vs House Price", fontsize=13)
        ax.set_xlabel("Area (sq ft)", fontsize=11)
        ax.set_ylabel("Price ($)", fontsize=11)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)
    else:
        st.info("Select parameters in the sidebar on the left and click **'Calculate Prediction'**.")

if __name__ == "__main__":
    create_streamlit_app()
