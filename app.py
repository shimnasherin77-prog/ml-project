import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)

# Load model
model = joblib.load("diamond_price_model.pkl")

# Sidebar
st.sidebar.header("🔧 About the App")
st.sidebar.write(
    """
    This app predicts the **diamond price**
    based on the **carat value** using
    a **Linear Regression model**.
    """
)

# Main title
st.markdown(
    "<h1 style='text-align: center;'>💎 Diamond Price Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Enter the carat value to get an estimated diamond price</p>",
    unsafe_allow_html=True
)

st.divider()

# Input section
st.subheader("📥 Input Details")

carat = st.slider(
    "Select carat value",
    min_value=0.1,
    max_value=10.0,
    step=0.1,
    value=0.5
)

# Prediction section
st.subheader("📊 Prediction")

if st.button("🔮 Predict Price", use_container_width=True):
    input_df = pd.DataFrame([[carat]], columns=['carat'])
    prediction = model.predict(input_df)

    st.success(f"💰 **Estimated Price:** ${prediction[0]:.2f}")

st.divider()


# Footer
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>"
    "Built with ❤️ using Streamlit & Machine Learning"
    "</p>",
    unsafe_allow_html=True
)