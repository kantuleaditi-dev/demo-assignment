import streamlit as st
import pandas as pd
import joblib
# Load Models
heart_model = joblib.load("best_classification_model.pkl")
heart_scaler = joblib.load("heart_scaler.pkl")
heart_columns = joblib.load("heart_columns.pkl")
house_model = joblib.load("best_regression_model.pkl")
house_scaler = joblib.load("house_scaler.pkl")
house_columns = joblib.load("house_columns.pkl")
st.title("Assignment 24 - Multi Model Prediction App")
problem = st.selectbox(
    "Select Problem Type",
    ["Classification", "Regression"]
)
# ==========================
# Classification
# ==========================
if problem == "Classification":
    st.header("Heart Disease Prediction")
    input_data = {}
    for col in heart_columns:
        input_data[col] = st.number_input(col, value=0.0)
    if st.button("Predict Heart Disease"):
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=heart_columns, fill_value=0)
        input_scaled = heart_scaler.transform(input_df)
        prediction = heart_model.predict(input_scaled)
        if prediction[0] == 1:
            st.success("Heart Disease Detected")
        else:
            st.success("No Heart Disease")
# ==========================
# Regression
# ==========================
else:
    st.header("House Price Prediction")
    input_data = {}
    for col in house_columns:
        input_data[col] = st.number_input(col, value=0.0)
    if st.button("Predict House Price"):
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=house_columns, fill_value=0)
        input_scaled = house_scaler.transform(input_df)
        prediction = house_model.predict(input_scaled)
        st.success(f"Predicted House Price : {prediction[0]}")