import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import load_iris
# Load model
model = joblib.load("best_model.pkl")
# Load iris dataset
iris = load_iris()
st.set_page_config(page_title="Iris Flower Prediction")
st.title("🌸 Iris Flower Prediction App")
st.write("Enter the flower measurements below:")
# User Input
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=4.0,
    max_value=8.0,
    value=5.1
)
sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=2.0,
    max_value=5.0,
    value=3.5
)
petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=1.0,
    max_value=7.0,
    value=1.4
)
petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.1,
    max_value=3.0,
    value=0.2
)
if st.button("Predict"):

    sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=iris.feature_names
    )
    prediction = model.predict(sample)[0]
    flower = iris.target_names[prediction]
    st.success(f"Predicted Flower: {flower}")
    st.write("Input Values")
    st.dataframe(sample)