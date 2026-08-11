import streamlit as st
import pandas as pd
import numpy as np
import pickle

MODEL_PATH = "Linear_regression.pkl"
DATA_PATH = "clean_data.xls"
CURRENT_YEAR = 2026

@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH, engine="xlrd")

@st.cache_data
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

st.title("Car Price Predictor")
st.write("Enter your car details below and click Predict.")

try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error("Failed to load model or data. Check that `clean_data.xls` and the pickle file are present.")
    st.stop()

companies = sorted(df["company"].dropna().unique())
years = sorted(df["year"].dropna().astype(int).unique(), reverse=True)
fuel_types = sorted(df["fuel_type"].dropna().unique())

company = st.selectbox("Company", [""] + companies)
available_models = sorted(df[df["company"] == company]["name"].dropna().unique()) if company else []
car_model = st.selectbox("Model", [""] + available_models)
year = st.selectbox("Year of Purchase", [""] + [str(y) for y in years])
fuel_type = st.selectbox("Fuel Type", [""] + fuel_types)
kms_driven = st.number_input("Kilometers Driven", min_value=0, value=10000, step=1000)

if st.button("Predict Price"):
    if not company or not car_model or not year or not fuel_type:
        st.error("Please select all car details before predicting.")
    else:
        car_age = CURRENT_YEAR - int(year)
        input_df = pd.DataFrame(
            [[car_model, company, fuel_type, kms_driven, car_age]],
            columns=["name", "company", "fuel_type", "kms_driven", "car_age"],
        )
        prediction = model.predict(input_df)
        try:
            predicted_price = np.expm1(prediction[0])
        except Exception:
            predicted_price = prediction[0]
        st.success(f"Estimated Price: Rs {predicted_price:,.2f}")
