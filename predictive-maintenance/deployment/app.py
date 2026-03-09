import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ---------------------------
# Load model
# ---------------------------

model_path = hf_hub_download(
    repo_id="ksricheenu/vehicle-engine-failure-prediction-adaboost",
    filename="vehicle_engine_failure_adaboost_model.pkl"
)

model = joblib.load(model_path)

st.title("Vehicle breakdown & Engine failures - Predictive Maintenance")

# ---------------------------
# User Inputs
# ---------------------------
engine_rpm = st.number_input("Engine Rpm", 500, 8000, 600)
lub_oil_pressure = st.number_input("Lub Oil Pressure", 0.00,20.99,0.00)
fuel_pressure = st.number_input("Fuel Pressure", 0.00,20.99,50.00)
coolant_pressure = st.number_input("Coolant Pressure", 0.00,20.00,0.00)
lube_oil_temp = st.number_input("Lub Oil Temperature", 60.00,200.99,60.00)
coolant_temp = st.number_input("Coolant Temperature", 60.00,300.00,60.00)

# ---------------------------
# Assemble EXACT training schema
# ---------------------------

input_data = {
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lube_oil_temp,
    "Coolant temp": coolant_temp
}

df = pd.DataFrame([input_data])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Vehicle & Engine predictive Maintenance"):
    prediction = model.predict(df)[0]
    result = "Engine condition is Abnormal or Faulty" if prediction == 1 else "Engine condition is Good or Normal"
    st.success(result)
