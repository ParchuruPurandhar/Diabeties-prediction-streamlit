import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction System")

st.markdown(
"""
Enter patient details below to predict diabetes risk.
"""
)

preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 300, 100)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 1000, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input(
    "Diabetes Pedigree Function",
    0.0,
    3.0,
    0.5
)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):

    input_data = np.array([[
        preg,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Diabetic")
    else:
        st.success("✅ Non-Diabetic")

    st.metric(
        "Risk Probability",
        f"{probability*100:.2f}%"
    )

    if probability < 0.3:
        st.success("Low Risk")
    elif probability < 0.7:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")