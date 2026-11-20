
import streamlit as st
import joblib
import pandas as pd
from genai import generate_explanation

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details to get an ML prediction and AI explanation.")

# Load trained model
data = joblib.load("models/heart_model.pkl")

model = data["model"]
scaler = data["scaler"]
feature_names = data["feature_names"]

st.subheader("Patient Information")

age = st.number_input("Age", 1, 100, 45)
resting_bp = st.number_input("Resting Blood Pressure", 0, 250, 140)
cholesterol = st.number_input("Cholesterol", 0, 600, 240)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
max_hr = st.number_input("Maximum Heart Rate", 0, 250, 150)
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.5)

sex = st.selectbox("Sex", ["Female", "Male"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
exercise_angina = st.selectbox("Exercise Angina", ["No", "Yes"])
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):

    patient_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "Male" else 0,
        "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
        "ChestPainType_TA": 1 if chest_pain == "TA" else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0
    }

    input_df = pd.DataFrame([patient_data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    scaled_data = scaler.transform(input_df)
    prediction = int(model.predict(scaled_data)[0])

    st.subheader("ML Prediction")
    st.write("Predicted class:", prediction)

    with st.spinner("Generating AI explanation..."):
        explanation = generate_explanation(prediction, patient_data)

    st.subheader("GenAI Explanation")
    st.write(explanation)

    st.warning("This is an educational prediction, not a medical diagnosis.")
