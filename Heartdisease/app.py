
import joblib
import pandas as pd
from genai import generate_explanation

# Load trained KNN model
data = joblib.load("models/heart_model.pkl")

model = data["model"]
scaler = data["scaler"]
feature_names = data["feature_names"]

# Patient data
patient_data = {
    "Age": 45,
    "RestingBP": 140,
    "Cholesterol": 240,
    "FastingBS": 0,
    "MaxHR": 150,
    "Oldpeak": 1.5,
    "Sex_M": 1,
    "ChestPainType_ATA": 0,
    "ChestPainType_NAP": 1,
    "ChestPainType_TA": 0,
    "RestingECG_Normal": 1,
    "RestingECG_ST": 0,
    "ExerciseAngina_Y": 0,
    "ST_Slope_Flat": 0,
    "ST_Slope_Up": 1
}

# Convert data into DataFrame
input_df = pd.DataFrame([patient_data])

# Arrange features in training order
input_df = input_df.reindex(
    columns=feature_names,
    fill_value=0
)

# Scale input
scaled_data = scaler.transform(input_df)

# ML prediction
prediction = int(model.predict(scaled_data)[0])

print("ML Prediction:", prediction)

# GenAI explanation
explanation = generate_explanation(
    prediction,
    patient_data
)

print("\nGenAI Explanation:")
print(explanation)

print("\nNote: This is not a medical diagnosis.")

