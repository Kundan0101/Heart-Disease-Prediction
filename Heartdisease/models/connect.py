import joblib

# Load saved ML model
data = joblib.load("models/heart_model.pkl")

model = data["model"]
scaler = data["scaler"]

print("ML model connected successfully!")