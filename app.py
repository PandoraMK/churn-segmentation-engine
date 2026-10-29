from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="FNB Churn & Segmentation Engine",
    description="Real-time inference API for customer churn risk prediction and persona clustering.",
    version="1.0"
)

# Load saved production artifacts on startup
try:
    xgb_model = joblib.load("models/churn_engine_xgb.pkl")
    kmeans_model = joblib.load("models/churn_engine_kmeans.pkl")
    scaler = joblib.load("models/churn_engine_scaler.pkl")
    print("[INFO] Production models loaded successfully into memory.")
except Exception as e:
    print(f"[ERROR] Failed to load model artifacts: {e}")

# Define the expected JSON input schema using Pydantic
class CustomerInput(BaseModel):
    CreditScore: float
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography_Germany: int
    Geography_Spain: int
    Geography_France: int
    Gender_Female: int
    Gender_Male: int

@app.get("/")
def home():
    return {"status": "online", "message": "FNB Churn Engine API is running."}

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    try:
        # Convert incoming JSON payload to DataFrame
        input_data = pd.DataFrame([customer.dict()])
        
        # Ensure column order matches training matrix exactly
        feature_columns = [
            "CreditScore", "Age", "Tenure", "Balance", 
            "NumOfProducts", "HasCrCard", "IsActiveMember", 
            "EstimatedSalary", "Geography_Germany", "Geography_Spain", 
            "Geography_France", "Gender_Female", "Gender_Male"
        ]
        input_data = input_data[feature_columns]
        
        # 1. Run Supervised XGBoost Prediction
        churn_prob = float(xgb_model.predict_proba(input_data)[0][1])
        churn_prediction = int(xgb_model.predict(input_data)[0])
        
        # 2. Run Unsupervised k-Means Clustering Assignment
        continuous_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
        scaled_continuous = input_data[continuous_cols].copy()
        scaled_continuous[continuous_cols] = scaler.transform(scaled_continuous[continuous_cols])
        
        cluster_assignment = int(kmeans_model.predict(scaled_continuous)[0])
        
        return {
            "churn_prediction": churn_prediction,
            "churn_probability": round(churn_prob, 4),
            "assigned_cluster": cluster_assignment,
            "risk_status": "HIGH RISK" if churn_prob > 0.5 else "STABLE"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))