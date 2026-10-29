from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="FNB Multi-Model Churn & Segmentation Engine",
    description="API supporting multi-model inference across Logistic Regression, Random Forest, XGBoost, and k-Means.",
    version="2.0"
)

# Initialize global containers to prevent NameError
models = {}
kmeans_model = None
scaler = None

# Load model registry into memory on startup
try:
    models = {
        "logistic_regression": joblib.load("models/churn_engine_lr.pkl"),
        "random_forest": joblib.load("models/churn_engine_rf.pkl"),
        "xgboost": joblib.load("models/churn_engine_xgb.pkl")
    }
    kmeans_model = joblib.load("models/churn_engine_kmeans.pkl")
    scaler = joblib.load("models/churn_engine_scaler.pkl")
    print("[INFO] All production models loaded into registry.")
except Exception as e:
    print(f"[ERROR] Failed to load model artifacts: {e}")

class CustomerInput(BaseModel):
    model_type: str = "xgboost"
    CreditScore: float
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography_France: int
    Geography_Germany: int
    Geography_Spain: int
    Gender_Female: int
    Gender_Male: int

@app.get("/")
def home():
    return {"status": "online", "available_models": list(models.keys())}

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    if not models or kmeans_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model artifacts failed to load into memory on startup.")
        
    if customer.model_type not in models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model_type. Choose from: {list(models.keys())}"
        )
    
    try:
        data_dict = customer.dict()
        selected_model_type = data_dict.pop("model_type")
        
        input_data = pd.DataFrame([data_dict])
        
        feature_columns = [
            "CreditScore", "Age", "Tenure", "Balance", 
            "NumOfProducts", "HasCrCard", "IsActiveMember", 
            "EstimatedSalary", "Geography_France", "Geography_Germany", "Geography_Spain", 
            "Gender_Female", "Gender_Male"
        ]
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)
        
        # 1. Run Selected Supervised Model Prediction
        selected_model = models[selected_model_type]
        churn_prob = float(selected_model.predict_proba(input_data)[0][1])
        churn_prediction = int(selected_model.predict(input_data)[0])
        
        # 2. Run Unsupervised k-Means Clustering Assignment
        continuous_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
        scaled_continuous = input_data[continuous_cols].copy()
        scaled_continuous[continuous_cols] = scaler.transform(scaled_continuous[continuous_cols])
        cluster_assignment = int(kmeans_model.predict(scaled_continuous)[0])
        
        return {
            "model_used": selected_model_type,
            "churn_prediction": churn_prediction,
            "churn_probability": round(churn_prob, 4),
            "assigned_cluster": cluster_assignment,
            "risk_status": "HIGH RISK" if churn_prob > 0.5 else "STABLE"
        }
        
    except Exception as e:
        import traceback
        print("[ERROR TRACEBACK]:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))