import json
import joblib
import pandas as pd
from kafka import KafkaConsumer

# 1. Load Model Artifacts Once at Startup (Singleton Pattern)
print("[INFO] Loading production models into memory for event streaming...")
model = joblib.load("models/churn_engine_xgb.pkl")
scaler = joblib.load("models/churn_engine_scaler.pkl")
kmeans = joblib.load("models/churn_engine_kmeans.pkl")

feature_columns = [
    "CreditScore", "Age", "Tenure", "Balance", 
    "NumOfProducts", "HasCrCard", "IsActiveMember", 
    "EstimatedSalary", "Geography_France", "Geography_Germany", "Geography_Spain", 
    "Gender_Female", "Gender_Male"
]
continuous_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]

# 2. Initialize Kafka Consumer
consumer = KafkaConsumer(
    'customer-events',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='churn-inference-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("[INFO] Churn prediction event consumer active and listening to topic: 'customer-events'...")

# 3. Stream Processing Loop
for message in consumer:
    event_data = message.value
    customer_id = event_data.get("CustomerId", "UNKNOWN")
    
    try:
        # Extract features and align columns
        input_data = pd.DataFrame([event_data])
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)
        
        # Run XGBoost Churn Prediction
        churn_prob = float(model.predict_proba(input_data)[0][1])
        churn_pred = int(model.predict(input_data)[0])
        
        # Run k-Means Persona Clustering
        scaled_continuous = input_data[continuous_cols].copy()
        scaled_continuous[continuous_cols] = scaler.transform(scaled_continuous[continuous_cols])
        cluster = int(kmeans.predict(scaled_continuous)[0])
        
        # Construct scoring result
        result = {
            "customer_id": customer_id,
            "churn_probability": round(churn_prob, 4),
            "churn_prediction": churn_pred,
            "assigned_cluster": cluster,
            "risk_status": "HIGH RISK" if churn_prob > 0.5 else "STABLE"
        }
        
        # Action: Print alert or write to downstream data warehouse / notification queue
        if churn_prob > 0.7:
            print(f"[HIGH RISK ALERT] Customer {customer_id} flagged with {churn_prob:.2%} churn probability! Cluster: {cluster}")
        else:
            print(f"[PROCESSED] Customer {customer_id} - Score: {churn_prob:.4f}")
            
    except Exception as e:
        print(f"[ERROR] Failed to process event for customer {customer_id}: {e}")