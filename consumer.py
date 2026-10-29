import os
import json
import joblib
import pandas as pd
from kafka import KafkaConsumer

# 1. Load Model Artifacts Once at Startup
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

# 2. Dynamic Bootstrap Servers (Reads from Docker environment or defaults to localhost)
raw_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
bootstrap_servers = [s.strip() for s in raw_servers.split(",")]

print(f"[INFO] Connecting Kafka consumer to brokers: {bootstrap_servers}")

# 3. Initialize Kafka Consumer
consumer = KafkaConsumer(
    'customer-events',
    bootstrap_servers=bootstrap_servers,
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='churn-inference-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("[INFO] Churn prediction event consumer active and listening to topic: 'customer-events'...")

# 4. Stream Processing Loop
for message in consumer:
    event_data = message.value
    customer_id = event_data.get("CustomerId", "UNKNOWN")
    
    try:
        input_data = pd.DataFrame([event_data])
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)
        
        churn_prob = float(model.predict_proba(input_data)[0][1])
        churn_pred = int(model.predict(input_data)[0])
        
        scaled_continuous = input_data[continuous_cols].copy()
        scaled_continuous[continuous_cols] = scaler.transform(scaled_continuous[continuous_cols])
        cluster = int(kmeans.predict(scaled_continuous)[0])
        
        print(f"[PROCESSED] Customer {customer_id} - Score: {churn_prob:.4f} | Cluster: {cluster}")
        
    except Exception as e:
        print(f"[ERROR] Failed to process event for customer {customer_id}: {e}")