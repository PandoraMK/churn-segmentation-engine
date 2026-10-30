import json
import sqlite3
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from kafka import KafkaConsumer

# 1. Initialize SQLite Database Connection & Table
DB_NAME = "churn_predictions.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            customer_id INTEGER,
            churn_probability REAL,
            churn_prediction INTEGER,
            cluster_segment INTEGER,
            raw_payload TEXT
        )
    """)
    conn.commit()
    conn.close()
    print(f"[INFO] Local SQLite database '{DB_NAME}' initialized successfully.")

init_db()

# 2. Load Model Artifacts
print("[INFO] Loading production model artifacts...")
model = joblib.load("models/churn_engine_xgb.pkl")
scaler = joblib.load("models/churn_engine_scaler.pkl")
kmeans = joblib.load("models/churn_engine_kmeans.pkl")

continuous_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
feature_columns = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", 
    "HasCrCard", "IsActiveMember", "EstimatedSalary", 
    "Geography_France", "Geography_Germany", "Geography_Spain", 
    "Gender_Female", "Gender_Male"
]

# 3. Initialize Kafka Consumer
topic_name = 'customer-events'
consumer = KafkaConsumer(
    topic_name,
    bootstrap_servers=['kafka:9092'],  # Use 'localhost:9092' if running outside Docker
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='churn-consumer-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print(f"[INFO] Listening for real-time events on topic '{topic_name}'...")

# Database connection for the consumer loop
db_conn = sqlite3.connect(DB_NAME, check_same_thread=False)
db_cursor = db_conn.cursor()

try:
    for message in consumer:
        event_data = message.value
        customer_id = event_data.get("CustomerId", 0)
        
        # Prepare DataFrame for Inference
        input_df = pd.DataFrame([event_data])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        
        # Run ML Inference
        churn_prob = float(model.predict_proba(input_df)[0][1])
        churn_pred = int(model.predict(input_df)[0])
        
        scaled_cont = input_df[continuous_cols].copy()
        scaled_cont[continuous_cols] = scaler.transform(scaled_cont[continuous_cols])
        cluster = int(kmeans.predict(scaled_cont.values)[0])
        
        # Log to Console
        status_label = "WILL CHURN" if churn_pred == 1 else "RETAINED"
        print(f"[CONSUMER] Customer {customer_id} | Churn Prob: {churn_prob:.2%} ({status_label}) | Cluster: #{cluster}")
        
        # Save to SQLite Database
        timestamp = datetime.utcnow().isoformat()
        db_cursor.execute("""
            INSERT INTO predictions (timestamp, customer_id, churn_probability, churn_prediction, cluster_segment, raw_payload)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, customer_id, churn_prob, churn_pred, cluster, json.dumps(event_data)))
        db_conn.commit()

except KeyboardInterrupt:
    print("\n[INFO] Shutting down consumer worker...")
finally:
    db_conn.close()