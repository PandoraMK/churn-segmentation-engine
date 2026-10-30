import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import os          # <--- Make sure this is here
import sqlite3     # <--- Make sure this is here
import os

# Page Configuration
st.set_page_config(page_title="Customer Churn Prediction Engine", page_icon="⚡", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_engine_xgb.pkl")
    scaler = joblib.load("models/churn_engine_scaler.pkl")
    kmeans = joblib.load("models/churn_engine_kmeans.pkl")
    return model, scaler, kmeans

model, scaler, kmeans = load_artifacts()

st.title("⚡ Real-Time Customer Churn & Behavior Analytics")
st.markdown("Predict customer churn probability and cluster behavioral segments using your production XGBoost & k-Means pipeline.")

# Sidebar Inputs for Customer Profile
st.sidebar.header("Customer Profile Parameters")
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
age = st.sidebar.slider("Age", 18, 92, 38)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance ($)", 0.0, 250000.0, 60000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.sidebar.selectbox("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
is_active = st.sidebar.selectbox("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
estimated_salary = st.sidebar.number_input("Estimated Salary ($)", 10000.0, 200000.0, 75000.0)

geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

# Process categorical features matching training schema
geo_france = 1 if geography == "France" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_female = 1 if gender == "Female" else 0
gender_male = 1 if gender == "Male" else 0

input_dict = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": estimated_salary,
    "Geography_France": geo_france,
    "Geography_Germany": geo_germany,
    "Geography_Spain": geo_spain,
    "Gender_Female": gender_female,
    "Gender_Male": gender_male
}

feature_columns = list(input_dict.keys())
continuous_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]

if st.button("Run Churn Prediction & Segmentation", type="primary"):
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    # Inference
    churn_prob = float(model.predict_proba(input_df)[0][1])
    churn_pred = int(model.predict(input_df)[0])
    
    scaled_cont = input_df[continuous_cols].copy()
    scaled_cont[continuous_cols] = scaler.transform(scaled_cont[continuous_cols])
    cluster = int(kmeans.predict(scaled_cont)[0])
    
    # Display Results Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Churn Probability", 
            value=f"{churn_prob:.2%}", 
            delta="High Risk" if churn_prob > 0.5 else "Low Risk"
        )
    with col2:
        st.metric(label="Predicted Status", value="Will Churn" if churn_pred == 1 else "Retained")
    with col3:
        st.metric(label="Behavioral Cluster Segment", value=f"Cluster #{cluster}")
        
    st.markdown("---")
    
    # Interactive Visualization: Customer Financial & Behavioral Profile Chart
    st.subheader("📊 Customer Profile & Risk Indicators")
    
    chart_data = pd.DataFrame({
        "Feature": ["Credit Score (scaled)", "Age (scaled)", "Tenure (Years)", "Balance ($10k)", "Products", "Salary ($10k)"],
        "Value": [
            credit_score / 850.0,
            age / 100.0,
            tenure / 10.0,
            balance / 250000.0,
            num_products / 4.0,
            estimated_salary / 200000.0
        ]
    })
    
    fig = px.bar(
        chart_data, 
        x="Value", 
        y="Feature", 
        orientation="h",
        title="Normalized Customer Feature Impact Vectors",
        labels={"Value": "Normalized Magnitude (0 to 1)", "Feature": "Customer Attribute"},
        color="Value",
        color_continuous_scale="Tealgrn"
    )
    fig.update_layout(xaxis_range=[0, 1], height=350, margin=dict(l=20, r=20, t=40, b=20))
    
    st.plotly_chart(fig, use_container_width=True)
    st.success("Inference executed successfully via model artifacts!")

    import os
import sqlite3

# --- Real-Time Pipeline Audit Logs Section ---
st.markdown("---")
st.subheader("📜 Real-Time Pipeline Audit Logs (Kafka & SQLite)")

DB_PATH = "churn_predictions.db"

if os.path.exists(DB_PATH):
    try:
        conn = sqlite3.connect(DB_PATH)
        log_df = pd.read_sql(
            "SELECT timestamp, customer_id, churn_probability, churn_prediction, cluster_segment FROM predictions ORDER BY id DESC LIMIT 20", 
            conn
        )
        conn.close()
        
        if not log_df.empty:
            # Quick summary metrics for the streaming stream
            col_l1, col_l2, col_l3 = st.columns(3)
            col_l1.metric("Recent Logged Events", len(log_df))
            col_l2.metric("Avg Churn Risk in Stream", f"{log_df['churn_probability'].mean():.2%}")
            col_l3.metric("High-Risk Alerts Detected", int((log_df['churn_probability'] > 0.5).sum()))
            
            # Interactive data frame
            st.dataframe(log_df, use_container_width=True)
            
            if st.button("🔄 Refresh Live Audit Logs"):
                st.rerun()
        else:
            st.info("Database table is active, but no streaming predictions have been logged yet. Run your Kafka producer to push live events!")
            
    except Exception as e:
        st.error(f"Error reading local audit database: {e}")
else:
    st.info(f"Audit database (`{DB_PATH}`) not found locally yet. Ensure your Kafka consumer container is running with the shared volume mapped.")