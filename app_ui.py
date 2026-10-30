import streamlit as st
import pandas as pd
import joblib
import numpy as np

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
    
    # Display Results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Churn Probability", value=f"{churn_prob:.2%}", delta="High Risk" if churn_prob > 0.5 else "Low Risk")
    with col2:
        st.metric(label="Predicted Status", value="Will Churn" if churn_pred == 1 else "Retained")
    with col3:
        st.metric(label="Behavioral Cluster Segment", value=f"Cluster #{cluster}")
        
    st.success("Inference executed successfully via model artifacts!")