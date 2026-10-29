import requests

payload = {
    "model_type": "xgboost",
    "CreditScore": 600,
    "Age": 42,
    "Tenure": 3,
    "Balance": 125000.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 50000.0,
    "Geography_France": 1,
    "Geography_Germany": 0,
    "Geography_Spain": 0,
    "Gender_Female": 1,
    "Gender_Male": 0
}

# Test Random Forest
payload["model_type"] = "random_forest"

# Test Logistic Regression
payload["model_type"] = "logistic_regression"

response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print("API Response Status:", response.status_code)
print("Raw Response Text:", repr(response.text))

if response.status_code == 200:
    print("API Response JSON:", response.json())