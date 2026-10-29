import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

class ChurnPipelineEngine:
    def __init__(self, data_path="data/Churn_Modelling.csv"):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.lr_model = None
        self.rf_model = None
        self.xgb_model = None

    def load_and_preprocess(self):
        print("[INFO] Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Drop irrelevant columns
        drop_cols = ['RowNumber', 'CustomerId', 'Surname']
        df_clean = self.df.drop(columns=[c for c in drop_cols if c in self.df.columns])
        
        # One-hot encode categorical features (Geography, Gender)
        df_encoded = pd.get_dummies(df_clean, columns=['Geography', 'Gender'], drop_first=False)
        
        # Split features and target
        self.X = df_encoded.drop(columns=['Exited'])
        self.y = df_encoded['Exited']
        
        # Train unsupervised k-Means clusterer on continuous features
        print("[INFO] Training k-Means clustering persona segmenter...")
        continuous_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
        scaled_continuous = self.scaler.fit_transform(self.X[continuous_cols])
        
        self.kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.kmeans_model.fit(scaled_continuous)
        
        return self

    def train_all_models(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # 1. Logistic Regression (Balanced)
        print("[INFO] Training Logistic Regression...")
        self.lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state)
        self.lr_model.fit(X_train, y_train)
        
        # 2. Random Forest
        print("[INFO] Training Random Forest...")
        self.rf_model = RandomForestClassifier(random_state=random_state)
        self.rf_model.fit(X_train, y_train)
        
        # 3. XGBoost
        print("[INFO] Training XGBoost...")
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        self.xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=random_state, eval_metric="logloss")
        self.xgb_model.fit(X_train, y_train)
        
        return self

    def save_all_artifacts(self, prefix="churn_engine"):
        os.makedirs("models", exist_ok=True)
        
        joblib.dump(self.lr_model, f"models/{prefix}_lr.pkl")
        joblib.dump(self.rf_model, f"models/{prefix}_rf.pkl")
        joblib.dump(self.xgb_model, f"models/{prefix}_xgb.pkl")
        joblib.dump(self.kmeans_model, f"models/{prefix}_kmeans.pkl")
        joblib.dump(self.scaler, f"models/{prefix}_scaler.pkl")
        
        print("[SUCCESS] All model artifacts, scaler, and clusterer saved successfully inside /models/.")

if __name__ == "__main__":
    pipeline = ChurnPipelineEngine()
    pipeline.load_and_preprocess()
    pipeline.train_all_models()
    pipeline.save_all_artifacts()