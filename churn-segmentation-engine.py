from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# (Inside your ChurnPipelineEngine class, add methods for LR and RF)
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
        joblib.dump(self.lr_model, f"models/{prefix}_lr.pkl")
        joblib.dump(self.rf_model, f"models/{prefix}_rf.pkl")
        joblib.dump(self.xgb_model, f"models/{prefix}_xgb.pkl")
        joblib.dump(self.kmeans_model, f"models/{prefix}_kmeans.pkl")
        joblib.dump(self.scaler, f"models/{prefix}_scaler.pkl")
        print("[SUCCESS] All model artifacts saved successfully.")