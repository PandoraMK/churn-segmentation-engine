# FNB Churn Prediction & Customer Segmentation Engine

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red.svg)](https://xgboost.readthedocs.io/)

A modular, production-ready machine learning backend providing real-time customer churn risk scoring and behavioral persona segmentation. Built with an extensible model registry pattern that allows dynamic routing across multiple supervised classifiers while simultaneously running unsupervised clustering.

---

## ystem Architecture & Workflow

The pipeline is engineered following strict software design and data engineering principles, separating data ingestion, preprocessing, multi-model training, artifact serialization, and API inference.

### **Core Components:**
1. **Data Engineering (`ChurnPipelineEngine`):** Handles missing value checks, column dropping (`RowNumber`, `CustomerId`, `Surname`), robust one-hot encoding for categorical variables (`Geography`, `Gender`), and strict feature alignment matrices.
2. **Supervised Model Registry:** Trains, evaluates, and serializes three distinct benchmark classifiers to compare performance:
   - **Logistic Regression** (with class balancing enabled)
   - **Random Forest Classifier**
   - **XGBoost Classifier** (with optimized class-weight scaling)
3. **Unsupervised Persona Clustering:** Fits a `k-Means` clustering model ($k=4$) on standardized continuous banking features (`CreditScore`, `Age`, `Tenure`, `Balance`, `EstimatedSalary`) to tag customer behavioral clusters at runtime.
4. **FastAPI Serving Layer:** Features dynamic model selection (`model_type` payload parameter), strict Pydantic payload validation, and robust exception-handled inference routes.

---

## Installation & Local Setup

### **1. Clone the Repository**
```bash
git clone [https://github.com/PandoraMK/churn-segmentation-engine.git](https://github.com/PandoraMK/churn-segmentation-engine.git)
cd churn-segmentation-engine


