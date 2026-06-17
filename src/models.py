"""
models.py — Model definitions and wrappers
Project: ML-based Predictive Maintenance for Rural Borehole Water Pumps
Course:  COEN807 Machine Learning, ABU Zaria
"""
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib, os

RANDOM_SEED = 42

def get_random_forest(class_weight="balanced_subsample"):
    return RandomForestClassifier(
        n_estimators=100, class_weight=class_weight,
        random_state=RANDOM_SEED, n_jobs=-1)

def get_xgboost(scale_pos_weight=10):
    return XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=RANDOM_SEED, n_jobs=-1)

def get_svm():
    return SVC(kernel="rbf", class_weight="balanced",
               probability=True, random_state=RANDOM_SEED)

def get_isolation_forest(contamination=0.07):
    return IsolationForest(
        n_estimators=200, contamination=contamination,
        random_state=RANDOM_SEED, n_jobs=-1)

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved: {path}")

def load_model(path: str):
    return joblib.load(path)
