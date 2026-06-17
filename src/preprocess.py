"""
preprocess.py — Reusable preprocessing functions
Project: ML-based Predictive Maintenance for Rural Borehole Water Pumps
Course:  COEN807 Machine Learning, ABU Zaria
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib, os

RANDOM_SEED = 42

def load_dataset(path: str) -> pd.DataFrame:
    """Load and sort the pump sensor CSV by timestamp."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.set_index("timestamp", inplace=True)
    return df

def get_sensor_columns(df: pd.DataFrame) -> list:
    """Return list of sensor column names."""
    return [c for c in df.columns if c.startswith("sensor")]

def drop_high_missing(df: pd.DataFrame, threshold: float = 0.30) -> tuple:
    """Drop sensor columns with missing proportion above threshold."""
    sensor_cols = get_sensor_columns(df)
    missing_pct = df[sensor_cols].isnull().mean()
    to_drop = missing_pct[missing_pct > threshold].index.tolist()
    df = df.drop(columns=to_drop)
    return df, to_drop

def impute_sensors(df: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """Forward-fill then backward-fill gaps up to max_gap consecutive NaNs."""
    sensor_cols = get_sensor_columns(df)
    df[sensor_cols] = (df[sensor_cols]
                       .ffill(limit=max_gap)
                       .bfill(limit=max_gap))
    return df

def drop_low_variance(df: pd.DataFrame, threshold: float = 0.01) -> tuple:
    """Remove near-constant sensor columns."""
    sensor_cols = get_sensor_columns(df)
    var = df[sensor_cols].var()
    to_drop = var[var < threshold].index.tolist()
    df = df.drop(columns=to_drop)
    return df, to_drop

def time_split(df: pd.DataFrame, train_ratio: float = 0.80) -> tuple:
    """Strict time-ordered train/test split — no shuffling."""
    n = len(df)
    split_idx = int(n * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   save_path: str = None) -> tuple:
    """Fit StandardScaler on train, transform both sets."""
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                              columns=X_train.columns, index=X_train.index)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),
                              columns=X_test.columns,  index=X_test.index)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
    return X_train_s, X_test_s, scaler

def encode_labels(series: pd.Series) -> tuple:
    """Encode machine_status strings to integers."""
    mapping = {"NORMAL": 0, "BROKEN": 1, "RECOVERING": 2}
    return series.map(mapping), mapping
