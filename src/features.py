"""
features.py — Feature engineering functions
Project: ML-based Predictive Maintenance for Rural Borehole Water Pumps
Course:  COEN807 Machine Learning, ABU Zaria
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib, os

WINDOWS = [5, 15, 60]   # rolling window sizes in minutes

def rolling_statistics(df: pd.DataFrame, windows: list = WINDOWS) -> pd.DataFrame:
    """Compute rolling mean, std, min, max, RMS, skew for each sensor."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor")]
    features = {}
    for w in windows:
        for col in sensor_cols:
            s = df[col]
            features[f"{col}_mean_{w}"]  = s.rolling(w, min_periods=1).mean()
            features[f"{col}_std_{w}"]   = s.rolling(w, min_periods=1).std().fillna(0)
            features[f"{col}_min_{w}"]   = s.rolling(w, min_periods=1).min()
            features[f"{col}_max_{w}"]   = s.rolling(w, min_periods=1).max()
            features[f"{col}_rms_{w}"]   = (s**2).rolling(w, min_periods=1).mean().apply(np.sqrt)
            features[f"{col}_skew_{w}"]  = s.rolling(w, min_periods=1).skew().fillna(0)
    return pd.DataFrame(features, index=df.index)

def rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """First-order difference (rate of change) for each sensor."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor")]
    diff_df = df[sensor_cols].diff().fillna(0)
    diff_df.columns = [f"{c}_diff1" for c in sensor_cols]
    return diff_df

def lag_features(df: pd.DataFrame, top_sensors: list, lags: list = [1, 5, 15]) -> pd.DataFrame:
    """Lag values for the most important sensors."""
    lag_dict = {}
    for sensor in top_sensors:
        if sensor in df.columns:
            for lag in lags:
                lag_dict[f"{sensor}_lag{lag}"] = df[sensor].shift(lag).fillna(method="bfill")
    return pd.DataFrame(lag_dict, index=df.index)

def apply_pca(X_train: pd.DataFrame, X_test: pd.DataFrame,
              variance_threshold: float = 0.95,
              save_path: str = None) -> tuple:
    """Reduce to components explaining variance_threshold of variance."""
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_train_pca = pd.DataFrame(pca.fit_transform(X_train),
                                index=X_train.index,
                                columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    X_test_pca  = pd.DataFrame(pca.transform(X_test),
                                index=X_test.index,
                                columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(pca, save_path)
    print(f"  PCA: {X_train.shape[1]} features → {pca.n_components_} components "
          f"({variance_threshold*100:.0f}% variance retained)")
    return X_train_pca, X_test_pca, pca

def build_feature_matrix(df: pd.DataFrame,
                         top_sensors: list = None,
                         include_raw: bool = True) -> pd.DataFrame:
    """Assemble all features into one DataFrame."""
    parts = []
    if include_raw:
        sensor_cols = [c for c in df.columns if c.startswith("sensor")]
        parts.append(df[sensor_cols])
    parts.append(rolling_statistics(df))
    parts.append(rate_of_change(df))
    if top_sensors:
        parts.append(lag_features(df, top_sensors))
    return pd.concat(parts, axis=1)
