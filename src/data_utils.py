"""Data loading and preprocessing for PS-Q1."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

RANDOM_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "q1"

def load_data():
    train = pd.read_csv(DATA_DIR / "ecg_train.csv")
    val = pd.read_csv(DATA_DIR / "ecg_val.csv")
    test = pd.read_csv(DATA_DIR / "ecg_test_features.csv")
    feature_cols = [c for c in train.columns if c != "label"]
    X_train = train[feature_cols].values.astype(np.float64)
    y_train = train["label"].values.astype(int)
    X_val = val[feature_cols].values.astype(np.float64)
    y_val = val["label"].values.astype(int)
    sample_ids = test["sample_id"].values
    X_test = test[feature_cols].values.astype(np.float64)
    return X_train, y_train, X_val, y_val, X_test, sample_ids, feature_cols

def fit_preprocess(X_train, n_components=4, scale_range=(0, np.pi)):
    """Fit PCA + MinMax on train only. Returns transformed data and transformers."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    # Scale to [0, pi] for angle encoding
    minmax = MinMaxScaler(feature_range=scale_range)
    X_final = minmax.fit_transform(X_pca)
    return X_final, {"scaler": scaler, "pca": pca, "minmax": minmax, "explained_variance": pca.explained_variance_ratio_.sum()}

def transform_preprocess(X, transformers):
    X_scaled = transformers["scaler"].transform(X)
    X_pca = transformers["pca"].transform(X_scaled)
    X_final = transformers["minmax"].transform(X_pca)
    return X_final
