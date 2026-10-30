"""Classical baselines for PS-Q1."""
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_SEED = 42

def train_classical(X_train, y_train, X_val, y_val):
    results = {}
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED))
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1))
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED, probability=True))
        ]),
    }
    best_name, best_f1, best_model = None, -1, None
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        y_pred = model.predict(X_val)
        macro_f1 = f1_score(y_val, y_pred, average="macro")
        acc = accuracy_score(y_val, y_pred)
        results[name] = {
            "macro_f1": float(macro_f1),
            "accuracy": float(acc),
            "precision_per_class": precision_score(y_val, y_pred, average=None).tolist(),
            "recall_per_class": recall_score(y_val, y_pred, average=None).tolist(),
            "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
            "train_time_sec": train_time,
        }
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = name
            best_model = model
    return best_model, best_name, results
