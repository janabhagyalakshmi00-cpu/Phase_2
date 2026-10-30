import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

folder = Path("data/q1")
train = pd.read_csv(folder / "ecg_train.csv")
val = pd.read_csv(folder / "ecg_val.csv")

features = [c for c in train.columns if c != "label"]
X_train = train[features].to_numpy(dtype=float)
y_train = train["label"].to_numpy()
X_val = val[features].to_numpy(dtype=float)
y_val = val["label"].to_numpy()

print("\n=== DATASET CHECK ===")
print("Training rows:", len(train))
print("Validation rows:", len(val))
print("Features:", len(features))
print("Training labels:", train["label"].value_counts().to_dict())
print("Validation labels:", val["label"].value_counts().to_dict())

print("\n=== DUPLICATE CHECK ===")
train_rows = set(map(tuple, X_train))
val_rows = set(map(tuple, X_val))
print("Exact feature rows shared between train and validation:",
      len(train_rows.intersection(val_rows)))

print("\n=== FEATURE / LABEL CHECK ===")
for col in features:
    a = train[col].to_numpy()
    if np.all(np.isfinite(a)) and np.all(np.isfinite(y_train)):
        if np.array_equal(a, y_train) or np.array_equal(a, 1 - y_train):
            print("WARNING: Feature exactly matches label:", col)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
)

print("\n=== FIVE-FOLD TRAINING CROSS-VALIDATION ===")
scores = cross_val_score(
    model, X_train, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy"
)
print("Fold accuracies:", np.round(scores, 4).tolist())
print("Mean CV accuracy:", round(float(scores.mean()), 4))

print("\n=== VALIDATION PERFORMANCE ===")
model.fit(X_train, y_train)
pred = model.predict(X_val)
print("Validation accuracy:", round(accuracy_score(y_val, pred), 4))
print("Validation Macro-F1:", round(f1_score(y_val, pred, average="macro"), 4))
print("\nCHECK COMPLETE")
