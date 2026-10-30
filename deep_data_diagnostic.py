import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.data_utils import load_data

X_train, y_train, X_val, y_val, _, _, feature_cols = load_data()

train_df = pd.read_csv("data/q1/ecg_train.csv")
val_df = pd.read_csv("data/q1/ecg_val.csv")

print("=" * 70)
print("DEEP DATASET AND CLASSICAL MODEL DIAGNOSTIC")
print("=" * 70)

print("\n1. DATASET INFORMATION")
print(f"Training shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Feature count: {len(feature_cols)}")
print(f"Training labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Validation labels: {dict(zip(*np.unique(y_val, return_counts=True)))}")
print(f"Missing values in training: {int(train_df.isna().sum().sum())}")
print(f"Missing values in validation: {int(val_df.isna().sum().sum())}")

print("\n2. BEST SINGLE-FEATURE DECISION TREES")
single_feature_results = []

for i, feature in enumerate(feature_cols):
    tree = DecisionTreeClassifier(max_depth=1, random_state=42)
    tree.fit(X_train[:, [i]], y_train)
    pred = tree.predict(X_val[:, [i]])
    acc = accuracy_score(y_val, pred)

    single_feature_results.append({
        "feature": feature,
        "validation_accuracy": float(acc)
    })

single_feature_results.sort(
    key=lambda item: item["validation_accuracy"], reverse=True
)

for item in single_feature_results[:10]:
    print(f"{item['feature']:<25} accuracy={item['validation_accuracy']:.2%}")

print("\n3. SHALLOW DECISION TREE")
for depth in [1, 2, 3, 5]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_val)
    print(
        f"Max depth {depth}: "
        f"accuracy={accuracy_score(y_val, pred):.2%}, "
        f"Macro-F1={f1_score(y_val, pred, average='macro'):.4f}"
    )

print("\n4. CHECK WHETHER TRAIN AND VALIDATION FEATURE RANGES OVERLAP")
range_results = []

for i, feature in enumerate(feature_cols):
    train_min, train_max = X_train[:, i].min(), X_train[:, i].max()
    val_min, val_max = X_val[:, i].min(), X_val[:, i].max()

    overlaps = not (val_max < train_min or val_min > train_max)

    range_results.append({
        "feature": feature,
        "train_min": float(train_min),
        "train_max": float(train_max),
        "val_min": float(val_min),
        "val_max": float(val_max),
        "ranges_overlap": bool(overlaps)
    })

print(
    "Features whose training and validation ranges do not overlap:",
    sum(not item["ranges_overlap"] for item in range_results),
    "of", len(feature_cols)
)

print("\n5. SUMMARY")
print(
    "A single feature or shallow tree with near-perfect accuracy would "
    "suggest that the classes are very easily separated by these features."
)
print(
    "That alone does not prove data leakage. We must also verify how the "
    "features and train/validation splits were created."
)

output = {
    "single_feature_results": single_feature_results,
    "feature_ranges": range_results
}

Path("results").mkdir(exist_ok=True)
Path("results/deep_data_diagnostic.json").write_text(
    json.dumps(output, indent=2), encoding="utf-8"
)

print("\nDiagnostic saved to results/deep_data_diagnostic.json")
