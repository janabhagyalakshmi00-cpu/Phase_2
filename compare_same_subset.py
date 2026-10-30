import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.data_utils import load_data

SEED = 42

X_train, y_train, X_val, y_val, *_ = load_data()

# Use exactly the same fixed-seed validation subset as the quantum experiment.
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X_val), min(300, len(X_val)), replace=False)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ))
    ]),
    "random_forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=42, n_jobs=-1
        ))
    ]),
    "svm_rbf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf", class_weight="balanced",
            random_state=42
        ))
    ])
}

results = {}

print("\nCLASSICAL MODELS: SAME VALIDATION SAMPLES")
print("-" * 65)

for name, model in models.items():
    model.fit(X_train, y_train)

    full_preds = model.predict(X_val)
    subset_preds = model.predict(X_val[idx])

    full_accuracy = accuracy_score(y_val, full_preds)
    full_f1 = f1_score(y_val, full_preds, average="macro")
    subset_accuracy = accuracy_score(y_val[idx], subset_preds)
    subset_f1 = f1_score(y_val[idx], subset_preds, average="macro")

    results[name] = {
        "full_validation_accuracy": float(full_accuracy),
        "full_validation_macro_f1": float(full_f1),
        "same_300_samples_accuracy": float(subset_accuracy),
        "same_300_samples_macro_f1": float(subset_f1)
    }

    print(f"\n{name}")
    print(f"  Full validation accuracy: {full_accuracy:.2%}")
    print(f"  Full validation Macro-F1: {full_f1:.4f}")
    print(f"  Same 300 samples accuracy: {subset_accuracy:.2%}")
    print(f"  Same 300 samples Macro-F1: {subset_f1:.4f}")

quantum_path = Path("results/quantum_reps3_full_validation.json")
quantum = json.loads(quantum_path.read_text(encoding="utf-8"))

print("\nQUANTUM MODEL")
print("-" * 65)
print(f"  Full validation accuracy: "
      f"{quantum['ideal_full_validation_accuracy']:.2%}")
print(f"  Full validation Macro-F1: "
      f"{quantum['ideal_full_validation_macro_f1']:.4f}")
print(f"  Noisy 300-sample accuracy: "
      f"{quantum['noisy_subset_accuracy']:.2%}")
print(f"  Noisy 300-sample Macro-F1: "
      f"{quantum['noisy_subset_macro_f1']:.4f}")

output = Path("results/classical_same_subset_comparison.json")
output.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nClassical comparison saved to: {output}")
print("Existing results were not overwritten.")
