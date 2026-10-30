import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.data_utils import load_data
from src.ablation import preprocess, train_one, predict_proba

SEED = 42
N_QUBITS = 4
ANSATZ_REPS = 3
MAXITER = 30
TRAIN_SUBSET = 200
NOISY_SUBSET = 300

print("=" * 65)
print("PHASE 2: QUANTUM VQC - THREE REPETITION EXPERIMENT")
print("=" * 65)

X_train, y_train, X_val, y_val, *_ = load_data()

print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Qubits: {N_QUBITS}")
print(f"Ansatz repetitions: {ANSATZ_REPS}")

print("\n[1/4] Preprocessing data using training data only...")
X_train_p, X_val_p, explained_variance = preprocess(
    X_train, X_val, N_QUBITS
)
print(f"PCA explained variance: {explained_variance:.4f}")

print("\n[2/4] Training quantum model...")
start = time.time()

theta, history, train_seconds, n_params = train_one(
    X_train_p,
    y_train,
    n_qubits=N_QUBITS,
    fm_reps=1,
    ansatz_reps=ANSATZ_REPS,
    maxiter=MAXITER,
    n_sub=TRAIN_SUBSET,
    seed=SEED
)

print(f"Trainable parameters: {n_params}")
print(f"Training time: {train_seconds:.2f} seconds")

print("\n[3/4] Evaluating on ALL validation samples...")
start = time.time()

ideal_probs = predict_proba(
    X_val_p,
    theta,
    n_qubits=N_QUBITS,
    fm_reps=1,
    ansatz_reps=ANSATZ_REPS,
    noisy=False,
    shots=256,
    seed=SEED
)

ideal_preds = (ideal_probs >= 0.5).astype(int)
ideal_accuracy = accuracy_score(y_val, ideal_preds)
ideal_f1 = f1_score(y_val, ideal_preds, average="macro")

print(f"Full validation accuracy: {ideal_accuracy:.4f}")
print(f"Full validation Macro-F1: {ideal_f1:.4f}")
print(f"Full validation evaluation time: {time.time() - start:.2f} seconds")

print("\n[4/4] Evaluating noisy quantum model on 300 samples...")
rng = np.random.default_rng(SEED)
noisy_idx = rng.choice(len(X_val_p), min(NOISY_SUBSET, len(X_val_p)), replace=False)

noisy_probs = predict_proba(
    X_val_p[noisy_idx],
    theta,
    n_qubits=N_QUBITS,
    fm_reps=1,
    ansatz_reps=ANSATZ_REPS,
    noisy=True,
    shots=256,
    seed=SEED
)

noisy_preds = (noisy_probs >= 0.5).astype(int)
y_noisy = y_val[noisy_idx]

noisy_accuracy = accuracy_score(y_noisy, noisy_preds)
noisy_f1 = f1_score(y_noisy, noisy_preds, average="macro")

ideal_subset_preds = ideal_preds[noisy_idx]
ideal_subset_f1 = f1_score(y_noisy, ideal_subset_preds, average="macro")

results = {
    "experiment": "four_qubits_three_ansatz_repetitions",
    "seed": SEED,
    "n_qubits": N_QUBITS,
    "ansatz_reps": ANSATZ_REPS,
    "maxiter": MAXITER,
    "training_subset_size": min(TRAIN_SUBSET, len(X_train_p)),
    "validation_size_ideal": len(y_val),
    "noisy_validation_subset_size": len(y_noisy),
    "pca_explained_variance": float(explained_variance),
    "n_params": int(n_params),
    "training_time_sec": float(train_seconds),
    "ideal_full_validation_accuracy": float(ideal_accuracy),
    "ideal_full_validation_macro_f1": float(ideal_f1),
    "ideal_full_validation_confusion_matrix": confusion_matrix(
        y_val, ideal_preds
    ).tolist(),
    "ideal_noisy_subset_macro_f1": float(ideal_subset_f1),
    "noisy_subset_accuracy": float(noisy_accuracy),
    "noisy_subset_macro_f1": float(noisy_f1),
    "note": (
        "Ideal metrics use the full validation set. Noisy metrics use a "
        "fixed-seed 300-sample subset. Compare noisy and ideal subset scores "
        "on the same samples. This experiment alone does not establish "
        "quantum advantage."
    )
}

output_path = Path("results/quantum_reps3_full_validation.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

print("\n" + "=" * 65)
print("EXPERIMENT FINISHED")
print("=" * 65)
print(f"Ideal accuracy (full validation): {ideal_accuracy:.2%}")
print(f"Ideal Macro-F1 (full validation): {ideal_f1:.4f}")
print(f"Ideal Macro-F1 (same noisy subset): {ideal_subset_f1:.4f}")
print(f"Noisy accuracy (300 samples): {noisy_accuracy:.2%}")
print(f"Noisy Macro-F1 (300 samples): {noisy_f1:.4f}")
print(f"\nSaved results to: {output_path}")
print("\nExisting quantum_results.json was not modified.")
