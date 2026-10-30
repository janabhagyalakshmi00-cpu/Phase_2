"""Hybrid VQC for PS-Q1 using Qiskit 2.x.

Design choices (justified):
- Reduce 30 features → 4 via PCA (fit on train only) to match qubit count and keep circuit shallow.
- Angle encoding with ZZFeatureMap (linear entanglement) — captures pairwise interactions useful for statistical features.
- RealAmplitudes ansatz (reps=2, linear) — low parameter count, nearest-neighbour CX matches linear Processor D.
- 4 qubits, ~16 trainable parameters.
- COBYLA + statevector training for speed; noisy eval with official QFF2026-v2.
- Random seed 42 everywhere.
"""
from __future__ import annotations
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noise_model.benchmark_noise_model import prepare_for_noisy_run, get_benchmark_noise_model

RANDOM_SEED = 42
N_QUBITS = 4
FEATURE_MAP_REPS = 1
ANSATZ_REPS = 2
MAXITER = 100
SHOTS_EVAL = 2048

def get_feature_map_ansatz(n_qubits=N_QUBITS):
    fm = ZZFeatureMap(feature_dimension=n_qubits, reps=FEATURE_MAP_REPS, entanglement="linear")
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=ANSATZ_REPS, entanglement="linear")
    return fm, ansatz

def build_bound_circuit(x, theta, n_qubits=N_QUBITS):
    fm, ansatz = get_feature_map_ansatz(n_qubits)
    qc = QuantumCircuit(n_qubits)
    qc.compose(fm.assign_parameters(x), inplace=True)
    qc.compose(ansatz.assign_parameters(theta), inplace=True)
    qc.measure_all()
    # Decompose to elementary gates so Aer understands them
    return qc.decompose().decompose()

def expectation_z(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    exp = 0.0
    for bitstr, cnt in counts.items():
        # Use last qubit Z
        bit = int(bitstr[-1])
        exp += (1 - 2 * bit) * cnt
    return exp / total

def predict_proba(X, theta, shots=SHOTS_EVAL, noisy=False, seed=RANDOM_SEED):
    probs = []
    if noisy:
        sim = AerSimulator(noise_model=get_benchmark_noise_model(), seed_simulator=seed)
    else:
        sim = AerSimulator(method="statevector", seed_simulator=seed)
    for x in X:
        qc = build_bound_circuit(x, theta)
        if noisy:
            qc = prepare_for_noisy_run(qc)
        counts = sim.run(qc, shots=shots).result().get_counts()
        exp = expectation_z(counts)
        p1 = (1.0 - exp) / 2.0
        probs.append(float(np.clip(p1, 0.0, 1.0)))
    return np.array(probs)


def train_vqc(
    X_train,
    y_train,
    maxiter=MAXITER,
    n_sub=400,
    batch_size=100,
    seed=RANDOM_SEED,
):
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int).ravel()

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train lengths do not match.")
    if len(np.unique(y_train)) != 2:
        raise ValueError("VQC requires both binary classes.")

    # Stratified subset: preserve class proportions.
    n_sub = min(int(n_sub), len(y_train))
    all_idx = np.arange(len(y_train))

    if n_sub < len(y_train):
        idx, _ = train_test_split(
            all_idx,
            train_size=n_sub,
            stratify=y_train,
            random_state=seed,
        )
    else:
        idx = all_idx

    X_sub = X_train[idx]
    y_sub = y_train[idx]

    # Fixed batch prevents the optimizer objective changing randomly.
    batch_size = min(int(batch_size), len(y_sub))
    batch_idx, _ = train_test_split(
        np.arange(len(y_sub)),
        train_size=batch_size,
        stratify=y_sub,
        random_state=seed,
    ) if batch_size < len(y_sub) else (
        np.arange(len(y_sub)), np.array([], dtype=int)
    )

    X_batch = X_sub[batch_idx]
    y_batch = y_sub[batch_idx]

    _, ansatz = get_feature_map_ansatz()
    n_params = ansatz.num_parameters

    theta0 = rng.uniform(0, 2 * np.pi, size=n_params)
    history = []

    def objective(theta):
        probs = predict_proba(
            X_batch,
            theta,
            shots=512,
            noisy=False,
            seed=seed,
        )
        eps = 1e-7
        probs = np.clip(probs, eps, 1 - eps)

        loss = -np.mean(
            y_batch * np.log(probs)
            + (1 - y_batch) * np.log(1 - probs)
        )
        history.append(float(loss))
        return float(loss)

    t0 = time.time()
    res = minimize(
        objective,
        theta0,
        method="COBYLA",
        options={"maxiter": maxiter, "disp": False},
    )

    return res.x, history, time.time() - t0, n_params



def evaluate_vqc(theta, X, y, noisy=False, shots=SHOTS_EVAL):
    
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    y_true = np.asarray(y, dtype=int).ravel()

    probs = np.asarray(
        predict_proba(X, theta, shots=shots, noisy=noisy),
        dtype=float
    ).ravel()

    if len(y_true) != len(probs):
        raise ValueError(
            f"Number of labels ({len(y_true)}) does not match "
            f"number of predictions ({len(probs)})."
        )

    preds = (probs >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, preds)
    macro_f1 = f1_score(
        y_true,
        preds,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100),
        "macro_f1": float(macro_f1),
        "probs": probs.tolist(),
        "preds": preds.tolist()
    }