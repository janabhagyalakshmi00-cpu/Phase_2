"""Controlled ablation studies for PS-Q1 (bonus points).

All ablations use validation data only.
PCA / scalers always fitted on training data only.
Seeds fixed at 42.
"""
from __future__ import annotations
import time
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noise_model.benchmark_noise_model import prepare_for_noisy_run, get_benchmark_noise_model

RANDOM_SEED = 42
SHOTS = 256

def make_circuit(x, theta, n_qubits, fm_reps=1, ansatz_reps=2):
    fm = ZZFeatureMap(feature_dimension=n_qubits, reps=fm_reps, entanglement="linear")
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=ansatz_reps, entanglement="linear")
    qc = QuantumCircuit(n_qubits)
    qc.compose(fm.assign_parameters(x), inplace=True)
    qc.compose(ansatz.assign_parameters(theta), inplace=True)
    qc.measure_all()
    return qc.decompose().decompose()

def expect_z(counts):
    total = sum(counts.values()) or 1
    exp = sum((1 - 2 * int(b[-1])) * c for b, c in counts.items())
    return exp / total

def predict_proba(X, theta, n_qubits, fm_reps=1, ansatz_reps=2, noisy=False, shots=SHOTS, seed=RANDOM_SEED):
    sim = AerSimulator(noise_model=get_benchmark_noise_model(), seed_simulator=seed) if noisy \
          else AerSimulator(method="statevector", seed_simulator=seed)
    probs = []
    for x in X:
        qc = make_circuit(x, theta, n_qubits, fm_reps, ansatz_reps)
        if noisy:
            qc = prepare_for_noisy_run(qc)
        counts = sim.run(qc, shots=shots).result().get_counts()
        p1 = (1.0 - expect_z(counts)) / 2.0
        probs.append(float(np.clip(p1, 0.0, 1.0)))
    return np.array(probs)


def train_one(
    X, y, n_qubits, fm_reps=1, ansatz_reps=2,
    maxiter=12, n_sub=80, seed=RANDOM_SEED
):
    rng = np.random.default_rng(seed)

    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=ansatz_reps,
        entanglement="linear"
    )
    n_params = ansatz.num_parameters
    theta0 = rng.uniform(0, 2 * np.pi, n_params)

    # Reproducible training subset
    idx = rng.choice(len(X), min(n_sub, len(X)), replace=False)
    Xs, ys = X[idx], np.asarray(y)[idx]

    # Fixed objective batch: same samples for every objective evaluation
    batch_size = min(24, len(Xs))
    batch_idx = rng.choice(len(Xs), batch_size, replace=False)
    Xb, yb = Xs[batch_idx], ys[batch_idx]

    history = []

    def obj(theta):
        p = predict_proba(
            Xb, theta, n_qubits,
            fm_reps=fm_reps,
            ansatz_reps=ansatz_reps,
            noisy=False,
            shots=128,
            seed=seed
        )
        eps = 1e-8
        loss = -np.mean(
            yb * np.log(p + eps)
            + (1 - yb) * np.log(1 - p + eps)
        )
        history.append(float(loss))
        return float(loss)

    t0 = time.time()
    res = minimize(
        obj,
        theta0,
        method="COBYLA",
        options={"maxiter": maxiter, "disp": False}
    )

    return res.x, history, time.time() - t0, n_params



def preprocess(X_train, X_val, n_comp):
    scaler = StandardScaler().fit(X_train)
    Xts = scaler.transform(X_train)
    Xvs = scaler.transform(X_val)

    pca = PCA(
        n_components=n_comp,
        random_state=RANDOM_SEED
    )
    Xtp = pca.fit_transform(Xts)
    Xvp = pca.transform(Xvs)

    mm = MinMaxScaler(feature_range=(0, np.pi)).fit(Xtp)
    Xtp = mm.transform(Xtp)
    Xvp = mm.transform(Xvp)

    explained_variance = pca.explained_variance_ratio_.sum()

    return Xtp, Xvp, explained_variance


def run_ablations(X_train, y_train, X_val, y_val, out_path="results/ablation_results.json"):
    np.random.seed(RANDOM_SEED)
    # fixed small val subset for fair comparison across configs
    val_idx = np.random.choice(len(X_val), 60, replace=False)
    yv = y_val[val_idx]

    ablations = []

    # ---- 1. PCA dimension / qubit count ----
    print("Ablation 1: PCA dimension / qubit count")
    for n_comp in [2, 4, 6]:
        Xt, Xv, var = preprocess(X_train, X_val, n_comp)
        Xv_sub = Xv[val_idx]
        theta, hist, ttime, npar = train_one(Xt, y_train, n_qubits=n_comp, maxiter=10, n_sub=70)
        ideal = predict_proba(Xv_sub, theta, n_comp, noisy=False)
        f1_i = float(f1_score(yv, (ideal >= 0.5).astype(int), average="macro"))
        
        # Noisy evaluation only for n=4
        f1_n = None
        if n_comp == 4:
            noisy = predict_proba(
                Xv_sub,
                theta,
                n_comp,
                noisy=True,
                shots=256
            )
            f1_n = float(
                f1_score(
                    yv,
                    (noisy >= 0.5).astype(int),
                    average="macro"
                )
            )

        entry = {
            "ablation": "pca_qubits",
            "n_components": n_comp,
            "n_qubits": n_comp,
            "explained_variance": round(float(var), 4),
            "n_params": npar,
            "train_time_sec": round(ttime, 2),
            "ideal_macro_f1": round(f1_i, 4),
            "noisy_macro_f1": round(f1_n, 4) if f1_n is not None else None,
            "final_loss": round(hist[-1], 4) if hist else None
        }
        ablations.append(entry)
        print(f"  PCA={n_comp} → ideal F1={f1_i:.3f}" + (f", noisy={f1_n:.3f}" if f1_n else ""))

    # ---- 2. Ansatz depth (reps) ----
    print("\nAblation 2: Ansatz depth (reps)")
    Xt, Xv, _ = preprocess(X_train, X_val, 4)
    Xv_sub = Xv[val_idx]
    for reps in [1, 2, 3]:
        theta, hist, ttime, npar = train_one(Xt, y_train, n_qubits=4, ansatz_reps=reps, maxiter=10, n_sub=70)
        ideal = predict_proba(Xv_sub, theta, 4, ansatz_reps=reps, noisy=False)
        f1_i = float(f1_score(yv, (ideal >= 0.5).astype(int), average="macro"))
        entry = {
            "ablation": "ansatz_depth",
            "ansatz_reps": reps,
            "n_qubits": 4,
            "n_params": npar,
            "train_time_sec": round(ttime, 2),
            "ideal_macro_f1": round(f1_i, 4),
            "final_loss": round(hist[-1], 4) if hist else None
        }
        ablations.append(entry)
        print(f"  reps={reps} (params={npar}) → ideal F1={f1_i:.3f}")

    # ---- 3. Feature map reps ----
    print("\nAblation 3: Feature-map reps")
    for fm_reps in [1, 2]:
        theta, hist, ttime, npar = train_one(Xt, y_train, n_qubits=4, fm_reps=fm_reps, ansatz_reps=2, maxiter=10, n_sub=70)
        ideal = predict_proba(Xv_sub, theta, 4, fm_reps=fm_reps, ansatz_reps=2, noisy=False)
        f1_i = float(f1_score(yv, (ideal >= 0.5).astype(int), average="macro"))
        entry = {
            "ablation": "feature_map_reps",
            "fm_reps": fm_reps,
            "n_qubits": 4,
            "n_params": npar,
            "train_time_sec": round(ttime, 2),
            "ideal_macro_f1": round(f1_i, 4),
            "final_loss": round(hist[-1], 4) if hist else None
        }
        ablations.append(entry)
        print(f"  fm_reps={fm_reps} → ideal F1={f1_i:.3f}")

    # ---- 4. Random seed sensitivity ----
    print("\nAblation 4: Seed sensitivity (n_qubits=4, reps=2)")
    seed_f1s = []
    for seed in [42, 7, 123]:
        theta, hist, ttime, npar = train_one(Xt, y_train, n_qubits=4, maxiter=8, n_sub=60, seed=seed)
        ideal = predict_proba(Xv_sub, theta, 4, noisy=False, seed=seed)
        f1_i = float(f1_score(yv, (ideal >= 0.5).astype(int), average="macro"))
        seed_f1s.append(f1_i)
        ablations.append({
            "ablation": "seed_sensitivity",
            "seed": seed,
            "ideal_macro_f1": round(f1_i, 4),
            "train_time_sec": round(ttime, 2)
        })
        print(f"  seed={seed} → ideal F1={f1_i:.3f}")

    summary = {
        "note": "All ablations use a fixed 60-sample validation subset for fair comparison. PCA/scalers fitted on train only. Full noisy evaluation only for main 4-qubit config.",
        "seed_mean_f1": round(float(np.mean(seed_f1s)), 4),
        "seed_std_f1": round(float(np.std(seed_f1s)), 4),
        "ablations": ablations
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")
    return summary

if __name__ == "__main__":
    from data_utils import load_data
    X_train, y_train, X_val, y_val, *_ = load_data()
    run_ablations(X_train, y_train, X_val, y_val)
