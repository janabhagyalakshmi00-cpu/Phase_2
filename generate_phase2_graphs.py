import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
RESULTS = ROOT / "results"
OUT = RESULTS / "graphs"
OUT.mkdir(parents=True, exist_ok=True)

def load_json(name):
    path = RESULTS / name
    if not path.exists():
        print(f"SKIPPED: {path} not found")
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT / name}")

def bar_plot(labels, values, title, ylabel, filename, ylim=None):
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    if ylim:
        plt.ylim(*ylim)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    save_plot(filename)

# 1. Classical model comparison
classical = load_json("classical_results.json")
if classical and "all" in classical:
    models = list(classical["all"])
    accuracy = [classical["all"][m]["accuracy"] for m in models]
    f1 = [classical["all"][m]["macro_f1"] for m in models]
    x = np.arange(len(models))
    width = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, accuracy, width, label="Accuracy")
    plt.bar(x + width/2, f1, width, label="Macro-F1")
    plt.xticks(x, models, rotation=15, ha="right")
    plt.ylim(0, 1.12)
    plt.ylabel("Score")
    plt.title("Classical Model Performance — Full Validation")
    plt.legend()
    save_plot("01_classical_model_comparison.png")

    # 2. Classical training time
    times = [classical["all"][m]["train_time_sec"] for m in models]
    bar_plot(models, times, "Classical Model Training Time",
             "Seconds", "02_classical_training_time.png")

# 3. Full validation comparison: ideal VQC and classical models
qfull = load_json("quantum_reps3_full_validation.json")
if classical and qfull:
    labels = list(classical["all"]) + ["VQC ideal"]
    accuracy = [classical["all"][m]["accuracy"] for m in classical["all"]]
    accuracy.append(qfull["ideal_full_validation_accuracy"])
    f1 = [classical["all"][m]["macro_f1"] for m in classical["all"]]
    f1.append(qfull["ideal_full_validation_macro_f1"])
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(10, 5))
    plt.bar(x-width/2, accuracy, width, label="Accuracy")
    plt.bar(x+width/2, f1, width, label="Macro-F1")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, 1.12)
    plt.ylabel("Score")
    plt.title("Classical Models vs VQC — Full Validation")
    plt.legend()
    save_plot("03_classical_vs_vqc_full_validation.png")

# 4. Same 300-sample subset comparison
same = load_json("classical_same_subset_comparison.json")
if same and qfull:
    labels = list(same) + ["VQC noisy"]
    acc = [same[m]["same_300_samples_accuracy"] for m in same]
    f1 = [same[m]["same_300_samples_macro_f1"] for m in same]
    acc.append(qfull["noisy_subset_accuracy"])
    f1.append(qfull["noisy_subset_macro_f1"])
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(10, 5))
    plt.bar(x-width/2, acc, width, label="Accuracy")
    plt.bar(x+width/2, f1, width, label="Macro-F1")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, 1.12)
    plt.ylabel("Score")
    plt.title("Performance on 300 Samples — Classical vs Noisy VQC")
    plt.legend()
    save_plot("04_same_300_sample_comparison.png")

# 5. VQC confusion matrix from saved full-validation metrics
if qfull and "ideal_full_validation_confusion_matrix" in qfull:
    cm = np.array(qfull["ideal_full_validation_confusion_matrix"])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(label="Number of samples")
    plt.xticks([0, 1], ["Normal", "Arrhythmia"])
    plt.yticks([0, 1], ["Normal", "Arrhythmia"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("VQC Ideal Full-Validation Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    save_plot("05_vqc_confusion_matrix.png")

# 6–9. Ablation studies
ablation = load_json("ablation_results.json")
if ablation:
    rows = ablation.get("ablations", [])

    pca = [r for r in rows if r.get("ablation") == "pca_qubits"]
    if pca:
        pca.sort(key=lambda r: r["n_components"])
        xlabels = [str(r["n_components"]) for r in pca]
        bar_plot(xlabels, [r["ideal_macro_f1"] for r in pca],
                 "Ablation: PCA Components / Qubits",
                 "Ideal Macro-F1", "06_ablation_pca_qubits.png",
                 (0, 1.05))
        bar_plot(xlabels, [r["explained_variance"] for r in pca],
                 "PCA Explained Variance",
                 "Explained variance ratio",
                 "07_pca_explained_variance.png", (0, 1.05))

    ansatz = [r for r in rows if r.get("ablation") == "ansatz_depth"]
    if ansatz:
        ansatz.sort(key=lambda r: r["ansatz_reps"])
        bar_plot([str(r["ansatz_reps"]) for r in ansatz],
                 [r["ideal_macro_f1"] for r in ansatz],
                 "Ablation: Ansatz Repetitions",
                 "Ideal Macro-F1", "08_ablation_ansatz_repetitions.png",
                 (0, 1.05))

    fm = [r for r in rows if r.get("ablation") == "feature_map_reps"]
    if fm:
        fm.sort(key=lambda r: r["fm_reps"])
        bar_plot([str(r["fm_reps"]) for r in fm],
                 [r["ideal_macro_f1"] for r in fm],
                 "Ablation: Feature Map Repetitions",
                 "Ideal Macro-F1", "09_ablation_feature_map_repetitions.png",
                 (0, 1.05))

    seeds = [r for r in rows if r.get("ablation") == "seed_sensitivity"]
    if seeds:
        seeds.sort(key=lambda r: r["seed"])
        bar_plot([str(r["seed"]) for r in seeds],
                 [r["ideal_macro_f1"] for r in seeds],
                 "Ablation: Random Seed Sensitivity",
                 "Ideal Macro-F1", "10_seed_sensitivity.png",
                 (0, 1.05))

    timed = [r for r in rows if "train_time_sec" in r
             and r.get("ablation") in ("pca_qubits", "ansatz_depth",
                                       "feature_map_reps")]
    if timed:
        labels = [f'{r["ablation"]}\n{r.get("n_components", r.get("ansatz_reps", r.get("fm_reps")))}'
                  for r in timed]
        bar_plot(labels, [r["train_time_sec"] for r in timed],
                 "Ablation Experiment Training Times",
                 "Seconds", "11_ablation_training_times.png")

# 12. Task A ideal vs noisy Bell fidelity
task = load_json("task_a_results.json")
if task:
    ideal = task["ideal"]
    noisy = task["noisy"]
    labels = ["XX", "YY", "ZZ", "Fidelity"]
    keys = ["XX", "YY", "ZZ", "fidelity"]
    ideal_vals = [ideal[k] for k in keys]
    noisy_vals = [noisy[k] for k in keys]
    ideal_err = [ideal[k + "_se"] for k in keys]
    noisy_err = [noisy[k + "_se"] for k in keys]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar(x-width/2, ideal_vals, width, yerr=ideal_err,
            capsize=4, label="Ideal")
    plt.bar(x+width/2, noisy_vals, width, yerr=noisy_err,
            capsize=4, label="Noisy")
    plt.xticks(x, labels)
    plt.ylabel("Expectation / Fidelity")
    plt.title("Task A: Ideal vs Noisy Bell-State Results")
    plt.legend()
    save_plot("12_task_a_ideal_vs_noisy.png")

# 13–14. Dataset class counts and feature distribution, if CSVs exist
csvs = list((ROOT / "data").rglob("ecg_train.csv"))
vals = list((ROOT / "data").rglob("ecg_val.csv"))

if csvs:
    train = pd.read_csv(csvs[0])
    if "label" in train.columns:
        counts = train["label"].value_counts().sort_index()
        bar_plot([f"Class {i}" for i in counts.index], counts.values,
                 "Training Dataset Class Distribution",
                 "Number of samples", "13_training_class_distribution.png")

        if vals:
            val = pd.read_csv(vals[0])
            if "label" in val.columns:
                combined = pd.DataFrame({
                    "Class": (["Normal" if x == 0 else "Arrhythmia"
                               for x in train["label"]] +
                              ["Normal" if x == 0 else "Arrhythmia"
                               for x in val["label"]]),
                    "Dataset": (["Train"] * len(train) +
                                ["Validation"] * len(val))
                })
                counts2 = combined.groupby(["Dataset", "Class"]).size().unstack(fill_value=0)
                counts2.plot(kind="bar", figsize=(8, 5))
                plt.ylabel("Number of samples")
                plt.title("Training vs Validation Class Distribution")
                plt.xticks(rotation=0)
                save_plot("14_train_validation_class_distribution.png")

        if "w5_mean" in train.columns and "label" in train.columns:
            plt.figure(figsize=(7, 5))
            data = [train.loc[train["label"] == c, "w5_mean"].dropna()
                    for c in sorted(train["label"].unique())]
            plt.boxplot(data, tick_labels=["Normal", "Arrhythmia"])
            plt.ylabel("w5_mean amplitude")
            plt.title("Training Feature Distribution: w5_mean by Class")
            save_plot("15_w5_mean_by_class.png")
else:
    print("Dataset CSVs named ecg_train.csv were not found under data/.")

print("\nGraph generation finished.")
print(f"Output directory: {OUT.resolve()}")
print(f"PNG files generated: {len(list(OUT.glob('*.png')))}")
