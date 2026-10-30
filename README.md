# PS-Q1 Biomedical Signal Classifier — QFF 2026 Phase 2

**Problem Statement:** PS-Q1 (Biomedical Signal Classification)  
**Track:** Quantum Machine Learning  
**Continuation of Phase 1 repository:** https://github.com/kasi-2006-bit/Biomedical-Signal-Classifier

## Overview
Hybrid quantum-classical VQC (4-qubit ZZFeatureMap + RealAmplitudes) for Normal vs Arrhythmia classification on QML-BIO-MITBIH features, compared against classical baselines. Includes official Task A (Bell fidelity) and Task B (Processor D QFT-3 benchmark) under the QFF2026-v2 noise model.

## Setup
```bash
python3 -m venv qml-env
source qml-env/bin/activate
pip install -r requirements.txt
```

## Self-tests (required)
```bash
python noise_model/benchmark_noise_model.py
python benchmark/qff_qml_benchmark.py processor_d_config.json
python tools/validate_predictions.py q1 predictions_q1.csv
```

## Key Results (actual measured)
- Classical (Logistic Regression) val macro-F1: **1.000**
- VQC ideal (subset): **0.661**
- VQC noisy (subset): **0.623**
- Task A ideal fidelity: **1.000**
- Task A noisy fidelity: **0.880 ± 0.003**
- Processor D QFT-3 noisy fidelity: **0.612**

See `results.md` for full analysis.

## Predictions
`predictions_q1.csv` generated from the best classical model (perfect validation performance) after training on `ecg_train.csv` only. Validated with official tool.

## Ablation Studies
Controlled ablations (PCA dim, ansatz depth, feature-map reps, seed sensitivity) are reported in `results.md` and `results/ablation_results.json`.

## Phase 1 Preservation
Original Phase 1 code is retained under `phase1/`.

## Reproducibility
All random seeds set to 42. Preprocessing (StandardScaler + PCA + MinMax) fitted exclusively on training data. Test data never used for fitting or tuning.
