# QFF 2026 Phase 2 — PS-Q1 Biomedical Signal Classifier

## 1. Executive Summary

This project evaluates classical machine-learning baselines and a Variational Quantum Classifier (VQC) for biomedical signal classification. It also evaluates a Bell-state circuit and a QFT-3 circuit under ideal and noisy simulation conditions.

The saved results show perfect validation performance for the three classical baselines. The latest saved VQC experiment achieved a Macro-F1 score of 0.8516 on the full ideal validation set. Its noisy result was evaluated on a separate 300-sample subset, so the two scores must not be treated as a strict same-sample comparison.

The experiments demonstrate the feasibility of quantum-circuit simulation for this task, but they do not establish a quantum advantage over classical methods.

## 2. Experimental Setup

- Task: Biomedical signal classification.
- Classical models: Logistic Regression, Random Forest, and RBF-kernel SVM.
- Quantum model: Variational Quantum Classifier (VQC).
- Feature processing: PCA and a quantum feature map.
- Optimizer: COBYLA for the saved VQC experiments.
- Quantum evaluation: Ideal simulation and noisy simulation where recorded.
- Random seeds, validation sizes, circuit settings, and run times are reported from the saved result files.

The classical and quantum experiments use different model structures and, in some cases, different evaluation subsets. Results should therefore be interpreted with their corresponding experiment details.

## 3. Classical Model Comparison

Saved results from `results/classical_results.json`:

| Model | Accuracy | Macro-F1 | Training Time (s) |
|---|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 0.0177 |
| Random Forest | 1.0000 | 1.0000 | 0.5150 |
| SVM (RBF) | 1.0000 | 1.0000 | 0.1598 |

All three models have the saved confusion matrix:

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 899 | 0 |
| Class 1 | 0 | 601 |

The saved best model is Logistic Regression. It reaches the same recorded validation scores as the other classical models and has the shortest training time among the three.

These perfect scores warrant further dataset and feature-provenance checks before making claims about generalization to new patients or independently collected data. They should not, by themselves, be taken as proof of data leakage.

## 4. Quantum Model Results

### 4.1 Earlier VQC experiment

Saved in `results/quantum_results.json`:

| Setting | Value |
|---|---|
| Feature encoding | ZZFeatureMap with PCA(4) |
| Qubits | 4 |
| Ansatz | RealAmplitudes, 2 repetitions, linear entanglement |
| Trainable parameters | 12 |
| Optimizer | COBYLA |
| Maximum iterations | 15 |
| Training subset | 100 samples |
| Training time | 1.4015 s |
| Ideal subset Macro-F1 | 0.6612 |
| Noisy subset Macro-F1 | 0.6229 |

This is a subset-based experiment and should not be presented as the latest full-validation result.

### 4.2 Latest VQC experiment

Saved in `results/quantum_reps3_full_validation.json`:

| Setting | Value |
|---|---|
| Qubits | 4 |
| Ansatz repetitions | 3 |
| Trainable parameters | 16 |
| Optimizer iterations (maximum) | 30 |
| Training subset | 200 samples |
| PCA explained variance | 0.8128 |
| Training time | 7.9478 s |
| Ideal validation size | 1,500 samples |
| Ideal accuracy | 0.8653 |
| Ideal Macro-F1 | 0.8516 |
| Noisy validation subset | 300 samples |
| Noisy subset accuracy | 0.8633 |
| Noisy subset Macro-F1 | 0.8520 |

Ideal full-validation confusion matrix:

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 877 | 22 |
| Class 1 | 180 | 421 |

The latest VQC's ideal Macro-F1 is higher than the earlier VQC's saved subset score. However, these runs differ in configuration, training subset, and evaluation protocol, so the difference should not be attributed to ansatz repetitions alone.

The latest ideal result uses 1,500 validation samples, whereas its noisy result uses a fixed-seed 300-sample subset. A fair ideal-versus-noisy comparison requires evaluating both conditions on the same 300 samples.

## 5. Classical Versus Quantum Comparison

| Experiment | Evaluation condition | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Logistic Regression | Saved classical validation | 1.0000 | 1.0000 |
| Random Forest | Saved classical validation | 1.0000 | 1.0000 |
| SVM (RBF) | Saved classical validation | 1.0000 | 1.0000 |
| Earlier VQC | Ideal subset | Not saved | 0.6612 |
| Earlier VQC | Noisy subset | Not saved | 0.6229 |
| Latest VQC | Ideal, full validation (1,500) | 0.8653 | 0.8516 |
| Latest VQC | Noisy, subset (300) | 0.8633 | 0.8520 |

**Interpretation:** the classical models have the highest saved validation metrics. The VQC results show useful experimental behavior but do not outperform the classical baselines in the recorded results. Because evaluation protocols and sample counts differ, this table is descriptive rather than a controlled head-to-head comparison.

## 6. Ablation Study

The saved ablation file specifies a fixed 60-sample validation subset. PCA and scaling were fitted on training data only. Full noisy evaluation was performed only for the main four-qubit configuration. These results are separate from the latest full-validation experiment above.

### 6.1 PCA dimensions and qubit count

| PCA Components / Qubits | Explained Variance | Parameters | Ideal Macro-F1 | Noisy Macro-F1 |
|---:|---:|---:|---:|---:|
| 2 | 0.6930 | 6 | 0.7962 | Not evaluated |
| 4 | 0.8128 | 12 | 0.6760 | 0.6419 |
| 6 | 0.8708 | 18 | 0.7304 | Not evaluated |

Increasing the number of components increases explained variance and circuit parameter count, but it does not produce a monotonic improvement in Macro-F1. Noisy scores are unavailable for the two- and six-qubit settings in the saved results.

### 6.2 Ansatz repetitions

| Ansatz Repetitions | Parameters | Ideal Macro-F1 | Training Time (s) |
|---:|---:|---:|---:|
| 1 | 8 | 0.6267 | 1.52 |
| 2 | 12 | 0.6760 | 2.67 |
| 3 | 16 | 0.8565 | 3.69 |

Within this particular ablation, three repetitions achieved the highest recorded ideal Macro-F1. This is a result for the fixed ablation subset, not a guarantee that deeper circuits will always perform better.

### 6.3 Feature-map repetitions

| Feature-Map Repetitions | Parameters | Ideal Macro-F1 | Training Time (s) |
|---:|---:|---:|---:|
| 1 | 12 | 0.6760 | 3.04 |
| 2 | 12 | 0.7306 | 3.97 |

Two feature-map repetitions improved the recorded ideal Macro-F1 in this ablation, at the cost of additional training time.

### 6.4 Random seed sensitivity

| Seed | Ideal Macro-F1 | Training Time (s) |
|---:|---:|---:|
| 42 | 0.6571 | 3.49 |
| 7 | 0.8369 | 3.38 |
| 123 | 0.8942 | 3.58 |
| Saved mean ± standard deviation | 0.7961 ± 0.1010 | — |

The variation across seeds indicates sensitivity to initialization and/or optimization. The saved aggregate statistics are retained as recorded; small differences may occur if the aggregate is recalculated from rounded table values.

## 7. Task A — Bell-State Benchmark

The Bell-state benchmark uses 8,192 shots and the recorded noise model `QFF2026-v2`.

The estimated fidelity is:

F = (1 + <XX> - <YY> + <ZZ>) / 4

| Measurement | Ideal | Noisy |
|---|---:|---:|
| XX | 1.0000 ± 0.0000 | 0.8191 ± 0.0063 |
| YY | -1.0000 ± 0.0000 | -0.8279 ± 0.0062 |
| ZZ | 1.0000 ± 0.0000 | 0.8735 ± 0.0054 |
| Fidelity | 1.0000 ± 0.0000 | 0.8801 ± 0.0026 |

The ideal circuit produces unit fidelity in the saved simulation. The noisy simulation reduces fidelity to approximately 0.8801, illustrating the effect of the configured noise model.

Noise configuration recorded in the report: CX depolarizing probability 0.05 and measurement readout probability 0.02.

## 8. Task B — Processor D QFT-3 Benchmark

Saved benchmark configuration: `psq1_linear4_v1`.

| Metric | Recorded Result |
|---|---|
| Processor configuration | Processor D |
| Physical qubits | 4 |
| Topology | Linear, bidirectional |
| QFT-3 depth before transpilation | 6 |
| Depth after transpilation | 20 |
| Two-qubit gates | 12 |
| SWAP overhead | 1 |
| Ideal fidelity | 1.0000 |
| Noisy fidelity | 0.611593 |

The transpiled depth is greater than the original depth because the circuit must be mapped to the processor topology. The noisy fidelity is lower than the ideal fidelity, indicating sensitivity to the simulated hardware constraints.

Only Processor D has verified results in the current report. Processor A/B comparisons must be added only if corresponding measurements are available.

## 9. Resource and Runtime Analysis

| Resource | Recorded Information |
|---|---|
| Main VQC qubits | 4 |
| Older VQC trainable parameters | 12 |
| Latest VQC trainable parameters | 16 |
| Approximate logical circuit depth | 15–25, as previously recorded |
| Earlier VQC training subset/time | 100 samples / 1.4015 s |
| Latest VQC training subset/time | 200 samples / 7.9478 s |
| Classical Logistic Regression training time | 0.0177 s |
| Classical Random Forest training time | 0.5150 s |
| Classical SVM training time | 0.1598 s |

Runtime depends on the experiment configuration, hardware, optimizer settings, and simulation method. The recorded timings are local experiment measurements and should not be treated as universal hardware benchmarks.

## 10. Limitations and Threats to Validity

1. All three classical models achieve perfect saved validation performance. Dataset provenance, feature construction, class separability, and split independence should be reviewed before making generalization claims.
2. The VQC does not outperform the classical baselines in the recorded experiments.
3. The latest ideal and noisy VQC results use different validation sample counts. A matched-subset comparison is needed for a strict noise-impact estimate.
4. Ablations use a fixed 60-sample validation subset, so their scores may be noisy and should not be directly compared with full-validation scores.
5. Noisy quantum simulation can be computationally expensive, particularly when evaluating large validation sets.
6. Results depend on initialization, feature-map settings, ansatz depth, optimization, noise assumptions, and shot count.
7. Only Processor D has verified hardware-topology benchmark results in the current report.
8. These results are experimental and do not demonstrate a quantum advantage or clinical readiness.

## 11. Reproducibility and Saved Artifacts

The report uses saved experiment results, including:

- `results/classical_results.json`
- `results/quantum_results.json`
- `results/quantum_reps3_full_validation.json`
- `results/ablation_results.json`
- `results/task_a_results.json`

The report also refers to the existing benchmark configuration and graph artifacts under `results/graphs/`.

For the Processor D benchmark, the previously recorded command is:

`python benchmark/qff_qml_benchmark.py processor_d_config.json`

Run a benchmark only after confirming the required files, dependencies, and working directory are present. The numerical tables in this report are based on saved results and do not require retraining to read.

## 12. Graphs

The existing graph files are under `results/graphs/`. If a graph does not render in a Markdown viewer, check that the image link uses standard Markdown syntax and a relative path, for example:

`![Classical model comparison](results/graphs/01_classical_model_comparison.png)`

Existing graph filenames include:

- `01_classical_model_comparison.png`
- `02_classical_training_time.png`
- `03_classical_vs_vqc_full_validation.png`
- `04_same_300_sample_comparison.png`
- `05_vqc_confusion_matrix.png`
- `06_ablation_pca_qubits.png`
- `07_pca_explained_variance.png`
- `08_ablation_ansatz_repetitions.png`
- `09_ablation_feature_map_repetitions.png`
- `10_seed_sensitivity.png`
- `11_ablation_training_times.png`
- `12_task_a_ideal_vs_noisy.png`
- `13_training_class_distribution.png`
- `14_train_validation_class_distribution.png`
- `15_w5_mean_by_class.png`

## 13. Conclusion

The experiments compare classical baselines with a variational quantum classifier and evaluate quantum circuits under ideal and noisy simulation. Logistic Regression, Random Forest, and RBF-SVM achieved perfect saved validation metrics, while the latest VQC achieved an ideal full-validation Macro-F1 of 0.8516. The Bell-state benchmark retained approximately 0.8801 fidelity under the configured noise model, and Processor D's QFT-3 benchmark recorded noisy fidelity of 0.611593.

The results demonstrate that circuit depth, feature-map settings, ansatz repetitions, random initialization, and noise affect quantum-model performance. However, the current evidence does not show quantum advantage over classical methods. Future work should include matched-subset ideal/noisy evaluation, additional seeds, independent dataset validation, and further verified processor comparisons.

## 14. Fair Same-Subset Comparison: Classical Models vs Noisy VQC

The saved file `results/classical_same_subset_comparison.json` records classical metrics on the same 300 validation samples used for the latest noisy VQC evaluation.

| Model | Evaluation subset | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Logistic Regression | Same 300 samples | 1.0000 | 1.0000 |
| Random Forest | Same 300 samples | 1.0000 | 1.0000 |
| SVM (RBF) | Same 300 samples | 1.0000 | 1.0000 |
| Latest VQC | Noisy, same 300 samples | 0.8633 | 0.8520 |

This is the fairest available comparison for the noisy VQC result because all models are evaluated on the same sample count and validation subset. The classical models outperform the noisy VQC on both recorded metrics. This comparison does not establish how either approach will perform on an independent external dataset.

## 15. Feature-Level Dataset Diagnostics

The saved diagnostic file is `results/deep_data_diagnostic.json`. It evaluates individual engineered features on the validation set.

| Feature | Validation Accuracy |
|---|---:|
| w5_mean | 1.0000 |
| w5_max | 0.9980 |
| w3_kurt | 0.9907 |
| w2_std | 0.9887 |
| w3_min | 0.9740 |
| w4_std | 0.9713 |
| w3_mean | 0.9687 |
| w5_min | 0.9673 |
| w5_std | 0.9647 |
| w4_max | 0.9627 |

The `w5_mean` feature alone reaches 100% validation accuracy in this diagnostic. The high scores for several other individual features indicate that the engineered features carry strong class-separating information.

The saved feature-range checks report overlapping training and validation ranges for all 30 inspected features. Range overlap is useful evidence, but it does not independently prove that the dataset split is free of leakage or duplicate records.

### Required interpretation

The perfect classical results may be genuine for this dataset and feature representation, but the exceptionally strong single-feature performance deserves investigation. Before making generalization claims:

1. Verify the origin and meaning of each engineered feature.
2. Confirm that feature generation does not use the target label or information unavailable at prediction time.
3. Check that training and validation records are split appropriately, ideally by patient or original recording when those identifiers are available.
4. Evaluate on a genuinely independent dataset if available.

These diagnostics are reasons for careful validation, not proof of data leakage.

## 16. Updated Overall Conclusion

The saved experiments show that Logistic Regression, Random Forest, and RBF-SVM achieved perfect validation scores, including on the same 300 samples used for the latest noisy VQC evaluation. The latest VQC achieved an ideal full-validation Macro-F1 of 0.8516 and a noisy-subset Macro-F1 of 0.8520; the noisy score is lower than the classical models' score on that same subset.

The Bell-state and Processor D QFT-3 benchmarks also show reduced fidelity under the configured noisy simulation. Ablation results indicate that model configuration and random seed affect VQC performance.

The evidence supports a comparison of classical and quantum approaches, but it does not demonstrate quantum advantage. The unusually high classical scores and single-feature accuracy should be investigated, and independent validation is needed before making claims about generalization or practical biomedical use.

## 17. Numerical Comparison: Classical ML vs Quantum ML

### 17.1 Overall Model Performance

| Model | Accuracy | Macro-F1 | Training Time (s) |
|---|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 0.0177 |
| Random Forest | 1.0000 | 1.0000 | 0.5150 |
| SVM (RBF) | 1.0000 | 1.0000 | 0.1598 |
| VQC - Earlier Ideal Subset | Not saved | 0.6612 | 1.4015 |
| VQC - Earlier Noisy Subset | Not saved | 0.6229 | 1.4015* |
| VQC - Latest Ideal Full Validation | 0.8653 | 0.8516 | 7.9478 |
| VQC - Latest Noisy Subset | 0.8633 | 0.8520 | 7.9478* |

*VQC training time is the recorded training time for that experiment, not a separate noisy-evaluation runtime.

### 17.2 Fair Comparison on the Same 300 Validation Samples

| Model | Accuracy | Macro-F1 | Rank |
|---|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 1 (tie) |
| Random Forest | 1.0000 | 1.0000 | 1 (tie) |
| SVM (RBF) | 1.0000 | 1.0000 | 1 (tie) |
| Noisy VQC | 0.8633 | 0.8520 | 2 |

The three classical models outperform the noisy VQC on both metrics in the saved same-subset comparison. The classical models tie on performance, so training time distinguishes Logistic Regression as the most efficient of the three in this experiment.

### 17.3 Training-Time Comparison

| Model | Recorded Training Time (s) | Approximate Time Relative to Logistic Regression |
|---|---:|---:|
| Logistic Regression | 0.0177 | 1x |
| SVM (RBF) | 0.1598 | 9.0x |
| Random Forest | 0.5150 | 29.1x |
| Latest VQC | 7.9478 | 449.0x |

These are recorded experiment times, not a controlled runtime benchmark: the models use different training procedures and sample sizes. The ratios describe the observed timings only.

### 17.4 Quantum Model Ablation Results

| Experiment | Configuration | Ideal Macro-F1 | Training Time (s) |
|---|---|---:|---:|
| PCA/Qubits | 2 components/qubits | 0.7962 | 1.42 |
| PCA/Qubits | 4 components/qubits | 0.6760 | 2.49 |
| PCA/Qubits | 6 components/qubits | 0.7304 | 4.21 |
| Ansatz repetitions | 1 | 0.6267 | 1.52 |
| Ansatz repetitions | 2 | 0.6760 | 2.67 |
| Ansatz repetitions | 3 | 0.8565 | 3.69 |
| Feature-map repetitions | 1 | 0.6760 | 3.04 |
| Feature-map repetitions | 2 | 0.7306 | 3.97 |

These ablations use a fixed 60-sample validation subset and should not be ranked directly against the main full-validation results.

### 17.5 Quantum Circuit Benchmark Results

| Benchmark | Ideal Fidelity | Noisy Fidelity | Observed Change |
|---|---:|---:|---:|
| Bell-state benchmark | 1.0000 | 0.8801 | -0.1199 |
| Processor D QFT-3 | 1.0000 | 0.6116 | -0.3884 |

These results show that the configured noise model reduces the fidelity of both benchmark circuits. They are circuit benchmarks, not classification-performance metrics.

### 17.6 Final Decision Based on Recorded Evidence

| Decision Criterion | Winner | Evidence |
|---|---|---|
| Highest saved classification accuracy | Classical ML | 1.0000 on the same 300 samples |
| Highest saved Macro-F1 | Classical ML | 1.0000 on the same 300 samples |
| Shortest recorded training time | Logistic Regression | 0.0177 seconds |
| Best classical model for this experiment | Logistic Regression | Ties on saved metrics and has the shortest recorded training time |
| Quantum-circuit research objective | VQC | Provides a quantum-model comparison and supports circuit ablation experiments |

**Overall winner: Classical ML, specifically Logistic Regression, for the current experiment.**

Logistic Regression matches the saved validation performance of Random Forest and RBF-SVM while having the lowest recorded training time. On the same 300 validation samples, all three classical models also outperform the noisy VQC.

This conclusion applies to the saved experiments only. The perfect classical scores and the individual feature `w5_mean` achieving 1.0000 validation accuracy warrant checks for target leakage, overlapping windows, duplicate records, and patient-level or recording-level split independence. The results do not establish quantum advantage or guarantee generalization to independent biomedical data.

