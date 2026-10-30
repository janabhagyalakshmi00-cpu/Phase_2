# QFF 2026 Phase 2 — Biomedical Signal Classifier

### Quantum Machine Learning Track | PS-Q1

A hybrid Quantum Machine Learning project for biomedical signal classification using classical machine learning models and a Variational Quantum Classifier (VQC).

The project evaluates classical baselines and quantum classification under ideal and simulated noisy conditions. It also explores model configurations, PCA dimensionality reduction, circuit parameters, noise effects, and processor configuration.

---

## 1. Project Overview

Biomedical signal classification uses extracted signal features to distinguish between different classes.

This project investigates whether a Variational Quantum Classifier can classify biomedical signal features and compares its performance with classical machine learning algorithms.

### Objectives

- Build classical machine learning baselines.
- Implement a Variational Quantum Classifier.
- Evaluate quantum classification under ideal and noisy simulation conditions.
- Compare validation accuracy, F1-score, and execution time.
- Investigate PCA components, ansatz repetitions, feature-map repetitions, and random seed sensitivity.
- Generate classification predictions and visualizations.
- Document the experimental results for reproducibility.

---

## 2. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Jupyter Notebook | Experiment execution and analysis |
| NumPy | Numerical computation |
| Pandas | Data handling |
| Scikit-learn | Classical ML models, preprocessing, and evaluation |
| Qiskit | Quantum circuit development and simulation |
| Matplotlib | Results visualization |
| JSON | Experiment results and configuration files |
| Git and GitHub | Version control and project hosting |

Refer to `requirements.txt` for the project's Python dependencies.

---

## 3. Models Implemented

### Classical Machine Learning

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (RBF kernel)

### Quantum Machine Learning

- Variational Quantum Classifier (VQC)
- Four-qubit quantum circuit
- Parameterized quantum ansatz
- PCA-based dimensionality reduction
- Ideal quantum simulation
- Simulated noisy quantum evaluation

The quantum circuit is evaluated using simulated conditions. The reported results do not establish execution on real quantum hardware.

---

## 4. Project Structure

```text
Biomedical-Signal-Classifier-Phase2/
│
├── README.md
├── requirements.txt
├── solution.ipynb
├── executed_solution.ipynb
├── executed_solution.html
├── predictions_q1.csv
├── results.md
├── results_before_graph_fix.md
│
├── data/
│   ├── q1/
│   ├── q2/
│   └── q3/
│
├── src/
│   ├── classical_baseline.py
│   ├── data_utils.py
│   ├── quantum_vqc.py
│   └── ablation.py
│
├── models/
│   └── vqc_theta.npy
│
├── benchmark/
│   └── qff_qml_benchmark.py
│
├── noise_model/
│   └── benchmark_noise_model.py
│
├── schemas/
│   ├── q1_feature_dictionary.md
│   ├── q2_feature_dictionary.md
│   ├── q3_feature_dictionary.md
│   └── processor_d_schema.json
│
├── submission_templates/
│
├── tools/
│   └── validate_predictions.py
│
├── results/
│   ├── classical_results.json
│   ├── quantum_results.json
│   ├── quantum_reps3_full_validation.json
│   ├── task_a_results.json
│   ├── ablation_results.json
│   └── graphs/
│       ├── 01_classical_model_comparison.png
│       ├── 02_classical_training_time.png
│       ├── 03_classical_vs_vqc_full_validation.png
│       ├── 04_same_300_sample_comparison.png
│       ├── 05_vqc_confusion_matrix.png
│       ├── 06_ablation_pca_qubits.png
│       ├── 07_pca_explained_variance.png
│       ├── 08_ablation_ansatz_repetitions.png
│       ├── 09_ablation_feature_map_repetitions.png
│       ├── 10_seed_sensitivity.png
│       ├── 11_ablation_training_times.png
│       ├── 12_task_a_ideal_vs_noisy.png
│       ├── 13_training_class_distribution.png
│       ├── 14_train_validation_class_distribution.png
│       ├── 15_w5_mean_by_class.png
│       └── 16_validation_accuracy_and_loss.png
│
└── phase1/
    └── Previous phase implementation and results
```

*Note: This structure summarizes the main project files. Some directories contain additional experiment outputs, configuration files, and tests.*

---

## 5. System Requirements

- Windows, Linux, or macOS
- Python 3.10 or another version compatible with the installed dependencies
- Git
- Visual Studio Code (recommended)
- Jupyter Notebook support

Check your Python installation:

```bash
python --version
```

Check Git:

```bash
git --version
```

---

## 6. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/janabhagyalakshmi00-cpu/Phase_2.git
```

Navigate into the repository:

```bash
cd Phase_2
```

### Step 2: Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

If installation fails, check the error message and use the Python and dependency versions supported by the project.

---

## 7. How to Run the Project

### Option A: Run in Jupyter Notebook

Install Jupyter if it is not already available:

```bash
python -m pip install jupyter
```

Start Jupyter:

```bash
python -m jupyter notebook
```

Open `solution.ipynb` in the browser.

Execute the notebook cells in order, starting from the first cell.

### Option B: Execute the Notebook from PowerShell

From the project directory, run:

```powershell
python -m jupyter nbconvert --to notebook --execute solution.ipynb --output executed_solution.ipynb
```

This executes the notebook and saves the executed version as `executed_solution.ipynb`.

Export it as HTML:

```powershell
python -m jupyter nbconvert --to html executed_solution.ipynb --output executed_solution.html
```

Open the generated report:

```powershell
Start-Process .\executed_solution.html
```

The HTML file can be used to review notebook outputs without executing the notebook again.

---

## 8. Prediction Output

The project generates a prediction file:

```text
predictions_q1.csv
```

The previously executed notebook produced a CSV with **1,500 rows and 2 columns**.

Check the output using Python:

```python
import pandas as pd

predictions = pd.read_csv("predictions_q1.csv")

print(predictions.head())
print("Shape:", predictions.shape)
print("Columns:", predictions.columns.tolist())
```

Expected shape from the recorded execution:

```text
Shape: (1500, 2)
```

The exact column names and prediction format should be verified against the competition's official submission requirements.

Validate the file using the project's validation utility:

```powershell
python tools/validate_predictions.py
```

If the utility requires command-line arguments, check its usage in `tools/validate_predictions.py` before running it.

---

## 9. Experimental Results

The following results were recorded in the Phase 2 experiments.

### Classical Model Performance

| Model | Validation Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic Regression | 100.00% | 1.0000 |
| Random Forest | 100.00% | 1.0000 |
| SVM (RBF) | 100.00% | 1.0000 |

### Quantum Model Performance

| Model | Evaluation Samples | Accuracy | F1-Score |
|---|---:|---:|---:|
| VQC — Ideal | 1,500 | 86.53% | 0.8516 |
| VQC — Noisy | 300 | 86.33% | 0.8520 |

**Important:** The ideal and noisy VQC results above use different evaluation sample counts. They should not be interpreted as a strict, same-sample comparison.

The classical models' perfect scores should also be independently verified using a leakage-free evaluation pipeline before making final performance claims.

Results may vary if preprocessing, data splits, random seeds, package versions, or circuit settings change.

---

## 10. Validation Accuracy and Error Percentage

The project includes the graph:

```text
results/graphs/16_validation_accuracy_and_loss.png
```

The displayed error percentage is calculated as:

```text
Error Percentage = 100 - Validation Accuracy
```

### Recorded Results

| Model | Accuracy | Error Percentage |
|---|---:|---:|
| Logistic Regression | 100.00% | 0.00% |
| Random Forest | 100.00% | 0.00% |
| SVM (RBF) | 100.00% | 0.00% |
| VQC — Ideal | 86.53% | 13.47% |
| VQC — Noisy | 86.33% | 13.67% |

**Metric clarification:** Error percentage is not the same as the training loss function. It represents the percentage of incorrectly classified samples when computed as 100% minus accuracy.

Other visualizations are available in `results/graphs/`, including:

- Classical model comparisons
- Classical training time comparisons
- Classical versus quantum model comparisons
- VQC confusion matrix
- PCA ablation experiments
- Ansatz repetition experiments
- Feature-map repetition experiments
- Random seed sensitivity
- Training and validation class distributions

---

## 11. Results and Analysis

The experiments provide a comparison between classical classification algorithms and a parameterized quantum classifier.

Key observations:

1. The recorded classical results show 100% accuracy, which requires further validation to rule out data leakage or an overly easy evaluation split.
2. The VQC achieved 86.53% accuracy in the recorded ideal evaluation.
3. The noisy VQC evaluation achieved 86.33% accuracy on a smaller evaluation subset.
4. PCA and circuit configuration experiments help investigate the effects of dimensionality and parameterization on performance.
5. Quantum simulation results are dependent on the circuit, optimizer, data encoding, measurement shots, and noise assumptions.

These results do not demonstrate quantum advantage. The current experiments should be treated as an experimental comparison between classical and quantum machine learning approaches.

---

## 12. Reproducibility

Important files for reviewing the experiments include:

| File | Purpose |
|---|---|
| `solution.ipynb` | Main experiment notebook |
| `executed_solution.ipynb` | Notebook containing saved execution outputs |
| `executed_solution.html` | HTML version of the executed notebook |
| `results.md` | Experiment summary |
| `results/classical_results.json` | Classical model results |
| `results/quantum_results.json` | Earlier quantum experiment results |
| `results/quantum_reps3_full_validation.json` | Recorded VQC evaluation results |
| `results/ablation_results.json` | Ablation experiment results |
| `results/task_a_results.json` | Task A experiment results |
| `processor_d_config.json` | Processor D configuration |
| `tools/validate_predictions.py` | Prediction validation utility |

For reproducibility, preserve the data splits, random seeds, dependency versions, preprocessing configuration, and model settings used during each experiment.

---

## 13. Limitations

- The reported results depend on the current dataset and evaluation procedure.
- Perfect classical accuracy requires an independent leakage audit.
- Ideal and noisy VQC results were evaluated using different sample counts in the reported experiments.
- No real quantum hardware execution is established by these simulation results.
- No quantum advantage has been demonstrated.
- Re-running experiments may produce different results if seeds or dependencies differ.

---

## 14. Future Improvements

Potential improvements include:

- Perform a strict data leakage audit.
- Compare all models on the same held-out evaluation samples.
- Add cross-validation and confidence intervals.
- Evaluate precision, recall, F1-score, and confusion matrices for every model.
- Compare multiple quantum circuit configurations.
- Record complete dependency versions and experiment configurations.
- Improve noisy quantum circuit performance.
- Investigate execution on supported quantum hardware if available.

---

## 15. GitHub Repository

Project repository:

https://github.com/janabhagyalakshmi00-cpu/Phase_2

---

## 16. Conclusion

This project explores biomedical signal classification using classical machine learning and Variational Quantum Classifiers.

It includes model comparisons, quantum simulation, noisy evaluation, ablation studies, prediction generation, and result visualizations.

The current experiments provide a foundation for further evaluation of hybrid quantum machine learning methods. Additional leakage checks and consistent evaluation procedures are necessary before treating the reported scores as definitive generalization performance.

---

**Project:** QFF 2026 Phase 2 — PS-Q1 Biomedical Signal Classification  
**Track:** Quantum Machine Learning  
**Language:** Python  
**Primary Notebook:** `solution.ipynb`
