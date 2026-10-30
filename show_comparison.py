import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


def load_json(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        print(f"Warning: {filename} not found.")
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    classical = load_json("classical_results.json")
    quantum = load_json("quantum_results.json")
    ablation = load_json("ablation_results.json")

    classical_models = classical.get("all", {})
    best_name = classical.get("best", "unknown")
    best_model = classical_models.get(best_name, {})

    classical_f1 = best_model.get("macro_f1")
    classical_accuracy = best_model.get("accuracy")

    ideal_f1 = quantum.get("ideal_subset_macro_f1")
    noisy_f1 = quantum.get("noisy_subset_macro_f1")

    print("=" * 90)
    print("BIOMEDICAL SIGNAL CLASSIFICATION - MODEL COMPARISON")
    print("=" * 90)

    print(f"{'Metric':<28}{'Classical ML':<22}"
          f"{'Quantum Ideal':<20}{'Quantum Noisy':<20}")
    print("-" * 90)

    print(f"{'Best model':<28}{best_name:<22}"
          f"{'VQC':<20}{'VQC':<20}")

    print(f"{'Qubits':<28}{'N/A':<22}"
          f"{str(quantum.get('n_qubits', 'N/A')):<20}"
          f"{str(quantum.get('n_qubits', 'N/A')):<20}")

    def fmt(value):
        return f"{value:.3f}" if isinstance(value, (int, float)) else "Unavailable"

    print(f"{'Macro-F1':<28}{fmt(classical_f1):<22}"
          f"{fmt(ideal_f1):<20}{fmt(noisy_f1):<20}")

    print(f"{'Accuracy':<28}{fmt(classical_accuracy):<22}"
          f"{'Not recorded':<20}{'Not recorded':<20}")

    print(f"{'Quantum noise':<28}{'N/A':<22}"
          f"{'Disabled':<20}{'Enabled':<20}")

    print("-" * 90)
    print("\nQUANTUM NOISE ANALYSIS")
    print("-" * 40)

    if isinstance(ideal_f1, (int, float)) and isinstance(noisy_f1, (int, float)):
        decrease = ideal_f1 - noisy_f1
        print(f"Ideal VQC Macro-F1       : {ideal_f1:.3f}")
        print(f"Noisy VQC Macro-F1       : {noisy_f1:.3f}")
        print(f"Score decrease           : {decrease:.3f}")
        print(f"Percentage-point decrease: {decrease * 100:.2f}")
    else:
        print("Noise comparison unavailable: required scores are missing.")

    print("\nCLASSICAL MODEL RESULTS")
    print("-" * 40)

    for name, metrics in classical_models.items():
        print(
            f"{name:<25} "
            f"Macro-F1={fmt(metrics.get('macro_f1'))}, "
            f"Accuracy={fmt(metrics.get('accuracy'))}"
        )

    print("\nABLATION RESULTS")
    print("-" * 40)

    entries = ablation.get("ablations", [])

    if not entries:
        print("No ablation results available.")
    else:
        for item in entries:
            name = item.get("ablation", "unknown")
            score = item.get("ideal_macro_f1")
            details = ", ".join(
                f"{key}={item[key]}"
                for key in (
                    "n_qubits",
                    "n_components",
                    "ansatz_reps",
                    "fm_reps",
                    "seed"
                )
                if key in item
            )
            print(
                f"{name:<20} "
                f"Macro-F1={fmt(score)}"
                + (f" ({details})" if details else "")
            )

            noisy_score = item.get("noisy_macro_f1")
            if noisy_score is not None:
                print(f"{'':20} Noisy Macro-F1={fmt(noisy_score)}")

    print("\nINTERPRETATION")
    print("-" * 40)
    print("These are reported experiment results, not proof of quantum advantage.")
    print("Classical scores should be checked for data leakage.")
    print("Main quantum and ablation scores may use different configurations")
    print("or evaluation subsets and should not be compared without verification.")
    print("=" * 90)


if __name__ == "__main__":
    main()
