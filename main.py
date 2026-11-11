from sklearn.metrics import classification_report
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
import time

from lib.split import select_z3, select_z1
from lib.shared import calculate_error, predict, normalize_datasets, ensemble_varying_k, ensemble_varying_z1

def evaluate_model(predictions, true_labels):
    error = calculate_error(predictions, true_labels)

    print(f"Error rate: {error:.4f}")

    print(classification_report(true_labels, predictions, zero_division=0))


if __name__ == '__main__':

    datasets = ["Iris", "Breast Cancer", "Wine", "Digits"]
    loaders = [load_iris, load_breast_cancer, load_wine, load_digits]

    benchmark_best_k = [9, 3, 3, 3]  # Best K values for each dataset based on previous experiments
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17]  # K values to test in the ensemble experiment
    seeds = list(range(42, 51))

    for dataset_name, loader, best_k in zip(datasets, loaders, benchmark_best_k):

        z3 = select_z3(loader, r3=0.5, seed=42)

        z3_indices = {sample[2] for sample in z3}
        z3_data = [(sample[0], sample[1]) for sample in z3]

        z1 = select_z1(loader, z3_indices, seed=42, r1=0.5)

        z1 = select_z1(loader, z3_indices, seed=42, r1=0.5)
        z1_normalized, z3_normalized = normalize_datasets(z1, z3_data)
        true_labels = [sample[1] for sample in z3_data]

        #### Benchmark: single KNN ####
        benchmark_start = time.perf_counter()
        print(f"\nBenchmark {dataset_name} com K={best_k}")

        benchmark_predictions = predict(z1_normalized, z3_normalized, best_k)

        evaluate_model(
            benchmark_predictions,
            true_labels
        )

        print(f"t: {time.perf_counter() - benchmark_start:.4f} seconds")
        print("--------------------------------------")

        #### Experiment 1: ensemble varying K ####
        experiment_1_start = time.perf_counter()
        print(f"\nCombinação {dataset_name} variando K")

        varying_k_predictions = ensemble_varying_k(z1_normalized, z3_normalized, k_values)

        evaluate_model(
            varying_k_predictions,
            true_labels
        )

        print(f"Execution time: {time.perf_counter() - experiment_1_start:.4f} seconds")
        print("--------------------------------------")

        #### Experiment 2: ensemble varying Z1 ####
        experiment_2_start = time.perf_counter()
        print(f"\nCombinação {dataset_name} variando Z1")

        # New normalized subsets of Z1 and Z3 for each seed
        z1_normalized_subsets = []
        z3_normalized_subsets = []
        
        for seed in seeds:
            z1_subset = select_z1(loader, z3_indices, seed, r1=0.5)
            
            # Normalize Z1 and Z3 according to the specific Z1 subset
            z1_normalized, z3_normalized = normalize_datasets(z1_subset, z3_data)

            z1_normalized_subsets.append(z1_normalized)
            z3_normalized_subsets.append(z3_normalized)

        ensemble_predictions = ensemble_varying_z1(
            z1_normalized_subsets,
            z3_normalized_subsets,
            k=best_k
        )

        evaluate_model(
            ensemble_predictions,
            true_labels
        )
        print(f"Execution time: {time.perf_counter() - experiment_2_start:.4f} seconds")
        print("--------------------------------------")
