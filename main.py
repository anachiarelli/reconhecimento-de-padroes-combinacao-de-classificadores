from sklearn.metrics import classification_report
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits

from lib.split import simple_z1_z3_split
from lib.shared import (calculate_error, predict, ensemble_varying_k)

def evaluate_model(predictions, true_labels):
    error = calculate_error(predictions, true_labels)

    print(f"Error rate: {error:.4f}")

    print(classification_report(true_labels, predictions, zero_division=0))
    print("--------------------------------------")


if __name__ == '__main__':

    datasets = ["Iris", "Breast Cancer", "Wine"]#, "Digits"]
    loaders = [load_iris, load_breast_cancer, load_wine]#, load_digits]

    benchmark_best_k = [9, 3, 3, 3]  # Best K values for each dataset based on previous experiments
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]  # K values to test in the ensemble experiment

    seeds = [42, 43, 44, 45, 46]

    for dataset_name, loader, best_k in zip(
        datasets, loaders, benchmark_best_k
    ):

        z1, z3 = simple_z1_z3_split(loader, r1=0.25, r3=0.5)

        true_labels = [sample[1] for sample in z3]

        # Benchmark: single KNN
        print(f"\nBenchmark {dataset_name} com K={best_k}")

        benchmark_predictions = predict(z1, z3, best_k)

        evaluate_model(
            benchmark_predictions,
            true_labels,
            dataset_name,
            f"Benchmark K={best_k}"
        )

        # Experiment 1: ensemble varying K
        print(f"\nCombinação {dataset_name} variando K")

        varying_k_predictions = ensemble_varying_k(z1, z3, k_values)

        evaluate_model(
            varying_k_predictions,
            true_labels,
            dataset_name,
            f"Ensemble K={k_values}"
        )

        # Experiment 2: ensemble varying Z1
        '''
        print(f"\nCombinação {dataset_name} variando Z1")

        Z1_subsets = [
            split_z1_z3(loader, seed=seed)[0]
            for seed in seeds
        ]

        Z3_subset = split_z1_z3(loader, seed=42)[1]
        true_labels = [sample[1] for sample in Z3_subset]

        ensemble_predictions = ensemble_varying_z1(
            Z1_subsets,
            Z3_subset,
            best_k
        )

        ensemble_error = calculate_error(
            ensemble_predictions,
            true_labels
        )

        print(f"Erro: {ensemble_error:.4f}")
        print("--------------------------------------")
        '''