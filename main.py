from lib.split import split_z1_z3
from lib.shared import (
    ensemble_varying_k,
    ensemble_varying_z1,
    calculate_error,
    predict
)

from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    load_wine,
    load_digits
)


if __name__ == '__main__':

    datasets = ["Iris", "Breast Cancer", "Wine", "Digits"]
    loaders = [load_iris, load_breast_cancer, load_wine, load_digits]

    benchmark_best_k = [3, 5, 9, 3]
    k_values = [3, 5, 7, 9, 11, 13, 15]

    seeds = [42, 43, 44, 45, 46]

    for dataset_name, loader, best_k in zip(
        datasets, loaders, benchmark_best_k
    ):

        Z1, Z3 = split_z1_z3(loader)

        true_labels = [sample[1] for sample in Z3]

        # Benchmark: single KNN
        print(f"\nBenchmark {dataset_name} com K={best_k}")

        benchmark_predictions = predict(Z1, Z3, best_k)

        benchmark_error = calculate_error(
            benchmark_predictions,
            true_labels
        )

        print(f"Erro: {benchmark_error:.4f}")

        # Experiment 1: ensemble varying K
        print(f"\nCombinação {dataset_name} variando K")

        predictions = ensemble_varying_k(Z1, Z3, k_values)

        error = calculate_error(predictions, true_labels)

        print(f"Erro: {error:.4f}")

        # Experiment 2: ensemble varying Z1
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