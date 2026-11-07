from lib.split import split
from lib.shared import ensemble_varying_k, calculate_error, predict
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits

if __name__ == '__main__':

    datasets = ["Iris", "Breast Cancer", "Wine", "Digits"]
    loaders = [load_iris, load_breast_cancer, load_wine, load_digits]

    benchmark_best_k = [3, 5, 9, 3]
    k_values = [3, 5, 7, 9, 11, 13, 15]

    for dataset_name, loader, best_k in zip(datasets, loaders, benchmark_best_k):

        Z1, Z2, Z3 = split(loader)

        true_labels = [sample[1] for sample in Z3]

        # Benchmark: single KNN using the best K from previous assignment
        print(f"\nBenchmark {dataset_name} com K={best_k}")

        benchmark_predictions = predict(Z1, Z3, best_k)

        benchmark_error = calculate_error(
            benchmark_predictions,
            true_labels
        )

        print(f"Benchmark Error: {benchmark_error:.4f}")

        # Experiment 1: ensemble varying K
        print(f"\nCombinação {dataset_name} variando K")

        predictions = ensemble_varying_k(Z1, Z3, k_values)

        error = calculate_error(predictions, true_labels)

        print(f"Ensemble Error: {error:.4f}")