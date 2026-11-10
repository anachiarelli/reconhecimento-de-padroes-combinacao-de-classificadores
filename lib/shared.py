import statistics
import random

def euclidean_distance(a:list[float], b:list[float]) -> float:
    sum = 0.0
    for x_i, y_i in zip(a, b):
        sum += (x_i - y_i) ** 2
    return sum ** 0.5

def predict(base, target, k):
    base_X = [b[0] for b in base]
    base_y = [b[1] for b in base]

    target_X = [t[0] for t in target]

    predictions = []
    for t in target_X:
        distances_by_class = [(euclidean_distance(t, b), b_class) for b, b_class in zip(base_X, base_y)]
        distances_by_class.sort()
        k_nearest_neighbors = distances_by_class[:k]

        predicted_class = statistics.mode([neighbor[1] for neighbor in k_nearest_neighbors])
        predictions.append(predicted_class)

    return predictions

def calculate_error(predictions, true_labels):
    errors = 0
    for predicted_class, true_class in zip(predictions, true_labels):
        if predicted_class != true_class:
            errors += 1
    return errors / len(true_labels)

def ensemble_varying_k(base, target, k_values):
    predictions_by_classifier = []

    for k in k_values:
        predictions = predict(base, target, k)
        predictions_by_classifier.append(predictions)

    ensemble_predictions = []

    for sample_predictions in zip(*predictions_by_classifier):
        votes = {}

        for prediction in sample_predictions:
            votes[prediction] = votes.get(prediction, 0) + 1
        max_votes = max(votes.values())

        winners = [
            target_class
            for target_class, count in votes.items()
            if count == max_votes
        ]

        # Solves ties
        ensemble_predictions.append(random.choice(winners))

    return ensemble_predictions

def ensemble_varying_z1(Z1_subsets, Z3, k):
    predictions_by_classifier = []

    for z1 in Z1_subsets:
        predictions = predict(z1, Z3, k)
        predictions_by_classifier.append(predictions)

    ensemble_predictions = []

    for sample_predictions in zip(*predictions_by_classifier):
        votes = {}

        for prediction in sample_predictions:
            votes[prediction] = votes.get(prediction, 0) + 1
        max_votes = max(votes.values())

        winners = [
            target_class
            for target_class, count in votes.items()
            if count == max_votes
        ]

        # Solves ties
        ensemble_predictions.append(random.choice(winners))

    return ensemble_predictions