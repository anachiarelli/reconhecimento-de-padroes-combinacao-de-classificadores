import random

def select_z3(loader, r3=0.5, seed=42):
    dataset = loader()
    X, y = dataset.data, dataset.target

    # Divide the dataset into classes using original indexes to ensure that we can remove the selected samples from Z to create Z1 later.
    indices_by_target = {}

    for index, target in enumerate(y):
        if target not in indices_by_target:
            indices_by_target[target] = []

        indices_by_target[target].append(index)

    z3 = []
    rng = random.Random(seed)

    # Split each class independently according to the specified ratio.
    for target, indices in indices_by_target.items():

        # Shuffling samples
        indices = list(indices)
        rng.shuffle(indices)

        # Number of samples for Z3
        n = len(indices)
        n3 = int(n * r3)

        z3.extend([
            (X[index], target, index)
            for index in indices[:n3]
        ])

    return z3

def select_z1(loader, z3_indices, seed, r1=0.8):
    dataset = loader()
    X, y = dataset.data, dataset.target

    # Divide the dataset into classes
    indices_by_target = {}

    for index, target in enumerate(y):
        if target not in indices_by_target:
            indices_by_target[target] = []

        indices_by_target[target].append(index)

    # Remove samples in z3 from Z
    for target in indices_by_target:
        indices_by_target[target] = [
            index for index in indices_by_target[target]
            if index not in z3_indices
        ]

    # Randomizes the remaining samples
    rng = random.Random(seed)
    for target, indices in indices_by_target.items():
        rng.shuffle(indices)

    # Selects a subset of the remaining samples for Z1 with r1 ratio items
    z1 = []
    for target, data_points in indices_by_target.items():
        n = len(data_points)
        n1 = int(n * r1)

        for index in data_points[:n1]:
            data = X[index]
            z1.append((data, target))

    return z1