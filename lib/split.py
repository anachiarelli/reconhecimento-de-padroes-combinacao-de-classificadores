import random
from sklearn.preprocessing import StandardScaler


def simple_z1_z3_split(loader, r1, r3, seed=42):

    dataset = loader()
    X, y = dataset.data, dataset.target

    # Divide the dataset into classes
    data_by_target = {}

    for data, target in zip(X, y):
        if target not in data_by_target:
            data_by_target[target] = []

        data_by_target[target].append(data)

    Z1, Z3 = [], []
    rng = random.Random(seed)

    # Split each class independently according to the specified ratios.
    for target, data_points in data_by_target.items():

        # Shuffling samples
        data_points = list(data_points)
        rng.shuffle(data_points)

        # number of samples for each subset
        n = len(data_points)
        n1 = int(n * r1)
        n3 = int(n * r3)

        Z1.extend([(d, target) for d in data_points[:n1]])
        Z3.extend([(d, target) for d in data_points[n1+n3:]])

    # Normalize using Z1
    scaler = StandardScaler()

    Z1_X = [data for data, _ in Z1]
    Z3_X = [data for data, _ in Z3]

    Z1_X = scaler.fit_transform(Z1_X)
    Z3_X = scaler.transform(Z3_X)

    # Reconstruct the subsets, keeping the original target labels
    # associated with their corresponding normalized feature vectors.
    Z1 = [(d, target) for d, (_, target) in zip(Z1_X, Z1)]
    Z3 = [(d, target) for d, (_, target) in zip(Z3_X, Z3)]

    return Z1, Z3


'''
def split_z1_z3(loader, r1=0.5, seed=42):
    # We don't need Z2 for the current experiments, because we already have the best K values from the previous assignment. So we can just split into Z1 and Z3.

    dataset = loader()
    X, y = dataset.data, dataset.target

    # Divide the dataset into classes
    data_by_target = {}

    for data, target in zip(X, y):
        if target not in data_by_target:
            data_by_target[target] = []

        data_by_target[target].append(data)

    Z1, Z3 = [], []
    rng = random.Random(seed)

    # Split each class independently according to the specified ratio.
    for target, data_points in data_by_target.items():

        # Shuffling samples
        data_points = list(data_points)
        rng.shuffle(data_points)

        # Number of samples for each subset
        n = len(data_points)
        n1 = int(n * r1)

        Z1.extend([(d, target) for d in data_points[:n1]])
        Z3.extend([(d, target) for d in data_points[n1:]])

    # Normalize using only Z1
    scaler = StandardScaler()

    Z1_X = [data for data, _ in Z1]
    Z3_X = [data for data, _ in Z3]

    Z1_X = scaler.fit_transform(Z1_X)
    Z3_X = scaler.transform(Z3_X)

    # Reconstruct the subsets, keeping the original target labels
    Z1 = [(d, target) for d, (_, target) in zip(Z1_X, Z1)]
    Z3 = [(d, target) for d, (_, target) in zip(Z3_X, Z3)]

    return Z1, Z3
'''