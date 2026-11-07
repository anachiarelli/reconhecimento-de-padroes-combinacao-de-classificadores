import random
from sklearn.preprocessing import StandardScaler


def split(loader, r1=0.25, r2=0.25, seed=42):

    dataset = loader()
    X, y = dataset.data, dataset.target

    # Divide the dataset into classes
    data_by_target = {}

    for data, target in zip(X, y):
        if target not in data_by_target:
            data_by_target[target] = []

        data_by_target[target].append(data)

    Z1, Z2, Z3 = [], [], []
    rng = random.Random(seed)

    # Split each class independently according to the specified ratios.
    for target, data_points in data_by_target.items():

        # Shuffling samples
        data_points = list(data_points)
        rng.shuffle(data_points)

        # number of samples for each subset
        n = len(data_points)
        n1 = int(n * r1)
        n2 = int(n * r2)

        Z1.extend([(d, target) for d in data_points[:n1]])
        Z2.extend([(d, target) for d in data_points[n1:n1+n2]])
        Z3.extend([(d, target) for d in data_points[n1+n2:]])

    # Normalize using only Z1
    scaler = StandardScaler()

    Z1_X = [data for data, _ in Z1]
    Z2_X = [data for data, _ in Z2]
    Z3_X = [data for data, _ in Z3]

    Z1_X = scaler.fit_transform(Z1_X)
    Z2_X = scaler.transform(Z2_X)
    Z3_X = scaler.transform(Z3_X)

    # Reconstruct the subsets, keeping the original target labels
    # associated with their corresponding normalized feature vectors.
    Z1 = [(d, target) for d, (_, target) in zip(Z1_X, Z1)]
    Z2 = [(d, target) for d, (_, target) in zip(Z2_X, Z2)]
    Z3 = [(d, target) for d, (_, target) in zip(Z3_X, Z3)]

    return Z1, Z2, Z3
