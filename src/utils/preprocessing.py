import numpy as np
import math


def scaler(X):
    return X / 255


def data_split(
    X: np.ndarray, y: np.ndarray, frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    total_size = y.size

    indices = np.arange(total_size)
    np.random.shuffle(indices)

    split_idx = int(total_size * frac)

    index_1st = indices[:split_idx]
    index_2nd = indices[split_idx:]

    X_1, y_1 = X[index_1st], y[index_1st]
    X_2, y_2 = X[index_2nd], y[index_2nd]

    return X_1, y_1, X_2, y_2


def get_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> list[tuple]:
    
    total_size = y.size
    total_batches = math.ceil(total_size / batch_size)
    batches = [
        (
            X[i * batch_size : min((i + 1) * batch_size, total_size)],
            y[i * batch_size : min((i + 1) * batch_size, total_size)],
        )
        for i in range(total_batches)
    ]
    return batches
