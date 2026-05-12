import numpy as np
from collections.abc import Iterator


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


def get_batches(
    X: np.ndarray, y: np.ndarray, batch_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    total_size = y.size

    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        yield X[start:end, :], y[start:end]
