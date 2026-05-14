import numpy as np
from collections.abc import Iterator


def scaler(X):
    """
    Scale pixel values to the `[0, 1]` range.

    Args:
        X: Input array.

    Returns:
        np.ndarray: Scaled input array.
    """
    return X / 255


def data_split(
    X: np.ndarray, y: np.ndarray, frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split a dataset into two partitions.

    Args:
        X (np.ndarray): Input samples.
        y (np.ndarray): Target labels.
        frac (float): Fraction of samples assigned to the first split.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: First split inputs and
            labels followed by second split inputs and labels.
    """

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
    """
    Yield mini-batches from the provided dataset.

    Args:
        X (np.ndarray): Input samples.
        y (np.ndarray): Target labels.
        batch_size (int): Number of samples per batch.

    Returns:
        Iterator[tuple[np.ndarray, np.ndarray]]: Iterator over input-label mini-batches.
    """
    total_size = y.size

    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        yield X[start:end, :], y[start:end]
