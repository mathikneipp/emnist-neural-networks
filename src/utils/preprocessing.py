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


def stratified_split(
    X: np.ndarray, y: np.ndarray, frac: float = 0.8, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits a dataset into training and validation sets while preserving
    class distribution.

    Works for binary and multiclass classification.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        frac (float): Fraction of samples per class assigned to training.
        seed (int): Random seed.

    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)

        split = int(len(cls_indices) * frac)

        train_indices.append(cls_indices[:split])
        val_indices.append(cls_indices[split:])

    train_indices = np.concatenate(train_indices)
    val_indices = np.concatenate(val_indices)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return (
        X[train_indices],
        X[val_indices],
        y[train_indices],
        y[val_indices],
    )


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
