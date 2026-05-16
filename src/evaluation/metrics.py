import numpy as np


def f1_score(
    y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0
) -> float:
    """
    Compute the binary F1-score from true and predicted labels.

    Args:
        y_true (np.ndarray): Ground-truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.
        zero_division (float): Value returned when precision or recall are undefined.

    Returns:
        float: F1-score value.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp == 0 and fp == 0 and fn == 0:
        return float(zero_division)

    prec = tp / (tp + fp) if (tp + fp) != 0 else zero_division
    rec = tp / (tp + fn) if (tp + fn) != 0 else zero_division

    if prec + rec == 0:
        return float(zero_division)

    return float(2 * (prec * rec) / (prec + rec))


def compute_accuracy(y_true, y_pred):
    """
    Compute overall multiclass accuracy.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        float: Fraction of correctly classified samples.
    """
    return float(np.mean(y_true == y_pred))


def compute_macro_f1_ova(
    y_true, y_pred, classes: np.ndarray | list | None = None, zero_division: float = 0.0
):
    """
    Compute macro F1-score using a one-vs-all strategy.

    Args:
        y_true: Ground-truth multiclass labels.
        y_pred: Predicted multiclass labels.
        classes: Class labels to include in the macro average. If `None`, uses
            the sorted unique labels present in `y_true`.
        zero_division: Value returned for per-class F1 when the metric is undefined.

    Returns:
        tuple: Mean macro F1-score and the evaluated classes.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if classes is None:
        classes = np.unique(y_true)
    else:
        classes = np.asarray(classes)

    f1_scores = []

    for cls in classes:
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        f1_scores.append(
            f1_score(y_true_binary, y_pred_binary, zero_division=zero_division)
        )

    return float(np.mean(f1_scores)), classes


def compute_multiclass_confusion_matrix(y_true, y_pred):
    """
    Compute the multiclass confusion matrix.

    Args:
        y_true: Ground-truth multiclass labels.
        y_pred: Predicted multiclass labels.

    Returns:
        tuple: Confusion matrix and the class order used to build it.
    """
    classes = np.union1d(y_true, y_pred)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[class_to_idx[true_label], class_to_idx[pred_label]] += 1

    return cm, classes
