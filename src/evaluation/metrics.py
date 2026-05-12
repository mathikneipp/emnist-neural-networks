import numpy as np

def _get_values(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    """
    Calculates the fundamental components of a binary confusion matrix.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.

    Returns:
        tuple[float, float, float, float]: (True Positives, True Negatives, 
            False Positives, False Negatives).
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes a 2x2 confusion matrix.

    The matrix is structured as:
    [[TP, FN],
     [FP, TN]]

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.

    Returns:
        np.ndarray: 2x2 confusion matrix.
    """
    tp, tn, fp, fn = _get_values(y_true, y_pred)
    return np.array([[tp, fn], [fp, tn]])

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the ratio of correctly predicted samples to the total samples.
    Formula: (TP + TN) / Total.
    """
    tp, tn, fp, fn = _get_values(y_true, y_pred)
    N = tp + tn + fp + fn
    return (tp + tn) / N

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the harmonic mean of precision and recall.
    Formula: 2 * (Precision * Recall) / (Precision + Recall).
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    prec = tp / (tp + fp) if (tp + fp) != 0 else 1.
    rec = tp / (tp + fn)
    return 2*(prec*rec) / (prec + rec + 1e-15)
