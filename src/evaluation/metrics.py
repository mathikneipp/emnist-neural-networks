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

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the ability of the classifier not to label a negative sample as positive.
    Formula: TP / (TP + FP).
    """
    tp, _, fp, _ = _get_values(y_true, y_pred)
    if (tp + fp) == 0: return 1.
    return tp / (tp + fp)

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the ability of the classifier to find all the positive samples.
    Formula: TP / (TP + FN).
    """
    tp, _, _, fn = _get_values(y_true, y_pred)
    return tp / (tp + fn)

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
    return 2*(prec*rec) / (prec + rec)

def fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the False Positive Rate (Fall-out).
    Formula: FP / (FP + TN).
    """
    _, tn, fp, _ = _get_values(y_true, y_pred)
    return fp / (fp + tn)

def tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the True Positive Rate (Sensitivity/Recall).
    Formula: TP / (TP + FN).
    """
    tp, _, _, fn = _get_values(y_true, y_pred)
    return tp / (tp + fn)

def pr(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[list, list, float]:
    """
    Computes Precision-Recall pairs for different probability thresholds 
    and the Area Under the Curve (AUC-PR).

    Args:
        y_true (np.ndarray): True binary labels.
        y_prob (np.ndarray): Target scores (probabilities) of the positive class.

    Returns:
        tuple[list, list, float]: (Recall values, Precision values, AUC-PR).
    """
    y_true = y_true.flatten()
    y_prob = y_prob.flatten()
    
    sorted_index = np.argsort(y_prob)[::-1]
    y_true = y_true[sorted_index]
    y_prob = y_prob[sorted_index]
    
    recall_values = [0]
    precision_values = [1]
    
    distinct_value_indices = np.where(np.diff(y_prob) != 0)[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    
    precision_values.extend((tps / (tps + fps).tolist()))
    
    total_positives = tps[-1]
    recall_values.extend((tps / total_positives).tolist())
    
    auc_pr = np.trapezoid(precision_values, recall_values)
    
    return recall_values, precision_values, auc_pr

def roc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[list, list, float]:
    """
    Computes Receiver Operating Characteristic (ROC) curve points and 
    the Area Under the Curve (AUC-ROC).

    Args:
        y_true (np.ndarray): True binary labels.
        y_prob (np.ndarray): Target scores (probabilities) of the positive class.

    Returns:
        tuple[list, list, float]: (FPR values, TPR values, AUC-ROC).
    """
    y_true = y_true.flatten()
    y_prob = y_prob.flatten()
    
    sorted_index = np.argsort(y_prob)[::-1]
    y_true = y_true[sorted_index]
    y_prob = y_prob[sorted_index]
    
    fpr_values = [0]
    tpr_values = [0]
    
    distinct_value_indices = np.where(np.diff(y_prob) != 0)[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    total_positivos = np.sum(y_true)
    total_negativos = len(y_true) - total_positivos

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    fns = total_positivos - tps
    tns = total_negativos - fps
    
    fpr_values.extend((fps / (fps + tns)).tolist())
    tpr_values.extend((tps / (tps + fns)).tolist())
    
    auc_roc = np.trapezoid(tpr_values, fpr_values)
    
    return fpr_values, tpr_values, auc_roc
