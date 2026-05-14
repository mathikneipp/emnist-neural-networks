import numpy as np

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the binary F1-score from true and predicted labels.

    Args:
        y_true (np.ndarray): Ground-truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.

    Returns:
        float: F1-score value.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    prec = tp / (tp + fp) if (tp + fp) != 0 else 1.
    rec = tp / (tp + fn)
    return 2*(prec*rec) / (prec + rec + 1e-15)

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
    
def compute_macro_f1_ova(y_true, y_pred):
    """
    Compute macro F1-score using a one-vs-all strategy.

    Args:
        y_true: Ground-truth multiclass labels.
        y_pred: Predicted multiclass labels.

    Returns:
        tuple: Mean macro F1-score and the evaluated classes.
    """
    classes = np.union1d(y_true, y_pred)
    f1_scores = []

    for cls in classes:
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        f1_scores.append(f1_score(y_true_binary, y_pred_binary))

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
