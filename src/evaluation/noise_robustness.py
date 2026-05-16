import numpy as np
import pandas as pd

from .predictions import get_predictions

from .metrics import compute_accuracy, compute_macro_f1_ova


def add_gaussian_noise(X, sigma, clip=True):
    """
    Add Gaussian noise to the input samples.

    Args:
        X: Input samples.
        sigma: Standard deviation of the Gaussian noise.
        clip: Whether to clip the noisy samples to the `[0.0, 1.0]` range.

    Returns:
        np.ndarray: Noisy input samples.
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
    X_noisy = X + noise

    if clip:
        X_noisy = np.clip(X_noisy, 0.0, 1.0)

    return X_noisy


def evaluate_noise_robustness(
    models: dict, X_test: np.ndarray, y_test: np.ndarray, noise_levels: list, device
):
    """
    Evaluate model performance under different Gaussian noise levels.

    Args:
        models (dict): Mapping from model names to trained model instances.
        X_test (np.ndarray): Test input samples.
        y_test (np.ndarray): Test target labels.
        noise_levels (list): Noise standard deviations to evaluate.
        device: Device used for PyTorch-based models.

    Returns:
        pd.DataFrame: Table with accuracy and macro F1-score for each model and noise level.
    """
    results = []
    metric_classes = np.arange(int(np.max(y_test)) + 1)

    for model_name, model in models.items():

        y_pred_clean = get_predictions(model, X_test, device)

        results.append(
            {
                "model": model_name,
                "noise_level": 0.0,
                "accuracy": compute_accuracy(y_test, y_pred_clean),
                "f1_score": compute_macro_f1_ova(
                    y_test, y_pred_clean, classes=metric_classes
                )[0],
            }
        )

        for sigma in noise_levels:
            X_noisy = add_gaussian_noise(X_test, sigma)

            y_pred = get_predictions(model, X_noisy, device)

            results.append(
                {
                    "model": model_name,
                    "noise_level": sigma,
                    "accuracy": compute_accuracy(y_test, y_pred),
                    "f1_score": compute_macro_f1_ova(
                        y_test, y_pred, classes=metric_classes
                    )[0],
                }
            )

    return pd.DataFrame(results)
