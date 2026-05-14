import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    """
    Base interface for loss functions used by the custom neural network.
    """

    @staticmethod
    @abstractmethod
    def fn(y: np.ndarray, y_pred: np.ndarray):
        """
        Compute the loss value for a batch of predictions.

        Args:
            y (np.ndarray): Ground-truth labels.
            y_pred (np.ndarray): Predicted outputs.

        Returns:
            float: Loss value for the batch.
        """
        pass

    @staticmethod
    @abstractmethod
    def grad(y: np.ndarray, y_pred: np.ndarray):
        """
        Compute the gradient of the loss with respect to the predictions.

        Args:
            y (np.ndarray): Ground-truth labels.
            y_pred (np.ndarray): Predicted outputs.

        Returns:
            np.ndarray: Loss gradient with respect to the predictions.
        """
        pass


class CrossEntropy(Loss):
    """
    Cross-entropy loss for multiclass classification.
    """

    @staticmethod
    def fn(y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross-entropy loss for a batch of predictions.

        Args:
            y (np.ndarray): Ground-truth class labels.
            y_pred (np.ndarray): Predicted class probabilities.

        Returns:
            float: Mean cross-entropy loss over the batch.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        batch_size = y.shape[0]
        correct_probs = y_pred[np.arange(batch_size), y]

        return -np.mean(np.log(correct_probs))

    @staticmethod
    def grad(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the cross-entropy gradient with respect to the predictions.

        Args:
            y (np.ndarray): Ground-truth class labels.
            y_pred (np.ndarray): Predicted class probabilities.

        Returns:
            np.ndarray: Gradient of the loss with respect to `y_pred`.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        batch_size = y.shape[0]
        grad = y_pred.copy()
        grad[np.arange(batch_size), y] -= 1
        grad /= batch_size

        return grad
