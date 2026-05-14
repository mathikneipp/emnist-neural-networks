import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    """
    Base interface for loss functions used by the custom neural network.
    """

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

    def __init__(self, smoothing: float = 0.0):
        self.smoothing = smoothing

    def fn(self, y: np.ndarray, y_pred: np.ndarray) -> float:
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
        num_classes = y_pred.shape[1]

        # one-hot
        y_one_hot = np.zeros((batch_size, num_classes))
        y_one_hot[np.arange(batch_size), y] = 1

        # label smoothing
        y_smooth = y_one_hot * (1 - self.smoothing) + self.smoothing / num_classes

        # cross entropy
        loss = -np.sum(y_smooth * np.log(y_pred), axis=1)

        return np.mean(loss)

    def grad(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
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
