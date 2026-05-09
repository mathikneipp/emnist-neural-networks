import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):

    @staticmethod
    @abstractmethod
    def fn(y: np.ndarray, y_pred: np.ndarray):
        pass

    @staticmethod
    @abstractmethod
    def grad(y: np.ndarray, y_pred: np.ndarray):
        pass


class CrossEntropy(Loss):

    @staticmethod
    def fn(y: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        batch_size = y.shape[0]
        correct_probs = y_pred[np.arange(batch_size), y]

        return -np.mean(np.log(correct_probs))

    @staticmethod
    def grad(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        batch_size = y.shape[0]
        grad = y_pred.copy()
        grad[np.arange(batch_size), y] -= 1
        grad /= batch_size

        return grad
