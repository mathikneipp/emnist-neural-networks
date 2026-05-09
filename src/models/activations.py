import numpy as np
from scipy import special
from abc import ABC, abstractmethod


class Activation(ABC):

    @staticmethod
    @abstractmethod
    def fn(x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def grad(x: np.ndarray) -> np.ndarray:
        pass


class ReLU(Activation):

    @staticmethod
    def fn(X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)

    @staticmethod
    def grad(X: np.ndarray) -> np.ndarray:
        return (X > 0).astype(X.dtype)


class SoftMax(Activation):

    @staticmethod
    def fn(X: np.ndarray) -> np.ndarray:
        return special.softmax(X, axis=1)

    @staticmethod
    def grad(X: np.ndarray) -> np.ndarray:
        softmax_X = special.softmax(X, axis=1)
        return softmax_X * (1 - softmax_X)
