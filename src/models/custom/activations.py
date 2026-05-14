import numpy as np
from scipy import special
from abc import ABC, abstractmethod


class Activation(ABC):
    """
    Base interface for activation functions used in custom layers.
    """

    @staticmethod
    @abstractmethod
    def fn(x: np.ndarray) -> np.ndarray:
        """
        Apply the activation function to the input values.

        Args:
            x (np.ndarray): Input values.

        Returns:
            np.ndarray: Activated output values.
        """
        pass

    @staticmethod
    @abstractmethod
    def grad(x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the activation function.

        Args:
            x (np.ndarray): Input values where the derivative is evaluated.

        Returns:
            np.ndarray: Activation derivative evaluated at the input values.
        """
        pass


class ReLU(Activation):
    """
    Rectified Linear Unit activation function.
    """

    @staticmethod
    def fn(X: np.ndarray) -> np.ndarray:
        """
        Apply the ReLU activation function.

        Args:
            X (np.ndarray): Input values.

        Returns:
            np.ndarray: Element-wise ReLU activations.
        """
        return np.maximum(0, X)

    @staticmethod
    def grad(X: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the ReLU activation function.

        Args:
            X (np.ndarray): Input values.

        Returns:
            np.ndarray: Element-wise ReLU derivatives.
        """
        return (X > 0).astype(X.dtype)


class SoftMax(Activation):
    """
    Softmax activation function for multiclass outputs.
    """

    @staticmethod
    def fn(X: np.ndarray) -> np.ndarray:
        """
        Apply the softmax activation function across classes.

        Args:
            X (np.ndarray): Input logits.

        Returns:
            np.ndarray: Class probabilities for each sample.
        """
        return special.softmax(X, axis=1)

    @staticmethod
    def grad(X: np.ndarray) -> np.ndarray:
        """
        Compute an element-wise softmax derivative approximation.

        Args:
            X (np.ndarray): Input logits.

        Returns:
            np.ndarray: Element-wise derivative values of the softmax output.
        """
        softmax_X = special.softmax(X, axis=1)
        return softmax_X * (1 - softmax_X)
