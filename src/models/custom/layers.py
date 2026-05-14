import numpy as np
import math

from .activations import Activation


class DenseLayer:
    """
    Fully connected layer with configurable activation and L2 regularization.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Activation,
        l2_regularization: float,
    ):
        """
        Initialize a fully connected layer.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of neurons in the layer.
            activation (Activation): Activation function applied after the linear transformation.
            l2_regularization (float): L2 regularization coefficient for the layer weights.
        """
        # Dims
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Activation
        self.activation = activation

        # Regularization
        self.l2_regularization = l2_regularization

        self.W = np.array(
            [
                np.random.normal(0, math.sqrt(2 / input_dim), input_dim)
                for _ in range(output_dim)
            ]
        )
        
        self.bias = np.zeros(output_dim)

        self.z_in: None | np.ndarray = None  # z^{(l-1)}
        self.W_d: None | np.ndarray = None  # dLi / dW(l)
        self.b_d: None | np.ndarray = None  # dLi / db(l)
        self.a: None | np.ndarray = None  # a^{(l)}

    def forward(self, z_in: np.ndarray, training: bool) -> np.ndarray:
        """
        Compute the forward pass of the layer.

        Args:
            z_in (np.ndarray): Input activations from the previous layer.
            training (bool): Whether the pass is performed during training.

        Returns:
            np.ndarray: Output activations of the current layer.
        """
        self.z_in = z_in
        if training:
            self.a = z_in @ self.W.T + self.bias
            z_out = self.activation.fn(self.a)
        else:
            a = z_in @ self.W.T + self.bias
            z_out = self.activation.fn(a)

        return z_out

    def backward(self, W_lnext: np.ndarray, delta_lnext: np.ndarray) -> np.ndarray:
        """
        Compute the local gradients for backpropagation.

        Args:
            W_lnext (np.ndarray): Weight matrix of the next layer.
            delta_lnext (np.ndarray): Backpropagated error term from the next layer.

        Returns:
            np.ndarray: Tuple containing the weight gradients and the local error term.
        """
        delta_l = (delta_lnext @ W_lnext) * self.activation.grad(self.a)
        self.W_d = delta_l.T @ self.z_in

        if self.l2_regularization > 0:
            self.W_d += self.l2_regularization * self.W

        self.b_d = np.sum(delta_l, axis=0)
        return self.W_d, delta_l
