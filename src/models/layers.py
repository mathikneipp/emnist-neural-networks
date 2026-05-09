import numpy as np
import math

from .activations import Activation


class DenseLayer:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Activation,
        l2_regularization: float,
    ):
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
        self.z_in = z_in
        if training:
            self.a = z_in @ self.W.T + self.bias
            z_out = self.activation.fn(self.a)
        else:
            a = z_in @ self.W.T + self.bias
            z_out = self.activation.fn(a)

        return z_out

    def backward(self, W_lnext: np.ndarray, delta_lnext: np.ndarray) -> np.ndarray:
        delta_l = (delta_lnext @ W_lnext) * self.activation.grad(self.a)
        self.W_d = delta_l.T @ self.z_in

        if self.l2_regularization > 0:
            self.W_d += self.l2_regularization * self.W

        self.b_d = np.sum(delta_l, axis=0)
        return self.W_d, delta_l
