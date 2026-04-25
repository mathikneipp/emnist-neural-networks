import numpy as np
import math
from abc import ABC, abstractmethod

from .activations import Activation

class Layer (ABC):
    def __init__(self, input_dim: int, output_dim: int, activation: Activation, l2_regularization: float):
        # Dims
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Activation
        self.activation = activation
        
        # Regularization
        self.l2_regularization = l2_regularization
        
    @abstractmethod
    def forward(self, z_in: np.ndarray, training: bool) -> np.ndarray:
        pass
    
    def backward(self, W_lnext: np.ndarray, delta_lnext: np.ndarray) -> np.ndarray:
        pass
        

class DenseLayer (Layer):
    def __init__(self, input_dim: int, output_dim: int, activation: Activation, l2_regularization: float):
        super().__init__(input_dim, output_dim, activation, l2_regularization)
        
        self.W = np.array([np.random.normal(0, math.sqrt(2/input_dim), input_dim) for _ in output_dim])
        self.bias = np.random.uniform(1e-3, 1e-1, output_dim)
        
        self.z_in: None | np.ndarray = None   # z^{(l-1)}
        self.W_d: None | np.ndarray = None # ∂Li / ∂W(l)
        self.b_d: None | np.ndarray = None # ∂Li / ∂b(l)
        self.a: None | np.ndarray = None      # a^{(l)}
        
    def forward(self, z_in: np.ndarray, training: bool) -> np.ndarray:
        self.z_in = z_in
        if training:
            self.a = self.W @ z_in + self.bias
            z_out = self.activation.fn(self.a)
        else:
            a = self.W @ z_in + self.bias
            z_out = self.activation.fn(a)
            
        return z_out
    
    def backward(self, W_lnext: np.ndarray, delta_lnext: np.ndarray) -> np.ndarray:
        delta_l = self.activation.grad(self.a) * W_lnext.T @ delta_lnext
        self.W_d = delta_l @ self.z_in.T
        self.b_d = delta_l
        return self.W_d, delta_l
        