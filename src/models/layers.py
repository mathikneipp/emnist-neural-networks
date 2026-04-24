import numpy as np
import math
from abc import ABC, abstractmethod

class Layer (ABC):
    def __init__(self, input_dim, output_dim, activation, l2_regularization):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.l2_regularization = l2_regularization
        
    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass
    
    def backward(self):
        pass
        

class DenseLayer (Layer):
    def __init__(self, input_dim, output_dim, activation, l2_regularization):
        super().__init__(input_dim, output_dim, activation, l2_regularization)
        
        self.W = np.array([np.random.normal(0, math.sqrt(2/input_dim), input_dim) for _ in output_dim])
        self.bias = np.random.uniform(1e-3, 1e-1, output_dim)
        
    def forward(self, z: np.ndarray) -> np.ndarray:
        