import numpy as np
from abc import ABC, abstractmethod


class Activation (ABC):

    @abstractmethod
    def fn(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray:
        pass

class ReLU (Activation):
    
    def fn(self, x: np.ndarray) -> np.ndarray:
        return np.max(np.zeros_like(x), x, axis=0)
    
    def grad(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)