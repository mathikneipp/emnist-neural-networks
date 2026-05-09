from abc import ABC, abstractmethod

from .layers import DenseLayer


class Optimizer(ABC):
    @abstractmethod
    def step(self, layers) -> None:
        pass


class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def step(self, layers: list[DenseLayer]) -> None:
        for layer in layers:
            if layer.W_d is not None:
                layer.W -= self.learning_rate * layer.W_d

            if layer.b_d is not None:
                layer.bias -= self.learning_rate * layer.b_d
