from abc import ABC, abstractmethod
import numpy as np

from .layers import DenseLayer


class Optimizer(ABC):
    @abstractmethod
    def step(self, layers: list, t: int) -> None:
        pass

    @abstractmethod
    def scheduling_step(self, epoch: int) -> None:
        pass


class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.01, scheduling: dict[str] = None):
        self.lr_0 = learning_rate
        self.learning_rate = learning_rate
        self.scheduling = scheduling
        self.t = 0

    def step(self, layers: list[DenseLayer]) -> None:
        for layer in layers:
            if layer.W_d is not None:
                layer.W -= self.learning_rate * layer.W_d

            if layer.b_d is not None:
                layer.bias -= self.learning_rate * layer.b_d

    def scheduling_step(self, epoch: int) -> None:
        if self.scheduling is not None:
            if self.scheduling.get("type") == "linear":
                self._linear_scheduling_step(epoch)
            elif self.scheduling.get("type") == "exponential":
                self._exponential_scheduling_step(epoch)

    def _linear_scheduling_step(self, epoch: int) -> None:
        lr_min = self.scheduling.get("lr_min")
        k = self.scheduling.get("k")

        self.learning_rate = max(lr_min, self.lr_0 - k * epoch)

    def _exponential_scheduling_step(self, epoch: int) -> None:
        gamma = self.scheduling.get("gamma")

        self.learning_rate = self.lr_0 * (gamma**epoch)


class ADAM(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        scheduling: dict[str] = None,
    ):
        self.lr_0 = learning_rate
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scheduling = scheduling
        self.t = 0

        self.W_mt = 0.0
        self.W_st = 0.0

        self.b_mt = 0.0
        self.b_st = 0.0

    def step(self, layers: list[DenseLayer]) -> None:
        self.t += 1

        for layer in layers:

            if not hasattr(layer, "W_mt"):
                layer.W_mt = np.zeros_like(layer.W)
                layer.W_st = np.zeros_like(layer.W)
                layer.b_mt = np.zeros_like(layer.bias)
                layer.b_st = np.zeros_like(layer.bias)

            if layer.W_d is not None:
                layer.W_mt = self.beta1 * layer.W_mt + (1 - self.beta1) * layer.W_d
                layer.W_st = self.beta2 * layer.W_st + (1 - self.beta2) * (layer.W_d**2)

                W_mt_hat = layer.W_mt / (1 - self.beta1**self.t)
                W_st_hat = layer.W_st / (1 - self.beta2**self.t)

                layer.W -= (
                    self.learning_rate * W_mt_hat / (np.sqrt(W_st_hat) + self.epsilon)
                )

            if layer.b_d is not None:
                layer.b_mt = self.beta1 * layer.b_mt + (1 - self.beta1) * layer.b_d
                layer.b_st = self.beta2 * layer.b_st + (1 - self.beta2) * (layer.b_d**2)

                b_mt_hat = layer.b_mt / (1 - self.beta1**self.t)
                b_st_hat = layer.b_st / (1 - self.beta2**self.t)

                layer.bias -= (
                    self.learning_rate * b_mt_hat / (np.sqrt(b_st_hat) + self.epsilon)
                )

    def scheduling_step(self, epoch: int) -> None:
        if self.scheduling is not None:
            if self.scheduling.get("type") == "linear":
                self._linear_scheduling_step(epoch)
            elif self.scheduling.get("type") == "exponential":
                self._exponential_scheduling_step(epoch)

    def _linear_scheduling_step(self, epoch: int) -> None:
        lr_min = self.scheduling.get("lr_min")
        k = self.scheduling.get("k")

        self.learning_rate = max(lr_min, self.lr_0 - k * epoch)

    def _exponential_scheduling_step(self, epoch: int) -> None:
        gamma = self.scheduling.get("gamma")

        self.learning_rate = self.lr_0 * (gamma**epoch)
