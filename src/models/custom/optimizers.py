from abc import ABC, abstractmethod
import numpy as np

from .layers import DenseLayer


class Optimizer(ABC):
    """
    Base interface for optimizers used by the custom neural network.
    """
    @abstractmethod
    def step(self, layers: list, t: int) -> None:
        """
        Update model parameters using the stored layer gradients.

        Args:
            layers (list): Layers whose parameters will be updated.
            t (int): Optimization step index.
        """
        pass

    @abstractmethod
    def scheduling_step(self, epoch: int) -> None:
        """
        Update the learning rate according to the configured schedule.

        Args:
            epoch (int): Current training epoch.
        """
        pass


class GradientDescent(Optimizer):
    """
    Gradient descent optimizer with optional learning rate scheduling.
    """
    def __init__(self, learning_rate: float = 0.01, scheduling: dict[str] = None):
        """
        Initialize a gradient descent optimizer.

        Args:
            learning_rate (float, optional): Initial learning rate. Defaults to 0.01.
            scheduling (dict[str], optional): Learning rate scheduling configuration.
                Defaults to None.
        """
        self.lr_0 = learning_rate
        self.learning_rate = learning_rate
        self.scheduling = scheduling
        self.t = 0

    def step(self, layers: list[DenseLayer]) -> None:
        """
        Apply a gradient descent update to each layer parameter.

        Args:
            layers (list[DenseLayer]): Layers to update.
        """
        for layer in layers:
            if layer.W_d is not None:
                layer.W -= self.learning_rate * layer.W_d

            if layer.b_d is not None:
                layer.bias -= self.learning_rate * layer.b_d

    def scheduling_step(self, epoch: int) -> None:
        """
        Apply one learning rate scheduling step if configured.

        Args:
            epoch (int): Current training epoch.
        """
        if self.scheduling is not None:
            if self.scheduling.get("type") == "linear":
                self._linear_scheduling_step(epoch)
            elif self.scheduling.get("type") == "exponential":
                self._exponential_scheduling_step(epoch)

    def _linear_scheduling_step(self, epoch: int) -> None:
        """
        Update the learning rate using a linear decay schedule.

        Args:
            epoch (int): Current training epoch.
        """
        lr_min = self.scheduling.get("lr_min")
        k = self.scheduling.get("k")

        self.learning_rate = max(lr_min, self.lr_0 - k * epoch)

    def _exponential_scheduling_step(self, epoch: int) -> None:
        """
        Update the learning rate using an exponential decay schedule.

        Args:
            epoch (int): Current training epoch.
        """
        gamma = self.scheduling.get("gamma")

        self.learning_rate = self.lr_0 * (gamma**epoch)


class ADAM(Optimizer):
    """
    Adam optimizer with optional learning rate scheduling.
    """
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        scheduling: dict[str] = None,
    ):
        """
        Initialize an Adam optimizer.

        Args:
            learning_rate (float, optional): Initial learning rate. Defaults to 0.001.
            beta1 (float, optional): Exponential decay rate for the first moment. Defaults to 0.9.
            beta2 (float, optional): Exponential decay rate for the second moment. Defaults to 0.999.
            epsilon (float, optional): Numerical stability constant. Defaults to 1e-8.
            scheduling (dict[str], optional): Learning rate scheduling configuration.
                Defaults to None.
        """
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
        """
        Apply an Adam update to each layer parameter.

        Args:
            layers (list[DenseLayer]): Layers to update.
        """
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
        """
        Apply one learning rate scheduling step if configured.

        Args:
            epoch (int): Current training epoch.
        """
        if self.scheduling is not None:
            if self.scheduling.get("type") == "linear":
                self._linear_scheduling_step(epoch)
            elif self.scheduling.get("type") == "exponential":
                self._exponential_scheduling_step(epoch)

    def _linear_scheduling_step(self, epoch: int) -> None:
        """
        Update the learning rate using a linear decay schedule.

        Args:
            epoch (int): Current training epoch.
        """
        lr_min = self.scheduling.get("lr_min")
        k = self.scheduling.get("k")

        self.learning_rate = max(lr_min, self.lr_0 - k * epoch)

    def _exponential_scheduling_step(self, epoch: int) -> None:
        """
        Update the learning rate using an exponential decay schedule.

        Args:
            epoch (int): Current training epoch.
        """
        gamma = self.scheduling.get("gamma")

        self.learning_rate = self.lr_0 * (gamma**epoch)
