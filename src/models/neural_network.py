import numpy as np
from tqdm import tqdm
from copy import deepcopy

from .layers import DenseLayer
from .optimizers import Optimizer
from .loss import Loss

from ..utils.preprocessing import get_batches


class SecuentialNeuralNetwork:
    def __init__(
        self, layers: list[DenseLayer], optimizer: Optimizer, loss_function: Loss
    ):
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.fit_loss = []
        self.val_loss = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping: int = 5,
    ):
        indices = np.arange(y.size)

        # Early stopping settings
        wait = 0
        best_val_loss = np.inf
        best_layers = None

        # tqdm bar
        bar = tqdm(range(epochs), desc="Training", unit="ep")

        for epoch in bar:
            # Data shuffle
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            batches = get_batches(X, y, batch_size)

            for batch_X, batch_y in batches:
                y_pred = self._forward_pass(batch_X)

                self._backward_pass(batch_y, y_pred)

                self.optimizer.step(self.layers)
            
            
            fit_loss = self.loss_function.fn(y, self.predict(X))
            self.fit_loss.append(fit_loss)

            # Early stopping
            if (X_val is not None) and (y_val is not None):
                val_loss = self.loss_function.fn(y_val, self.predict(X_val))
                self.val_loss.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                    best_layers = deepcopy(self.layers)

                else:
                    wait += 1

                    if wait >= early_stopping:
                        print("Early stopping.")
                        break

        if best_layers is not None:
            self.layers = best_layers

    def predict(self, X: np.ndarray):
        return self._forward_pass(X, training=False)

    def _forward_pass(self, X, training: bool = True) -> np.ndarray:
        z_l = X
        for layer in self.layers:
            z_l = layer.forward(z_l, training)
        return z_l

    def _backward_pass(self, y: np.ndarray, y_pred: np.ndarray) -> None:
        gradients = []

        last_layer = self.layers[-1]

        y_d = self.loss_function.grad(y, y_pred)
        delta_l = last_layer.activation.grad(last_layer.a) * y_d

        W_d = delta_l.T @ last_layer.z_in

        gradients.append((W_d, delta_l))

        for l in range(len(self.layers) - 2, -1, -1):
            W_d, delta_l = self.layers[l].backward(self.layers[l + 1].W, delta_l)
            gradients.append((W_d, delta_l))

        gradients.reverse()
