import numpy as np
from typing import Callable
from tqdm import tqdm
from copy import deepcopy

from .layers import Layer
from .optimizers import Optimizer
from .loss import Loss

from ..utils.preprocessing import get_batches


class SecuentialNeuralNetwork:
    def __init__(
        self, layers: list[Layer], optimizer: Optimizer, loss_function: Loss
    ):
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping: int = 5,
    ):
        x, y = x.copy(), y.copy()

        indices = np.arange(y.size)

        # Early stopping settings
        wait = 0
        best_val_loss = np.inf
        best_layers = None

        # tqdm bar
        bar = tqdm(range(epochs), desc="Training", unit="ep")

        for epoch in bar:
            # Data shuffle
            indices = np.random.shuffle(indices)
            x, y = x[indices], y[indices]

            batches = get_batches(x, y, batch_size)

            for batch_x, batch_y in batches:
                y_pred = self._forward_pass(batch_x)
                fit_loss = self.loss_function.fn(batch_y, y_pred)
                self._backward_pass(batch_y, y_pred)
                
            # Early stopping
            if (x_val is not None) and (y_val is not None):
                val_loss = self.loss_function(y_val, self.predict(x_val))
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
            
                    
    def predict(self, x: np.ndarray):
        return self._forward_pass(x, training=False)

    def _forward_pass(self, x, training: bool = True) -> np.ndarray:
        z_l = x
        for layer in self.layers:
            z_l = layer.forward(z_l, training)
        return z_l

    def _backward_pass(self, y: np.ndarray, y_pred: np.ndarray) -> None:
        gradients = []
        
        y_d = self.loss_function.grad(y, y_pred)
        
        delta_l = self.layers[-1].activation.grad(self.layers[-1].a) * y_d
        W_d = delta_l @ self.layers[-1].z_in.T
        
        gradients.append((W_d, delta_l))
        
        for l in range(len(self.layers) - 1, -1, -1):
            W_d, delta_l = self.layers[l].backward(self.layers[l + 1].W, delta_l)
            gradients.append((W_d, delta_l))