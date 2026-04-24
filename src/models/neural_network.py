import numpy as np
from typing import Callable
from tqdm import tqdm
from copy import deepcopy

from .layers import Layer
from .optimizers import Optimizer

from ..utils.preprocessing import get_batches


class NeuralNetwork:
    def __init__(
        self, layers: list[Layer], optimizer: Optimizer, loss_function: Callable
    ):
        self.layers = layers
        self.loss_function = loss_function

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
        X, y = X.copy(), y.copy()

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
            X, y = X[indices], y[indices]

            batches = get_batches(X, y, batch_size)

            for batch_X, batch_y in batches:
                y_pred = self._forward_pass(batch_X)
                fit_loss = self.loss_function(batch_y, y_pred)
                self._back_propagation(batch_y, y_pred)
                
            # Early stopping
            if (X_val is not None) and (y_val is not None):
                val_loss = self.loss_function(y_val, self.predict(X_val))
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
        

    def _back_propagation(self):
        pass
