import numpy as np
from tqdm import tqdm
from copy import deepcopy

from .layers import DenseLayer
from .optimizers import Optimizer
from .loss import Loss

from ...utils.preprocessing import get_batches


class SecuentialNeuralNetwork:
    """
    Sequential neural network built from custom dense layers.
    """

    def __init__(
        self, layers: list[DenseLayer], optimizer: Optimizer, loss_function: Loss
    ):
        """
        Initialize the neural network with its layers, optimizer, and loss function.

        Args:
            layers (list[DenseLayer]): Ordered list of network layers.
            optimizer (Optimizer): Optimizer used to update the model parameters.
            loss_function (Loss): Loss function used during training.
        """
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.train_loss = []
        self.val_loss = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int | None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping: int | None = 5,
        epsilon: float = 1e-3,
    ) -> int:
        """
        Train the neural network using mini-batch gradient-based optimization.

        Args:
            X (np.ndarray): Training input samples.
            y (np.ndarray): Training target labels.
            epochs (int): Maximum number of training epochs.
            batch_size (int | None): Number of samples per batch. If `None`, uses full-batch training.
            X_val (np.ndarray | None, optional): Validation input samples. Defaults to None.
            y_val (np.ndarray | None, optional): Validation target labels. Defaults to None.
            early_stopping (int | None, optional): Number of epochs to wait for validation
                improvement before stopping. Defaults to 5.
            epsilon (float): Minimum improvement in validation loss required to reset early stopping
                patience. Defaults to 1e-3.

        Returns:
            int: Number of epochs effectively completed or the best epoch when early stopping is used.
        """
        indices = np.arange(y.size)

        # Early stopping settings
        wait = 0
        best_val_loss = np.inf
        best_layers = None
        best_epoch = None

        # tqdm bar
        bar = tqdm(range(epochs), desc="Training", unit="ep")

        for epoch in bar:
            # Data shuffle
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            if batch_size is not None:
                batches = get_batches(X, y, batch_size)
            else:
                batches = [(X, y)]

            epoch_loss, seen = 0.0, 0

            # Batches iteration
            for batch_X, batch_y in batches:
                y_pred = self._forward_pass(batch_X)

                batch_loss = self.loss_function.fn(batch_y, y_pred)
                batch_size_temp = batch_y.shape[0]

                epoch_loss += batch_loss * batch_size_temp
                seen += batch_size_temp

                self._backward_pass(batch_y, y_pred)
                self.optimizer.step(self.layers)

            train_loss = epoch_loss / seen
            self.train_loss.append(train_loss)

            # Val. eval
            if (X_val is not None) and (y_val is not None):
                val_loss = self.loss_function.fn(y_val, self.predict(X_val))
                self.val_loss.append(val_loss)

                # Early stopping
                if early_stopping is not None:
                    if val_loss + epsilon < best_val_loss:
                        best_val_loss = val_loss
                        wait = 0
                        best_layers = deepcopy(self.layers)
                        best_epoch = epoch

                    else:
                        wait += 1

                        if wait >= early_stopping:
                            print(f"Early stopping after epoch: {epoch}")
                            break

            self.optimizer.scheduling_step(epoch)

        if best_layers is not None:
            self.layers = best_layers

        if best_epoch is None:
            return epoch + 1

        return best_epoch + 1

    def predict(self, X: np.ndarray):
        """
        Generate predictions for the provided input samples.

        Args:
            X (np.ndarray): Input samples to evaluate.

        Returns:
            np.ndarray: Network output predictions.
        """
        return self._forward_pass(X, training=False)

    @classmethod
    def build_from_config(
        cls,
        input_dim,
        output_dim,
        activation,
        output_activation,
        config: dict,
        optimizer,
        loss,
    ):
        """
        Build a neural network instance from a configuration dictionary.

        Args:
            input_dim (_type_): Number of input features.
            output_dim (_type_): Number of output units.
            activation (_type_): Activation class or factory used in hidden layers.
            output_activation (_type_): Activation class or factory used in the output layer.
            config (dict): Model and optimization hyperparameter configuration.
            optimizer (_type_): Optimizer instance or optimizer class to instantiate.
            loss (_type_): Loss instance or loss class to instantiate.

        Returns:
            _type_: Configured neural network instance.
        """
        layers = []
        last_dim = input_dim

        # Hidden layers
        for next_dim in config["layers"]:
            layers.append(
                DenseLayer(
                    input_dim=last_dim,
                    output_dim=next_dim,
                    activation=activation(),
                    l2_regularization=config["l2"],
                )
            )
            last_dim = next_dim

        # Out layer
        layers.append(
            DenseLayer(
                input_dim=last_dim,
                output_dim=output_dim,
                activation=output_activation(),
                l2_regularization=config["l2"],
            )
        )

        optimizer_instance = optimizer(
            learning_rate=config["lr"], scheduling=config["scheduling"]
        )

        loss_instance = (
            loss(config["label_smoothing"]) if isinstance(loss, type) else loss
        )

        return cls(
            layers=layers,
            optimizer=optimizer_instance,
            loss_function=loss_instance,
        )

    def _forward_pass(self, X, training: bool = True) -> np.ndarray:
        """
        Run a forward pass through all network layers.

        Args:
            X (_type_): Input batch.
            training (bool, optional): Whether to store intermediate activations for backpropagation.
                Defaults to True.

        Returns:
            np.ndarray: Network output for the input batch.
        """
        z_l = X
        for layer in self.layers:
            z_l = layer.forward(z_l, training)
        return z_l

    def _backward_pass(self, y: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Run backpropagation and store gradients for every layer.

        Args:
            y (np.ndarray): Ground-truth labels for the current batch.
            y_pred (np.ndarray): Predicted outputs for the current batch.
        """
        last_layer = self.layers[-1]

        delta_l = self.loss_function.grad(y, y_pred)
        last_layer.W_d = delta_l.T @ last_layer.z_in
        last_layer.b_d = np.sum(delta_l, axis=0)

        if last_layer.l2_regularization > 0:
            last_layer.W_d += last_layer.l2_regularization * last_layer.W

        for l in range(len(self.layers) - 2, -1, -1):
            _, delta_l = self.layers[l].backward(self.layers[l + 1].W, delta_l)
