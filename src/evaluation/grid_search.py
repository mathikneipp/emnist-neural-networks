import random
from tqdm import tqdm
from collections import defaultdict
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..models.custom.layers import DenseLayer
from ..models.custom.activations import ReLU, SoftMax
from ..models.custom.neural_network import SecuentialNeuralNetwork
from ..models.custom.loss import CrossEntropy
from ..models.custom.optimizers import ADAM

from ..models.torch.mlp import MLP

from ..training.train import train_and_eval


def random_grid_search_custom(
    input_dim,
    output_dim,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    K_models,
    possible_configs,
    early_stopping,
):
    """
    Train multiple custom neural networks using random hyperparameter sampling.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output classes.
        X_train: Training input samples.
        y_train: Training target labels.
        X_val: Validation input samples.
        y_val: Validation target labels.
        epochs: Maximum number of training epochs per model.
        K_models: Number of random configurations to evaluate.
        possible_configs: Search space for each hyperparameter.
        early_stopping: Early stopping patience used during training.

    Returns:
        tuple: Trained models and their sampled configurations.
    """
    models = []
    model_config = defaultdict(dict)

    rgen = random.Random(42)

    for i in tqdm(range(K_models)):
        print("\nModel:", i)

        for _ in range(10):
            new_config = defaultdict()
            for k, v in possible_configs.items():
                new_config[k] = rgen.choice(v)

            if new_config not in list(model_config.values()):
                model_config[i] = new_config
                break

        # No new model found
        if len(model_config) < i + 1:
            break

        print("Config:", model_config[i], end=2 * "\n")

        # Layers
        layers = []

        last_dim = input_dim
        for next_dim in model_config[i]["layers"]:
            layers.append(
                DenseLayer(
                    input_dim=last_dim,
                    output_dim=next_dim,
                    activation=ReLU(),
                    l2_regularization=model_config[i]["l2"],
                )
            )
            last_dim = next_dim

        layers.append(
            DenseLayer(
                input_dim=last_dim,
                output_dim=output_dim,
                activation=SoftMax(),
                l2_regularization=model_config[i]["l2"],
            )
        )

        # Neural Network
        models.append(
            SecuentialNeuralNetwork(
                layers,
                ADAM(
                    learning_rate=model_config[i]["lr"],
                    scheduling=model_config[i]["scheduling"],
                ),
                CrossEntropy(model_config[i]["label_smoothing"]),
            )
        )

        # Model Fitting
        models[i].fit(
            X_train,
            y_train,
            epochs,
            batch_size=model_config[i]["batch_size"],
            X_val=X_val,
            y_val=y_val,
            early_stopping=early_stopping,
        )

        print()

    return models, model_config


def random_grid_search_torch(
    input_dim,
    output_dim,
    train_dataset,
    val_dataset,
    epochs,
    K_models,
    possible_configs,
    early_stopping,
    device,
):
    """
    Train multiple PyTorch MLP models using random hyperparameter sampling.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output classes.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        epochs: Maximum number of training epochs per model.
        K_models: Number of random configurations to evaluate.
        possible_configs: Search space for each hyperparameter.
        early_stopping: Early stopping patience used during training.
        device: Device used to train the models.

    Returns:
        tuple: Trained models and their sampled configurations.
    """
    models = []
    model_config = defaultdict(dict)

    rgen = random.Random(42)

    for i in tqdm(range(K_models)):
        print("\nModel:", i)

        for _ in range(10):
            new_config = defaultdict()
            for k, v in possible_configs.items():
                new_config[k] = rgen.choice(v)

            if new_config not in list(model_config.values()):
                model_config[i] = new_config
                break

        # No new model found
        if len(model_config) < i + 1:
            break

        print("Config:", model_config[i], end=2 * "\n")

        model = MLP(
            input_dim,
            model_config[i]["layers"],
            output_dim,
            dropout=model_config[i]["dropout"],
            activation=model_config[i]["activation"],
            batch_norm=model_config[i]["batch_norm"],
        )

        optimizer = model_config[i]["optimizer"](
            model.parameters(),
            lr=model_config[i]["lr"],
            weight_decay=model_config[i]["l2"],
        )

        loss_fn = nn.CrossEntropyLoss(
            label_smoothing=model_config[i]["label_smoothing"]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=model_config[i]["batch_size"],
            shuffle=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset, batch_size=model_config[i]["batch_size"], shuffle=False
        )

        model.to(device)

        _, _, _ = train_and_eval(
            train_loader,
            val_loader,
            model,
            loss_fn,
            optimizer,
            device,
            epochs=epochs,
            early_stopping=early_stopping,
            scheduling=model_config[i]["scheduling"],
        )

        models.append(model)

    return models, model_config
