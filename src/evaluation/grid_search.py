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
    models = []
    model_config = defaultdict(dict)

    rgen = random.Random(42)

    for i in tqdm(range(K_models)):
        print("\nModel:", i)

        for k, v in possible_configs.items():
            model_config[i][k] = rgen.choice(v)

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
                layers, ADAM(scheduling=model_config[i]["scheduling"]), CrossEntropy()
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
    models = []
    model_config = defaultdict(dict)

    rgen = random.Random(42)

    for i in tqdm(range(K_models)):
        print("\nModel:", i)

        for k, v in possible_configs.items():
            model_config[i][k] = rgen.choice(v)

        print("Config:", model_config[i], end=2 * "\n")

        model = MLP(
            input_dim,
            model_config[i]["layers"],
            output_dim,
            dropout=model_config[i]["dropout"],
            activation=model_config[i]["activation"],
        )

        optimizer = model_config[i]["optimizer"](
            model.parameters(), weight_decay=model_config[i]["l2"]
        )

        loss_fn = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            train_dataset, batch_size=model_config[i]["batch_size"], shuffle=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=model_config[i]["batch_size"], shuffle=False
        )

        model.to(device)

        _, _ = train_and_eval(
            train_loader,
            val_loader,
            model,
            loss_fn,
            optimizer,
            device,
            epochs=epochs,
            early_stopping=early_stopping,
            scheduling=model_config[i]["scheduling"],
            lr_0=0.001,
        )

        models.append(model)

        print()

    return models, model_config
