import torch
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from ..utils.utils import get_scheduler


def train_loop(dataloader, model, loss_fn, optimizer, device) -> tuple[float, list]:
    """
    Run one training epoch over the dataloader.

    Args:
        dataloader: Iterable of training batches.
        model: Model to optimize.
        loss_fn: Loss function used for optimization.
        optimizer: Optimizer used to update model parameters.
        device: Device where computations are performed.

    Returns:
        tuple[float, list]: Mean training loss over the epoch and per-batch loss values.
    """
    model.train()
    loss = 0.0
    total_samples = 0
    losses = []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred = model(X)  # Forward pass
        batch_loss = loss_fn(y_pred, y)  # Loss eval
        batch_loss_value = batch_loss.item()
        batch_size = y.shape[0]
        loss += batch_loss_value * batch_size
        total_samples += batch_size
        losses.append(batch_loss_value)
        batch_loss.backward()  # Backward pass
        optimizer.step()  # Step

    return loss / total_samples, losses


def test_loop(dataloader, model, loss_fn, device) -> tuple[float, list]:
    """
    Evaluate a model over the validation or test dataloader.

    Args:
        dataloader: Iterable of evaluation batches.
        model: Model to evaluate.
        loss_fn: Loss function used for evaluation.
        device: Device where computations are performed.

    Returns:
        tuple[float, list]: Mean loss and per-batch loss values.
    """
    model.eval()

    loss = 0.0
    total_samples = 0

    losses = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)  # Forward pass
            batch_loss = loss_fn(y_pred, y).item()  # Loss eval
            batch_size = y.shape[0]
            loss += batch_loss * batch_size
            total_samples += batch_size
            losses.append(batch_loss)

    return loss / total_samples, losses


def train_and_eval(
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    device,
    epochs,
    early_stopping,
    scheduling,
    epsilon: float = 1e-3,
    grid_search: bool = False
) -> tuple[list, list]:
    """
    Train a PyTorch model and track training and validation history.

    Args:
        train_loader: Dataloader for the training split.
        val_loader: Dataloader for the validation split.
        model: Model to train.
        loss_fn: Loss function used for optimization and evaluation.
        optimizer: Optimizer used to update model parameters.
        device: Device where computations are performed.
        epochs: Maximum number of training epochs.
        early_stopping: Number of epochs without improvement before stopping.
        scheduling: Optional learning rate scheduling configuration.
        epsilon (float): Minimum improvement in validation loss required to reset
            early stopping patience. Defaults to 1e-3.

    Returns:
        tuple[list, list]: Training loss history, validation loss history, and best epoch.
    """
    model = model.to(device)

    best_val_loss = np.inf
    best_model = None

    epoch_train_losses = []
    epoch_val_losses = []
    batch_train_losses = []
    batch_val_losses = []

    counter = 0
    best_epoch = 0

    scheduler = get_scheduler(scheduling, optimizer)

    iter = range(epochs) if grid_search else tqdm(range(epochs))
    
    for t in iter:

        train_loss, train_losses = train_loop(
            train_loader, model, loss_fn, optimizer, device
        )
        val_loss, val_losses = test_loop(val_loader, model, loss_fn, device)

        batch_train_losses.extend(train_losses)
        batch_val_losses.extend(val_losses)
        epoch_train_losses.append(float(train_loss))
        epoch_val_losses.append(float(val_loss))

        if val_loss + epsilon < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model.state_dict())
            counter = 0
            best_epoch = t
        else:
            counter += 1

        # Early stopping
        if counter >= early_stopping:
            if not grid_search: 
                print(f"Early stopping after epoch: {t}")
            break

        if scheduler is not None:
            scheduler.step()

    # Restore best weights
    if best_model is not None:
        model.load_state_dict(best_model)

    model.train_loss = epoch_train_losses
    model.val_loss = epoch_val_losses
    model.train_batch_loss = batch_train_losses
    model.val_batch_loss = batch_val_losses

    return epoch_train_losses, epoch_val_losses, best_epoch + 1
