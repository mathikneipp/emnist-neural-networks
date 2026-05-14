import torch
from tqdm import tqdm
import numpy as np
from copy import deepcopy


def train_loop(dataloader, model, loss_fn, optimizer, device) -> tuple[list]:
    """
    Run one training epoch over the dataloader.

    Args:
        dataloader: Iterable of training batches.
        model: Model to optimize.
        loss_fn: Loss function used for optimization.
        optimizer: Optimizer used to update model parameters.
        device: Device where computations are performed.

    Returns:
        tuple[list]: Batch loss values collected during the epoch.
    """
    model.train()
    losses = []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred = model(X)  # Forward pass
        batch_loss = loss_fn(y_pred, y)  # Loss eval
        losses.append(batch_loss.item())
        batch_loss.backward()  # Backward pass
        optimizer.step()  # Step

    return losses


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

    num_batches = len(dataloader)
    loss = 0.0

    losses = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)  # Forward pass
            batch_loss = loss_fn(y_pred, y).item()  # Loss eval
            loss += batch_loss
            losses.append(batch_loss)

    return loss / num_batches, losses


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
    scheduler = None
    best_epoch = 0
    
    if scheduling is not None:
        if scheduling["type"] == "linear":
            lr_0 = optimizer.param_groups[0]["lr"]
            lr_lambda = lambda epoch: max(
                scheduling["lr_min"] / lr_0, 1 - (scheduling["k"] / lr_0) * epoch
            )
        elif scheduling["type"] == "exponential":
            lr_lambda = lambda epoch: scheduling["gamma"] ** epoch

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    for t in tqdm(range(epochs)):

        train_losses = train_loop(train_loader, model, loss_fn, optimizer, device)
        val_loss, val_losses = test_loop(val_loader, model, loss_fn, device)

        batch_train_losses.extend(train_losses)
        batch_val_losses.extend(val_losses)
        epoch_train_losses.append(float(np.mean(train_losses)))
        epoch_val_losses.append(float(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model.state_dict())
            counter = 0
            best_epoch = t
        else:
            counter += 1

        # Early stopping
        if counter >= early_stopping:
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
