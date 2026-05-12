import torch
from tqdm import tqdm
import numpy as np
from copy import deepcopy


def train_loop(dataloader, model, loss_fn, optimizer, device) -> tuple[list]:
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
    lr_0
) -> tuple[list, list]:
    model = model.to(device)

    best_val_loss = np.inf
    best_model = None

    epoch_train_losses = []
    epoch_val_losses = []
    batch_train_losses = []
    batch_val_losses = []

    counter = 0
    scheduler = None
    
    if scheduling is not None:
        if scheduling["type"] == "linear":
            lr_lambda = lambda epoch: max(scheduling["lr_min"], lr_0 - scheduling["k"] * epoch)
        elif scheduling["type"] == "exponential":
            lr_lambda = lambda epoch: lr_0 * (scheduling["gamma"]**epoch)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda
        )
    
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

    return epoch_train_losses, epoch_val_losses
