import torch


def get_best_config(models: list, configs: list):
    """
    Select the configuration associated with the best validation loss.

    Args:
        models (list): Trained models with validation loss history.
        configs (list): Configurations associated with each model.

    Returns:
        dict: Configuration of the best-performing model.
    """
    best_index = 0
    best_val = float("inf")
    for i in range(len(models)):
        best_model_val = min(models[i].val_loss)
        if best_model_val < best_val:
            best_val = best_model_val
            best_index = i

    config = configs[best_index]
    return config


def get_scheduler(scheduling, optimizer):
    scheduler = None

    if scheduling is not None:
        if scheduling["type"] == "linear":
            lr_0 = optimizer.param_groups[0]["lr"]
            lr_lambda = lambda epoch: max(
                scheduling["lr_min"] / lr_0, 1 - (scheduling["k"] / lr_0) * epoch
            )
        elif scheduling["type"] == "exponential":
            lr_lambda = lambda epoch: scheduling["gamma"] ** epoch

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return scheduler
