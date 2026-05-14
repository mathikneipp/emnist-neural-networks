

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
    
