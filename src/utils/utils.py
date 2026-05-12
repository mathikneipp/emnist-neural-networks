

def get_best_config(models: list, configs: list):
    best_index = 0
    best_val = float("inf")
    for i in range(len(models)):
        best_model_val = min(models[i].val_loss)
        if best_model_val < best_val:
            best_val = best_model_val
            best_index = i

    config = configs[best_index]
    return config
    