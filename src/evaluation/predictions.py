import numpy as np
import torch

from ..models.custom.neural_network import SecuentialNeuralNetwork
from ..models.torch.mlp import MLP

def get_predictions(model, X: np.ndarray, device) -> np.ndarray:
    """
    Generate class predictions from either a custom or PyTorch model.

    Args:
        model: Trained model instance.
        X (np.ndarray): Input samples to evaluate.
        device: Device used for PyTorch-based models.

    Returns:
        np.ndarray: Predicted class indices.
    """
    if isinstance(model, SecuentialNeuralNetwork):
        y_pred_one_hot = model.predict(X)
        y_pred = np.argmax(y_pred_one_hot, axis=1)
        
    elif isinstance(model, MLP):
        X_tensor = torch.from_numpy(X).float().to(device)
        
        model.eval()

        with torch.no_grad():
            logits = model(X_tensor)
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    
    return y_pred
