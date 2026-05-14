from torch import nn

class MLP(nn.Module):
    """
    Configurable multilayer perceptron implemented with PyTorch modules.
    """
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float, activation, batch_norm: bool = False):
        """
        Initialize the multilayer perceptron architecture.

        Args:
            input_dim (int): Number of input features.
            hidden_dims (list): Sizes of the hidden layers.
            output_dim (int): Number of output units.
            dropout (float): Dropout probability applied after each hidden layer.
            activation: Activation class used in hidden layers.
            batch_norm (bool, optional): Whether to add batch normalization after each
                hidden linear layer. Defaults to False.
        """
        super().__init__()
        self.linear_relu_stack = self._build_mlp(input_dim, hidden_dims, output_dim, dropout, activation, batch_norm)
        
    def forward(self, x):
        """
        Run a forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.linear_relu_stack(x)
    
    def _build_mlp(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float, activation, batch_norm: bool):
        """
        Assemble the sequential stack of layers for the MLP.

        Args:
            input_dim (int): Number of input features.
            hidden_dims (list): Sizes of the hidden layers.
            output_dim (int): Number of output units.
            dropout (float): Dropout probability applied after each hidden layer.
            activation: Activation class used in hidden layers.
            batch_norm (bool): Whether to add batch normalization after each hidden layer.

        Returns:
            nn.Sequential: Sequential container with the MLP layers.
        """
        layers = []

        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))

            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)
