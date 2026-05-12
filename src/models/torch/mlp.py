from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float, activation, batch_norm: bool = False):
        super().__init__()
        self.linear_relu_stack = self._build_mlp(input_dim, hidden_dims, output_dim, dropout, activation, batch_norm)
        
    def forward(self, x):
        return self.linear_relu_stack(x)
    
    def _build_mlp(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float, activation, batch_norm: bool):
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
