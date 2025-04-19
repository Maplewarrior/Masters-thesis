import torch
import torch.nn as nn
from src.models.base_model import BaseModel

class NeuralNet(BaseModel):
    def __init__(self, M: int, n_classes: int, seed: int = None) -> None:
        super().__init__()
        """
            M (int): Feature dimension of the data
            n_classes (int): Number of classes in the dataset.
            seed (int, optional): Random seed for weight initialization.
        """
        # Set seed for reproducible weight initialization
        self.set_seed(seed)
        self.M = M
        self.n_classes = n_classes
        self.net = nn.Sequential(nn.Linear(self.M, self.M * 8),
                                 nn.ReLU(),
                                 nn.Linear(self.M * 8, self.M * 4),
                                 nn.ReLU(),
                                 nn.Linear(self.M * 4, self.n_classes))
        
        

    def forward(self, x, start_idx: int = 0, stop_idx: int = None):
        logits = self.net[start_idx:stop_idx](x)
        return {'logits': logits, 'probabilities': self.softmax(logits), 'predictions': torch.argmax(logits, dim=-1)}

class ResidualSkipLinearLayer(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        assert in_features == out_features, 'Cannot apply residual skip connection without preserving dimensionality.'
        self.in_features = in_features
        self.out_features = out_features
        self.lin_layer = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        return self.lin_layer(x) + x

class NeuralNetRS(BaseModel):
    def __init__(self, M: int, n_classes: int, n_layers: int = 3, width_factor: int = 4, seed = None) -> None:
        super().__init__()
        """
            M (int): Feature dimension of the data
            n_classes (int): Number of classes in the dataset.
        """
        # Set seed for reproducible weight initialization
        self.set_seed(seed)
        self.M = int(M)
        self.n_classes = n_classes
        self.width_factor = width_factor
        self.n_layers = int(n_layers)
        hidden_dim = int(self.M * self.width_factor)

        layers = [
                    nn.Linear(self.M, hidden_dim), 
                    nn.ReLU()
                 ]
        for _ in range(self.n_layers):
            layers.append(ResidualSkipLinearLayer(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, self.n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x, start_idx: int = 0, stop_idx: int = None):
        logits = self.net[start_idx:stop_idx](x)
        return {'logits': logits, 'probabilities': self.softmax(logits), 'predictions': torch.argmax(logits, dim=-1)}        

class NeuralNetRS2(BaseModel):
    def __init__(self, M: int, n_classes: int, n_layers: int = 3, width_factor: int = 4, seed = None) -> None:
        super().__init__()
        """
            M (int): Feature dimension of the data
            n_classes (int): Number of classes in the dataset.
        """
        # Set seed for reproducible weight initialization
        self.set_seed(seed)
        self.M = int(M)
        self.n_classes = n_classes
        self.width_factor = width_factor
        self.n_layers = int(n_layers)
        hidden_dim = int(self.M * self.width_factor)
        """
        3 RS connection layers, width_factor=4
        train/accuracy=0.403 train/loss=1.644 val/accuracy=0.379 val/loss=1.788 epoch=8.000

        3 RS connection layers, width_factor=6
        train/accuracy=0.379 train/loss=1.716 val/accuracy=0.361 val/loss=1.779 epoch=8.000

        8 RS connection layers, width_factor=4
        train/accuracy=0.295 train/loss=1.880 val/accuracy=0.280 val/loss=1.920 epoch=11.000

        12 RS connection layers,
        width_factor = 4
        train/accuracy=0.352 train/loss=1.762 val/accuracy=0.318 val/loss=1.899 epoch=8.000:  23%

        """
        
        layers = [
                    nn.Linear(self.M, hidden_dim), 
                    nn.ReLU()
                 ]
        for _ in range(self.n_layers):
            layers.append(ResidualSkipLinearLayer(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
        
        layers.append(nn.Linear(hidden_dim, self.n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x, start_idx: int = 0, stop_idx: int = None):
        logits = self.net[start_idx:stop_idx](x)
        return {'logits': logits, 'probabilities': self.softmax(logits), 'predictions': torch.argmax(logits, dim=-1)}