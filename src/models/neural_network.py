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
        self.M = M
        self.n_classes = n_classes
        self.net = nn.Sequential(nn.Linear(self.M, self.M * 8),
                                 nn.ReLU(),
                                 nn.Linear(self.M * 8, self.M * 4),
                                 nn.ReLU(),
                                 nn.Linear(self.M * 4, self.n_classes))
        
        # Set seed for reproducible weight initialization
        self.set_seed(seed)

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
    def __init__(self, M: int, n_classes: int) -> None:
        super().__init__()
        """
            M (int): Feature dimension of the data
            n_classes (int): Number of classes in the dataset.
        """
        self.M = M
        self.n_classes = n_classes
        self.net = nn.Sequential(nn.Linear(self.M, self.M * 4),
                                 nn.ReLU(),
                                 ResidualSkipLinearLayer(self.M*4, self.M*4),
                                 nn.ReLU(),
                                 ResidualSkipLinearLayer(self.M*4, self.M*4),
                                 nn.ReLU(),
                                 ResidualSkipLinearLayer(self.M*4, self.M*4),
                                 nn.ReLU(),
                                 nn.Linear(self.M * 4, self.n_classes))

    def forward(self, x, start_idx: int = 0, stop_idx: int = None):
        logits = self.net[start_idx:stop_idx](x)
        return {'logits': logits, 'probabilities': self.softmax(logits), 'predictions': torch.argmax(logits, dim=-1)}


