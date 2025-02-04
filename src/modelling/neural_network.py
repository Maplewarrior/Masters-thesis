import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, M: int, n_classes: int) -> None:
        super().__init__()
        """
            M (int): Feature dimension of the data
            n_classes (int): Number of classes in the dataset.
        """
        self.M = M
        self.n_classes = n_classes
        self.net = nn.Sequential([nn.Linear(self.M, self.M),
                                  nn.ReLU(),
                                  nn.Linear(self.M, self.n_classes)
                                  ])
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        logits = self.net(x)
        return {'logits': logits, 'probabilities': self.softmax(logits)}