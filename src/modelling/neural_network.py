import torch
import torch.nn as nn
import pdb

class NeuralNet(nn.Module):
    def __init__(self, M: int, n_classes: int) -> None:
        super().__init__()
        """
            M (int): Feature dimension of the data
            n_classes (int): Number of classes in the dataset.
        """
        self.M = M
        self.n_classes = n_classes
        self.net = nn.Sequential(nn.Linear(self.M, self.M * 8),
                                 nn.ReLU(),
                                 nn.Linear(self.M * 8, self.M * 4),
                                 nn.ReLU(),
                                 nn.Linear(self.M * 4, self.n_classes))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, start_idx: int = 0, stop_idx: int = None):
        logits = self.net[start_idx:stop_idx](x)
        return {'logits': logits, 'probabilities': self.softmax(logits)}

    def inference(self, x, start_idx: int = 0, stop_idx: int = None):
        self.eval()
        with torch.no_grad():
            return self(x, start_idx, stop_idx)

class NeuralNetwork(nn.Module):
    def __init__(self, M: int, n_classes: int) -> None:
        """
        Initialize the neural network.
        
        Args:
            M (int): Feature dimension of the data
            n_classes (int): Number of classes in the dataset.
        """
        super().__init__()
        # Input validation
        if M <= 0 or n_classes <= 0:
            raise ValueError("Feature dimension and number of classes must be positive")
            
        self.M = M
        self.n_classes = n_classes
        self.net = nn.Sequential(nn.Linear(self.M, self.M),
                               nn.ReLU(),
                               nn.Linear(self.M, self.n_classes)
                               )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        logits = self.net(x)
        return {'logits': logits, 'probabilities': self.softmax(logits)}
    
    def fit(self, X, y, n_epochs: int = 10, batch_size: int = 32, lr: float = 0.01):
        # Convert inputs to torch tensors if they aren't already
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.LongTensor(y)
        
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Convert to DataLoader for batching
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(n_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self(batch_X)
                loss = criterion(output['logits'], batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
        print(f'Average Loss: {avg_loss:.4f}')

        return self
    
    def predict(self, X):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = self(X)
            predictions = torch.argmax(output['probabilities'], dim=1)
        self.train()  # Set the model back to training mode
        return predictions
    
    def predict_proba(self, X):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = self(X)
            probabilities = output['probabilities']
        self.train()  # Set the model back to training mode
        return probabilities
    