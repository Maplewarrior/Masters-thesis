# FROM: https://github.com/cleverhans-lab/machine-unlearning/blob/master/architectures/purchase.py
import torch
from torch.nn import Module, Linear
from torch.nn.functional import tanh
from torch.nn import Softmax, CrossEntropyLoss

class Model(Module):
    def __init__(self, input_shape, nb_classes, *args, **kwargs):
        super(Model, self).__init__()
        self.fc1 = Linear(input_shape[0], 128)
        self.fc2 = Linear(128, nb_classes)
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = tanh(x)
        x = self.fc2(x)
        return {'logits': x, 'probabilities': self.softmax(x)}

    def fit(self, X, y, n_epochs: int = 10, batch_size: int = 32, lr: float = 0.01):
        # Convert inputs to torch tensors if they aren't already
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.LongTensor(y)
        
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = CrossEntropyLoss()
        
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