import torch 
from torch.utils.data import Dataset
class SyntheticDataset(Dataset):
    def __init__(self, X, y, n_classes, dataset_name: str = None):
        """
        A PyTorch Dataset for synthetic data with optional metadata.
        
        Args:
            X (numpy.ndarray or torch.Tensor): Features of shape (num_samples, num_features).
            y (numpy.ndarray or torch.Tensor): Labels of shape (num_samples,).
            n_classes (int, optional): Number of classes in the dataset. If None, will be inferred from y.
            dataset_name (str, optional): Name identifier for the dataset.
        
        Note:
            The dataset automatically converts numpy arrays to torch tensors if needed.
            Labels are kept as integers (not one-hot encoded).
        """
        # Convert features to float32
        self.X = torch.tensor(X, dtype=torch.float32)
        # Convert labels to long (integer)
        self.y = torch.tensor(y, dtype=torch.long)
        self.indices = torch.arange(len(self.X))

        self.n_classes = n_classes

        self.name = dataset_name

        # if it is not onehot endcoded, do it
        if y.dim() == 1 and n_classes is not None and n_classes > 1:
            self.y = self.onehot_encode_labels(self.y, self.n_classes)
        elif y.dim() == 1 and n_classes == 1:
            self.y = torch.zeros(len(y), 1)
            self.y[y == 1] = 1

    def __len__(self):
        return len(self.X)
    
    def onehot_encode_labels(self, y, n_classes):
        if n_classes is None:
            n_classes = y.max() + 1
        return torch.zeros(len(y), n_classes).scatter_(1, y.unsqueeze(1), 1)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.indices[idx]
        