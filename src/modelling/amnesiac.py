import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
from src.data_utils.synthetic_data import create_dataloaders, DataGenerator
from tqdm import tqdm
import pdb


class AmnesiacModel(nn.Module):
    def __init__(self, M: int, n_classes: int, track_all=True):
        """
        M: Number of input features
        n_classes: Number of output classes
        track_all: If True, track gradients for all points, else track only the forget set
        """
        super().__init__()
        self.M = M
        self.n_classes = n_classes
        self.track_all = track_all  # Determines whether to track all gradients or just the forget set
        
        # Define simple neural network for multiclass classification
        self.net = nn.Sequential(
            nn.Linear(self.M, self.M),
            nn.ReLU(),
            nn.Linear(self.M, self.n_classes)
        )
        self.softmax = nn.Softmax(dim=-1)

        # Dictionary to store gradients
        self.gradient_storage = defaultdict(list)
    

        # batch_mappings: {epoch: {sample_idx: batch_idx}}
        self.batch_mapping = {}
        # batch_grads: {epoch: {batch_idx: {param_name: param_grad}}}
        self.batch_grads = {}

    def forward(self, x):
        logits = self.net(x)
        return {'logits': logits, 'probabilities': self.softmax(logits)}

    def train_model(self, train_loader, val_loader, epochs=10, lr=0.001):
        """
        Train the model while keeping track of gradients.
        If self.track_all is False, gradients are only tracked for forget_loader data points.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        with tqdm(range(epochs)) as pbar:
            for epoch in pbar:
                # Training
                self.train()
                total_loss = 0.0

                for batch_idx, (x, y, indices) in enumerate(train_loader):
                    x, y = x.float(), y.long()
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self(x)


                    preds = torch.argmax(outputs['probabilities'], dim=1)
                    
                    loss = criterion(outputs['logits'], y)

                    # Backward pass
                    loss.backward()

                    # Saving the gradients and batch mapping
                    self.batch_mapping.setdefault(epoch, {}).update({idx.item(): batch_idx for idx in indices})
                    self.batch_grads.setdefault(epoch, {}).update({batch_idx: {name: param.grad for name, param in self.named_parameters()}})

                    optimizer.step()
                    total_loss += loss.item()

                train_loss = total_loss/len(train_loader)

                # Validation
                self.eval()
                val_loss = 0
                correct = 0
                total = 0

                val_loss = 0
                correct = 0
                total = 0

                with torch.no_grad():
                    for (x, y, indices) in val_loader:
                        x, y = x.float(), y.long()
                        outputs = self(x)
                        
                        loss = criterion(outputs['probabilities'], y)
                        val_loss += loss.item()


                        
                        preds = torch.argmax(outputs['probabilities'], dim=1)
                        correct += (preds == y).sum().item()
                        total += y.size(0)
                
                val_loss = val_loss/len(val_loader)
                val_acc = 100 * correct / total

                pbar.set_description(f"Train Loss: {train_loss:.4f}, Val Loss: {str(val_loss)}, Val Acc: {str(val_acc)}%")

    def forget(self, indices_to_forget):
        """
        Amnesiac unlearning: Reverts parameter updates from batches containing the forget set.
        """

        batches = {}

        # obtain all the batches that contain the forget set for each epoch
        for epoch in self.batch_mapping:
            batches_to_forget = list(set([self.batch_mapping[epoch][idx] for idx in indices_to_forget]))
            batches[epoch] = batches_to_forget


        sensitive_batches = []


        with torch.no_grad():
            # now get the gradients for each batch
            for epoch in batches:
                for batch_idx in batches[epoch]:
                    grads = self.batch_grads[epoch][batch_idx]
                    sensitive_batches.append(grads)

            print("Sensitive batches collected using forget set ids.")

            # Sum all the sensitive batches 
            for grads in sensitive_batches:
                for name, param in self.named_parameters():
                    param -= grads[name]


        print("Forgetting complete.")


if __name__ == "__main__":
    # Define data generation parameters
    n_classes = 4
    n_samples = 1000
    n_features = 2
    n_informative = 2
    n_redundant = 0
    n_outliers = 50
    outlier_scale = 4.0
    outlier_variance = 0.4
    outlier_class = 1

    # Define data splitting parameters 
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # Forget set parameters
    forget_n_points = 20
    forget_class = 1
    forget_ood_ratio = 0.5

    # Define model parameters
    M = n_features
    n_classes = 4
    track_all = False

    # Generate synthetic data
    data_generator = DataGenerator(random_state=42)
    data_generator.generate_data(n_samples=n_samples, n_features=n_features, 
                                 n_informative=n_informative, n_redundant=n_redundant, 
                                 n_outliers=n_outliers, n_classes=n_classes, outlier_scale=outlier_scale, 
                                 outlier_variance=outlier_variance, outlier_class=outlier_class)
    data_generator.split_data(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    data_generator.draw_forget_set(n_points=forget_n_points, class_idx=forget_class, ood_ratio=forget_ood_ratio)


    # Create data loaders
    dataloaders = create_dataloaders(data_generator, batch_size=32, use_indices=True)

    # Create data loaders
    train_full_loader = dataloaders["train_full_loader"]
    val_loader = dataloaders["val_loader"]
    test_loader = dataloaders["test_loader"]

    indices_to_forget = dataloaders["forget_idx_in_train"]
    indices_to_retain = dataloaders["retain_idx_in_train"]

    model = AmnesiacModel(M=M, n_classes=n_classes, track_all=track_all)
    model.train_model(train_full_loader, val_loader, epochs=200, lr=0.01)
    model.forget(indices_to_forget)
    # model.test(test_loader)



