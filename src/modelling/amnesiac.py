import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
from src.data_utils.synthetic_data import create_dataloaders, DataGenerator
from tqdm import tqdm
import pdb


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class AmnesiacModel(nn.Module):
    def __init__(self, M: int, n_classes: int):
        super().__init__()
        self.M = M
        self.n_classes = n_classes
        
        # Define simple neural network for multiclass classification
        self.net = nn.Sequential(
            nn.Linear(self.M, self.M),
            nn.ReLU(),
            nn.Linear(self.M, self.n_classes)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.net(x)
        return {'logits': logits, 'probabilities': self.softmax(logits)}

class AmnesiacTrainer:
    def __init__(self, model: AmnesiacModel, lr=0.001):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_mapping = {}
        self.batch_params = {}
    
    def train(self, train_loader, val_loader, epochs=10):
        with tqdm(range(epochs)) as pbar:
            for epoch in pbar:
                self.model.train()
                total_loss = 0.0
                
                for batch_idx, (x, y, indices) in enumerate(train_loader):
                    x, y = x.float(), y.long()
                    self.optimizer.zero_grad()
                    
                    before_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}
                    
                    outputs = self.model(x)
                    loss = self.criterion(outputs['logits'], y)
                    loss.backward()
                    self.optimizer.step()
                    
                    after_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}
                    
                    self.batch_mapping.setdefault(epoch, {}).update({idx.item(): batch_idx for idx in indices})
                    self.batch_params.setdefault(epoch, {}).update({batch_idx: {name: after_params[name] - before_params[name] for name in before_params}})
                    
                    total_loss += loss.item()
                
                train_loss = total_loss / len(train_loader)
                val_loss, val_acc = self.evaluate(val_loader)
                pbar.set_description(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    def evaluate(self, loader):
        self.model.eval()
        val_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for (x, y, _) in loader:
                x, y = x.float(), y.long()
                outputs = self.model(x)
                
                loss = self.criterion(outputs['logits'], y)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs['probabilities'], dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        return val_loss / len(loader), 100 * correct / total
    
    def forget(self, indices_to_forget):
        batches = {}
        for epoch in self.batch_mapping:
            batches_to_forget = list(set([self.batch_mapping[epoch][idx] for idx in indices_to_forget if idx in self.batch_mapping[epoch]]))
            if batches_to_forget:
                batches[epoch] = batches_to_forget
        
        if not batches:
            print("No batches found for the forget set.")
            return
        
        print("Found batches containing forget set, reverting updates...")
        with torch.no_grad():
            for epoch in batches:
                for batch_idx in batches[epoch]:
                    if batch_idx in self.batch_params[epoch]:
                        grads = self.batch_params[epoch][batch_idx]
                        for name, param in self.model.named_parameters():
                            param -= grads[name]
        
        print("Forgetting complete.")
    
    def test(self, test_loader):
        _, test_acc = self.evaluate(test_loader)
        print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    # Define data generation parameters
    n_classes = 4
    n_samples = 10000
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
    forget_n_points = 5
    forget_class = None
    forget_ood_ratio = 1.0

    # Generate synthetic data
    data_generator = DataGenerator(random_state=42)
    data_generator.generate_data(n_samples=n_samples, n_features=n_features, 
                                 n_informative=n_informative, n_redundant=n_redundant, 
                                 n_outliers=n_outliers, n_classes=n_classes, outlier_scale=outlier_scale, 
                                 outlier_variance=outlier_variance, outlier_class=outlier_class)
    data_generator.split_data(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    data_generator.draw_forget_set(n_points=forget_n_points, class_idx=forget_class, ood_ratio=forget_ood_ratio)

    # Create data loaders
    dataloaders = create_dataloaders(data_generator, batch_size=8, use_indices=True)

    train_full_loader = dataloaders["train_full_loader"]
    val_loader = dataloaders["val_loader"]
    test_loader = dataloaders["test_loader"]
    indices_to_forget = dataloaders["forget_idx_in_train"]

    # Instantiate model and trainer
    model = AmnesiacModel(M=n_features, n_classes=n_classes)
    trainer = AmnesiacTrainer(model, lr=0.1)

    # Train and test model
    trainer.train(train_full_loader, val_loader, epochs=10)
    trainer.test(test_loader)

    # Forget samples and test again
    trainer.forget(indices_to_forget)
    trainer.test(test_loader)