import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_utils.synthetic_data import create_dataloaders, DataGenerator
import os

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
    """A trainer class for the AmnesiacModel that supports unlearning through gradient reversal.
    
    This trainer implements a training loop with checkpoint saving and validation evaluation.
    It tracks parameter updates for each batch to enable selective forgetting of training data.

    Args:
        model (AmnesiacModel): The model to train
        lr (float, optional): Learning rate for optimization. Defaults to 0.001.
        criterion (nn.Module, optional): Loss function. Defaults to CrossEntropyLoss.
        optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to Adam.
        device (str, optional): Device to use for training. Defaults to "cpu".

    Attributes:
        device (str): Device used for training ("cpu", "cuda", etc)
        model (AmnesiacModel): The model being trained
        criterion (nn.Module): Loss function used for training
        optimizer (torch.optim.Optimizer): Optimizer used for training
        batch_mapping (dict): Maps data indices to batch indices for each epoch
        batch_params (dict): Stores parameter updates for each batch
        cache_gradients (bool): Whether to cache gradients for each batch or save them to file
    """

    def __init__(self, model: AmnesiacModel, 
                 lr: float=0.001, 
                 criterion: nn.Module = None,
                  optimizer: optim.Optimizer = None,
                  device: str=None, 
                  cache_gradients: bool = True):
        
        self.device = "cpu" if device is None else device
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr) if optimizer is None else optimizer
        # {epoch: {x_idx: batch_idx}}
        self.batch_mapping = {}
        # {epoch: {batch_idx: param_diff}}
        self.batch_params = {}
        self.cache_gradients = cache_gradients
        if cache_gradients:
            print(f"Caching gradients. This could take a lot of memory.")
    
    def train(self, train_loader, val_loader=None, epochs=10, ckpt=False):
        """Train the model.

        Args:
            train_loader (DataLoader): DataLoader for training data
            val_loader (DataLoader): DataLoader for validation data  
            epochs (int, optional): Number of epochs to train for. Defaults to 10.
            ckpt (bool, optional): Whether to save checkpoints. Defaults to False.
        """
        with tqdm(range(epochs)) as pbar:
            for epoch in pbar:
                self.model.train()
                total_loss = 0.0
                
                for batch_idx, (x, y, indices) in enumerate(train_loader):
                    x, y = x.float(), y.long()
                    x = x.to(self.device)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()
                    
                    before_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}
                    
                    outputs = self.model(x)
                    # Criterion expects logits if outputs is a dictionary, otherwise it expects the output directly (used for resnet)
                    loss = self.criterion(outputs['logits'] if isinstance(outputs, dict) and 'logits' in outputs else outputs, y)
                    loss.backward()
                    self.optimizer.step()
                    
                    after_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}
                    

                    # save difference to file
                    self.batch_mapping.setdefault(epoch, {}).update({idx.item(): batch_idx for idx in indices})


                    param_diff = {name: after_params[name] - before_params[name] for name in before_params}

                    if not self.cache_gradients:    
                        # Save difference to file, and save path to memory
                        param_diff_path = f"results/amnesiac/gradients/epoch_{epoch}/gradients_{epoch}_{batch_idx}.pth"
                        if not os.path.exists(os.path.dirname(param_diff_path)):
                            os.makedirs(os.path.dirname(param_diff_path))


                        # save difference to file, this folder will be bloated quickly
                        with open(param_diff_path, "wb") as f:
                            torch.save(param_diff, f)

                        self.batch_params.setdefault(epoch, {}).update({batch_idx: param_diff_path})
                    else:
                        # Save difference in memory
                        self.batch_params.setdefault(epoch, {}).update({batch_idx: param_diff})


                    total_loss += loss.item()

                if ckpt:
                    self.save_checkpoint(epoch)
                
                train_loss = total_loss / len(train_loader)

                if val_loader is not None:
                    val_loss, val_acc = self.evaluate(val_loader)
                    pbar.set_description(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                else:
                    pbar.set_description(f"Train Loss: {train_loss:.4f}")
    
    def save_checkpoint(self, epoch):
        """Save a checkpoint of the model and training state.

        Args:
            epoch (int): Current epoch number
        """
        results_folder = "results/amnesiac"
        checkpoint_folder = os.path.join(results_folder, "checkpoints")
        checkpoint_path = f"{checkpoint_folder}/checkpoint_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_mapping': self.batch_mapping,
            'batch_params': self.batch_params
        }, checkpoint_path)

    def evaluate(self, loader):
        """Evaluate the model on a data loader.

        Args:
            loader (DataLoader): DataLoader to evaluate on

        Returns:
            tuple: (average loss, accuracy percentage)
        """
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
        """Unlearn specific training examples by reverting their parameter updates.

        Args:
            indices_to_forget (list): List of data indices to forget
        """
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
                        if not self.cache_gradients:
                            grads = torch.load(self.batch_params[epoch][batch_idx])
                        else:
                            grads = self.batch_params[epoch][batch_idx]
                        for name, param in self.model.named_parameters():
                            param -= grads[name]
        
        print("Forgetting complete.")
    
    def test(self, test_loader, dname=None):
        """Evaluate the model on test data.

        Args:
            test_loader (DataLoader): DataLoader for test data
        """
        _, test_acc = self.evaluate(test_loader)
        if dname is not None:
            print(f"{dname} - Test Accuracy: {test_acc:.2f}%")
        else:
            print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":

    # Check for device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    device = "cpu"


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
    forget_ood_ratio = 0.0

    # Generate synthetic data
    data_generator = DataGenerator(random_state=42)
    data_generator.generate_data(n_samples=n_samples, n_features=n_features, 
                                 n_informative=n_informative, n_redundant=n_redundant, 
                                 n_outliers=n_outliers, n_classes=n_classes, outlier_scale=outlier_scale, 
                                 outlier_variance=outlier_variance, outlier_class=outlier_class)
    data_generator.split_data(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    data_generator.draw_forget_set(n_points=forget_n_points, class_idx=forget_class, ood_ratio=forget_ood_ratio)

    # Create data loaders
    dataloaders = create_dataloaders(data_generator, batch_size=8, use_indices=True, device=device)

    train_full_loader = dataloaders["train_full_loader"]
    val_loader = dataloaders["val_loader"]
    test_loader = dataloaders["test_loader"]
    indices_to_forget = dataloaders["forget_idx_in_train"]

    # Instantiate model and trainer
    model = AmnesiacModel(M=n_features, n_classes=n_classes)
    trainer = AmnesiacTrainer(model, lr=0.1, device=device)

    # Train and test model
    trainer.train(train_full_loader, val_loader, epochs=10)
    trainer.test(test_loader)

    # Forget samples and test again
    trainer.forget(indices_to_forget)
    trainer.test(test_loader)
