import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_utils.synthetic_data import create_dataloaders, DataGenerator
import os
import json

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
                 lr: float=0.0001, 
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
            print(f"Caching gradients. This could take a lot of memory. Download some more RAM if you run out.")
        else:
            print(f"Save gradients to file. This will save memory. Not storage space though :'(")
    
    def train(self, 
              train_loader, 
              val_loader=None, 
              epochs=10, 
              ckpt=False, 
              class_to_forget=None, 
              indices_to_forget=None, 
              repair=False, 
              save_accuracy_to_file=False,
              save_accuracy_to_file_name=None,
              retain_loader=None,
              forget_loader=None):
        """Train the model.

        Args:
            train_loader (DataLoader): DataLoader for training data
            val_loader (DataLoader): DataLoader for validation data  
            epochs (int, optional): Number of epochs to train for. Defaults to 10.
            ckpt (bool, optional): Whether to save checkpoints. Defaults to False.
            class_to_forget (int, optional): Class to forget. Defaults to None. If both this and indices_to_forget are None, gradients for all classes are cached.
            indices_to_forget (list, optional): List of data indices to forget. Defaults to None. If both this and class_to_forget are None, gradients for all classes are cached.
            repair (bool, optional): Whether to train in the repair phase. Defaults to False. If True, no gradients are stored for forgetting.
            save_accuracy_to_file (bool, otional): Whether to save accuracy to file. Defaults to False.
            save_accuracy_to_file_name (str, optional): Name of file to save accuracy to. Defaults to None. If None, accuracy is saved to file with name "accuracies.txt".
            retain_loader (DataLoader, optional): DataLoader for retain set. Defaults to None. Used to evaluate accuracy on retain set. If not None, this accuracy is not saved to file if save_accuracy_to_file is True.
            forget_loader (DataLoader, optional): DataLoader for forget set. Defaults to None. Used to evaluate accuracy on forget set. If not None, this accuracy is not saved to file if save_accuracy_to_file is True.
        """

        # not both class_to_forget and indices_to_forget can be specified
        if class_to_forget is not None and indices_to_forget is not None:
            raise ValueError("Cannot specify both class_to_forget and indices_to_forget. Please specify only one.")

        accuracies_validation = []
        accuracies_forget = []
        accuracies_retain = []

        with tqdm(range(epochs)) as pbar:
            for epoch in pbar:
                self.model.train()
                total_loss = 0.0


                forget_accuracy = []
                retain_accuracy = []
                
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
                
                    param_diff = {name: after_params[name] - before_params[name] for name in before_params}
                    

                    # Save batch mapping and param diff if either:
                    # 1. We're not targeting a specific class (class_to_forget is None)
                    # 2. The batch contains samples from the class we want to forget
                    # 3. The batch contains indices we want to forget
                    should_save = ((class_to_forget is None) and (indices_to_forget is None)) or \
                                ((class_to_forget is not None) and (class_to_forget in y)) or \
                                ((indices_to_forget is not None) and any(idx.item() in indices_to_forget for idx in indices))
                    
                    if repair:
                        should_save = False
                    
                    if should_save:
                        self.batch_mapping.setdefault(epoch, {}).update({idx.item(): batch_idx for idx in indices})
                        param_diff_path = self.save_param_diff(param_diff, epoch, batch_idx)

                    total_loss += loss.item()

                if ckpt:
                    self.save_checkpoint(epoch)
                
                train_loss = total_loss / len(train_loader)

                if val_loader is not None:
                    val_loss, val_acc = self.evaluate(val_loader)
                    pbar.set_description(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

                    accuracies_validation.append(val_acc)

                else:
                    pbar.set_description(f"Train Loss: {train_loss:.4f}")

                if forget_loader is not None:
                    forget_loss, forget_acc = self.evaluate(forget_loader)
                    accuracies_forget.append(forget_acc)

                if retain_loader is not None:
                    retain_loss, retain_acc = self.evaluate(retain_loader)
                    accuracies_retain.append(retain_acc)

        accuracies_dict = {
            "validation": accuracies_validation,
        }

        if forget_loader is not None:
            accuracies_dict["forget"] = accuracies_forget
        if retain_loader is not None:
            accuracies_dict["retain"] = accuracies_retain

        # save accuracies to json file
        if save_accuracy_to_file is not None:

            if save_accuracy_to_file_name is not None:
                file_name = f"results/amnesiac/accuracy/{save_accuracy_to_file_name}_accuracies.json"
            else:
                file_name = f"results/amnesiac/accuracy/accuracies.json"
            if not os.path.exists(os.path.dirname(file_name)):
                os.makedirs(os.path.dirname(file_name))
            with open(file_name, "w") as f:
                json.dump(accuracies_dict, f)

    
    def save_param_diff(self, param_diff: torch.Tensor, epoch: int, batch_idx: int) -> str:
        """
        Save gradient differences either to file or memory.
        
        Args:
            param_diff: The parameter difference tensor to save
            epoch: Current epoch number
            batch_idx: Current batch index
            
        Returns:
            str: Path to saved gradients if cache_gradients is False, else empty string
        """
        if not self.cache_gradients:    
            # Save difference to file, and save path to memory
            param_diff_path = f"results/amnesiac/gradients/epoch_{epoch}/gradients_{epoch}_{batch_idx}.pth"
            if not os.path.exists(os.path.dirname(param_diff_path)):
                os.makedirs(os.path.dirname(param_diff_path))

            # save difference to file
            with open(param_diff_path, "wb") as f:
                torch.save(param_diff, f)

            self.batch_params.setdefault(epoch, {}).update({batch_idx: param_diff_path})
            return param_diff_path
        else:
            # Save difference in memory
            self.batch_params.setdefault(epoch, {}).update({batch_idx: param_diff})
            return ""

    def save_checkpoint(self, epoch):
        """Save a checkpoint of the model and training state.

        Args:
            epoch (int): Current epoch number
        """
        results_folder = "results/amnesiac"
        checkpoint_folder = os.path.join(results_folder, "checkpoints")
        checkpoint_path = f"{checkpoint_folder}/checkpoint_{epoch}.pth"
        # create the directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
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
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)

                loss = self.criterion(outputs['logits'] if isinstance(outputs, dict) and 'logits' in outputs else outputs, y)
                val_loss += loss.item()
                

                # depending on the model, resnet does not return probabilities
                if isinstance(outputs, dict) and 'probabilities' in outputs:
                    preds = torch.argmax(outputs['probabilities'], dim=1)
                else:
                    preds = torch.argmax(outputs, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)
        
        return val_loss / len(loader), correct / total
    
    def forget(self, indices_to_forget=None):
        """Unlearn specific training examples by reverting their parameter updates.

        Args:
            indices_to_forget (list): List of data indices to forget. If None, all sensitive batches stored are reverted. Be sure to specifiy while training which should be sensitive batches.
        """
        batches = {}
        
        for epoch in self.batch_mapping:
            if indices_to_forget is not None:
                batches_to_forget = list(set([self.batch_mapping[epoch][idx] for idx in indices_to_forget if idx in self.batch_mapping[epoch]]))
            else:
                batches_to_forget = list(set(self.batch_mapping[epoch].values()))
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
    
    def test(self, test_loader, dname=None, save_to_file_name=None):
        """Evaluate the model on test data.

        Args:
            test_loader (DataLoader): DataLoader for test data
        """
        _, test_acc = self.evaluate(test_loader)


        if dname is not None:
            print(f"{dname}: {test_acc*100:.2f}%")
        else:
            print(f"Test Accuracy: {test_acc*100:.2f}%")

        if save_to_file_name is not None:
            file_name = f"results/amnesiac/accuracy/{save_to_file_name}.txt"
            if not os.path.exists(os.path.dirname(file_name)):
                os.makedirs(os.path.dirname(file_name))
            with open(file_name, "a") as f:
                f.write(f"{test_acc:.4f}\n")

        return test_acc

if __name__ == "__main__":

    # Check for device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    device = "cpu"


    # Define data generation parameters
    n_classes = 4
    n_samples = 1000
    n_features = 2
    n_informative = n_features
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
    forget_n_points = 10
    forget_class = 1 # Forget a specific class
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
    dataloaders = create_dataloaders(data_generator, batch_size=1, use_indices=True, device=device)

    train_full_loader = dataloaders["train_full_loader"]
    retain_loader = dataloaders["train_retain_loader"]
    forget_loader = dataloaders["train_forget_loader"]
    val_loader = dataloaders["val_loader"]
    test_loader = dataloaders["test_loader"]
    indices_to_forget = dataloaders["forget_idx_in_train"]

    # Instantiate model and trainer
    model = AmnesiacModel(M=n_features, n_classes=n_classes)
    trainer = AmnesiacTrainer(model, lr=0.1, device=device)

    # ================================================
    # Examples of usage below
    # ================================================

    # %%
    # ============= Only store gradients for forget set =============
    trainer.train(train_full_loader, val_loader, epochs=20, 
                  indices_to_forget=indices_to_forget, 
                  save_accuracy_to_file=True, 
                  save_accuracy_to_file_name="training_phase", 
                  retain_loader=retain_loader, 
                  forget_loader=forget_loader)
    
    trainer.test(test_loader, dname="Before forgetting, Test accuracy", save_to_file_name="before_forgetting_test_accuracy")
    trainer.test(retain_loader, dname="Before forgetting, Retain accuracy", save_to_file_name="before_forgetting_retain_accuracy")
    trainer.test(forget_loader, dname="Before forgetting, Forget accuracy", save_to_file_name="before_forgetting_forget_accuracy")

    trainer.forget(indices_to_forget=None) # Set to None to forget all sensitive batches, used when storing only gradients for sensitive batches

    trainer.test(test_loader, dname="After forgetting, Test accuracy", save_to_file_name="after_forgetting_test_accuracy")
    trainer.test(retain_loader, dname="After forgetting, Retain accuracy", save_to_file_name="after_forgetting_retain_accuracy")
    trainer.test(forget_loader, dname="After forgetting, Forget accuracy", save_to_file_name="after_forgetting_forget_accuracy")

    # %%
    # ============= Repair performance on retain set =============
    trainer.train(retain_loader, 
                  val_loader, 
                  epochs=10, 
                  repair=True, 
                  save_accuracy_to_file=True, 
                  save_accuracy_to_file_name="repair_phase") # repair performance on retain set
    
    trainer.test(test_loader, dname="After repair, Test accuracy", save_to_file_name="after_repair_test_accuracy")
    trainer.test(retain_loader, dname="After repair, Retain accuracy", save_to_file_name="after_repair_retain_accuracy")
    trainer.test(forget_loader, dname="After repair, Forget accuracy", save_to_file_name="after_repair_forget_accuracy")

    # %%
    # ============= Store all gradients, pick indices to forget =============
    trainer.train(train_full_loader, val_loader, epochs=10)
    trainer.test(test_loader)
    trainer.forget(indices_to_forget)
    trainer.test(test_loader)

    # %%
    # ============= Forget specific class =============
    # Train and test model only storing gradients for the forget set if it is in specified class
    trainer.train(train_full_loader, val_loader, epochs=10, class_to_forget=forget_class)
    trainer.test(test_loader)

    # Forget samples and test again
    trainer.forget(indices_to_forget=None) # Set to None to forget all sensitive batches, used when storing only sensitive batches
    trainer.test(test_loader)



