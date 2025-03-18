import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def build_model(model_type, model_parameters):
    if model_type == 'neural-network':
        return NeuralNetwork(**model_parameters)
    

class BaseTrainer:
    def __init__(self, 
                 model: nn.Module,
                 optimizer, 
                 train_dataloader, 
                 val_dataloader,
                 logger,
                 disable_tqdm: bool,
                 do_early_stopping: bool) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.disable_tqdm = disable_tqdm
        self.do_early_stopping = do_early_stopping
    
    def train_one_epoch(self) -> tuple[list[float], float]:
        self.model.train()
        epoch_loss = []
        epoch_acc = 0
        n_samples = 0

        for x, y in self.train_dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            # perform forward and backward pass
            out, loss = self.step(x, y)

            # Update metrics
            batch_correct = (out['probabilities'].argmax(dim=1) == y.argmax(dim=1)).sum().item()
            epoch_acc += batch_correct
            epoch_loss.append(loss.item())
            n_samples += len(y)
        
        epoch_acc = epoch_acc / n_samples

        return epoch_loss, epoch_acc

    def train(self):
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        self.model = self._initialize_model()
        self.optimizer = self._initialize_optimizer()

        with tqdm(range(self.train_parameters['n_epochs']), disable=self.train_parameters['disable_tqdm']) as epoch_pbar:
            for epoch in epoch_pbar:
                # train one epoch
                epoch_loss, epoch_accuracy = self.train_one_epoch()

                self.model.step()
                # run inference on validation set
                val_loss, val_acc = self.eval()

                # store dynamics
                train_losses.append(epoch_loss)
                train_accs.append(epoch_accuracy)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                avg_epoch_loss = np.mean(epoch_loss)
                
                # log validation loss and accuracy
                if self.logger:
                    self.logger.log({
                        f"train/loss/{self.dataset_name}": avg_epoch_loss,
                        f"train/accuracy/{self.dataset_name}": epoch_accuracy,
                        f"validation/loss/{self.dataset_name}": val_loss,
                        f"validation/accuracy/{self.dataset_name}": val_acc,
                        "epoch": epoch
                    })

                # Update progress bar
                epoch_pbar.set_description(
                    f"epoch={epoch+1}/{self.train_parameters['n_epochs']}, "
                    f"loss={avg_epoch_loss:.4f}, "
                    f"acc={epoch_accuracy:.4f}, "
                    f"val_acc: {val_acc}"
                )
                
                if self.train_parameters['do_early_stopping'] and epoch > 4 and not val_losses[-1] < np.mean(val_losses[:-4:-1]):
                    print("\Ending due to early stopping")
                    return train_losses, train_accs, val_losses, val_accs

        return train_losses, train_accs, val_losses, val_accs

    def step(self, x: torch.tensor, y: torch.tensor) -> tuple[dict, torch.tensor]:
        self.optimizer.zero_grad()
        out = self.model(x)
        # Compute loss and update
        loss = self.model.loss(out, y)
        loss.backward()
        self.optimizer.step()
        return out, loss

    def eval(self):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in self.val_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                out = self.model(x)
                loss = self.criterion(out['logits'], y)
                
                total_loss += loss.item()
                total_correct += ((out['probabilities'].argmax(dim=1)) == y.argmax(dim=1)).sum().item()
                total_samples += len(y)
        
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def __call__(self):
        return self.train()
    


class SISATrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def train_client_models(self):
        # TODO: Call self.train() as many times as there are n_clients 
        raise NotImplementedError()
    
    def __call__(self):
        return self.train_client_models()
        