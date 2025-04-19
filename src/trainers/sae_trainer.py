import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.trainers.base_trainer import BaseTrainer


class SAETrainer(BaseTrainer):
    def __init__(self, 
                 model: nn.Module,
                 train_dataloader, 
                 val_dataloader,
                 logger=None,
                 device="cpu",
                 learning_rate=0.001,
                 n_epochs=100,
                 disable_tqdm=False,
                 do_early_stopping=True,
                 **kwargs) -> None:
        """
        A class that supports training a SAE to reconstruct the activations of a trained neural network.
        This class is compatible with the SAEUnlearner class.
        """
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        super().__init__(model, self.optimizer, train_dataloader, val_dataloader, logger, disable_tqdm, do_early_stopping, n_epochs, device)

    
    
    def init_epoch_metrics(self):
        return {'accuracy': 0, 'loss': 0, 'l0-norm': 0}
    
    def update_epoch_metrics(self, metrics: dict, out: dict, y, N: int):
        metrics['accuracy'] += (out['probabilities'].argmax(dim=1) == y.argmax(dim=1)).sum().item() / (y.size(0) * N)
        metrics['l0-norm'] += out['z'].norm(p=0, dim=-1).mean() / N
        return metrics
    
    def init_train_metrics(self):
        return {'train/accuracy': [], 'train/loss': [], 'train/l0-norm': [],
                'val/accuracy': [], 'val/loss': [], 'val/l0-norm': [], 'epoch': []}
    
    def update_train_metrics(self, train_metrics: dict, epoch_metrics: dict, epoch_type: str):
        assert epoch_type in ['train', 'val'], f'Invalid epoch_type: "{epoch_type}". Options are: {["train", "val"]}'
        train_metrics[f'{epoch_type}/accuracy'].append(epoch_metrics['accuracy'])
        train_metrics[f'{epoch_type}/loss'].append(epoch_metrics['loss'])
        train_metrics[f'{epoch_type}/l0-norm'].append(epoch_metrics['l0-norm'])
        return train_metrics
    
    # def step(self, x: torch.tensor, y: torch.tensor):
    #     self.optimizer.zero_grad()
    #     out = self.model(x)
    #     loss = self.model.loss(out)
    #     loss.backward()
    #     self.optimizer.step()
    #     # compute remaning forward pass of neural network using reconstructed activation
    #     pred = self.model.predict_from_reconstruction(out['xhat'])
    #     out['probabilities'] = pred['probabilities']
    #     return out, loss
    
