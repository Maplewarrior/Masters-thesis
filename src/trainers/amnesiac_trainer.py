import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import numpy as np
from src.trainers.base_trainer import BaseTrainer

from src.trainers.base_trainer import BaseTrainer


class AmnesiacTrainer(BaseTrainer):
    def __init__(self, 
                 model: nn.Module,
                 learning_rate: float,
                 train_dataloader, 
                 val_dataloader,
                 logger,
                 disable_tqdm: bool,
                 do_early_stopping: bool,
                 n_epochs: int = 20,
                 device: str = 'cpu',
                 cache_gradients: bool = True) -> None:
        self.model = model
        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.disable_tqdm = disable_tqdm
        self.do_early_stopping = do_early_stopping
        self.n_epochs = n_epochs
        self.device = device
        self.cache_gradients = cache_gradients
        self.model.cache_gradients = cache_gradients
        # {epoch: {x_idx: batch_idx}}
        self.batch_mapping = {}
        # {epoch: {batch_idx: param_diff}}
        self.batch_params = {}

        super().__init__(model, optimizer, train_dataloader, val_dataloader, logger, disable_tqdm, do_early_stopping, n_epochs, device)

    def init_epoch_metrics(self):
        return {'accuracy': 0, 'loss': 0}
    
    def init_train_metrics(self):
        return {'train/accuracy': [], 'train/loss': [],
                'val/accuracy': [], 'val/loss': [], 'epoch': []}

    def train_one_epoch(self, epoch: int, **kwargs) -> tuple[list[float], float]:
        # self.model.train()
        # epoch_losses = []
        # epoch_metrics = self.init_epoch_metrics()

        # for batch in self.train_dataloader:
        #     x = batch[0].to(self.device)
        #     y = batch[1].to(self.device)
        #     # perform forward and backward pass
        #     out, loss = self.step(x, y)
        #     # Update/store metrics
        #     epoch_metrics = self.update_epoch_metrics(epoch_metrics, out, y)
        #     epoch_losses.append(loss.item())
        
        # epoch_metrics['loss'] = sum(epoch_losses) / len(epoch_losses)
        # epoch_metrics['accuracy'] = epoch_metrics['accuracy'] / len(epoch_losses)
        # return epoch_metrics

        class_to_forget = kwargs.get('class_to_forget', None)
        indices_to_forget = kwargs.get('indices_to_forget', None)
        repair = kwargs.get('repair', False)
        ckpt = kwargs.get('ckpt', False)


        self.model.train()
        epoch_losses = []
        epoch_metrics = self.init_epoch_metrics()
        # total_loss = 0.0

        for batch_idx, batch in enumerate(self.train_dataloader):
            assert len(batch) == 3, f"Batch must contain x, y, and indices. Got length {len(batch)}"
            x, y, indices = batch[0], batch[1], batch[2]
            x = x.to(self.device)
            y = y.to(self.device)
            out, loss, param_diff = self.step(x, y)

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

                # Update the model's batch mapping and param diff
                self.model.batch_mapping.setdefault(epoch, {}).update({idx.item(): batch_idx for idx in indices})
                self.save_param_diff(param_diff, epoch, batch_idx)

            # Update/store metrics
            epoch_metrics = self.update_epoch_metrics(epoch_metrics, out, y)
            epoch_losses.append(loss.item())
        
        epoch_metrics['loss'] = sum(epoch_losses) / len(epoch_losses)
        epoch_metrics['accuracy'] = epoch_metrics['accuracy'] / len(epoch_losses)

        if ckpt:
            self.save_checkpoint(epoch)
    
        return epoch_metrics
    
    def update_epoch_metrics(self, metrics: dict, out: dict, y):
        metrics['accuracy'] += (out['probabilities'].argmax(dim=1) == y.argmax(dim=1)).sum().item() / y.size(0)
        return metrics

    def update_train_metrics(self, train_metrics: dict, epoch_metrics: dict, epoch_type: str):
        assert epoch_type in ['train', 'val'], f'Invalid epoch_type: "{epoch_type}". Options are: {["train", "val"]}'
        train_metrics[f'{epoch_type}/accuracy'].append(epoch_metrics['accuracy'])
        train_metrics[f'{epoch_type}/loss'].append(epoch_metrics['loss'])
        return train_metrics

    def train(self,
              class_to_forget=None,
              indices_to_forget=None,
              repair=False,
              ckpt=False):
        train_metrics = self.init_train_metrics()

        # either class_to_forget or indices_to_forget must be provided, or repair must be True
        if class_to_forget is None and indices_to_forget is None and not repair:
            raise ValueError("Either class_to_forget or indices_to_forget must be provided, or repair must be True")

        with tqdm(range(self.n_epochs), disable=self.disable_tqdm) as epoch_pbar:
            for epoch in epoch_pbar:

                train_epoch_metrics = self.train_one_epoch(epoch, 
                                                           class_to_forget=class_to_forget, 
                                                           indices_to_forget=indices_to_forget, 
                                                           repair=repair, 
                                                           ckpt=ckpt)

                # run inference on validation set
                val_metrics = self.eval()
                
                # update train/val dynamics
                self.update_train_metrics(train_metrics, train_epoch_metrics, epoch_type='train')
                self.update_train_metrics(train_metrics, val_metrics, epoch_type='val')
                train_metrics['epoch'].append(int(epoch + 1))

                train_dataset_name = self.train_dataloader.dataset.name
                validation_dataset_name = self.val_dataloader.dataset.name

                # log validation loss and accuracy
                if self.logger:
                    self.logger.log({
                        f"train/loss/{train_dataset_name}": train_epoch_metrics['accuracy'],
                        f"train/accuracy/{train_dataset_name}": train_epoch_metrics['loss'],
                        f"validation/loss/{validation_dataset_name}": val_metrics['loss'],
                        f"validation/accuracy/{validation_dataset_name}": val_metrics['accuracy'],
                        "epoch": epoch
                    })

                # Update progress bar
                pbar_strings = ' '.join([f'{k}={v[-1]:.3f}' for k, v in train_metrics.items()])
                epoch_pbar.set_description(pbar_strings)
                        
                
                if self.do_early_stopping and epoch > 4 and not train_metrics['val/loss'][-1] <= np.mean(train_metrics['val/loss'][:-4:-1]):
                    print("\Ending due to early stopping")
                    return train_metrics
                
        return train_metrics

    def step(self, x: torch.tensor, y: torch.tensor) -> tuple[dict, torch.tensor]:
        """Perform a forward pass, compute loss, and the difference in parameters.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            out: Output tensor
            loss: Loss tensor
            param_diff: Difference in parameters
        """
        self.optimizer.zero_grad()
            
        before_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        
        out = self.model(x)
        # Criterion expects logits if outputs is a dictionary, otherwise it expects the output directly (used for resnet)
        loss = self.model.loss(out, y)
        loss.backward()
        self.optimizer.step()
        
        after_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        param_diff = {name: after_params[name] - before_params[name] for name in before_params}

        return out, loss, param_diff
    
    def eval(self):
        self.model.eval()
        losses = []
        
        epoch_metrics = self.init_epoch_metrics()
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                
                # forward pass and calculate loss
                out = self.model(x)
                loss = self.model.loss(out, y)
                
                # update metrics
                epoch_metrics = self.update_epoch_metrics(epoch_metrics, out, y)
                losses.append(loss.item())
        
        epoch_metrics['loss'] = sum(losses) / len(losses)
        epoch_metrics['accuracy'] = epoch_metrics['accuracy'] / len(losses)
        return epoch_metrics
    
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

            self.model.batch_params.setdefault(epoch, {}).update({batch_idx: param_diff_path})
            return param_diff_path
        else:
            # Save difference in memory
            self.model.batch_params.setdefault(epoch, {}).update({batch_idx: param_diff})

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

    def delete_stored_gradients(self):
        """Delete all cached gradients from memory."""
        if self.cache_gradients:
            raise ValueError("Cannot delete cached gradients if cache_gradients is True. They have to be saved to file.")
        # delete from file
        for epoch in self.batch_params:
            for batch_idx in self.batch_params[epoch]:
                if not self.cache_gradients:
                    os.remove(self.batch_params[epoch][batch_idx])

    def __call__(self,
                 class_to_forget=None,
                 indices_to_forget=None,
                 repair=False,
                 ckpt=False):
        return self.train(class_to_forget=class_to_forget,
                          indices_to_forget=indices_to_forget,
                          repair=repair,
                          ckpt=ckpt)
    
