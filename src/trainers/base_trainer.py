import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import pdb

class BaseTrainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer,
                 train_dataloader,
                 val_dataloader,
                 logger,
                 disable_tqdm: bool,
                 do_early_stopping: bool,
                 lr_scheduler = None,
                 save_checkpoints: bool = False,
                 checkpoint_dir: str = None,
                 n_epochs: int = 20,
                 device: str = 'cpu') -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.disable_tqdm = disable_tqdm
        self.do_early_stopping = do_early_stopping
        self.lr_scheduler = lr_scheduler
        self.n_epochs = n_epochs
        self.device = device
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = checkpoint_dir

    def init_epoch_metrics(self):
        return {'accuracy': 0, 'loss': 0}

    def init_train_metrics(self):
        return {'train/accuracy': [], 'train/loss': [],
                'val/accuracy': [], 'val/loss': [], 'epoch': []}

    def train_one_epoch(self) -> tuple[list[float], float]:
        self.model.train()
        N_steps = len(self.train_dataloader)
        epoch_losses = []
        epoch_metrics = self.init_epoch_metrics()
        for batch in self.train_dataloader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            # perform forward and backward pass
            out, loss = self.step(x, y)
            # Update/store metrics
            epoch_metrics = self.update_epoch_metrics(epoch_metrics, out, y, N_steps)
            epoch_losses.append(loss.item())

        epoch_metrics['loss'] = sum(epoch_losses) / N_steps
        # epoch_metrics['accuracy'] = epoch_metrics['accuracy']
        return epoch_metrics

    def update_epoch_metrics(self, metrics: dict, out: dict, y, N: int):
        metrics['accuracy'] += (out['probabilities'].argmax(dim=1) == y.argmax(dim=1)).sum().item() / (y.size(0) * N)
        return metrics

    def update_train_metrics(self, train_metrics: dict, epoch_metrics: dict, epoch_type: str):
        assert epoch_type in ['train', 'val'], f'Invalid epoch_type: "{epoch_type}". Options are: {["train", "val"]}'
        train_metrics[f'{epoch_type}/accuracy'].append(epoch_metrics['accuracy'])
        train_metrics[f'{epoch_type}/loss'].append(epoch_metrics['loss'])
        return train_metrics

    def train(self):
        train_metrics = self.init_train_metrics()

        with tqdm(range(self.n_epochs), disable=self.disable_tqdm) as epoch_pbar:
            for epoch in epoch_pbar:
                # train one epoch
                train_epoch_metrics = self.train_one_epoch()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # run inference on validation set
                val_metrics = self.eval()

                # update train/val dynamics
                train_metrics = self.update_train_metrics(train_metrics, train_epoch_metrics, epoch_type='train')
                train_metrics = self.update_train_metrics(train_metrics, val_metrics, epoch_type='val')
                train_metrics['epoch'].append(int(epoch + 1))
                train_dataset_name = self.train_dataloader.dataset.name
                validation_dataset_name = self.val_dataloader.dataset.name

                # log validation loss and accuracy
                if self.logger:
                    self.logger.log({
                        f"train/loss/{train_dataset_name}": train_epoch_metrics['loss'],
                        f"train/accuracy/{train_dataset_name}": train_epoch_metrics['accuracy'],
                        f"validation/loss/{validation_dataset_name}": val_metrics['loss'],
                        f"validation/accuracy/{validation_dataset_name}": val_metrics['accuracy'],
                        "epoch": epoch + 1
                    })

                # Update progress bar
                pbar_strings = ' '.join([f'{k}={v[-1]:.3f}' for k, v in train_metrics.items()])
                epoch_pbar.set_description(pbar_strings)
                # import pdb; pdb.set_trace()
                if self.save_checkpoints and epoch > 2 and train_metrics['val/accuracy'][-1] == np.max(train_metrics['val/accuracy']):
                    self.save_state_dict(self.checkpoint_dir,
                                         model_name=self.checkpoint_dir.split('/')[-1],
                                         val_acc=train_metrics['val/accuracy'][-1],
                                         epoch=epoch + 1
                                         )

                if self.do_early_stopping and epoch > 4 and not train_metrics['val/loss'][-1] <= np.mean(train_metrics['val/loss'][:-4:-1]):
                    print("\Ending due to early stopping")
                    return train_metrics
            
        if self.save_checkpoints: # load the best model checkpoint
            best_idx = np.argmax(train_metrics['val/accuracy'])
            sd_opt = self.load_state_dict(self.checkpoint_dir,
                                 model_name=self.checkpoint_dir.split('/')[-1],
                                 val_acc=train_metrics['val/accuracy'][best_idx],
                                 epoch=train_metrics['epoch'][best_idx]
                                 )
            self.model.load_state_dict(sd_opt)

        return train_metrics

    def step(self, x: torch.tensor, y: torch.tensor) -> tuple[dict, torch.tensor]:
        self.optimizer.zero_grad()
        out = self.model(x)
        # Compute loss and update
        loss = self.model.loss(out, y)
        loss.backward()
        self.optimizer.step()
        return out, loss

    def eval_step(self, x):
        return self.model(x)

    def eval(self):
        self.model.eval()
        losses = []
        N_steps = len(self.val_dataloader)
        epoch_metrics = self.init_epoch_metrics()

        with torch.no_grad():
            for batch in self.val_dataloader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)

                # forward pass and calculate loss
                # out = self.model(x)
                out = self.eval_step(x)
                loss = self.model.loss(out, y)

                # update metrics
                epoch_metrics = self.update_epoch_metrics(epoch_metrics, out, y, N_steps)
                losses.append(loss.item())

        epoch_metrics['loss'] = sum(losses) / len(losses)
        return epoch_metrics

    def save_state_dict(self, save_dir: str, model_name: str, val_acc: float, epoch: int):
        torch.save(self.model.state_dict(), f'{save_dir}/{model_name}_val-acc={val_acc}_epoch={epoch}.pt')
    
    def load_state_dict(self, load_dir: str, model_name: str, val_acc: float, epoch: int):
         opt_checkpoint_path = f'{load_dir}/{model_name}_val-acc={val_acc}_epoch={epoch}.pt'
         self.remove_suboptimal_checkpoints(load_dir, opt_checkpoint_path) # free up space
         return torch.load(f'{load_dir}/{model_name}_val-acc={val_acc}_epoch={epoch}.pt')
    
    def remove_suboptimal_checkpoints(self, dir_path: str, optimal_checkpoint_path: str):
        filenames = os.listdir(dir_path)
        optimal_checkpoint_file = optimal_checkpoint_path.split('/')[-1]
        remove_files = [e for e in filenames if e != optimal_checkpoint_file]
        for fn in remove_files:
            os.remove(f'{dir_path}/{fn}')

    def __call__(self):
        return self.train()

