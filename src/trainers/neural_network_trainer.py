from src.trainers.base_trainer import BaseTrainer
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetworkTrainer(BaseTrainer):
    def __init__(self, 
                 model: nn.Module,
                 train_dataloader, 
                 val_dataloader,
                 logger=None,
                 device="cpu",
                 learning_rate=0.001,
                 weight_decay=0.0,
                 optimizer_name: str = 'Adam',
                 lr_scheduler: str = None,
                 save_checkpoints: bool = False,
                 checkpoint_dir: str = '/work3/204138/MachineUnlearning/weights/CIFAR10/original_model',
                 n_epochs=100,
                 disable_tqdm=False,
                 do_early_stopping=True,
                 **kwargs) -> None:
        
        # Initialize optimizer
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        else:
            raise NotImplementedError(f"The optimizer {optimizer_name} is not supported.")

        # Initialize LR scheduler
        if lr_scheduler is not None:
            if lr_scheduler == 'cosine_annealing':
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
            else:
                raise NotImplementedError(f"The lr scheduler {lr_scheduler} is not supported")

        # Set device
        self.device = device
        model = model.to(device)
        
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            logger=logger,
            lr_scheduler=lr_scheduler,
            save_checkpoints=save_checkpoints,
            checkpoint_dir=checkpoint_dir,
            n_epochs=n_epochs,
            disable_tqdm=disable_tqdm,
            do_early_stopping=do_early_stopping,
            device=device)