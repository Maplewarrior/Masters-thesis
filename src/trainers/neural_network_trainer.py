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
                 save_checkpoints: bool = False,
                 checkpoint_dir: str = '/work3/204138/MachineUnlearning/weights/CIFAR10/original_model',
                 n_epochs=100,
                 disable_tqdm=False,
                 do_early_stopping=True,
                 **kwargs) -> None:
        
        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Set device
        self.device = device
        model = model.to(device)
        
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            logger=logger,
            save_checkpoints=save_checkpoints,
            checkpoint_dir=checkpoint_dir,
            n_epochs=n_epochs,
            disable_tqdm=disable_tqdm,
            do_early_stopping=do_early_stopping,
            device=device)