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
                 n_epochs=100,
                 disable_tqdm=False,
                 do_early_stopping=True,
                 **kwargs) -> None:
        
        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Set device
        self.device = device
        model = model.to(device)
        
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            logger=logger,
            n_epochs=n_epochs,
            disable_tqdm=disable_tqdm,
            do_early_stopping=do_early_stopping,
            device=device
        )