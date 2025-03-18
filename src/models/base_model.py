import torch
import torch.nn as nn
import numpy as np
import random


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.seed = None

    def set_seed(self, seed: int = None):
        """
        Set random seed for reproducibility across all random operations.
        
        Args:
            seed (int, optional): Random seed to use. If None, no seed is set.
        """
        if seed is not None:
            self.seed = seed
            # Set seeds for all random number generators
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # For multi-GPU
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Reset parameters for all modules that support it
            self.reset_parameters()
    
    def reset_parameters(self):
        """
        Reset parameters for all modules that have a reset_parameters method.
        This provides deterministic initialization when used with set_seed.
        """
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            else: 
                raise NotImplementedError(f"Module {module} does not have a reset_parameters method.")


    def forward(self, x: torch.tensor, 
                start_idx: int = 0, 
                stop_idx: int | None = None) -> dict:
        raise NotImplementedError()
    
    def inference(self, x: torch.tensor, start_idx: int = 0, stop_idx: int | None = None) -> dict:
        self.eval()
        with torch.no_grad():
            return self(x, start_idx, stop_idx)
        
    def loss(self, out: dict, y: torch.tensor):
        """
        @param out: A dictionary containing the output of a forward pass.
                    - out should have a logits key where the value is a tensor of shape batch_size x n_classes.
        @param y: A tensor of shape batch_size x n_classes containing one-hot encoded class labels.
        """
        return nn.functional.cross_entropy(input=out['logits'], target=y)
    
