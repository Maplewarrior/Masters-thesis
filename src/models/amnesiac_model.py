from src.models.base_model import BaseModel

from torch import nn


class AmnesiacModel(BaseModel):
    def __init__(self, base_model: nn.Module) -> None:
        """
        A wrapper model that adds additional parameters to any base model.
        
        Args:
            base_model (nn.Module): The model to wrap
            parameters (dict, optional): Additional parameters to store with the model
            seed (int, optional): Random seed for reproducibility
        """
        super().__init__()
        self.base_model = base_model

        # Set these empty parameters to be set later
        self.batch_mapping = {}
        self.batch_params = {}
        self.cache_gradients = None
    
    def forward(self, x, start_idx: int = 0, stop_idx: int = None):
        # Pass the input through the base model
        return self.base_model(x, start_idx, stop_idx)
    
    def get_parameter(self, key, default=None):
        """Get a parameter by key with an optional default value"""
        return self.parameters_dict.get(key, default)
    
    def set_parameter(self, key, value):
        """Set or update a parameter"""
        self.parameters_dict[key] = value
        
    def get_all_parameters(self):
        """Get all stored parameters"""
        return self.parameters_dict