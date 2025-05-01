from src.models.base_model import BaseModel
from src.models.neural_network import NeuralNet, NeuralNetRS
from torch import nn


class AmnesiacModelRS(NeuralNetRS):
    def __init__(self, base_model: NeuralNetRS) -> None:
        """
        A wrapper model that adds additional parameters to any base model.
        
        Args:
            base_model (nn.Module): The model to wrap
        """
        # Check if NeuralNet
        if not isinstance(base_model, NeuralNetRS):
            raise ValueError("base_model must be a NeuralNet")
        
        # Unpack params of the neural net and pass to init of NeuralNet
        M = base_model.M
        n_classes = base_model.n_classes
        seed = base_model.seed
        width_factor = base_model.width_factor
        n_layers = base_model.n_layers
        super().__init__(M, n_classes, n_layers, width_factor, seed)
        # self.base_model = base_model

        # Set these empty parameters to be set later
        self.batch_mapping = {}
        self.batch_params = {}
        self.cache_gradients = None
        
        # Initialize parameters dictionary
        self.parameters_dict = {}
        
        # No need to set seed - base model should already have it set
    
"""
NOTE: We cannot make AmnesiacModel compatible with both NeuralNet and NeuralNetRS because of class inheritance...
"""
class AmnesiacModel(NeuralNet):
    def __init__(self, base_model: NeuralNet) -> None:
        """
        A wrapper model that adds additional parameters to any base model.
        
        Args:
            base_model (nn.Module): The model to wrap
        """
        # Check if NeuralNet
        if not isinstance(base_model, NeuralNet): # or isinstance(base_model, NeuralNetRS)):
            raise ValueError("base_model must be a NeuralNet")
        
        # Unpack params of the neural net and pass to init of NeuralNet
        M, n_classes, seed = base_model.M, base_model.n_classes, base_model.seed
        super().__init__(M, n_classes, seed)
        # self.base_model = base_model

        # Set these empty parameters to be set later
        self.batch_mapping = {}
        self.batch_params = {}
        self.cache_gradients = None
        
        # Initialize parameters dictionary
        self.parameters_dict = {}
        
        # No need to set seed - base model should already have it set
    