import torch.nn as nn

class BaseUnlearner:
    def __init__(self, model: nn.Module, unlearn_parameters: dict) -> None:
        """
        Arguments:
            @param model: A model that has been trained on the full training set that should unlearn the forget set.
            @param unlearn_parameters: The hyperparameters of the unlearning algorithm. 
        """
        raise NotImplementedError()
    
    def __call__(self, **kwargs):
        """
        This function applies the unlearning algorithm on the model.
        """
        raise NotImplementedError()