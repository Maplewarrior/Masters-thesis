import torch.nn as nn
import torch

from src.unlearners.base_unlearner import BaseUnlearner

class AmnesiacUnlearner(BaseUnlearner):
    def __init__(self, model: nn.Module, unlearn_parameters: dict) -> None:
        """'
        Arguments:
            @param model: A model that has been trained on the full training set that should unlearn the forget set.
            @param unlearn_parameters: The hyperparameters of the unlearning algorithm. 
        """
        self.model = model
    
    def forget(self, indices_to_forget=None):
        """Unlearn specific training examples by reverting their parameter updates."""
        batches = {}

        batch_mapping = self.model.batch_mapping
        batch_params = self.model.batch_params

        cache_gradients = self.model.cache_gradients
        
        for epoch in batch_mapping:
            if indices_to_forget is not None:
                batches_to_forget = list(set([batch_mapping[epoch][idx] for idx in indices_to_forget if idx in batch_mapping[epoch]]))
            else:
                batches_to_forget = list(set(batch_mapping[epoch].values()))
            if batches_to_forget:
                batches[epoch] = batches_to_forget
        
        if not batches:
            print("No batches found for the forget set.")
            return
        
        print("Found batches containing forget set, reverting updates...")
        with torch.no_grad():
            for epoch in batches:
                for batch_idx in batches[epoch]:
                    if batch_idx in batch_params[epoch]:
                        if not cache_gradients:
                            grads = torch.load(batch_params[epoch][batch_idx])
                        else:
                            grads = batch_params[epoch][batch_idx]
                        for name, param in self.model.named_parameters():
                            param -= grads[name]
        
        print("Forgetting complete.")


    def __call__(self, **kwargs):
        """
        This function applies the unlearning algorithm on the model.
        """

        indices_to_forget = kwargs.get('indices_to_forget', None)

        self.forget(indices_to_forget)

        # Repair phase