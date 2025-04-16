import torch
import torch.optim as optim
from src.unlearners.base_unlearner import BaseUnlearner
import pdb

"""
This version of SSD only applies dampening in the final layer of the model
"""

from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening as SSD

class SelectiveSynapticDampening(SSD):
    def __init__(self, 
                 model,
                 alpha: float,
                 _lambda: float,
                 device: str = 'cpu') -> None:
        super().__init__(model, alpha, _lambda, device)
        

    def __call__(self, 
                 full_dataloader,
                 forget_dataloader,
                 FIM_full: dict = None, 
                 FIM_forget: dict = None
                 ):
        
        # calculate FIM matrices if necessary
        if FIM_forget is None:
            FIM_forget = self.calculate_FIM(forget_dataloader)
        
        if FIM_full is None:
            FIM_full = self.calculate_FIM(full_dataloader)
        
        # go through the parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # skip the first layers :D
                if int(name.split('.')[1]) <= 2:
                    continue
                updated_parameter = param.data.clone()
                dampen_mask = FIM_forget[name] > self.alpha * FIM_full[name] # find which paramters to dampen
                # print(f'Original parameter: {param}')
                # print(f'Dampen mask: {dampen_mask}')
                # calculate dampening factors and apply them
                beta = torch.min((self._lambda * FIM_full[name][dampen_mask] / FIM_forget[name][dampen_mask]), torch.tensor(1))
                updated_parameter[dampen_mask] = beta * updated_parameter[dampen_mask]
               
                # update parameter in the model
                param.copy_(updated_parameter)
                # print(f'Updated parameter: {param}')
        