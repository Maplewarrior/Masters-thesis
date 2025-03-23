import torch
import torch.optim as optim
from src.unlearners.base_unlearner import BaseUnlearner
import pdb


"""
This version of SSD only applies dampening in the final layer of the model
"""

class SelectiveSynapticDampening(BaseUnlearner):
    def __init__(self, 
                 model, 
                 criterion,
                 alpha: float,
                 _lambda: float) -> None:
        super().__init__(model, {'alpha': alpha, '_lambda': _lambda})
        self.criterion = criterion
        self.alpha = alpha
        self._lambda = _lambda
        
    
    def calculate_FIM(self, dataloader) -> dict:
        """
        A function that calculates the diagonal of the Fisher Information of a model for a specific dataset.

        @param dataloader: An iterable dataloader for which the FIM diagonal should be calculated.
        returns: A dictionary where keys are the names of parameters and values are the FIM of that parameter
        """
        FIM = {k: 0 for k in self.model.state_dict().keys()}
        # define optimizer to allow gradient computation
        optimizer = optim.SGD(self.model.parameters())
        # turn of dropout if applicable
        self.model.eval()
        
        for i, batch in enumerate(dataloader):
            x, y = batch[0], batch[1]
            optimizer.zero_grad()
            # forward pass
            logits = self.model(x)['logits']
            # calculate loss
            loss = self.criterion(logits, y)
            # calculate gradients
            loss.backward()
            for i, (name, param) in enumerate(self.model.named_parameters()):
                FIM[name] += param.grad.data.clone().pow(2) #optimizer.param_groups[0]['params'][i].grad.pow(2)  
        # account for batched inference
        for k in FIM.keys():
            FIM[k] = FIM[k] / len(dataloader)
        
        return FIM
    
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
        