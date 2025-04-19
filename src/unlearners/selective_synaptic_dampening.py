import torch
import torch.optim as optim
from src.unlearners.base_unlearner import BaseUnlearner
import pdb

"""
How will the input space be partitioned based on the neural network.
mad max: affine spline insights into deep learning.

Use for func equivalence: If two NN's create the same partitioning in input space, then they are roughly the same

More convex in deeper layers...
Maybe dampening in last layer the model is 

Evaluation
-----------
When can we compare 
"""

import torch
import torch.optim as optim
from src.unlearners.base_unlearner import BaseUnlearner
import pdb

"""
How will the input space be partitioned based on the neural network.
mad max: affine spline insights into deep learning.

Use for func equivalence: If two NN's create the same partitioning in input space, then they are roughly the same

More convex in deeper layers...
Maybe dampening in last layer the model is 

Evaluation
-----------
When can we compare 
"""

class SelectiveSynapticDampening(BaseUnlearner):
    def __init__(self, 
                 model,
                 alpha: float,
                 _lambda: float,
                 device: str = 'cpu') -> None:
        super().__init__(model, {'alpha': alpha, '_lambda': _lambda})
        self.alpha = alpha
        self._lambda = _lambda
        self.device = device
    
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
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            optimizer.zero_grad()
            # forward pass
            out = self.model(x)
            # calculate loss
            loss = self.model.loss(out, y)
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
    