import torch
import torch.optim as optim
from src.unlearners.base_unlearner import BaseUnlearner
import pdb

class AdaptiveSSD(BaseUnlearner):
    def __init__(self, 
                 model) -> None:
        super().__init__(model, {})
        
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
            out = self.model(x)
            # calculate loss
            loss = self.model.loss(out, y)
            # calculate gradients
            loss.backward()
            # calculate FIM diagonal
            for i, (name, param) in enumerate(self.model.named_parameters()):
                FIM[name] += param.grad.data.clone().pow(2) #optimizer.param_groups[0]['params'][i].grad.pow(2)  
        # account for batched inference
        for k in FIM.keys():
            FIM[k] = FIM[k] / len(dataloader)
        
        return FIM
    
    def calculate_FIM_ratio(self, FIM_full, FIM_forget):
        FIM_ratio_dist = []
        for key in self.model.state_dict().keys():
            ratio = FIM_full[key] / FIM_forget[key] #torch.nan_to_num(FIM_full[key] / FIM_forget[key], 0.0)
            FIM_ratio = ratio[~torch.isnan(ratio) & ~torch.isinf(ratio)]
            FIM_ratio_dist.append(FIM_ratio)
            
        return torch.cat(FIM_ratio_dist)

    def calculate_alpha(self, FIM_full, FIM_forget, p):
        FIM_ratio_dist = self.calculate_FIM_ratio(FIM_full, FIM_forget)
        alpha = FIM_ratio_dist.quantile(p/100)
        return alpha

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
        
        p = 100 - torch.log(torch.tensor(1 + len(forget_dataloader.dataset) / len(full_dataloader.dataset) * 100))
        alpha = self.calculate_alpha(FIM_full, FIM_forget, p)

        # go through the parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                updated_parameter = param.data.clone()
                dampen_mask = FIM_forget[name] > alpha * FIM_full[name] # find which paramters to dampen
                # print(f'Original parameter: {param}')
                # print(f'Dampen mask: {dampen_mask}')
                # calculate dampening factors and apply them
                beta = torch.min((FIM_full[name][dampen_mask] / FIM_forget[name][dampen_mask]), torch.tensor(1))
                updated_parameter[dampen_mask] = beta * updated_parameter[dampen_mask]
                # update parameter in the model
                param.copy_(updated_parameter)
                # print(f'Updated parameter: {param}')
        
        return {'alpha': alpha, '_lambda': 1}
        