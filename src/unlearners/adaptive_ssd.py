import torch
import torch.optim as optim
from src.unlearners.base_unlearner import BaseUnlearner
import pdb
from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening as SSD

class AdaptiveSSD(SSD):
    def __init__(self, 
                 model,
                 device: str = 'cpu') -> None:
        super().__init__(model, alpha=None, _lambda=None, device=device)
    
    
    def calculate_FIM_ratio(self, FIM_full, FIM_forget):
        FIM_ratio_dist = []
        for key in self.model.state_dict().keys():
            """
            NOTE: 
            - In their paper they use the formula: FIM_full / FIM_forget
            - In their code, they use the formula: FIM_forget / FIM_full
            The latter makes more sense, so we evaluate against that.
            """
            # ratio = FIM_full[key] / FIM_forget[key] #torch.nan_to_num(FIM_full[key] / FIM_forget[key], 0.0)
            ratio = FIM_forget[key] / FIM_full[key]
            FIM_ratio = ratio[~torch.isnan(ratio) & ~torch.isinf(ratio)]
            FIM_ratio_dist.append(FIM_ratio)
            
        return torch.cat(FIM_ratio_dist)

    def calculate_alpha(self, FIM_full, FIM_forget, p):
        # 
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
        
        p = (100 - torch.log(torch.tensor(1 + len(forget_dataloader.dataset) / len(full_dataloader.dataset) * 100))).to(self.device)
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
        