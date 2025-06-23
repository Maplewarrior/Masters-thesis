import torch
import pdb
from src.unlearners.base_unlearner import BaseUnlearner

class SAEUnlearner(BaseUnlearner):
    def __init__(self, model, alpha: float = 0.9, device : str = 'cpu') -> None:
        super().__init__(model, {'alpha': alpha})
        self.model = model
        self.alpha = alpha # the higher alpha, the lower the dampening
        self.device = device

    def get_Z_matrix(self, dataloader):
        Z_sum = torch.zeros(self.model.sae.m, device=self.device) # [feature dim x dictionary size]
        Z_count = torch.zeros(self.model.sae.m, device=self.device) # [dictionary size]

        for batch in dataloader:
            x = batch[0].to(self.device)
            z = self.model.inference(x)['z']
            Z_sum += z.sum(dim=0)
            idxs, counts = torch.unique(torch.where(z > 0)[1], return_counts=True)
            Z_count[idxs] += counts
        
        active_idxs = torch.where(Z_count > 0)[0]
        active_counts = Z_count[active_idxs]
        return Z_sum, active_idxs, active_counts
    
    def calculate_dampening_factors(self, retain_loader, forget_loader):
        Z_retain, retain_feature_idxs, retain_feature_counts = self.get_Z_matrix(retain_loader)
        Z_forget, forget_feature_idxs, forget_feature_counts = self.get_Z_matrix(forget_loader)

        # normalized_retain_counts = retain_feature_counts / Z_retain.size(0)
        # normalized_forget_counts = forget_feature_counts / Z_forget.size(0)

        dampening_factors = []
        
        mean_retain_activations = (Z_retain[retain_feature_idxs] / retain_feature_counts).to(self.device)
        mean_forget_activations = (Z_forget[forget_feature_idxs] / forget_feature_counts).to(self.device)
        
        # assert alpha >= 1, 'alpha must be greater than 1'
        for i, forget_idx in enumerate(forget_feature_idxs):
            retain_idx = torch.where(forget_idx == retain_feature_idxs)[0]
            if len(retain_idx): # feature is also used in retain
                # dampening_factors.append(torch.min(alpha * normalized_forget_counts[i] / normalized_retain_counts[retain_idx], torch.tensor(1.)))
                dampening_factors.append(torch.min(self.alpha * mean_retain_activations[retain_idx] / mean_forget_activations[i], torch.tensor(1., device=self.device)))
            else:
                dampening_factors.append(torch.tensor([0.], device=self.device))
                
        dampening_factors = torch.cat(dampening_factors)

        return dampening_factors, forget_feature_idxs
        
    def unlearn_features(self, retain_loader, forget_loader):
        """
        A method that dampens the dictionary matrix features based on how active they are in the forget vs retain set.
        """
        
        Z_retain = self.get_Z_matrix(dataloader=retain_loader)
        Z_forget = self.get_Z_matrix(dataloader=forget_loader)

        dampening_factors, forget_feature_idxs = self.calculate_dampening_factors(retain_loader, forget_loader)

        with torch.no_grad():
            W_dec = self.model.sae.decoder.weight.data.clone()
            W_dec[:, forget_feature_idxs] = W_dec[:, forget_feature_idxs] * dampening_factors
            self.model.sae.decoder.weight.copy_(W_dec)
        
    # def unlearn_weights(self, retain_loader, forget_loader, val_loader, n_epochs: int, lr: float,):
          # # TODO: Make this compatible with current setup  
    #     Z_retain = self.get_Z_matrix(dataloader=retain_loader)
    #     Z_forget = self.get_Z_matrix(dataloader=forget_loader)
    #     alpha = 0.9 # the higher alpha, the lower the dampening
    #     _lambda = 2.
    #     dampening_factors, forget_feature_idxs = self.calculate_dampening_factors(Z_retain, Z_forget, alpha)
        
    #     penalty_terms = torch.zeros_like(Z_retain[0,:])
    #     penalty_terms[forget_feature_idxs] = 1 - dampening_factors
    #     self.model.train()
    #     trainer = Trainer(model=self, 
    #                       train_dataloader=retain_loader, 
    #                       val_dataloader=val_loader, 
    #                       n_epochs=n_epochs, 
    #                       lr=lr,
    #                       loss = CustomSAECriterion(penalty_terms, 
    #                                                 _lambda=_lambda))
    
    def __call__(self, retain_dataloader, forget_dataloader):
        self.unlearn_features(retain_dataloader, forget_dataloader)
    
    