import torch
import torch.nn as nn
import pdb
# from src.models.neural_network import NeuralNet
# from src.models.SAE import SAE
from src.unlearners.base_unlearner import BaseUnlearner


class SAEUnlearner(BaseUnlearner):
    def __init__(self, model, sae, layer_num: int, alpha: float = 0.9) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        self.sae = sae
        self.layer_num = layer_num # which layer to apply the SAE to
        self.alpha = alpha # the higher alpha, the lower the dampening
    

    def forward(self, x, return_reconstruction: bool = True):
        x_act = self.model.inference(x, start_idx=0, stop_idx=self.layer_num)['logits'] # activations at layer_num
        sae_out = self.sae(x_act) # reconstruction

        if return_reconstruction:
            sae_out['xact'] = x_act
            return sae_out
        # class prediction
        logits = self.model.inference(sae_out['xhat'], start_idx=self.layer_num, stop_idx=None)['logits']
        return {'logits': logits}
    
    def predict_from_reconstruction(self, xhat):
        return self.model.inference(xhat, start_idx=self.layer_num, stop_idx=None)

    def get_Z_matrix(self, dataloader):
        Z = []
        self.sae.eval()
        with torch.no_grad():
            for ipt, _ in dataloader:
                z = self(ipt, return_reconstruction=True)['z']
                Z.append(z)
        return torch.cat(Z)
    
    def inference(self, x):
        self.sae.eval()
        with torch.no_grad():
            return self(x, return_reconstruction=False)
    
    def calculate_dampening_factors(self, Z_retain, Z_forget):
        retain_feature_idxs, retain_feature_counts = torch.unique(torch.where(Z_retain > 0)[1], return_counts=True)
        forget_feature_idxs, forget_feature_counts = torch.unique(torch.where(Z_forget > 0)[1], return_counts=True)

        # normalized_retain_counts = retain_feature_counts / Z_retain.size(0)
        # normalized_forget_counts = forget_feature_counts / Z_forget.size(0)

        dampening_factors = []
        
        mean_retain_activations = Z_retain.sum(dim=0)[retain_feature_idxs] / retain_feature_counts
        mean_forget_activations = Z_forget.sum(dim=0)[forget_feature_idxs] / forget_feature_counts
        
        # assert alpha >= 1, 'alpha must be greater than 1'
        for i, forget_idx in enumerate(forget_feature_idxs):
            retain_idx = torch.where(forget_idx == retain_feature_idxs)[0]
            if len(retain_idx): # feature is also used in retain
                # dampening_factors.append(torch.min(alpha * normalized_forget_counts[i] / normalized_retain_counts[retain_idx], torch.tensor(1.)))
                dampening_factors.append(torch.min(self.alpha * mean_retain_activations[retain_idx] / mean_forget_activations[i], torch.tensor(1.)))
            else:
                dampening_factors.append(torch.tensor([0.]))
                
        dampening_factors = torch.cat(dampening_factors)

        return dampening_factors, forget_feature_idxs

    
    def unlearn(self, retain_loader, forget_loader):
        Z_retain = self.get_Z_matrix(dataloader=retain_loader)
        Z_forget = self.get_Z_matrix(dataloader=forget_loader)
        dampening_factors, forget_feature_idxs = self.calculate_dampening_factors(Z_retain, Z_forget, alpha)

        with torch.no_grad():
            W_dec = self.sae.decoder.weight.data.clone()
            W_dec[:, forget_feature_idxs] = W_dec[:, forget_feature_idxs] * dampening_factors
            self.sae.decoder.weight.copy_(W_dec)
    
    def __call__(self, retain_dataloader, forget_dataloader):
        self.unlearn(retain_dataloader, forget_dataloader)
    
    