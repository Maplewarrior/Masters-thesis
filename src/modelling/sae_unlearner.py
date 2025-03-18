import torch
import torch.nn as nn
import pdb
from src.modelling.neural_network import NeuralNet
from src.modelling.SAE import SAE
from src.modelling.trainer import Trainer

class CustomSAECriterion(nn.Module):
    def __init__(self, penalty_terms, _lambda) -> None:
        super().__init__()
        self.CE = nn.CrossEntropyLoss()
        self.penalty_terms = penalty_terms
        self._lambda = _lambda
    
    def forward(self, logits, label, z):
        fit_term = self.CE(logits, label)
        reg_term = (self._lambda * z @ self.penalty_terms).mean()
        return fit_term + reg_term

class SAEUnlearner(nn.Module):
    def __init__(self, model, sae, layer_num: int) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        self.sae = sae
        self.layer_num = layer_num # which layer to apply the SAE to
    
    def forward(self, x, return_reconstruction: bool = True, train_sae: bool = True):
        if train_sae:
            x_act = self.model.inference(x, start_idx=0, stop_idx=self.layer_num)['logits'] # activations at layer_num
            sae_out = self.sae(x_act) # reconstruction

            if return_reconstruction:
                sae_out['xact'] = x_act
                return sae_out
            # class prediction
            logits = self.model.inference(sae_out['xhat'], start_idx=self.layer_num, stop_idx=None)['logits']
            return {'logits': logits}
        else:
            x_act = self.model(x, start_idx=0, stop_idx=self.layer_num)['logits']
            sae_out = self.sae.inference(x_act)
            out = self.model(sae_out['xhat'], start_idx=self.layer_num, stop_idx=None)
            out['z'] = sae_out['z']
            return out
    
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
            return self(x, return_reconstruction=False, train_sae=False)
    
    def calculate_dampening_factors(self, Z_retain, Z_forget, alpha: float):
        retain_feature_idxs, retain_feature_counts = torch.unique(torch.where(Z_retain > 0)[1], return_counts=True)
        forget_feature_idxs, forget_feature_counts = torch.unique(torch.where(Z_forget > 0)[1], return_counts=True)

        normalized_retain_counts = retain_feature_counts / Z_retain.size(0)
        normalized_forget_counts = forget_feature_counts / Z_forget.size(0)
    
        mean_retain_activations = Z_retain.sum(dim=0)[retain_feature_idxs] / retain_feature_counts
        mean_forget_activations = Z_forget.sum(dim=0)[forget_feature_idxs] / forget_feature_counts

        dampening_factors = []
        # assert alpha >= 1, 'alpha must be greater than 1'

        for i, forget_idx in enumerate(forget_feature_idxs):
            retain_idx = torch.where(forget_idx == retain_feature_idxs)[0]
            if len(retain_idx): # feature is also used in retain
                # dampening_factors.append(torch.min(alpha * normalized_forget_counts[i] / normalized_retain_counts[retain_idx], torch.tensor(1.)))
                dampening_factors.append(torch.min(alpha * mean_retain_activations[retain_idx] / mean_forget_activations[i], torch.tensor(1.)))
                # dampening_factors.append(torch.min(alpha * mean_forget_activations[i] / mean_retain_activations[retain_idx], torch.tensor(1.)))
            else:
                dampening_factors.append(torch.tensor([0.]))

        dampening_factors = torch.cat(dampening_factors)
        return dampening_factors, forget_feature_idxs
    
    def unlearn_features(self, retain_loader, forget_loader):
        Z_retain = self.get_Z_matrix(dataloader=retain_loader)
        Z_forget = self.get_Z_matrix(dataloader=forget_loader)

        alpha = 0.75 # 0.9 # the higher alpha, the lower the dampening
        dampening_factors, forget_feature_idxs = self.calculate_dampening_factors(Z_retain, Z_forget, alpha)

        with torch.no_grad():
            W_dec = self.sae.decoder.weight.data.clone()
            W_dec[:, forget_feature_idxs] = W_dec[:, forget_feature_idxs] * dampening_factors
            self.sae.decoder.weight.copy_(W_dec)
    
    def unlearn_weights(self, retain_loader, forget_loader, val_loader, n_epochs: int, lr: float,):
        Z_retain = self.get_Z_matrix(dataloader=retain_loader)
        Z_forget = self.get_Z_matrix(dataloader=forget_loader)
        alpha = 0.9 # the higher alpha, the lower the dampening
        _lambda = 2.
        dampening_factors, forget_feature_idxs = self.calculate_dampening_factors(Z_retain, Z_forget, alpha)
        
        penalty_terms = torch.zeros_like(Z_retain[0,:])
        penalty_terms[forget_feature_idxs] = 1 - dampening_factors
        self.model.train()
        trainer = Trainer(model=self, 
                          train_dataloader=retain_loader, 
                          val_dataloader=val_loader, 
                          n_epochs=n_epochs, 
                          lr=lr,
                          loss = CustomSAECriterion(penalty_terms, 
                                                    _lambda=_lambda))
        
        trainer.fine_tune_w_sae()
    