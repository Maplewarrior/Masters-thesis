import torch.nn as nn
import torch
import pdb
class SAE(nn.Module):
    def __init__(self, d: int, m: int, _lambda: float):
        super().__init__()
        self.d = d # dimensionality of the model activations
        self.m = m # dimensionality of the dictionary space
        self.encoder = nn.Sequential(nn.Linear(self.d, self.m),
                                     nn.ReLU())
        
        self.decoder = nn.Linear(self.m, self.d)
        self.mse = nn.MSELoss(reduction='none')
        self._lambda = _lambda

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return {'xhat': xhat, 
                'z': z}

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)
    
    def loss(self, x, forward_out: dict):
        # x er 1 x 100
        # W er 100 x m
        z = forward_out['z']
        x_hat = forward_out['xhat']
        # calculate reconstruction_term 
        reconstruction_term = self.mse(x, x_hat).mean(dim=-1)
        # calculate regularization term
        W_dec = self.decoder.weight
        l2_norm_dictionary = W_dec.norm(p=2, dim=0) # torch.linalg.norm(self.decoder.weight, dim=0, ord=2)
        # regularization_term = z @ l2_norm_dictionary # no need to abs z because of ReLU
        # MHA update:
        regularization_term = (z @ l2_norm_dictionary) / x.size(0) # no need to abs z because of ReLU
        
        # MHA update 2: Favor near orthogonal W values:
        # regularization_term = (z @ l2_norm_dictionary) / x.size(0) + torch.sum(torch.abs(W_dec @ W_dec.T - torch.diag(torch.diag(W_dec @ W_dec.T))))
        
        # MHA update 3: Actual orthogonality constraint:
        # mprod = W_dec.T @ W_dec # n_features x n_features
        # regularization_term = ((z @ l2_norm_dictionary) + torch.sum(torch.abs(mprod - torch.diag(torch.diag(mprod))))) / x.size(0)
        
        # MHA update 4: Orthogonal + diversity constraint
        # sparsity_constraint = z @ l2_norm_dictionary #/ x.size(0)
        # orthogonality_constraint = torch.sum(torch.abs(mprod - torch.diag(torch.diag(mprod))))
        # diversity_constraint = torch.log(torch.sum(torch.abs(mprod / torch.outer(l2_norm_dictionary, l2_norm_dictionary) - torch.eye(W_dec.size(1)))))
        # regularization_term = (sparsity_constraint + orthogonality_constraint) / x.size(0) + diversity_constraint
        
        loss = reconstruction_term + self._lambda * regularization_term
        return loss.mean()
class Crosscoder(nn.Module):
    def __init__(self, d: int, m: int) -> None:
        super().__init__()
        self.d = d # dimensionality of the model activations
        self.m = m # dimensionality of the dictionary space

    def forward(self):
        raise NotImplementedError()