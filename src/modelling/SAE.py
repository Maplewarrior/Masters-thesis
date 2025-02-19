import torch.nn as nn
import torch

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
    
    def loss(self, x, forward_out: dict):
        z = forward_out['z']
        x_hat = forward_out['xhat']
        # calculate reconstruction_term 
        reconstruction_term = self.mse(x, x_hat).mean(dim=-1)
        # calculate regularization term
        l2_norm_dictionary = torch.linalg.norm(self.decoder.weight, dim=0, ord=2)
        regularization_term = z @ l2_norm_dictionary # no need to abs z because of ReLU
        loss = reconstruction_term + self._lambda * regularization_term
        return loss.mean()
    

class Crosscoder(nn.Module):
    def __init__(self, d: int, m: int) -> None:
        super().__init__()
        self.d = d # dimensionality of the model activations
        self.m = m # dimensionality of the dictionary space

    def forward(self):
        raise NotImplementedError()