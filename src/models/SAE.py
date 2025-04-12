import torch.nn as nn
import torch
from src.models.base_model import BaseModel
import pdb

class SAE(BaseModel):
    def __init__(self, d: int, m: int, _lambda: float):
        super().__init__()
        self.d = d # dimensionality of the model activations
        self.m = m # dimensionality of the dictionary space
        
        self.decoder = nn.Linear(self.m, self.d)
        self.encoder = nn.Sequential(nn.Linear(self.d, self.m),
                                     nn.ReLU())
        
        self.mse = nn.MSELoss(reduction='none')
        self._lambda = _lambda
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        # see: https://transformer-circuits.pub/2024/april-update/index.html on initialization
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'bias' in name:
                    # set all bias terms to zero
                    param.copy_(torch.zeros_like(param.detach()))
                elif name == 'decoder.weight':
                    Wdec = torch.randn_like(param)
                    Wdec = Wdec / Wdec.norm(p=2, dim=0)
                    param.copy_(Wdec)

                elif name == 'encoder.0.weight':
                    param.copy_(self.state_dict()['decoder.weight'].T)
        
        assert torch.isclose(self.state_dict()['encoder.0.weight'].norm(p=2, dim=1), torch.ones(self.m)).all(), 'Encoder weights not initialized to unit norm!'
        assert torch.isclose(self.state_dict()['encoder.0.bias'], torch.zeros(self.m)).all(), 'Encoder bias not initalized to zeros'
        assert torch.isclose(self.state_dict()['decoder.bias'], torch.zeros(self.d)).all(), 'Decoder bias not initialized to zeros'


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
        W_dec = self.decoder.weight
        l2_norm_dictionary = W_dec.norm(p=2, dim=0) # torch.linalg.norm(self.decoder.weight, dim=0, ord=2)
        
        # regularization_term = z @ l2_norm_dictionary # no need to abs z because of ReLU
        # MHA update:
        regularization_term = (z @ l2_norm_dictionary) / x.size(0) # no need to abs z because of ReLU
        
        # MHA update 2: Favor near orthogonal W values:
        # regularization_term = (z @ l2_norm_dictionary) / x.size(0) + torch.sum(torch.abs(W_dec @ W_dec.T - torch.diag(torch.diag(W_dec @ W_dec.T))))
        
        # MHA update 3: Actual orthogonality constraint:
        # mprod = W_dec.T @ W_dec
        # regularization_term = ((z @ l2_norm_dictionary) + torch.sum(torch.abs(mprod - torch.diag(torch.diag(mprod))))) / x.size(0)

        loss = reconstruction_term + self._lambda * regularization_term
        return loss.mean()

class Crosscoder(nn.Module):
    def __init__(self, d: int, m: int) -> None:
        super().__init__()
        self.d = d # dimensionality of the model activations
        self.m = m # dimensionality of the dictionary space

    def forward(self):
        raise NotImplementedError()