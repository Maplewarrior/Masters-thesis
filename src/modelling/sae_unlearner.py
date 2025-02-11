import torch
import torch.nn as nn
from src.modelling.neural_network import NeuralNet
from src.modelling.SAE import SAE


class SAEUnlearner(nn.Module):
    def __init__(self, model, sae, layer_num: int) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        self.sae = sae
        self.layer_num = layer_num # which layer to apply the SAE to
    
    def forward(self, x, return_reconstruction: bool = True):
        x_act = self.model.predict(x, start_idx=0, stop_idx=self.layer_num)['logits'] # activations at layer_num
        sae_out = self.sae(x_act) # reconstruction

        if return_reconstruction:
            sae_out['xact'] = x_act
            return sae_out
        # class prediction
        logits = self.model.predict(sae_out['xhat'], start_idx=self.layer_num, stop_idx=None)['logits']
        return {'logits': logits}
    
    def predict_from_reconstruction(self, xhat):
        return self.model.predict(xhat, start_idx=self.layer_num, stop_idx=None)

if __name__ == '__main__':
    import pdb
    nnet = NeuralNet(M = 10, n_classes = 3)
    sae = SAE(d=10, m=40, _lambda=0.5)
    x = torch.randn(size=(5, 10))
    SAEU = SAEUnlearner(nnet, sae, layer_num=4) # after 2nd ReLU
    pdb.set_trace()
    