import torch
import torch.nn as nn
import pdb
from src.models.base_model import BaseModel

class NeuralNetWithSAE(BaseModel):
    def __init__(self, neural_net: nn.Module, sae: nn.Module, layer_num: int) -> None:
        super().__init__()
        self.neural_net = neural_net
        self.sae = sae
        self.layer_num = layer_num # where to apply the SAE on the neural network's forward pass
        self.freeze_nn_parameters()
    
    def freeze_nn_parameters(self):
        self.neural_net.eval()
        for param in self.neural_net.parameters():
            param.requires_grad = False


    def forward(self, x, start_idx = None, stop_idx = None):
        """
        Computes a forward pass through the neural network up to layer num.
        This is followed by the SAE reconstructing the output activation at layer num.
        The forward pass of the neural is continued using the SAE reconstruction.
        NOTE: start_idx and stop_idx args are provided for backwards compatibility.
        """
        # compute forward pass up to layer_num
        x_act = self.neural_net.inference(x, start_idx=0, stop_idx=self.layer_num)['logits']
        original_shape = x_act.size()
        if x_act.ndim > 2:
            x_act = x_act.view(-1, self.sae.d)  # collapse batch_size and sequence_length dimensions
        
        # compute sae reconstruction
        sae_out = self.sae(x_act.view(-1, self.sae.d))
        sae_out['xact'] = x_act

        # compute remaining forward pass using reconstruction
        if len(original_shape) > 2:
            nn_out = self.neural_net.inference(sae_out['xhat'].view(original_shape), start_idx=self.layer_num, stop_idx=None)
        else:
            nn_out = self.neural_net.inference(sae_out['xhat'], start_idx=self.layer_num, stop_idx=None)

        nn_out.update(sae_out) # update output dictionary
        return nn_out
    
    def predict_from_reconstruction(self, xhat):
        return self.neural_net.inference(xhat, start_idx=self.layer_num, stop_idx=None)
    
    def loss(self, out: dict, *args):
        return self.sae.loss(x=out['xact'], forward_out=out)
        