import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTModel
from src.models.base_model import BaseModel
import pdb

def build_backbone():
    model = ViTModel.from_pretrained("WinKawaks/vit-tiny-patch16-224", add_pooling_layer=False)
    return model

class ViT(BaseModel):
    def __init__(self, 
                 d_patch: int,
                 d_hidden: int,
                 d_ff: int,
                 d_k: int,
                 n_layers: int,
                 n_heads: int,
                 n_classes: int,
                 n_patches: int,
                 dropout_prob: float,
                 tau: float,
                 pooling_type: str
                 ) -> None:
        super().__init__()
        self.d_patch = d_patch
        self.d_hidden = d_hidden
        self.d_ff = d_ff
        self.d_k = d_k
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.n_patches = n_patches
        self.dropout_prob = dropout_prob
        self.tau = tau
        self.pooling_type = pooling_type
        self.backbone = build_backbone()
        self.linear_probe = nn.Linear(d_hidden, n_classes)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.softmax = nn.Softmax(dim=-1)        
    
    def forward(self, x, start_idx = 0, stop_idx=None):
        # process using pretrained vit
        if start_idx == 0 and stop_idx == None:
            return self._forward_full(x)
        else:
            return self._forward_selective(x, start_idx, stop_idx)
    
    def _forward_full(self, x):
        x = self.backbone(x).last_hidden_state
        x = self.dropout(x)
        if self.pooling_type == 'max':
            x = x.max(dim=1)[0]
        
        elif self.pooling_type == 'mean':
            x = x.mean(dim=1)
        
        elif self.pooling_type == 'cls':
            x = x[:, 0]
        
        logits = self.linear_probe(x)
        return {'logits': logits, 'probabilities': self.softmax(logits), 'predictions': torch.argmax(logits, dim=-1)}
    
    def _forward_selective(self, x, start_idx, stop_idx):
        if start_idx == 0:
            x = self.backbone.embeddings(x)
        if stop_idx == None:
            stop_idx = len(self.backbone.encoder.layer)
        for i in range(start_idx, min(stop_idx, len(self.backbone.encoder.layer))):
            x = self.backbone.encoder.layer[i](x)[0]
        
        # return intermediate state
        if stop_idx != len(self.backbone.encoder.layer):
            return {'logits': x}
        
        # complete forward pass
        x = self.backbone.layernorm(x)
        x = self.dropout(x)
        if self.pooling_type == 'max':
            x = x.max(dim=1)[0]
        
        elif self.pooling_type == 'mean':
            x = x.mean(dim=1)
        
        elif self.pooling_type == 'cls':
            x = x[:, 0]
        
        logits = self.linear_probe(x)
        return {'logits': logits, 'probabilities': self.softmax(logits), 'predictions': torch.argmax(logits, dim=-1)}