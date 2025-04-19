import torch
import torch.nn as nn
from src.models.base_model import BaseModel

class MultiHeadAttention(nn.Module):
    def __init__(self, n_patches: int, d_hidden: int, n_heads: int, d_k: int, tau: float) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.d_k = d_k
        self.n_heads = n_heads
        self.softmax = nn.softmax(dim=-1)
        self.key_projection = nn.Linear(d_hidden, d_k* n_heads, bias=False)
        self.query_projection = nn.Linear(d_hidden, d_k * n_heads, bias=False)
        self.value_projection = nn.Linear(d_hidden, d_k * n_heads, bias=False)
        self.output_projection = nn.Linear(d_k * n_heads, d_hidden)
        self.tau = torch.tensor(tau)

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """
        x: A 3d tensor of dimension batch_size x sequence_length x d_hidden
        mask: A 1d tensor of dimension sequence_length with boolean values
        """
        # calculate query, key, and value.
        # reshaped to batch_size x n_heads x n_patches x d_k
        Q = self.query_projection(x).view(-1, self.n_heads, self.n_patches, self.d_k)
        K = self.key_projection(x).view(-1, self.n_heads, self.n_patches, self.d_k)
        V = self.value_projection(x).view(-1, self.n_heads, self.n_patches, self.d_k)

        attn = self.softmax((Q @ K.transpose()) / self.tau.sqrt()) @ V
        return self.output_projection(attn)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_hidden: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
                                 nn.Linear(d_hidden, d_ff),
                                 nn.ReLU(),
                                 nn.Linear(d_ff, d_hidden)
                                )
    
    def forward(self, x: torch.tensor):
        z = self.net(x)
        return z

class EncoderBlock(nn.Module):
    def __init__(self, h: int, d_k: int, d_hidden: int, d_ff, tau: float):
        
        # build multi-head attention module
        self.mha = MultiHeadAttention(d_hidden, h, d_k, tau)
        # build feed forward network
        self.ffn = FeedForwardNetwork(d_hidden, d_ff)
    
    def forward(self, x):
        z = self.mha(x) + x # multi head attention + skip connection
        z = self.ffn(z)
        return z


class ViT(BaseModel):
    def __init__(self, 
                 img_size: int, 
                 patch_size: int, 
                 n_classes: int) -> None:
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError()
        