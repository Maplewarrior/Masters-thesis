import torch
import torch.nn as nn
import copy
from src.models.base_model import BaseModel
import pdb

class PositionalEncoding(nn.Module):
    def __init__(self, n_patches: int, d_hidden: int) -> None:
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, n_patches, d_hidden))
    
    def forward(self, x):
        return x + self.positional_embedding
    
class Embedding(nn.Module):
    def __init__(self, d_patch: int, d_hidden: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_patch)
        self.embedding_projection = nn.Linear(d_patch, d_hidden, bias=False)
        self.layer_norm2 = nn.LayerNorm(d_hidden)
    
    def forward(self, x):
        x_emb = self.layer_norm2(self.embedding_projection(self.layer_norm1(x)))
        return x_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, n_patches: int, d_hidden: int, n_heads: int, d_k: int, tau: float) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.d_k = d_k
        self.n_heads = n_heads
        self.softmax = nn.Softmax(dim=-1)
        self.key_projection = nn.Linear(d_hidden, d_k* n_heads, bias=False)
        self.query_projection = nn.Linear(d_hidden, d_k * n_heads, bias=False)
        self.value_projection = nn.Linear(d_hidden, d_k * n_heads, bias=False)
        self.output_projection = nn.Linear(d_k * n_heads, d_hidden)
        self.tau = torch.tensor(tau)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: A 3d tensor of dimension batch_size x sequence_length x d_hidden
        mask: A 1d tensor of dimension sequence_length with boolean values
        """
        # calculate query, key, and value.
        # reshaped to batch_size x n_heads x n_patches x d_k
        Q = self.query_projection(x).view(x.size(0), self.n_heads, self.n_patches, self.d_k)
        K = self.key_projection(x).view(x.size(0), self.n_heads, self.n_patches, self.d_k)
        V = self.value_projection(x).view(x.size(0), self.n_heads, self.n_patches, self.d_k)

        attn = self.softmax((Q @ K.transpose(-2, -1)) / self.tau.sqrt()) @ V # compute attention for each head
        attn = attn.view(x.size(0), self.n_patches, self.n_heads * self.d_k) # "concatenate" attention activations
        return self.output_projection(attn)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_hidden: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
                                 nn.Linear(d_hidden, d_ff),
                                 nn.GELU(),
                                 nn.Linear(d_ff, d_hidden)
                                )
    
    def forward(self, x: torch.tensor):
        z = self.net(x)
        return z

class EncoderBlock(nn.Module):
    def __init__(self, n_patches: int, n_heads: int, d_k: int, d_hidden: int, d_ff, tau: float):
        super().__init__()
        # build multi-head attention module
        self.mha = MultiHeadAttention(n_patches, d_hidden, n_heads, d_k, tau)
        # define layer norms
        self.layer_norm1 = nn.LayerNorm(d_hidden)
        self.layer_norm2 = nn.LayerNorm(d_hidden)
        # build feed forward network
        self.ffn = FeedForwardNetwork(d_hidden, d_ff)
        
    def forward(self, x):
        x = self.mha(self.layer_norm1(x)) + x # pre-norm + multi head attention + skip-connection
        x = self.ffn(self.layer_norm2(x)) + x # pre-norm + feed forward + skip-connection  
        return x

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

        if self.pooling_type == 'cls':
            self.cls_token = nn.Parameter(torch.randn((1, 1, d_patch)))
            self.n_patches = n_patches + 1 # add one m

        self.positional_encoding = PositionalEncoding(self.n_patches, self.d_hidden)
        self.embed = Embedding(d_patch, d_hidden)
        encoder_block = EncoderBlock(self.n_patches, self.n_heads, self.d_k, self.d_hidden, self.d_ff, self.tau)
        self.encoder_blocks = nn.Sequential(*[copy.deepcopy(encoder_block) for _ in range(self.n_layers)])
        
        self.layer_norm_L = nn.LayerNorm(self.d_hidden)
        self.linear_probe = nn.Linear(self.d_hidden, self.n_classes)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.softmax = nn.Softmax(dim=-1)        
    
    def forward(self, x, start_idx: int = 0, stop_idx: int = None):
        
        if self.pooling_type == 'cls':
            x = torch.cat([self.cls_token.repeat((x.size(0), 1, 1)), x], dim=1)
        # embed the input and add positional encodings 
        x = self.positional_encoding(self.embed(x))
        x = self.dropout(x)
        # pass embedded image through encoder
        x = self.encoder_blocks[start_idx:stop_idx](x)
        if stop_idx != None: # return model activation at some layer
            raise NotImplementedError("We need to reconstruct need feed forward layer of encoder block. This is a TODO...")
            # return x
        else:
            x = self.layer_norm_L(x)
            x = self.dropout(x)
            if self.pooling_type == 'max':
                # max pooling
                x = x.max(dim=1)[0]
            elif self.pooling_type == 'mean':
                x = x.mean(dim=1)
            elif self.pooling_type == 'cls':
                x = x[:, 0]

            logits = self.linear_probe(x)
            return {'logits': logits, 'probabilities': self.softmax(logits)}
