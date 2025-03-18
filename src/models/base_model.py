import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, M: int, n_classes: int) -> None:
        super().__init__()
        self.M = M
        self.n_classes = n_classes
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.tensor, 
                start_idx: int = 0, 
                stop_idx: int | None = None) -> dict:
        raise NotADirectoryError()
    
    def inference(self, x: torch.tensor, start_idx: int = 0, stop_idx: int | None = None) -> dict:
        self.eval()
        with torch.no_grad():
            return self(x, start_idx, stop_idx)
        

    def loss(self, out: dict, y: torch.tensor):
        """
        @param out: A dictionary containing the output of a forward pass.
                    - out should have a logits key where the value is a tensor of shape batch_size x n_classes.
        @oparam y: A tensor of shape batch_size x n_classes containing one-hot encoded class labels.
        """
        return nn.functional.cross_entropy(input=out['logits'], target=y)
    
