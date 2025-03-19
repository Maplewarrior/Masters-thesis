from src.unlearners.base_unlearner import BaseUnlearner
from src.sisa_implementation.sisa_class import SISA
import torch.nn as nn

class SISAUnlearner(BaseUnlearner):
    def __init__(self, sisa: SISA) -> None:
        self.sisa = sisa

    def __call__(self, forget_indices: list[int]):  
        # Get the indices of the datapoints to forget
        self.sisa.forget_datapoints(forget_indices)

        return