import hydra
import wandb
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
from omegaconf import OmegaConf
import copy
import graphviz
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import RandomSampler, SequentialSampler

from src.models.neural_network import NeuralNetRS
from src.models.vision_transformer_tiny import ViT
from src.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.evaluation.membership_inference_attack import MIA
from src.evaluation.unlearning_evaluator import UnlearningEvaluator
from prepare_image_data import get_image_unlearn_data
from prepare_image_data_v2 import get_image_unlearn_data as get_image_unlearn_data_tsne_box
import time
import json

def has_shuffle(dataloader) -> bool:
    """
    Returns True if dataloader was instantiated with shuffle=True and False otherwise.
    """
    if isinstance(dataloader.sampler, SequentialSampler):
        return False
    elif isinstance(dataloader.sampler, RandomSampler):
        return True
    else:
        raise NotImplementedError(f"The sampler: {type(dataloader.sampler)} is not supported by this function!")

def re_instantiate_dataloaders(dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, seed: int):
    """
    A function used for re-instantiating dataloaders. This is for reproducibility purposes.
    If this was not used, the order of execution would affect the resulting model when shuffle=True.
    """
    def recreate(dataloader):
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            dataloader.dataset, 
            shuffle=has_shuffle(dataloader), 
            batch_size=dataloader.batch_size, 
            generator=g
        )
    dataloader_train = recreate(dataloader_train)
    dataloader_retain = recreate(dataloader_retain)
    dataloader_forget = recreate(dataloader_forget)
    dataloader_val = recreate(dataloader_val)
    return dataloader_train, dataloader_retain, dataloader_forget, dataloader_val

def build_nn(M: int, n_classes: int, n_layers: int, width_factor: int, seed: int) -> nn.Module:
    model = NeuralNetRS(M=M, 
                        n_classes=n_classes, 
                        n_layers=n_layers, 
                        width_factor=width_factor, 
                        seed=seed)
    return model