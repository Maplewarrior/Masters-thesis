import torch.nn as nn


class ActivationVisualizer:
    def __init__(self, model: nn.Modlue) -> None:
        self.model = model

    def visualize_activations(self, retain_loader, forget_loader):
        pass

    def visualize_FIM(self, SSD_unlearner, retain_loader, forget_loader):
        pass


