import torch

from src.modelling.trainer import Trainer
from src.modelling.neural_network import NeuralNet

class SplineDistance:
    def __init__(self, xmin, xmax, N) -> None:
        self.xmin = xmin
        self.xmax = xmax
        self.N = N
        self.domain = torch.linspace(start=xmin, end=xmax, steps=N)
    
    def inner_product(self, a1, b1, a2, b2):
        """
        int(f, g; a, b) = int_b^a (f(x) g(x)) dx = f(b)g(b) * dx + f(b+dx)g(b+dx) * dx + ... + f(a)g(a) * dx
        """
        f = a1 * self.domain + b1
        g = a2 * self.domain + b2 # conjugate(g(x)) = g(x) because we only have real coefficients...
        # g_conjugate = 
        dx = 1/self.N
        return (f @ g * dx).sum()


