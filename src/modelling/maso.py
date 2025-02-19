"""
This script attempts to implement the Max Affine Spline Operator for a simple DNN.

The idea is to use the max affine spline operator to approximate the decision boundary of the model.

Assumptions:
- Each layer in the DNN is a MASO, but the composition of layers is only a MASO iff all component operators are non-decreasing in output dimensions.
- The DNN uses convex affine operators (Such as ReLU and its variants). Sigmoid and Tanh are not convex.

"""

from src.modelling.neural_network import NeuralNetwork
from src.modelling.trainer import Trainer

class MASO:
    def __init__(self, model: NeuralNetwork):
        self.model = model

    def fit(self, X, y):
        trainer = Trainer(self.model)
        trainer.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
