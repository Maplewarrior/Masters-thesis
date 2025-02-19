"""
This script attempts to implement the Max Affine Spline Operator for a simple DNN.

The idea is to use the max affine spline operator to approximate the decision boundary of the model.

Assumptions:
- Each layer in the DNN is a MASO, but the composition of layers is only a MASO iff all component operators are non-decreasing in output dimensions.
- The DNN uses convex affine operators (Such as ReLU and its variants). Sigmoid and Tanh are not convex.

"""

from src.modelling.neural_network import NeuralNet
import torch.nn as nn
from src.data_utils.synthetic_data import create_dataloaders
from src.data_utils.synthetic_data import DataGenerator
from src.modelling.trainer import Trainer
import matplotlib.pyplot as plt
import pdb
import numpy as np

class MASO:
    def __init__(self, model: NeuralNet, train_dataloader):
        self.model = model
        self.maso_params = self.extract_maso_params()
        self.X = train_dataloader.dataset.X
        self.y = train_dataloader.dataset.y
        
    def fit(self, X, y):
        trainer = Trainer(self.model)
        trainer.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def print_layerwise_parameters(self):
        for name, param in self.model.named_parameters():
            print(f"Layer: {name}, Shape: {param.shape}")

    def extract_maso_params(self):
        maso_params = []
        
        # Use named_modules() instead of children() to get all nested modules
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):
                W = layer.weight.detach().numpy()  # Extract weights
                b = layer.bias.detach().numpy()    # Extract biases
                maso_params.append((W, b))
        
        return maso_params

    def plot_input_space_partitions(self, layer_idx: int):
        """
        Plot the input space partitions up to and including the specified layer.
        layer_idx represents how many layers to include (1 = first layer, 2 = first+second layer, etc.)
        """
        # Create a grid of points
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000),
                            np.linspace(y_min, y_max, 1000))
        
        # Reshape the input points (10000, 2)
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T  # Shape: (10000, 2)
        
        # Forward pass through the layers
        current_activation = grid_points
        for i in range(layer_idx):
            W, b = self.maso_params[i]
            # Linear transformation
            current_activation = np.dot(current_activation, W.T) + b
            # Apply ReLU activation
            current_activation = np.maximum(0, current_activation)
        
        # Reshape the final activation for plotting
        Z = current_activation.reshape(xx.shape[0], xx.shape[1], -1)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Convert one-hot encoded labels back to class indices
        y_classes = np.argmax(self.y, axis=1)
        
        # Plot the data points
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=y_classes, cmap='viridis')
        
        # Plot the decision boundary for each neuron in the final layer
        for i in range(Z.shape[-1]):
            plt.contour(xx, yy, Z[:,:,i], levels=[0], colors='red', alpha=0.5)
        
        # Add colorbar for the scatter plot
        plt.colorbar(scatter, label='Class')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Input Space Partitions - Through Layer {layer_idx}')
        plt.show()

if __name__ == "__main__":
    # ==============================
    # Define dataset structure
    # ==============================
    n_samples = 1000
    n_features = 2
    n_classes = 4
    # ==============================
    # Generate data
    # ==============================
    data_generator = DataGenerator(random_state=42)
    # generate data
    data_generator.generate_data(n_samples=n_samples, 
                                 n_features=n_features, 
                                 n_classes=n_classes)
    data_generator.split_data(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # create dataloader
    dataloader = create_dataloaders(data_generator, batch_size=16, use_indices=False, device='cpu', onehot_labels=True)

    train_loader = dataloader["train_full_loader"]
    val_loader = dataloader["val_loader"]
    test_loader = dataloader["test_loader"]

    # ==============================
    # Create model
    # ==============================
    model = NeuralNet(M=n_features, n_classes=n_classes)

    # ==============================
    # Train model
    # ==============================
    trainer = Trainer(model, 
                      train_dataloader=train_loader, 
                      val_dataloader=val_loader, 
                      lr=0.01, 
                      device='cpu') # This trains with Cross Entropy Loss
    trainer.train(n_epochs=20)

    maso = MASO(model, train_loader)
    maso.plot_input_space_partitions(layer_idx=3)