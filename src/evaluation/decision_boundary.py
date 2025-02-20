import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

class DecisionBoundaryCreator:
    def __init__(self, model: nn.Module, dataloader: DataLoader):
        self.model = model
        self.dataloader = dataloader

    def create_decision_boundary(self, x_interval: tuple[float, float], y_interval: tuple[float, float]):
        # Mesh grid
        x = torch.linspace(x_interval[0], x_interval[1], 1000)
        y = torch.linspace(y_interval[0], y_interval[1], 1000)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        # Reshape the grid into a 2D array of points
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get predictions for all points
        with torch.no_grad():
            output = self.model(grid)
            # Handle dictionary output
            if isinstance(output, dict):
                probabilities = output['probabilities']  # Use predictions key
                predictions = torch.argmax(probabilities, dim=1)
            elif isinstance(output, tuple):
                predictions = output[0]
            else:
                predictions = output
            
            # Ensure predictions are integers to avoid interpolation
            predictions = predictions.long()
        
        # Reshape predictions back to grid shape
        decision_boundary = predictions.reshape(xx.shape)
        
        return xx, yy, decision_boundary.float()  # Convert back to float for plotting

    def plot_decision_boundary(self, x_interval: tuple[float, float], y_interval: tuple[float, float]):
        """
        Plot the decision boundary and data points.
        
        Args:
            x_interval: Tuple of (min_x, max_x) for plotting
            y_interval: Tuple of (min_y, max_y) for plotting
        """
        # Get decision boundary
        xx, yy, decision_boundary = self.create_decision_boundary(x_interval, y_interval)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Get number of unique classes from the dataloader
        X, y = self.dataloader.dataset.X, self.dataloader.dataset.y

        # If y is onehot, convert it to class indices
        if len(y.shape) == 2 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1) 

        classes = np.unique(y)
        num_classes = len(classes)

        print("There are ", num_classes, " classes")

        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
        custom_cmap = ListedColormap(colors)


        # Plot decision boundary
        # plt.pcolormesh(xx.numpy(), yy.numpy(), decision_boundary.numpy(), 
        #             alpha=0.4, cmap=custom_cmap, levels=num_classes)
        plt.pcolormesh(xx.numpy(), yy.numpy(), decision_boundary.numpy(), 
                      alpha=0.4, cmap=custom_cmap, shading='auto')        

        for class_idx, color in zip(classes, colors):
            plt.scatter(X[y == class_idx, 0], X[y == class_idx, 1], color=color, s=10, label=f"Class {class_idx}", alpha=0.5)
        
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt


if __name__ == "__main__":
    from src.modelling.neural_network import NeuralNet
    from src.data_utils.synthetic_data import create_dataloaders
    from src.data_utils.synthetic_data import DataGenerator
    from src.modelling.trainer import Trainer

    import pdb


    # ==============================
    # Generate data
    # ==============================
    data_generator = DataGenerator(random_state=42)
    # generate data
    data_generator.generate_data(n_samples=1000, n_features=2, n_classes=4)
    data_generator.split_data(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # create dataloader
    dataloader = create_dataloaders(data_generator, batch_size=16, use_indices=False, device='cpu', onehot_labels=True)

    train_loader = dataloader["train_full_loader"]
    val_loader = dataloader["val_loader"]
    test_loader = dataloader["test_loader"]

    # ==============================
    # Create model
    # ==============================
    model = NeuralNet(M=2, n_classes=4)


    # ==============================
    # Train model
    # ==============================
    trainer = Trainer(model, 
                      train_dataloader=train_loader, 
                      val_dataloader=val_loader, 
                      lr=0.01, 
                      device='cpu')
    trainer.train(n_epochs=20)


    creator = DecisionBoundaryCreator(model, train_loader)
    creator.plot_decision_boundary((-4, 4), (-4, 4))
    plt.show()