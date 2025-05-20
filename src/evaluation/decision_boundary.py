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

    # ... existing code ...

    def plot_retain_data(self, X, y, classes, colors):
        for class_idx, color in zip(classes, colors):
            # Use a slightly darker color for the edge
            if isinstance(color, str):
                # Convert hex to RGBA if it's a string
                color = plt.matplotlib.colors.to_rgba(color)
            
            # Create darker version for edge
            edge_color = np.array(color) * 0.7  # Multiply by 0.7 to make it darker
            edge_color[3] = 1.0  # Keep alpha at 1.0 for the edge
            
            plt.scatter(X[y == class_idx, 0], X[y == class_idx, 1], 
                    color=color, 
                    s=70,  # Larger point size
                    label=f"Class {class_idx}", 
                    alpha=0.8,
                    edgecolor=edge_color,
                    linewidth=0.8)

    def plot_decision_boundary(self, x_interval: tuple[float, float], y_interval: tuple[float, float], 
                              alpha=0.4, custom_colors=None):
        """
        Plot the decision boundary and data points.
        
        Args:
            x_interval: Tuple of (min_x, max_x) for plotting
            y_interval: Tuple of (min_y, max_y) for plotting
            alpha: Transparency level for the decision boundary (default: 0.4)
            custom_colors: Optional list of custom colors for classes
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

        # Use professional colors if provided, otherwise use viridis
        if custom_colors is None:
            # Professional color palette
            professional_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']
            # If we have more classes than colors, fall back to viridis
            if len(classes) <= len(professional_colors):
                colors = [professional_colors[i % len(professional_colors)] for i in range(len(classes))]
                # Convert hex to RGBA for easier manipulation
                colors = [plt.matplotlib.colors.to_rgba(color) for color in colors]
            else:
                colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
        else:
            # Use provided custom colors
            colors = [plt.matplotlib.colors.to_rgba(color) for color in custom_colors[:len(classes)]]
        
        custom_cmap = ListedColormap(colors)

        # Plot decision boundary with the specified alpha
        plt.pcolormesh(xx.numpy(), yy.numpy(), decision_boundary.numpy(), 
                    alpha=alpha, cmap=custom_cmap, shading='auto')        

        self.plot_retain_data(X, y, classes, colors)
        
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.title('Decision Boundary', fontsize=14, fontweight='bold')
        plt.legend(
            frameon=True,
            framealpha=0.95,
            facecolor='white',
            edgecolor='lightgray',
            loc='best',
            fontsize=10
        )
        plt.grid(True, alpha=0.3)
        
        # Set white background
        ax = plt.gca()
        ax.set_facecolor('white')
        
        # Improve ticks
        plt.tick_params(direction='out', length=6, width=1)
        
        # Add a subtle border
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_color('lightgray')
        
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