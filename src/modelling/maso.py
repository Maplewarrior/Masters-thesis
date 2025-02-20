"""
This script attempts to implement the Max Affine Spline Operator for a simple DNN.

The idea is to use the max affine spline operator to approximate the decision boundary of the model.

Assumptions:
- Each layer in the DNN is a MASO, but the composition of layers is only a MASO iff all component operators are non-decreasing in output dimensions.
- The DNN uses convex affine operators (Such as ReLU and its variants). Sigmoid and Tanh are not convex.

"""

from src.modelling.neural_network import NeuralNet
import torch.nn as nn
import torch
from src.data_utils.synthetic_data import create_dataloaders
from src.data_utils.synthetic_data import DataGenerator
from src.modelling.trainer import Trainer
import matplotlib.pyplot as plt
import pdb
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.evaluation.decision_boundary import DecisionBoundaryCreator
from matplotlib.colors import ListedColormap

class MasoDataset(Dataset):
    def __init__(self, X, y, use_indices: bool = False, onehot_labels: bool = False, n_classes: int = None, device="cpu"):
        self.device = device
        self.X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.y = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        self.indices = torch.arange(len(self.X), device=self.device) if use_indices else None
        self.onehot_labels = onehot_labels

        if self.onehot_labels:
            self.y = self.onehot_encode_labels(self.y, n_classes)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.indices is not None:
            return self.X[idx], self.y[idx], self.indices[idx]
        else:
            return self.X[idx], self.y[idx]
        
    def onehot_encode_labels(self, y, n_classes):
        """
        Convert labels to one-hot encoding.
        If labels are already one-hot encoded, return as is.
        """
        # Check if labels are already one-hot encoded
        if len(y.shape) > 1 and y.shape[1] > 1:
            return y
            
        if n_classes is None:
            n_classes = int(y.max() + 1)
        return torch.zeros(len(y), n_classes, device=self.device).scatter_(1, y.unsqueeze(1), 1)

class MASO:
    def __init__(self, model: NeuralNet, train_dataloader, _lambda: float = 0.0):
        self.model = model
        self.maso_params = self.extract_maso_params()
        self.X = train_dataloader.dataset.X
        self.y = train_dataloader.dataset.y
        self.partition_map = self.compute_partition_indices(self.X)
        self.train_dataloader = train_dataloader
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self._lambda = _lambda

    def loss(self, logits, y):
        # logits = self.model(X)['logits']
        # pdb.set_trace()
        fit_term = self.cross_entropy_loss(logits, y)
        W_L = self.model.net[-1].weight
        W = W_L @ W_L.T
        reg_term = torch.sum(torch.abs(W - torch.diag(torch.diag(W))))
        return fit_term + self._lambda * reg_term
    
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

    def plot_input_space_partitions(self, layer_idx: int, activation_fn: nn.Module):
        """
        Plot the input space partitions up to and including the specified layer.
        layer_idx represents how many layers to include (1 = first layer, 2 = first+second layer, etc.)
        Previous layers' splines are shown in grey.
        """
        # Create a grid of points
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000),
                            np.linspace(y_min, y_max, 1000))
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Convert one-hot encoded labels back to class indices
        y_classes = np.argmax(self.y, axis=1)
        
        # Plot the data points
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=y_classes, cmap='viridis')
        
        # Plot splines for each layer
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        current_activation = grid_points
        
        # Plot each layer's splines
        for layer in range(layer_idx):
            W, b = self.maso_params[layer]
            # Linear transformation
            current_activation = np.dot(current_activation, W.T) + b
            # Apply ReLU activation
            current_activation = activation_fn(torch.tensor(current_activation, dtype=torch.float32)).numpy()
            
            # Reshape for plotting
            Z = current_activation.reshape(xx.shape[0], xx.shape[1], -1)
            
            # If this is not the final layer, plot in grey
            if layer < layer_idx - 1:
                color = 'grey'
                alpha = 0.6 * ((layer + 1) / layer_idx)  # Scale alpha by layer depth
            else:
                color = 'red'
                alpha = 0.8
                
            # Plot the decision boundary for each neuron
            for i in range(Z.shape[-1]):
                plt.contour(xx, yy, Z[:,:,i], levels=[0], colors=color, alpha=alpha)
        
        # Add colorbar for the scatter plot
        plt.colorbar(scatter, label='Class')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Input Space Partitions - Through Layer {layer_idx}\n(Previous layers shown in grey)\nActivation Function: {activation_fn.__name__}')
        plt.show()

    def compute_partition_indices(self, X):
        """
        Based on Eq. (14) in Balestriero et al.
        Computes the partition index t^{(ℓ)}(x) for each input X at each layer ℓ.
        Returns a dictionary mapping layer indices to partition indices.
        """

        partition_indices = {}  # Stores partition indices per layer
        current_activation = torch.tensor(X, dtype=torch.float32)  # Input

        for layer_idx, (W, b) in enumerate(self.maso_params):
            W_torch = torch.tensor(W, dtype=torch.float32)  # Convert to tensors
            b_torch = torch.tensor(b, dtype=torch.float32)

            # Compute the affine transformation for each partition
            # Equivalent to just using a linear transformation, but this looks cooler.
            affine_values = torch.einsum("rd,bd->br", W_torch, current_activation) + b_torch

            # Compute argmax over partitions (r)
            # Based on Eq. (14) in Balestriero et al.
            t_l_x = torch.argmax(affine_values, dim=1)  # Shape: (batch_size,)

            # Store results
            partition_indices[layer_idx] = t_l_x.numpy()

            # Apply ReLU activation for the next layer
            current_activation = torch.relu(affine_values)

        return partition_indices

    def distance(self, x1, x2):
        """
        We implement Eq. (15) in Balestriero et al. An near implementation of Nearest Neighbours.
        The distance between two partition indices t^(l)(x1) and t^(l)(x2) is:
        d(t^(l)(x1), t^(l)(x2)) = 1 - (number of matching indices) / (total indices)
        
        Where:
        - t^(l)(x) is the partition index at layer l for input x
        - The numerator counts how many corresponding indices match between x1 and x2
        - D^(l) in denominator is the total number of indices at layer l

        This assumes an Activiation MASO, meaning that the activation function is a ReLU or other affine non-decreasing function.
        If a max-pooling MASO is used then the distance will be unintpreted.
        """
        distance = 0
        for layer_idx in range(len(self.partition_map)):
            distance += 1 - (self.partition_map[layer_idx][x1] == self.partition_map[layer_idx][x2])
        return distance / len(self.partition_map)

    def plot_combined_visualization(self, layer_idx: int, activation_fn: nn.Module, x_interval=(-10, 10), y_interval=(-10, 10)):
        """
        Plot both the decision boundary and input space partitions in a single plot.
        layer_idx represents how many layers to include for partitions.
        """
        # Create a grid of points
        x_min, x_max = x_interval
        y_min, y_max = y_interval
        xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 1000),
                            torch.linspace(y_min, y_max, 1000))
        
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Plot decision boundary first
        decision_boundary_creator = DecisionBoundaryCreator(self.model, self.train_dataloader)
        xx_db, yy_db, decision_boundary = decision_boundary_creator.create_decision_boundary(x_interval, y_interval)
        
        # Get number of unique classes
        y = self.y
        if len(y.shape) == 2 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)
        classes = np.unique(y)
        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
        custom_cmap = ListedColormap(colors)
        
        # Plot decision boundary
        plt.pcolormesh(xx_db.numpy(), yy_db.numpy(), decision_boundary.numpy(), 
                      alpha=0.2, cmap=custom_cmap, shading='auto')

        # Plot splines for each layer
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        current_activation = grid_points
        
        # Plot each layer's splines
        for layer in range(layer_idx):
            W, b = self.maso_params[layer]
            # Linear transformation
            current_activation = np.dot(current_activation, W.T) + b
            # Apply activation
            current_activation = activation_fn(torch.tensor(current_activation, dtype=torch.float32)).numpy()
            
            # Reshape for plotting
            Z = current_activation.reshape(xx.shape[0], xx.shape[1], -1)
            
            # If this is not the final layer, plot in grey
            if layer < layer_idx - 1:
                color = 'grey'
                alpha = 0.3 * ((layer + 1) / layer_idx)  # Scale alpha by layer depth
            else:
                color = 'red'
                alpha = 0.5
                
            # Plot the splines for each neuron
            for i in range(Z.shape[-1]):
                plt.contour(xx, yy, Z[:,:,i], levels=[0], colors=color, alpha=alpha, linewidths=1)
        
        # Plot the data points
        for class_idx, color in zip(classes, colors):
            plt.scatter(self.X[y == class_idx, 0], self.X[y == class_idx, 1], 
                       color=color, s=20, label=f"Class {class_idx}", alpha=0.6)
        
        plt.colorbar(label='Class')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Decision Boundary and Input Space Partitions Through Layer {layer_idx}\n'
                 f'(Previous layers shown in grey)\nActivation Function: {activation_fn.__name__}')
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.show()

def maso_dataset(n_samples: int, n_features: int, n_classes: int, random_state: int = 42):
    """
    Generate synthetic dataset with random patterns (circular, semicircular, or gaussian) for different classes.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features (should be 2 for visualization)
        n_classes: Number of classes to generate
    
    Returns:
        X: Features array of shape (n_samples, n_features)
        y: One-hot encoded labels of shape (n_samples, n_classes)
    """
    assert n_features == 2, "This generator only works with 2D data"
    
    # Initialize arrays
    X = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples, n_classes))
    samples_per_class = n_samples // n_classes

    np.random.seed(random_state)
    
    for class_idx in range(n_classes):
        start_idx = class_idx * samples_per_class
        end_idx = start_idx + samples_per_class
        
        # Randomly choose pattern type for this class
        pattern_type = np.random.choice(['circular', 'semicircular', 'gaussian'])
        
        if pattern_type == 'gaussian':
            # Random center point for the gaussian
            center = np.random.uniform(-5, 5, 2)
            X[start_idx:end_idx] = np.random.multivariate_normal(
                mean=center,
                cov=[[0.5, 0], [0, 0.5]],
                size=samples_per_class
            )
        else:
            # Generate radius and angles
            radius = 2 * (class_idx + 1)  # Increasing radii for each class
            
            if pattern_type == 'circular':
                theta = np.random.uniform(0, 2*np.pi, samples_per_class)
            else:  # semicircular
                # Random starting angle for the semicircle
                start_angle = np.random.uniform(0, np.pi)
                theta = np.random.uniform(
                    start_angle, 
                    start_angle + np.pi, 
                    samples_per_class
                )
            
            # Add some noise to radius
            radius = np.random.normal(radius, 0.2, samples_per_class)
            
            # Convert to Cartesian coordinates
            X[start_idx:end_idx, 0] = radius * np.cos(theta)
            X[start_idx:end_idx, 1] = radius * np.sin(theta)
            
            # Add small random offset to center
            center_offset = np.random.uniform(-2, 2, 2)
            X[start_idx:end_idx] += center_offset
        
        # Add one-hot encoded labels
        y[start_idx:end_idx, class_idx] = 1
    
    return X, y

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
    # data_generator = DataGenerator(random_state=42)
    # # generate data
    # data_generator.generate_data(n_samples=n_samples, 
    #                              n_features=n_features, 
    #                              n_classes=n_classes)
    # data_generator.split_data(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # # create dataloader
    # dataloader = create_dataloaders(data_generator, batch_size=16, use_indices=False, device='cpu', onehot_labels=True)

    # train_loader = dataloader["train_full_loader"]
    # val_loader = dataloader["val_loader"]
    # test_loader = dataloader["test_loader"]

    X, y = maso_dataset(n_samples=n_samples, 
                       n_features=n_features, 
                       n_classes=n_classes,
                       random_state=69)
    
    # Train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # Since y is already one-hot encoded from maso_dataset
    train_dataset = MasoDataset(X_train, y_train, onehot_labels=True, n_classes=n_classes)
    val_dataset = MasoDataset(X_val, y_val, onehot_labels=True, n_classes=n_classes)
    test_dataset = MasoDataset(X_test, y_test, onehot_labels=True, n_classes=n_classes)

    
    # Convert to DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
    # ==============================
    # Create model
    # ==============================
    model = NeuralNet(M=n_features, n_classes=n_classes)

    # ==============================
    # Train model
    # ==============================
    maso = MASO(model, train_loader, _lambda=0.3)
    trainer = Trainer(model, 
                     train_dataloader=train_loader, 
                     val_dataloader=val_loader, 
                     lr=0.01, 
                     device='cpu',
                     loss=maso.loss)
    
    trainer.train(n_epochs=100)

    maso.plot_combined_visualization(layer_idx=3, 
                                   activation_fn=nn.functional.gelu,
                                   x_interval=(-10, 10),
                                   y_interval=(-10, 10))
    # evaluate model
    trainer.eval()


    # Compute distance matrix for all points in train_loader
    # n_samples = len(train_loader.dataset)
    # distance_matrix = np.zeros((n_samples, n_samples))
    # for i in range(n_samples):
    #     for j in range(i+1, n_samples):
    #         distance_matrix[i, j] = maso.distance(i, j)
    #         distance_matrix[j, i] = distance_matrix[i, j]
    
    # # Plot distance matrix
    # plt.imshow(distance_matrix, cmap='viridis')
    # plt.colorbar()
    # plt.show()
