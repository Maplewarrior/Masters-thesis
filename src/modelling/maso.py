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
from src.modelling.trainer import Trainer
import matplotlib.pyplot as plt
import pdb
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.evaluation.decision_boundary import DecisionBoundaryCreator
from matplotlib.colors import ListedColormap
import os

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
        self.model_train_weights = None
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
            Z = current_activation.reshape(xx.shape[0], xx.shape[1], -1).numpy()
            
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
            affine_values = torch.matmul(current_activation, W_torch.T) + b_torch

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

    def plot_combined_visualization(self, layer_idx: int, activation_fn: nn.Module, x_interval=(-10, 10), y_interval=(-10, 10), save_path: str = None):
        """
        Plot both the decision boundary and input space partitions in a single plot.
        layer_idx represents how many layers to include for partitions.
        """
        # Create a grid of points
        x = torch.linspace(x_interval[0], x_interval[1], 1000)
        y = torch.linspace(y_interval[0], y_interval[1], 1000)
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
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
        current_activation = grid
        
        # Plot each layer's splines
        for layer in range(layer_idx):
            W, b = self.maso_params[layer]
            # Linear transformation
            current_activation = np.dot(current_activation, W.T) + b
            
            # If this is not the final layer, plot in grey
            if layer < layer_idx - 1:
                color = 'grey'
                
                current_activation = activation_fn(torch.tensor(current_activation, dtype=torch.float32)).numpy()
                Z = current_activation.reshape(xx.shape[0], xx.shape[1], -1)
            else:
                color = 'red'

                Z = current_activation.reshape(xx.shape[0], xx.shape[1], -1)

                if Z.shape[-1] > 2:  # multiclass case
                    # Get regions where each class has maximum logit
                    max_indices = np.argmax(Z, axis=2)
                    for i in range(Z.shape[-1]):
                        class_region = (max_indices == i)
                        plt.contour(xx, yy, class_region, colors=colors[i], linewidths=2, zorder=1)
                else:  # binary case
                    decision_boundary = Z[:,:,0] - Z[:,:,1]
                    plt.contour(xx, yy, decision_boundary, levels=[0], 
                              colors=color, linewidths=2, zorder=1)
                
            for i in range(Z.shape[-1]):
                plt.contour(xx, yy, Z[:,:,i], levels=[0], colors=color, linewidths=1)
        
        # Plot the data points
        for class_idx, color in zip(classes, colors):
            plt.scatter(self.X[y == class_idx, 0], self.X[y == class_idx, 1], 
                       color=color, s=20, label=f"Class {class_idx}", alpha=0.6)
            
        # Add area, by amount of gridpoints in each decision region to legend
        decision_boundary = np.unique(decision_boundary.numpy(), return_counts=True)
        total_points = grid.shape[0] * grid.shape[1]
        for i in range(decision_boundary[0].shape[0]):
            plt.scatter([], [], color=colors[i], label=f"Decision region {i} has {decision_boundary[1][i]/total_points*100:.2f}% of gridpoints", alpha=0.6)

        plt.colorbar(label='Class')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Decision Boundary and Input Space Partitions Through Layer {layer_idx}\n'
                 f'(Previous layers shown in grey)\nActivation Function: {activation_fn.__name__}')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.2)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

def maso_dataset(n_samples: int, n_features: int, n_classes: int):
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
    
    # Calculate base samples per class and remaining samples
    base_samples_per_class = n_samples // n_classes
    remaining_samples = n_samples % n_classes
    
    current_idx = 0
    for class_idx in range(n_classes):
        # Add one extra sample to this class if there are remaining samples
        samples_this_class = base_samples_per_class + (1 if class_idx < remaining_samples else 0)
        end_idx = current_idx + samples_this_class
        
        # Randomly choose pattern type for this class
        pattern_type = np.random.choice(['circular', 'semicircular', 'gaussian'])
        
        if pattern_type == 'gaussian':
            # Random center point for the gaussian
            center = np.random.uniform(-5, 5, 2)
            X[current_idx:end_idx] = np.random.multivariate_normal(
                mean=center,
                cov=[[0.5, 0], [0, 0.5]],
                size=samples_this_class
            )
        else:
            # Generate radius and angles
            radius = 2 * (class_idx + 1)  # Increasing radii for each class
            
            if pattern_type == 'circular':
                theta = np.random.uniform(0, 2*np.pi, samples_this_class)
            else:  # semicircular
                # Random starting angle for the semicircle
                start_angle = np.random.uniform(0, np.pi)
                theta = np.random.uniform(
                    start_angle, 
                    start_angle + np.pi, 
                    samples_this_class
                )
            
            # Add some noise to radius
            radius = np.random.normal(radius, 0.2, samples_this_class)
            
            # Convert to Cartesian coordinates
            X[current_idx:end_idx, 0] = radius * np.cos(theta)
            X[current_idx:end_idx, 1] = radius * np.sin(theta)
            
            # Add small random offset to center
            center_offset = np.random.uniform(-2, 2, 2)
            X[current_idx:end_idx] += center_offset
        
        # Add one-hot encoded labels
        y[current_idx:end_idx, class_idx] = 1
        current_idx = end_idx
    
    return X, y

def perturb_and_gif(model: NeuralNet, 
                    maso: MASO, 
                    output_dir: str,
                    layer_idx: int,
                    weight_x_idx: int,
                    weight_y_idx: int):
    frames = []
    # Create dir for layer
    os.makedirs(f"{output_dir}/layer{layer_idx}", exist_ok=True)
    # Make one dir for gifs
    os.makedirs(f"{output_dir}/gifs", exist_ok=True)
    for weight_perturbation in np.linspace(1, 0, 20):
        save_path = f"{output_dir}/layer{layer_idx}/weight_perturbed_{layer_idx}_{weight_x_idx}_{weight_y_idx}_{weight_perturbation}.png"
        frames.append(save_path)
        model.net[layer_idx].weight.data[weight_x_idx, weight_y_idx] *= weight_perturbation

        # save_path = "src/modelling/maso_model/original_model.png"

        # ==============================
        # Plot results
        # ==============================
        maso.plot_combined_visualization(layer_idx=3, 
                                    activation_fn=nn.functional.relu,
                                    x_interval=(-10, 10),
                                    y_interval=(-10, 10),
                                    save_path=save_path)
    
    gif_path = f"{output_dir}/gifs/weight_perturbation_layer{layer_idx}_{weight_x_idx}_{weight_y_idx}.gif"
    with imageio.get_writer(gif_path, mode='I', duration=10) as writer:
        for frame_path in frames:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    
    print(f"GIF created at: {gif_path}")

def pertubation_db_distributions(model: NeuralNet, 
                                 maso: MASO, 
                                 output_dir: str):
    """
    Function to perturb each weight individually (set it to 0), and then save the distribution of the decision boundary.
    We use this to visualise the change in decision boundary from the original model.
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/heatmaps", exist_ok=True)
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    
    # Define grid parameters for decision boundary calculation
    x_interval = (-10, 10)
    y_interval = (-10, 10)
    
    # Create decision boundary creator
    decision_boundary_creator = DecisionBoundaryCreator(model, maso.train_dataloader)
    
    # Get original model's decision boundary
    _, _, original_db = decision_boundary_creator.create_decision_boundary(x_interval, y_interval)
    original_db_np = original_db.numpy()
    
    # Calculate original class distribution (number of grid points per class)
    original_class_counts = np.unique(original_db_np, return_counts=True)
    original_counts_dict = dict(zip(original_class_counts[0], original_class_counts[1]))
    
    # Save original model visualization
    plt.figure(figsize=(12, 10))
    maso.plot_combined_visualization(layer_idx=3, 
                                    activation_fn=nn.functional.relu,
                                    x_interval=x_interval,
                                    y_interval=y_interval,
                                    save_path=f"{output_dir}/original_model.png")
    
    # Store results for all perturbations
    perturbation_results = []
    
    # Iterate through each layer
    for layer_idx, layer in enumerate(model.net):
        if not isinstance(layer, nn.Linear):
            continue
            
        # Get the original weights
        original_weights = layer.weight.data.clone()
        
        # Create heatmap data structures for this layer
        layer_shape = original_weights.shape
        area_change_heatmap = np.zeros(layer_shape)
        
        # Iterate through each weight in the layer
        for i in range(layer_shape[0]):
            for j in range(layer_shape[1]):
                # Set the weight to 0
                layer.weight.data[i, j] = 0.0
                
                # Recalculate decision boundary
                _, _, perturbed_db = decision_boundary_creator.create_decision_boundary(x_interval, y_interval)
                perturbed_db_np = perturbed_db.numpy()
                
                # Calculate perturbed class distribution
                perturbed_class_counts = np.unique(perturbed_db_np, return_counts=True)
                perturbed_counts_dict = dict(zip(perturbed_class_counts[0], perturbed_class_counts[1]))
                
                # Calculate area changes for each class
                class_area_changes = {}
                total_area_change = 0
                
                # Calculate absolute changes in area for each class
                for class_idx in set(list(original_counts_dict.keys()) + list(perturbed_counts_dict.keys())):
                    original_count = original_counts_dict.get(class_idx, 0)
                    perturbed_count = perturbed_counts_dict.get(class_idx, 0)
                    absolute_change = abs(perturbed_count - original_count)
                    relative_change = absolute_change / original_count if original_count > 0 else float('inf')
                    
                    class_area_changes[class_idx] = {
                        'original': original_count,
                        'perturbed': perturbed_count,
                        'absolute_change': absolute_change,
                        'relative_change': relative_change
                    }
                    
                    total_area_change += absolute_change
                
                # Store the total area change in the heatmap
                area_change_heatmap[i, j] = total_area_change
                
                # Store detailed results
                perturbation_results.append({
                    'layer_idx': layer_idx,
                    'weight_i': i,
                    'weight_j': j,
                    'total_area_change': total_area_change,
                    'class_area_changes': class_area_changes
                })
                
                # Restore the original weight
                layer.weight.data[i, j] = original_weights[i, j]
        
        # Create and save heatmap for this layer
        plt.figure(figsize=(10, 8))
        plt.imshow(area_change_heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Total Decision Boundary Area Change')
        plt.title(f'Impact of Weight Perturbation on Decision Boundary - Layer {layer_idx}')
        plt.xlabel('Input Neuron Index')
        plt.ylabel('Output Neuron Index')
        plt.savefig(f"{output_dir}/heatmaps/layer{layer_idx}_heatmap.png")
        plt.close()
    
    # Create summary visualization: bar chart of most influential weights
    # Sort perturbation results by total area change
    sorted_results = sorted(perturbation_results, key=lambda x: x['total_area_change'], reverse=True)
    top_n = 20  # Show top 20 most influential weights
    
    plt.figure(figsize=(14, 8))
    
    # Extract data for top weights
    top_weights = sorted_results[:top_n]
    labels = [f"L{r['layer_idx']}_W{r['weight_i']}_{r['weight_j']}" for r in top_weights]
    values = [r['total_area_change'] for r in top_weights]
    
    # Create bar chart
    bars = plt.bar(range(len(labels)), values, color='skyblue')
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.xlabel('Weight (Layer_OutputNeuron_InputNeuron)')
    plt.ylabel('Total Decision Boundary Area Change')
    plt.title('Top Influential Weights on Decision Boundary')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_influential_weights.png")
    plt.close()
    
    # Save detailed results as numpy array for further analysis
    np.save(f"{output_dir}/perturbation_results.npy", np.array(perturbation_results, dtype=object))
    
    print(f"Perturbation analysis completed. Results saved to {output_dir}")
    
    return perturbation_results

def plot_during_training(maso: MASO, 
                         output_dir: str,
                         epochs: int):
    """
    Function to plot the decision boundary during training.
    We expect to have run the training loop and saved the weights for each epoch.
    """
    from tqdm import tqdm
    # for each epoch we load the weights into the model
    # We create a gif of the decision boundary changing over time
    frames = []
    for epoch in tqdm(range(epochs)):
        model = NeuralNet(M=2, n_classes=3)
        save_path = f"{output_dir}/epoch_{epoch+1}.png"
        weights = torch.load(f"{output_dir}/epoch_train_weights_{epoch+1}.pth")
        model.load_state_dict(weights)
        maso.model = model
        maso.maso_params = maso.extract_maso_params()
        maso.plot_combined_visualization(layer_idx=3, 
                                        activation_fn=nn.functional.relu,
                                        x_interval=(-10, 10),
                                        y_interval=(-10, 10),
                                        save_path=save_path)
        frames.append(save_path)
    # Create a gif of the decision boundary changing over time
    gif_path = f"{output_dir}/training_gif.gif"
    with imageio.get_writer(gif_path, mode='I', duration=10) as writer:
        for frame_path in frames:
            image = imageio.imread(frame_path)
            writer.append_data(image)

def plot_with_varying_widths(maso: MASO,
                             epochs: int,
                             output_dir: str,
                             train_dataloader: any,
                             val_dataloader: any,
                             widths: list[float]):
    """
    Function to plot the decision boundary with varying widths of the network.
    We expect to have run the training loop and saved the weights for each epoch.
    """
    from tqdm import tqdm
    frames = []
    os.makedirs(output_dir, exist_ok=True)
    for width in tqdm(widths):
        model = NeuralNet(M=2, n_classes=3, width=width)
        trainer = Trainer(model, 
                         train_dataloader=train_dataloader,
                         val_dataloader=val_dataloader,
                         lr=0.001, 
                         device='cpu',
                         loss=maso.loss)
        trainer.train(n_epochs=epochs)

        maso.model = model
        maso.maso_params = maso.extract_maso_params()
        save_path = f"{output_dir}/width_{width}.png"
        frames.append(save_path)
        maso.plot_combined_visualization(layer_idx=3, 
                                        activation_fn=nn.functional.relu,
                                        x_interval=(-10, 10),
                                        y_interval=(-10, 10),
                                        save_path=save_path)
        
    gif_path = f"{output_dir}/width_varying_gif.gif"
    with imageio.get_writer(gif_path, mode='I', duration=10) as writer:
        for frame_path in frames:
            image = imageio.imread(frame_path)
            writer.append_data(image)

def compare_splines(spline_1: nn.Linear,
                    spline_2: nn.Linear):
    """
    Function to parameterise the splines, integrate them.
    Construct a grid to approximate the integral of the splines.
    Compare the integrated values of the splines.
    """
    # Parameterise the splines
    # Since we can consider splines as affine linear functions, we can parameterise them as such.
    # This means we just have to integrate the linear function over the interval.
    # the integrated linear function is: F(x) = (a/2)x^2 + bx + c
    raise NotImplementedError("Not implemented")
    
    grid = torch.linspace(0, 1, 100)

    spline_1_weights = spline_1.weight.data / 2
    spline_2_weights = spline_2.weight.data / 2
    spline_1_bias = spline_1.bias.data
    spline_2_bias = spline_2.bias.data
    
    # Integrate the splines
    # reshape grid to match weight dimensions
    grid = grid.view(1, -1)
    integral_1 = torch.sum(spline_1_weights * grid**2 + spline_1_bias * grid)
    integral_2 = torch.sum(spline_2_weights * grid**2 + spline_2_bias * grid)

    # We can then compare them using the L2 norm
    return torch.norm(integral_1 - integral_2, p=2)

if __name__ == "__main__":
    import imageio.v2 as imageio
    import os
    # ==============================
    # Define dataset structure
    # ==============================
    n_samples = 1000
    n_features = 2
    n_classes = 3
    seed = 42

    np.random.seed(seed)

    X, y = maso_dataset(n_samples=n_samples, 
                       n_features=n_features, 
                       n_classes=n_classes)
    
    # Train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)
    
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

    # Load model
    model.load_state_dict(torch.load("src/modelling/maso_model/model.pth"))

    maso = MASO(model, train_loader, _lambda=0.3)

    # ==============================
    # Train model
    # ==============================
    trainer = Trainer(model, 
                     train_dataloader=train_loader, 
                     val_dataloader=val_loader, 
                     lr=0.001, 
                     device='cpu',
                     loss=maso.loss)
    
    # trainer.train(n_epochs=40, save_weights=True)
    # trainer.eval()

    # # Save model
    # torch.save(model.state_dict(), "src/modelling/maso_model/model.pth")
    
    # ==============================
    # We tamper with the model to see effects on the splines
    # ==============================
    perturb_and_gif(model, 
                    maso, "src/modelling/maso_model/weight_perturbation", 
                    layer_idx=4, weight_x_idx=2, weight_y_idx=0)
    
    # pertubation_db_distributions(model, 
    #                              maso, "src/modelling/maso_model/perturbation_analysis")
    
    # plot_during_training(maso, "src/modelling/maso_model/train_weights", 60)

    # plot_with_varying_widths(maso, 
    #                         epochs=60, 
    #                         output_dir="src/modelling/maso_model/width_varying", 
    #                         train_dataloader=train_loader,
    #                         val_dataloader=val_loader,
    #                         widths=[1, 2, 4, 8, 16, 32, 64, 128])
    # print(compare_splines(maso.model.net[0], maso.model.net[0]))