import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class NeuronActivationAnalyzer:
    def __init__(self, model):
        """
        Initialize with a PyTorch model whose activations we want to visualize
        """
        self.model = model
        self.activation_maps = {}
        self.hooks = []
        
        # Register hooks for all layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(
                    lambda module, input, output, name=name: 
                    self.activation_maps.update({name: output.detach()})
                )
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks to avoid memory issues"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self, inputs):
        """
        Get activations for given inputs
        
        Args:
            inputs: Tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary of activations for each layer
        """
        self.activation_maps = {}
        with torch.no_grad():
            _ = self.model(inputs)
        return self.activation_maps
    
    def find_most_active_neurons(self, inputs, layer_name=None, top_k=5):
        """
        Find the neurons that fire the most for given inputs
        
        Args:
            inputs: Tensor of shape (batch_size, input_dim)
            layer_name: Optional name of layer to analyze. If None, use the last layer
            top_k: Number of top neurons to return
            
        Returns:
            Tuple of (neuron_indices, activation_values)
        """
        activations = self.get_activations(inputs)
        
        if layer_name is None:
            # Use the last layer by default
            layer_name = list(activations.keys())[-1]
        
        layer_activations = activations[layer_name]
        
        # For a batch, take the mean activation across all inputs
        if len(layer_activations.shape) > 1 and layer_activations.shape[0] > 1:
            mean_activations = torch.mean(layer_activations, dim=0)
        else:
            mean_activations = layer_activations.squeeze()
        
        # Get top-k neuron indices and their activation values
        top_values, top_indices = torch.topk(mean_activations, min(top_k, len(mean_activations)))
        
        return top_indices.cpu().numpy(), top_values.cpu().numpy()
    
    def compare_activations(self, forget_inputs, retain_inputs, layer_name=None):
        """
        Compare activations between forget and retain datasets
        
        Args:
            forget_inputs: Tensor of shape (batch_size, input_dim) for forget set
            retain_inputs: Tensor of shape (batch_size, input_dim) for retain set
            layer_name: Optional name of layer to analyze. If None, use the last layer
            
        Returns:
            Dictionary with comparison metrics
        """
        forget_activations = self.get_activations(forget_inputs)
        retain_activations = self.get_activations(retain_inputs)
        
        if layer_name is None:
            # Use the last layer by default
            layer_name = list(forget_activations.keys())[-1]
        
        forget_layer_activations = forget_activations[layer_name]
        retain_layer_activations = retain_activations[layer_name]
        
        # Calculate mean activations
        if len(forget_layer_activations.shape) > 1 and forget_layer_activations.shape[0] > 1:
            forget_mean = torch.mean(forget_layer_activations, dim=0)
        else:
            forget_mean = forget_layer_activations.squeeze()
            
        if len(retain_layer_activations.shape) > 1 and retain_layer_activations.shape[0] > 1:
            retain_mean = torch.mean(retain_layer_activations, dim=0)
        else:
            retain_mean = retain_layer_activations.squeeze()
        
        # Calculate activation difference
        activation_diff = forget_mean - retain_mean
        
        # Calculate top different neurons
        abs_diff = torch.abs(activation_diff)
        top_diff_values, top_diff_indices = torch.topk(abs_diff, min(10, len(abs_diff)))
        
        result = {
            'layer_name': layer_name,
            'forget_mean': forget_mean.cpu().numpy(),
            'retain_mean': retain_mean.cpu().numpy(),
            'activation_diff': activation_diff.cpu().numpy(),
            'top_diff_indices': top_diff_indices.cpu().numpy(),
            'top_diff_values': top_diff_values.cpu().numpy(),
        }
        
        return result
    
    def visualize_activations_comparison(self, forget_inputs, retain_inputs, layer_name=None):
        """
        Visualize the comparison between activations for forget and retain datasets
        
        Args:
            forget_inputs: Tensor of shape (batch_size, input_dim) for forget set
            retain_inputs: Tensor of shape (batch_size, input_dim) for retain set
            layer_name: Optional name of layer to analyze. If None, use the last layer
        """
        comparison = self.compare_activations(forget_inputs, retain_inputs, layer_name)
        layer_name = comparison['layer_name']
        forget_mean = comparison['forget_mean']
        retain_mean = comparison['retain_mean']
        activation_diff = comparison['activation_diff']
        top_diff_indices = comparison['top_diff_indices']
        
        # Plot the activations
        plt.figure(figsize=(15, 12))
        
        # Plot mean activations
        plt.subplot(3, 1, 1)
        x = np.arange(len(forget_mean))
        plt.bar(x, forget_mean, alpha=0.5, label='Forget Set', color='red')
        plt.bar(x, retain_mean, alpha=0.5, label='Retain Set', color='blue')
        plt.title(f'Mean Neuron Activations: Layer {layer_name}')
        plt.xlabel('Neuron Index')
        plt.ylabel('Mean Activation')
        plt.legend()
        
        # Plot activation differences
        plt.subplot(3, 1, 2)
        plt.bar(x, activation_diff, color='purple')
        plt.title(f'Activation Difference (Forget - Retain): Layer {layer_name}')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Difference')
        
        # Highlight top different neurons
        plt.subplot(3, 1, 3)
        plt.bar(top_diff_indices, forget_mean[top_diff_indices], alpha=0.5, label='Forget Set', color='red')
        plt.bar(top_diff_indices, retain_mean[top_diff_indices], alpha=0.5, label='Retain Set', color='blue')
        plt.title(f'Top Different Neurons: Layer {layer_name}')
        plt.xlabel('Neuron Index')
        plt.ylabel('Mean Activation')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return top_diff_indices
    
    def analyze_all_layers(self, forget_inputs, retain_inputs):
        """
        Analyze neuron activations across all layers
        
        Args:
            forget_inputs: Tensor of shape (batch_size, input_dim) for forget set
            retain_inputs: Tensor of shape (batch_size, input_dim) for retain set
        """
        # Get all layer names with their activations
        _ = self.get_activations(forget_inputs)
        layer_names = list(self.activation_maps.keys())
        
        # Analyze each layer
        results = []
        for layer_name in layer_names:
            print(f"\nAnalyzing layer: {layer_name}")
            comparison = self.compare_activations(forget_inputs, retain_inputs, layer_name)
            
            # Get top different neurons
            top_diff_indices = comparison['top_diff_indices']
            top_diff_values = comparison['top_diff_values']
            
            print(f"Top neurons with different activations:")
            for i, (idx, val) in enumerate(zip(top_diff_indices[:5], top_diff_values[:5])):
                print(f"  Neuron {idx}: Difference = {val:.4f}")
            
            results.append(comparison)
        
        return results
    
    def visualize_2d_neuron_response(self, X, neuron_index, layer_name=None):
        """
        Visualize how a specific neuron responds to 2D data points
        
        Args:
            X: 2D data points of shape (n_samples, 2)
            neuron_index: Index of neuron to analyze
            layer_name: Optional name of layer to analyze. If None, use the last layer
        """
        if X.shape[1] != 2:
            raise ValueError("This visualization only works for 2D input data")
        
        # Create a grid over the data range
        margin = 0.5
        x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
        y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # Prepare grid points as torch tensors
        grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        
        # Use the same device as the model
        device = next(self.model.parameters()).device
        grid_points = grid_points.to(device)
        
        # Get activations for grid points
        activations = self.get_activations(grid_points)
        
        if layer_name is None:
            layer_name = list(activations.keys())[-1]
        
        # Get activations for the specified neuron
        neuron_activations = activations[layer_name][:, neuron_index].cpu().numpy()
        
        # Reshape back to grid shape
        Z = neuron_activations.reshape(xx.shape)
        
        # Plot the contour
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.8)
        
        # Plot the original data points
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        plt.scatter(X[:, 0], X[:, 1], c='red', edgecolor='k', s=50)
        
        plt.title(f'Activation of Neuron {neuron_index} in Layer {layer_name}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Activation Value')
        plt.tight_layout()
        plt.show()

    def visualize_decision_regions(self, X, y, resolution=0.02):
        """
        Visualize decision regions for a 2D dataset
        
        Args:
            X: Input features (n_samples, 2)
            y: Labels (n_samples,)
            resolution: Grid resolution for visualization
        """
        # Set up the mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                             np.arange(y_min, y_max, resolution))
        
        # Flatten the grid points
        grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        
        # Move to the same device as the model
        device = next(self.model.parameters()).device
        grid_points = grid_points.to(device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            # Handle different model output types
            model_output = self.model(grid_points)
            
            # If model returns a dictionary, try to get the logits/predictions
            if isinstance(model_output, dict):
                # Common keys for predictions in model outputs
                possible_keys = ['logits', 'predictions', 'probs', 'output', 'outputs']
                
                # Try to find a suitable key
                for key in possible_keys:
                    if key in model_output:
                        Z = model_output[key].argmax(dim=1).cpu().numpy()
                        break
                else:
                    # If no known key is found, try the first value that's a tensor
                    for value in model_output.values():
                        if isinstance(value, torch.Tensor) and value.dim() > 1:
                            Z = value.argmax(dim=1).cpu().numpy()
                            break
                    else:
                        raise ValueError("Could not find suitable prediction tensor in model output dictionary")
            else:
                # If model returns a tensor directly
                Z = model_output.argmax(dim=1).cpu().numpy()
        
        # Reshape back to the grid shape
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        
        # Plot the data points
        unique_classes = torch.unique(y)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
        
        for i, cls in enumerate(unique_classes):
            plt.scatter(X[y == cls, 0], X[y == cls, 1], 
                       c=[colors[i]], label=f'Class {cls.item()}',
                       edgecolors='k', alpha=0.8)
        
        plt.title('Decision Regions')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return plt

        
def analyze_forget_retain_activation(model, dataset_forget, dataset_retain, batch_size=None):
    """
    Analyze the activation patterns for forget and retain datasets
    
    Args:
        model: Trained PyTorch model
        dataset_forget: Forget dataset (instance of torch.utils.data.Dataset)
        dataset_retain: Retain dataset (instance of torch.utils.data.Dataset)
        batch_size: Optional batch size for DataLoader
    """
    # Create the analyzer
    analyzer = NeuronActivationAnalyzer(model)
    
    # Get data from datasets
    if batch_size:
        # Use DataLoader if batch_size is specified
        forget_loader = DataLoader(dataset_forget, batch_size=batch_size, shuffle=False)
        retain_loader = DataLoader(dataset_retain, batch_size=batch_size, shuffle=False)
        
        # Get a batch of data from each
        forget_batch = next(iter(forget_loader))
        retain_batch = next(iter(retain_loader))
        
        # Handle different dataset formats
        if len(forget_batch) == 2:
            forget_X, forget_y = forget_batch
        elif len(forget_batch) == 3:  # If dataset returns (X, y, idx)
            forget_X, forget_y, _ = forget_batch
            
        if len(retain_batch) == 2:
            retain_X, retain_y = retain_batch
        elif len(retain_batch) == 3:  # If dataset returns (X, y, idx)
            retain_X, retain_y, _ = retain_batch
    else:
        # Otherwise, get all data at once
        # Handle different dataset item formats
        if len(dataset_forget[0]) == 2:
            forget_X = torch.stack([dataset_forget[i][0] for i in range(len(dataset_forget))])
            # Handle multi-dimensional labels by directly stacking them instead of using torch.tensor
            forget_y_list = [dataset_forget[i][1] for i in range(len(dataset_forget))]
            if isinstance(forget_y_list[0], torch.Tensor):
                forget_y = torch.stack(forget_y_list)
            else:
                forget_y = torch.tensor(forget_y_list)
        elif len(dataset_forget[0]) == 3:  # If dataset returns (X, y, idx)
            forget_X = torch.stack([dataset_forget[i][0] for i in range(len(dataset_forget))])
            # Handle multi-dimensional labels
            forget_y_list = [dataset_forget[i][1] for i in range(len(dataset_forget))]
            if isinstance(forget_y_list[0], torch.Tensor):
                forget_y = torch.stack(forget_y_list)
            else:
                forget_y = torch.tensor(forget_y_list)
        
        # For retain, we might have a lot of data, so let's sample a comparable amount to forget
        retain_indices = np.random.choice(len(dataset_retain), 
                                         min(len(dataset_retain), len(forget_X)*10), 
                                         replace=False)
        
        if len(dataset_retain[0]) == 2:
            retain_X = torch.stack([dataset_retain[i][0] for i in retain_indices])
            # Handle multi-dimensional labels
            retain_y_list = [dataset_retain[i][1] for i in retain_indices]
            if isinstance(retain_y_list[0], torch.Tensor):
                retain_y = torch.stack(retain_y_list)
            else:
                retain_y = torch.tensor(retain_y_list)
        elif len(dataset_retain[0]) == 3:  # If dataset returns (X, y, idx)
            retain_X = torch.stack([dataset_retain[i][0] for i in retain_indices])
            # Handle multi-dimensional labels
            retain_y_list = [dataset_retain[i][1] for i in retain_indices]
            if isinstance(retain_y_list[0], torch.Tensor):
                retain_y = torch.stack(retain_y_list)
            else:
                retain_y = torch.tensor(retain_y_list)
    
    # Convert one-hot encoded labels to class indices if needed
    if len(forget_y.shape) == 2 and forget_y.shape[1] > 1:
        forget_y = torch.argmax(forget_y, dim=1)
    
    if len(retain_y.shape) == 2 and retain_y.shape[1] > 1:
        retain_y = torch.argmax(retain_y, dim=1)
    
    # Use the same device as the model
    device = next(model.parameters()).device
    forget_X = forget_X.to(device)
    retain_X = retain_X.to(device)
    
    print(f"Analyzing activations for {len(forget_X)} forget samples and {len(retain_X)} retain samples")
    
    # Visualize decision regions if data is 2D
    if forget_X.shape[1] == 2:
        print("Visualizing decision regions...")
        # Combine datasets for visualization
        X_all = torch.cat([forget_X, retain_X], dim=0)
        y_all = torch.cat([forget_y, retain_y], dim=0)
        analyzer.visualize_decision_regions(X_all, y_all)
    
    # Analyze each layer
    print("\nAnalyzing neuron activations across all layers...")
    layer_results = analyzer.analyze_all_layers(forget_X, retain_X)
    
    # Get all hidden layer names
    hidden_layer_names = [name for name in analyzer.activation_maps.keys() 
                        if "weight" not in name.lower() and "bias" not in name.lower()]
    
    # Analyze each hidden layer
    for i, layer_name in enumerate(hidden_layer_names):
        print(f"\n{'='*20} Layer {i+1}/{len(hidden_layer_names)}: {layer_name} {'='*20}")
        
        # Visualize activation comparison for this layer
        print(f"Visualizing activation comparison for layer: {layer_name}")
        top_diff_neurons = analyzer.visualize_activations_comparison(forget_X, retain_X, layer_name)
        
        # If data is 2D, visualize how the most different neuron responds to data
        if forget_X.shape[1] == 2 and len(top_diff_neurons) > 0:
            most_diff_neuron = top_diff_neurons[0]
            print(f"Visualizing how neuron {most_diff_neuron} responds to data...")
            X_all = torch.cat([forget_X, retain_X], dim=0).cpu()
            analyzer.visualize_2d_neuron_response(X_all, most_diff_neuron, layer_name)
    
    # Clean up
    analyzer.remove_hooks()
    
    return layer_results


# Example usage with your code:
"""
# After you've trained your model, you can analyze the activations:

# Assuming model is your trained NeuralNet model
# and dataset_forget and dataset_retain are your SyntheticDataset instances

analysis_results = analyze_forget_retain_activation(
    model=model,  # Your trained neural network
    dataset_forget=dataset_forget,  # The forget dataset
    dataset_retain=dataset_retain,  # The retain dataset
    batch_size=None  # Set to None to use all data, or specify a batch size
)

# The analysis results contain information about each layer's activation differences
"""


def load_and_analyze_model(model_path, dataset_forget, dataset_retain):
    """
    Load a saved model and analyze its activations
    
    Args:
        model_path: Path to the saved model
        dataset_forget: Forget dataset
        dataset_retain: Retain dataset
    """
    # Load model
    model = torch.load(model_path)
    model.eval()  # Set to evaluation mode
    
    # Analyze activations
    results = analyze_forget_retain_activation(model, dataset_forget, dataset_retain)
    
    return results