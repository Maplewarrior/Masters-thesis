import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects
import numpy as np
import torch

def visualize_parameter_dampening(state_dict, figsize=(10, 6)):
    """
    Visualize neural network parameter dampening with a structured layout.
    
    Parameters:
    state_dict: Dictionary containing dampening values for each parameter
    figsize: Tuple for figure size
    """
    
    # Network architecture
    layer_sizes = [2, 16, 8, 3]  # input, hidden1, hidden2, output
    layer_names = ['', 'Input', 'Hidden', 'Output']
    
    # Create figure and axis with better proportions
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(2, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Create custom colormap with better contrast
    colors = ['#ffa600', '#ff7c43', '#f95d6a', '#d45087', '#a05195', '#665191', '#2f4b7c', '#003f5c']
    n_bins = 512
    cmap = LinearSegmentedColormap.from_list('dampening', colors, N=n_bins)
    
    # Position parameters with better spacing
    layer_x_positions = [0.05, 3, 6.5, 10.7]
    weight_circle_radius = 0.08
    bias_circle_radius = 0.08
    def calculate_grid_layout(rows, cols, center_x, center_y, spacing=0.12):
        """Calculate positions for a grid of circles"""
        positions = []
        start_x = center_x - (cols - 1) * spacing / 2
        start_y = center_y + (rows - 1) * spacing / 2
        
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * spacing
                y = start_y - row * spacing
                positions.append((x, y))
        return positions
    
    # Draw nodes for each layer
    for layer_idx, (layer_size, layer_name, x_pos) in enumerate(zip(layer_sizes, layer_names, layer_x_positions)):
        
        # Draw weight matrices
        if layer_idx > 0:
            # Get weight matrix
            
            weight_keys = [k for k in state_dict.keys() if 'weight' in k and 'bias' not in k]
            if len(weight_keys) > layer_idx - 1:
                weights = state_dict[weight_keys[layer_idx-1]]
            
            if weights is not None:
                if isinstance(weights, torch.Tensor):
                    weights = weights.detach().numpy()
                
                # Draw weight matrix as a grid with better positioning
                prev_layer_size = layer_sizes[layer_idx - 1]
                current_layer_size = layer_size
                
                # Adjust weight position based on layer to prevent overlap
                if layer_idx == 1:  # First hidden layer
                    weight_x = x_pos - 1.8
                    weight_spacing = 0.2
                elif layer_idx == 2:  # Second hidden layer  
                    weight_x = x_pos - 1.5
                    weight_spacing = 0.2
                else:  # Output layer
                    weight_x = x_pos - 1.5
                    weight_spacing = 0.2
                
                # Calculate grid positions for weight matrix
                grid_positions = calculate_grid_layout(current_layer_size, prev_layer_size, 
                                                     weight_x, 4, spacing=weight_spacing)
                
                for target_idx in range(current_layer_size):
                    for source_idx in range(prev_layer_size):
                        pos_idx = target_idx * prev_layer_size + source_idx
                        if pos_idx < len(grid_positions):
                            x, y = grid_positions[pos_idx]
                            
                            if target_idx < weights.shape[0] and source_idx < weights.shape[1]:
                                weight_val = weights[target_idx, source_idx]
                                color = cmap(weight_val)
                                circle = patches.Circle((x, y), weight_circle_radius, 
                                                      facecolor=color, edgecolor='black', 
                                                      linewidth=0.2, alpha=0.9)
                                ax.add_patch(circle)
            
            # Draw bias vector with better positioning
            bias_key = f'layer{layer_idx-1}.bias'
            possible_bias_keys = [
                f'layer{layer_idx-1}.bias',
                f'{layer_idx-1}.bias',
                f'linear{layer_idx}.bias',
                f'fc{layer_idx}.bias',
                f'layers.{layer_idx-1}.bias'
            ]
            
            bias_values = None
            for key in possible_bias_keys:
                if key in state_dict:
                    bias_values = state_dict[key]
                    break
            
            if bias_values is None:
                bias_keys = [k for k in state_dict.keys() if 'bias' in k]
                if len(bias_keys) >= layer_idx:
                    bias_values = state_dict[bias_keys[layer_idx-1]]
            
            if bias_values is not None:
                if isinstance(bias_values, torch.Tensor):
                    bias_values = bias_values.detach().numpy()
                
                # Draw bias vector with proper spacing to avoid overlap
                if layer_idx == 1:  # First hidden layer
                    bias_x = x_pos - 1.2
                elif layer_idx == 2:  # Second hidden layer
                    bias_x = x_pos + 0.4
                else:  # Output layer
                    bias_x = x_pos - 0.4
                    
                bias_spacing = 0.2
                bias_positions = calculate_grid_layout(layer_size, 1, bias_x, 4, spacing=bias_spacing)
                
                for i, bias_val in enumerate(bias_values):
                    if i < len(bias_positions):
                        x, y = bias_positions[i]
                        color = cmap(bias_val)
                        circle = patches.Circle((x, y), bias_circle_radius, 
                                              facecolor=color, edgecolor='black', 
                                              linewidth=0.2, alpha=0.9)
                        ax.add_patch(circle)
                        
                        # Add 'B' for bias with better styling
                        # ax.text(x, y, '', ha='center', va='center', 
                        #        fontsize=6, fontweight='bold', color='white',
                        #        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=1, foreground='black')])
        
        
        # Add weight and bias labels for non-input layers with better positioning
        if layer_idx > 0:
            if layer_idx == 1:
                weight_label_x = x_pos - 1.9
                bias_label_x = x_pos - 0.5
            elif layer_idx == 2:
                weight_label_x = x_pos - 1.5
                bias_label_x = x_pos + 0.4
            else:
                weight_label_x = x_pos - 1.5
                bias_label_x = x_pos - 0.3
            layer_label_x = (weight_label_x + bias_label_x) / 2
                
            # Add layer labels with better styling
            ax.text(weight_label_x, 6.8, f'{layer_name}\n$w$: {layer_size}×{layer_sizes[layer_idx-1]}  $b$: {layer_size}×1', 
                   ha='center', va='center', fontsize=11, color='black',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='gray'))
    

    # Add colorbar with horizontal orientation at the bottom
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])

    # Create horizontal colorbar at the bottom
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.8, aspect=20, pad=0.1, fraction=0.05)

    # Set label and tick size
    cbar.set_label(f'$\\beta$\n(1 = No Dampening, 0 = Full Dampening)', 
                fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    cbar.ax.invert_xaxis()
    
    # Add title with better styling
    plt.suptitle('SSD Parameter Dampening', 
                fontsize=16, y=0.95, color='black')
    
    # Add subtle grid lines for better visual separation
    for x in [2.75, 7.75]:
        ax.axvline(x=x, color='lightgray', linestyle='--', alpha=0.4, linewidth=0.8)
    
    plt.tight_layout()
    return fig, ax

# Example usage:
if __name__ == "__main__":
    # Create example state_dict with dampening values
    example_state_dict = {
        'layer0.weight': torch.tensor([[0.9, 0.8], [0.7, 1.0], [0.6, 0.9], [1.0, 0.5],
                                      [0.8, 0.9], [0.7, 0.8], [0.9, 1.0], [0.6, 0.7],
                                      [1.0, 0.8], [0.5, 0.9], [0.8, 0.6], [0.9, 1.0],
                                      [0.7, 0.8], [1.0, 0.9], [0.6, 0.7], [0.8, 1.0]]),
        'layer0.bias': torch.tensor([0.9, 0.8, 0.7, 1.0, 0.6, 0.9, 1.0, 0.5, 
                                    0.8, 0.9, 0.7, 0.8, 0.9, 1.0, 0.6, 0.7]),
        'layer1.weight': torch.tensor([[0.8, 0.9, 0.7, 1.0, 0.6, 0.8, 0.9, 0.7, 
                                       0.5, 0.8, 0.9, 1.0, 0.6, 0.7, 0.8, 0.9],
                                      [0.7, 0.8, 0.9, 0.6, 1.0, 0.7, 0.8, 0.5, 
                                       0.9, 0.6, 0.8, 0.7, 1.0, 0.9, 0.8, 0.6],
                                      [0.9, 0.6, 0.8, 0.7, 0.5, 1.0, 0.8, 0.9, 
                                       0.7, 0.8, 0.6, 0.9, 0.8, 0.7, 1.0, 0.5],
                                      [0.8, 0.7, 0.9, 0.8, 0.6, 0.7, 1.0, 0.8, 
                                       0.9, 0.5, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8],
                                      [0.6, 0.9, 0.7, 0.8, 0.9, 0.6, 0.8, 0.7, 
                                       1.0, 0.8, 0.9, 0.5, 0.7, 0.8, 0.9, 0.6],
                                      [0.7, 0.8, 0.6, 0.9, 0.8, 0.7, 0.6, 1.0, 
                                       0.8, 0.9, 0.7, 0.8, 0.5, 0.6, 0.8, 0.9],
                                      [0.8, 0.6, 0.9, 0.7, 0.8, 0.9, 0.7, 0.6, 
                                       0.8, 0.7, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                                      [0.9, 0.7, 0.8, 0.6, 0.7, 0.8, 0.9, 0.8, 
                                       0.6, 1.0, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8]]),
        'layer1.bias': torch.tensor([0.8, 0.9, 0.7, 1.0, 0.6, 0.8, 0.9, 0.5]),
        'layer2.weight': torch.tensor([[0.9, 0.8, 0.7, 1.0, 0.6, 0.9, 0.8, 0.7],
                                      [0.8, 0.7, 0.9, 0.6, 1.0, 0.8, 0.7, 0.9],
                                      [0.7, 0.9, 0.8, 0.7, 0.8, 0.6, 1.0, 0.8]]),
        'layer2.bias': torch.tensor([0.9, 0.8, 0.7])
    }
    
    # Create visualization
    fig, ax = visualize_parameter_dampening(example_state_dict)
    plt.show()