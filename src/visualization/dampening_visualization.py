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
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(0.5, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Create custom colormap with better contrast
    colors = ['#8B0000', '#B22222', '#DC143C', '#FF6347', '#FFA500', '#FFD700', '#E0E6FF', '#87CEEB', '#4169E1', '#191970']
    n_bins = 512
    cmap = LinearSegmentedColormap.from_list('dampening', colors, N=n_bins)
    
    # Position parameters with better spacing
    layer_x_positions = [0.05, 2.5, 6.5, 10.5]
    weight_circle_radius = 0.08
    bias_circle_radius = 0.1
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
                    weight_spacing = 0.17
                elif layer_idx == 2:  # Second hidden layer  
                    weight_x = x_pos - 1.5
                    weight_spacing = 0.19
                else:  # Output layer
                    weight_x = x_pos - 1.5
                    weight_spacing = 0.18
                
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
                    bias_x = x_pos - 0.5
                elif layer_idx == 2:  # Second hidden layer
                    bias_x = x_pos + 0.3
                else:  # Output layer
                    bias_x = x_pos - 0.3
                    
                bias_spacing = 0.22 if layer_size > 8 else 0.23
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
                        ax.text(x, y, 'B', ha='center', va='center', 
                               fontsize=6, fontweight='bold', color='white',
                               path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=1, foreground='black')])
        
        
        # Add weight and bias labels for non-input layers with better positioning
        if layer_idx > 0:
            if layer_idx == 1:
                weight_label_x = x_pos - 1.8
                bias_label_x = x_pos - 0.5
            elif layer_idx == 2:
                weight_label_x = x_pos - 1.5
                bias_label_x = x_pos + 0.17
            else:
                weight_label_x = x_pos - 1.5
                bias_label_x = x_pos - 0.3
            layer_label_x = (weight_label_x + bias_label_x) / 2
                
            # Add layer labels with better styling
            ax.text(layer_label_x, 7.0, layer_name, ha='center', va='center', 
               fontsize=12, fontweight='bold', color='#2F4F4F',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8, edgecolor='gray'))
        
            ax.text(weight_label_x, 6.2, f'Weights\n{layer_size}×{layer_sizes[layer_idx-1]}', 
                   ha='center', va='center', fontsize=10, style='italic', color='#4A4A4A',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
            ax.text(bias_label_x, 6.2, f'Biases\n{layer_size}×1', 
                   ha='center', va='center', fontsize=10, style='italic', color='#4A4A4A',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    

    # Add colorbar with tighter layout next to the output layer
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])

    # Reduce whitespace by tweaking pad and fraction
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=12, pad=-0.1, fraction=0.03)

    # Set label and tick size
    cbar.set_label('Dampening Factor\n(1 = No Dampening, 0 = Full Dampening)', 
                rotation=270, labelpad=28, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    # Add title with better styling
    plt.suptitle('SSD Parameter Dampening', 
                fontsize=18, fontweight='bold', y=0.95, color='#2F4F4F')
    
    # # Add legend with updated elements
    # legend_elements = [
    #     plt.Line2D([], [], color='white', marker='o', markerfacecolor='lightgray', label='Weight Parameters'),
    #     plt.Line2D([], [], color='white', marker='o', markerfacecolor='darkgray', label='Bias Parameters')
    # ]
    
    # ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 0.98), 
    #          frameon=True, fancybox=True, shadow=False, fontsize=9)
    
    # Add explanation text with better styling
    explanation = ("Matrix grids showing parameter dampening factors with color-coded circles.\n"
                  "Cold colors indicate no dampening, warm colors indicate high dampening.")
    ax.text(6, 1.2, explanation, ha='center', va='center', fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F8FF", alpha=0.9, 
                    edgecolor='#4682B4', linewidth=1.5))
    
    # Add subtle grid lines for better visual separation
    for x in [2.75, 7.75]:
        ax.axvline(x=x, color='lightgray', linestyle='--', alpha=0.3, linewidth=0.8)
    
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
    fig, ax = visualize_network_dampening(example_state_dict)
    plt.show()