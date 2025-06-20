import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects
import numpy as np
import torch
from torch import tensor

def visualize_parameter_dampening_stacked(
    state_dict,
    layer_sizes, # Now an explicit argument
    figsize=(12, 8),
    title=None,
    ax=None,
    title_fontsize=16,
    circle_radius=0.15,
    h_spacing_factor=5.0,
    v_spacing_factor=1.2
):
    """
    Visualize neural network parameter dampening with a precisely aligned,
    stacked two-column layout. Automatically discovers layer keys from the state_dict,
    making it compatible with various naming conventions (e.g., from nn.Sequential).
    """
    # --- Automatically discover and sort layer keys ---
    try:
        weight_keys = sorted([k for k in state_dict.keys() if 'weight' in k])
        bias_keys = sorted([k for k in state_dict.keys() if 'bias' in k])
        
        # Ensure the discovered keys match the expected number of layers
        num_param_layers = len(layer_sizes) - 1
        if len(weight_keys) != num_param_layers or len(bias_keys) != num_param_layers:
            raise ValueError(
                f"Mismatch between layer_sizes ({num_param_layers} parameter layers) "
                f"and found keys in state_dict ({len(weight_keys)} weights, {len(bias_keys)} biases)."
            )
    except Exception as e:
        print(f"Error processing state_dict keys: {e}")
        return None, None


    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output'] # Can be customized if needed

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')
    fig.patch.set_facecolor('white')

    colors = ['#ffa600', '#ff7c43', '#f95d6a', '#d45087', '#a05195', '#665191', '#2f4b7c', '#003f5c']
    cmap = LinearSegmentedColormap.from_list('dampening', colors, N=512)
    
    all_coords = []
    circle_diameter = 2 * circle_radius
    h_gap = circle_diameter * h_spacing_factor
    v_spacing = circle_diameter * v_spacing_factor
    label_box_height = circle_diameter * 4.5

    # --- Block 1: Draw "Hidden 1" ---
    l1_size, l0_size = layer_sizes[1], layer_sizes[0]
    l1_circle_h = (l1_size - 1) * v_spacing
    l1_weight_w = (l0_size - 1) * v_spacing
    l1_x_center = l1_weight_w / 2
    l1_y_center = 0

    weights_l1 = state_dict[weight_keys[0]].detach().numpy()
    x_start_w1 = l1_x_center - l1_weight_w / 2
    y_start_w1 = l1_y_center + l1_circle_h / 2
    for r in range(l1_size):
        for c in range(l0_size):
            x, y = x_start_w1 + c * v_spacing, y_start_w1 - r * v_spacing
            all_coords.append((x, y))
            ax.add_patch(patches.Circle((x, y), circle_radius, facecolor=cmap(weights_l1[r, c]), ec='black', lw=0.2))

    x_start_b1 = x_start_w1 + l1_weight_w + h_gap
    biases_l1 = state_dict[bias_keys[0]].detach().numpy()
    for r in range(l1_size):
        x, y = x_start_b1, y_start_w1 - r * v_spacing
        all_coords.append((x, y))
        ax.add_patch(patches.Circle((x, y), circle_radius, facecolor=cmap(biases_l1[r]), ec='black', lw=0.2))

    master_y_bottom_edge = (l1_y_center - l1_circle_h / 2) - circle_radius
    master_y_top_edge = (l1_y_center + l1_circle_h / 2) + circle_radius
    l1_label_y_center = master_y_top_edge + (label_box_height / 2)
    master_y_top_label = l1_label_y_center + (label_box_height / 2)
    ax.text(x_start_b1 / 2, l1_label_y_center, f'{layer_names[1]}\n$w$: ${l1_size} \\times {l0_size}$, $b$: ${l1_size} \\times 1$',
            ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec='gray'))

    # --- Column 2 Horizontal Position ---
    col2_x_start = x_start_b1 + circle_diameter + h_gap

    # --- Block 2: Draw "Hidden 2" ---
    l2_size, l1_size_ = layer_sizes[2], layer_sizes[1]
    l2_circle_h = (l2_size - 1) * v_spacing
    l2_weight_w = (l1_size_ - 1) * v_spacing
    l2_label_y_center = master_y_top_label - label_box_height / 2
    l2_y_center = l2_label_y_center - label_box_height / 2 - (l2_circle_h / 2)
    l2_x_center = col2_x_start + l2_weight_w / 2
    
    weights_l2 = state_dict[weight_keys[1]].detach().numpy()
    x_start_w2 = l2_x_center - l2_weight_w / 2
    y_start_w2 = l2_y_center + l2_circle_h / 2
    for r in range(l2_size):
        for c in range(l1_size_):
            x, y = x_start_w2 + c * v_spacing, y_start_w2 - r * v_spacing
            all_coords.append((x, y))
            ax.add_patch(patches.Circle((x, y), circle_radius, facecolor=cmap(weights_l2[r, c]), ec='black', lw=0.2))
    x_start_b2 = x_start_w2 + l2_weight_w + h_gap
    biases_l2 = state_dict[bias_keys[1]].detach().numpy()
    for r in range(l2_size):
        x, y = x_start_b2, y_start_w2 - r * v_spacing
        all_coords.append((x, y))
        ax.add_patch(patches.Circle((x, y), circle_radius, facecolor=cmap(biases_l2[r]), ec='black', lw=0.2))
    ax.text(l2_x_center, l2_label_y_center, f'{layer_names[2]}\n$w$: ${l2_size} \\times {l1_size_}$, $b$: ${l2_size} \\times 1$',
            ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec='gray'))
    l2_bottom_edge = (l2_y_center - l2_circle_h / 2) - circle_radius

    # --- Block 3: Draw "Output" ---
    l3_size, l2_size_ = layer_sizes[3], layer_sizes[2]
    l3_circle_h = (l3_size - 1) * v_spacing
    l3_weight_w = (l2_size_ - 1) * v_spacing
    l3_y_center = master_y_bottom_edge + circle_radius + l3_circle_h / 2
    l3_label_y_center = l3_y_center + l3_circle_h / 2 + circle_radius + label_box_height / 2
    l3_x_center = col2_x_start + l2_weight_w / 2
    l3_top_edge_label = l3_label_y_center - label_box_height / 2
    
    weights_l3 = state_dict[weight_keys[2]].detach().numpy()
    x_start_w3 = l3_x_center - l3_weight_w / 2
    y_start_w3 = l3_y_center + l3_circle_h / 2
    for r in range(l3_size):
        for c in range(l2_size_):
            x, y = x_start_w3 + c * v_spacing, y_start_w3 - r * v_spacing
            all_coords.append((x, y))
            ax.add_patch(patches.Circle((x, y), circle_radius, facecolor=cmap(weights_l3[r, c]), ec='black', lw=0.2))
    x_start_b3 = x_start_w3 + l3_weight_w + h_gap
    biases_l3 = state_dict[bias_keys[2]].detach().numpy()
    for r in range(l3_size):
        x, y = x_start_b3, y_start_w3 - r * v_spacing
        all_coords.append((x, y))
        ax.add_patch(patches.Circle((x, y), circle_radius, facecolor=cmap(biases_l3[r]), ec='black', lw=0.2))
    ax.text(l3_x_center, l3_label_y_center, f'{layer_names[3]}\n$w$: ${l3_size} \\times {l2_size_}$, $b$: ${l3_size} \\times 1$',
            ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec='gray'))


    # --- Finalize Plot ---
    if all_coords:
        x_coords, y_coords = zip(*all_coords)
        padding_x = (max(x_coords) - min(x_coords)) * 0.05
        ax.set_xlim(min(x_coords) - padding_x, max(x_coords) + padding_x)
        ax.set_ylim(master_y_bottom_edge - circle_diameter, master_y_top_label + circle_diameter)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.08, aspect=40, shrink=0.5)
    cbar.set_label(r'$\beta$ (1 = No Dampening, 0 = Full Dampening)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.invert_xaxis()

    if title:
        ax.set_title(title, fontsize=title_fontsize, y=0.98, color='black', weight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig, ax

def visualize_parameter_dampening_original(state_dict, figsize=(12, 8), title=None, ax=None, title_fontsize=14):
    """
    Visualize neural network parameter dampening with a structured layout.
    
    Parameters:
    state_dict: Dictionary containing dampening values for each parameter
    figsize: Tuple for figure size (ignored if ax is provided)
    ax: Optional matplotlib axis to plot on (if None, creates new figure)
    """
    
    # Network architecture
    layer_sizes = [2, 16, 8, 3]  # input, hidden1, hidden2, output
    layer_names = ['', 'Input', 'Hidden', 'Output']
    
    # Create figure and axis with better proportions or use provided ax
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False
        
    ax.set_xlim(0, 10.5)
    ax.set_ylim(2, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if created_fig:
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
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, color='black')
    
    # Add subtle grid lines for better visual separation
    for x in [2.75, 7.75]:
        ax.axvline(x=x, color='lightgray', linestyle='--', alpha=0.4, linewidth=0.8)
    
    if created_fig:
        plt.tight_layout()
    return fig, ax

def visualize_parameter_dampening(state_dict, 
                                    figsize=(12, 8),
                                    title=None,
                                    ax=None,
                                    title_fontsize=16,
                                    circle_radius=0.15,
                                    h_spacing_factor=5.0,
                                    v_spacing_factor=1.2,
                                    type=None,
                                    layer_sizes=None):
    if type == 'stacked':
        if layer_sizes is None:
            raise ValueError("layer_sizes must be provided for stacked visualization")
        fig, ax = visualize_parameter_dampening_stacked(state_dict, layer_sizes=layer_sizes, figsize=figsize, title=title, ax=ax, title_fontsize=title_fontsize, circle_radius=circle_radius, h_spacing_factor=h_spacing_factor, v_spacing_factor=v_spacing_factor)
    else:
        fig, ax = visualize_parameter_dampening_original(state_dict, figsize=figsize, title=title, ax=ax, title_fontsize=title_fontsize)

    return fig, ax
# Example usage:
if __name__ == "__main__":
    # Create example state_dict with dampening values
    example_state_dict = {
        'net.0.weight': torch.rand(16, 2),
        'net.0.bias': torch.rand(16),
        'net.2.weight': torch.rand(8, 16),
        'net.2.bias': torch.rand(8),
        'net.4.weight': torch.rand(3, 8),
        'net.4.bias': torch.rand(3)
    }
    

    model_architecture = [2, 16, 8, 3]
    # Create visualization
    fig, ax = visualize_parameter_dampening(example_state_dict, type='original', layer_sizes=model_architecture)
    # Save the figure
    plt.savefig('dampening_visualization.pdf')
    plt.show()