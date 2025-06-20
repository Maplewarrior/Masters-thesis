import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects
import numpy as np
import torch
from torch import tensor
import torch.nn as nn

def get_architecture_from_state_dict(state_dict: dict) -> list:
    """
    Dynamically inspects a PyTorch state_dict and deduces the network architecture.

    This function is designed for sequential models where weight keys follow a
    sortable naming convention (e.g., 'layer.0.weight', 'net.2.weight').

    Args:
        state_dict (dict): The PyTorch state_dict object.

    Returns:
        list: A list of integers representing the layer sizes,
              e.g., [input_features, hidden1_features, ..., output_features].
              Returns an empty list if no weight keys are found.
    """
    # 1. Find all keys that correspond to weights.
    weight_keys = [k for k in state_dict.keys() if 'weight' in k]

    if not weight_keys:
        print("Warning: No 'weight' keys found in the state_dict.")
        return []

    # 2. Sort keys to ensure correct layer order.
    #    This sort is robust for multi-digit layer numbers (e.g., 'net.10.weight').
    try:
        weight_keys.sort(key=lambda x: int(x.split('.')[1]))
    except (IndexError, ValueError):
        print("Warning: Could not sort keys by layer number. Using simple alphabetical sort.")
        weight_keys.sort()

    # 3. Build the architecture list from the tensor shapes.
    arch = []
    
    # The input feature size of the network is the 'in_features' of the first layer.
    # For a weight tensor, shape is (out_features, in_features).
    first_layer_shape = state_dict[weight_keys[0]].shape
    arch.append(first_layer_shape[1])

    # The subsequent sizes are the 'out_features' of each layer.
    for key in weight_keys:
        shape = state_dict[key].shape
        arch.append(shape[0])

    return arch



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
    Visualize neural network parameter dampening with a precisely aligned,
    stacked two-column layout, including vertical and horizontal separator lines.
    """
    # Network architecture
    layer_sizes = [2, 16, 8, 3]
    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')
    fig.patch.set_facecolor('white')

    # Custom colormap
    colors = ['#ffa600', '#ff7c43', '#f95d6a', '#d45087', '#a05195', '#665191', '#2f4b7c', '#003f5c']
    cmap = LinearSegmentedColormap.from_list('dampening', colors, N=512)

    # --- Layout Constants ---
    all_coords = []
    circle_diameter = 2 * circle_radius
    h_gap = circle_diameter * h_spacing_factor
    v_spacing = circle_diameter * v_spacing_factor
    label_box_height = circle_diameter * 4.5

    # --- Block 1: Draw "Hidden 1" (The Reference Column) ---
    l1_idx, l1_size, l0_size = 1, layer_sizes[1], layer_sizes[0]
    l1_circle_h = (l1_size - 1) * v_spacing
    l1_weight_w = (l0_size - 1) * v_spacing
    l1_x_center = l1_weight_w / 2
    l1_y_center = 0

    weights_l1 = state_dict.get(f'layer{l1_idx-1}.weight').detach().numpy()
    x_start_w1 = l1_x_center - l1_weight_w / 2
    y_start_w1 = l1_y_center + l1_circle_h / 2
    for r in range(l1_size):
        for c in range(l0_size):
            x, y = x_start_w1 + c * v_spacing, y_start_w1 - r * v_spacing
            all_coords.append((x, y))
            ax.add_patch(patches.Circle((x, y), circle_radius, facecolor=cmap(weights_l1[r, c]), ec='black', lw=0.2))

    x_start_b1 = x_start_w1 + l1_weight_w + h_gap
    biases_l1 = state_dict.get(f'layer{l1_idx-1}.bias').detach().numpy()
    for r in range(l1_size):
        x, y = x_start_b1, y_start_w1 - r * v_spacing
        all_coords.append((x, y))
        ax.add_patch(patches.Circle((x, y), circle_radius, facecolor=cmap(biases_l1[r]), ec='black', lw=0.2))

    # --- Establish Master Vertical Boundaries from Column 1 ---
    # Using circle radius for precise edge calculation
    master_y_bottom_edge = (l1_y_center - l1_circle_h / 2) - circle_radius
    master_y_top_edge = (l1_y_center + l1_circle_h / 2) + circle_radius
    l1_label_y_center = master_y_top_edge + (label_box_height / 2)
    master_y_top_label = l1_label_y_center + (label_box_height / 2)
    ax.text(x_start_b1 / 2, l1_label_y_center, f'{layer_names[1]}\n$w$: ${l1_size} \\times {l0_size}$, $b$: ${l1_size} \\times 1$',
            ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec='gray'))

    # --- Column 2 Horizontal Position ---
    col2_x_start = x_start_b1 + circle_diameter + h_gap

    # --- Block 2: Draw "Hidden 2" (Top of Column 2) ---
    l2_idx, l2_size, l1_size_ = 2, layer_sizes[2], layer_sizes[1]
    l2_circle_h = (l2_size - 1) * v_spacing
    l2_weight_w = (l1_size_ - 1) * v_spacing
    l2_label_y_center = master_y_top_label - label_box_height / 2
    l2_y_center = l2_label_y_center - label_box_height / 2 - (l2_circle_h / 2)
    l2_x_center = col2_x_start + l2_weight_w / 2
    
    weights_l2 = state_dict.get(f'layer{l2_idx-1}.weight').detach().numpy()
    x_start_w2 = l2_x_center - l2_weight_w / 2
    y_start_w2 = l2_y_center + l2_circle_h / 2
    for r in range(l2_size):
        for c in range(l1_size_):
            x, y = x_start_w2 + c * v_spacing, y_start_w2 - r * v_spacing
            all_coords.append((x, y))
            ax.add_patch(patches.Circle((x, y), circle_radius, facecolor=cmap(weights_l2[r, c]), ec='black', lw=0.2))
    x_start_b2 = x_start_w2 + l2_weight_w + h_gap
    biases_l2 = state_dict.get(f'layer{l2_idx-1}.bias').detach().numpy()
    for r in range(l2_size):
        x, y = x_start_b2, y_start_w2 - r * v_spacing
        all_coords.append((x, y))
        ax.add_patch(patches.Circle((x, y), circle_radius, facecolor=cmap(biases_l2[r]), ec='black', lw=0.2))
    ax.text(l2_x_center, l2_label_y_center, f'{layer_names[2]}\n$w$: ${l2_size} \\times {l1_size_}$, $b$: ${l2_size} \\times 1$',
            ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec='gray'))
    l2_bottom_edge = (l2_y_center - l2_circle_h / 2) - circle_radius

    # --- Block 3: Draw "Output" (Bottom of Column 2) ---
    l3_idx, l3_size, l2_size_ = 3, layer_sizes[3], layer_sizes[2]
    l3_circle_h = (l3_size - 1) * v_spacing
    l3_weight_w = (l2_size_ - 1) * v_spacing
    l3_y_center = master_y_bottom_edge + circle_radius + l3_circle_h / 2
    l3_label_y_center = l3_y_center + l3_circle_h / 2 + circle_radius + label_box_height / 2
    l3_x_center = col2_x_start + l2_weight_w / 2
    l3_top_edge_label = l3_label_y_center - label_box_height / 2
    
    weights_l3 = state_dict.get(f'layer{l3_idx-1}.weight').detach().numpy()
    x_start_w3 = l3_x_center - l3_weight_w / 2
    y_start_w3 = l3_y_center + l3_circle_h / 2
    for r in range(l3_size):
        for c in range(l2_size_):
            x, y = x_start_w3 + c * v_spacing, y_start_w3 - r * v_spacing
            all_coords.append((x, y))
            ax.add_patch(patches.Circle((x, y), circle_radius, facecolor=cmap(weights_l3[r, c]), ec='black', lw=0.2))
    x_start_b3 = x_start_w3 + l3_weight_w + h_gap
    biases_l3 = state_dict.get(f'layer{l3_idx-1}.bias').detach().numpy()
    for r in range(l3_size):
        x, y = x_start_b3, y_start_w3 - r * v_spacing
        all_coords.append((x, y))
        ax.add_patch(patches.Circle((x, y), circle_radius, facecolor=cmap(biases_l3[r]), ec='black', lw=0.2))
    ax.text(l3_x_center, l3_label_y_center, f'{layer_names[3]}\n$w$: ${l3_size} \\times {l2_size_}$, $b$: ${l3_size} \\times 1$',
            ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec='gray'))

    # --- Add Separator Lines ---
    # MODIFICATION 1: Trim vertical line to the exact height of the first layer's circles.
    # vline_x = (x_start_b1 + circle_diameter + col2_x_start) / 2
    # ax.vlines(x=vline_x, ymin=master_y_bottom_edge, ymax=master_y_top_edge,
    #           color='gray', linestyle='--', linewidth=1, alpha=0.6)

    # MODIFICATION 2: Center horizontal line between the bottom edge of H2 circles and top edge of O label.
    # hline_y = (l2_bottom_edge + l3_top_edge_label) / 2
    # ax.hlines(hline_y, xmin=col2_x_start, xmax=(x_start_b2 + circle_diameter),
    #           color='gray', linestyle='--', linewidth=1, alpha=0.6)

    # --- Finalize Plot ---
    if all_coords:
        x_coords, y_coords = zip(*all_coords)
        padding_x = (max(x_coords) - min(x_coords)) * 0.05
        ax.set_xlim(min(x_coords) - padding_x, max(x_coords) + padding_x)
        ax.set_ylim(master_y_bottom_edge - circle_diameter, master_y_top_label + circle_diameter)

    # Add Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.08, aspect=40, shrink=0.5)
    cbar.set_label(r'$\beta$ (1 = No Dampening, 0 = Full Dampening)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.invert_xaxis()

    if title:
        ax.set_title(title, fontsize=title_fontsize, y=0.98, color='black', weight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
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
        layer_sizes = get_architecture_from_state_dict(state_dict)
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
    

    # Create visualization
    fig, ax = visualize_parameter_dampening(example_state_dict, type='stacked')
    # Save the figure
    plt.savefig('dampening_visualization.pdf')
    plt.show()