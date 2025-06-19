import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects
import numpy as np
import torch

def visualize_parameter_dampening_stacked(
    state_dict,
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


def visualize_parameter_dampening_original(
    state_dict,
    figsize=(14, 7), # Adjusted for potentially wider layout
    title=None,
    ax=None,
    title_fontsize=16,
    circle_radius=0.15,
    h_spacing_factor=3.0,
    v_spacing_factor=1.2
):
    """
    Visualize neural network parameter dampening with a dynamic, space-optimized layout,
    aligned labels, and vertical separators.
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

    # --- Dynamic Layout Calculation ---
    all_coords = []
    current_x = 0
    circle_diameter = 2 * circle_radius
    h_gap = circle_diameter * h_spacing_factor
    v_spacing = circle_diameter * v_spacing_factor

    # --- NEW: Calculate a single Y position for all labels based on the tallest layer ---
    max_rows = max(layer_sizes)
    max_height = (max_rows - 1) * v_spacing
    label_y_pos = (max_height / 2) + circle_diameter * 4 # Use max height for consistent y-position

    for i in range(1, len(layer_sizes)):
        prev_layer_size = layer_sizes[i - 1]
        current_layer_size = layer_sizes[i]

        # --- Position Weight Matrix ---
        weight_width = (prev_layer_size - 1) * v_spacing
        weight_height = (current_layer_size - 1) * v_spacing
        y_start_w = -(weight_height / 2)
        x_start_w = current_x

        weight_key = f'layer{i-1}.weight'
        weights = state_dict.get(weight_key)
        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().numpy()

            for r in range(current_layer_size):
                for c in range(prev_layer_size):
                    x = x_start_w + c * v_spacing
                    y = y_start_w + r * v_spacing
                    all_coords.append((x, y))
                    color = cmap(weights[r, c])
                    circle = patches.Circle((x, y), circle_radius, facecolor=color, edgecolor='black', linewidth=0.2)
                    ax.add_patch(circle)

        current_x += weight_width + h_gap

        # --- Position Bias Vector ---
        bias_height = (current_layer_size - 1) * v_spacing
        y_start_b = -(bias_height / 2)
        x_start_b = current_x

        bias_key = f'layer{i-1}.bias'
        biases = state_dict.get(bias_key)
        if biases is not None:
            if isinstance(biases, torch.Tensor):
                biases = biases.detach().numpy()

            for r in range(current_layer_size):
                x = x_start_b
                y = y_start_b + r * v_spacing
                all_coords.append((x, y))
                color = cmap(biases[r])
                circle = patches.Circle((x, y), circle_radius, facecolor=color, edgecolor='black', linewidth=0.2)
                ax.add_patch(circle)

        # --- Add Labels using the pre-calculated shared Y position ---
        weight_label_x = x_start_w + weight_width / 2
        ax.text(
            weight_label_x, label_y_pos, # MODIFIED: Using shared y-position
            f'{layer_names[i]}\n$w$: ${current_layer_size} \\times {prev_layer_size}$, $b$: ${current_layer_size} \\times 1$',
            ha='center', va='center', fontsize=11, color='black',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='gray')
        )

        # Update current_x for the next layer's gap
        current_x += circle_diameter

        # --- NEW: Add a transparent vertical divider between layers ---
        if i < len(layer_sizes) - 1:
            divider_x = current_x + h_gap / 2
            ax.axvline(
                x=divider_x, color='lightgray', linestyle='--',
                linewidth=1, alpha=0.7, ymin=0.1, ymax=0.9 # ymin/max constrain line vertically
            )

        # Add the rest of the gap after the divider's position is calculated
        current_x += h_gap

    # --- Auto-set axis limits based on content ---
    if all_coords:
        x_coords, y_coords = zip(*all_coords)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Ensure y-limits accommodate the high labels
        y_max = max(y_max, label_y_pos)

        padding_x = circle_diameter * 3
        padding_y = circle_diameter * 3
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)


    # --- Add Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.08, aspect=50, shrink=0.75)
    cbar.set_label(r'$\beta$ (1 = No Dampening, 0 = Full Dampening)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.invert_xaxis()

    if title:
        ax.set_title(title, fontsize=title_fontsize, y=1.0, pad=20, color='black', weight='bold')

    return fig, ax


def visualize_parameter_dampening(state_dict, 
                                    figsize=(12, 8),
                                    title=None,
                                    ax=None,
                                    title_fontsize=16,
                                    circle_radius=0.15,
                                    h_spacing_factor=5.0,
                                    v_spacing_factor=1.2,
                                    type=None):
    if type == 'stacked':
        fig, ax = visualize_parameter_dampening_stacked(state_dict, figsize, title, ax, title_fontsize, circle_radius, h_spacing_factor, v_spacing_factor)
    else:
        fig, ax = visualize_parameter_dampening_original(state_dict, figsize, title, ax, title_fontsize, circle_radius, h_spacing_factor, v_spacing_factor)

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
    fig, ax = visualize_parameter_dampening(example_state_dict, type='stacked')
    # Save the figure
    plt.show()