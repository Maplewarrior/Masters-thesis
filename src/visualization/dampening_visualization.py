import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects
import numpy as np
import torch

def visualize_parameter_dampening(
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
    # Save the figure
    plt.show()