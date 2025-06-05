import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def plot_accuracy_metrics(results, retrained_model_results, figsize=(15, 5)):
    """
    Plot scrubbing metrics showing retain, forget, and val accuracies along with their losses.
    
    Args:
        results: Dictionary containing metrics with 'retain', 'forget', 'val' keys
        retrained_model_results: Dictionary containing metrics for the retrained model
        figsize: Tuple for figure size (width, height)
    
    Returns:
        matplotlib.figure.Figure: Figure containing accuracy and loss plots
    """
    fig_acc, axes_acc = plt.subplots(1, 3, figsize=figsize)
    fig_acc.suptitle('Scrubbing Results Across Epochs', fontsize=16, fontweight='bold')
    
    metrics = ['retain', 'forget', 'val']
    metric_titles = ['Retain', 'Forget', 'Validation']
    colors = ['#f95d6a','#665191','#ffa600']
    
    # Create empty lists to store line objects for the legend
    lines = []
    labels = []
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes_acc[i]
        ax2 = ax.twinx()  # Create a second y-axis sharing the same x-axis
        
        acc_values = results[metric]['acc']
        loss_values = results[metric]['loss']
        epochs = range(1, len(acc_values) + 1)

        # Plot accuracy values (left y-axis)
        # Make a horizontal line at the retrained model accuracy
        retrain_acc_value = retrained_model_results[metric]['acc'][0]
        retrain_line = ax.axhline(y=retrain_acc_value, color=colors[0], linestyle='--', alpha=0.8, linewidth=1.5)
        
        # Plot accuracy values
        scrub_line = ax.plot(epochs, acc_values, 
                           marker='o', linewidth=2, markersize=4,
                           color=colors[1], alpha=0.7,
                           label='Accuracy')[0]
        
        # Plot loss values (right y-axis)
        # retrain_loss_value = retrained_model_results[metric]['loss'][0]
        # retrain_loss_line = ax2.axhline(y=retrain_loss_value, color=colors[0], linestyle='--', alpha=0.4, linewidth=1.5)
        
        loss_line = ax2.plot(epochs, loss_values,
                           marker='s', linewidth=2, markersize=4,
                           color=colors[2], alpha=0.7,
                           label='Loss')[0]
        
        # Only store legend handles from the first plot
        if i == 0:
            lines.extend([retrain_line, scrub_line, loss_line])
            labels.extend(['Retrained Model', 'Unlearned Model (SCRUB) - Acc', 'Unlearned Model (SCRUB) - Loss'])

        # Set labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax2.set_ylabel('Loss')
        ax.set_title(f'{title} Metrics')
        
        # Set grid and limits
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        # Set reasonable y-limits for loss based on your typical values
        # You might need to adjust these based on your actual loss ranges
        ax2.set_ylim(bottom=0)
        
        # Set colors for the y-axis labels
        ax.yaxis.label.set_color(colors[1])
        ax2.yaxis.label.set_color(colors[2])
    
    # Add a single legend at the bottom center of the figure
    fig_acc.legend(lines, labels, 
                  loc='center', 
                  bbox_to_anchor=(0.5, 0.02),
                  ncol=3,
                  frameon=False)
    
    # Adjust the subplot parameters to give specified padding
    plt.subplots_adjust(bottom=0.2, wspace=0.3)
    
    return fig_acc

def plot_mia_metrics(mia_results, retrained_model_results, figsize=(5, 5)):
    """
    Plot MIA probabilities across epochs.
    
    Args:
        mia_results: Dictionary containing MIA probabilities
        figsize: Tuple for figure size (width, height)
    
    Returns:
        matplotlib.figure.Figure: Figure containing MIA plot
    """

    colors = ['#f95d6a','#665191','#ffa600']
    fig_mia, ax_mia = plt.subplots(1, 1, figsize=figsize)
    fig_mia.suptitle('MIA Probability Across Epochs', fontsize=16, fontweight='bold')
    
    mia_values = mia_results
    epochs = range(1, len(mia_values) + 1)

    retrain_mia_value = retrained_model_results['mia']['acc'][0]
    ax_mia.axhline(y=retrain_mia_value, color=colors[0], linestyle='--', alpha=0.8, linewidth=1.5, label='Retrained Model')
    
    ax_mia.plot(epochs, mia_values,
                marker='o', linewidth=2, markersize=4,
                color=colors[2], alpha=0.7,
                label='Unlearned Model (SCRUB)')
    

    ax_mia.set_xlabel('Epoch')
    ax_mia.set_ylabel('MIA Probability')
    ax_mia.grid(True, alpha=0.3)
    ax_mia.set_ylim(0, 1.1)
    ax_mia.legend()
    
    plt.tight_layout()
    return fig_mia


def plot_js_divergence_metrics(js_divergence_results, figsize=(5, 5)):
    """
    Plot MIA probabilities across epochs.
    
    Args:
        mia_results: Dictionary containing MIA probabilities
        figsize: Tuple for figure size (width, height)
    
    Returns:
        matplotlib.figure.Figure: Figure containing MIA plot
    """
    datasets = ['retain', 'forget', 'val']
    dataset_labels = ['Retain', 'Forget', 'Validation']

    colors = ['#f95d6a','#665191','#ffa600', '#dd5182']
    fig_js, ax_js = plt.subplots(1, 1, figsize=figsize)
    fig_js.suptitle('JS Divergence between Retrained and Unlearned Model Across Epochs', fontsize=16, fontweight='bold')
    
    for i, dataset in enumerate(datasets):
        js_div_values = js_divergence_results[dataset]
        epochs = range(1, len(js_div_values) + 1)

        ax_js.plot(epochs, js_div_values,
                    marker='o', linewidth=2, markersize=4,
                    color=colors[i], alpha=0.7,
                    label=dataset_labels[i])
    

    ax_js.set_xlabel('Epoch')
    ax_js.set_ylabel('JS Divergence')
    ax_js.grid(True, alpha=0.3)
    ax_js.set_ylim(0, 1.1)
    ax_js.legend()
    
    plt.tight_layout()
    return fig_js

def plot_js_div_vs_retain_acc(results, retrained_model_results, figsize=(7, 7)):
    """
    Plot JS Divergence on the forget set vs. Retain Accuracy.

    Args:
        results: Dictionary containing metrics.
        retrained_model_results: Dictionary containing metrics for the retrained model.
        figsize: Tuple for figure size (width, height).

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle('JS Divergence (Forget) vs. Retain Accuracy', fontsize=16, fontweight='bold')

    # Color for retrained model, SCRUB points will use a colormap
    retrained_model_color = '#f95d6a'
   
    # Define your custom color palette
    custom_colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
    custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", custom_colors)


    # SCRUB model results
    retain_acc_values = results['retain']['acc']
    js_div_forget_values = results['js_div']['forget']
    epochs = range(1, len(retain_acc_values) + 1)

    scatter = ax.scatter(retain_acc_values, js_div_forget_values,
                         marker='o', s=36,
                         c=epochs, cmap=custom_cmap,  # Color by epoch using viridis colormap
                         alpha=0.8, label='Unlearned Model (SCRUB)')

    # Retrained model result (single point)
    retrained_retain_acc = retrained_model_results['retain']['acc'][0]
    if 'js_div' in retrained_model_results and 'forget' in retrained_model_results['js_div'] and retrained_model_results['js_div']['forget']:
        retrained_js_div_forget = retrained_model_results['js_div']['forget'][0]
        ax.scatter(retrained_retain_acc, retrained_js_div_forget,
                   marker='*', s=150, color=retrained_model_color,
                   label='Retrained Model', zorder=5)
    else:
        # If no JS-div for retrained model, we can just mark its accuracy level
        # or simply not plot a specific point for it in this type of plot.
        # For now, let's just note its retain accuracy as a vertical line for context if no JS Div.
        ax.axvline(x=retrained_retain_acc, color=retrained_model_color, linestyle='--', alpha=0.7, linewidth=1.5,
                   label=f'Retrained Model Retain Acc ({retrained_retain_acc:.2f})')

    ax.set_xlabel('Retain Accuracy')
    ax.set_ylabel('JS Divergence between unlearned and retrained model (Forget Set)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(bottom=0) # JS Divergence is non-negative

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Epoch')

    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    return fig

def plot_mia_vs_retain_acc(results, retrained_model_results, figsize=(7, 7)):
    """
    Plot MIA Probability vs. Retain Accuracy as a scatter plot.

    Args:
        results: Dictionary containing metrics, including 'mia' and 'retain'.
        retrained_model_results: Dictionary containing metrics for the retrained model.
        figsize: Tuple for figure size (width, height).

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle('MIA Probability vs. Retain Accuracy', fontsize=16, fontweight='bold')

    # Color for retrained model, SCRUB points will use a colormap
    retrained_model_color = '#f95d6a'

    # Define your custom color palette
    custom_colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
    custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", custom_colors)

    # SCRUB model results
    retain_acc_values = results['retain']['acc']
    mia_values = results['mia'] 
    epochs = range(1, len(retain_acc_values) + 1)

    if len(retain_acc_values) != len(mia_values):
        print("Warning: Retain accuracy and MIA value lists have different lengths.")
        # Potentially handle this error more gracefully, e.g., by returning None or raising an exception

    scatter = ax.scatter(retain_acc_values, mia_values,
                         marker='o', s=36, 
                         c=epochs, cmap=custom_cmap, # Use custom colormap
                         alpha=0.8, label='Unlearned Model (SCRUB)')

    # Retrained model result (single point)
    retrained_retain_acc = retrained_model_results['retain']['acc'][0]
    # Assuming retrained_model_results['mia']['acc'][0] is the MIA prob for the retrained model
    retrained_mia_prob = retrained_model_results['mia']['acc'][0] 
    
    ax.scatter(retrained_retain_acc, retrained_mia_prob,
               marker='*', s=150, color=retrained_model_color,
               label='Retrained Model', zorder=5)

    ax.set_xlabel('Retain Accuracy')
    ax.set_ylabel('MIA Probability')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1) # MIA Probability is also typically between 0 and 1

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Epoch')

    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    return fig

def plot_retain_vs_forget_js_div(results, retrained_model_results, figsize=(7, 7)):
    """
    Plot Retain JS Divergence vs. Forget JS Divergence as a scatter plot.
    Points are colored by epoch.

    Args:
        results: Dictionary containing metrics, including 'js_div'.
        retrained_model_results: Dictionary containing metrics for the retrained model.
        figsize: Tuple for figure size (width, height).

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle('Retain JS Div vs. Forget JS Div', fontsize=16, fontweight='bold')

    retrained_model_color = '#f95d6a' # Color for the retrained model marker

    # Define your custom color palette (consistent with other plots)
    custom_colors_palette = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
    custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", custom_colors_palette)

    # SCRUB model results
    retain_js_div_values = results['js_div']['retain']
    forget_js_div_values = results['js_div']['forget']
    epochs = range(1, len(retain_js_div_values) + 1)

    if len(retain_js_div_values) != len(forget_js_div_values):
        print("Warning: Retain JS Div and Forget JS Div lists have different lengths.")
        # Handle error or return

    scatter = ax.scatter(retain_js_div_values, forget_js_div_values,
                         marker='o', s=36,
                         c=epochs, cmap=custom_cmap,
                         alpha=0.8, label='Unlearned Model (SCRUB)')

    # Retrained model result (single point)
    # This assumes that JS divergence for a retrained model is typically low or zero
    # against the data it was trained on (for retain) and against a hypothetical forget set.
    # Adjust if your retrained_model_results have meaningful non-zero JS Div values.
    retrained_retain_js_div = 0.0 # Default assumption
    retrained_forget_js_div = 0.0 # Default assumption

    if 'js_div' in retrained_model_results:
        if 'retain' in retrained_model_results['js_div'] and retrained_model_results['js_div']['retain']:
            retrained_retain_js_div = retrained_model_results['js_div']['retain'][0]
        if 'forget' in retrained_model_results['js_div'] and retrained_model_results['js_div']['forget']:
            retrained_forget_js_div = retrained_model_results['js_div']['forget'][0]
            
    ax.scatter(retrained_retain_js_div, retrained_forget_js_div,
               marker='*', s=150, color=retrained_model_color,
               label='Retrained Model', zorder=5)

    ax.set_xlabel('Retain Set JS Divergence')
    ax.set_ylabel('Forget Set JS Divergence')
    ax.grid(True, alpha=0.3)
    # Set limits based on expected JS divergence range, typically 0 to 1 or log scale if very small
    ax.set_xlim(left=-0.05, right=max(retain_js_div_values + [retrained_retain_js_div, 0.5]) * 1.1) 
    ax.set_ylim(bottom=-0.05, top=max(forget_js_div_values + [retrained_forget_js_div, 0.5]) * 1.1)


    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Epoch')

    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
    return fig

def plot_retain_acc_vs_forget_acc(results, retrained_model_results, figsize=(7, 7)):
    """
    Plot Retain Accuracy vs. Forget Accuracy as a scatter plot.
    Points are colored by epoch.

    Args:
        results: Dictionary containing metrics, including 'retain' and 'forget' accuracies.
        retrained_model_results: Dictionary containing metrics for the retrained model.
        figsize: Tuple for figure size (width, height).

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle('Retain Accuracy vs. Forget Accuracy', fontsize=16, fontweight='bold')

    retrained_model_color = '#f95d6a' # Color for the retrained model marker

    # Define your custom color palette (consistent with other plots)
    custom_colors_palette = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
    custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", custom_colors_palette)

    # SCRUB model results
    retain_acc_values = results['retain']['acc']
    forget_acc_values = results['forget']['acc']
    epochs = range(1, len(retain_acc_values) + 1)

    if len(retain_acc_values) != len(forget_acc_values):
        print("Warning: Retain accuracy and Forget accuracy lists have different lengths.")
        # Handle error or return

    scatter = ax.scatter(retain_acc_values, forget_acc_values,
                         marker='o', s=36,
                         c=epochs, cmap=custom_cmap,
                         alpha=0.8, label='Unlearned Model (SCRUB)')

    # Retrained model result (single point)
    retrained_retain_acc = retrained_model_results['retain']['acc'][0]
    retrained_forget_acc = retrained_model_results['forget']['acc'][0]
            
    ax.scatter(retrained_retain_acc, retrained_forget_acc,
               marker='*', s=150, color=retrained_model_color,
               label='Retrained Model', zorder=5)

    ax.set_xlabel('Retain Accuracy')
    ax.set_ylabel('Forget Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.1) 
    ax.set_ylim(0, 1.1)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Epoch')

    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
    return fig


def plot_retain_forget_accuracy_epochs(results, retrained_model_results, figsize=(8, 6)):
    """
    Plot Retain and Forget accuracies against epochs on the same plot.

    Args:
        results: Dictionary containing metrics for the unlearned model.
        retrained_model_results: Dictionary containing metrics for the retrained model.
        figsize: Tuple for figure size (width, height).

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle('Retain vs. Forget Accuracy Across Epochs', fontsize=16, fontweight='bold')

    # Colors from your palette
    retain_color = '#003f5c'  # For SCRUB Retain Acc
    forget_color = '#ffa600'  # For SCRUB Forget Acc
    retrained_line_color = '#f95d6a' # For retrained model lines

    # SCRUB model results
    retain_acc_values = results['retain']['acc']
    forget_acc_values = results['forget']['acc']
    epochs = range(1, len(retain_acc_values) + 1)

    # Plot SCRUB model accuracies
    ax.plot(epochs, retain_acc_values,
            marker='o', linewidth=2, markersize=5,
            color=retain_color, alpha=0.8, label='Unlearned Model - Retain Acc')
    
    ax.plot(epochs, forget_acc_values,
            marker='s', linewidth=2, markersize=5, # Different marker for forget
            color=forget_color, alpha=0.8, label='Unlearned Model - Forget Acc')

    # Retrained model results (horizontal lines)
    retrained_retain_acc = retrained_model_results['retain']['acc'][0]
    retrained_forget_acc = retrained_model_results['forget']['acc'][0]
            
    ax.axhline(y=retrained_retain_acc, color=retrained_line_color, linestyle='--', 
               alpha=0.9, linewidth=1.5, label=f'Retrained Model - Retain Acc ({retrained_retain_acc:.2f})')
    ax.axhline(y=retrained_forget_acc, color=retrained_line_color, linestyle=':', 
               alpha=0.9, linewidth=1.5, label=f'Retrained Model - Forget Acc ({retrained_forget_acc:.2f})') # Dotted line

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1) 

    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
    return fig


def main():
    # Example results
    results_dir = 'results/MNIST/seed_42/scrub'
    retain_split_type = "random"

    # Make folder for retain_split_type
    plots_dir = os.path.join("plots/scrub", retain_split_type)
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(results_dir, f'scrub_metrics_{retain_split_type}.json'), 'r') as f:
        results = json.load(f)
    mia_results = results['mia']
    js_div_results = results['js_div']
    
    # Create and save plots
    os.makedirs(results_dir, exist_ok=True)

    # load retrained model results
    with open(os.path.join(results_dir, f'retrained_model_metrics_{retain_split_type}.json'), 'r') as f:
        retrained_model_results = json.load(f)
        
    fig_acc = plot_accuracy_metrics(results, retrained_model_results)
    fig_mia_epochs = plot_mia_metrics(mia_results, retrained_model_results)
    fig_js_div_epochs = plot_js_divergence_metrics(js_div_results)
    fig_js_vs_acc = plot_js_div_vs_retain_acc(results, retrained_model_results)
    fig_mia_vs_acc = plot_mia_vs_retain_acc(results, retrained_model_results)
    fig_retain_vs_forget_js = plot_retain_vs_forget_js_div(results, retrained_model_results)
    fig_acc_tradeoff = plot_retain_acc_vs_forget_acc(results, retrained_model_results)
    fig_retain_forget_epochs = plot_retain_forget_accuracy_epochs(results, retrained_model_results)
    # Save plots
    os.makedirs(plots_dir, exist_ok=True)
    fig_acc.savefig(os.path.join(plots_dir, f'scrub_accuracy_loss_results_{retain_split_type}.png'), bbox_inches='tight')
    fig_mia_epochs.savefig(os.path.join(plots_dir, f'scrub_mia_epochs_results_{retain_split_type}.png'), bbox_inches='tight')
    fig_js_div_epochs.savefig(os.path.join(plots_dir, f'scrub_js_div_epochs_results_{retain_split_type}.png'), bbox_inches='tight')
    fig_js_vs_acc.savefig(os.path.join(plots_dir, f'scrub_js_div_vs_retain_acc_{retain_split_type}.png'), bbox_inches='tight')
    fig_mia_vs_acc.savefig(os.path.join(plots_dir, f'scrub_mia_vs_retain_acc_{retain_split_type}.png'), bbox_inches='tight')
    fig_retain_vs_forget_js.savefig(os.path.join(plots_dir, f'scrub_retain_vs_forget_js_div_{retain_split_type}.png'), bbox_inches='tight')
    fig_acc_tradeoff.savefig(os.path.join(plots_dir, f'scrub_retain_vs_forget_acc_{retain_split_type}.png'), bbox_inches='tight')
    fig_retain_forget_epochs.savefig(os.path.join(plots_dir, f'scrub_retain_forget_accuracy_epochs_{retain_split_type}.png'), bbox_inches='tight')
    plt.close('all')

if __name__ == "__main__":
    main() 