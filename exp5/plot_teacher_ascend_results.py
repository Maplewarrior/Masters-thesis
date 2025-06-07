import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import sys
from matplotlib.ticker import MaxNLocator

def plot_accuracy_metrics(results, retrained_model_results, figsize=(16, 5)):
    """
    Plot teacher ascent metrics showing retain, forget, and val accuracies.
    
    Args:
        results (dict): Dictionary containing metrics with 'retain', 'forget', 'val' keys.
        retrained_model_results (dict): Dictionary containing metrics for the retrained model.
        figsize (tuple): Tuple for figure size (width, height).
    
    Returns:
        matplotlib.figure.Figure: Figure containing the accuracy plots.
    """
    fig_acc, axes_acc = plt.subplots(1, 3, figsize=figsize, sharey=True)
    fig_acc.suptitle('Teacher Ascent Accuracy vs. Retrained Model', fontsize=16, fontweight='bold')
    
    metrics = ['retain', 'forget', 'val']
    metric_titles = ['Retain Set Accuracy', 'Forget Set Accuracy', 'Validation Set Accuracy']
    colors = {'retrained': '#f95d6a', 'unlearned': '#665191'}
    
    lines = []
    labels = []
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes_acc[i]
        
        acc_values = results[metric]['acc']
        epochs = range(len(acc_values))

        # Plot a horizontal line for the retrained model's accuracy
        retrain_acc_value = retrained_model_results[metric]['acc'][0]
        retrain_line = ax.axhline(y=retrain_acc_value, color=colors['retrained'], linestyle='--', alpha=0.9, linewidth=2)
        
        # Plot unlearned model's accuracy values
        ta_line = ax.plot(epochs, acc_values, 
                           marker='o', linewidth=2.5, markersize=5,
                           color=colors['unlearned'], alpha=0.8,
                           label='Accuracy')[0]
        
        # Only store legend handles from the first plot
        if i == 0:
            lines.extend([retrain_line, ta_line])
            labels.extend(['Retrained Model', 'Unlearned Model (TA)'])

        # Set labels and title
        ax.set_xlabel('Epoch', fontsize=12)
        if i == 0: # Set y-label only for the first plot
            ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Set grid and limits
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, len(epochs) - 1 if len(epochs) > 1 else 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # Add a single legend at the bottom center of the figure
    fig_acc.legend(lines, labels, 
                  loc='lower center', 
                  bbox_to_anchor=(0.5, -0.02),
                  ncol=2,
                  frameon=False,
                  fontsize=12)
    
    # Adjust the subplot parameters for a tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
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
                label='Unlearned Model (TA)')
    

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

    # Color for retrained model, TA points will use a colormap
    retrained_model_color = '#f95d6a'
   
    # Define your custom color palette
    custom_colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
    custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", custom_colors)


    # TA model results
    retain_acc_values = results['retain']['acc']
    js_div_forget_values = results['js_div']['forget']
    epochs = range(1, len(retain_acc_values) + 1)

    scatter = ax.scatter(retain_acc_values, js_div_forget_values,
                         marker='o', s=36,
                         c=epochs, cmap=custom_cmap,  # Color by epoch using viridis colormap
                         alpha=0.8, label='Unlearned Model (TA)')

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

    # Color for retrained model, TA points will use a colormap
    retrained_model_color = '#f95d6a'

    # Define your custom color palette
    custom_colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
    custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", custom_colors)

    # TA model results
    retain_acc_values = results['retain']['acc']
    mia_values = results['mia'] 
    epochs = range(1, len(retain_acc_values) + 1)

    if len(retain_acc_values) != len(mia_values):
        print("Warning: Retain accuracy and MIA value lists have different lengths.")
        # Potentially handle this error more gracefully, e.g., by returning None or raising an exception

    scatter = ax.scatter(retain_acc_values, mia_values,
                         marker='o', s=36, 
                         c=epochs, cmap=custom_cmap, # Use custom colormap
                         alpha=0.8, label='Unlearned Model (TA)')

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

    # TA model results
    retain_js_div_values = results['js_div']['retain']
    forget_js_div_values = results['js_div']['forget']
    epochs = range(1, len(retain_js_div_values) + 1)

    if len(retain_js_div_values) != len(forget_js_div_values):
        print("Warning: Retain JS Div and Forget JS Div lists have different lengths.")
        # Handle error or return

    scatter = ax.scatter(retain_js_div_values, forget_js_div_values,
                         marker='o', s=36,
                         c=epochs, cmap=custom_cmap,
                         alpha=0.8, label='Unlearned Model (TA)')

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

    # TA model results
    retain_acc_values = results['retain']['acc']
    forget_acc_values = results['forget']['acc']
    epochs = range(1, len(retain_acc_values) + 1)

    if len(retain_acc_values) != len(forget_acc_values):
        print("Warning: Retain accuracy and Forget accuracy lists have different lengths.")
        # Handle error or return

    scatter = ax.scatter(retain_acc_values, forget_acc_values,
                         marker='o', s=36,
                         c=epochs, cmap=custom_cmap,
                         alpha=0.8, label='Unlearned Model (TA)')

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
    retain_color = '#003f5c'  # For TA Retain Acc
    forget_color = '#ffa600'  # For TA Forget Acc
    retrained_line_color = '#f95d6a' # For retrained model lines

    # TA model results
    retain_acc_values = results['retain']['acc']
    forget_acc_values = results['forget']['acc']
    epochs = range(1, len(retain_acc_values) + 1)

    # Plot TA model accuracies
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

def plot_loss_components(loss_terms: dict, figsize=(12, 7)):
    """
    Plots the different components of the loss over training steps.

    Args:
        loss_terms (dict): A dictionary where keys are loss component names
                           (e.g., 'full', 'ascend', 'reg_weighted') and
                           values are lists of loss values per batch.
        figsize (tuple): The size of the figure.

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {
        'full': '#003f5c',
        'ascend': '#d45087',
        'repair': '#ff7c43',
        'reg_weighted': '#ffa600',
        'reg': '#2f4b7c'
    }
    
    linestyles = {
        'full': '-',
        'ascend': '--',
        'repair': ':',
        'reg_weighted': '-.',
        'reg': ':'
    }

    legend_labels = {
        'full': 'Total Loss',
        'ascend': 'Ascend Term (Forget Loss)',
        'repair': 'Repair Term (Retain Loss)',
        'reg_weighted': 'Regularization (Weighted)',
        'reg': 'Regularization (Unweighted)'
    }

    max_steps = 0
    if loss_terms:
        # Find the length of the longest list to set the x-axis
        valid_terms = {k: v for k, v in loss_terms.items() if v}
        
        # Exclude the 'full' loss term from the plot as requested
        if 'full' in valid_terms:
            del valid_terms['full']
            
        if valid_terms:
            max_steps = max(len(v) for v in valid_terms.values())
    
    steps = range(max_steps)

    for term, values in valid_terms.items():
        # Plot only if the term exists and has data
        ax.plot(steps[:len(values)], values, 
                label=legend_labels.get(term, term.replace('_', ' ').title()),
                color=colors.get(term, '#000000'),
                linestyle=linestyles.get(term, '-'),
                linewidth=2, 
                alpha=0.85)

    ax.set_title('Loss Components During Unlearning', fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Batch Step', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(title='Loss Components', fontsize=10, frameon=True)
    ax.set_xlim(0, max_steps - 1 if max_steps > 1 else 1)
    
    plt.tight_layout()
    
    return fig

def _select_from_list(options: list, prompt_message: str) -> str:
    """Helper function to prompt user to select from a list of options."""
    if not options:
        return None

    print(prompt_message)
    for i, option in enumerate(options):
        print(f"  {i + 1}: {option}")

    while True:
        try:
            choice = input(f"Select an option (1-{len(options)}) or 'q' to quit: ")
            if choice.lower() == 'q':
                return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                return options[choice_idx]
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled.")
            return None


def select_experiment_folder(base_dir: str = 'results/teacher_ascend') -> str:
    """Guides the user to interactively select an experiment results folder."""
    # Level 1: Select split type
    print(f"Searching for experiment results in: {os.path.abspath(base_dir)}")
    try:
        experiment = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        if not experiment:
            print("No subdirectories found for split types. Make sure results are in the correct folder.")
            return None
    except FileNotFoundError:
        print(f"Error: Base directory '{base_dir}' not found.")
        return None
    
    selected_experiment = _select_from_list(experiment, "\nSelect the experiment:")
    if not selected_experiment:
        return None

    return selected_experiment


def main():

    base_results_dir = "results/teacher_ascend"
    base_plots_dir = "plots/teacher_ascend"

    results_dir = select_experiment_folder(base_results_dir)
    if not results_dir:
        print("No experiment folder selected. Exiting.")
        sys.exit(1)

    experiment_dir = os.path.join(base_results_dir, results_dir)

    print("\n--- Folder Selection Successful ---")
    print(f"Analyzing results from: {experiment_dir}")
    print("------------------------------------")

    print("\nContents of the selected directory:")
    files = os.listdir(experiment_dir)
    # filter out only the ones with ta_metrics in the name 
    files = [f for f in files if "ta_metrics" in f]
    selected_file = _select_from_list(files, "\nSelect the TA version:")
    selected_file_path = os.path.join(experiment_dir, selected_file)

    # load the file
    with open(selected_file_path, 'r') as f:
        results = json.load(f)


    # remove extension from selected file
    selected_file_name = selected_file.split('.')[0]

    plots_dir = os.path.join(base_plots_dir, results_dir)
    plots_dir_subfolder = os.path.join(plots_dir, selected_file_name)

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(plots_dir_subfolder, exist_ok=True)

    mia_results = results['mia']
    js_div_results = results['js_div']
    
    # Create and save plots
    os.makedirs(results_dir, exist_ok=True)

    # load retrained model results
    with open(os.path.join(experiment_dir, f'retrained_model_metrics.json'), 'r') as f:
        retrained_model_results = json.load(f)

    # load original model results
    with open(os.path.join(experiment_dir, f'original_model_metrics.json'), 'r') as f:
        original_model_results = json.load(f)
    
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
    fig_acc.savefig(os.path.join(plots_dir_subfolder, f'accuracy_loss_results.png'), bbox_inches='tight')
    fig_mia_epochs.savefig(os.path.join(plots_dir_subfolder, f'mia_epochs_results.png'), bbox_inches='tight')
    fig_js_div_epochs.savefig(os.path.join(plots_dir_subfolder, f'js_div_epochs_results.png'), bbox_inches='tight')
    fig_acc_tradeoff.savefig(os.path.join(plots_dir_subfolder, f'retain_vs_forget_acc.png'), bbox_inches='tight')
    fig_retain_forget_epochs.savefig(os.path.join(plots_dir_subfolder, f'retain_forget_accuracy_epochs.png'), bbox_inches='tight')
    plt.close('all')

    # Now, plot the loss components
    if 'loss_terms' in results and any(results['loss_terms'].values()):
        print("Plotting loss components...")
        fig_loss = plot_loss_components(results['loss_terms'])
        
        # Save the figure
        save_path = os.path.join(plots_dir_subfolder, f'loss_components_plot.png')
        fig_loss.savefig(save_path, bbox_inches='tight')
        print(f"Loss components plot saved to: {save_path}")
        
        plt.close(fig_loss) # Close the figure to free up memory
    else:
        print("No 'loss_terms' data found to plot.")

    # except FileNotFoundError:
    #     print(f"Error: The selected directory '{results_dir}' was not found.")
    # except Exception as e:
    #     print(f"An error occurred while listing directory contents: {e}")

if __name__ == "__main__":
    main() 