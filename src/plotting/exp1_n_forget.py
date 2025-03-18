from wandb_client import WandbClient
from datetime import datetime
import json
import os
import argparse
import matplotlib.pyplot as plt
from typing import Dict, Any


def plot_performance_metrics(results: Dict[str, Dict[str, Any]], 
                             colors: Dict[str, str] = None, 
                             save_to_pdf: bool = False, 
                             save_to_png: bool = False, 
                             experiments_folder: str = "results/n_forget"):
    """
    Plot performance metrics for each unlearning method.
    """
    # Define performance metrics
    time_metrics = ['performance_tracking/unlearning/time_seconds',
                    'performance_tracking/retrained/training_time_seconds',
                    'performance_tracking/original/training_time_seconds']
    
    peak_memory_metrics = ['performance_tracking/unlearning/cpu_peak_memory_mb', 
                          'performance_tracking/retrained/cpu_peak_memory_mb', 
                          'performance_tracking/original/cpu_peak_memory_mb']
    
    memory_usage_metrics = ['performance_tracking/unlearning/cpu_memory_usage_mb', 
                           'performance_tracking/retrained/cpu_memory_usage_mb', 
                           'performance_tracking/original/cpu_memory_usage_mb']
    
    # Define colors for each unlearning method using a subtle professional palette
    if colors is None:
        colors = {
            'amnesiac': '#4285F4',  # Google blue
            'sisa': '#34A853',      # Google green
            'scrub+r': '#EA4335',   # Google red
            'ssd': '#8E44AD'        # Rich purple
        }
    
    # Plot time metrics
    plt, fig, axs = _plot_performance_grid(results, time_metrics, colors, 
                          title='Unlearning Time',
                          ylabel='Time (seconds)')
    
    if save_to_pdf:
        os.makedirs(experiments_folder, exist_ok=True)
        plt.savefig(f"{experiments_folder}/time_metrics_grid.pdf", bbox_inches='tight')
    
    if save_to_png:
        os.makedirs(experiments_folder, exist_ok=True)
        plt.savefig(f"{experiments_folder}/time_metrics_grid.png", dpi=300, bbox_inches='tight')
    
    # Plot peak memory metrics
    plt, fig, axs = _plot_performance_grid(results, peak_memory_metrics, colors, 
                          title='Peak Memory Usage',
                          ylabel='Memory (MB)')
    
    if save_to_pdf:
        plt.savefig(f"{experiments_folder}/peak_memory_metrics_grid.pdf", bbox_inches='tight')
    
    if save_to_png:
        plt.savefig(f"{experiments_folder}/peak_memory_metrics_grid.png", dpi=300, bbox_inches='tight')

    # Plot memory usage metrics
    plt, fig, axs = _plot_performance_grid(results, memory_usage_metrics, colors, 
                          title='Memory Usage',
                          ylabel='Memory (MB)')

    if save_to_pdf:
        plt.savefig(f"{experiments_folder}/memory_usage_metrics_grid.pdf", bbox_inches='tight')
    
    if save_to_png:
        plt.savefig(f"{experiments_folder}/memory_usage_metrics_grid.png", dpi=300, bbox_inches='tight')

    return plt, fig, axs


def _plot_performance_grid(results, metrics, colors, title, ylabel):
    """
    Helper function to plot a grid of performance metrics.
    """
    # Create a grid of plots (1 row x N columns)
    num_cols = len(metrics)
    fig, axs = plt.subplots(1, num_cols, figsize=(5*num_cols, 5), sharex=True)
    plt.subplots_adjust(wspace=0.3, left=0.1, right=0.95, top=0.85, bottom=0.15)
    
    # Handle the case where there's only one metric (axs is not an array)
    if num_cols == 1:
        axs = [axs]
    
    # Set a common title
    fig.suptitle(title, fontsize=16, y=0.95)
    
    # Find the maximum and minimum y values across all metrics to set consistent y-axis limits
    max_y_value = 0
    min_y_value = float('inf')
    for metric in metrics:
        for unlearn_type, unlearn_data in results.items():
            for x_str in unlearn_data:
                if metric in unlearn_data[x_str]:
                    # Consider both the mean and potential error bar maximum/minimum
                    mean_val = unlearn_data[x_str][metric]['mean']
                    std_val = unlearn_data[x_str][metric]['std']
                    potential_max = mean_val + std_val
                    potential_min = mean_val - std_val
                    max_y_value = max(max_y_value, potential_max)
                    min_y_value = min(min_y_value, potential_min)
    
    # Add buffers to the maximum and minimum values
    y_max_limit = max_y_value + 0.1 * max_y_value  # Add 10% to the max
    y_min_limit = max(0, min_y_value - 0.1 * min_y_value)  # Subtract 10% from min, but not below 0
    
    # For each metric, create a plot
    for i, metric in enumerate(metrics):
        ax = axs[i]
        
        # Extract model type for the title
        parts = metric.split('/')
        model_type = parts[1].capitalize()
        
        # Set the title for this subplot
        ax.set_title(model_type, fontsize=14)
        
        # For each unlearning method, plot the data
        for unlearn_type, unlearn_data in results.items():
            # Extract x values (n_forget points) and sort them
            # Extract the n_forget value from the group_key
            x_values = []
            for key in unlearn_data.keys():
                if 'group_key' in unlearn_data[key]:
                    group_key = unlearn_data[key]['group_key']
                    # Extract n_points value from the group_key
                    if 'n_points_' in group_key:
                        n_points = int(group_key.split('n_points_')[1])
                        x_values.append(n_points)
                    else:
                        # If we can't extract n_points, use the key itself
                        x_values.append(key)
                else:
                    # If there's no group_key, use the key itself
                    x_values.append(key)
            
            # Create a mapping from x_values to keys
            x_to_key = {}
            for key in unlearn_data.keys():
                if 'group_key' in unlearn_data[key]:
                    group_key = unlearn_data[key]['group_key']
                    if 'n_points_' in group_key:
                        n_points = int(group_key.split('n_points_')[1])
                        x_to_key[n_points] = key
                else:
                    x_to_key[key] = key
            
            # Sort x_values
            x_values = sorted(x_values)
            
            # Prepare y values and error bars
            y_values = []
            y_errors = []
            
            for x in x_values:
                key = x_to_key.get(x)
                if key and metric in unlearn_data[key]:
                    y_values.append(unlearn_data[key][metric]['mean'])
                    y_errors.append(unlearn_data[key][metric]['std'])
                else:
                    # Handle missing data
                    y_values.append(None)
                    y_errors.append(None)
            
            # Filter out None values
            valid_points = [(x, y, e) for x, y, e in zip(x_values, y_values, y_errors) if y is not None]
            if not valid_points:
                continue
                
            valid_x, valid_y, valid_e = zip(*valid_points)
            
            # Plot with error bars
            ax.errorbar(valid_x, valid_y, yerr=valid_e, 
                       fmt='o-', label=unlearn_type, 
                       color=colors.get(unlearn_type, 'gray'),
                       capsize=4, markersize=6, linewidth=2)
        
        # Set labels and grid
        ax.set_xlabel('Number of Forget Points')
        if i == 0:  # Only leftmost column gets y-axis label
            ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis limits based on the maximum and minimum values found
        ax.set_ylim(y_min_limit, y_max_limit)
        
        # Add a legend to the first plot only
        if i == 0:
            ax.legend(loc='upper left')

    return plt, fig, axs

def plot_accuracy_metrics(results: Dict[str, Dict[str, Any]], colors: Dict[str, str] = None, save_to_pdf: bool = False, save_to_png: bool = False, experiments_folder: str = "results/n_forget"):
    """
    Plot accuracy metrics for each unlearning method.
    """
    # Define accuracy metrics
    accuracy_metrics = ['validation/accuracy/unlearned', 
                        'validation/accuracy/retrained', 
                        'validation/accuracy/original', 
                        'retain/accuracy/unlearned', 
                        'retain/accuracy/retrained', 
                        'retain/accuracy/original', 
                        'forget/accuracy/unlearned', 
                        'forget/accuracy/retrained', 
                        'forget/accuracy/original']
    
    # Define colors for each unlearning method using a subtle professional palette
    if colors is None:
        colors = {
            'amnesiac': '#4285F4',  # Google blue
            'sisa': '#34A853',      # Google green
            'scrub+r': '#EA4335',   # Google red
            'ssd': '#8E44AD'        # Rich purple
        }
    
    # Define row labels
    row_labels = ['Validation', 'Retain', 'Forget']
    
    # Create a 3x3 grid of plots
    fig, axs = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.3, left=0.1, right=0.95, top=0.9, bottom=0.1)
    
    # Set a common title
    fig.suptitle('Accuracy Metrics Across Unlearning Methods', fontsize=16, y=0.98)
    
    # Add row labels (on the left side)
    for i, label in enumerate(row_labels):
        fig.text(0.02, 0.77 - i*0.27, label, rotation=90, 
                 ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Find the maximum and minimum y values across all metrics to set consistent y-axis limits
    max_y_value = 0
    min_y_value = float('inf')
    for metric in accuracy_metrics:
        for unlearn_type, unlearn_data in results.items():
            for x_str in unlearn_data:
                if metric in unlearn_data[x_str]:
                    # Consider both the mean and potential error bar maximum/minimum
                    mean_val = unlearn_data[x_str][metric]['mean']
                    std_val = unlearn_data[x_str][metric]['std']
                    potential_max = mean_val + std_val
                    potential_min = mean_val - std_val
                    max_y_value = max(max_y_value, potential_max)
                    min_y_value = min(min_y_value, potential_min)
    
    # Add buffers to the maximum and minimum values
    y_max_limit = min(1.0, max_y_value + 0.1)  # Cap at 1.0 for accuracy
    y_min_limit = max(0, min_y_value - 0.1)    # Ensure we don't go below 0 for accuracy
    
    # Column titles
    col_titles = ['Unlearned', 'Retrained', 'Original']
    
    # For each metric, create a plot
    for i, metric in enumerate(accuracy_metrics):
        # Calculate row and column indices
        row = i // 3
        col = i % 3
        
        ax = axs[row, col]
        
        # Add column title only to the top row
        if row == 0:
            ax.set_title(col_titles[col], fontsize=14)
        
        # For each unlearning method, plot the data
        for unlearn_type, unlearn_data in results.items():
            # Extract x values (n_forget points) and sort them
            # Extract the n_forget value from the group_key
            x_values = []
            for key in unlearn_data.keys():
                if 'group_key' in unlearn_data[key]:
                    group_key = unlearn_data[key]['group_key']
                    # Extract n_points value from the group_key
                    if 'n_points_' in group_key:
                        n_points = int(group_key.split('n_points_')[1])
                        x_values.append(n_points)
                    else:
                        # If we can't extract n_points, use the key itself
                        x_values.append(key)
                else:
                    # If there's no group_key, use the key itself
                    x_values.append(key)
            
            # Create a mapping from x_values to keys
            x_to_key = {}
            for key in unlearn_data.keys():
                if 'group_key' in unlearn_data[key]:
                    group_key = unlearn_data[key]['group_key']
                    if 'n_points_' in group_key:
                        n_points = int(group_key.split('n_points_')[1])
                        x_to_key[n_points] = key
                else:
                    x_to_key[key] = key
            
            # Sort x_values
            x_values = sorted(x_values)
            
            # Prepare y values and error bars
            y_values = []
            y_errors = []
            
            for x in x_values:
                key = x_to_key.get(x)
                if key and metric in unlearn_data[key]:
                    y_values.append(unlearn_data[key][metric]['mean'])
                    y_errors.append(unlearn_data[key][metric]['std'])
                else:
                    # Handle missing data
                    y_values.append(None)
                    y_errors.append(None)
            
            # Filter out None values
            valid_points = [(x, y, e) for x, y, e in zip(x_values, y_values, y_errors) if y is not None]
            if not valid_points:
                continue
                
            valid_x, valid_y, valid_e = zip(*valid_points)
            
            # Plot with error bars
            ax.errorbar(valid_x, valid_y, yerr=valid_e, 
                       fmt='o-', label=unlearn_type, 
                       color=colors.get(unlearn_type, 'gray'),
                       capsize=4, markersize=6, linewidth=2)
        
        # Set labels and grid
        if row == 2:  # Only bottom row gets x-axis label
            ax.set_xlabel('Number of Forget Points')
        if col == 0:  # Only leftmost column gets y-axis label
            ax.set_ylabel('Accuracy')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis limits based on the maximum and minimum values found
        ax.set_ylim(y_min_limit, y_max_limit)
        
        # Add a legend to the top-left plot only
        if row == 0 and col == 0:
            ax.legend(loc='lower right')
    
    # Save the figure
    if save_to_pdf:
        os.makedirs(experiments_folder, exist_ok=True)
        plt.savefig(f"{experiments_folder}/accuracy_metrics_grid.pdf", bbox_inches='tight')
    
    if save_to_png:
        os.makedirs(experiments_folder, exist_ok=True)
        plt.savefig(f"{experiments_folder}/accuracy_metrics_grid.png", dpi=300, bbox_inches='tight')
    
    return plt, fig, axs

def plot_equiv_metrics(results: Dict[str, Dict[str, Any]], colors: Dict[str, str] = None, save_to_pdf: bool = False, save_to_png: bool = False, experiments_folder: str = "results/n_forget"):
    """
    Plot equivalence metrics for each unlearning method.
    """
    # Define equivalence metrics
    hamming_pd_metrics = ['validation/unlearned_vs_original/hamming_pd', 
                     'validation/retrained_vs_original/hamming_pd', 
                     'validation/unlearned_vs_retrained/hamming_pd', 
                     'retain/unlearned_vs_original/hamming_pd', 
                     'retain/retrained_vs_original/hamming_pd', 
                     'retain/unlearned_vs_retrained/hamming_pd', 
                     'forget/unlearned_vs_original/hamming_pd', 
                     'forget/retrained_vs_original/hamming_pd', 
                     'forget/unlearned_vs_retrained/hamming_pd']
    
    js_divergence_metrics = ['validation/unlearned_vs_original/js_divergence', 
                            'validation/retrained_vs_original/js_divergence', 
                            'validation/unlearned_vs_retrained/js_divergence', 
                            'retain/unlearned_vs_original/js_divergence', 
                            'retain/retrained_vs_original/js_divergence', 
                            'retain/unlearned_vs_retrained/js_divergence', 
                            'forget/unlearned_vs_original/js_divergence', 
                            'forget/retrained_vs_original/js_divergence', 
                            'forget/unlearned_vs_retrained/js_divergence']
    
    # Define colors for each unlearning method using a subtle professional palette
    if colors is None:
        colors = {
            'amnesiac': '#4285F4',  # Google blue
            'sisa': '#34A853',      # Google green
            'scrub+r': '#EA4335',   # Google red
            'ssd': '#8E44AD'        # Rich purple
        }
    
    # Plot Hamming PD metrics
    plt, fig, axs = _plot_metric_grid(results, hamming_pd_metrics, colors, 
                     title='Hamming Prediction Distance Across Unlearning Methods',
                     ylabel='Hamming PD')
    
    if save_to_pdf:
        os.makedirs(experiments_folder, exist_ok=True)
        plt.savefig(f"{experiments_folder}/hamming_pd_metrics_grid.pdf", bbox_inches='tight')
    
    if save_to_png:
        os.makedirs(experiments_folder, exist_ok=True)
        plt.savefig(f"{experiments_folder}/hamming_pd_metrics_grid.png", dpi=300, bbox_inches='tight')

    # Plot JS Divergence metrics
    plt, fig, axs = _plot_metric_grid(results, js_divergence_metrics, colors, 
                     title='JS Divergence Across Unlearning Methods',
                     ylabel='JS Divergence')
    
    if save_to_pdf:
        os.makedirs(experiments_folder, exist_ok=True)
        plt.savefig(f"{experiments_folder}/js_divergence_metrics_grid.pdf", bbox_inches='tight')
    
    if save_to_png:
        os.makedirs(experiments_folder, exist_ok=True)
        plt.savefig(f"{experiments_folder}/js_divergence_metrics_grid.png", dpi=300, bbox_inches='tight')
    
    return plt, fig, axs

def _plot_metric_grid(results, metrics, colors, title, ylabel):
    """
    Helper function to plot a grid of metrics with row and column labels.
    """
    # Define row labels
    row_labels = ['Validation', 'Retain', 'Forget']
    
    # Define column titles
    col_titles = ['Unlearned vs Original', 'Retrained vs Original', 'Unlearned vs Retrained']
    
    # Create a 3x3 grid of plots
    fig, axs = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.3, left=0.1, right=0.95, top=0.9, bottom=0.1)
    
    # Set a common title
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Add row labels (on the left side)
    for i, label in enumerate(row_labels):
        fig.text(0.02, 0.77 - i*0.27, label, rotation=90, 
                 ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Find the maximum and minimum y values across all metrics to set consistent y-axis limits
    max_y_value = 0
    min_y_value = float('inf')
    for metric in metrics:
        for unlearn_type, unlearn_data in results.items():
            for x_str in unlearn_data:
                if metric in unlearn_data[x_str]:
                    # Consider both the mean and potential error bar maximum/minimum
                    mean_val = unlearn_data[x_str][metric]['mean']
                    std_val = unlearn_data[x_str][metric]['std']
                    potential_max = mean_val + std_val
                    potential_min = mean_val - std_val
                    max_y_value = max(max_y_value, potential_max)
                    min_y_value = min(min_y_value, potential_min)
    
    # Add buffers to the maximum and minimum values
    y_max_limit = max_y_value + 0.1 * max_y_value
    y_min_limit = max(0, min_y_value - 0.1 * min_y_value)  # Ensure we don't go below 0 for most metrics
    
    # For each metric, create a plot
    for i, metric in enumerate(metrics):
        # Calculate row and column indices
        row = i // 3
        col = i % 3
        
        ax = axs[row, col]
        
        # Add column title only to the top row
        if row == 0:
            ax.set_title(col_titles[col], fontsize=14)
        
        # For each unlearning method, plot the data
        for unlearn_type, unlearn_data in results.items():
            # Extract x values (n_forget points) and sort them
            # Extract the n_forget value from the group_key
            x_values = []
            for key in unlearn_data.keys():
                if 'group_key' in unlearn_data[key]:
                    group_key = unlearn_data[key]['group_key']
                    # Extract n_points value from the group_key
                    if 'n_points_' in group_key:
                        n_points = int(group_key.split('n_points_')[1])
                        x_values.append(n_points)
                    else:
                        # If we can't extract n_points, use the key itself
                        x_values.append(key)
                else:
                    # If there's no group_key, use the key itself
                    x_values.append(key)
            
            # Create a mapping from x_values to keys
            x_to_key = {}
            for key in unlearn_data.keys():
                if 'group_key' in unlearn_data[key]:
                    group_key = unlearn_data[key]['group_key']
                    if 'n_points_' in group_key:
                        n_points = int(group_key.split('n_points_')[1])
                        x_to_key[n_points] = key
                else:
                    x_to_key[key] = key
            
            # Sort x_values
            x_values = sorted(x_values)
            
            # Prepare y values and error bars
            y_values = []
            y_errors = []
            
            for x in x_values:
                key = x_to_key.get(x)
                if key and metric in unlearn_data[key]:
                    y_values.append(unlearn_data[key][metric]['mean'])
                    y_errors.append(unlearn_data[key][metric]['std'])
                else:
                    # Handle missing data
                    y_values.append(None)
                    y_errors.append(None)
            
            # Filter out None values
            valid_points = [(x, y, e) for x, y, e in zip(x_values, y_values, y_errors) if y is not None]
            if not valid_points:
                continue
                
            valid_x, valid_y, valid_e = zip(*valid_points)
            
            # Plot with error bars
            ax.errorbar(valid_x, valid_y, yerr=valid_e, 
                       fmt='o-', label=unlearn_type, 
                       color=colors.get(unlearn_type, 'gray'),
                       capsize=4, markersize=6, linewidth=2)
        
        # Set labels and grid
        if row == 2:  # Only bottom row gets x-axis label
            ax.set_xlabel('Number of Forget Points')
        if col == 0:  # Only leftmost column gets y-axis label
            ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis limits based on the maximum and minimum values found
        ax.set_ylim(y_min_limit, y_max_limit)
        
        # Add a legend to the top-left plot only
        if row == 0 and col == 0:
            ax.legend(loc='upper left')
    
    return plt, fig, axs

def plot(results: Dict[str, Dict[str, Any]], save_to_pdf: bool = False, save_to_png: bool = False, experiments_folder: str = "results/n_forget"):
    """
    Plot separate grids of metrics for validation, retain, and forget categories.
    
    Args:
        results: Dictionary mapping unlearn_type to experiment results
    """
    colors = {
        'amnesiac': '#4285F4',  # Google blue
        'sisa': '#34A853',      # Google green
        'scrub+r': '#EA4335',   # Google red
        'ssd': '#8E44AD'        # Rich purple
    }
    
    plot_performance_metrics(results, colors, save_to_pdf, save_to_png, experiments_folder)
    plot_accuracy_metrics(results, colors, save_to_pdf, save_to_png, experiments_folder)
    plot_equiv_metrics(results, colors, save_to_pdf, save_to_png, experiments_folder)


if __name__ == "__main__":

    experiments_folder = "results/n_forget"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str.lower,
        default="experiment",
        choices=["data", "plot"],
        help="extract data from wandb or plot the data",
    )

    args = parser.parse_args()

    if args.mode == "data":

        # Example usage of the WandbClient class
        unlearn_types = ["amnesiac", "sisa", "scrub+r", "ssd"]

        entity = "machine-unlearning-thesis"
        project = "unlearning-experiments"

        results = {}
        for unlearn_type in unlearn_types:
            # Get today's date in ISO format (YYYY-MM-DD)
            today = datetime.now().strftime("%Y-%m-%d")
            
            filters = {
                "config.experiment.experiment_group": "sweep_forget_n_points",
                "config.experiment.unlearn_type": unlearn_type,
                "created_at": {"$gte": "2025-03-06"}  # Filter for runs created on or after May 3, 2025
            }
            
            client = WandbClient(entity=entity, project=project)
            runs = client.get_runs(filters=filters)

            experiment_means = client.group_runs_by_experiment_id(runs, sweep_metric="forget.n_forget")

            results[unlearn_type] = experiment_means


            # print(experiment_means)

        os.makedirs(experiments_folder, exist_ok=True)

        # save results
        with open(f"{experiments_folder}/results_n_forget.json", "w") as f:
            json.dump(results, f)

    elif args.mode == "plot":
        # from exp1_n_ood import plot
        # raise error if results_ood_ratio.json does not exist
        if not os.path.exists(f"{experiments_folder}/results_n_forget.json"):
            raise FileNotFoundError(f"results_n_forget.json does not exist in {experiments_folder}")

        # Load results
        with open(f"{experiments_folder}/results_n_forget.json", "r") as f:
            results = json.load(f)
        
        # plot the data
        plot(results, save_to_pdf=True, save_to_png=True, experiments_folder=experiments_folder)
