import argparse
from datetime import datetime
from wandb_client import WandbClient
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


model_colors = {
    'amnesiac': '#4285F4',  # Google blue
    'sisa': '#34A853',      # Google green
    'scrub+r': '#EA4335',   # Google red
    'ssd': '#8E44AD'        # Rich purple
}

def plot_metrics_grid(data_dict_ood, data_dict_id, metric_name="Accuracy", ylim=(0.5, 1.0), save_path=None):
    """
    Create a grid of barplots from two nested dictionaries with the structure:
    {row_name: {column_name: {model_name: {mean: value, std: value}}}}
    
    Args:
        data_dict_ood: Nested dictionary with OOD metrics data
        data_dict_id: Nested dictionary with ID metrics data
        metric_name: Name of the metric being plotted (for title and y-axis)
        ylim: Tuple with (min, max) for y-axis limits
        save_path: Path to save the figure (optional)
    """
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')  # This is still ok as it's a matplotlib style
    plt.rcParams.update({'font.size': 12})  # Increase font size instead of using sns.set_context
    
    # Extract row and column names from OOD data (assuming both dicts have same structure)
    row_names = list(data_dict_ood.keys())
    column_names = list(data_dict_ood[row_names[0]].keys())
    
    # Get all unique model names across both dictionaries
    model_names = set()
    for row in row_names:
        for col in column_names:
            if row in data_dict_ood and col in data_dict_ood[row]:
                model_names.update(data_dict_ood[row][col].keys())
            if row in data_dict_id and col in data_dict_id[row]:
                model_names.update(data_dict_id[row][col].keys())
    model_names = sorted(list(model_names))
    
    # Define colors for models
    model_colors = {
        'amnesiac': '#4285F4',  # Google blue
        'sisa': '#34A853',      # Google green
        'scrub+r': '#EA4335',   # Google red
        'ssd': '#8E44AD'        # Rich purple
    }
    
    # Create figure with GridSpec
    n_rows = len(row_names)
    n_cols = len(column_names)
    fig = plt.figure(figsize=(4*n_cols, 3.5*n_rows))
    
    # Create GridSpec with extra space for row and column labels
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.3, hspace=0.4)
    
    # Loop through each row and column to create subplots
    for i, row in enumerate(row_names):
        for j, col in enumerate(column_names):
            # Create subplot
            ax = fig.add_subplot(gs[i, j])
            
            # Extract data for each model in this cell
            means_ood = []
            stds_ood = []
            means_id = []
            stds_id = []
            models_present = []
            
            for model in model_names:
                # Check if model exists in OOD data
                if row in data_dict_ood and col in data_dict_ood[row] and model in data_dict_ood[row][col]:
                    means_ood.append(data_dict_ood[row][col][model]['mean'])
                    stds_ood.append(data_dict_ood[row][col][model]['std'])
                    if model not in models_present:
                        models_present.append(model)
                else:
                    means_ood.append(0)  # Placeholder if no OOD data
                    stds_ood.append(0)
                
                # Check if model exists in ID data
                if row in data_dict_id and col in data_dict_id[row] and model in data_dict_id[row][col]:
                    means_id.append(data_dict_id[row][col][model]['mean'])
                    stds_id.append(data_dict_id[row][col][model]['std'])
                    if model not in models_present:
                        models_present.append(model)
                else:
                    means_id.append(0)  # Placeholder if no ID data
                    stds_id.append(0)
            
            # Create x positions for grouped bars
            bar_width = 0.35
            x_pos = np.arange(len(models_present))
            
            # Create bars with error bars for OOD
            bars1 = ax.bar(x_pos - bar_width/2, means_ood[:len(models_present)], bar_width, yerr=stds_ood[:len(models_present)], 
                   alpha=0.8, capsize=5, edgecolor='black', linewidth=1,
                   color=[model_colors.get(model, 'gray') for model in models_present],
                   label='OOD')
            
            # Create bars with error bars for ID
            bars2 = ax.bar(x_pos + bar_width/2, means_id[:len(models_present)], bar_width, yerr=stds_id[:len(models_present)], 
                   alpha=0.8, capsize=5, edgecolor='black', linewidth=1, hatch='//',
                   color=[model_colors.get(model, 'gray') for model in models_present],
                   label='ID')
            
            # Add value labels on top of bars
            for bar, mean in zip(bars1, means_ood[:len(models_present)]):
                if mean > 0:  # Only add label if there's a value
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{mean:.2f}', ha='center', va='bottom', fontsize=8)
            
            for bar, mean in zip(bars2, means_id[:len(models_present)]):
                if mean > 0:  # Only add label if there's a value
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{mean:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Remove x-tick labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels([])
            
            # Set y-axis limits
            ax.set_ylim(ylim)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add column labels at the top (only for the first row)
            if i == 0:
                ax.annotate(col.capitalize(), xy=(0.5, 1.15), xycoords='axes fraction',
                           ha='center', va='center', fontsize=14, fontweight='bold')
            
            # Add row labels on the left (only for the first column)
            if j == 0:
                ax.annotate(row.capitalize(), xy=(-0.35, 0.5), xycoords='axes fraction',
                           ha='center', va='center', fontsize=14, fontweight='bold',
                           rotation=90)
    
    # Add overall title
    plt.suptitle(f'{metric_name} Metrics Across Datasets and Model Types', 
                fontsize=16, y=0.98, fontweight='bold')
    
    # Add model legend at the bottom
    handles_models = [plt.Rectangle((0,0),1,1, color=model_colors.get(model, 'gray')) for model in model_names]
    first_legend = fig.legend(handles_models, [model.capitalize() for model in model_names], 
              loc='lower center', ncol=len(model_names), 
              bbox_to_anchor=(0.5, 0.02), frameon=True, title="Models")
    
    # Add dataset type legend (OOD vs ID)
    handles_datasets = [
        plt.Rectangle((0,0),1,1, color='gray', alpha=0.8),
        plt.Rectangle((0,0),1,1, color='gray', hatch='//', alpha=0.8)
    ]
    fig.legend(handles_datasets, ['OOD', 'ID'], 
              loc='lower center', ncol=2, 
              bbox_to_anchor=(0.5, 0.07), frameon=True, title="Dataset Types")
    
    # Add the first legend manually
    plt.gca().add_artist(first_legend)
    
    # Adjust layout with more space for row and column labels and legends
    plt.tight_layout(rect=[0.05, 0.12, 0.95, 0.92])
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig


def extract_accuracy_values(results):
    """
    Extract accuracy values from results dictionary.
    
    Args:
        results: Dictionary containing model results
        
    Returns:
        Dictionary with structure {row: {col: {model: {mean: value, std: value}}}}
    """
    accuracy_values = {}
    
    for model in results.keys():
        accuracy_keys = [key for key in results[model].keys() if "accuracy" in key and len(key.split("/")) > 1]
        for accuracy_key in accuracy_keys:
            split_key = accuracy_key.split("/")
            new_key = f"{split_key[0]}/{split_key[-1]}"

            col = split_key[-1]
            row = split_key[0]

            if row not in accuracy_values:
                accuracy_values[row] = {}

            if col not in accuracy_values[row]:
                accuracy_values[row][col] = {}

            accuracy_values[row][col][model] = results[model][accuracy_key]
    
    return accuracy_values


def extract_hamming_pd_values(results):
    """
    Extract Hamming PD values from results dictionary.
    
    Args:
        results: Dictionary containing model results
        
    Returns:
        Dictionary with structure {row: {col: {model: {mean: value, std: value}}}}
    """
    hamming_pd_values = {}
    
    for model in results.keys():
        hamming_pd_keys = [key for key in results[model].keys() if "hamming_pd" in key and len(key.split("/")) > 1]
        for hamming_pd_key in hamming_pd_keys:
            split_key = hamming_pd_key.split("/")
            new_key = f"{split_key[0]}/{split_key[-1]}"

            col = split_key[1]
            row = split_key[0]

            if row not in hamming_pd_values:
                hamming_pd_values[row] = {}

            if col not in hamming_pd_values[row]:
                hamming_pd_values[row][col] = {}

            hamming_pd_values[row][col][model] = results[model][hamming_pd_key]
    
    return hamming_pd_values


def extract_js_divergence_values(results):
    """
    Extract JS divergence values from results dictionary.
    
    Args:
        results: Dictionary containing model results
        
    Returns:
        Dictionary with structure {row: {col: {model: {mean: value, std: value}}}}
    """
    js_divergence_values = {}
    
    for model in results.keys():
        js_divergence_keys = [key for key in results[model].keys() if "js_divergence" in key and len(key.split("/")) > 1]
        for js_divergence_key in js_divergence_keys:
            split_key = js_divergence_key.split("/")
            new_key = f"{split_key[0]}/{split_key[-1]}"

            col = split_key[1]
            row = split_key[0]

            if row not in js_divergence_values:
                js_divergence_values[row] = {}

            if col not in js_divergence_values[row]:
                js_divergence_values[row][col] = {}

            js_divergence_values[row][col][model] = results[model][js_divergence_key]
    
    return js_divergence_values


if __name__ == "__main__":

    experiments_folder = "results/exp2_ood_synthetic"

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
        unlearn_types = ["amnesiac", "scrub+r", "ssd"]

        entity = "machine-unlearning-thesis"
        project = "unlearning-experiments"

        results = {}
        for unlearn_type in unlearn_types:
            # Get today's date in ISO format (YYYY-MM-DD)
            today = datetime.now().strftime("%Y-%m-%d")
            
            filters = {
                "config.experiment.experiment_group": "synthetic_ood_cluster_exp1_id",
                "config.experiment.unlearn_type": unlearn_type,
                "created_at": {"$gte": "2025-03-17"}  # Filter for runs created on or after May 3, 2025
            }
            
            client = WandbClient(entity=entity, project=project)
            runs = client.get_runs(filters=filters)


            experiment_means = client.group_runs_by_experiment_id(runs)

            # we will only have one key in the dictionary for this experiment
            keys = list(experiment_means.keys())


            if len(keys) > 1 or len(keys) < 1:
                raise ValueError(f"Expected 1 key in the dictionary, got {len(keys)}")
            

            results[unlearn_type] = experiment_means[keys[0]]


            # print(experiment_means)

        os.makedirs(experiments_folder, exist_ok=True)

        # save results
        with open(f"{experiments_folder}/results_ood_ratio_id.json", "w") as f:
            json.dump(results, f)

    elif args.mode == "plot":

        # raise error if results_ood_ratio.json does not exist
        if not os.path.exists(f"{experiments_folder}/results_ood_ratio_id.json"):
            raise FileNotFoundError(f"results_ood_ratio.json does not exist in {experiments_folder}")

        # Load results
        with open(f"{experiments_folder}/results_ood_ratio.json", "r") as f:
            results = json.load(f)
        

        with open(f"{experiments_folder}/results_ood_ratio_id.json", "r") as f:
            results_id = json.load(f)

        # ========= Extract accuracy values =========

        accuracy_values = extract_accuracy_values(results)
        accuracy_values_id = extract_accuracy_values(results_id)


        # ========= Extract equiv values =========

        hamming_pd_values = extract_hamming_pd_values(results)
        hamming_pd_values_id = extract_hamming_pd_values(results_id)

        js_divergence_values = extract_js_divergence_values(results)
        js_divergence_values_id = extract_js_divergence_values(results_id)


        import pdb; pdb.set_trace()

        # ========= Plot metrics grid =========

        plot_metrics_grid(accuracy_values, accuracy_values_id, metric_name="Accuracy", ylim=(0.5, 1.0), save_path=f"{experiments_folder}/accuracy_metrics_grid.png")
        # plot the data

        plot_metrics_grid(hamming_pd_values, hamming_pd_values_id, metric_name="Hamming PD", ylim=(0, 0.5), save_path=f"{experiments_folder}/hamming_pd_metrics_grid.png")

        plot_metrics_grid(js_divergence_values, js_divergence_values_id, metric_name="JS Divergence", ylim=(0, 0.3), save_path=f"{experiments_folder}/js_divergence_metrics_grid.png")

        import pdb; pdb.set_trace()
