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

def plot_metrics_grid(data_dict, metric_name="Accuracy", ylim=(0.5, 1.0), save_path=None):
    """
    Create a grid of barplots from a nested dictionary with the structure:
    {row_name: {column_name: {model_name: {mean: value, std: value}}}}
    
    Args:
        data_dict: Nested dictionary with metrics data
        metric_name: Name of the metric being plotted (for title and y-axis)
        ylim: Tuple with (min, max) for y-axis limits
        save_path: Path to save the figure (optional)
    """
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')  # This is still ok as it's a matplotlib style
    plt.rcParams.update({'font.size': 12})  # Increase font size instead of using sns.set_context
    
    # Extract row and column names
    row_names = list(data_dict.keys())
    column_names = list(data_dict[row_names[0]].keys())
    
    # Get all unique model names across all rows and columns
    model_names = set()
    for row in row_names:
        for col in column_names:
            model_names.update(data_dict[row][col].keys())
    model_names = sorted(list(model_names))
    
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
            means = []
            stds = []
            models_present = []
            
            for model in model_names:
                if model in data_dict[row][col]:
                    means.append(data_dict[row][col][model]['mean'])
                    stds.append(data_dict[row][col][model]['std'])
                    models_present.append(model)
                else:
                    # Skip models not present in this cell
                    continue
            
            # Create x positions for bars
            x_pos = np.arange(len(models_present))
            
            # Create bars with error bars
            bars = ax.bar(x_pos, means, yerr=stds, align='center', 
                   alpha=0.8, capsize=5, edgecolor='black', linewidth=1,
                   color=[model_colors[model] for model in models_present])
            
            # Add value labels on top of bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{mean:.2f}', ha='center', va='bottom', fontsize=9)
            
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
    
    # Add legend at the bottom
    handles = [plt.Rectangle((0,0),1,1, color=model_colors[model]) for model in model_names]
    fig.legend(handles, [model.capitalize() for model in model_names], 
              loc='lower center', ncol=len(model_names), 
              bbox_to_anchor=(0.5, 0.01), frameon=True)
    
    # Adjust layout with more space for row and column labels
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.92])
    
    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig


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
                "config.experiment.experiment_group": "synthetic_ood_cluster_exp1",
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
        with open(f"{experiments_folder}/results_ood_ratio.json", "w") as f:
            json.dump(results, f)

    elif args.mode == "plot":

        # raise error if results_ood_ratio.json does not exist
        if not os.path.exists(f"{experiments_folder}/results_ood_ratio.json"):
            raise FileNotFoundError(f"results_ood_ratio.json does not exist in {experiments_folder}")

        # Load results
        with open(f"{experiments_folder}/results_ood_ratio.json", "r") as f:
            results = json.load(f)
        


        # ========= Extract accuracy values =========

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



        # ========= Extract equiv values =========

        js_divergence_values = {}
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
                    
                



        plot_metrics_grid(accuracy_values, metric_name="Accuracy", ylim=(0.5, 1.0), save_path=f"{experiments_folder}/accuracy_metrics_grid.png")
        # plot the data

        plot_metrics_grid(hamming_pd_values, metric_name="Hamming PD", ylim=(0, 0.5), save_path=f"{experiments_folder}/hamming_pd_metrics_grid.png")

        plot_metrics_grid(js_divergence_values, metric_name="JS Divergence", ylim=(0, 0.3), save_path=f"{experiments_folder}/js_divergence_metrics_grid.png")

        import pdb; pdb.set_trace()
