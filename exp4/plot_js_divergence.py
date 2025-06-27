import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. Data Preparation ---
try:
    # TODO: Make this configurable if needed, e.g., "MNIST"
    dataset_name = "MNIST" 
    results_folder = f"real_data_results/{dataset_name}"
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
    file_path = os.path.join(script_dir, results_folder, f'{dataset_name}_js_div_all.csv')
    df = pd.read_csv(file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {file_path}")

# --- 2. Data Aggregation ---
metrics_to_plot = ['JS div. retrained', 'JS div. original']

# Aggregate data for each metric and dataset type
aggregated_df = df.groupby(['model-name', 'dataset'])[metrics_to_plot].agg(['mean', 'std'])
aggregated_df.columns = ['_'.join(col) for col in aggregated_df.columns.values]
aggregated_df = aggregated_df.reset_index()

print("--- Aggregated Data (JS Divergence) ---")
print(aggregated_df.to_string())
print("\n" + "="*80 + "\n")

# --- 3. Setup for Plotting and Color Palette ---
COLOR_PALETTE = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']

# Define font sizes for consistency
TITLE_FONTSIZE = 26
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18

# --- 4. Plotting ---
print("--- Generating JS Divergence Plots ---")
dataset_order = ['retain', 'forget', 'val']
# Ensure all requested datasets are present in the dataframe
if not all(d in df['dataset'].unique() for d in dataset_order):
    dataset_order = list(df['dataset'].unique())
    print(f"Warning: Not all datasets found. Plotting for: {', '.join(dataset_order)}")


for dataset_type in dataset_order:
    for metric in metrics_to_plot:
        metric_mean_col = f'{metric}_mean'
        metric_std_col = f'{metric}_std'
        
        # Filter for the specific dataset
        plot_df = aggregated_df[aggregated_df['dataset'] == dataset_type].copy()

        # Sort models by JS divergence for the current metric
        plot_df = plot_df.sort_values(by=metric_mean_col, ascending=True)

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))
        
        plot_df.plot(
            kind='bar', x='model-name', y=metric_mean_col, yerr=metric_std_col,
            ax=ax, capsize=4, width=0.6, legend=False,
            color=COLOR_PALETTE[0], edgecolor='black'
        )

        # Formatting the plot
        plot_title_metric = metric.replace('JS div. ', '').replace('_', ' ').title()
        dataset_title = dataset_type.title()
        ax.set_title(f'JS Divergence to {plot_title_metric} Model on {dataset_title} Set', 
                     fontsize=TITLE_FONTSIZE, pad=20)
        ax.set_ylabel('JS Divergence (log scale)', fontsize=LABEL_FONTSIZE)
        ax.set_xlabel('')
        
        plt.xticks(rotation=45, ha='right', fontsize=TICK_FONTSIZE)
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
        ax.grid(True, which="both", ls="--", axis='y', color='grey', alpha=0.7)
        ax.set_axisbelow(True)

        ax.set_yscale('log')

        plt.tight_layout()
        
        plot_filename_metric = metric.replace(' ', '_').replace('.', '')
        print(f"Generating and saving plot for {metric} on {dataset_type}...")
        save_dir = os.path.join(script_dir, results_folder, 'plots')
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f'{plot_filename_metric}_{dataset_type}_comparison.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

print("\nJS Divergence plots have been generated and saved.") 