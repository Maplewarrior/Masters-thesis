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
    # Filter for the specific dataset
    plot_df = aggregated_df[aggregated_df['dataset'] == dataset_type].copy()

    # Sort models by the 'original' JS divergence for consistent ordering
    sort_metric = 'JS div. original_mean'
    if sort_metric in plot_df.columns:
        plot_df = plot_df.sort_values(by=sort_metric, ascending=True)

    plot_df = plot_df.set_index('model-name')

    # Data to plot
    mean_cols = ['JS div. retrained_mean', 'JS div. original_mean']
    std_cols = ['JS div. retrained_std', 'JS div. original_std']
    means_to_plot = plot_df[mean_cols]
    stds_to_plot = plot_df[std_cols].copy()
    
    # Rename std columns to match mean columns for yerr alignment
    stds_to_plot.columns = means_to_plot.columns

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    
    means_to_plot.plot(
        kind='bar', yerr=stds_to_plot, ax=ax, capsize=4, width=0.8,
        color=COLOR_PALETTE[:2], edgecolor='black'
    )

    # Formatting the plot
    dataset_title = dataset_type.title()
    ax.set_title(f'JS Divergence Comparison on {dataset_title} Set', 
                 fontsize=TITLE_FONTSIZE, pad=20)
    ax.set_ylabel('JS Divergence (log scale)', fontsize=LABEL_FONTSIZE)
    ax.set_xlabel('')
    
    plt.xticks(rotation=45, ha='right', fontsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.grid(True, which="both", ls="--", axis='y', color='grey', alpha=0.7)
    ax.set_axisbelow(True)

    ax.set_yscale('log')

    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    cleaned_labels = [l.replace('_mean', '').replace('JS div. ', '').title() for l in labels]
    ax.legend(handles=handles, labels=cleaned_labels, 
              title='Comparison Model', fontsize=TICK_FONTSIZE-2, title_fontsize=TICK_FONTSIZE-2,
              bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    
    print(f"Generating and saving plot for {dataset_type} dataset...")
    save_dir = os.path.join(script_dir, results_folder, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f'js_div_comparison_{dataset_type}.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

print("\nJS Divergence plots have been generated and saved.") 