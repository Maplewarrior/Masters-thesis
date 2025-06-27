import pandas as pd
import matplotlib.pyplot as plt
import io
import os

# --- 1. Data Preparation ---
try:
    results_folder = "real_data_results/MNIST"
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
    file_path = os.path.join(script_dir, results_folder, 'MNIST_acc_mia_all.csv')
    df = pd.read_csv(file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {file_path}")


# Define models to be excluded from the main comparison plots
original_model_names = ["Original model", "SAE original", "Amnesiac original", "SISA original"]

# Filter out original models for the main comparison plot
df_main = df[~df['model-name'].isin(original_model_names)]


# --- 2. Data Aggregation ---
metrics_to_plot = [
    'MIA-probability',
    'retain-accuracy',
    'forget-accuracy',
    'val-accuracy',
    'time (sec)'
]
aggregated_df = df_main.groupby('model-name')[metrics_to_plot].agg(['mean', 'std'])
print("--- Aggregated Data (Unlearning Models) ---")
print(aggregated_df)
print("\n" + "="*80 + "\n")


# --- 2.1. Aggregate Original Model Data & Calculate Shared Y-axis for Time ---
df_originals = df[df['model-name'].isin(original_model_names)]
agg_originals_df = pd.DataFrame()
if not df_originals.empty:
    agg_originals_df = df_originals.groupby('model-name')[metrics_to_plot].agg(['mean', 'std'])

time_metric = 'time (sec)'
global_max_time = None
use_log_scale_for_time = False

if time_metric in metrics_to_plot:
    # Combine mean and std from both dataframes to find the overall max for the y-axis
    unlearning_tops = aggregated_df.get((time_metric, 'mean'), pd.Series(dtype=float)) + aggregated_df.get((time_metric, 'std'), pd.Series(dtype=float)).fillna(0)
    original_tops = agg_originals_df.get((time_metric, 'mean'), pd.Series(dtype=float)) + agg_originals_df.get((time_metric, 'std'), pd.Series(dtype=float)).fillna(0)
    all_tops = pd.concat([unlearning_tops, original_tops])
    
    # Combine means to decide if a log scale is needed
    unlearning_means = aggregated_df.get((time_metric, 'mean'), pd.Series(dtype=float))
    original_means = agg_originals_df.get((time_metric, 'mean'), pd.Series(dtype=float))
    all_means = pd.concat([unlearning_means, original_means])

    if not all_tops.empty:
        global_max_time = all_tops.max()
        
    if not all_means.empty:
        min_mean = all_means[all_means > 0].min()
        max_mean = all_means.max()
        # Use log scale if the range of means is large enough
        if pd.notna(min_mean) and min_mean > 0 and (max_mean / min_mean > 10):
            use_log_scale_for_time = True


# --- 3. Setup for Plotting and Color Palette ---

# Define constants for colors
RETRAINED_MODEL_NAME = 'Retrained model'

COLOR_PALETTE = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']


RETRAINED_COLOR = COLOR_PALETTE[-3]  # A distinct color for the retrained model
# OTHER_MODEL_COLOR = COLOR_PALETTE[3] # A single color for all other unlearning models
OTHER_MODEL_COLOR = COLOR_PALETTE[1]
ORIGINAL_MODEL_COLOR = OTHER_MODEL_COLOR

# Define font sizes for consistency
TITLE_FONTSIZE = 26
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18

# --- 4. Plotting Full Comparison ---
print("--- Generating Full Comparison Plots ---")
for metric in metrics_to_plot:
    if metric not in aggregated_df.columns.get_level_values(0):
        continue

    means = aggregated_df[(metric, 'mean')]
    stds = aggregated_df[(metric, 'std')].fillna(0)
    is_accuracy = 'accuracy' in metric
    sorted_model_names = list(means.sort_values(ascending=not is_accuracy).index)
    plot_means = means.reindex(sorted_model_names)
    plot_stds = stds.reindex(sorted_model_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Assign a distinct color to the retrained model and a single color to all others
    colors = [RETRAINED_COLOR if model == RETRAINED_MODEL_NAME else OTHER_MODEL_COLOR for model in plot_means.index]

    ax.bar(plot_means.index, plot_means, yerr=plot_stds, capsize=5, color=colors, edgecolor='black')

    if RETRAINED_MODEL_NAME in plot_means.index:
        retrained_mean = plot_means[RETRAINED_MODEL_NAME]
        retrained_std = plot_stds.get(RETRAINED_MODEL_NAME, 0)
        ax.axhline(y=retrained_mean, color="black", linestyle='--', linewidth=2)
        
        # make axhline for each std dev
        for i in range(1, 2):
            ax.axhline(y=retrained_mean + i*retrained_std, color="black", linestyle='--', linewidth=1)
            ax.axhline(y=retrained_mean - i*retrained_std, color="black", linestyle='--', linewidth=1)

        ax.axhspan(retrained_mean - retrained_std, retrained_mean + retrained_std, color=RETRAINED_COLOR, alpha=0.40)

    plot_title = metric.replace('-', ' ').replace('_', ' ').title()
    ax.set_title(f'Model Comparison: {plot_title}', fontsize=TITLE_FONTSIZE)
    ax.set_ylabel(plot_title, fontsize=LABEL_FONTSIZE)
    plt.xticks(rotation=45, ha='right', fontsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    ax.set_axisbelow(True)

    if 'accuracy' in metric:
        ax.set_ylim(0, 1.05)
    elif metric == time_metric and global_max_time is not None:
        if use_log_scale_for_time:
            ax.set_yscale('log')
            ax.set_ylabel(f'{plot_title} (Log Scale)', fontsize=LABEL_FONTSIZE)
            ax.set_ylim(top=global_max_time * 1.15)
        else:
            ax.set_ylim(bottom=0, top=global_max_time * 1.1)

    plt.tight_layout()
    print(f"Generating and saving plot for {metric}...")
    save_dir = os.path.join(script_dir, results_folder, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{metric}_comparison.pdf'))
    plt.close()

print("\nFull comparison plots have been generated and saved.")
print("\n" + "="*80 + "\n")


# --- 5. Plotting for Original Model Baselines ---
print("--- Generating Original Model Baseline Plots ---")

if df_originals.empty:
    print("No data found for original models. Skipping baseline plots.")
else:
    for metric in metrics_to_plot:
        if metric not in agg_originals_df.columns.get_level_values(0):
            continue

        means = agg_originals_df[(metric, 'mean')]
        stds = agg_originals_df[(metric, 'std')].fillna(0)
        is_accuracy = 'accuracy' in metric
        sorted_model_names = list(means.sort_values(ascending=not is_accuracy).index)
        plot_means = means.reindex(sorted_model_names)
        plot_stds = stds.reindex(sorted_model_names)

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use a single, distinct color for these baseline plots
        ax.bar(plot_means.index, plot_means, yerr=plot_stds, capsize=5, color=ORIGINAL_MODEL_COLOR, edgecolor='black')

        plot_title = metric.replace('-', ' ').replace('_', ' ').title()
        ax.set_title(f'Original Models: {plot_title}', fontsize=TITLE_FONTSIZE)
        ax.set_ylabel(plot_title, fontsize=LABEL_FONTSIZE)
        plt.xticks(rotation=45, ha='right', fontsize=TICK_FONTSIZE)
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
        ax.set_axisbelow(True)

        if 'accuracy' in metric:
            ax.set_ylim(0, 1.05)
        elif metric == time_metric and global_max_time is not None:
            if use_log_scale_for_time:
                ax.set_yscale('log')
                ax.set_ylabel(f'{plot_title} (Log Scale)', fontsize=LABEL_FONTSIZE)
                ax.set_ylim(top=global_max_time * 1.15)
            else:
                ax.set_ylim(bottom=0, top=global_max_time * 1.1)

        plt.tight_layout()
        print(f"Generating and saving baseline plot for {metric}...")
        save_dir = os.path.join(script_dir, results_folder, 'plots')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{metric}_originals_baseline.pdf'))
        plt.close()

    print("\nOriginal model baseline plots have been generated and saved.")
