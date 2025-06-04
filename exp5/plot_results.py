import os
import json
import matplotlib.pyplot as plt

def plot_mia_metrics(models, figsize=(12, 8)):
    """
    Plot MIA metrics for multiple models.
    
    Args:
        models: Dictionary where keys are model names and values are dictionaries
               containing MIA metrics
        figsize: Tuple for figure size (width, height)
    """
    plt.style.use('default')
    colors = [ '#2f4b7c','#a05195','#f95d6a','#ffa600','#003f5c','#665191','#d45087','#ff7c43']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle('Model MIA Comparison Across Epochs', fontsize=16, fontweight='bold')
    

    # Plot each model
    for j, (model_name, model_data) in enumerate(models.items()):
        if 'mia' in model_data:
            acc_values = model_data['mia']
            epochs = range(1, len(acc_values) + 1)
            
            ax.plot(epochs, acc_values, 
                   marker='o', linewidth=2, markersize=4,
                   color=colors[j % len(colors)], 
                   label=model_name)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MIA Probability')
    ax.set_title('MIA Probability')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return plt

def plot_model_accuracies(models, figsize=(12, 8)):
    """
    Plot accuracy curves for multiple models showing retain, forget, and val accuracies.
    
    Args:
        models: Dictionary where keys are model names and values are dictionaries
               with 'retain', 'forget', 'val' keys containing accuracy lists
        figsize: Tuple for figure size (width, height)
    """
    plt.style.use('default')
    colors = [ '#2f4b7c','#a05195','#f95d6a','#ffa600','#003f5c','#665191','#d45087','#ff7c43']
    
    # Create subplots - one for each metric
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Model Accuracy Comparison Across Epochs', fontsize=16, fontweight='bold')
    
    metrics = ['retain', 'forget', 'val']
    metric_titles = ['Retain Accuracy', 'Forget Accuracy', 'Validation Accuracy']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i]
        
        # Plot each model
        for j, (model_name, model_data) in enumerate(models.items()):
            if metric in model_data and 'acc' in model_data[metric]:
                acc_values = model_data[metric]['acc']
                epochs = range(1, len(acc_values) + 1)
                
                ax.plot(epochs, acc_values, 
                       marker='o', linewidth=2, markersize=4,
                       color=colors[j % len(colors)], 
                       label=model_name)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return plt

def main():
    # Define the datasets and seeds to process
    datasets = ['MNIST']  # Add more datasets as needed
    seeds = [42]  # Add more seeds as needed
    
    for dataset in datasets:
        for seed in seeds:
            # Define paths
            results_dir = os.path.join('results', dataset, f'seed_{seed}')
            metrics_file = os.path.join(results_dir, 'all_metrics.json')
            
            # Check if metrics file exists
            if not os.path.exists(metrics_file):
                print(f"No metrics file found for {dataset} with seed {seed}")
                continue
                
            print(f"Processing {dataset} with seed {seed}")
            
            # Load metrics
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
            
            # Create plots
            # MIA metrics plot
            mia_plot = plot_mia_metrics(all_metrics)
            # mia_plot.savefig(os.path.join(results_dir, 'mia_metrics.png'))
            plt.show()
            plt.close()
            
            # Model accuracies plot
            acc_plot = plot_model_accuracies(all_metrics)

            # plt.show()
            # acc_plot.savefig(os.path.join(results_dir, 'model_accuracies.png'))
            
            plt.close()
            
            print(f"Plots saved in {results_dir}")

if __name__ == "__main__":
    main()
