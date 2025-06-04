import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy_metrics(results, retrained_model_results, figsize=(15, 5)):
    """
    Plot scrubbing metrics showing retain, forget, and val accuracies.
    
    Args:
        results: Dictionary containing metrics with 'retain', 'forget', 'val' keys
        figsize: Tuple for figure size (width, height)
    
    Returns:
        matplotlib.figure.Figure: Figure containing accuracy plots
    """
    fig_acc, axes_acc = plt.subplots(1, 3, figsize=figsize)
    fig_acc.suptitle('Scrubbing Accuracy Results Across Epochs', fontsize=16, fontweight='bold')
    
    metrics = ['retain', 'forget', 'val']
    metric_titles = ['Retain Accuracy', 'Forget Accuracy', 'Validation Accuracy']
    colors = ['#f95d6a','#665191','#ffa600']
    
    # Create empty lists to store line objects for the legend
    lines = []
    labels = []
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes_acc[i]
        
        acc_values = results[metric]['acc']
        epochs = range(1, len(acc_values) + 1)

        # Make a horizontal line at the retrained model accuracy
        retrain_acc_value = retrained_model_results[metric]['acc'][0]
        retrain_line = ax.axhline(y=retrain_acc_value, color=colors[0], linestyle='--', alpha=0.8, linewidth=1.5)
        
        # Plot accuracy values
        scrub_line = ax.plot(epochs, acc_values, 
                           marker='o', linewidth=2, markersize=4,
                           color=colors[1], alpha=0.7)[0]
        
        # Only store legend handles from the first plot
        if i == 0:
            lines.extend([retrain_line, scrub_line])
            labels.extend(['Retrained Model', 'Unlearned Model (SCRUB)'])

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    # Add a single legend at the bottom center of the figure
    fig_acc.legend(lines, labels, 
                  loc='center', 
                  bbox_to_anchor=(0.5, 0.02),
                  ncol=2,
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

def main():
    # Example results
    results_dir = 'results/MNIST/seed_42/scrub'
    with open(os.path.join(results_dir, 'scrub_metrics.json'), 'r') as f:
        results = json.load(f)
    mia_results = results['mia']
    
    # Create and save plots
    os.makedirs(results_dir, exist_ok=True)

    # load retrained model results
    with open(os.path.join(results_dir, 'retrained_model_metrics.json'), 'r') as f:
        retrained_model_results = json.load(f)
    
    fig_acc = plot_accuracy_metrics(results, retrained_model_results)
    fig_mia = plot_mia_metrics(mia_results, retrained_model_results)
    
    # Save plots
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    fig_acc.savefig(os.path.join(plots_dir, 'scrub_accuracy_results.png'), bbox_inches='tight')
    fig_mia.savefig(os.path.join(plots_dir, 'scrub_mia_results.png'), bbox_inches='tight')
    
    plt.close('all')

if __name__ == "__main__":
    main() 