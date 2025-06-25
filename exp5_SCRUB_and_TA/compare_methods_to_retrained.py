import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def load_mnist_results(json_path):
    """Load the MNIST results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_method_scores(data):
    """Calculate scores for each method - similarity to retrained for accuracy metrics, speed for time"""
    
    # Find retrained model baseline
    retrained = None
    methods_data = []
    
    for entry in data:
        if entry["Model name"] == "Retrained model":
            retrained = entry
        else:
            methods_data.append(entry)
    
    if retrained is None:
        raise ValueError("Retrained model not found in data")
    
    # Calculate scores for each method
    scored_data = []
    
    # Get min/max values for normalization
    all_times = [method["Time (sec)"][0] for method in methods_data]
    min_time = min(all_times)
    max_time = max(all_times)
    
    for method in methods_data:
        # Calculate absolute differences from retrained model for accuracy metrics
        mia_diff = abs(method["MIA-probability"][0] - retrained["MIA-probability"][0])
        retain_diff = abs(method["Retain-accuracy"][0] - retrained["Retain-accuracy"][0])
        forget_diff = abs(method["Forget-accuracy"][0] - retrained["Forget-accuracy"][0])
        val_diff = abs(method["Val-accuracy"][0] - retrained["Val-accuracy"][0])
        
        # Calculate error propagation for differences: sqrt(σ1² + σ2²)
        mia_diff_err = np.sqrt(method["MIA-probability"][1]**2 + retrained["MIA-probability"][1]**2)
        retain_diff_err = np.sqrt(method["Retain-accuracy"][1]**2 + retrained["Retain-accuracy"][1]**2)
        forget_diff_err = np.sqrt(method["Forget-accuracy"][1]**2 + retrained["Forget-accuracy"][1]**2)
        val_diff_err = np.sqrt(method["Val-accuracy"][1]**2 + retrained["Val-accuracy"][1]**2)
        
        # Convert differences to similarities (1 = identical, 0 = very different)
        mia_sim = 1 - min(mia_diff / 0.5, 1.0)  # Max reasonable MIA diff = 0.5
        retain_sim = 1 - min(retain_diff / 0.2, 1.0)  # Max reasonable accuracy diff = 0.2
        forget_sim = 1 - min(forget_diff / 0.2, 1.0)
        val_sim = 1 - min(val_diff / 0.2, 1.0)
        
        # For time: normalize speed score (faster = better, 1 = fastest, 0 = slowest)
        time_score = 1 - (method["Time (sec)"][0] - min_time) / (max_time - min_time) if max_time > min_time else 1.0
        
        # Overall score combining similarity to retrained (accuracy metrics) + speed
        overall_score = (mia_sim + retain_sim + forget_sim + val_sim + time_score) / 5
        
        scored_data.append({
            'method': method["Model name"],
            'mia_diff': mia_diff,
            'retain_diff': retain_diff,
            'forget_diff': forget_diff,
            'val_diff': val_diff,
            'mia_diff_err': mia_diff_err,
            'retain_diff_err': retain_diff_err,
            'forget_diff_err': forget_diff_err,
            'val_diff_err': val_diff_err,
            'time_raw': method["Time (sec)"][0],
            'time_err': method["Time (sec)"][1],
            'mia_sim': mia_sim,
            'retain_sim': retain_sim,
            'forget_sim': forget_sim,
            'val_sim': val_sim,
            'time_score': time_score,
            'overall_score': overall_score,
            'raw_data': method
        })
    
    return retrained, scored_data

def create_comparison_plots(retrained, scored_data, save_path=None):
    """Create plots showing method performance with error bars"""
    
    # Sort by overall score
    scored_data.sort(key=lambda x: x['overall_score'], reverse=True)
    
    methods = [item['method'] for item in scored_data]
    overall_scores = [item['overall_score'] for item in scored_data]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('MNIST Methods: Performance Comparison with Error Bars\n(Similarity to Retrained for Accuracy + Speed for Time)', 
                 fontsize=16, fontweight='bold')
    
    # Color scheme - green for high scores, red for low scores
    colors = plt.cm.RdYlGn([score for score in overall_scores])
    
    # 1. Overall Performance Score (no error bars since it's a composite score)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(methods)), overall_scores, color=colors, alpha=0.8)
    ax1.set_title('Overall Performance Score\n(Similarity to Retrained + Speed)', 
                  fontweight='bold', fontsize=12)
    ax1.set_ylabel('Performance Score (0-1)')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Add score labels on bars
    for i, (bar, score) in enumerate(zip(bars1, overall_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. MIA Probability Difference
    ax2 = axes[0, 1]
    mia_diffs = [item['mia_diff'] for item in scored_data]
    mia_errs = [item['mia_diff_err'] for item in scored_data]
    bars2 = ax2.bar(range(len(methods)), mia_diffs, yerr=mia_errs, 
                   color=colors, alpha=0.8, capsize=3)
    ax2.set_title(f'MIA Probability Difference\nRetrained: {retrained["MIA-probability"][0]:.3f}±{retrained["MIA-probability"][1]:.3f}', 
                  fontweight='bold', fontsize=12)
    ax2.set_ylabel('|Method - Retrained|')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Retain Accuracy Difference
    ax3 = axes[0, 2]
    retain_diffs = [item['retain_diff'] for item in scored_data]
    retain_errs = [item['retain_diff_err'] for item in scored_data]
    bars3 = ax3.bar(range(len(methods)), retain_diffs, yerr=retain_errs, 
                   color=colors, alpha=0.8, capsize=3)
    ax3.set_title(f'Retain Accuracy Difference\nRetrained: {retrained["Retain-accuracy"][0]:.3f}±{retrained["Retain-accuracy"][1]:.3f}', 
                  fontweight='bold', fontsize=12)
    ax3.set_ylabel('|Method - Retrained|')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Forget Accuracy Difference
    ax4 = axes[1, 0]
    forget_diffs = [item['forget_diff'] for item in scored_data]
    forget_errs = [item['forget_diff_err'] for item in scored_data]
    bars4 = ax4.bar(range(len(methods)), forget_diffs, yerr=forget_errs, 
                   color=colors, alpha=0.8, capsize=3)
    ax4.set_title(f'Forget Accuracy Difference\nRetrained: {retrained["Forget-accuracy"][0]:.3f}±{retrained["Forget-accuracy"][1]:.3f}', 
                  fontweight='bold', fontsize=12)
    ax4.set_ylabel('|Method - Retrained|')
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Validation Accuracy Difference
    ax5 = axes[1, 1]
    val_diffs = [item['val_diff'] for item in scored_data]
    val_errs = [item['val_diff_err'] for item in scored_data]
    bars5 = ax5.bar(range(len(methods)), val_diffs, yerr=val_errs, 
                   color=colors, alpha=0.8, capsize=3)
    ax5.set_title(f'Val Accuracy Difference\nRetrained: {retrained["Val-accuracy"][0]:.3f}±{retrained["Val-accuracy"][1]:.3f}', 
                  fontweight='bold', fontsize=12)
    ax5.set_ylabel('|Method - Retrained|')
    ax5.set_xticks(range(len(methods)))
    ax5.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Execution Time (Raw Values)
    ax6 = axes[1, 2]
    
    # Sort by time for this plot to show speed ranking clearly
    time_sorted_data = sorted(scored_data, key=lambda x: x['time_raw'])
    time_methods = [item['method'] for item in time_sorted_data]
    time_values = [item['time_raw'] for item in time_sorted_data]
    time_errors = [item['time_err'] for item in time_sorted_data]
    time_colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(time_values)))  # Reverse colors for time
    
    bars6 = ax6.bar(range(len(time_methods)), time_values, yerr=time_errors,
                   color=time_colors, alpha=0.8, capsize=3)
    ax6.axhline(y=retrained["Time (sec)"][0], color='red', linestyle='--', linewidth=2, 
               label=f'Retrained: {retrained["Time (sec)"][0]:.1f}±{retrained["Time (sec)"][1]:.1f}s')
    
    # Add error bar for retrained baseline
    ax6.fill_between([-0.5, len(time_methods)-0.5], 
                    retrained["Time (sec)"][0] - retrained["Time (sec)"][1],
                    retrained["Time (sec)"][0] + retrained["Time (sec)"][1],
                    color='red', alpha=0.2)
    
    ax6.set_title(f'Execution Time', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Time (seconds)')
    ax6.set_yscale('log')  # Log scale for better visualization
    ax6.set_xticks(range(len(time_methods)))
    ax6.set_xticklabels(time_methods, rotation=45, ha='right', fontsize=10)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig

def create_raw_metrics_plot(retrained, scored_data, save_path=None):
    """Create a plot showing raw metric values with error bars"""
    
    # Sort by overall score
    scored_data.sort(key=lambda x: x['overall_score'], reverse=True)
    
    methods = [item['method'] for item in scored_data]
    
    # Create figure with subplots for raw metrics
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('MNIST Methods: Raw Metric Values with Error Bars', 
                 fontsize=16, fontweight='bold')
    
    # Color scheme
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    retrained_color = 'red'
    
    # 1. MIA Probability
    ax1 = axes[0, 0]
    mia_values = [item['raw_data']["MIA-probability"][0] for item in scored_data]
    mia_errors = [item['raw_data']["MIA-probability"][1] for item in scored_data]
    bars1 = ax1.bar(range(len(methods)), mia_values, yerr=mia_errors, 
                   color=colors, alpha=0.7, capsize=3)
    ax1.axhline(y=retrained["MIA-probability"][0], color=retrained_color, 
               linestyle='--', linewidth=2, label=f'Retrained: {retrained["MIA-probability"][0]:.3f}±{retrained["MIA-probability"][1]:.3f}')
    ax1.fill_between([-0.5, len(methods)-0.5], 
                    retrained["MIA-probability"][0] - retrained["MIA-probability"][1],
                    retrained["MIA-probability"][0] + retrained["MIA-probability"][1],
                    color=retrained_color, alpha=0.2)
    ax1.set_title('MIA Probability', fontweight='bold')
    ax1.set_ylabel('MIA Probability')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Retain Accuracy
    ax2 = axes[0, 1]
    retain_values = [item['raw_data']["Retain-accuracy"][0] for item in scored_data]
    retain_errors = [item['raw_data']["Retain-accuracy"][1] for item in scored_data]
    bars2 = ax2.bar(range(len(methods)), retain_values, yerr=retain_errors, 
                   color=colors, alpha=0.7, capsize=3)
    ax2.axhline(y=retrained["Retain-accuracy"][0], color=retrained_color, 
               linestyle='--', linewidth=2, label=f'Retrained: {retrained["Retain-accuracy"][0]:.3f}±{retrained["Retain-accuracy"][1]:.3f}')
    ax2.fill_between([-0.5, len(methods)-0.5], 
                    retrained["Retain-accuracy"][0] - retrained["Retain-accuracy"][1],
                    retrained["Retain-accuracy"][0] + retrained["Retain-accuracy"][1],
                    color=retrained_color, alpha=0.2)
    ax2.set_title('Retain Accuracy', fontweight='bold')
    ax2.set_ylabel('Retain Accuracy')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Forget Accuracy
    ax3 = axes[0, 2]
    forget_values = [item['raw_data']["Forget-accuracy"][0] for item in scored_data]
    forget_errors = [item['raw_data']["Forget-accuracy"][1] for item in scored_data]
    bars3 = ax3.bar(range(len(methods)), forget_values, yerr=forget_errors, 
                   color=colors, alpha=0.7, capsize=3)
    ax3.axhline(y=retrained["Forget-accuracy"][0], color=retrained_color, 
               linestyle='--', linewidth=2, label=f'Retrained: {retrained["Forget-accuracy"][0]:.3f}±{retrained["Forget-accuracy"][1]:.3f}')
    ax3.fill_between([-0.5, len(methods)-0.5], 
                    retrained["Forget-accuracy"][0] - retrained["Forget-accuracy"][1],
                    retrained["Forget-accuracy"][0] + retrained["Forget-accuracy"][1],
                    color=retrained_color, alpha=0.2)
    ax3.set_title('Forget Accuracy', fontweight='bold')
    ax3.set_ylabel('Forget Accuracy')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Validation Accuracy
    ax4 = axes[1, 0]
    val_values = [item['raw_data']["Val-accuracy"][0] for item in scored_data]
    val_errors = [item['raw_data']["Val-accuracy"][1] for item in scored_data]
    bars4 = ax4.bar(range(len(methods)), val_values, yerr=val_errors, 
                   color=colors, alpha=0.7, capsize=3)
    ax4.axhline(y=retrained["Val-accuracy"][0], color=retrained_color, 
               linestyle='--', linewidth=2, label=f'Retrained: {retrained["Val-accuracy"][0]:.3f}±{retrained["Val-accuracy"][1]:.3f}')
    ax4.fill_between([-0.5, len(methods)-0.5], 
                    retrained["Val-accuracy"][0] - retrained["Val-accuracy"][1],
                    retrained["Val-accuracy"][0] + retrained["Val-accuracy"][1],
                    color=retrained_color, alpha=0.2)
    ax4.set_title('Validation Accuracy', fontweight='bold')
    ax4.set_ylabel('Validation Accuracy')
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Time (sorted by speed)
    ax5 = axes[1, 1]
    time_sorted_data = sorted(scored_data, key=lambda x: x['time_raw'])
    time_methods = [item['method'] for item in time_sorted_data]
    time_values = [item['time_raw'] for item in time_sorted_data]
    time_errors = [item['time_err'] for item in time_sorted_data]
    time_colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(time_values)))
    
    bars5 = ax5.bar(range(len(time_methods)), time_values, yerr=time_errors,
                   color=time_colors, alpha=0.7, capsize=3)
    ax5.axhline(y=retrained["Time (sec)"][0], color=retrained_color, 
               linestyle='--', linewidth=2, label=f'Retrained: {retrained["Time (sec)"][0]:.1f}±{retrained["Time (sec)"][1]:.1f}s')
    ax5.set_title('Execution Time', fontweight='bold')
    ax5.set_ylabel('Time (seconds)')
    ax5.set_yscale('log')
    ax5.set_xticks(range(len(time_methods)))
    ax5.set_xticklabels(time_methods, rotation=45, ha='right', fontsize=10)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Combined Privacy-Utility Score
    ax6 = axes[1, 2]
    # Calculate combined score with error propagation
    combined_scores = []
    combined_errors = []
    
    for item in scored_data:
        method = item['raw_data']
        # Privacy score: 1 - MIA (higher = better privacy)
        privacy_score = 1 - method["MIA-probability"][0]
        privacy_error = method["MIA-probability"][1]  # Error propagates directly
        
        # Utility score: retain accuracy
        utility_score = method["Retain-accuracy"][0]
        utility_error = method["Retain-accuracy"][1]
        
        # Combined score (average)
        combined = (privacy_score + utility_score) / 2
        # Error propagation for average: sqrt((σ1/2)² + (σ2/2)²)
        combined_err = np.sqrt((privacy_error/2)**2 + (utility_error/2)**2)
        
        combined_scores.append(combined)
        combined_errors.append(combined_err)
    
    # Calculate retrained baseline
    retrained_privacy = 1 - retrained["MIA-probability"][0]
    retrained_utility = retrained["Retain-accuracy"][0]
    retrained_combined = (retrained_privacy + retrained_utility) / 2
    retrained_combined_err = np.sqrt((retrained["MIA-probability"][1]/2)**2 + (retrained["Retain-accuracy"][1]/2)**2)
    
    bars6 = ax6.bar(range(len(methods)), combined_scores, yerr=combined_errors,
                   color=colors, alpha=0.7, capsize=3)
    ax6.axhline(y=retrained_combined, color=retrained_color, 
               linestyle='--', linewidth=2, label=f'Retrained: {retrained_combined:.3f}±{retrained_combined_err:.3f}')
    ax6.fill_between([-0.5, len(methods)-0.5], 
                    retrained_combined - retrained_combined_err,
                    retrained_combined + retrained_combined_err,
                    color=retrained_color, alpha=0.2)
    ax6.set_title('Privacy-Utility Score', fontweight='bold')
    ax6.set_ylabel('Combined Score')
    ax6.set_xticks(range(len(methods)))
    ax6.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        raw_path = save_path.parent / "mnist_raw_metrics.pdf"
        plt.savefig(raw_path, dpi=300, bbox_inches='tight')
        print(f"Raw metrics plot saved to {raw_path}")
    
    plt.show()
    
    return fig

def main():
    # Path to the JSON file
    json_path = Path("exp5_SCRUB_and_TA/json_results/mnist.json")
    
    # Load data
    print("Loading MNIST results...")
    data = load_mnist_results(json_path)
    print(f"Loaded {len(data)} methods")
    
    # Calculate performance scores
    print("\nCalculating performance scores...")
    retrained, scored_data = calculate_method_scores(data)
    
    # Create comparison plots
    print("\nCreating performance comparison plots...")
    save_path = Path("exp5_SCRUB_and_TA/plots/mnist_performance_comparison.pdf")
    save_path.parent.mkdir(exist_ok=True)
    
    fig1 = create_comparison_plots(retrained, scored_data, save_path)
    
    # Create raw metrics plot
    print("\nCreating raw metrics plots...")
    fig2 = create_raw_metrics_plot(retrained, scored_data, save_path)
    
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 