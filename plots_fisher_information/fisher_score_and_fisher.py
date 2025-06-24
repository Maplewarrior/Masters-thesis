#!/usr/bin/env python3
"""
Gaussian Score Function Visualization

This script implements and visualizes the score function for the Gaussian (Normal) distribution.
The score function is the derivative of the log-likelihood with respect to the parameter.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from typing import Callable
from scipy.stats import gaussian_kde

# Set plot style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})


def normal_score(x: np.ndarray, theta: float) -> np.ndarray:
    """
    Score function for normal distribution with mean theta and variance 1.
    
    Args:
        x: Sample data
        theta: Mean parameter
        
    Returns:
        Score function values
    """
    return x - theta  # For normal with mean theta and variance 1, score = x - theta


def generate_samples(dist_func: Callable, true_param: float, n_samples: int, 
                    n_datasets: int, **kwargs) -> np.ndarray:
    """
    Generate multiple datasets from a distribution.
    
    Args:
        dist_func: Distribution function from scipy.stats
        true_param: True parameter value
        n_samples: Number of samples per dataset
        n_datasets: Number of datasets to generate
        **kwargs: Additional parameters for the distribution
        
    Returns:
        Array of shape (n_datasets, n_samples) containing the samples
    """
    samples = np.zeros((n_datasets, n_samples))
    for i in range(n_datasets):
        samples[i] = dist_func.rvs(true_param, size=n_samples, **kwargs)
    return samples


def plot_gaussian_score_function(samples: np.ndarray, param_range: np.ndarray, 
                               true_param: float, n_samples: int, ax=None) -> None:
    """
    Plot score functions for multiple Gaussian datasets.
    
    Args:
        samples: Sample datasets
        param_range: Range of parameter values to evaluate
        true_param: True parameter value
        n_samples: Number of samples per dataset
        ax: Matplotlib axis to plot on (optional)
    """
    # If no axis is provided, create a new figure
    if ax is None:
        # Reset any previous plot settings
        plt.close('all')
        plt.figure(figsize=(8, 6), dpi=100)
        
        # Set up the figure with LaTeX rendering
        plt.rcParams.update({
            'font.size': 14,
            'mathtext.fontset': 'cm',
            'figure.figsize': (8, 6),
            'figure.dpi': 100,  # Reduced DPI for display
            'savefig.dpi': 600,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use a single blue color instead of viridis colormap
    blue_color = '#665191'  # A nice matplotlib blue


    # Plot individual score values for each sample point
    for i in range(samples.shape[0]):
        sample_dataset = samples[i]
        
        # For each data point in the dataset
        for j in range(len(sample_dataset)):
            x_value = sample_dataset[j]
            
            # Calculate score function for this individual point across parameter range
            individual_scores = np.array([x_value - theta for theta in param_range])
            
            # Plot with increased alpha for better visibility
            ax.plot(param_range, individual_scores, color=blue_color, alpha=0.2, linewidth=0.8)
    
    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    # ax.axvline(x=true_param, color='gray', linestyle='--', alpha=0.7)
    
    # Add annotation for true parameter
    ax.text(true_param, ax.get_ylim()[0]*0.9, 
            f"True $\\theta={true_param}$", 
            ha='center', va='bottom', color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
    
    # Add labels and title
    ax.set_xlabel(r'$\theta$', fontsize=16)
    ax.set_ylabel('Score Function Value', fontsize=16)
    ax.set_title(f"Gaussian Score Function - Individual Values", 
                fontsize=18, fontweight='bold')
    
    # Add formulas as text
    pdf_formula = r"$p(x|\theta) = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(x-\theta)^2}$"
    score_formula = r"$s(\theta) = x - \theta$"
    formula_text = pdf_formula + '\n' + score_formula
    
    ax.text(0.98, 0.98, formula_text,
            ha='right', va='top', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5),
            fontsize=14)
    
    # # Add explanation text
    # explanation = (
    #     "Each line represents the score function for an individual data point.\n"
    #     "The score function equals zero when $\\theta = x$.\n"
    #     "At the true parameter, scores are distributed around zero."
    # )Now 
    # ax.text(0.02, 0.02, explanation,
    #         ha='left', va='bottom', transform=ax.transAxes,
    #         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5),
    #         fontsize=12, style='italic')
    
    # Clean up the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # If we created a new figure, show it
    if ax is None:
        plt.tight_layout()
        plt.show()



def main():
    """Main function to generate plots in a grid layout with aligned axes."""
    # Parameters
    n_samples = 25  # Number of samples per dataset
    n_datasets = 20  # Number of datasets to generate
    bins = 30
    true_theta_normal = 5  # True mean parameter
    param_range_normal = np.linspace(2, 8, 100)  # Range of parameter values to evaluate
    
    # Generate samples
    normal_samples = generate_samples(stats.norm, true_theta_normal, n_samples, n_datasets, scale=1)
    
    # Set up the figure with LaTeX rendering
    plt.rcParams.update({
        'font.size': 14,
        'mathtext.fontset': 'cm',
        'figure.figsize': (12, 8),  # Reduced height from 10 to 6
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Create a figure with GridSpec for custom layout
    fig = plt.figure(figsize=(12, 8))  # Reduced height from 10 to 6
    
    
    gs = fig.add_gridspec(1, 4)
    
    # Create main score function plot (3/4 of width)
    ax_score = fig.add_subplot(gs[0, :3])
    
    # Create distribution plot (1/4 of width)
    ax_dist = fig.add_subplot(gs[0, 3], sharey=ax_score)
    
    # Plot score function on the left
    plot_gaussian_score_function(normal_samples, param_range_normal, true_theta_normal, n_samples, ax=ax_score)
    
    # Calculate score values at true parameter for distribution plot
    all_scores = []
    for i in range(normal_samples.shape[0]):
        sample = normal_samples[i]
        scores = normal_score(sample, theta=true_theta_normal)
        all_scores.extend(scores)
    
    all_scores = np.array(all_scores)
    
    # Plot horizontal histogram (rotated distribution)
    histogram_color = '#ff7c43'  # Medium Purple - sophisticated purple tone
    sns.histplot(y=all_scores, color=histogram_color, alpha=0.8, ax=ax_dist, bins=bins)


    # Add reference line at zero
    ax_dist.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Calculate and display variance
    variance = np.var(all_scores)
    mean_score = np.mean(all_scores)
    std_dev = np.sqrt(variance)
    
    # Add standard deviation lines (±1σ)
    std_line_color = 'black'  # Classic black for maximum clarity
    ax_dist.axhline(y=mean_score + std_dev, color=std_line_color, linestyle='-.', alpha=0.8, linewidth=1.5)
    ax_dist.text(0.1, mean_score + std_dev + 0.2, "$+1\\sigma$", color=std_line_color, ha='left', va='bottom', fontweight='bold')
    
    # Add -1σ line
    ax_dist.axhline(y=mean_score - std_dev, color=std_line_color, linestyle='-.', alpha=0.8, linewidth=1.5)
    ax_dist.text(0.1, mean_score - std_dev - 0.2, "$-1\\sigma$", color=std_line_color, ha='left', va='top', fontweight='bold')
    
    # Add Fisher information horizontal line
    # Calculate the x-coordinates for the line (from 0 to 50% of the x-axis)
    x_min, x_max = ax_dist.get_xlim()
    x_range = x_max - x_min
    x_start = x_min
    x_end = x_min + x_range * 0.5
    
    # Draw a horizontal line with arrows at both ends
    fisher_info_color = 'black'  # Classic black for Fisher information
    ax_dist.annotate('', xy=(x_start, mean_score + std_dev), xytext=(x_start, mean_score - std_dev),
                    arrowprops=dict(arrowstyle='<->', color=fisher_info_color, lw=2))
    
    # Add Fisher information text with smaller font size
    ax_dist.text(x_start + x_range * 0.05, mean_score, 
                "Fisher Information = $I(\\theta) = 1$",
                color=fisher_info_color, ha='left', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3),
                fontsize=10)  # Reduced font size
    
    # Add variance annotation with smaller font size
    ax_dist.text(0.95, 0.05, 
                f"Variance = {variance:.4f}\n$\\approx$ Fisher Information", 
                ha='right', va='bottom', transform=ax_dist.transAxes,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3),
                fontsize=10)  # Reduced font size
    
    # Add title and labels
    ax_dist.set_title("Score Distribution\nat True Parameter", fontsize=12)  # Reduced font size
    ax_dist.set_xlabel("Frequency", fontsize=12)
    ax_dist.set_ylabel("")  # Remove y-label as it's shared with score plot
    
    # Hide y-ticks for distribution plot since they're shared
    ax_dist.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    # Clean up the distribution plot
    ax_dist.spines['top'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    
    # Create a vertical line with varying transparency based on density
    kde = gaussian_kde(all_scores)
    
    # Create y-values for the vertical line segments
    y_vals = np.linspace(ax_score.get_ylim()[0], ax_score.get_ylim()[1], 100)
    
    # Calculate density at each y-value
    density_vals = kde(y_vals)
    
    # Normalize density values to range from 0.1 to 1.0 for alpha
    min_alpha = 0.1
    max_alpha = 1.0
    normalized_density = min_alpha + (max_alpha - min_alpha) * (density_vals / np.max(density_vals))
    
    # Plot vertical line segments with varying transparency
    for i in range(len(y_vals)-1):
        ax_score.plot([true_theta_normal, true_theta_normal], 
                     [y_vals[i], y_vals[i+1]], 
                     color=histogram_color,  # Match histogram color
                     alpha=normalized_density[i],
                     linewidth=2)
    
    # Adjust layout and save
    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95])  # Reduced padding and adjusted rect
    
    # Save as PDF with high resolution
    plt.savefig('gaussian_score_and_fisher_information.pdf', format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
    
    # Also save as PNG for quick viewing
    plt.savefig('gaussian_score_and_fisher_information.png', dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    main()            
