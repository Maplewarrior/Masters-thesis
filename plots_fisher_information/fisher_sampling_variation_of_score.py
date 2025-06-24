#!/usr/bin/env python3
"""
Fisher Information and Score Functions Visualization

This script implements and visualizes score functions for different probability distributions:
- Normal
- Poisson
- Bernoulli
- Cauchy

The score function is the derivative of the log-likelihood with respect to the parameter.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from typing import Callable, List, Tuple

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


def poisson_score(x: np.ndarray, theta: float) -> np.ndarray:
    """
    Score function for Poisson distribution with rate parameter theta.
    
    Args:
        x: Sample data
        theta: Rate parameter
        
    Returns:
        Score function values
    """
    return x/theta - 1  # For Poisson with rate theta, score = x/theta - 1


def bernoulli_score(x: np.ndarray, theta: float) -> np.ndarray:
    """
    Score function for Bernoulli distribution with probability theta.
    
    Args:
        x: Sample data (0 or 1)
        theta: Probability parameter
        
    Returns:
        Score function values
    """
    return x/theta - (1-x)/(1-theta)  # For Bernoulli with prob theta, score = x/theta - (1-x)/(1-theta)


def cauchy_score(x: np.ndarray, theta: float) -> np.ndarray:
    """
    Score function for Cauchy distribution with location parameter theta.
    
    Args:
        x: Sample data
        theta: Location parameter
        
    Returns:
        Score function values
    """
    return 2 * (x - theta) / (1 + (x - theta)**2)  # For Cauchy with location theta


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


def plot_score_functions(ax, samples: np.ndarray, score_func: Callable, 
                        param_range: np.ndarray, true_param: float, 
                        distribution_name: str, n_samples: int,
                        pdf_formula: str, score_formula: str) -> None:
    """
    Plot score functions for multiple datasets.
    
    Args:
        ax: Matplotlib axis
        samples: Sample datasets
        score_func: Score function
        param_range: Range of parameter values to evaluate
        true_param: True parameter value
        distribution_name: Name of the distribution
        n_samples: Number of samples per dataset
        pdf_formula: Formula for the PDF/PMF
        score_formula: Formula for the score function
    """
    # Use a modern color palette
    custom_palette = ['#d45087', '#f95d6a', '#ff7c43', '#ffa600']
    colors = sns.color_palette(custom_palette, n_colors=samples.shape[0])
    
    for i in range(samples.shape[0]):
        sample = samples[i]
        scores = np.array([np.mean(score_func(sample, theta)) for theta in param_range])
        ax.plot(param_range, scores, color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=true_param, color='gray', linestyle='--', alpha=0.7)
    
    # Add annotation for true parameter
    ax.text(true_param, ax.get_ylim()[0]*0.9, 
            f"True $\\theta={true_param}$", 
            ha='center', va='bottom', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
    
    ax.set_xlabel(r'$\theta$', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    
    # Create a title with the distribution and formulas
    title = f"{distribution_name} Distribution (n={n_samples})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add formulas as text in the top right corner
    formula_text = pdf_formula + '\n' + score_formula
    ax.text(0.98, 0.98, formula_text,
            ha='right', va='top', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5),
            fontsize=11)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    """Main function to generate plots."""
    # Set a modern style with LaTeX rendering
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 14,
        'mathtext.fontset': 'cm',  # Use Computer Modern for math text
        'figure.figsize': (15, 5),  # Modified for single row
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    n_samples = 30
    n_datasets = 10
    
    # Create figure with subplots - single row of three
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Add a main title to the figure
    fig.suptitle("Sampling Variation of Score Functions", fontsize=18, fontweight='bold', y=1.05)
    
    # 1. Normal distribution
    true_theta_normal = 6
    param_range_normal = np.linspace(4, 10, 100)
    normal_samples = generate_samples(stats.norm, true_theta_normal, n_samples, n_datasets, scale=1)
    normal_pdf = r"$p(x|\theta) = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(x-\theta)^2}$"
    normal_score_formula = r"$s(\theta) = x - \theta$"
    plot_score_functions(axes[0], normal_samples, normal_score, 
                        param_range_normal, true_theta_normal, 
                        "Normal", n_samples,
                        normal_pdf, normal_score_formula)
    
    # 2. Poisson distribution
    true_theta_poisson = 6
    param_range_poisson = np.linspace(3, 10, 100)
    poisson_samples = generate_samples(stats.poisson, true_theta_poisson, n_samples, n_datasets)
    poisson_pmf = r"$p(x|\theta) = \frac{\theta^x e^{-\theta}}{x!}$"
    poisson_score_formula = r"$s(\theta) = \frac{x}{\theta} - 1$"
    plot_score_functions(axes[1], poisson_samples, poisson_score, 
                        param_range_poisson, true_theta_poisson, 
                        "Poisson", n_samples,
                        poisson_pmf, poisson_score_formula)
    
    # 3. Bernoulli distribution
    true_theta_bernoulli = 0.6
    param_range_bernoulli = np.linspace(0.1, 0.9, 100)
    bernoulli_samples = generate_samples(stats.bernoulli, true_theta_bernoulli, n_samples, n_datasets)
    bernoulli_pmf = r"$p(x|\theta) = \theta^x (1-\theta)^{1-x}$"
    bernoulli_score_formula = r"$s(\theta) = \frac{x}{\theta} - \frac{1-x}{1-\theta}$"
    plot_score_functions(axes[2], bernoulli_samples, bernoulli_score, 
                        param_range_bernoulli, true_theta_bernoulli, 
                        "Bernoulli", n_samples,
                        bernoulli_pmf, bernoulli_score_formula)
    
    # Adjust spacing between subplots for better readability
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save as PDF with high resolution
    plt.savefig('score_functions.pdf', format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
    
    # Also save as PNG for quick viewing
    plt.savefig('score_functions.png', dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    main()