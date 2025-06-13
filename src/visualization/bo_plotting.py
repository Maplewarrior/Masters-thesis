import matplotlib.pyplot as plt
import numpy as np
import pydantic
from matplotlib.colors import LinearSegmentedColormap

# Professional color palette
professional_colors = ['#ffa600', '#a05195', '#f95d6a']

class ParamItem(pydantic.BaseModel):
    alpha_0: float
    _lambda_0: float
    alpha_1: float
    _lambda_1: float


def plot_bo_samples_basic(bo_result: list[ParamItem], savepath=None):
    """Plot the basic objective function samples from BoTorch optimization"""
    # Extract data from results
    alphas = np.array([bo_result[i]['params']['alpha_0'] for i in range(len(bo_result))])
    lambdas = np.array([bo_result[i]['params']['_lambda_0'] for i in range(len(bo_result))])
    targets = np.array([bo_result[i]['target'] for i in range(len(bo_result))])
    
    # Create sampling order for color coding
    sampling_order = np.linspace(0, 1, len(alphas))
    
    # Create custom colormap from professional colors
    custom_cmap = LinearSegmentedColormap.from_list("professional", professional_colors)
    
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Objective vs Alpha
    plt.subplot(1, 2, 1)
    sc = plt.scatter(alphas, targets, c=sampling_order, cmap=custom_cmap, s=50, edgecolor='k', alpha=0.7)
    plt.xlabel('Alpha')
    plt.ylabel('Objective Value')
    plt.title('Objective vs Alpha')
    plt.colorbar(sc, label='Sampling Order (0=early, 1=late)')
    
    # Plot 2: Objective vs Lambda
    plt.subplot(1, 2, 2)
    sc = plt.scatter(lambdas, targets, c=sampling_order, cmap=custom_cmap, s=50, edgecolor='k', alpha=0.7)
    plt.xlabel('Lambda')
    plt.ylabel('Objective Value')
    plt.title('Objective vs Lambda')
    plt.colorbar(sc, label='Sampling Order (0=early, 1=late)')
        
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()

def plot_convergence_analysis(bo_result: list[ParamItem], savepath=None):
    """Plot convergence behavior of the BoTorch optimization"""
    
    targets = np.array([bo_result[i]['target'] for i in range(len(bo_result))])
    iterations = np.arange(len(targets))
    
    # Calculate running maximum (best found so far)
    running_max = np.maximum.accumulate(targets)
    total_best_ind = np.argmax(running_max)
    
    plt.figure(figsize=(10, 5))
    
    # Plot 1: Objective value over iterations
    plt.plot(iterations, targets, 'o-', alpha=0.7, color=professional_colors[0], label='Sampled Points')
    plt.plot(iterations, running_max, '-', linewidth=2, color=professional_colors[1], label='Best So Far')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Optimization Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(total_best_ind, color=professional_colors[2], linestyle='--', linewidth=2, label=f'Best: {targets[total_best_ind]:.4f} at iteration {total_best_ind}')
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()