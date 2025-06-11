import matplotlib.pyplot as plt
import numpy as np
import pydantic

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
    
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Objective vs Alpha
    plt.subplot(1, 2, 1)
    sc = plt.scatter(alphas, targets, c=sampling_order, cmap='coolwarm', s=50, edgecolor='k', alpha=0.7)
    plt.xlabel('Alpha')
    plt.ylabel('Objective Value')
    plt.title('Objective vs Alpha')
    plt.colorbar(sc, label='Sampling Order (0=early, 1=late)')
    
    # Plot 2: Objective vs Lambda
    plt.subplot(1, 2, 2)
    sc = plt.scatter(lambdas, targets, c=sampling_order, cmap='coolwarm', s=50, edgecolor='k', alpha=0.7)
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
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Objective value over iterations
    plt.subplot(1, 3, 1)
    plt.plot(iterations, targets, 'o-', alpha=0.7, label='Sampled Points')
    plt.plot(iterations, running_max, 'r-', linewidth=2, label='Best So Far')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Optimization Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Improvement over iterations
    plt.subplot(1, 3, 2)
    improvements = np.diff(running_max)
    plt.plot(iterations[1:], improvements, 'g-', marker='o', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Improvement')
    plt.title('Improvement per Iteration')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of objective values
    plt.subplot(1, 3, 3)
    plt.hist(targets, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(targets.max(), color='red', linestyle='--', linewidth=2, label=f'Best: {targets.max():.4f}')
    plt.xlabel('Objective Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Objective Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()