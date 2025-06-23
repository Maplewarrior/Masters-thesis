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

def plot_convergence_analysis(bo_result: list[ParamItem], savepath=None, exploration_factor=None):
    """Plot convergence behavior of the BoTorch optimization"""
    
    targets = np.array([bo_result[i]['target'] for i in range(len(bo_result))])
    iterations = np.arange(len(targets))
    
    # Calculate running maximum (best found so far)
    running_max = np.maximum.accumulate(targets.round(4))
    total_best_ind = np.argmax(targets)
    
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['mathtext.fontset'] = 'stix'  # Use a LaTeX-like font for math text

    # Color palette from objective plot
    colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot 1: Objective value over iterations
    ax.plot(iterations, targets, 'o-', alpha=0.7, color=colors[0], label='Sampled Points')
    ax.plot(iterations, running_max, '-', linewidth=2.5, color=colors[6], label='Best So Far')
    ax.set_xlabel('Iteration', fontsize=26)
    ax.set_ylabel('$\\mathcal{L}_{BO}$', fontsize=26)
    ax.set_title(f'Optimization Convergence (Exploration Factor: ${exploration_factor}$)', fontsize=30)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axvline(total_best_ind, color=colors[4], linestyle='--', linewidth=2, label=f'Best: ${targets[total_best_ind]:.4f}$ at iteration ${total_best_ind}$')
    ax.legend(fontsize=24)

    ax.tick_params(axis='both', which='major', labelsize=22)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close(fig)