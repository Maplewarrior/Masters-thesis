import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import sys
from matplotlib.ticker import MaxNLocator

def _select_from_list(options: list, prompt_message: str) -> str:
    """Helper function to prompt user to select from a list of options."""
    if not options:
        return None

    print(prompt_message)
    for i, option in enumerate(options):
        print(f"  {i + 1}: {option}")

    while True:
        try:
            choice = input(f"Select an option (1-{len(options)}) or 'q' to quit: ")
            if choice.lower() == 'q':
                return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                return options[choice_idx]
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled.")
            return None

def select_experiment_folder(base_dir: str = 'results/teacher_ascend') -> str:
    """Guides the user to interactively select an experiment results folder."""
    # Level 1: Select split type
    print(f"Searching for experiment results in: {os.path.abspath(base_dir)}")
    try:
        experiment = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        if not experiment:
            print("No subdirectories found for split types. Make sure results are in the correct folder.")
            return None
    except FileNotFoundError:
        print(f"Error: Base directory '{base_dir}' not found.")
        return None
    
    selected_experiment = _select_from_list(experiment, "\nSelect the experiment:")
    if not selected_experiment:
        return None

    return selected_experiment



def get_objective_function_latex(experiment_name: str) -> str:
    """Get the LaTeX string representation of the objective function."""
    objective = ""
    if "ce" in experiment_name:
        if "fimratio" in experiment_name:
            objective = r"$- \frac{1}{|\mathcal{D}_f|}\sum_{(\boldsymbol{x}_i,y_i)\in \mathcal{D}_f} \mathcal{L}_{CE}(\boldsymbol{x}_i,y_i; \boldsymbol{\theta}_u) +  \frac{\lambda}{2} \sum_j \frac{i_j^{\mathcal{D}_r}}{i_j^{\mathcal{D}_f}}\left(\theta_{u, j}-\theta_{\text {orig }, j}\right)^2 $"
        else:
            objective = r"$- \frac{1}{|\mathcal{D}_f|}\sum_{(\boldsymbol{x}_i,y_i)\in \mathcal{D}_f} \mathcal{L}_{CE}(\boldsymbol{x}_i,y_i; \boldsymbol{\theta}_u) + \frac{\lambda}{2} \sum_j i_j^{\left(\mathcal{D}_r\right)}\left(\theta_{u, j}-\theta_{\text {orig }, j}\right)^2$"
    elif "entropy" in experiment_name:
        if "fimratio" in experiment_name:
            objective = r"$-\frac{1}{\left|\mathcal{D}_f\right|} \sum_{(\boldsymbol{x}_i, y_i)\in\mathcal{D}_f} H_{\mathcal{M}_{\theta_u}}\left(\boldsymbol{x}_i\right) + \frac{\lambda}{2} \sum_j \frac{i_j^{\mathcal{D}_r}}{i_j^{\mathcal{D}_f}}\left(\theta_{u, j}-\theta_{\text {orig }, j}\right)^2$"
        else:   
            objective = r"$-\frac{1}{\left|\mathcal{D}_f\right|} \sum_{(\boldsymbol{x}_i, y_i)\in\mathcal{D}_f} H_{\mathcal{M}_{\theta_u}}\left(\boldsymbol{x}_i\right) + \frac{\lambda}{2} \sum_j i_j^{\left(\mathcal{D}_r\right)}\left(\theta_{u, j}-\theta_{\text {orig }, j}\right)^2$"

    if "retain" in experiment_name:
        # Cross entropy loss over one batch of the retained data is added 
        objective = objective[:-1]  # Remove the closing $
        if "ce" in experiment_name:
            if "fimratio" in experiment_name:
                objective += r" \\ \phantom{- \frac{1}{|\mathcal{D}_f|}\sum_{(\boldsymbol{x}_i,y_i)\in \mathcal{D}_f} \mathcal{L}_{CE}(\boldsymbol{x}_i,y_i; \boldsymbol{\theta}_u)} + \frac{1}{\left|\mathcal{B}_r \right|}\sum_{(\boldsymbol{x}_k,y_k)\in\mathcal{B}_r} \mathcal{L}_{C E}\left(\boldsymbol{x}_k, y_k ; \boldsymbol{\theta}_u\right)$"
            else:
                objective += r" \\ \phantom{- \frac{1}{|\mathcal{D}_f|}\sum_{(\boldsymbol{x}_i,y_i)\in \mathcal{D}_f} \mathcal{L}_{CE}(\boldsymbol{x}_i,y_i; \boldsymbol{\theta}_u)} + \frac{1}{\left|\mathcal{B}_r \right|}\sum_{(\boldsymbol{x}_k,y_k)\in\mathcal{B}_r} \mathcal{L}_{C E}\left(\boldsymbol{x}_k, y_k ; \boldsymbol{\theta}_u\right)$"
        elif "entropy" in experiment_name:
            if "fimratio" in experiment_name:
                objective += r" \\ \phantom{-\frac{1}{\left|\mathcal{D}_f\right|} \sum_{(\boldsymbol{x}_i, y_i)\in\mathcal{D}_f} H_{\mathcal{M}_{\theta_u}}\left(\boldsymbol{x}_i\right)} + \frac{1}{\left|\mathcal{B}_r \right|}\sum_{(\boldsymbol{x}_k,y_k)\in\mathcal{B}_r} \mathcal{L}_{C E}\left(\boldsymbol{x}_k, y_k ; \boldsymbol{\theta}_u\right)$"
            else:
                objective += r" \\ \phantom{-\frac{1}{\left|\mathcal{D}_f\right|} \sum_{(\boldsymbol{x}_i, y_i)\in\mathcal{D}_f} H_{\mathcal{M}_{\theta_u}}\left(\boldsymbol{x}_i\right)} + \frac{1}{\left|\mathcal{B}_r \right|}\sum_{(\boldsymbol{x}_k,y_k)\in\mathcal{B}_r} \mathcal{L}_{C E}\left(\boldsymbol{x}_k, y_k ; \boldsymbol{\theta}_u\right)$"

    return objective


def main():
    # Set paper-friendly style at the start
    # Set a standard, paper-friendly style
    plt.style.use('seaborn-v0_8-paper')
    
    # Setup a STANDARD global font (e.g., sans-serif)
    # Turn off usetex globally
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'text.usetex': False,  # Disable LaTeX for everything by default
        'text.latex.preamble': (
            r'\usepackage{amsmath}'
            r'\usepackage{amssymb}'
            r'\usepackage{bm}'
        )
    })
    
    # Define your color palette
    retain_color = '#003f5c'
    forget_color = '#ffa600'
    val_color = '#dd5182'

    model_options = os.listdir("results")
    # filter model options to only include teacher_ascend
    model_options = [f for f in model_options if "teacher_ascend" in f]
    model_folder = _select_from_list(model_options, "Select the model to plot:")
 
    base_results_dir = f"results/{model_folder}"
    base_plots_dir = f"plots/{model_folder}"

    results_dir = select_experiment_folder(base_results_dir)
    if not results_dir:
        print("No experiment folder selected. Exiting.")
        sys.exit(1)

    experiment_dir = os.path.join(base_results_dir, results_dir)

    print("\n--- Folder Selection Successful ---")
    print(f"Analyzing results from: {experiment_dir}")
    print("------------------------------------")

    print("\nContents of the selected directory:")
    files = os.listdir(experiment_dir)
    # filter out only the ones with ta_metrics in the name 
    files = [f for f in files if "ta_metrics" in f or "scrub_metrics" in f]

    # load retrained_model_results
    retrained_path = os.path.join(experiment_dir, "retrained_model_metrics.json")
    retrained_model_results = json.load(open(retrained_path, "r"))

    # get the retrained_retain_acc
    retrained_retain_acc = retrained_model_results["retain"]["acc"][0]
    retrained_forget_acc = retrained_model_results["forget"]["acc"][0]
    retrained_val_acc = retrained_model_results["val"]["acc"][0]

    all_results = {}


    for selected_file in files:
        selected_file_path = os.path.join(experiment_dir, selected_file)

        # load the file
        with open(selected_file_path, 'r') as f:
            results = json.load(f)




        # remove extension from selected file
        selected_file_name = selected_file.split('.')[0]

        # get the objective function latex
        objective_function_latex = get_objective_function_latex(selected_file_name)

        all_results[selected_file_name] = results


    fig, axs = plt.subplots(2, 4, figsize=(22, 12))  # Made even taller to accommodate legend
    fig.suptitle('Teacher Ascend Results', fontsize=22)

    all_results_no_reg = {k: v for k, v in all_results.items() if "no-reg" not in k}
    axs_flat = axs.flatten()
    plot_idx = 0
    

    # Separate results into FIM ratio and non-FIM ratio
    fimratio_results = {k: v for k, v in all_results_no_reg.items() if "fimratio" in k}
    non_fimratio_results = {k: v for k, v in all_results_no_reg.items() if "fimratio" not in k}

    # sort in the order ce, ce-retain, entropy, entropy-retain
    non_fimratio_results = dict(sorted(non_fimratio_results.items(), key=lambda x: x[0]))
    # Sort in the order 'ta_metrics_ce_fimratio', 'ta_metrics_ce-retain_fimratio',  'ta_metrics_entropy_fimratio' 'ta_metrics_entropy-retain_fimratio',
    fimratio_results = {
        'ta_metrics_ce_fimratio': fimratio_results['ta_metrics_ce_fimratio'],
        'ta_metrics_ce-retain_fimratio': fimratio_results['ta_metrics_ce-retain_fimratio'],
        'ta_metrics_entropy_fimratio': fimratio_results['ta_metrics_entropy_fimratio'],
        'ta_metrics_entropy-retain_fimratio': fimratio_results['ta_metrics_entropy-retain_fimratio']
    }
    

    # Create two figures
    fig1, axs1 = plt.subplots(2, 2, figsize=(17, 14))
    fig2, axs2 = plt.subplots(2, 2, figsize=(17, 14))
    
    # Set titles with LaTeX formatting
    fig1.suptitle(r'Teacher Ascend Results (with FIM ratio)', fontsize=22)
    fig2.suptitle(r'Teacher Ascend Results (without FIM ratio)', fontsize=22)

    # Function to plot on a specific axis
    def plot_experiment(ax, results, selected_file_name):
        retain_acc = results["retain"]["acc"]
        forget_acc = results["forget"]["acc"]
        val_acc = results["val"]["acc"]
        
        epochs = range(0, len(retain_acc))
        
        # Plot with consistent styling
        lines = []
        lines.append(ax.plot(epochs, retain_acc, 
               marker='o', linewidth=2, markersize=5,
               color=retain_color, alpha=0.8, label='Retain')[0])
        
        lines.append(ax.plot(epochs, forget_acc, 
               marker='s', linewidth=2, markersize=5,
               color=forget_color, alpha=0.8, label='Forget')[0])
        
        lines.append(ax.plot(epochs, val_acc, 
               marker='^', linewidth=2, markersize=5,
               color=val_color, alpha=0.8, label='Validation')[0])
        
        retrained_line_color = '#f95d6a'
        retain_line = ax.axhline(y=retrained_retain_acc, color=retrained_line_color, linestyle='--', 
           alpha=0.9, linewidth=1.5, label=f'Retrained Model - Retain Acc ({retrained_retain_acc:.2f})')
        forget_line = ax.axhline(y=retrained_forget_acc, color=retrained_line_color, linestyle=':', 
           alpha=0.9, linewidth=1.5, label=f'Retrained Model - Forget Acc ({retrained_forget_acc:.2f})')
        
        lines.extend([retain_line, forget_line])
        
        # Regular styling for labels - will use the global 'sans-serif' font
        ax.set_xlabel('Epoch', fontsize=18)
        ax.set_ylabel('Accuracy', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        latex_objective_function = get_objective_function_latex(selected_file_name)
        ax.set_title(
            latex_objective_function,
            fontsize=18,  # Changed from 16 to 18
            usetex=True,
            fontfamily='serif'
        )
        
        return lines


    # Plot FIM ratio results and get legend lines from first plot
    axs1_flat = axs1.flatten()
    legend_lines = None
    for idx, (name, results) in enumerate(fimratio_results.items()):
        if idx >= len(axs1_flat):
            print(f"Warning: More FIM ratio experiments than available subplots. Skipping {name}")
            continue
        lines = plot_experiment(axs1_flat[idx], results, name)
        if idx == 0:  # Save lines from first plot for legend
            legend_lines = lines

    # Plot non-FIM ratio results
    axs2_flat = axs2.flatten()
    for idx, (name, results) in enumerate(non_fimratio_results.items()):
        if idx >= len(axs2_flat):
            print(f"Warning: More non-FIM ratio experiments than available subplots. Skipping {name}")
            continue
        plot_experiment(axs2_flat[idx], results, name)

    # Remove any empty subplots
    for idx in range(len(fimratio_results), len(axs1_flat)):
        fig1.delaxes(axs1_flat[idx])
    for idx in range(len(non_fimratio_results), len(axs2_flat)):
        fig2.delaxes(axs2_flat[idx])

    # Legend labels
    legend_labels = [
        'Retain',
        'Forget',
        'Validation',
        f'Retrained Model - Retain Acc ({retrained_retain_acc:.2f})',
        f'Retrained Model - Forget Acc ({retrained_forget_acc:.2f})'
    ]
   # Add legends to both figures, using tight_layout to prevent overlap
    for fig in [fig1, fig2]:
        # We've reduced the margins here:
        # bottom=0.1: Shrinks the bottom margin from 15% to 10%.
        # top=0.95: Shrinks the top margin from 8% to 5%.
        fig.tight_layout(rect=[0, 0.1, 1, 0.95])

        # Add the shared legend, positioning it within the new, smaller margin.
        fig.legend(legend_lines, legend_labels, 
                  loc='upper center',
                  # This y-value is reduced to fit the new margin
                  bbox_to_anchor=(0.5, 0.08), 
                  ncol=5,
                  fontsize=14,
                  frameon=True,
                  bbox_transform=fig.transFigure)


    # save as pdf
    fig1.savefig(f"plots/{model_folder}/teacher_ascend_fimratio.pdf", bbox_inches='tight')
    fig2.savefig(f"plots/{model_folder}/teacher_ascend_no_fimratio.pdf", bbox_inches='tight')

    # Create individual plots for specific experiments
    individual_experiments = [
        'ta_metrics_ce',
        'ta_metrics_entropy', 
        'ta_metrics_entropy-retain',
        'ta_metrics_entropy-retain_fimratio'
    ]
    
    for exp_name in individual_experiments:
        if exp_name in all_results:
            # Create individual figure with consistent dimensions
            fig_ind, ax_ind = plt.subplots(1, 1, figsize=(8, 6))
            
            # Get results for this experiment
            results = all_results[exp_name]
            retain_acc = results["retain"]["acc"]
            forget_acc = results["forget"]["acc"]
            val_acc = results["val"]["acc"]
            
            epochs = range(0, len(retain_acc))
            
            # Plot with consistent styling
            ax_ind.plot(epochs, retain_acc, 
                       marker='o', linewidth=2, markersize=5,
                       color=retain_color, alpha=0.8, label='Retain')
            
            ax_ind.plot(epochs, forget_acc, 
                       marker='s', linewidth=2, markersize=5,
                       color=forget_color, alpha=0.8, label='Forget')
            
            ax_ind.plot(epochs, val_acc, 
                       marker='^', linewidth=2, markersize=5,
                       color=val_color, alpha=0.8, label='Validation')
            
            # Add retrained model lines
            retrained_line_color = '#f95d6a'
            ax_ind.axhline(y=retrained_retain_acc, color=retrained_line_color, linestyle='--', 
                          alpha=0.9, linewidth=1.5, label=f'Retrained Model - Retain Acc ({retrained_retain_acc:.2f})')
            ax_ind.axhline(y=retrained_forget_acc, color=retrained_line_color, linestyle=':', 
                          alpha=0.9, linewidth=1.5, label=f'Retrained Model - Forget Acc ({retrained_forget_acc:.2f})')
            
            # Styling
            ax_ind.set_xlabel('Epoch', fontsize=18)
            ax_ind.set_ylabel('Accuracy', fontsize=18)
            ax_ind.tick_params(axis='both', which='major', labelsize=16)
            ax_ind.grid(True, alpha=0.3)
            ax_ind.set_ylim(0, 1.1)
            ax_ind.legend(fontsize=14, loc='best')
            
            # Save individual plot
            fig_ind.tight_layout()
            fig_ind.savefig(f"plots/{model_folder}/{exp_name}.pdf", bbox_inches='tight')
            plt.close(fig_ind)
            
            print(f"Created individual plot for {exp_name}")
        else:
            print(f"Warning: {exp_name} not found in results")


if __name__ == "__main__":
    main() 