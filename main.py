import os
import json
import sys
from rich.console import Console
from rich.table import Table
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import logging
import uuid

from rich.pretty import Pretty
from rich.panel import Panel

from src.evaluation.results_table import create_latex_table
from src.utils.hydra_config import validate_config
from src.experiment.experiment_runner import run_experiment

logger = logging.getLogger(__name__)
console = Console()

def get_latex_results():
    """
    Loads all .json results in `experiments/results` and prints a combined LaTeX table.
    """
    result_dir = "experiments/results"
    result_files = [f for f in os.listdir(result_dir) if f.endswith(".json")]
    
    if not result_files:
        console.print("[bold red]No result files found in experiments/results![/bold red]")
        return
    
    # Create a table to display available results
    table = Table(title="Available Result Files")
    table.add_column("Index", style="cyan")
    table.add_column("Filename", style="green")
    table.add_column("Experiment ID", style="yellow")
    table.add_column("Unlearn Type", style="red")
    
    file_info = []
    for i, file in enumerate(result_files):
        try:
            with open(os.path.join(result_dir, file), "r") as f:
                data = json.load(f)
                exp_id = data.get("experiment_id", "Unknown")
                unlearn_type = data.get("unlearn_type", "Unknown")
                file_info.append((i, file, exp_id, unlearn_type))
                table.add_row(str(i), file, exp_id, unlearn_type)
        except:
            table.add_row(str(i), file, "Error loading", "Error loading")
    
    console.print(table)
    
    # Ask user which files to include
    console.print("[bold]Enter indices of files to include (comma-separated) or 'all':[/bold]")
    selection = input().strip()
    
    selected_files = []
    if selection.lower() == 'all':
        selected_files = result_files
    else:
        try:
            indices = [int(idx.strip()) for idx in selection.split(',')]
            selected_files = [result_files[idx] for idx in indices if 0 <= idx < len(result_files)]
        except:
            console.print("[bold red]Invalid selection. Using all files.[/bold red]")
            selected_files = result_files
    
    all_results = {}
    for file in selected_files:
        with open(os.path.join(result_dir, file), "r") as f:
            results = json.load(f)
        all_results.update(results)
    
    latex_table = create_latex_table(all_results)
    
    # Print to console with syntax highlighting
    console.print("\n[bold blue]LaTeX Table:[/bold blue]")
    console.print(f"```latex\n{latex_table}\n```")
    
    # Save to file
    latex_file = os.path.join(result_dir, "latex_table.tex")
    with open(latex_file, "w") as f:
        f.write(latex_table)
    
    console.print(f"[green]LaTeX table saved to {latex_file}[/green]")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for the application.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set experiment_id if not provided
    if not cfg.experiment.experiment_id:
        cfg.experiment.experiment_id = str(uuid.uuid4())[:8]
    
    # If we're in a multirun, add parameter info to experiment_id
    if hasattr(hydra, 'multirun') and hydra.multirun:
        # Extract the parameter being swept
        sweep_params = [k for k, v in OmegaConf.to_container(cfg).items() 
                       if isinstance(v, list) or (hasattr(v, '__iter__') and not isinstance(v, str))]
        
        if sweep_params:
            param_name = sweep_params[0].split('.')[-1]
            param_value = cfg[sweep_params[0]]
            cfg.experiment.experiment_id = f"{cfg.experiment.experiment_id}_{param_name}_{param_value}"
    
    # Validate configuration
    try:
        config = validate_config(cfg)
    except ValueError as e:
        console.print(f"[bold red]Configuration error: {str(e)}[/bold red]")
        return
    
    # Print the config that is being used
    console.print("[bold blue]Configuration:[/bold blue]")
    console.print(config.model_dump_json(indent=2))
    
    if config.experiment.mode == "experiment":
        run_experiment(config)
    elif config.experiment.mode == "get_latex_results":
        get_latex_results()
    else:
        console.print("[bold yellow]Visualization mode not implemented in this refactoring.[/bold yellow]")

if __name__ == "__main__":
    main()
