import argparse
import json
import os
from rich.console import Console
from rich.table import Table

from src.evaluation.results_table import create_latex_table
from src.utils.config_loader import load_config
from src.experiment.experiment_runner import run_experiment


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    # Core arguments
    parser.add_argument(
        "mode",
        type=str.lower,
        default="experiment",
        choices=["experiment", "visualize", "get_latex_results"],
        help="What to do when running the script",
    )
    parser.add_argument(
        "--unlearn-type",
        type=str.lower,
        default="ssd",
        choices=["ssd", "scrub+r", "sisa", "amnesiac", "sae"],
        help="What type of unlearning algorithm to apply",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to experiment configuration file",
    )
    
    # Data generation arguments
    data_group = parser.add_argument_group('data generation')
    data_group.add_argument("--n-samples", type=int, default=1000)
    data_group.add_argument("--n-features", type=int, default=25)
    data_group.add_argument("--n-classes", type=int, default=4)
    data_group.add_argument("--n-informative", type=int, default=25)
    data_group.add_argument("--n-redundant", type=int, default=0)
    data_group.add_argument("--n-outliers", type=int, default=50)
    data_group.add_argument("--outlier-scale", type=float, default=4.0)
    data_group.add_argument("--outlier-variance", type=float, default=0.4)
    data_group.add_argument("--outlier-class", type=int, default=1)
    data_group.add_argument("--random-state", type=int, default=42)
    data_group.add_argument("--train-ratio", type=float, default=0.8)
    data_group.add_argument("--val-ratio", type=float, default=0.1)
    data_group.add_argument("--test-ratio", type=float, default=0.1)

    # Experiment arguments
    exp_group = parser.add_argument_group('experiment')
    exp_group.add_argument("--n-repeats", type=int, default=5,
                          help="Number of experiment repeats")
    exp_group.add_argument("--n-epochs", type=int, default=20,
                          help="Number of training epochs")
    exp_group.add_argument("--n-forget-trials", type=int, default=2,
                          help="Number of different forget sets to try")

    # Forget set arguments
    forget_group = parser.add_argument_group('forget')
    forget_group.add_argument("--forget-n-points", type=int, default=50,
                            help="Number of points to include in forget set")
    forget_group.add_argument("--forget-class-idx", type=int, default=None,
                            help="Specific class to draw forget points from")
    forget_group.add_argument("--forget-ood-ratio", type=float, default=0.5,
                            help="Ratio of out-of-distribution points in forget set")

    # System arguments
    sys_group = parser.add_argument_group('system')
    sys_group.add_argument(
        "--device",
        type=str.lower,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
    )
    sys_group.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    
    args = parser.parse_args()
    
    # First load the config file
    config = load_config(args.config)
    
    # Only override config with explicitly provided command-line arguments
    # (not using defaults from argparse)
    arg_dict = vars(args)
    provided_args = {k: v for k, v in arg_dict.items() 
                    if k in parser._option_string_actions and 
                    parser._option_string_actions[k].dest in arg_dict and
                    arg_dict[parser._option_string_actions[k].dest] is not parser.get_default(parser._option_string_actions[k].dest)}
     
    # Update config with explicitly provided arguments
    if provided_args:
        config.update_from_dict(provided_args)
    
    return args, config



def get_latex_results():
    """
    Loads all .json results in `experiments/results` and prints a combined LaTeX table.
    """
    console = Console()
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


def main():
    args, config = parse_arguments()

    # print the config that is being used
    print(config)
    
    if config.experiment.mode == "experiment":
        run_experiment(config)
    elif config.experiment.mode == "get_latex_results":
        get_latex_results()
    else:
        print("Visualization mode not implemented in this refactoring.")

if __name__ == "__main__":
    main()
