import os
import json
import sys
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import logging
import uuid
from rich.table import Table

from src.evaluation.results_table import create_latex_table
from src.utils.hydra_config import validate_config
from src.experiment.experiment_runner import run_experiment

logger = logging.getLogger(__name__)

def get_latex_results():
    """
    Loads all .json results in `experiments/results` and prints a combined LaTeX table.
    """
    # Import rich inside the function to avoid serialization issues
    from rich.console import Console
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

    print(selected_files)
    
    all_results = {}
    for file in selected_files:
        with open(os.path.join(result_dir, file), "r") as f:
            results = json.load(f)
        all_results.update(results)

    print(all_results)
    
    latex_table = create_latex_table(all_results)
    
    # Print to console with syntax highlighting
    console.print("\n[bold blue]LaTeX Table:[/bold blue]")
    console.print(f"```latex\n{latex_table}\n```")
    
    # Save to file
    latex_file = os.path.join(result_dir, "latex_table.tex")
    with open(latex_file, "w") as f:
        f.write(latex_table)
    
    console.print(f"[green]LaTeX table saved to {latex_file}[/green]")

def sync_wandb_runs(config: DictConfig):
    """
    Syncs all offline wandb runs to the server in parallel using Ray with optimizations for maximum speed.
    This function should be called after experiments are complete.
    """
    from rich.console import Console
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    import subprocess
    import os
    import ray
    import time
    import shutil
    
    console = Console()
    
    # Get the wandb directory from config
    wandb_dir = config.wandb.dir
    
    if not os.path.exists(wandb_dir):
        console.print("[bold red]No wandb directory found![/bold red]")
        return
    
    # Find all run directories (they have a 'files' subdirectory)
    run_dirs = []
    for root, dirs, files in os.walk(wandb_dir):
        if 'files' in dirs:
            run_dirs.append(root)
    
    if not run_dirs:
        console.print("[bold yellow]No offline wandb runs found to sync.[/bold yellow]")
        return
    
    console.print(f"[bold green]Found {len(run_dirs)} offline wandb runs to sync.[/bold green]")
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        # Use more CPUs for faster syncing
        num_cpus = os.cpu_count() or 8  # Default to 8 if can't determine
        ray.init(num_cpus=num_cpus)
    
    # Define a Ray remote function for syncing a single run with optimizations
    @ray.remote(num_cpus=0.5)  # Allow more tasks than CPUs for I/O-bound operations
    def sync_single_run(run_dir):
        run_id = os.path.basename(run_dir)
        start_time = time.time()
        
        try:
            # Use --no-verbose to reduce output and speed up sync
            # Use --sync-all to ensure all files are synced
            result = subprocess.run(
                ["wandb", "sync", "--sync-all", run_dir],
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5-minute timeout per run
            )
            duration = time.time() - start_time
            return {"success": True, "run_id": run_id, "duration": duration}
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            return {"success": False, "run_id": run_id, "error": e.stderr, "duration": duration}
        except subprocess.TimeoutExpired:
            return {"success": False, "run_id": run_id, "error": "Sync operation timed out after 5 minutes", "duration": 300}
    
    # Group runs by size for better load balancing
    # Smaller runs first to get quick wins
    run_sizes = []
    for run_dir in run_dirs:
        size = 0
        for root, dirs, files in os.walk(run_dir):
            size += sum(os.path.getsize(os.path.join(root, file)) for file in files if os.path.isfile(os.path.join(root, file)))
        run_sizes.append((run_dir, size))
    
    # Sort by size (smallest first)
    run_sizes.sort(key=lambda x: x[1])
    sorted_run_dirs = [item[0] for item in run_sizes]
    
    # Submit all sync tasks to Ray
    batch_size = min(100, len(sorted_run_dirs))  # Process in batches to avoid overwhelming the system
    console.print(f"[bold blue]Processing in batches of {batch_size} runs[/bold blue]")
    
    total_synced = 0
    total_failed = 0
    total_time = 0
    
    # Process in batches
    for i in range(0, len(sorted_run_dirs), batch_size):
        batch = sorted_run_dirs[i:i+batch_size]
        pending_results = [sync_single_run.remote(run_dir) for run_dir in batch]
        
        # Track progress with Rich
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, style="blue", complete_style="bright_blue"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=False
        ) as progress:
            task = progress.add_task(f"[Syncing Batch {i//batch_size + 1}/{(len(sorted_run_dirs) + batch_size - 1)//batch_size}]", total=len(batch))
            
            # Process results as they complete
            while pending_results:
                # Get the next completed result
                done_id, pending_results = ray.wait(pending_results, num_returns=1)
                result = ray.get(done_id[0])
                
                # Update progress based on result
                if result["success"]:
                    total_synced += 1
                    total_time += result["duration"]
                    progress.update(task, advance=1, 
                                  description=f"[bold blue]Synced {result['run_id']} in {result['duration']:.1f}s")
                else:
                    total_failed += 1
                    console.print(f"[bold red]Error syncing run {result['run_id']}:[/bold red] {result.get('error', 'Unknown error')}")
                    progress.update(task, advance=1, 
                                  description=f"[bold red]Failed to sync {result['run_id']}")
    
    # Print summary statistics
    console.print(f"[bold green]Sync complete! Successfully synced {total_synced} runs, failed {total_failed} runs.[/bold green]")
    if total_synced > 0:
        console.print(f"[bold blue]Average sync time: {total_time/total_synced:.2f} seconds per run[/bold blue]")
    
    # Ask if user wants to clean up synced runs to save space
    if total_synced > 0:
        cleanup = input("Do you want to delete successfully synced runs to save disk space? (y/n): ").lower() == 'y'
        if cleanup:
            console.print("[bold yellow]Cleaning up synced runs...[/bold yellow]")
            cleaned = 0
            for run_dir in sorted_run_dirs:
                try:
                    # Check if run was successfully synced by looking for a wandb-summary.json file
                    summary_file = os.path.join(run_dir, "wandb-summary.json")
                    if os.path.exists(summary_file):
                        shutil.rmtree(run_dir)
                        cleaned += 1
                except Exception as e:
                    console.print(f"[bold red]Error removing {run_dir}: {str(e)}[/bold red]")
            console.print(f"[bold green]Cleaned up {cleaned} run directories[/bold green]")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for the application.
    
    Args:
        cfg: Hydra configuration object
    """
    # Create console inside the function
    from rich.console import Console
    console = Console()
    
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
    elif config.experiment.mode == "sync_wandb":
        sync_wandb_runs(config)
    else:
        console.print("[bold yellow]Mode not implemented: {config.experiment.mode}[/bold yellow]")

if __name__ == "__main__":
    main()
