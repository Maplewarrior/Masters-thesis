from src.utils.config_loader import Config
from src.data_utils.data_generation import generate_data
from src.visualization.results_processor import create_results_table, create_summary_table, collect_metrics
from src.modelling.UnlearningManager import UnlearningManager
from src.evaluation.evaluation_utils import evaluate_models
from src.data_utils.data_generation import create_forget_retain_split

import os
import json
import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import wandb
import time

def run_experiment(config: Config):
    """Main experiment flow with multiple forget set trials"""
    console = Console()
    
    # Generate data once
    data_generator = generate_data(config.data)
    
    console.print(Panel(
        f"[bold blue]Experiment: {config.experiment.experiment_id}[/bold blue]\n"
        f"Unlearning Method: [yellow]{config.experiment.unlearn_type}[/yellow]",
        title="Machine Unlearning Experiment"
    ))
    
    # Track metrics across all trials
    all_metrics = {
        "Hamming PD": [],
        "JS Divergence": [],
        "Accuracy (Unlearned)": [],
        "Accuracy (Retrained)": []
    }
    
    # Track dataset-specific metrics during the experiment
    dataset_metrics = {
        "Retain": {
            "Hamming PD": {"uo": [], "ur": [], "ro": []},
            "JS divergence": {"uo": [], "ur": [], "ro": []},
            "acc_comparison_model": {"uo": [], "ur": [], "ro": []},
            "acc_unlearned_model": {"uo": [], "ur": [], "ro": []}
        },
        "Forget": {
            "Hamming PD": {"uo": [], "ur": [], "ro": []},
            "JS divergence": {"uo": [], "ur": [], "ro": []},
            "acc_comparison_model": {"uo": [], "ur": [], "ro": []},
            "acc_unlearned_model": {"uo": [], "ur": [], "ro": []}
        },
        "Validation": {
            "Hamming PD": {"uo": [], "ur": [], "ro": []},
            "JS divergence": {"uo": [], "ur": [], "ro": []},
            "acc_comparison_model": {"uo": [], "ur": [], "ro": []},
            "acc_unlearned_model": {"uo": [], "ur": [], "ro": []}
        }
    }
    
    # Create progress display with Rich
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, style="blue", complete_style="bright_blue"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=False
    ) as progress:
        forget_task = progress.add_task("[Forget Trials]", total=config.experiment.n_forget_trials)
        repeat_task = progress.add_task("[Current Repeat]", total=config.experiment.n_repeats, visible=False)
        
        for forget_trial in range(config.experiment.n_forget_trials):
            dataloaders = create_forget_retain_split(data_generator, config)
            
            # Reset the repeats task for each new forget trial
            progress.reset(repeat_task)
            progress.update(repeat_task, description=f"[bold green]Repeat {forget_trial+1}/{config.experiment.n_forget_trials}", visible=True)
            
            for repeat in range(config.experiment.n_repeats):
                # Initialize wandb if enabled
                if config.wandb.enabled:
                    # Reset wandb mode for this run
                    if hasattr(wandb, "run") and wandb.run is not None:
                        wandb.finish()
                    
                    os.environ["WANDB_MODE"] = config.wandb.mode
                    wandb.setup(settings=wandb.Settings(mode=config.wandb.mode))
                    
                    # Individual run for each trial/repeat
                    run = wandb.init(
                        project=config.wandb.project,
                        name=f"trial_{forget_trial}_repeat_{repeat}",
                        group=config.experiment.experiment_group,
                        config=config.model_dump(),
                        mode=config.wandb.mode,
                        dir=config.wandb.dir,
                        tags=[config.experiment.experiment_id],
                        reinit=True
                    )
                
                # Train models
                unlearning_manager = UnlearningManager(config, config.system.device, 
                                                     wandb=wandb if config.wandb.enabled else None,
                                                     track_performance=config.experiment.track_performance)
                unlearned_model, retrained_model, original_model = unlearning_manager.apply_unlearning(
                    config.experiment.unlearn_type,
                    dataloaders,
                    config
                )
                
                # Evaluate
                results = evaluate_models(unlearned_model, retrained_model, original_model, dataloaders)
                
                # Store metrics for summary
                all_metrics["Hamming PD"].append(results["unlearned vs. retrained"]["validation"]["Hamming PD"])
                all_metrics["JS Divergence"].append(results["unlearned vs. retrained"]["validation"]["JS divergence"])
                all_metrics["Accuracy (Unlearned)"].append(results["unlearned vs. original"]["validation"]["accuracy_unlearned_model"])
                all_metrics["Accuracy (Retrained)"].append(results["unlearned vs. retrained"]["validation"]["accuracy_comparison_model"])
                
                # Collect metrics for detailed summary
                dataset_metrics = collect_metrics(results, dataset_metrics)
                
                # Log to wandb if enabled
                if config.wandb.enabled:
                    # Log metrics for this run using "/" for nesting
                    wandb.log({
                        # Validation: divergence and hamming pd
                        "validation/unlearned_vs_original/hamming_pd": results["unlearned vs. original"]["validation"]["Hamming PD"],
                        "validation/unlearned_vs_original/js_divergence": results["unlearned vs. original"]["validation"]["JS divergence"],
                        "validation/retrained_vs_original/hamming_pd": results["retrained vs. original"]["validation"]["Hamming PD"],
                        "validation/retrained_vs_original/js_divergence": results["retrained vs. original"]["validation"]["JS divergence"],
                        "validation/unlearned_vs_retrained/hamming_pd": results["unlearned vs. retrained"]["validation"]["Hamming PD"],
                        "validation/unlearned_vs_retrained/js_divergence": results["unlearned vs. retrained"]["validation"]["JS divergence"],

                        # Validation: accuracy
                        "validation/accuracy/unlearned": results["unlearned vs. original"]["validation"]["accuracy_unlearned_model"],
                        "validation/accuracy/retrained": results["unlearned vs. retrained"]["validation"]["accuracy_comparison_model"],
                        "validation/accuracy/original": results["unlearned vs. original"]["validation"]["accuracy_comparison_model"],
                        
                        # Retain: divergence and hamming pd
                        "retain/unlearned_vs_original/hamming_pd": results["unlearned vs. original"]["retain"]["Hamming PD"],
                        "retain/unlearned_vs_original/js_divergence": results["unlearned vs. original"]["retain"]["JS divergence"],
                        "retain/retrained_vs_original/hamming_pd": results["retrained vs. original"]["retain"]["Hamming PD"],
                        "retain/retrained_vs_original/js_divergence": results["retrained vs. original"]["retain"]["JS divergence"],
                        "retain/unlearned_vs_retrained/hamming_pd": results["unlearned vs. retrained"]["retain"]["Hamming PD"],
                        "retain/unlearned_vs_retrained/js_divergence": results["unlearned vs. retrained"]["retain"]["JS divergence"],

                        # Retain: accuracy
                        "retain/accuracy/unlearned": results["unlearned vs. original"]["retain"]["accuracy_unlearned_model"],
                        "retain/accuracy/retrained": results["unlearned vs. retrained"]["retain"]["accuracy_comparison_model"],
                        "retain/accuracy/original": results["unlearned vs. original"]["retain"]["accuracy_comparison_model"],

                        # Forget: divergence and hamming pd
                        "forget/unlearned_vs_original/hamming_pd": results["unlearned vs. original"]["forget"]["Hamming PD"],
                        "forget/unlearned_vs_original/js_divergence": results["unlearned vs. original"]["forget"]["JS divergence"],
                        "forget/retrained_vs_original/hamming_pd": results["retrained vs. original"]["forget"]["Hamming PD"],
                        "forget/retrained_vs_original/js_divergence": results["retrained vs. original"]["forget"]["JS divergence"],
                        "forget/unlearned_vs_retrained/hamming_pd": results["unlearned vs. retrained"]["forget"]["Hamming PD"],
                        "forget/unlearned_vs_retrained/js_divergence": results["unlearned vs. retrained"]["forget"]["JS divergence"],

                        # Forget: accuracy
                        "forget/accuracy/unlearned": results["unlearned vs. original"]["forget"]["accuracy_unlearned_model"],
                        "forget/accuracy/retrained": results["unlearned vs. retrained"]["forget"]["accuracy_comparison_model"],
                        "forget/accuracy/original": results["unlearned vs. original"]["forget"]["accuracy_comparison_model"],

                        "metadata/trial": forget_trial,
                        "metadata/repeat": repeat
                    })
                    run.finish()
                
                # Update the repeats progress
                progress.update(repeat_task, advance=1)
            
            # Update the forget trials progress
            progress.update(forget_task, advance=1)
    
    # Print summary statistics at the end
    console.print("\n[bold blue]===== Experiment Summary =====[/bold blue]")
    
    # Create and print summary table
    summary_table = create_summary_table(dataset_metrics, config.experiment.unlearn_type)
    console.print(summary_table)
    
    # Save summary to file
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(results_dir, f"summary_{config.experiment.unlearn_type}_{timestamp}.json")
    
    with open(summary_file, "w") as f:
        json.dump({
            "experiment_id": config.experiment.experiment_id,
            "unlearn_type": config.experiment.unlearn_type,
            "summary": dataset_metrics,
            "config": config.model_dump()
        }, f, indent=2)
    
    console.print(f"[green]Summary saved to {summary_file}[/green]")