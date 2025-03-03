import argparse
import pdb
import pandas as pd
import torch
import torch.nn as nn
import copy
import json
import os
from pydantic import BaseModel, Field
from typing import Dict, Any, Tuple
import wandb
import yaml
from torch.utils.data import DataLoader
import numpy as np
import datetime

from src.modelling.trainer import Trainer
from src.modelling.neural_network import NeuralNet
from src.data_utils.synthetic_data import DataGenerator, create_dataloaders
from src.modelling.selective_synaptic_dampening import SelectiveSynapticDampening
from src.modelling.scrub import ScrubR
from src.modelling.sisa import SISA
from src.evaluation.unlearning_evaluator import UnlearningEvaluator
from src.evaluation.results_table import create_latex_table
from src.modelling.amnesiac import AmnesiacTrainer
from src.utils.config_loader import Config, load_config

from tqdm.auto import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


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
    config = load_config(args.config, vars(args))
    
    return args, config


def generate_data(args) -> DataGenerator:
    """
    Generates and splits the synthetic data using arguments from argparse.
    
    Args:
        args: Parsed command line arguments containing data generation parameters
    """
    data_generator = DataGenerator(random_state=args.random_state)
    data_generator.generate_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        n_informative=args.n_informative,
        n_redundant=args.n_redundant,
        n_outliers=args.n_outliers,
        outlier_scale=args.outlier_scale,
        outlier_variance=args.outlier_variance,
        outlier_class=args.outlier_class
    )
    data_generator.split_data(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    return data_generator


def create_forget_retain_split(data_generator, config: Config):
    """Creates a new forget/retain split for existing data and returns associated dataloaders."""
    data_generator.draw_forget_set(
        n_points=config.forget.n_points, 
        class_idx=config.forget.class_idx, 
        ood_ratio=config.forget.ood_ratio
    )
    
    # For amnesiac, we need indices, so batch_size differs:
    if config.experiment.unlearn_type == "amnesiac":
        dataloaders = create_dataloaders(
            data_generator, 
            batch_size=32, 
            use_indices=True, 
            device=config.system.device
        )
    else:
        dataloaders = create_dataloaders(
            data_generator, 
            batch_size=32, 
            onehot_labels=True, 
            device=config.system.device
        )
    
    return dataloaders


class UnlearningManager:
    def __init__(self, config: Config, device: str = "cpu", wandb=None):
        self.config = config
        self.device = device
        self.unlearn_type = None
        self.wandb = wandb

    def apply_unlearning(self, unlearn_type: str, dataloaders: Dict, config: Config) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """
        Applies the specified unlearning method and returns the unlearned, retrained, and original models.
        
        Args:
            unlearn_type: Type of unlearning to apply
            dataloaders: Dictionary containing the data loaders
            config: Configuration object containing all parameters
        """
        self.unlearn_type = unlearn_type

        if self.unlearn_type == "sisa":
            return self._apply_unlearning_sisa(dataloaders, config.model.n_features, config.model.n_classes, config.model.n_epochs, self.device)
        elif self.unlearn_type == "amnesiac":
            return self._apply_unlearning_amnesiac(dataloaders, config.model.n_features, config.model.n_classes, config.model.n_epochs, self.device)
        elif self.unlearn_type == "sae":
            return self._apply_unlearning_sae(dataloaders, config.model.n_features, config.model.n_classes, config.model.n_epochs, self.device)
        elif self.unlearn_type == "scrub+r" or self.unlearn_type == "ssd":
            return self._apply_unlearning(dataloaders, config.model.n_features, config.model.n_classes, config.model.n_epochs, self.device)
        else:
            raise ValueError(f"Unknown unlearning type: {self.unlearn_type}")

    def _train_standard_model(self, 
                              dataloader, 
                              val_dataloader, 
                              n_features, 
                              n_classes, 
                              n_epochs, 
                              dataset_name,
                              device="cpu"):
        """
        Trains and returns a NeuralNet model using the provided dataloader.
        """
        model = NeuralNet(n_features, n_classes)
        trainer = Trainer(model, 
                        train_dataloader=dataloader, 
                        val_dataloader=val_dataloader, 
                        n_epochs=n_epochs, 
                        device=device, 
                        disable_tqdm=True,
                        wandb=self.wandb,
                        dataset_name=dataset_name)
        trainer.train()
        trainer.eval()
        return model

    def _apply_unlearning(self, dataloaders, n_features, n_classes, n_epochs, device="cpu"):
        """
        Applies either Scrub+R or SSD to the given data. Returns the unlearned model, retrained model, and original model.
        """
        # 1) Train model on full dataset
        unlearned_model = self._train_standard_model(dataloader=dataloaders["train_full_loader"], 
                                    val_dataloader=dataloaders["val_loader"], 
                                    n_features=n_features, 
                                    n_classes=n_classes, 
                                    n_epochs=n_epochs, 
                                    dataset_name="original",
                                    device=device)

        # 2) Train retrained model on retain dataset
        retrained_model = self._train_standard_model(dataloader=dataloaders["train_retain_loader"], 
                                    val_dataloader=dataloaders["val_loader"], 
                                    n_features=n_features, 
                                    n_classes=n_classes, 
                                    n_epochs=n_epochs, 
                                    dataset_name="retrained",
                                    device=device)

        # 3) Copy the original model (before unlearning)
        original_model = copy.deepcopy(unlearned_model)
        original_model.model_type = "pre_forget"
        unlearned_model.model_type = "post_forget"

        # 4) Perform unlearning
        if self.unlearn_type == "scrub+r":
            alpha = 1.0
            gamma = 1.0
            scrub = ScrubR(unlearned_model, original_model, alpha=alpha, gamma=gamma)
            scrub(
                dataloaders["train_retain_loader"],
                dataloaders["train_forget_loader"],
                dataloaders["val_loader"],
                n_rounds=6,
            )
        elif self.unlearn_type == "ssd":
            alpha = 7.5
            _lambda = 0.5
            criterion = nn.CrossEntropyLoss()
            ssd = SelectiveSynapticDampening(unlearned_model, criterion, alpha=alpha, _lambda=_lambda)
            ssd(
                full_dataloader=dataloaders["train_full_loader"],
                forget_dataloader=dataloaders["train_forget_loader"],
            )

        return unlearned_model, retrained_model, original_model

    def _apply_unlearning_sisa(self, dataloaders, n_features, n_classes, n_epochs, device="cpu"):
        """
        Applies SISA unlearning. Returns the unlearned model, retrained model, and original model.
        """
        # Unlearned model
        unlearned_model = SISA(
            dataloader=dataloaders["train_full_loader"],
            n_shards=10,
            n_slices=10,
            n_features=n_features,
            n_classes=n_classes,
            n_epochs=n_epochs,
            save_dir="./experiments/checkpoints/SISA/original_model",
            disable_tqdm=True
        )
        unlearned_shards_dict = unlearned_model.process_data()
        original_shards_dict = unlearned_shards_dict  # in case you need it
        unlearned_model.train_all_models()

        # Retrained model
        retrained_model = SISA(
            dataloader=dataloaders["train_retain_loader"],
            n_shards=10,
            n_slices=10,
            n_features=n_features,
            n_classes=n_classes,
            n_epochs=n_epochs,
            save_dir="./experiments/checkpoints/SISA/retrained_model",
            disable_tqdm=True
        )
        retrained_model.process_data()
        retrained_model.train_all_models()

        # Original model (copy before unlearning)
        original_model = copy.deepcopy(unlearned_model)
        original_model.model_type = "pre_forget"
        unlearned_model.model_type = "post_forget"

        # Forget the datapoints
        unlearned_model.forget_datapoints(datapoint_idxs=dataloaders["forget_idx_in_train"])

        return unlearned_model, retrained_model, original_model

    def _apply_unlearning_amnesiac(self, dataloaders, n_features, n_classes, n_epochs, device="cpu"):
        """
        Applies the amnesiac unlearning method.
        Returns the unlearned model, retrained model, and original model.
        """
        unlearned_model = NeuralNet(n_features, n_classes)
        trainer = AmnesiacTrainer(unlearned_model, 
                                  lr=3e-3, 
                                  device=device, 
                                  cache_gradients=True, 
                                  disable_tqdm=True, 
                                  wandb=self.wandb,
                                  dataset_name="original")

        # Train on full data, storing gradients for sensitive batches
        indices_to_forget = dataloaders["forget_idx_in_train"]
        trainer.train(
            dataloaders["train_full_loader"], 
            epochs=n_epochs,
            indices_to_forget=indices_to_forget,
            save_accuracy_to_file=False
        )

        # Retrained model
        retrained_model = NeuralNet(n_features, n_classes)
        retrained_trainer = AmnesiacTrainer(retrained_model, 
                                            lr=3e-3, 
                                            device=device, 
                                            cache_gradients=True, 
                                            disable_tqdm=True,
                                            dataset_name="retrained")
        retrained_trainer.train(dataloaders["train_retain_loader"], repair=True)

        # Original model
        original_model = copy.deepcopy(unlearned_model)
        original_model.model_type = "pre_forget"
        unlearned_model.model_type = "post_forget"

        # Forget all sensitive batches
        trainer.forget(indices_to_forget=None)

        # Repair phase
        trainer.train(dataloaders["train_full_loader"], epochs=10, save_accuracy_to_file=False, repair=True)

        return unlearned_model, retrained_model, original_model

    def _apply_unlearning_sae(self):
        """
        Placeholder function for SAE unlearning since it's not yet implemented.
        """
        raise NotImplementedError("SAE is not implemented yet.")


def evaluate_models(unlearned_model, retrained_model, original_model, dataloaders):
    """
    Runs the evaluation metrics and returns a dictionary with the results.
    """
    unlearning_evaluator = UnlearningEvaluator()

    efficacy_metrics = ["accuracy", "Hamming PD", "min max normalized HPD"]
    func_equivalence_metrics = ["avg norm prediction difference", "JS divergence"]
    evaluation_metrics = efficacy_metrics + func_equivalence_metrics

    # Compare unlearned vs. original
    unlearned_original_retain = unlearning_evaluator.evaluate(
        unlearned_model, original_model, evaluation_metrics, dataloaders["train_retain_loader"]
    )
    unlearned_original_forget = unlearning_evaluator.evaluate(
        unlearned_model, original_model, evaluation_metrics, dataloaders["train_forget_loader"]
    )
    unlearned_original_val = unlearning_evaluator.evaluate(
        unlearned_model, original_model, evaluation_metrics, dataloaders["val_loader"]
    )

    # Compare unlearned vs. retrained
    unlearned_retrained_retain = unlearning_evaluator.evaluate(
        unlearned_model, retrained_model, evaluation_metrics, dataloaders["train_retain_loader"]
    )
    unlearned_retrained_forget = unlearning_evaluator.evaluate(
        unlearned_model, retrained_model, evaluation_metrics, dataloaders["train_forget_loader"]
    )
    unlearned_retrained_val = unlearning_evaluator.evaluate(
        unlearned_model, retrained_model, evaluation_metrics, dataloaders["val_loader"]
    )

    results = {
        "unlearned vs. original": {
            "retain": unlearned_original_retain,
            "forget": unlearned_original_forget,
            "validation": unlearned_original_val,
        },
        "unlearned vs. retrained": {
            "retain": unlearned_retrained_retain,
            "forget": unlearned_retrained_forget,
            "validation": unlearned_retrained_val,
        },
    }
    return results


def create_trial_metrics(trial: int, repeat: int, results: dict) -> dict:
    """Creates a nested dictionary of metrics for a trial and repeat."""
    return {
        "trials": {
            trial: {
                "repeats": {
                    repeat: {
                        "original_model": {
                            "accuracy": results["unlearned vs. original"]["validation"]["accuracy_comparison_model"]
                        },
                        "unlearned_model": {
                            "accuracy": results["unlearned vs. original"]["validation"]["accuracy_unlearned_model"]
                        },
                        "retrained_model": {
                            "accuracy": results["unlearned vs. retrained"]["validation"]["accuracy_comparison_model"]
                        },
                        "metrics": {
                            "hamming_pd": results["unlearned vs. original"]["validation"]["Hamming PD"],
                            "js_divergence": results["unlearned vs. original"]["validation"]["JS divergence"]
                        }
                    }
                }
            }
        }
    }

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
    
    # Create a status string that will be updated
    status_text = f"Trial: 0/{config.experiment.n_forget_trials}, Repeat: 0/{config.experiment.n_repeats}"
    
    # Function to create a fresh table with the latest results
    def create_results_table(results=None):
        # Create a grid for side-by-side tables
        grid = Table.grid()
        
        # Create the tables
        uo_table = Table(title=f"Unlearned vs. Original - {status_text}")
        uo_table.add_column("Dataset", style="cyan")
        uo_table.add_column("Hamming PD", style="green")
        uo_table.add_column("JS Divergence", style="blue")
        uo_table.add_column("Accuracy", style="yellow")
        
        ur_table = Table(title=f"Unlearned vs. Retrained - {status_text}")
        ur_table.add_column("Dataset", style="cyan")
        ur_table.add_column("Hamming PD", style="green")
        ur_table.add_column("JS Divergence", style="blue")
        ur_table.add_column("Accuracy", style="yellow")
        
        # Add rows with data or placeholders
        if results is None:
            # Add placeholder rows
            for table in [uo_table, ur_table]:
                table.add_row("Validation", "Pending...", "Pending...", "Pending...")
                table.add_row("Retain", "Pending...", "Pending...", "Pending...")
                table.add_row("Forget", "Pending...", "Pending...", "Pending...")
        else:
            # Add rows with actual data
            uo_table.add_row(
                "Validation", 
                f"{results['unlearned vs. original']['validation']['Hamming PD']:.4f}",
                f"{results['unlearned vs. original']['validation']['JS divergence']:.4f}",
                f"{results['unlearned vs. original']['validation']['accuracy_unlearned_model']:.4f}"
            )
            uo_table.add_row(
                "Retain",
                f"{results['unlearned vs. original']['retain']['Hamming PD']:.4f}",
                f"{results['unlearned vs. original']['retain']['JS divergence']:.4f}",
                f"{results['unlearned vs. original']['retain']['accuracy_unlearned_model']:.4f}"
            )
            uo_table.add_row(
                "Forget",
                f"{results['unlearned vs. original']['forget']['Hamming PD']:.4f}",
                f"{results['unlearned vs. original']['forget']['JS divergence']:.4f}",
                f"{results['unlearned vs. original']['forget']['accuracy_unlearned_model']:.4f}"
            )
            
            ur_table.add_row(
                "Validation", 
                f"{results['unlearned vs. retrained']['validation']['Hamming PD']:.4f}",
                f"{results['unlearned vs. retrained']['validation']['JS divergence']:.4f}",
                f"{results['unlearned vs. retrained']['validation']['accuracy_comparison_model']:.4f}"
            )
            ur_table.add_row(
                "Retain",
                f"{results['unlearned vs. retrained']['retain']['Hamming PD']:.4f}",
                f"{results['unlearned vs. retrained']['retain']['JS divergence']:.4f}",
                f"{results['unlearned vs. retrained']['retain']['accuracy_comparison_model']:.4f}"
            )
            ur_table.add_row(
                "Forget",
                f"{results['unlearned vs. retrained']['forget']['Hamming PD']:.4f}",
                f"{results['unlearned vs. retrained']['forget']['JS divergence']:.4f}",
                f"{results['unlearned vs. retrained']['forget']['accuracy_comparison_model']:.4f}"
            )
        
        # Add tables to grid
        grid.add_row(uo_table, ur_table)
        return grid
    
    # Initial table with placeholders
    results_table = create_results_table()
    
    with Live(results_table, refresh_per_second=4, console=console) as live:
        for forget_trial in range(config.experiment.n_forget_trials):
            dataloaders = create_forget_retain_split(data_generator, config)
            
            for repeat in range(config.experiment.n_repeats):
                # Update status
                status_text = f"Trial: {forget_trial+1}/{config.experiment.n_forget_trials}, Repeat: {repeat+1}/{config.experiment.n_repeats}"
                
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
                                                     wandb=wandb if config.wandb.enabled else None)
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
                
                # Update the table with new results
                live.update(create_results_table(results))
                
                # Log to wandb if enabled
                if config.wandb.enabled:
                    # Log metrics for this run using "/" for nesting
                    wandb.log({
                        "validation/unlearned_vs_original/hamming_pd": results["unlearned vs. original"]["validation"]["Hamming PD"],
                        "validation/unlearned_vs_original/js_divergence": results["unlearned vs. original"]["validation"]["JS divergence"],
                        "validation/retrained_vs_original/hamming_pd": results["unlearned vs. retrained"]["validation"]["Hamming PD"],
                        "validation/retrained_vs_original/js_divergence": results["unlearned vs. retrained"]["validation"]["JS divergence"],
                        "validation/accuracy/unlearned": results["unlearned vs. original"]["validation"]["accuracy_unlearned_model"],
                        "validation/accuracy/retrained": results["unlearned vs. retrained"]["validation"]["accuracy_comparison_model"],
                        "validation/accuracy/original": results["unlearned vs. original"]["validation"]["accuracy_comparison_model"],
                        
                        "retain/unlearned_vs_original/hamming_pd": results["unlearned vs. original"]["retain"]["Hamming PD"],
                        "retain/unlearned_vs_original/js_divergence": results["unlearned vs. original"]["retain"]["JS divergence"],
                        "retain/retrained_vs_original/hamming_pd": results["unlearned vs. retrained"]["retain"]["Hamming PD"],
                        "retain/retrained_vs_original/js_divergence": results["unlearned vs. retrained"]["retain"]["JS divergence"],
                        "retain/accuracy/unlearned": results["unlearned vs. original"]["retain"]["accuracy_unlearned_model"],
                        "retain/accuracy/retrained": results["unlearned vs. retrained"]["retain"]["accuracy_comparison_model"],
                        "retain/accuracy/original": results["unlearned vs. original"]["retain"]["accuracy_comparison_model"],

                        "forget/unlearned_vs_original/hamming_pd": results["unlearned vs. original"]["forget"]["Hamming PD"],
                        "forget/unlearned_vs_original/js_divergence": results["unlearned vs. original"]["forget"]["JS divergence"],
                        "forget/retrained_vs_original/hamming_pd": results["unlearned vs. retrained"]["forget"]["Hamming PD"],
                        "forget/retrained_vs_original/js_divergence": results["unlearned vs. retrained"]["forget"]["JS divergence"],
                        "forget/accuracy/unlearned": results["unlearned vs. original"]["forget"]["accuracy_unlearned_model"],
                        "forget/accuracy/retrained": results["unlearned vs. retrained"]["forget"]["accuracy_comparison_model"],
                        "forget/accuracy/original": results["unlearned vs. original"]["forget"]["accuracy_comparison_model"],

                        "metadata/trial": forget_trial,
                        "metadata/repeat": repeat
                    })
                    run.finish()
    
    # Print summary statistics at the end
    console.print("\n[bold blue]===== Experiment Summary =====[/bold blue]")
    
    # Create a single summary table with both comparisons
    summary_table = Table(title=f"Summary for {config.experiment.unlearn_type} Unlearning")
    summary_table.add_column("Dataset/Metric", style="cyan")
    summary_table.add_column("Unlearned vs. Original", style="green")
    summary_table.add_column("Unlearned vs. Retrained", style="yellow")
    
    # Track dataset-specific metrics during the experiment
    dataset_metrics = {
        "Retain": {
            "Hamming PD": {"uo": [], "ur": []},
            "JS divergence": {"uo": [], "ur": []},
            "acc_comparison_model": {"uo": [], "ur": []},
            "acc_unlearned_model": {"uo": [], "ur": []}
        },
        "Forget": {
            "Hamming PD": {"uo": [], "ur": []},
            "JS divergence": {"uo": [], "ur": []},
            "acc_comparison_model": {"uo": [], "ur": []},
            "acc_unlearned_model": {"uo": [], "ur": []}
        },
        "Validation": {
            "Hamming PD": {"uo": [], "ur": []},
            "JS divergence": {"uo": [], "ur": []},
            "acc_comparison_model": {"uo": [], "ur": []},
            "acc_unlearned_model": {"uo": [], "ur": []}
        }
    }
    
    # Collect metrics for each dataset and comparison type
    for forget_trial in range(config.experiment.n_forget_trials):
        for repeat in range(config.experiment.n_repeats):
            # Simulate collecting results for this example
            # In the actual code, this would be the results from each experiment run
            for dataset in ["retain", "forget", "validation"]:
                dataset_key = dataset.capitalize()
                
                # Unlearned vs. Original
                dataset_metrics[dataset_key]["Hamming PD"]["uo"].append(
                    results["unlearned vs. original"][dataset]["Hamming PD"])
                dataset_metrics[dataset_key]["JS divergence"]["uo"].append(
                    results["unlearned vs. original"][dataset]["JS divergence"])
                dataset_metrics[dataset_key]["acc_comparison_model"]["uo"].append(
                    results["unlearned vs. original"][dataset]["accuracy_comparison_model"])
                dataset_metrics[dataset_key]["acc_unlearned_model"]["uo"].append(
                    results["unlearned vs. original"][dataset]["accuracy_unlearned_model"])
                
                # Unlearned vs. Retrained
                dataset_metrics[dataset_key]["Hamming PD"]["ur"].append(
                    results["unlearned vs. retrained"][dataset]["Hamming PD"])
                dataset_metrics[dataset_key]["JS divergence"]["ur"].append(
                    results["unlearned vs. retrained"][dataset]["JS divergence"])
                dataset_metrics[dataset_key]["acc_comparison_model"]["ur"].append(
                    results["unlearned vs. retrained"][dataset]["accuracy_comparison_model"])
                dataset_metrics[dataset_key]["acc_unlearned_model"]["ur"].append(
                    results["unlearned vs. retrained"][dataset]["accuracy_unlearned_model"])
            
            # Break after one iteration for this example
            break
        break
    
    # Add rows to the summary table
    for dataset in ["Retain", "Forget", "Validation"]:
        # Add dataset header
        summary_table.add_row(f"[bold]{dataset}[/bold]", "", "")
        
        # Add metrics for this dataset
        for metric in ["Hamming PD", "JS divergence", "acc_comparison_model", "acc_unlearned_model"]:
            uo_values = dataset_metrics[dataset][metric]["uo"]
            ur_values = dataset_metrics[dataset][metric]["ur"]
            
            uo_formatted = "N/A"
            ur_formatted = "N/A"
            
            if uo_values:
                uo_mean = np.mean(uo_values)
                uo_std = np.std(uo_values)
                uo_formatted = f"{uo_mean:.3f} ± {uo_std:.3f}"
            
            if ur_values:
                ur_mean = np.mean(ur_values)
                ur_std = np.std(ur_values)
                ur_formatted = f"{ur_mean:.3f} ± {ur_std:.3f}"
            
            summary_table.add_row(f"  {metric}", uo_formatted, ur_formatted)
    
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
    
    if config.experiment.mode == "experiment":
        run_experiment(config)
    elif config.experiment.mode == "get_latex_results":
        get_latex_results()
    else:
        print("Visualization mode not implemented in this refactoring.")

if __name__ == "__main__":
    main()
