import argparse
import pdb
import pandas as pd
import torch
import torch.nn as nn
import copy
import json
import os
from pydantic import BaseModel, Field
from typing import Dict, Any
import wandb
import yaml
from torch.utils.data import DataLoader
import numpy as np

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


def train_model(dataloader, val_dataloader, n_features, n_classes, n_epochs, device="cpu"):
    """
    Trains and returns a NeuralNet model using the provided dataloader.
    """
    model = NeuralNet(n_features, n_classes)
    trainer = Trainer(model, train_dataloader=dataloader, val_dataloader=val_dataloader, n_epochs=n_epochs, device=device, disable_tqdm=True)
    trainer.train()
    trainer.eval()
    return model

class UnlearningManager:
    def __init__(self, args: argparse.Namespace, config: Config, device: str = "cpu"):
        self.args = args
        self.config = config
        self.device = device
        self.unlearn_type = None

    def apply_unlearning(self, 
                         unlearn_type: str, 
                         dataloaders: Dict[str, DataLoader], 
                         n_features: int, 
                         n_classes: int, 
                         n_epochs: int):
        """
        Applies the unlearning method specified by the unlearn_type.
        """
        self.unlearn_type = unlearn_type

        if self.unlearn_type == "sisa":
            return self._apply_unlearning_sisa(dataloaders, n_features, n_classes, n_epochs, self.device)
        elif self.unlearn_type == "amnesiac":
            return self._apply_unlearning_amnesiac(dataloaders, n_features, n_classes, n_epochs, self.device)
        elif self.unlearn_type == "sae":
            return self._apply_unlearning_sae(dataloaders, n_features, n_classes, n_epochs, self.device)
        elif self.unlearn_type == "scrub_r" or self.unlearn_type == "ssd":
            return self._apply_unlearning(dataloaders, n_features, n_classes, n_epochs, self.device)
        else:
            raise ValueError(f"Unknown unlearning type: {self.unlearn_type}")

    def _apply_unlearning(self, dataloaders, n_features, n_classes, n_epochs, device="cpu"):
        """
        Applies either Scrub+R or SSD to the given data. Returns the unlearned model, retrained model, and original model.
        """
        # 1) Train model on full dataset
        unlearned_model = train_model(dataloader=dataloaders["train_full_loader"], 
                                    val_dataloader=dataloaders["val_loader"], 
                                    n_features=n_features, 
                                    n_classes=n_classes, 
                                    n_epochs=n_epochs, 
                                    device=device)

        # 2) Train retrained model on retain dataset
        retrained_model = train_model(dataloader=dataloaders["train_retain_loader"], 
                                    val_dataloader=dataloaders["val_loader"], 
                                    n_features=n_features, 
                                    n_classes=n_classes, 
                                    n_epochs=n_epochs, 
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
        trainer = AmnesiacTrainer(unlearned_model, lr=3e-3, device=device, cache_gradients=True, disable_tqdm=True)

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
        retrained_trainer = AmnesiacTrainer(retrained_model, lr=3e-3, device=device, cache_gradients=True, disable_tqdm=True)
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


def run_experiment(config: Config):
    """
    Main experiment flow with multiple forget set trials
    """

    console = Console()
    n_repeats = config.experiment.n_repeats
    n_epochs = config.experiment.n_epochs
    n_features = config.data.n_features
    n_classes = config.data.n_classes
    n_forget_trials = config.experiment.n_forget_trials

    # start wandb logging
    if config.wandb.enabled:
        os.makedirs(config.wandb.dir, exist_ok=True)
        wandb.init(
            project=config.wandb.project,
            mode=config.wandb.mode,
            dir=config.wandb.dir)

        # Set all experiment parameters
        # TODO: add all experiment parameters to wandb

    # Generate data once
    data_generator = generate_data(config.data)

    # Prepare structure to store aggregated results
    aggregated_results = {
        "unlearned vs. original": {"retain": [], "forget": [], "validation": []},
        "unlearned vs. retrained": {"retain": [], "forget": [], "validation": []},
    }

    # Create a panel for live updates
    info_panel = Panel("Starting experiments...", title="Current Status")
    
    # Create progress bars for both loops
    live = Live(info_panel, refresh_per_second=4)
    
    # Set up exception handler to clean up Live display
    def handle_pdb(*args):
        live.stop()  # Stop live display before pdb
        result = original_trace(*args)  # Run pdb
        live.start()  # Restart live display after pdb
        return result
    
    original_trace = pdb.set_trace
    pdb.set_trace = handle_pdb
    
    unlearning_manager = UnlearningManager(config.system, config.system.device)

    try:
        live.start()
        for forget_trial in tqdm(range(n_forget_trials), desc="Forget trials", position=0):
            # Create new forget/retain split
            dataloaders = create_forget_retain_split(data_generator, config)
            
            for i in tqdm(range(n_repeats), desc="Repeats", position=1, leave=False):
                # Update the panel with current status
                info_panel.title = f"[bold blue]Forget Trial {forget_trial + 1}/{n_forget_trials}, Iteration {i + 1}/{n_repeats}"
                info_panel.subtitle = f"[bold]Unlearning type:[/bold] {config.experiment.unlearn_type}"
                
                # Call the appropriate unlearning function based on the type
                unlearned_model, retrained_model, original_model = unlearning_manager.apply_unlearning(
                    config.experiment.unlearn_type,
                    dataloaders,
                    n_features,
                    n_classes,
                    n_epochs
                )

                # Evaluate
                results = evaluate_models(unlearned_model, retrained_model, original_model, dataloaders)

                # Aggregate results
                for comp_type in aggregated_results.keys():
                    for subset in aggregated_results[comp_type].keys():
                        aggregated_results[comp_type][subset].append(results[comp_type][subset])
                
                # Update the panel content with latest metrics
                info_panel.renderable = f"""[green]Key metrics for this iteration:[/green]
Hamming PD (validation): {results['unlearned vs. retrained']['validation']['Hamming PD']:.4f}
JS divergence (validation): {results['unlearned vs. retrained']['validation']['JS divergence']:.4f}"""
                
                live.refresh()

        # Save results and print final output
        os.makedirs("experiments/results", exist_ok=True)
        output_file = f"experiments/results/{config.experiment.unlearn_type}_results.json"
        with open(output_file, "w") as f:
            json.dump(aggregated_results, f)
        
        live.stop()
        
        console.print(f"[bold green]Results saved to {output_file}.")

        with open(f'experiments/{config.experiment.unlearn_type}_results.json', 'w') as f:
            json.dump(results, f)


        results_dict = results

        def calculate_averages(results_dict):
            averages = {}
            
            for comparison in results_dict:  # 'unlearned vs. original', 'unlearned vs. retrained'
                averages[comparison] = {}
                
                # Get all metrics from the first split to use as keys
                metrics = list(next(iter(results_dict[comparison].values())).keys())
                
                # Calculate average for each metric
                for metric in metrics:
                    values = [results_dict[comparison][split][metric] 
                            for split in ['retain', 'forget', 'validation']]
                    averages[comparison][metric] = sum(values) / len(values)
            
            return averages

        # Example usage:
        averages = calculate_averages(results_dict)

        # Pretty print the results
        for comparison, metrics in averages.items():
            print(f"\n{comparison}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        
        
    finally:
        # Restore original pdb.set_trace and ensure Live display is stopped
        pdb.set_trace = original_trace
        live.stop()


def get_latex_results():
    """
    Loads all .json results in `experiments/results` and prints a combined LaTeX table.
    """
    result_dir = "experiments/results"
    result_files = [f for f in os.listdir(result_dir) if f.endswith(".json")]
    all_results = {}

    for file in result_files:
        with open(os.path.join(result_dir, file), "r") as f:
            results = json.load(f)
        all_results.update(results)

    print("\n\n\n" + create_latex_table(all_results))


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
