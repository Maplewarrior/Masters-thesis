import argparse
import pdb
import pandas as pd
import torch
import torch.nn as nn
import copy
import json
import os

from src.modelling.trainer import Trainer
from src.modelling.neural_network import NeuralNet
from src.data_utils.synthetic_data import DataGenerator, create_dataloaders
from src.modelling.selective_synaptic_dampening import SelectiveSynapticDampening
from src.modelling.scrub import ScrubR
from src.modelling.sisa import SISA
from src.modelling.sae_unlearner import SAEUnlearner
from src.modelling.SAE import SAE
from src.evaluation.unlearning_evaluator import UnlearningEvaluator
from src.evaluation.results_table import create_latex_table
from src.modelling.amnesiac import AmnesiacTrainer


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="experiment",
        choices=["experiment", "visualize", "get_latex_results"],
        help="What to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--unlearn-type",
        type=str,
        default="SSD",
        choices=["SSD", "Scrub+R", "SISA", "amnesiac", "SAE"],
        help="What type of unlearning algorithm to apply.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Torch device (default: %(default)s)",
    )
    return parser.parse_args()


def generate_data(args, n_features=25, n_classes=4, random_state=42):
    """
    Generates and splits the synthetic data. Returns a data generator and associated dataloaders.
    """
    data_generator = DataGenerator(random_state=random_state)
    data_generator.generate_data(
        n_samples=1000,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=2,
        n_redundant=0,
        n_outliers=50,
        outlier_scale=4.0,
        outlier_variance=0.4,
        outlier_class=1
    )
    data_generator.split_data(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    data_generator.draw_forget_set(n_points=50, class_idx=None, ood_ratio=0.5)

    # For amnesiac, we need indices, so batch_size differs:
    if args.unlearn_type == "amnesiac":
        batch_size = 1
        dataloaders = create_dataloaders(
            data_generator, batch_size=batch_size, use_indices=True, device=args.device
        )
    else:
        batch_size = 32
        dataloaders = create_dataloaders(
            data_generator, batch_size=batch_size, onehot_labels=True, device=args.device
        )

    return data_generator, dataloaders


def train_model(dataloader, val_dataloader, n_features, n_classes, n_epochs, device="cpu"):
    """
    Trains and returns a NeuralNet model using the provided dataloader.
    """
    model = NeuralNet(n_features, n_classes)
    trainer = Trainer(model, train_dataloader=dataloader, val_dataloader=val_dataloader, n_epochs=n_epochs, device=device)
    trainer.train()
    trainer.eval()
    return model


def apply_unlearning_scrub_r_SSD(unlearn_type, dataloaders, n_features, n_classes, n_epochs, device="cpu"):
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
    if unlearn_type == "Scrub+R":
        alpha = 1.0
        gamma = 1.0
        scrub = ScrubR(unlearned_model, original_model, alpha=alpha, gamma=gamma)
        scrub(
            dataloaders["train_retain_loader"],
            dataloaders["train_forget_loader"],
            dataloaders["val_loader"],
            n_rounds=6,
        )
    elif unlearn_type == "SSD":
        alpha = 7.5
        _lambda = 0.5
        criterion = nn.CrossEntropyLoss()
        ssd = SelectiveSynapticDampening(unlearned_model, criterion, alpha=alpha, _lambda=_lambda)
        ssd(
            full_dataloader=dataloaders["train_full_loader"],
            forget_dataloader=dataloaders["train_forget_loader"],
        )

    return unlearned_model, retrained_model, original_model


def apply_unlearning_sisa(dataloaders, n_features, n_classes, n_epochs, device="cpu"):
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


def apply_unlearning_amnesiac(dataloaders, n_features, n_classes, n_epochs, device="cpu"):
    """
    Applies the amnesiac unlearning method.
    Returns the unlearned model, retrained model, and original model.
    """
    unlearned_model = NeuralNet(n_features, n_classes)
    trainer = AmnesiacTrainer(unlearned_model, lr=0.1, device=device, cache_gradients=True)

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
    retrained_trainer = AmnesiacTrainer(retrained_model, lr=0.1, device=device, cache_gradients=True)
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


def apply_unlearning_sae():
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


def run_experiment(args):
    """
    Main experiment flow:
    1) Generate data
    2) For each unlearning method, repeatedly run unlearning & evaluation
    3) Save results to JSON
    4) Print LaTeX table
    """
    n_repeats = 10
    n_epochs = 20
    n_features = 25
    n_classes = 4

    data_generator, dataloaders = generate_data(args, n_features, n_classes, random_state=42)

    # Prepare structure to store aggregated results over repeats
    aggregated_results = {
        "unlearned vs. original": {"retain": [], "forget": [], "validation": []},
        "unlearned vs. retrained": {"retain": [], "forget": [], "validation": []},
    }

    # Dictionary dispatch for different unlearning methods
    unlearning_dispatch = {
        "SSD": apply_unlearning_scrub_r_SSD,
        "Scrub+R": apply_unlearning_scrub_r_SSD,
        "SISA": apply_unlearning_sisa,
        "amnesiac": apply_unlearning_amnesiac,
        "SAE": apply_unlearning_sae,
    }

    for i in range(n_repeats):
        if args.unlearn_type in ["SSD", "Scrub+R"]:
            unlearned_model, retrained_model, original_model = unlearning_dispatch[args.unlearn_type](
                args.unlearn_type,
                dataloaders,
                n_features,
                n_classes,
                n_epochs,
                device=args.device
            )
        else:
            unlearned_model, retrained_model, original_model = unlearning_dispatch[args.unlearn_type](
                dataloaders, n_features, n_classes, n_epochs, device=args.device
            )

        # Evaluate
        results = evaluate_models(unlearned_model, retrained_model, original_model, dataloaders)

        # Aggregate
        for comp_type in aggregated_results.keys():
            for subset in aggregated_results[comp_type].keys():
                aggregated_results[comp_type][subset].append(results[comp_type][subset])

    # Wrap in a top-level dict keyed by unlearn type
    final_results = {args.unlearn_type: aggregated_results}

    # Save to file
    os.makedirs("experiments/results", exist_ok=True)
    output_file = f"experiments/results/{args.unlearn_type}_results.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f)
    print(f"Results saved to {output_file}.")

    # Print LaTeX table for immediate reference
    results_table = create_latex_table(final_results)
    print(results_table)
    pdb.set_trace()


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
    args = parse_arguments()

    if args.mode == "experiment":
        run_experiment(args)
    elif args.mode == "get_latex_results":
        get_latex_results()
    else:
        print("Visualization mode not implemented in this refactoring.")


if __name__ == "__main__":
    main()
