from wandb_client import WandbClient
from datetime import datetime
import json
import os
import argparse
import matplotlib.pyplot as plt
from typing import Dict, Any

if __name__ == "__main__":

    experiments_folder = "results/n_forget"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str.lower,
        default="experiment",
        choices=["data", "plot"],
        help="extract data from wandb or plot the data",
    )

    args = parser.parse_args()

    if args.mode == "data":

        # Example usage of the WandbClient class
        unlearn_types = ["amnesiac", "sisa", "scrub+r", "ssd"]

        entity = "machine-unlearning-thesis"
        project = "unlearning-experiments"

        results = {}
        for unlearn_type in unlearn_types:
            # Get today's date in ISO format (YYYY-MM-DD)
            today = datetime.now().strftime("%Y-%m-%d")
            
            filters = {
                "config.experiment.experiment_group": "sweep_forget_n_forget",
                "config.experiment.unlearn_type": unlearn_type,
                "created_at": {"$gte": "2025-03-06"}  # Filter for runs created on or after May 3, 2025
            }
            
            client = WandbClient(entity=entity, project=project)
            runs = client.get_runs(filters=filters)

            experiment_means = client.group_runs_by_experiment_id(runs, sweep_metric="forget.n_forget")

            results[unlearn_type] = experiment_means


            # print(experiment_means)

        os.makedirs(experiments_folder, exist_ok=True)

        # save results
        with open(f"{experiments_folder}/results_n_forget.json", "w") as f:
            json.dump(results, f)

    elif args.mode == "plot":
        from exp1_n_ood import plot
        # raise error if results_ood_ratio.json does not exist
        if not os.path.exists(f"{experiments_folder}/results_n_forget.json"):
            raise FileNotFoundError(f"results_n_forget.json does not exist in {experiments_folder}")

        # Load results
        with open(f"{experiments_folder}/results_n_forget.json", "r") as f:
            results = json.load(f)
        
        # plot the data
        plot(results, save_to_pdf=True, save_to_png=True, experiments_folder=experiments_folder)
