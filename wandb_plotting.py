import wandb
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union
import numpy as np
import concurrent.futures
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
import json

class WandbClient:
    """
    A client for interacting with Weights & Biases.
    
    This class encapsulates functionality for fetching runs, run history,
    projects, and entities from Weights & Biases.
    """
    
    def __init__(self, entity: str, project: str):
        """
        Initialize the WandbClient.
        
        Args:
            entity: The wandb entity (username or team name)
            project: The wandb project name
        """
        self.entity = entity
        self.project = project
        self.api = wandb.Api()
    
    def get_metrics(self, run_id: str) -> List[str]:
        """
        Get all metrics for a given run.
        """
        run = self.api.run(f"{self.entity}/{self.project}/{run_id}")
        return run.summary.keys()
    

    def get_run_results(self, run_id: str) -> Dict[str, Any]:
        """
        Get the final values of all metrics for a given run.
        
        Args:
            run_id: The ID of the run to fetch
            
        Returns:
            Dictionary containing the last recorded values for each metric
        """
        run = self.api.run(f"{self.entity}/{self.project}/{run_id}")
        
        # Define the metrics we want to extract
        metrics = [
            # Validation metrics
            "validation/unlearned_vs_original/hamming_pd",
            "validation/unlearned_vs_original/js_divergence",
            "validation/retrained_vs_original/hamming_pd",
            "validation/retrained_vs_original/js_divergence",
            "validation/accuracy/unlearned",
            "validation/accuracy/retrained",
            "validation/accuracy/original",
            
            # Retain metrics
            "retain/unlearned_vs_original/hamming_pd",
            "retain/unlearned_vs_original/js_divergence",
            "retain/retrained_vs_original/hamming_pd",
            "retain/retrained_vs_original/js_divergence",
            "retain/accuracy/unlearned",
            "retain/accuracy/retrained",
            "retain/accuracy/original",
            
            # Forget metrics
            "forget/unlearned_vs_original/hamming_pd",
            "forget/unlearned_vs_original/js_divergence",
            "forget/retrained_vs_original/hamming_pd",
            "forget/retrained_vs_original/js_divergence",
            "forget/accuracy/unlearned",
            "forget/accuracy/retrained",
            "forget/accuracy/original"
        ]
        
        # print(f"Fetching history for run {run_id}...")
        history = run.history()
        
        # Get the last values for each metric
        results = {}
        for metric in metrics:
            if metric in history.columns:
                # Convert pandas Series to list
                metric_values = history[metric].dropna().tolist()
                if metric_values:
                    results[metric] = metric_values[-1]  # Get the last value
                else:
                    results[metric] = None
            else:
                results[metric] = None
        
        # Add run information
        results['run_id'] = run_id
        results['run_name'] = run.name
        
        return results
    
    def get_multiple_run_results(self, run_ids: List[str], max_workers: int = 8) -> List[Dict[str, Any]]:
        """
        Get results for multiple runs in parallel.
        
        Args:
            run_ids: List of run IDs to fetch
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of dictionaries containing results for all runs
        """
        all_results = []
        
        # Define a worker function to process a single run
        def process_run(run_id):
            try:
                return self.get_run_results(run_id)
            except Exception as e:
                print(f"Error processing run {run_id}: {e}")
                return None
        
        # Use ThreadPoolExecutor for I/O-bound tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_run_id = {executor.submit(process_run, run_id): run_id for run_id in run_ids}
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_run_id), 
                              total=len(run_ids), 
                              desc="Processing runs"):
                run_id = future_to_run_id[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                except Exception as e:
                    print(f"Exception occurred while processing run {run_id}: {e}")
        
        return all_results
    

    def get_run_ids(self, runs, num_workers: int = 16) -> List[str]:
        """
        Get run IDs for runs matching the given filters.
        
        This method extracts run IDs directly from the runs list without making
        additional API calls when possible.
        """
        # First try to extract IDs directly from the runs list
        # This is much faster if the IDs are already loaded
        run_ids = []
        for run in runs:
            try:
                # Try to access the ID directly without triggering API calls
                run_ids.append(run.id)
            except:
                # If that fails, we'll collect the problematic runs for parallel processing
                pass
                
        return run_ids

    def get_runs(
        self,
        filters: Optional[Dict[str, Any]] = None,
        max_workers: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Fetch runs from Weights & Biases and return as a list of dictionaries.
        
        Args:
            filters: Dictionary of filters to apply (e.g., {"config.batch_size": 32})
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of dictionaries containing run information
        """
        # Build the query
        runs = self.api.runs(f"{self.entity}/{self.project}", filters=filters)
        

        print("Getting run ids")
        # Try to get all run IDs at once using the API's internal methods
        run_ids = self.get_run_ids(runs, num_workers=32)
        print(f"Found {len(run_ids)} runs matching filters")

            
        print("Getting run results")
        run_results = self.get_multiple_run_results(run_ids, max_workers=max_workers)
        print("done getting run results")

        return run_results
    
    def group_runs_by_experiment_id(self, runs: List[Dict[str, Any]], sweep_metric: str = None, max_workers: int = 8) -> Dict[str, Dict[str, float]]:
        """
        Group runs by experiment_id and calculate the mean of numeric metrics.
        
        Args:
            runs: List of run result dictionaries
            sweep_metric: overwrite experiment_id with metric, usually each experiment_id is a new value for a parameter in the sweep
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping experiment_ids to mean metrics
        """
        import concurrent.futures
        from tqdm import tqdm
        
        # Define a worker function to process a single run
        def process_run(run):
            try:
                run_obj = self.api.run(f"{self.entity}/{self.project}/{run['run_id']}")
                experiment_id = run_obj.config.get("experiment", {}).get("experiment_id", "unknown")
                
                # If sweep_metric is provided, use it instead of experiment_id
                if sweep_metric is not None:
                    keys = sweep_metric.split(".")
                    current_level = run_obj.config
                    for key in keys:
                        if isinstance(current_level, dict) and key in current_level:
                            current_level = current_level[key]
                        else:
                            current_level = None
                            break
                    
                    if current_level is not None:
                        experiment_id = str(current_level)
                
                return (experiment_id, run)
            except Exception as e:
                print(f"Error processing run {run['run_id']}: {e}")
                return None
        
        # First, we need to extract experiment_id from each run in parallel
        experiment_groups = {}
        
        # Use ThreadPoolExecutor for I/O-bound tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_run = {executor.submit(process_run, run): run for run in runs}
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_run), 
                              total=len(runs), 
                              desc="Grouping runs"):
                try:
                    result = future.result()
                    if result:
                        experiment_id, run = result
                        if experiment_id not in experiment_groups:
                            experiment_groups[experiment_id] = []
                        experiment_groups[experiment_id].append(run)
                except Exception as e:
                    print(f"Exception occurred while grouping runs: {e}")
        
        # Calculate means for each group
        experiment_means = {}
        
        for experiment_id, group_runs in experiment_groups.items():
            print(f"Calculating means for group: {experiment_id} ({len(group_runs)} runs)")
            
            # Initialize with metrics from the first run
            metrics = {}
            for key, value in group_runs[0].items():
                # Skip non-numeric and metadata fields
                if key in ['run_id', 'run_name'] or not isinstance(value, (int, float)) or value is None:
                    continue
                metrics[key] = []
            
            # Collect all values for each metric
            for run in group_runs:
                for key in metrics.keys():
                    if key in run and run[key] is not None and isinstance(run[key], (int, float)):
                        metrics[key].append(run[key])
            
            # Calculate means
            means = {}
            for key, values in metrics.items():
                if values:  # Only calculate if we have values
                    means[key] = sum(values) / len(values)
                else:
                    means[key] = None
            
            # Add metadata
            means['group_key'] = experiment_id
            means['num_runs'] = len(group_runs)
            
            experiment_means[experiment_id] = means
        
        return experiment_means
    
    @staticmethod
    def get_projects(entity: str) -> List[str]:
        """
        Get all projects for a given entity.
        
        Args:
            entity: The wandb entity (username or team name)
            
        Returns:
            List of project names
        """
        api = wandb.Api()
        projects = api.projects(entity)
        return [project.name for project in projects]
    
    @staticmethod
    def get_entities() -> List[str]:
        """
        Get all entities accessible to the current user.
        
        Returns:
            List of entity names
        """
        api = wandb.Api()
        current_user = api.viewer
        return [current_user.entity]

def plot_grid(results: Dict[str, Dict[str, float]], sweep_metric: str = "ood_ratio"):
    """
    Plot a grid of metrics for each unlearn type.
    
    Args:
        results: Dictionary mapping unlearn_type to experiment results
        sweep_metric: The metric used for the x-axis (default: "ood_ratio")
    """
    # Collect all metrics across all unlearn types
    all_metrics = set()
    for unlearn_type, experiments in results.items():
        for experiment_id, metrics in experiments.items():
            for metric_name in metrics.keys():
                # Skip non-metric fields
                if metric_name not in ['group_key', 'num_runs']:
                    all_metrics.add(metric_name)
    
    # Remove None values
    all_metrics = [m for m in all_metrics if m is not None]
    
    # Sort metrics by category
    metric_categories = ["validation", "retain", "forget"]
    metric_subcategories = ["unlearned_vs_original", "retrained_vs_original", "accuracy"]
    
    sorted_metrics = []
    for category in metric_categories:
        for subcategory in metric_subcategories:
            for metric in sorted(all_metrics):
                if f"{category}/{subcategory}" in metric:
                    sorted_metrics.append(metric)
    
    # Add any remaining metrics
    for metric in sorted(all_metrics):
        if metric not in sorted_metrics:
            sorted_metrics.append(metric)
    
    # Determine grid dimensions
    n_metrics = len(sorted_metrics)
    n_cols = 3  # You can adjust this
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure
    fig = plt.figure(figsize=(n_cols * 5, n_rows * 4))
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    # Plot each metric
    for i, metric in enumerate(sorted_metrics):
        row, col = i // n_cols, i % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Plot data for each unlearn type
        for unlearn_type, experiments in results.items():
            x_values = []
            y_values = []
            
            # Extract x and y values
            for experiment_id, metrics in experiments.items():
                if metric in metrics and metrics[metric] is not None:
                    try:
                        x_value = float(experiment_id)
                        x_values.append(x_value)
                        y_values.append(metrics[metric])
                    except ValueError:
                        # Skip if experiment_id can't be converted to float
                        pass
            
            # Sort by x values
            if x_values:
                sorted_pairs = sorted(zip(x_values, y_values))
                x_values, y_values = zip(*sorted_pairs)
                
                # Plot the line
                ax.plot(x_values, y_values, marker='o', label=unlearn_type)
        
        # Set labels and title
        ax.set_xlabel(sweep_metric)
        ax.set_ylabel(metric.split('/')[-1])
        
        # Create a more readable title from the metric name
        title_parts = metric.split('/')
        if len(title_parts) >= 2:
            title = f"{title_parts[0]}: {' '.join(title_parts[1:])}"
        else:
            title = metric
        ax.set_title(title, fontsize=10)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"metrics_by_{sweep_metric}.png", dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    # # Example usage of the WandbClient class
    # unlearn_types = ["amnesiac"]

    # entity = "machine-unlearning-thesis"
    # project = "unlearning-experiments"

    # results = {}
    # for unlearn_type in unlearn_types:
    #     filters = {
    #         "config.experiment.experiment_group": "sweep_forget_ood_ratio",
    #         "config.experiment.unlearn_type": unlearn_type
    #     }
        
    #     client = WandbClient(entity=entity, project=project)
    #     runs = client.get_runs(filters=filters)

    #     experiment_means = client.group_runs_by_experiment_id(runs, sweep_metric="forget.ood_ratio")

    #     results[unlearn_type] = experiment_means


    #     print(experiment_means)


    # # save results
    # with open("results.json", "w") as f:
    #     json.dump(results, f)


    # load results
    with open("results.json", "r") as f:
        results = json.load(f)

    plot_grid(results)