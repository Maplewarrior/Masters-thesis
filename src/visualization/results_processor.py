from rich.table import Table
import numpy as np

def create_results_table(results=None, status_text=""):
    """Creates a table to display experiment results."""
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

def create_summary_table(dataset_metrics, unlearn_type):
    """Creates a summary table from collected metrics."""
    summary_table = Table(title=f"Summary for {unlearn_type} Unlearning")
    summary_table.add_column("Dataset/Metric", style="cyan")
    summary_table.add_column("Unlearned vs. Original", style="green")
    summary_table.add_column("Unlearned vs. Retrained", style="yellow")
    summary_table.add_column("Retrained vs. Original", style="blue")
    
    # Add rows to the summary table
    for dataset in ["Retain", "Forget", "Validation"]:
        # Add dataset header
        summary_table.add_row(f"[bold]{dataset}[/bold]", "", "")
        
        # Add metrics for this dataset
        for metric in ["Hamming PD", "JS divergence", "acc_comparison_model", "acc_unlearned_model"]:
            uo_values = dataset_metrics[dataset][metric]["uo"]
            ur_values = dataset_metrics[dataset][metric]["ur"]
            ro_values = dataset_metrics[dataset][metric]["ro"]
            uo_formatted = "N/A"
            ur_formatted = "N/A"
            ro_formatted = "N/A"
            
            if uo_values:
                uo_mean = np.mean(uo_values)
                uo_std = np.std(uo_values)
                uo_formatted = f"{uo_mean:.3f} ± {uo_std:.3f}"
            
            if ur_values:
                ur_mean = np.mean(ur_values)
                ur_std = np.std(ur_values)
                ur_formatted = f"{ur_mean:.3f} ± {ur_std:.3f}"
            
            if ro_values:
                ro_mean = np.mean(ro_values)
                ro_std = np.std(ro_values)
                ro_formatted = f"{ro_mean:.3f} ± {ro_std:.3f}"
            
            summary_table.add_row(f"  {metric}", uo_formatted, ur_formatted, ro_formatted)
    
    return summary_table

def collect_metrics(results, dataset_metrics):
    """Collects metrics from results into the dataset_metrics dictionary."""
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
        dataset_metrics[dataset_key]["Hamming PD"]["ro"].append(
            results["retrained vs. original"][dataset]["Hamming PD"])
        dataset_metrics[dataset_key]["JS divergence"]["ro"].append(
            results["retrained vs. original"][dataset]["JS divergence"])
        dataset_metrics[dataset_key]["acc_comparison_model"]["ro"].append(
            results["retrained vs. original"][dataset]["accuracy_comparison_model"])
        dataset_metrics[dataset_key]["acc_unlearned_model"]["ro"].append(
            results["retrained vs. original"][dataset]["accuracy_unlearned_model"])
        
        # Unlearned vs. Retrained
        dataset_metrics[dataset_key]["Hamming PD"]["ur"].append(
            results["unlearned vs. retrained"][dataset]["Hamming PD"])
        dataset_metrics[dataset_key]["JS divergence"]["ur"].append(
            results["unlearned vs. retrained"][dataset]["JS divergence"])
        dataset_metrics[dataset_key]["acc_comparison_model"]["ur"].append(
            results["unlearned vs. retrained"][dataset]["accuracy_comparison_model"])
        dataset_metrics[dataset_key]["acc_unlearned_model"]["ur"].append(
            results["unlearned vs. retrained"][dataset]["accuracy_unlearned_model"])
        dataset_metrics[dataset_key]["Hamming PD"]["ro"].append(
            results["retrained vs. original"][dataset]["Hamming PD"])
        dataset_metrics[dataset_key]["JS divergence"]["ro"].append(
            results["unlearned vs. retrained"][dataset]["JS divergence"])
        dataset_metrics[dataset_key]["acc_comparison_model"]["ro"].append(
            results["unlearned vs. retrained"][dataset]["accuracy_comparison_model"])
        dataset_metrics[dataset_key]["acc_unlearned_model"]["ro"].append(
            results["unlearned vs. retrained"][dataset]["accuracy_unlearned_model"])
    
    return dataset_metrics