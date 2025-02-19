def create_latex_table(results_dict):
    """
    Convert nested results dictionary to a LaTeX table string.
    Handles different unlearning algorithms and model comparisons.
    
    Args:
        results_dict (dict): Nested dictionary containing experiment results
        {unlearning_algorithm: 
            {model_comparison: 
                {dataset_type: [evaluation_metrics]}}}
    
    Returns:
        str: Formatted LaTeX table string
    """
    # Get all unique metrics across all comparisons and algorithms
    all_metrics = set()
    unlearning_algorithms = list(results_dict.keys())
    
    # Get all model comparisons (they should be the same for each algorithm)
    model_comparisons = list(results_dict[unlearning_algorithms[0]].keys())
    
    # Collect all unique metrics
    for algorithm in results_dict:
        for comparison in results_dict[algorithm]:
            for dataset in ['retain', 'forget', 'validation']:
                if results_dict[algorithm][comparison][dataset]:
                    metrics = results_dict[algorithm][comparison][dataset][0].keys()
                    all_metrics.update(metrics)
    all_metrics = sorted(all_metrics)
    
    # Dataset types
    datasets = ['retain', 'forget', 'validation']
    
    # Calculate means for each combination
    means = {}
    for algorithm in unlearning_algorithms:
        means[algorithm] = {}
        for comparison in model_comparisons:
            means[algorithm][comparison] = {}
            for dataset in datasets:
                means[algorithm][comparison][dataset] = {}
                # Initialize all metrics with None
                for metric in all_metrics:
                    means[algorithm][comparison][dataset][metric] = None
                
                # Calculate means for available metrics
                if results_dict[algorithm][comparison][dataset]:
                    for metric in results_dict[algorithm][comparison][dataset][0].keys():
                        values = []
                        for metric_dict in results_dict[algorithm][comparison][dataset]:
                            values.append(metric_dict[metric])
                        
                        mean = sum(values) / len(values)
                        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                        means[algorithm][comparison][dataset][metric] = (mean, std)

    # Begin LaTeX table
    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scalebox{0.8}{"
        "\\begin{tabular}{l" + "c" * (len(model_comparisons) * len(unlearning_algorithms)) + "}",
        "\\toprule"
    ]
    
    # Create first header row with unlearning algorithms
    header1 = ["Dataset"]
    for algorithm in unlearning_algorithms:
        header1.extend([f"\\multicolumn{{{len(model_comparisons)}}}{{c}}{{{algorithm.replace('_', '\\_')}}}"])
    latex.append(" & ".join(header1) + " \\\\")
    
    # Create second header row with model comparisons
    header2 = [""]
    for _ in unlearning_algorithms:
        header2.extend([comp.replace("_", "\\_") for comp in model_comparisons])
    latex.append(" & ".join(header2) + " \\\\")
    
    latex.append("\\midrule")
    
    # Add data rows
    for dataset in datasets:
        # Add dataset name as a subheader
        latex.append(f"\\multicolumn{{{len(model_comparisons) * len(unlearning_algorithms) + 1}}}{{l}}{{\\textbf{{{dataset.capitalize()}}}}}" + " \\\\")
        
        # Add metric rows for this dataset
        for metric in all_metrics:
            row = [metric.replace("_", "\\_")]
            for algorithm in unlearning_algorithms:
                for comparison in model_comparisons:
                    result = means[algorithm][comparison][dataset][metric]
                    if result is None:
                        row.append("-")
                    else:
                        mean, std = result
                        row.append(f"{mean:.3f} ± {std:.3f}")
            latex.append(" & ".join(row) + " \\\\")
            
        # Add a small space between datasets
        if dataset != datasets[-1]:
            latex.append("\\addlinespace")
    
    # Close the table
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\caption{Experiment Results by Unlearning Algorithm}",
        "\\label{tab:experiment-results}",
        "\\end{table}"
    ])
    
    return "\n".join(latex)