import argparse
import yaml
from pathlib import Path
import copy
from tqdm import tqdm
import numpy as np
import subprocess
import os
import uuid

def run_parameter_sweep(base_config_path, param_name, param_values, output_dir="configs/sweep", models=None, experiment_group_prefix=None, use_wandb=False):
    """
    Run a parameter sweep by varying a single parameter across multiple values.
    
    Args:
        base_config_path: Path to the base config file
        param_name: Parameter to vary (in dot notation, e.g., 'data.n_samples')
        param_values: List of values to use for the parameter
        output_dir: Directory to store generated config files
        models: List of model types to run experiments for (if None, uses the model in base config)
        experiment_group_prefix: Prefix for the experiment group name (if None, uses "sweep")
        use_wandb: Whether to use Weights & Biases for logging
    """
    # Load base config
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Parse parameter path (e.g., 'data.n_samples' -> ['data', 'n_samples'])
    param_path = param_name.split('.')
    
    # If no models specified, use the one from base config
    if models is None:
        models = [base_config['experiment']['unlearn_type']]
    
    # Set default experiment group prefix if not provided
    if experiment_group_prefix is None:
        experiment_group_prefix = "sweep"
    
    # Run experiments for each parameter value and model
    for value in tqdm(param_values, desc=f"Sweeping {param_name}"):
        for model in models:
            # Create a copy of the base config
            config = copy.deepcopy(base_config)
            
            # Update the parameter value
            current = config
            for key in param_path[:-1]:
                current = current[key]
            current[param_path[-1]] = value

            config['experiment']['unlearn_type'] = model
            
            
            # Create a unique experiment group name based on the parameter and model
            if config['experiment']['experiment_group'] is None:
                config['experiment']['experiment_group'] = f"{experiment_group_prefix}"
            else:
                config['experiment']['experiment_group'] = f"{config['experiment']['experiment_group']}_{experiment_group_prefix}"
            
            # Append parameter and value to experiment ID
            param_short_name = param_path[-1]
            
            # Check if experiment_id exists in the config
            if 'experiment_id' not in config['experiment'] or config['experiment']['experiment_id'] is None:
                # Generate a unique ID using uuid and combine with parameter info
                unique_id = str(uuid.uuid4())[:8]
                config['experiment']['experiment_id'] = f"{unique_id}_{param_short_name}_{value}"
            else:
                config['experiment']['experiment_id'] = f"{config['experiment']['experiment_id']}_{param_short_name}_{value}"
            

            # Save the config to a file
            config_filename = f"{output_dir}/{param_path[-1]}_{value}_{model}.yaml"
            with open(config_filename, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Run the experiment with this config
            cmd = ["python", "main.py", "experiment", "--unlearn-type", model, "--config", config_filename]
            
            # Add wandb flag if requested
            if use_wandb:
                cmd.append("--wandb")

            subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameter sweep experiments")
    parser.add_argument("--base_config", type=str, required=True, help="Path to base config file")
    parser.add_argument("--param", type=str, required=True, help="Parameter to vary (in dot notation)")
    parser.add_argument("--min", type=float, required=True, help="Minimum parameter value")
    parser.add_argument("--max", type=float, required=True, help="Maximum parameter value")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps between min and max")
    parser.add_argument("--log_scale", action="store_true", help="Use logarithmic scale for values")
    parser.add_argument("--output_dir", type=str, default="configs/sweep", help="Directory for output configs")
    parser.add_argument("--models", type=str, nargs="+", help="List of model types to run experiments for")
    parser.add_argument("--experiment_group", type=str, default=None, 
                        help="Prefix for experiment group name (default: 'sweep')")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Generate parameter values
    if args.log_scale:
        param_values = np.logspace(np.log10(args.min), np.log10(args.max), args.steps)
    else:
        param_values = np.linspace(args.min, args.max, args.steps)
    
    # Convert to appropriate type (int or float)
    if all(float(x).is_integer() for x in param_values):
        param_values = [int(x) for x in param_values]
    else:
        param_values = [float(x) for x in param_values]
    
    run_parameter_sweep(args.base_config, args.param, param_values, args.output_dir, args.models, 
                        args.experiment_group, args.wandb) 