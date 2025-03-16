# Masters-thesis

## Quickstart

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the data is in the `data/CSV` directory.

3. Run the EDA:

```bash
python src/data-exploration/EDA.py
```

## Run experiments

We use Hydra for configuration management, which provides a flexible way to configure experiments and run parameter sweeps. Hydra allows you to override any configuration parameter from the command line without modifying config files.

### Basic experiment execution

To run a basic experiment with default settings:

```bash
python main.py
```

### Overriding configuration values

You can override any configuration value from the command line:

```bash
# Run with a specific unlearning method
python main.py experiment.unlearn_type=amnesiac

# Run on a specific device
python main.py system.device=cuda

# Change data parameters
python main.py data.n_samples=2000 data.n_features=50

# Enable WandB logging
python main.py wandb.enabled=true wandb.mode=online
```

### Running multiple experiments (parameter sweeps)

Hydra's multirun feature allows you to run parameter sweeps easily:

```bash
# Run experiments with different unlearning methods
python main.py --multirun experiment.unlearn_type=ssd,amnesiac,sisa,scrub+r

# Sweep over forget set sizes
python main.py --multirun forget.n_points=10,50,100,200,500

# Use ranges for more granular sweeps
python main.py --multirun "forget.n_points=range(10,100,10)"

# Combine multiple parameters
python main.py --multirun experiment.unlearn_type=ssd,amnesiac forget.n_points=10,50,100
```

### Using predefined sweep configurations

We've created predefined sweep configurations for common scenarios:

```bash
# Run the forget points sweep
python main.py --multirun --config-name=sweeps/n_forget

# Run the OOD ratio sweep
python main.py --multirun --config-name=sweeps/ood_ratio

# Run the outlier scale sweep
python main.py --multirun --config-name=sweeps/outlier_scale
```

## Configuration structure

The configuration is organized into the following sections: 

- `data`: Parameters for synthetic data generation
- `model`: Neural network model configuration
- `forget`: Forget set configuration
- `wandb`: Weights & Biases logging settings
- `system`: System settings (device, seed)
- `experiment`: Experiment parameters (unlearning method, repeats, etc.)

See `configs/config.yaml` for the complete configuration structure and default values.

## Creating custom sweep configurations

To create a custom sweep configuration, create a new YAML file in the `configs/sweeps` directory:

```yaml
# @package _global_

defaults:
  - /config
  - override hydra/sweeper: basic
  - _self_

hydra:
  sweeper:
    params:
      # Define parameters to sweep
      your.parameter: range(min,max,step)
      experiment.unlearn_type: method1,method2,method3

# Override other configuration values as needed
experiment:
  experiment_group: your_custom_sweep_name
  n_repeats: 5
```

Then run your custom sweep:

```bash
python main.py --multirun --config-name=sweeps/your_custom_sweep
``` 