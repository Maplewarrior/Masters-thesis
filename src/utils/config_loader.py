from pydantic import BaseModel, Field
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class DataConfig(BaseModel):
    n_samples: int = Field(default=1000, description="Number of samples to generate")
    n_features: int = Field(default=25, description="Number of features")
    n_classes: int = Field(default=4, description="Number of classes")
    n_informative: int = Field(default=25, description="Number of informative features")
    n_redundant: int = Field(default=0, description="Number of redundant features")
    n_outliers: int = Field(default=50, description="Number of outlier points")
    outlier_scale: float = Field(default=4.0, description="Scale factor for outliers")
    outlier_variance: float = Field(default=0.4, description="Variance of outliers")
    outlier_class: int = Field(default=1, description="Class index for outliers")
    random_state: int = Field(default=None, description="Random seed")
    train_ratio: float = Field(default=0.8, description="Ratio of training data")
    val_ratio: float = Field(default=0.1, description="Ratio of validation data")
    test_ratio: float = Field(default=0.1, description="Ratio of test data")

class ForgetConfig(BaseModel):
    n_points: int = Field(default=50, description="Number of points to include in forget set")
    class_idx: Optional[int] = Field(default=None, description="Specific class to draw forget points from")
    ood_ratio: float = Field(default=0.5, description="Ratio of out-of-distribution points in forget set")

class WandBConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable wandb logging")
    mode: str = Field(default="offline", description="Wandb mode: online/offline/disabled")
    project: str = Field(default="unlearning-experiments", description="WandB project name")
    dir: str = Field(default="./experiments/wandb", description="Local directory for wandb files")

class SystemConfig(BaseModel):
    device: str = Field(default="cpu", description="Device to use (cpu/cuda/mps)")
    seed: Optional[int] = Field(default=None, description="Global random seed")

class ExperimentConfig(BaseModel):
    mode: str = Field(default="experiment", description="Operation mode (experiment/visualize/get_latex_results)")
    unlearn_type: str = Field(default="ssd", description="Unlearning method to use")
    n_repeats: int = Field(default=5, description="Number of experiment repeats")
    n_epochs: int = Field(default=20, description="Number of training epochs")
    n_forget_trials: int = Field(default=2, description="Number of different forget sets to try")

class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    forget: ForgetConfig = Field(default_factory=ForgetConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

def load_config(config_path: str = "configs/config.yaml", args_dict: Optional[Dict] = None) -> Config:
    """
    Load and validate configuration from a yaml file and merge with command line arguments.
    
    Args:
        config_path: Path to the yaml config file
        args_dict: Optional dictionary of command line arguments to override config values
        
    Returns:
        Config: Validated configuration object
        
    Raises:
        ValueError: If config validation fails
    """
    # Load config file if it exists
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
    else:
        raw_config = {}

    try:
        # Create base config
        config = Config(**raw_config)
        
        # Update with command line arguments if provided
        if args_dict is not None:
            config_dict = config.dict()
            
            # Map command line args to config structure
            arg_mapping = {
                # Data arguments
                'n_samples': ('data', 'n_samples'),
                'n_features': ('data', 'n_features'),
                'n_classes': ('data', 'n_classes'),
                'n_informative': ('data', 'n_informative'),
                'n_redundant': ('data', 'n_redundant'),
                'n_outliers': ('data', 'n_outliers'),
                'outlier_scale': ('data', 'outlier_scale'),
                'outlier_variance': ('data', 'outlier_variance'),
                'outlier_class': ('data', 'outlier_class'),
                'random_state': ('data', 'random_state'),
                'train_ratio': ('data', 'train_ratio'),
                'val_ratio': ('data', 'val_ratio'),
                'test_ratio': ('data', 'test_ratio'),
                
                # System arguments
                'device': ('system', 'device'),
                
                # Experiment arguments
                'mode': ('experiment', 'mode'),
                'unlearn_type': ('experiment', 'unlearn_type'),
                'n_repeats': ('experiment', 'n_repeats'),
                'n_epochs': ('experiment', 'n_epochs'),
                'n_forget_trials': ('experiment', 'n_forget_trials'),
                
                # Forget arguments
                'forget_n_points': ('forget', 'n_points'),
                'forget_class_idx': ('forget', 'class_idx'),
                'forget_ood_ratio': ('forget', 'ood_ratio'),
                
                # WandB arguments
                'wandb': ('wandb', 'enabled')
            }
            
            # Update config with command line arguments
            for arg_name, value in args_dict.items():
                if arg_name in arg_mapping and value is not None:
                    section, key = arg_mapping[arg_name]
                    config_dict[section][key] = value
            
            # Recreate config with updated values
            config = Config(**config_dict)
        
        return config
        
    except Exception as e:
        raise ValueError(f"Config validation failed: {str(e)}")
