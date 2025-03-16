from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from pathlib import Path
import uuid
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)

class DataConfig(BaseModel):
    n_samples: int = Field(default=1000, description="Number of samples to generate")
    n_features: int = Field(default=25, description="Number of features")
    n_classes: int = Field(default=4, description="Number of classes")
    n_informative: int = Field(default=25, description="Number of informative features")
    n_redundant: int = Field(default=0, description="Number of redundant features")
    n_outliers: int = Field(default=50, description="Number of outlier points")
    outlier_scale: float = Field(default=4.0, description="Scale factor for outliers")
    outlier_variance: float = Field(default=0.4, description="Variance of outliers")
    outlier_class: Optional[int] = Field(default=None, description="Class index for outliers")
    random_state: Optional[int] = Field(default=None, description="Random seed")
    train_ratio: float = Field(default=0.8, description="Ratio of training data")
    val_ratio: float = Field(default=0.1, description="Ratio of validation data")
    test_ratio: float = Field(default=0.1, description="Ratio of test data")
    
    @validator('train_ratio', 'val_ratio', 'test_ratio')
    def validate_ratios(cls, v, values):
        if v < 0 or v > 1:
            raise ValueError(f"Ratio must be between 0 and 1, got {v}")
        return v
    
    @validator('test_ratio')
    def validate_total_ratio(cls, v, values):
        if 'train_ratio' in values and 'val_ratio' in values:
            total = values['train_ratio'] + values['val_ratio'] + v
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Sum of ratios must be 1.0, got {total}")
        return v

class ForgetConfig(BaseModel):
    n_points: int = Field(default=50, description="Number of points to include in forget set")
    class_idx: Optional[int] = Field(default=None, description="Specific class to draw forget points from")
    ood_ratio: float = Field(default=0.5, description="Ratio of out-of-distribution points in forget set")
    
    @validator('ood_ratio')
    def validate_ood_ratio(cls, v):
        if v < 0 or v > 1:
            raise ValueError(f"OOD ratio must be between 0 and 1, got {v}")
        return v

class WandBConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable wandb logging")
    mode: str = Field(default="offline", description="Wandb mode (online/offline)")
    project: str = Field(default="unlearning-experiments", description="Wandb project name")
    dir: str = Field(default="./experiments/wandb", description="Directory for wandb files")

class SystemConfig(BaseModel):
    device: str = Field(default="cpu", description="Device to use (cpu/cuda/mps)")
    seed: Optional[int] = Field(default=None, description="Global random seed")
    
    @validator('device')
    def validate_device(cls, v):
        if v not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Device must be one of 'cpu', 'cuda', or 'mps', got {v}")
        return v

class ExperimentConfig(BaseModel):
    mode: str = Field(default="experiment", description="Mode of operation")
    unlearn_type: str = Field(default="ssd", description="Type of unlearning to use")
    n_repeats: int = Field(default=30, description="Number of experiment repeats")
    n_epochs: int = Field(default=50, description="Number of training epochs")
    n_forget_trials: int = Field(default=10, description="Number of forget set trials")
    experiment_id: Optional[str] = None
    experiment_group: str = Field(default="default_group", description="Group name for the experiment")
    track_performance: bool = Field(default=False, description="Enable performance tracking")
    
    @validator('mode')
    def validate_mode(cls, v):
        if v not in ["experiment", "visualize", "get_latex_results"]:
            raise ValueError(f"Mode must be one of 'experiment', 'visualize', or 'get_latex_results', got {v}")
        return v
    
    @validator('unlearn_type')
    def validate_unlearn_type(cls, v):
        if v not in ["ssd", "scrub+r", "sisa", "amnesiac", "sae"]:
            raise ValueError(f"Unlearn type must be one of 'ssd', 'scrub+r', 'sisa', 'amnesiac', 'sae', got {v}")
        return v
    
    @validator('experiment_id')
    def set_experiment_id(cls, v):
        if v is None:
            return str(uuid.uuid4())[:8]
        return v

class ModelConfig(BaseModel):
    n_features: int = Field(default=25, description="Number of input features")
    n_classes: int = Field(default=4, description="Number of output classes")
    n_epochs: int = Field(default=20, description="Number of training epochs")

class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    forget: ForgetConfig
    wandb: WandBConfig
    system: SystemConfig
    experiment: ExperimentConfig

def validate_config(cfg: DictConfig) -> Config:
    """
    Validate the Hydra configuration using Pydantic models.
    
    Args:
        cfg: Hydra DictConfig object
        
    Returns:
        Config: Validated configuration object
    """
    # Convert OmegaConf to dict
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Remove Hydra-specific keys
    if "hydra" in config_dict:
        del config_dict["hydra"]
    
    try:
        # Create and validate config
        config = Config(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            forget=ForgetConfig(**config_dict.get("forget", {})),
            wandb=WandBConfig(**config_dict.get("wandb", {})),
            system=SystemConfig(**config_dict.get("system", {})),
            experiment=ExperimentConfig(**config_dict.get("experiment", {}))
        )
        
        # Ensure experiment_id is set
        if config.experiment.experiment_id is None:
            config.experiment.experiment_id = str(uuid.uuid4())[:8]
            
        return config
    except Exception as e:
        logger.error(f"Config validation failed: {str(e)}")
        raise ValueError(f"Config validation failed: {str(e)}") 