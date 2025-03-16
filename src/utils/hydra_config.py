from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import uuid
from omegaconf import DictConfig, OmegaConf
import logging
import os

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
    
    @field_validator('train_ratio', 'val_ratio', 'test_ratio')
    @classmethod
    def validate_ratios(cls, v):
        if v < 0 or v > 1:
            raise ValueError(f"Ratio must be between 0 and 1, got {v}")
        return v
    
    @field_validator('test_ratio')
    @classmethod
    def validate_total_ratio(cls, v, info):
        values = info.data
        if 'train_ratio' in values and 'val_ratio' in values:
            total = values['train_ratio'] + values['val_ratio'] + v
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Sum of ratios must be 1.0, got {total}")
        return v

class ForgetConfig(BaseModel):
    n_points: int = Field(default=50, description="Number of points to include in forget set")
    class_idx: Optional[int] = Field(default=None, description="Specific class to draw forget points from")
    ood_ratio: float = Field(default=0.5, description="Ratio of out-of-distribution points in forget set")
    
    @field_validator('ood_ratio')
    @classmethod
    def validate_ood_ratio(cls, v):
        if v < 0 or v > 1:
            raise ValueError(f"OOD ratio must be between 0 and 1, got {v}")
        return v

class WandBConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable wandb logging")
    mode: str = Field(default="offline", description="Wandb mode (online/offline)")
    project: str = Field(default="unlearning-experiments", description="Wandb project name")
    dir: str = Field(default="./experiments/wandb", description="Directory for wandb files")
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        if v not in ["online", "offline", "disabled"]:
            raise ValueError(f"WandB mode must be one of 'online', 'offline', or 'disabled', got {v}")
        return v
    
    @model_validator(mode='after')
    def create_wandb_dir(self):
        if self.enabled:
            # Create WandB directory if it doesn't exist
            os.makedirs(self.dir, exist_ok=True)
        return self

class SystemConfig(BaseModel):
    device: str = Field(default="cpu", description="Device to use (cpu/cuda/mps)")
    seed: Optional[int] = Field(default=None, description="Global random seed")
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        if v not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Device must be one of 'cpu', 'cuda', or 'mps', got {v}")
        return v

class RayInitConfig(BaseModel):
    address: Optional[str] = Field(default=None, description="Ray cluster address")
    num_cpus: Optional[int] = Field(default=None, description="Number of CPUs to use")
    num_gpus: Optional[int] = Field(default=None, description="Number of GPUs to use")
    object_store_memory: Optional[int] = Field(default=None, description="Object store memory in bytes")
    include_dashboard: bool = Field(default=True, description="Include Ray dashboard")
    dashboard_host: str = Field(default="127.0.0.1", description="Dashboard host")
    dashboard_port: int = Field(default=8265, description="Dashboard port")
    
    @field_validator('num_cpus', 'num_gpus')
    @classmethod
    def validate_resources(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"Resource count cannot be negative, got {v}")
        return v

class RayRemoteConfig(BaseModel):
    max_retries: int = Field(default=3, description="Maximum number of retries")
    num_cpus: Optional[int] = Field(default=1, description="CPUs per task")
    num_gpus: Optional[int] = Field(default=0, description="GPUs per task")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables for Ray workers")

class RayConfig(BaseModel):
    init: RayInitConfig = Field(default_factory=RayInitConfig)
    remote: RayRemoteConfig = Field(default_factory=RayRemoteConfig)

class LauncherConfig(BaseModel):
    ray: Optional[RayConfig] = Field(default=None, description="Ray launcher configuration")

class ExperimentConfig(BaseModel):
    mode: str = Field(default="experiment", description="Mode of operation")
    unlearn_type: str = Field(default="ssd", description="Type of unlearning to use")
    n_repeats: int = Field(default=30, description="Number of experiment repeats")
    n_epochs: int = Field(default=50, description="Number of training epochs")
    n_forget_trials: int = Field(default=10, description="Number of forget set trials")
    experiment_id: Optional[str] = None
    experiment_group: str = Field(default="default_group", description="Group name for the experiment")
    track_performance: bool = Field(default=False, description="Enable performance tracking")
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        if v not in ["experiment", "visualize", "get_latex_results", "sync_wandb"]:
            raise ValueError(f"Mode must be one of 'experiment', 'visualize', or 'get_latex_results', got {v}")
        return v
    
    @field_validator('unlearn_type')
    @classmethod
    def validate_unlearn_type(cls, v):
        if v not in ["ssd", "scrub+r", "sisa", "amnesiac", "sae"]:
            raise ValueError(f"Unlearn type must be one of 'ssd', 'scrub+r', 'sisa', 'amnesiac', 'sae', got {v}")
        return v
    
    @model_validator(mode='before')
    @classmethod
    def set_experiment_id(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if data.get('experiment_id') is None:
            data['experiment_id'] = str(uuid.uuid4())[:8]
        return data

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
    launcher: Optional[LauncherConfig] = None
    
    model_config = {
        # Allow extra fields for flexibility with Hydra
        "extra": "ignore"
    }
    
    @model_validator(mode='after')
    def ensure_compatibility(self):
        # Ensure WandB is properly configured for parallel runs
        if self.launcher and self.launcher.ray and self.wandb.enabled:
            # Make sure WandB is configured to handle parallel runs
            # Each Ray worker needs a unique WandB run
            if 'WANDB_RUN_GROUP' not in os.environ:
                os.environ['WANDB_RUN_GROUP'] = self.experiment.experiment_group
        return self

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
    
    # Extract launcher config if present
    launcher_config = None
    if "hydra" in config_dict and "launcher" in config_dict["hydra"]:
        launcher_data = config_dict["hydra"].get("launcher", {})
        if "ray" in launcher_data:
            launcher_config = {"ray": launcher_data.get("ray", {})}
    
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
            experiment=ExperimentConfig(**config_dict.get("experiment", {})),
            launcher=LauncherConfig(**launcher_config) if launcher_config else None
        )
        
        # Set up environment for parallel runs if needed
        if config.launcher and config.launcher.ray:
            # Set environment variables for Ray
            if config.wandb.enabled:
                os.environ['WANDB_RUN_GROUP'] = config.experiment.experiment_group
        
        return config
    except Exception as e:
        logger.error(f"Config validation failed: {str(e)}")
        raise ValueError(f"Config validation failed: {str(e)}") 