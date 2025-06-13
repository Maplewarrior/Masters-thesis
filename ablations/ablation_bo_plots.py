import hydra
import wandb
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from omegaconf import OmegaConf
from src.datasets.synthetic_dataset import SyntheticDataset
import copy
import matplotlib.pyplot as plt

from src.models.neural_network import NeuralNet
from src.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.visualization.dampening_visualization import visualize_parameter_dampening
from src.visualization.bo_plotting import plot_bo_samples_basic, plot_convergence_analysis

results_dir = os.path.join(os.path.dirname(__file__), "results_ablation_bo_plots")

def create_structured_savepath(base_dir, dataset_name, experiment_type, plot_type, 
                             P=None, exploration_factor=None, n_bo_iter=None, smooth_dampening=None):
    """
    Create a structured save path for organized file storage.
    
    Args:
        base_dir: Base directory for results
        dataset_name: Name of the dataset
        method: Unlearning method name
        experiment_type: Type of experiment (e.g., 'bayesian_optimization')
        plot_type: Type of plot (e.g., 'samples_basic', 'convergence')
        P: P parameter value
        exploration_factor: exploration factor for the UCB acquisition function
        n_bo_iter: Number of BO iterations
        smooth_dampening: Whether smooth dampening is enabled
    
    Returns:
        Complete structured file path
    """
    # Create subdirectory structure
    subdir = os.path.join(base_dir, dataset_name, experiment_type)
    os.makedirs(subdir, exist_ok=True)
    
    # Build descriptive filename
    filename_parts = [dataset_name, plot_type]
    
    # Add parameter information
    if P is not None:
        filename_parts.append(f"P{P}")
    if exploration_factor is not None:
        filename_parts.append(f"exploration_factor{exploration_factor}")
    if n_bo_iter is not None:
        filename_parts.append(f"iter{n_bo_iter}")
    if smooth_dampening is not None:
        filename_parts.append("smooth" if smooth_dampening else "nosmooth")
    
    filename = "_".join(filename_parts) + ".png"
    
    return os.path.join(subdir, filename)

def load_dataset(file):
    npz_file = np.load(file, allow_pickle=True)
    X = npz_file["X"]
    y = npz_file["y"]

    if "rogue_point_idx" in npz_file:
        try: # ? Not the nicest way to do this
            forget_idxs = torch.from_numpy(npz_file["rogue_point_idx"])
        except:
            forget_idxs = npz_file["rogue_point_idx"] 
    else:
        forget_idxs = None

    return torch.from_numpy(X), torch.from_numpy(y), forget_idxs


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    DEVICE = 'cpu' #('cuda' if torch.cuda.is_available() else 'cpu')
    # results_dir = os.path.join(os.path.dirname(__file__), "TA_results")
    os.makedirs(results_dir, exist_ok=True)

    # ============= Load data and prepare data =============
    dataset_path = cfg.data.dataset
    # path relative to current working directory
    dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)
    validation_path = os.path.join(os.path.dirname(__file__), "data/validation_data.npz")

    X, y, forget_idxs = load_dataset(dataset_path)
    # retrain data should be all the data except the index of the rogue point
    X_val, y_val, _ = load_dataset(validation_path)

    # Convert forget_idx to a set for faster lookup
    forget_idx_set = set(forget_idxs.tolist())

    # Use boolean indexing to filter out the indices
    mask_retrain = ~torch.tensor([int(i) in forget_idx_set for i in torch.arange(X.shape[0])])
    mask_forget = torch.tensor([int(i) in forget_idx_set for i in torch.arange(X.shape[0])])

    X_retrain, y_retrain = X[mask_retrain], y[mask_retrain]
    X_forget, y_forget = X[mask_forget], y[mask_forget]
    
    # X, y, X_val, y_val, X_retrain, y_retrain, X_forget, y_forget
    dataset_train = SyntheticDataset(X, y, dataset_name="train", n_classes=cfg.data.n_classes)
    dataset_val = SyntheticDataset(X_val, y_val, dataset_name="validation", n_classes=cfg.data.n_classes)
    dataset_retain = SyntheticDataset(X_retrain, y_retrain, dataset_name="retrain", n_classes=cfg.data.n_classes)
    dataset_forget = SyntheticDataset(X_forget, y_forget, dataset_name="forget", n_classes=cfg.data.n_classes)

    batch_size = cfg.data.batch_size
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    dataloader_retain = DataLoader(dataset_retain, batch_size=batch_size, shuffle=True)
    dataloader_forget = DataLoader(dataset_forget, batch_size=batch_size, shuffle=True)

    # ============= Initialize logger =============
    if cfg.logging.logger == "wandb":
        cfg_dict = OmegaConf.to_container(cfg, resolve=True) # ? Convert to dict to avoid issues with wandb

        os.environ["WANDB_MODE"] = cfg.logging.wandb.mode
        wandb.setup(settings=wandb.Settings(mode=cfg.logging.wandb.mode))
        logger = wandb.init(project=cfg.logging.project, 
                            config=cfg_dict, 
                            name=cfg.logging.name, 
                            group=cfg.logging.group,
                            mode=cfg.logging.wandb.mode,
                            dir=cfg.logging.dir)


        dataset_name = cfg.data.dataset.split("/")[-1].split(".")[0]

    else:
        raise NotImplementedError(f"Logger {cfg.logging.logger} not implemented")
    

    unlearned_model = NeuralNet(M=X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)
    NeuralNetworkTrainer(model=unlearned_model, 
                                train_dataloader=dataloader_train, 
                                val_dataloader=dataloader_val, 
                                logger=logger, 
                                device=cfg.model.device, 
                                learning_rate=cfg.trainer.lr, 
                                n_epochs=cfg.trainer.n_epochs, 
                                disable_tqdm=cfg.trainer.disable_tqdm, 
                                do_early_stopping=cfg.trainer.do_early_stopping)()
            
    from src.unlearners.selective_synaptic_dampening_v6 import SelectiveSynapticDampening
    hyperparams, dampenings = SelectiveSynapticDampening(unlearned_model,
                                                P=2,
                                                k=0.999,
                                                smooth_dampening=False,
                                                n_bo_iter=100,
                                                device=DEVICE)(full_dataloader=dataloader_train, 
                                                            forget_dataloader=dataloader_forget,
                                                            validation_dataloader=dataloader_val,
                                                            return_dampening=True)
    
    plot_bo_samples_basic(hyperparams['result'], savepath=create_structured_savepath(results_dir, dataset_name, 'nosmooth', "samples_basic", P=2, exploration_factor=2.5, n_bo_iter=100, smooth_dampening=False))
    plot_convergence_analysis(hyperparams['result'], savepath=create_structured_savepath(results_dir, dataset_name, 'nosmooth', "convergence", P=2, exploration_factor=2.5, n_bo_iter=100, smooth_dampening=False))

    
    from src.unlearners.selective_synaptic_dampening_v6 import SelectiveSynapticDampening
    hyperparams, dampenings = SelectiveSynapticDampening(unlearned_model,
                                                P=2,
                                                k=0.999,
                                                smooth_dampening=True,
                                                n_bo_iter=100,
                                                exploration_factor=2.5,
                                                device=DEVICE)(full_dataloader=dataloader_train, 
                                                            forget_dataloader=dataloader_forget,
                                                            validation_dataloader=dataloader_val,
                                                            return_dampening=True)
    
    plot_bo_samples_basic(hyperparams['result'], savepath=create_structured_savepath(results_dir, dataset_name, 'smooth', "samples_basic", P=2, exploration_factor=2.5, n_bo_iter=100, smooth_dampening=True))
    plot_convergence_analysis(hyperparams['result'], savepath=create_structured_savepath(results_dir, dataset_name, 'smooth', "convergence", P=2, exploration_factor=2.5, n_bo_iter=100, smooth_dampening=True))
            

if __name__ == "__main__":
    main()
