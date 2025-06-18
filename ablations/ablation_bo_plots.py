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

from src.unlearners.selective_synaptic_dampening_v6 import SelectiveSynapticDampening
from src.models.neural_network import NeuralNet
from src.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.visualization.dampening_visualization import visualize_parameter_dampening
from src.visualization.bo_plotting import plot_bo_samples_basic, plot_convergence_analysis
from src.evaluation.decision_boundary import DecisionBoundaryCreator

results_dir = os.path.join(os.path.dirname(__file__), "results_ablation_ssd_bo")

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

"""
SMOOTH DATASET 3:
Iteration 25/100 completed in 0.08s, best value: -0.297381
Iteration 26/100 completed in 0.08s, best value: -0.156777
"""
@hydra.main(config_path=".", config_name="config")
def main(cfg):
    DEVICE = 'cpu' #('cuda' if torch.cuda.is_available() else 'cpu')
    # results_dir = os.path.join(os.path.dirname(__file__), "TA_results")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/convergence', exist_ok=True)

    # Table for storing nosmooth vs smooth max value iteration
    # across all datasets
    convergence_table = {
        "dataset": [],
        "nosmooth_max_alpha": [],
        "smooth_max_alpha": [],
        "nosmooth_max_lambda": [],
        "smooth_max_lambda": [],
        "nosmooth_max_iteration": [],
        "smooth_max_iteration": [],
    }

    for i in range(1,6):
        for smooth_dampen in [True, False]:
            print(f"Processing dataset {i}")
            # ============= Load data and prepare data =============
            dataset_path = f'data/data_{i}.npz'
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

            dataset_name = f"data_{i}"

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


            else:
                raise NotImplementedError(f"Logger {cfg.logging.logger} not implemented")
            
            # ============= Initialize model =============
            unlearned_model_1 = NeuralNet(M=X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)
            NeuralNetworkTrainer(model=unlearned_model_1, 
                                 train_dataloader=dataloader_train, 
                                 val_dataloader=dataloader_val, 
                                 logger=logger, 
                                 device=cfg.model.device, 
                                 learning_rate=cfg.trainer.lr, 
                                 n_epochs=cfg.trainer.n_epochs, 
                                 disable_tqdm=cfg.trainer.disable_tqdm, 
                                 do_early_stopping=cfg.trainer.do_early_stopping)()
            
            hyperparams, dampenings = SelectiveSynapticDampening(unlearned_model_1,
                                                        P=1,
                                                        k=0.999,
                                                        smooth_dampening=smooth_dampen,
                                                        n_bo_iter=100,
                                                        device=DEVICE)(full_dataloader=copy.deepcopy(dataloader_train), 
                                                                    forget_dataloader=copy.deepcopy(dataloader_forget),
                                                                    validation_dataloader=copy.deepcopy(dataloader_val),
                                                                    return_dampening=True)
            smooth_str = "smooth" if smooth_dampen else "nosmooth"
            filename = f"{dataset_name}_{smooth_str}_convergence.png"
            savepath = f"{results_dir}/convergence/{filename}"

            
            plot_convergence_analysis(hyperparams['result'], 
                                    savepath=savepath,
                                    exploration_factor=2.5)

if __name__ == "__main__":
    main()
