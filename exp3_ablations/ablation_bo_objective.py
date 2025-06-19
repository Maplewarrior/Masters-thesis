import hydra
import wandb
import os
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.datasets.synthetic_dataset import SyntheticDataset
from src.models.neural_network import NeuralNet
from src.trainers.neural_network_trainer import NeuralNetworkTrainer
import pdb

# Correction of save path..
results_dir = os.path.join(os.path.dirname(__file__), "results_ablation_bo_objective")

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

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    DEVICE = 'cpu' #('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/objective_shape", exist_ok=True)

    N_points = 10000
    # set appropriate pbounds for each dataset (where objective is not always constant)
    dataset_number2pbounds = {1: (0.01, 160),
                              2: (0.01, 14),
                              3: (0.01, 200),
                              4: (0.01, 20),
                              5: (0.01, 120)}

    # ============= Load data and prepare data =============
    for i in range(1, 6):
        for smooth_dampen in [False]:#, True]:
            pbounds = dataset_number2pbounds[i]
            print('-'*100)
            print(f'Running experiment for data/data_{i}.npz')
            print('-'*100)
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

            alpha_values = np.linspace(start=pbounds[0], stop=pbounds[1], num=N_points)

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
            model = copy.deepcopy(NeuralNet(M=X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed))
            trainer = NeuralNetworkTrainer(model=model, 
                                        train_dataloader=dataloader_train, 
                                        val_dataloader=dataloader_val,
                                        logger=logger, 
                                        device=cfg.model.device, 
                                        learning_rate=cfg.trainer.lr, 
                                        n_epochs=cfg.trainer.n_epochs, 
                                        disable_tqdm=cfg.trainer.disable_tqdm, 
                                        do_early_stopping=cfg.trainer.do_early_stopping)()
            

            from src.unlearners.ssd_v6_bo_visualizer import SSDVisualizer
            ssd_visualizer = SSDVisualizer(model, 
                                        P=1, 
                                        smooth_dampening=smooth_dampen, 
                                        device=DEVICE)
            # from src.unlearners.selective_synaptic_dampening_v3 import SelectiveSynapticDampening
            # hyperparams, dampenings = SelectiveSynapticDampening(model, 
            #                                                      device=DEVICE)(full_dataloader=dataloader_train, 
            #                                                                 forget_dataloader=dataloader_forget,
            #                                                                 validation_dataloader=dataloader_val,
            #                                                                 return_dampening=True)
            # pdb.set_trace()
            objective_values = ssd_visualizer.search_hyperparameters_exhaustive(dataloader_train, dataloader_forget, dataloader_val, alpha_values)
            # pdb.set_trace()

            plt.style.use('seaborn-v0_8-whitegrid')

            # Your specified color palette
            colors = ['#ffa600', '#a05195', '#f95d6a', '#8172B3', '#CCB974', '#64B5CD', '#FC9E4F']

            # Create the figure with a nice size and DPI
            plt.figure(figsize=(10, 7), dpi=100)

            # Create the main plot with enhanced styling
            plt.plot(alpha_values, objective_values, 
                    color=colors[0], 
                    linewidth=3, 
                    # markerfacecolor=colors[1], 
                    # markeredgecolor=colors[2], 
                    # markeredgewidth=2,
                    alpha=0.9,
                    label='Objective value')
            
            # Add a subtle grid
            # plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

            # Customize the plot appearance
            plt.xlabel(r'$\alpha$', fontsize=14, fontweight='bold', color='#333333')
            plt.ylabel(r'$\mathcal{L}_{BO}$', fontsize=14, fontweight='bold', color='#333333')
            plt.title(r'Objective value as a function of $\alpha$. Here $\lambda=1$ is fixed.', fontsize=18, fontweight='bold', 
                    color='#2c3e50', pad=20)

            # Add legend with custom styling
            plt.legend(fontsize=14, frameon=True, fancybox=True, shadow=False, 
                    framealpha=0.9, edgecolor='#cccccc', loc='best')

            # Customize tick parameters
            plt.tick_params(axis='both', which='major', labelsize=12, colors='#333333')
            plt.tick_params(axis='both', which='minor', labelsize=10, colors='#666666')

            # Set background color
            # plt.gca().set_facecolor('#fafafa')

            # Add some padding around the plot
            plt.tight_layout(pad=0.0)

            # Optional: Add a subtle border around the plot area
            for spine in plt.gca().spines.values():
                spine.set_color('#cccccc')
                spine.set_linewidth(1.2)

            # Display the plot

            # Optional: Save the plot with high quality
            filename = f"{i}_objective_vs_alpha" if not smooth_dampen else f"{i}_objective_vs_alpha_smooth"
            plt.savefig(f'{results_dir}/objective_shape/{filename}.png', dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            

        
        


if __name__ == '__main__':
    main()


