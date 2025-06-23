import hydra
import os
import numpy as np
import copy
import matplotlib.pyplot as plt

from src.models.neural_network import NeuralNet
from src.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.utils.get_git_root_path import get_git_root
from src.utils.load_rogue_data import get_data

# Correction of save path..

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    DEVICE = 'cpu' #('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = os.path.join(os.path.dirname(__file__), "results", "ablation_bo_objective")
    os.makedirs(results_dir, exist_ok=True)

    N_points = 10000
    # set appropriate pbounds for each dataset (where objective is not always constant)
    dataset_number2pbounds = {1: (0.01, 160),
                              2: (0.01, 14),
                              3: (0.01, 200),
                              4: (0.01, 20),
                              5: (0.01, 120)}


    smooth_dampen = True if "ssd-bo-pairwise-smooth" == cfg.unlearn.method else False

    # ============= Preparing data and results directory =============
    git_root = get_git_root()
    dataset_path = cfg.data.dataset
    print('-'*100)
    print(f'Running experiment for {dataset_path}')
    print('-'*100)
    # path relative to current working directory
    dataset_path = os.path.join(git_root, dataset_path)
    validation_path = os.path.join("/",*dataset_path.split("/")[1:-1], "validation_data.npz")

    rogue = dataset_path.split("/")[-2] # rogue_many or rogue_one

    # ============== Load data ============== 
    dataloader_train, dataloader_val, dataloader_retain, dataloader_forget, forget_idxs = get_data(dataset_path, validation_path, cfg.data.batch_size, cfg.data.n_classes)
    
    dataset_name = cfg.data.dataset.split("/")[-1].split(".")[0]
    dataset_number = dataset_name.split("_")[1]

    pbounds = dataset_number2pbounds[int(dataset_number)]

    # No logger, this could be changed to a wandb logger if needed
    logger = None 
    alpha_values = np.linspace(start=pbounds[0], stop=pbounds[1], num=N_points)

    # ============= Initialize model =============
    model = copy.deepcopy(NeuralNet(M=dataloader_train.dataset.X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed))
    trainer = NeuralNetworkTrainer(model=model, 
                                train_dataloader=dataloader_train, 
                                val_dataloader=dataloader_val,
                                logger=logger, 
                                device=cfg.model.device, 
                                learning_rate=cfg.trainer.lr, 
                                n_epochs=cfg.trainer.n_epochs, 
                                disable_tqdm=cfg.trainer.disable_tqdm, 
                                do_early_stopping=cfg.trainer.do_early_stopping)()
    

    from src.unlearners.selective_synaptic_dampening_BO_visualizer import SSDVisualizer
    ssd_visualizer = SSDVisualizer(model, 
                                P=1, 
                                smooth_dampening=smooth_dampen, 
                                device=DEVICE)

    objective_values = ssd_visualizer.search_hyperparameters_exhaustive(dataloader_train, dataloader_forget, dataloader_val, alpha_values)

    plt.style.use('seaborn-v0_8-paper')

    # Color palette from SCRUB plots
    colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']

    # Create the figure with a consistent style
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(r'$\mathcal{L}_{BO}$ as a function of $\alpha$, with $\lambda=1$', fontsize=34)


    # Create the main plot
    ax.plot(alpha_values, objective_values, 
            color=colors[6], # Using a color from the SCRUB palette
            linewidth=2.5, 
            alpha=0.9,
            label='Objective value')

    # Customize the plot appearance
    ax.set_xlabel(r'$\alpha$', fontsize=28)
    ax.set_ylabel(r'$\mathcal{L}_{BO}$', fontsize=28)
    
    # Add legend with custom styling
    ax.legend(fontsize=24, frameon=True, fancybox=True, shadow=False, loc='best')

    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Add a grid
    ax.grid(True, linestyle=':', alpha=0.6)

    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot with high quality
    filename = f"{dataset_number}_LBO_vs_alpha_lambda1" if not smooth_dampen else f"{dataset_number}_LBO_vs_alpha_lambda1_smooth"
    plt.savefig(f'{results_dir}/{filename}.pdf', bbox_inches='tight') 


if __name__ == '__main__':
    main()


