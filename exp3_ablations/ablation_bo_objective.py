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
    

    from unlearners.selective_synaptic_dampening_BO_visualizer import SSDVisualizer
    ssd_visualizer = SSDVisualizer(model, 
                                P=1, 
                                smooth_dampening=smooth_dampen, 
                                device=DEVICE)

    objective_values = ssd_visualizer.search_hyperparameters_exhaustive(dataloader_train, dataloader_forget, dataloader_val, alpha_values)

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

    # Customize the plot appearance
    plt.xlabel(r'$\alpha$', fontsize=14, fontweight='bold', color='#333333')
    plt.ylabel(r'$\mathcal{L}_{BO}$', fontsize=14, fontweight='bold', color='#333333')
    plt.title(r'$\mathcal{L}_{BO}$ as a function of $\alpha$. $\lambda=1$', fontsize=18, fontweight='bold', 
            color='#2c3e50', pad=20)

    # Add legend with custom styling
    plt.legend(fontsize=14, frameon=True, fancybox=True, shadow=False, 
            framealpha=0.9, edgecolor='#cccccc', loc='best')

    # Customize tick parameters
    plt.tick_params(axis='both', which='major', labelsize=12, colors='#333333')
    plt.tick_params(axis='both', which='minor', labelsize=10, colors='#666666')

    plt.tight_layout(pad=0.0)

    # Optional: Add a subtle border around the plot area
    for spine in plt.gca().spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(1.2)


    # Optional: Save the plot with high quality
    filename = f"{dataset_number}_LBO_vs_alpha_lambda1" if not smooth_dampen else f"{dataset_number}_objective_vs_alpha_lambda1_smooth"
    plt.savefig(f'{results_dir}/{filename}.pdf') 


if __name__ == '__main__':
    main()


