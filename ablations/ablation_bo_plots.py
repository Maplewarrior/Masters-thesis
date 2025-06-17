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

results_dir = os.path.join(os.path.dirname(__file__), "results_ablation_bo_plots")

def decision_boundary_plot(model, original_model, dataloader_retrain, dataloader_train, 
                           dataset_name, X_forget, cfg, hyperparams,
                           make_pdfs=False, 
                           compress_pdfs=False,
                           savepath=None,
                           title=None,
                           ):
    dataset_number = dataset_name.split("_")[1]
    if title is None:
        title = f"{cfg.unlearn.method}"
    if savepath is None:
        savepath = f"{results_dir}/{dataset_number}_decision_boundary_{cfg.unlearn.method}"

    
    # Set common plot styling
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Get the classes of the rogue points
    X, y = dataloader_train.dataset.X, dataloader_train.dataset.y
    X_retain, y_retain = dataloader_retrain.dataset.X, dataloader_retrain.dataset.y
    
    # If y is onehot, convert it to class indices
    if len(y.shape) == 2 and y.shape[1] > 1:
        y = torch.argmax(y, dim=1)
        y_retain = torch.argmax(y_retain, dim=1)
    
    # Professional color palette
    professional_colors = ['#ffa600', '#a05195', '#f95d6a', '#8172B3', '#CCB974', '#64B5CD']
    
    # Find the classes of the rogue points
    rogue_classes = []
    rogue_colors = []
    for i in range(X_forget.shape[0]):
        distances = np.sum((X.numpy() - X_forget[i].numpy())**2, axis=1)
        rogue_class = y[np.argmin(distances)].item()
        rogue_classes.append(rogue_class)
        rogue_colors.append(professional_colors[rogue_class % len(professional_colors)])
    
    # List to store PDF files for later compression
    pdf_files = []
    
    # Create creators for both models
    unlearned_creator = DecisionBoundaryCreator(model, dataloader_retrain)
    original_creator = DecisionBoundaryCreator(original_model, dataloader_train)
    
    # Get the decision boundaries but don't plot them yet
    xx_o, yy_o, decision_boundary_original = original_creator.create_decision_boundary((-9, 9), (-9, 9))
    xx_u, yy_u, decision_boundary_unlearned = unlearned_creator.create_decision_boundary((-9, 9), (-9, 9))
    
    # Create a single figure
    plt.figure(figsize=(8, 7), dpi=200)
    
    # Get number of unique classes
    classes = np.unique(y)
    num_classes = len(classes)
    
    # Professional color palette for classes
    colors = [professional_colors[i % len(professional_colors)] for i in range(len(classes))]
    colors = [plt.matplotlib.colors.to_rgba(color) for color in colors]
    custom_cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Plot the unlearned decision boundary with lower opacity
    plt.pcolormesh(xx_u.numpy(), yy_u.numpy(), decision_boundary_unlearned.numpy(),
                 alpha=0.4, cmap=custom_cmap)
    

    unlearned_creator.plot_retain_data(X_retain, y_retain, classes, colors)


    for class_idx in range(num_classes):
        for neighbor_class in range(class_idx + 1, num_classes):
            # Original model boundaries - dashed lines
            original_mask = np.logical_or(
                decision_boundary_original.numpy() == class_idx,
                decision_boundary_original.numpy() == neighbor_class
            )
            # --- Before Unlearning ---
            contour_o = plt.contour(
                xx_o.numpy(), yy_o.numpy(), original_mask,
                colors=['#353535'],  # Jet
                linestyles='dotted',
                linewidths=2,
            )
            
            # Unlearned model boundaries
            unlearned_mask = np.logical_or(
                decision_boundary_unlearned.numpy() == class_idx,
                decision_boundary_unlearned.numpy() == neighbor_class
            )
            # --- After Unlearning ---
            contour_u = plt.contour(
                xx_u.numpy(), yy_u.numpy(), unlearned_mask,
                colors=['#FC9E4F'],  # Carribean Current
                linestyles='solid',
                linewidths=2,
            )
            
    # Get the axis to enhance
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Store the handles for legend
    legend_handles = []
    
    # Add custom legend entries for the two boundaries
    from matplotlib.lines import Line2D    
    original_legend = Line2D([0], [0], color='#353535', lw=2, linestyle='dotted', label='Before Unlearning')
    unlearned_legend = Line2D([0], [0], color='#FC9E4F', lw=2, linestyle='solid', label='After Unlearning')
    legend_handles.extend([original_legend, unlearned_legend])
    
    # Add rogue points with two different styles
    for i, (x_point, rogue_class, rogue_color) in enumerate(zip(X_forget, rogue_classes, rogue_colors)):
        rogue_point = plt.scatter(
            x_point[0], x_point[1], 
            color=rogue_color, 
            marker="X", 
            s=100,
            linewidth=1,
            edgecolor='white',
            zorder=10,  # Higher zorder to ensure visibility
            label=f"Forget observations" if i == 0 else "_nolegend_"
        )
        if i == 0:
            legend_handles.append(rogue_point)
    
    # Improve title and labels
    plt.title(f"Decision Boundary\nBefore and After Unlearning for {cfg.unlearn.method} ({title})", 
              fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    
    # Add legend with box and all custom entries
    legend = plt.legend(
        handles=legend_handles,
        frameon=True,
        framealpha=0.95,
        facecolor='white',
        edgecolor='lightgray',
        loc='best',
        fontsize=12
    )
    
    # Improve ticks
    ax.tick_params(direction='out', length=6, width=1)
    
    # Add a subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
    
    # Add grid with light alpha
    plt.grid(True, alpha=0.3)
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save with high DPI as PNG
    plt.savefig(savepath, bbox_inches='tight')
    
    # Save as PDF if enabled
    if make_pdfs:
        pdf_file = f"{results_dir}/{dataset_number}_decision_boundary_{cfg.unlearn.method}_superimposed.pdf"
        
        plt.savefig(pdf_file, 
                   bbox_inches='tight', 
                   format='pdf',
                   dpi=150)  # Reduced DPI for PDF
        
        # Add to list for later compression if enabled
        pdf_files.append(pdf_file)
    
    plt.close()
    
    # Further compress PDFs with Ghostscript if enabled
    if compress_pdfs and pdf_files:
        try:
            from src.utils.pdf_compression import compress_pdf_with_ghostscript
            
            print("Further compressing PDFs with Ghostscript...")
            for pdf_file in pdf_files:
                compress_pdf_with_ghostscript(pdf_file, quality='ebook')
        except ImportError:
            print("PDF compression module not found. PDFs saved with basic compression only.")
        except Exception as e:
            print(f"Error during PDF compression: {e}")
            print("PDFs saved with basic compression only.")


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
    filename_parts = [plot_type]
    
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
        
        original_model = copy.deepcopy(unlearned_model_1)
        unlearned_model_2 = copy.deepcopy(unlearned_model_1)

        # ============= Run NO SMOOTH BO =============
        hyperparams, dampenings = SelectiveSynapticDampening(unlearned_model_1,
                                                    P=1,
                                                    k=0.999,
                                                    smooth_dampening=False,
                                                    n_bo_iter=100,
                                                    device=DEVICE)(full_dataloader=dataloader_train, 
                                                                forget_dataloader=dataloader_forget,
                                                                validation_dataloader=dataloader_val,
                                                                return_dampening=True)
        savepath = create_structured_savepath(results_dir, 
                                              dataset_name, 'nosmooth', "decision_boundary", P=1, exploration_factor=2.5, n_bo_iter=100, smooth_dampening=False)
        total_parameters = sum(dampening.numel() for dampening in dampenings.values())
        total_dampening = total_parameters - sum(torch.sum(torch.abs(dampening)).item() for dampening in dampenings.values())
        title = f"{cfg.unlearn.method}\n$\\alpha$={hyperparams['max']['alpha_0']:.2f}, $\\lambda$={hyperparams['max']['_lambda_0']:.2f}\nTotal dampening: {total_dampening:.2f}"
        decision_boundary_plot(unlearned_model_1, 
                               original_model, 
                               dataloader_retain, 
                               dataloader_train, 
                               dataset_name, 
                               X_forget, 
                               cfg, 
                               hyperparams['max'], 
                               savepath=savepath,
                               title=title)
            
        fig = visualize_parameter_dampening(dampenings)
        plt.savefig(create_structured_savepath(results_dir, dataset_name, 'nosmooth', "dampening", P=1, exploration_factor=2.5, n_bo_iter=100, smooth_dampening=False))
        plt.close()

        plot_bo_samples_basic(hyperparams['result'], 
                              savepath=create_structured_savepath(results_dir, dataset_name, 'nosmooth', "samples_basic", P=1, exploration_factor=2.5, n_bo_iter=100, smooth_dampening=False),
                              exploration_factor=2.5)
        plot_convergence_analysis(hyperparams['result'], 
                                  savepath=create_structured_savepath(results_dir, dataset_name, 'nosmooth', "convergence", P=1, exploration_factor=2.5, n_bo_iter=100, smooth_dampening=False),
                                  exploration_factor=2.5)

        # ============= Run SMOOTH BO =============
        hyperparams_2, dampenings = SelectiveSynapticDampening(unlearned_model_2,
                                                    P=1,
                                                    k=0.999,
                                                    smooth_dampening=True,
                                                    n_bo_iter=100,
                                                    exploration_factor=2.5,
                                                    device=DEVICE)(full_dataloader=dataloader_train, 
                                                                forget_dataloader=dataloader_forget,
                                                                validation_dataloader=dataloader_val,
                                                                return_dampening=True)
        total_parameters = sum(dampening.numel() for dampening in dampenings.values())
        total_dampening = total_parameters - sum(torch.sum(torch.abs(dampening)).item() for dampening in dampenings.values())
        savepath = create_structured_savepath(results_dir, 
                                              dataset_name, 'smooth', "decision_boundary", P=1, exploration_factor=2.5, n_bo_iter=100, smooth_dampening=True)
        title = f"{cfg.unlearn.method}\n$\\alpha$={hyperparams_2['max']['alpha_0']:.2f}, $\\lambda$={hyperparams_2['max']['_lambda_0']:.2f}\nTotal dampening: {total_dampening:.2f}"
        decision_boundary_plot(unlearned_model_2, 
                               original_model, 
                               dataloader_retain, 
                               dataloader_train, 
                               dataset_name, 
                               X_forget, 
                               cfg, 
                               hyperparams_2['max'], 
                               savepath=savepath, 
                               title=title)

        fig = visualize_parameter_dampening(dampenings)
        plt.savefig(create_structured_savepath(results_dir, dataset_name, 'smooth', "dampening", P=1, exploration_factor=2.5, n_bo_iter=100, smooth_dampening=True))
        plt.close()
        
        plot_bo_samples_basic(hyperparams_2['result'], 
                            savepath=create_structured_savepath(results_dir, dataset_name, 'smooth', "samples_basic", P=1, exploration_factor=2.5, n_bo_iter=100, smooth_dampening=True),
                            exploration_factor=2.5)
        plot_convergence_analysis(hyperparams_2['result'], 
                                savepath=create_structured_savepath(results_dir, dataset_name, 'smooth', "convergence", P=1, exploration_factor=2.5, n_bo_iter=100, smooth_dampening=True),
                                exploration_factor=2.5)
        
        convergence_table["dataset"].append(dataset_name)
        convergence_table["nosmooth_max_alpha"].append(round(hyperparams['max']['alpha_0'], 3))
        convergence_table["smooth_max_alpha"].append(round(hyperparams_2['max']['alpha_0'], 3))
        convergence_table["nosmooth_max_lambda"].append(round(hyperparams['max']['_lambda_0'], 3))
        convergence_table["smooth_max_lambda"].append(round(hyperparams_2['max']['_lambda_0'], 3))
        targets = np.array([hyperparams['result'][i]['target'] for i in range(len(hyperparams['result']))])
        targets_2 = np.array([hyperparams_2['result'][i]['target'] for i in range(len(hyperparams_2['result']))])
        nosmooth_max_iteration = np.argmax(targets)
        smooth_max_iteration = np.argmax(targets_2)
        convergence_table["nosmooth_max_iteration"].append(int(nosmooth_max_iteration))
        convergence_table["smooth_max_iteration"].append(int(smooth_max_iteration))

        import pandas as pd; 
        df = pd.DataFrame(convergence_table)
        df.to_latex(os.path.join(results_dir, "convergence_table.tex"), index=False)

if __name__ == "__main__":
    main()
