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

from src.evaluation.decision_boundary import DecisionBoundaryCreator
from src.visualization.dampening_visualization import visualize_parameter_dampening

results_dir = os.path.join(os.path.dirname(__file__), "results_ablation_ssd")

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


def decision_boundary_plot(model, 
                           original_model, 
                           dataloader_retrain, 
                           dataloader_train, 
                           dataset_name, 
                           X_forget, cfg, 
                           make_pdfs=False, 
                           compress_pdfs=False,
                           layer_target=None,
                           hyperparams=None,
                           title=None):
    dataset_number = dataset_name.split("_")[1]
    print(f'Dataset number: {dataset_number}')
    
    # Create structured directory hierarchy
    layer_target_str = '_'.join(map(str, layer_target)) if layer_target else 'all'
    alpha_val = hyperparams.get('alpha', 'default') if hyperparams else 'default'
    lambda_val = hyperparams.get('lambda', 'default') if hyperparams else 'default'
    
    # Create nested directory structure
    plot_dir = os.path.join(
        results_dir,
        f"dataset_{dataset_number}",
        f"layer_target_{layer_target_str}",
        f"alpha_{alpha_val}_lambda_{lambda_val}"
    )
    os.makedirs(plot_dir, exist_ok=True)
    
    # Separate subdirectories for different formats
    png_dir = os.path.join(plot_dir, "png")
    pdf_dir = os.path.join(plot_dir, "pdf")
    os.makedirs(png_dir, exist_ok=True)
    if make_pdfs:
        os.makedirs(pdf_dir, exist_ok=True)
    
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
                colors=['#003f5c'],  # Blue like
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
    plt.title(title if title is not None else f"Decision Boundary\nBefore and After Unlearning for {cfg.unlearn.method}", 
              fontsize=18)
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
    
    # Save with cleaner filename structure
    base_filename = "decision_boundary"
    
    # Save PNG with high DPI
    png_filename = os.path.join(png_dir, f"{base_filename}.png")
    plt.savefig(png_filename, bbox_inches='tight', dpi=200)
    print(f"PNG saved to: {png_filename}")
    
    # Save as PDF if enabled
    if make_pdfs:
        pdf_filename = os.path.join(pdf_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_filename, 
                   bbox_inches='tight', 
                   format='pdf',
                   dpi=150)  # Reduced DPI for PDF
        
        # Add to list for later compression if enabled
        pdf_files.append(pdf_filename)
        print(f"PDF saved to: {pdf_filename}")
    
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
    
    return plot_dir  # Return the plot directory for reference

@hydra.main(config_path=".", config_name="config")
def main(cfg):
    DEVICE = 'cpu' #('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(results_dir, exist_ok=True)

    # ============= Load data and prepare data =============
    for i in range(1, 6):
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

        dataset_name = dataset_path.split("/")[-1].split(".")[0]

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
        model = NeuralNet(M=X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)

        # Layer block
        # Get all possible layer combinations
        layers = [0, 2, 4]
        from itertools import combinations
        layer_combinations = []
        for r in range(1, len(layers) + 1):
            layer_combinations.extend(list(combinations(layers, r)))

        for layer_target in layer_combinations:
            print(f'Running experiment for layer target {layer_target}')
            original_model = copy.deepcopy(model)

            print("Epochs: ", cfg.trainer.n_epochs)
            trainer = NeuralNetworkTrainer(model=original_model, 
                                            train_dataloader=dataloader_train, 
                                            val_dataloader=dataloader_val, 
                                            logger=logger,
                                            device=cfg.model.device,
                                            learning_rate=cfg.trainer.lr,
                                            n_epochs=cfg.trainer.n_epochs, 
                                            disable_tqdm=cfg.trainer.disable_tqdm, 
                                            do_early_stopping=cfg.trainer.do_early_stopping)()
            
            # We train on retrain as it is the Retrain unlearning method
            trainer = NeuralNetworkTrainer(model=model, 
                                            train_dataloader=dataloader_retain, 
                                            val_dataloader=dataloader_val, 
                                            logger=logger, 
                                            device=cfg.model.device, 
                                            learning_rate=cfg.trainer.lr, 
                                            n_epochs=cfg.trainer.n_epochs, 
                                            disable_tqdm=cfg.trainer.disable_tqdm, 
                                            do_early_stopping=cfg.trainer.do_early_stopping)()

            # decision_boundary_plot(model, original_model, dataloader_retain, dataloader_train, dataset_name, X_forget, cfg)


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

            original_model = copy.deepcopy(unlearned_model)
                
            from src.unlearners.ablation_ssd_layerwise import SelectiveSynapticDampening_layerwise as SelectiveSynapticDampening_ablation

            hyperparams = {
                'alpha': 5,
                'lambda': 3
            }

            dampenings = SelectiveSynapticDampening_ablation(unlearned_model, 
                                            alpha=hyperparams['alpha'], 
                                            _lambda=hyperparams['lambda'],
                                            layer_target=layer_target,
                                            device=DEVICE)(full_dataloader=dataloader_train, 
                                                            forget_dataloader=dataloader_forget,
                                                            return_dampening=True)
            
            plot_dir = decision_boundary_plot(unlearned_model, 
                                original_model, 
                                dataloader_retain, 
                                dataloader_train, 
                                dataset_name, 
                                X_forget, 
                                cfg, 
                                layer_target=layer_target,
                                hyperparams=hyperparams,
                                title=f'SSD\n${{L}}_{{target}}$ = {",".join(map(str, layer_target))}\n${{\\alpha}}$ = {hyperparams["alpha"]}, ${{\\lambda}}$ = {hyperparams["lambda"]}')

            # Create dampening visualization with structured path
            visualize_parameter_dampening(dampenings, figsize=(10, 6), title=f'${{L}}_{{target}}$ = {",".join(map(str, layer_target))}\n${{\\alpha}}$ = {hyperparams["alpha"]}, ${{\\lambda}}$ = {hyperparams["lambda"]}', title_fontsize=18)
            
            # Save dampening plot in the same structured directory
            dampening_filename = os.path.join(plot_dir, "png", "parameter_dampening.png")
            plt.savefig(dampening_filename, bbox_inches='tight', dpi=200)
            print(f"Parameter dampening plot saved to: {dampening_filename}")
            plt.close()

if __name__ == "__main__":
    main()