import hydra
import wandb
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from omegaconf import OmegaConf
from src.datasets.synthetic_dataset import SyntheticDataset
import copy
# import graphviz
import torch.nn as nn
import matplotlib.pyplot as plt

from src.models.neural_network import NeuralNet
from src.trainers.neural_network_trainer import NeuralNetworkTrainer

from src.evaluation.decision_boundary import DecisionBoundaryCreator

results_dir = os.path.join(os.path.dirname(__file__), "results")

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


def decision_boundary_plot(model, original_model, dataloader_retrain, dataloader_train, dataset_name, X_forget, cfg, make_pdfs=True, compress_pdfs=True):
    dataset_number = dataset_name.split("_")[1]
    
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
    professional_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']
    
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
    xx_o, yy_o, decision_boundary_original = original_creator.create_decision_boundary((-8.5, 8.5), (-8.5, 8.5))
    xx_u, yy_u, decision_boundary_unlearned = unlearned_creator.create_decision_boundary((-8.5, 8.5), (-8.5, 8.5))
    
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
    plt.title(f"Decision Boundary\nBefore and After Unlearning for {cfg.unlearn.method}", 
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
    superimposed_filename = f"{results_dir}/{dataset_number}_decision_boundary_{cfg.unlearn.method}_superimposed.png"
    plt.savefig(superimposed_filename, bbox_inches='tight')
    
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

# def create_simple_model_visualization(model, save_path, input_shape):
#     """Create a simple flowchart visualization of the model architecture.
    
#     Args:
#         model: The neural network model to visualize
#         save_path: Path to save the visualization
#         input_shape: Shape of the input data (for labeling input node)
#     """
#     dot = graphviz.Digraph(comment='Neural Network Architecture')
    
#     # Set graph attributes for better appearance
#     dot.attr(rankdir='LR', bgcolor='white', dpi='300', fontname='Helvetica')
    
#     # Set node attributes
#     dot.attr('node', shape='box', style='filled,rounded', 
#              fillcolor='#E8F0FE', color='#4285F4', 
#              fontname='Helvetica', fontsize='14', fontcolor='#333333')
    
#     # Set edge attributes
#     dot.attr('edge', color='#4285F4', penwidth='1.5', arrowsize='0.8')
    
#     # Add input node
#     dot.node('input', f'Input\n({input_shape} features)', shape='oval')
    
#     # Add nodes for each layer in the Sequential model
#     prev_node = 'input'
#     for i, layer in enumerate(model.net):
#         if isinstance(layer, nn.Linear):
#             node_name = f'linear_{i}'
#             label = f'Linear\n{layer.in_features} → {layer.out_features}'
#             dot.node(node_name, label)
#             dot.edge(prev_node, node_name)
#             prev_node = node_name
#         elif isinstance(layer, nn.ReLU):
#             node_name = f'relu_{i}'
#             dot.node(node_name, 'ReLU')
#             dot.edge(prev_node, node_name)
#             prev_node = node_name
    
#     # Add output node
#     dot.node('output', f'Output\n({model.net[-1].out_features} classes)', shape='oval')
#     dot.edge(prev_node, 'output')
    
#     # Render the visualization
#     dot.render(save_path, format='png')
#     return dot

@hydra.main(config_path=".", config_name="config")
def main(cfg):
    DEVICE = 'cpu' #('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    if cfg.unlearn.method == "retrain":

        model = NeuralNet(M=X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)
        # Generate a simplified visualization
        nn_model_image_path = os.path.join(results_dir, f"nn_model_architecture")
        # create_simple_model_visualization(model, nn_model_image_path, X.shape[1])

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

        decision_boundary_plot(model, original_model, dataloader_retain, dataloader_train, dataset_name, X_forget, cfg)

    else:
        if cfg.unlearn.method not in ['amnesiac', 'sisa']:
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
        
        if cfg.unlearn.method == "scrubr":

            from src.unlearners.scrub import ScrubR
            ScrubR(model=unlearned_model, 
                   original_model=original_model,
                   alpha=1,
                   gamma=1,
                   device=DEVICE)(retain_dataloader=dataloader_retain, 
                                  forget_dataloader=dataloader_forget, 
                                  val_dataloader=dataloader_val, 
                                  n_rounds=10)
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataset_name, X_forget, cfg)
        
        elif cfg.unlearn.method == "ssd":
            from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening

            SelectiveSynapticDampening(unlearned_model, 
                                       alpha=1, 
                                       _lambda=1,
                                       device=DEVICE)(full_dataloader=dataloader_train, 
                                                      forget_dataloader=dataloader_forget)
            
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataset_name, X_forget, cfg)
        
        elif cfg.unlearn.method == 'sae':
            from src.models.neural_network import NeuralNetRS
            from src.models.SAE import SAE
            from src.models.neural_net_with_sae import NeuralNetWithSAE
            from src.trainers.sae_trainer import SAETrainer
            from src.unlearners.sae_unlearner import SAEUnlearner
            # instantiate & train neural network
            neural_net = NeuralNetRS(X.shape[1], cfg.data.n_classes)
            nn_trainer = NeuralNetworkTrainer(neural_net, dataloader_train, dataloader_val)
            nn_trainer.train()
            original_model = copy.deepcopy(neural_net)
            
            # instantiate & train SAE
            sae = SAE(d=8, m=32, _lambda=1.).to(DEVICE)
            unlearned_model = NeuralNetWithSAE(neural_net, sae, layer_num=6)
            sae_trainer = SAETrainer(unlearned_model, dataloader_train, dataloader_val, logger=None, device=DEVICE)            
            sae_trainer.train()

            # unlearn with feature dampening
            sae_unlearner = SAEUnlearner(unlearned_model, alpha=0.9, device=DEVICE)
            sae_unlearner(dataloader_retain, dataloader_forget)

            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataset_name, X_forget, cfg)
      
        elif cfg.unlearn.method == "amnesiac":
            from src.unlearners.amnesiac_unlearner import AmnesiacUnlearner
            from src.models.amnesiac_model import AmnesiacModel
            from src.trainers.amnesiac_trainer import AmnesiacTrainer
            
            unlearned_model = NeuralNet(M=X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)
            # Wrap the unlearned model in an AmnesiacModel
            unlearned_model = AmnesiacModel(unlearned_model)

            # Train the unlearned model
            AmnesiacTrainer(model=unlearned_model, 
                            train_dataloader=dataloader_train, 
                            val_dataloader=dataloader_val, 
                            logger=logger, 
                            device=cfg.model.device,
                            learning_rate=cfg.trainer.lr,
                            weight_decay=cfg.trainer.weight_decay,
                            cache_gradients=True,
                            disable_tqdm=cfg.trainer.disable_tqdm,
                            do_early_stopping=cfg.trainer.do_early_stopping,
                            n_epochs=cfg.trainer.n_epochs,
                            save_dir=None)(indices_to_forget=forget_idxs)

            original_model = copy.deepcopy(unlearned_model)

            AmnesiacUnlearner(model=unlearned_model, 
                              unlearn_parameters=cfg.unlearn)(indices_to_forget=forget_idxs)

            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataset_name, X_forget, cfg)

        elif cfg.unlearn.method == "sisa":
            import shutil
            from src.unlearners.sisa_unlearner import SISAUnlearner
            from src.models.sisa_class import SISA
            from src.trainers.sisa_trainer import SISATrainer
            
            weights_dir = os.path.join(os.path.dirname(__file__), "weights")
            sisa_model_fn = NeuralNet
            sisa_model_parameters = {'M': X.shape[1],
                                     'n_classes': cfg.data.n_classes,
                                     'seed': cfg.model.seed}
            sisa_optimzier_parameters = {'lr': cfg.trainer.lr}

            sisa = SISA(dataloader_train,
                        dataloader_val,
                        n_shards=cfg.sisa.n_shards, 
                        n_slices=cfg.sisa.n_slices,
                        model_fn=sisa_model_fn,
                        model_params=sisa_model_parameters,
                        optimizer_parms=sisa_optimzier_parameters,
                        n_classes=cfg.data.n_classes, 
                        n_epochs=cfg.trainer.n_epochs,
                        batch_size=cfg.data.batch_size,
                        save_dir=weights_dir+'/sisa',
                        device=DEVICE,
                        seed=cfg.model.seed) 
            
            SISATrainer(
                model=None, 
                sisa=sisa, 
                train_dataloader=dataloader_train, 
                val_dataloader=dataloader_val, 
                logger=None, 
                device=cfg.model.device, 
                learning_rate=cfg.trainer.lr, 
                n_epochs=cfg.trainer.n_epochs, 
                disable_tqdm=cfg.trainer.disable_tqdm, 
                do_early_stopping=cfg.trainer.do_early_stopping)()
        
            # original SISA model
            sisa_pre_unlearning = sisa.copy()

            # get unlearned SISA model
            sisa_unlearner = SISAUnlearner(sisa)
            sisa_unlearner(forget_indices=forget_idxs)

            decision_boundary_plot(sisa, sisa_pre_unlearning, dataloader_retain, dataloader_train, dataset_name, X_forget, cfg,
                                  compress_pdfs=cfg.get('compress_pdfs', True), make_pdfs=True)#   cfg.get('make_pdfs',True))
                                
            # remove saved SISA weights
            shutil.rmtree(f'{weights_dir}/sisa/{sisa.experiment_id}')

        else:
            raise NotImplementedError(f"Unlearning method {cfg.unlearn.method} not implemented")

if __name__ == "__main__":
    main()