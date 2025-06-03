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
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from src.models.neural_network import NeuralNet
from src.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.evaluation.decision_boundary import DecisionBoundaryCreator
from src.evaluation.membership_inference_attack import MIA


results_dir = os.path.join(os.path.dirname(__file__), "results")

def decision_boundary_plot(model, original_model, dataloader_retrain, dataloader_train, dataset_name, X_forget, cfg):
    dataset_number = dataset_name.split("_")[1]

    creator = DecisionBoundaryCreator(model, dataloader_retrain)
    plot1 = creator.plot_decision_boundary((-8.5, 8.5), (-8.5, 8.5))
    plot1.title("After unlearning")
    plot1.scatter(X_forget[:, 0], X_forget[:, 1], color="red", marker="x", alpha=0.3)
    plot1.savefig(f"{results_dir}/{dataset_number}_decision_boundary_{cfg.unlearn.method}_unlearned.png")

    creator = DecisionBoundaryCreator(original_model, dataloader_train)
    plot2 = creator.plot_decision_boundary((-8.5, 8.5), (-8.5, 8.5))
    plot2.scatter(X_forget[:, 0], X_forget[:, 1], color="red", marker="x")
    plot2.title("Before unlearning")
    plot2.savefig(f"{results_dir}/{dataset_number}_decision_boundary_{cfg.unlearn.method}_original.png")


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

def load_dataset_single(file):
    npz_file = np.load(file, allow_pickle=True)
    X = npz_file["X"]
    y = npz_file["y"]

    if "rogue_point_idx" in npz_file and npz_file["rogue_point_idx"].item() is not None:
        forget_idx = int(npz_file["rogue_point_idx"])
    else:
        forget_idx = None

    return torch.from_numpy(X), torch.from_numpy(y), forget_idx

def plot_nn_heatmaps(state_dict):
    """
    Plot heatmaps for weights and biases of neural network layers in a single figure.
    
    Args:
        state_dict: PyTorch state_dict containing weights and biases
    """
    # Get all layers with weights and biases
    layers = {}
    for key, value in state_dict.items():
        if 'weight' in key or 'bias' in key:
            layer_name = key.rsplit('.', 1)[0]
            param_type = key.rsplit('.', 1)[1]
            
            if layer_name not in layers:
                layers[layer_name] = {}
            
            layers[layer_name][param_type] = value.detach().cpu().numpy()
    
    # Create a single figure with subplots for each layer
    num_layers = len(layers)
    fig, axes = plt.subplots(num_layers, 2, figsize=(12, 4 * num_layers))
    
    # If only one layer, make axes 2D
    if num_layers == 1:
        axes = np.array([axes])
    
    # Plot each layer
    for i, (layer_name, params) in enumerate(layers.items()):
        # Plot weight heatmap
        if 'weight' in params:
            weight = params['weight']
            ax = axes[i, 0]
            
            # Get color scale limits
            vmax = np.abs(weight).max()
            vmin = 0
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vmax/2, vmax=vmax)
            
            sns.heatmap(weight, norm=norm, ax=ax, cbar=True)
            ax.set_title(f"{layer_name} - weight\nShape: {weight.shape}")
            ax.set_xlabel("Input Features")
            ax.set_ylabel("Output Features")
        
        # Plot bias heatmap
        if 'bias' in params:
            bias = params['bias']
            ax = axes[i, 1]
            
            # Reshape bias to 2D for heatmap
            bias_2d = bias.reshape(1, -1)
            
            # Get color scale limits
            vmax = np.abs(bias).max()
            vmin = 0
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vmax/2, vmax=vmax)
            
            sns.heatmap(bias_2d, norm=norm, ax=ax, cbar=True)
            ax.set_title(f"{layer_name} - bias\nShape: {bias.shape}")
            ax.set_xlabel("Neurons")
            ax.set_yticks([])
    
    plt.tight_layout()
    return fig


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

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # ============= Load data and prepare data =============
    dataset_path = cfg.data.dataset
    # path relative to current working directory
    dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)
    validation_path = os.path.join(os.path.dirname(__file__), "data/validation_data.npz")

    ### NOTE: This is for rogue-one data:
    # X, y, forget_idx = load_dataset(dataset_path)
    # # retrain data should be all the data except the index of the rogue point
    # X_val, y_val, _ = load_dataset(validation_path)

    # # retrain data should be all the data except the index of the rogue point
    # all_indices = torch.arange(X.shape[0])
    # X_retrain, y_retrain = X[all_indices != forget_idx], y[all_indices != forget_idx]
    # X_forget, y_forget = X[all_indices == forget_idx], y[all_indices == forget_idx]
    
    ### NOTE: This is for rogue-many data:
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
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    dataloader_retain = DataLoader(dataset_retain, batch_size=batch_size, shuffle=True)
    dataloader_forget = DataLoader(dataset_forget, batch_size=batch_size, shuffle=True)

    dataloader_train_bs1 = DataLoader(dataset_train, batch_size=1, shuffle=False)

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

    original_model = copy.deepcopy(unlearned_model)
    
    if cfg.unlearn.method == "ssd":
        from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening
        hyperparams = {'alpha': 5.,
                        '_lambda': 3.}
        
        ssd = SelectiveSynapticDampening(unlearned_model, 
                                    criterion=nn.CrossEntropyLoss(), 
                                    alpha=hyperparams["alpha"], 
                                    _lambda=hyperparams["_lambda"])
        
        # FIM_forget = ssd.calculate_FIM(dataloader_forget)
        # FIM_full = ssd.calculate_FIM(dataloader_train)
        # FIM_full_bs1 = ssd.calculate_FIM(dataloader_train_bs1)
        # import pdb; pdb.set_trace()
        
        # forget_heatmap_fig = plot_nn_heatmaps(FIM_forget)
        # forget_heatmap_fig.savefig(f'{results_dir}/forget_heatmap_fig.png')
        # full_heatmap_fig = plot_nn_heatmaps(FIM_full)
        # full_heatmap_fig.savefig(f'{results_dir}/full_heatmap_fig.png')

        retrained_model = NeuralNet(M=X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)
        NeuralNetworkTrainer(model=retrained_model, 
                                train_dataloader=dataloader_retain, 
                                val_dataloader=dataloader_val, 
                                logger=logger, 
                                device=cfg.model.device, 
                                learning_rate=cfg.trainer.lr, 
                                n_epochs=cfg.trainer.n_epochs, 
                                disable_tqdm=cfg.trainer.disable_tqdm, 
                                do_early_stopping=cfg.trainer.do_early_stopping)()

        mia_model = MIA()
        mia_prob_original = mia_model(original_model, dataloader_retain, dataloader_forget, dataloader_val)
        mia_prob_unlearned = mia_model(unlearned_model, dataloader_retain, dataloader_forget, dataloader_val)
        mia_prob_retrained = mia_model(retrained_model, dataloader_retain, dataloader_forget, dataloader_val)
        print(f'MIA prob original: {mia_prob_original}')
        print(f'MIA prob unlearned: {mia_prob_unlearned}')
        print(f'MIA prob retrained: {mia_prob_retrained}')
   
    elif cfg.unlearn.method =='teacher-ascend':
        from src.unlearners.teacher_ascend import TeacherAscender
        ta = TeacherAscender(unlearned_model, n_epochs=50)
        ta(dataloader_retain, dataloader_forget)
        decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataset_name, X_forget, cfg)

        retrained_model = NeuralNet(M=X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)
        NeuralNetworkTrainer(model=retrained_model, 
                                train_dataloader=dataloader_retain, 
                                val_dataloader=dataloader_val, 
                                logger=logger, 
                                device=cfg.model.device, 
                                learning_rate=cfg.trainer.lr, 
                                n_epochs=cfg.trainer.n_epochs, 
                                disable_tqdm=cfg.trainer.disable_tqdm, 
                                do_early_stopping=cfg.trainer.do_early_stopping)()


        mia_model = MIA()
        mia_prob_original = mia_model(original_model, dataloader_retain, dataloader_forget, dataloader_val)
        mia_prob_unlearned = mia_model(unlearned_model, dataloader_retain, dataloader_forget, dataloader_val)
        mia_prob_retrained = mia_model(retrained_model, dataloader_retain, dataloader_forget, dataloader_val)
        print(f'MIA prob original: {mia_prob_original}')
        print(f'MIA prob unlearned: {mia_prob_unlearned}')
        print(f'MIA prob retrained: {mia_prob_retrained}')

    else:
        raise NotImplementedError(f"Unlearning method {cfg.unlearn.method} not implemented")

if __name__ == "__main__":
    main()
