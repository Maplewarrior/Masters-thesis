import hydra
import wandb
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from omegaconf import OmegaConf
from src.datasets.synthetic_dataset import SyntheticDataset
import copy
import graphviz
import torch.nn as nn

from src.models.neural_network import NeuralNet
from src.trainers.neural_network_trainer import NeuralNetworkTrainer

from src.evaluation.decision_boundary import DecisionBoundaryCreator

results_dir = os.path.join(os.path.dirname(__file__), "results")

def load_dataset(file):
    npz_file = np.load(file, allow_pickle=True)
    X = npz_file["X"]
    y = npz_file["y"]

    if "rogue_point_idx" in npz_file and npz_file["rogue_point_idx"].item() is not None:
        forget_idx = int(npz_file["rogue_point_idx"])
    else:
        forget_idx = None

    return torch.from_numpy(X), torch.from_numpy(y), forget_idx

def decision_boundary_plot(model, original_model, dataloader_retrain, dataloader_train, dataset_name, X_forget, cfg):
        dataset_number = dataset_name.split("_")[1]
    
        creator = DecisionBoundaryCreator(model, dataloader_retrain)
        plot1 = creator.plot_decision_boundary((-8.5, 8.5), (-8.5, 8.5))
        plot1.title("Unlearned")
        plot1.scatter(X_forget[0, 0], X_forget[0, 1], color="red", marker="x")
        plot1.savefig(f"{results_dir}/{dataset_number}_decision_boundary_{cfg.unlearn.method}_unlearned.png")

        creator = DecisionBoundaryCreator(original_model, dataloader_train)
        plot2 = creator.plot_decision_boundary((-8.5, 8.5), (-8.5, 8.5))
        plot2.scatter(X_forget[0, 0], X_forget[0, 1], color="red", marker="x", alpha=0.3)
        plot2.title("Original")
        plot2.savefig(f"{results_dir}/{dataset_number}_decision_boundary_{cfg.unlearn.method}_original.png")

def create_simple_model_visualization(model, save_path, input_shape):
    """Create a simple flowchart visualization of the model architecture.
    
    Args:
        model: The neural network model to visualize
        save_path: Path to save the visualization
        input_shape: Shape of the input data (for labeling input node)
    """
    dot = graphviz.Digraph(comment='Neural Network Architecture')
    
    # Set graph attributes for better appearance
    dot.attr(rankdir='LR', bgcolor='white', dpi='300', fontname='Helvetica')
    
    # Set node attributes
    dot.attr('node', shape='box', style='filled,rounded', 
             fillcolor='#E8F0FE', color='#4285F4', 
             fontname='Helvetica', fontsize='14', fontcolor='#333333')
    
    # Set edge attributes
    dot.attr('edge', color='#4285F4', penwidth='1.5', arrowsize='0.8')
    
    # Add input node
    dot.node('input', f'Input\n({input_shape} features)', shape='oval')
    
    # Add nodes for each layer in the Sequential model
    prev_node = 'input'
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.Linear):
            node_name = f'linear_{i}'
            label = f'Linear\n{layer.in_features} → {layer.out_features}'
            dot.node(node_name, label)
            dot.edge(prev_node, node_name)
            prev_node = node_name
        elif isinstance(layer, nn.ReLU):
            node_name = f'relu_{i}'
            dot.node(node_name, 'ReLU')
            dot.edge(prev_node, node_name)
            prev_node = node_name
    
    # Add output node
    dot.node('output', f'Output\n({model.net[-1].out_features} classes)', shape='oval')
    dot.edge(prev_node, 'output')
    
    # Render the visualization
    dot.render(save_path, format='png')
    return dot

@hydra.main(config_path=".", config_name="config")
def main(cfg):

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # ============= Load data and prepare data =============
    dataset_path = cfg.data.dataset
    # path relative to current working directory
    dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)
    validation_path = os.path.join(os.path.dirname(__file__), "data/validation_data.npz")

    X, y, forget_idx = load_dataset(dataset_path)
    # retrain data should be all the data except the index of the rogue point
    X_val, y_val, _ = load_dataset(validation_path)

    # retrain data should be all the data except the index of the rogue point
    all_indices = torch.arange(X.shape[0])
    X_retrain, y_retrain = X[all_indices != forget_idx], y[all_indices != forget_idx]
    X_forget, y_forget = X[all_indices == forget_idx], y[all_indices == forget_idx]
    
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
        create_simple_model_visualization(model, nn_model_image_path, X.shape[1])

        original_model = copy.deepcopy(model)

        # TODO Train a model on the retrain dataset, X_retrain, y_retrain
        print("Epochs: ", cfg.trainer.n_epochs)
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

        trainer = NeuralNetworkTrainer(model=original_model, 
                                       train_dataloader=dataloader_train, 
                                       val_dataloader=dataloader_val, 
                                       logger=logger, 
                                       device=cfg.model.device, 
                                       learning_rate=cfg.trainer.lr, 
                                       n_epochs=cfg.trainer.n_epochs, 
                                       disable_tqdm=cfg.trainer.disable_tqdm, 
                                       do_early_stopping=cfg.trainer.do_early_stopping)()


        decision_boundary_plot(model, original_model, dataloader_retain, dataloader_train, dataset_name, X_forget, cfg)


    else:

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
                           gamma=1)(retain_dataloader=dataloader_retain, 
                                    forget_dataloader=dataloader_forget, 
                                    val_dataloader=dataloader_val, 
                                    n_rounds=10)
            
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataset_name, X_forget, cfg)
        elif cfg.unlearn.method == "ssd":
            from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening


            SelectiveSynapticDampening(unlearned_model, 
                                       criterion=nn.CrossEntropyLoss(), 
                                       alpha=1, 
                                       _lambda=1)(full_dataloader=dataloader_train, 
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
            sae = SAE(d=8, m=32, _lambda=1.)
            unlearned_model = NeuralNetWithSAE(neural_net, sae, layer_num=6)
            sae_trainer = SAETrainer(unlearned_model, dataloader_train, dataloader_val)
            sae_trainer.train()

            # unlearn with feature dampening
            sae_unlearner = SAEUnlearner(unlearned_model, alpha=0.9)
            sae_unlearner(dataloader_retain, dataloader_forget)

            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataset_name, X_forget, cfg)

            
        elif cfg.unlearn.method == "amnesiac":
            from src.unlearners.amnesiac_unlearner import AmnesiacUnlearner
            from src.models.amnesiac_model import AmnesiacModel
            from src.trainers.amnesiac_trainer import AmnesiacTrainer


            # Wrap the unlearned model in an AmnesiacModel
            unlearned_model = AmnesiacModel(unlearned_model)

            # Train the unlearned model
            AmnesiacTrainer(model=unlearned_model, 
                            train_dataloader=dataloader_train, 
                            val_dataloader=dataloader_val, 
                            logger=logger, 
                            device=cfg.model.device,
                            learning_rate=cfg.trainer.lr,
                            cache_gradients=True,
                            disable_tqdm=cfg.trainer.disable_tqdm,
                            do_early_stopping=cfg.trainer.do_early_stopping,
                            n_epochs=cfg.trainer.n_epochs)(indices_to_forget=[forget_idx])

            original_model = copy.deepcopy(unlearned_model)

            AmnesiacUnlearner(model=unlearned_model, 
                              unlearn_parameters=cfg.unlearn)(indices_to_forget=[forget_idx])

            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataset_name, X_forget, cfg)


        elif cfg.unlearn.method == "sisa":
            from src.unlearners.sisa_unlearner import SISAUnlearner
            from src.sisa_implementation.sisa_class import SISA
            from src.trainers.sisa_trainer import SISATrainer
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
        
            sisa = SISA(dataloader_train, 
                        n_classes=cfg.data.n_classes, 
                        n_features=X.shape[1], 
                        n_epochs=cfg.trainer.n_epochs, 
                        n_shards=cfg.sisa.n_shards, 
                        n_slices=cfg.sisa.n_slices,
                        save_dir=results_dir+'/sisa')
            sisa.process_data()
            
            SISATrainer(
                model=sisa.model, 
                sisa=sisa, 
                train_dataloader=dataloader_train, 
                val_dataloader=dataloader_val, 
                logger=None, 
                device=cfg.model.device, 
                learning_rate=cfg.trainer.lr, 
                n_epochs=cfg.trainer.n_epochs, 
                disable_tqdm=cfg.trainer.disable_tqdm, 
                do_early_stopping=cfg.trainer.do_early_stopping)()
            
            sisa_pre_unlearning = sisa.copy()

            batch = next(iter(dataloader_retain))
            X_batch, y_batch, _ = batch

            sisa_unlearner = SISAUnlearner(sisa)
            sisa_unlearner(forget_indices=[forget_idx])

            #! Monkey mode
            x_interval = (-8.5, 8.5)
            y_interval = (-8.5, 8.5)

            x = torch.linspace(x_interval[0], x_interval[1], 1000)
            y = torch.linspace(y_interval[0], y_interval[1], 1000)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            
            # Reshape the grid into a 2D array of points
            grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

            #! Post unlearning
            decision_boundary_post_unlearning = sisa.predict(grid).reshape(xx.shape)

            #! Pre unlearning
            decision_boundary_pre_unlearning = sisa_pre_unlearning.predict(grid).reshape(xx.shape)

            print(decision_boundary_pre_unlearning)
            import pdb; pdb.set_trace()

            X, y = dataloader_retrain.dataset.X, dataloader_retrain.dataset.y

            # If y is onehot, convert it to class indices
            if len(y.shape) == 2 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1) 

            classes = np.unique(y)

            colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
            custom_cmap = ListedColormap(colors)

            # ! We start with pre unlearning
            plt.figure(figsize=(10, 8))

            plt.pcolormesh(xx.numpy(), yy.numpy(), decision_boundary_pre_unlearning, 
                                         alpha=0.4, cmap=custom_cmap, shading='auto')        
                
            for class_idx, color in zip(classes, colors):
                plt.scatter(X[y == class_idx, 0], X[y == class_idx, 1], color=color, s=10, label=f"Class {class_idx}", alpha=0.5)

            plt.title = "Pre unlearning"
            plt.scatter(X_forget[0, 0], X_forget[0, 1], color="red", marker="x")
            plt.legend()
            plt.savefig(f"{results_dir}/sisa/decision_boundary_pre_unlearning.png")


            # ! We end with unlearning
            plt.figure(figsize=(10, 8))

            plt.pcolormesh(xx.numpy(), yy.numpy(), decision_boundary_post_unlearning, 
                                         alpha=0.4, cmap=custom_cmap, shading='auto')  
            for class_idx, color in zip(classes, colors):
                plt.scatter(X[y == class_idx, 0], X[y == class_idx, 1], color=color, s=10, label=f"Class {class_idx}", alpha=0.5)

            plt.title = "Post unlearning"
            plt.scatter(X_forget[0, 0], X_forget[0, 1], color="red", marker="x")
            plt.legend()
            plt.savefig(f"{results_dir}/sisa/decision_boundary_post_unlearning.png")
        else:
            raise NotImplementedError(f"Unlearning method {cfg.unlearn.method} not implemented")

if __name__ == "__main__":
    main()
