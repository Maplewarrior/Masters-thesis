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
from src.plotting.decision_boundary_plot import decision_boundary_plot

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

def count_updated_params(original_model, unlearned_model):
    pass

def ssd_bo_title(hyperparams):
    
    plt_tit = ""

    n_hyperparam_pairs = int((len(hyperparams["max"]) - 1)/2)
    for i in range(n_hyperparam_pairs):
        alpha = hyperparams["max"][f"alpha_{i}"]
        lamb = hyperparams["max"][f"_lambda_{i}"]
        num = i + 1
        plt_tit += f"$\\alpha_{num}={alpha:.2f}$, $\\lambda_{num}={lamb:.2f}$"
        if i < n_hyperparam_pairs - 1:  # Add comma and space except for last item
            plt_tit += ",   "

    plt_tit = plt_tit.rstrip(', ')

    return plt_tit

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
    
    dataset_name = cfg.data.dataset.split("/")[-1].split(".")[0]
    dataset_number = dataset_name.split("_")[1]

    # No logger, this could be changed to a wandb logger if needed
    logger = None 
    
    if cfg.unlearn.method == "retrain":

        model = NeuralNet(M=X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)

        # Generate a simplified visualization
        nn_model_image_path = os.path.join(results_dir, f"nn_model_architecture")
        #create_simple_model_visualization(model, nn_model_image_path, X.shape[1])

        original_model = copy.deepcopy(model)

        # TODO Train a model on the retrain dataset, X_retrain, y_retrain
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

        decision_boundary_filename = f"{dataset_number}_decision_boundary_{cfg.unlearn.method}"
        plot_title = f"Retrained"
        decision_boundary_plot(model, original_model, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)

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
        
        if cfg.unlearn.method == "ssd":
            from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening
            # hyperparams = {'alpha': 30.,
            #                '_lambda': 5.}
            hyperparams = {'alpha': 1.,
                           '_lambda': 1.}
            dampenings = SelectiveSynapticDampening(unlearned_model,
                                       alpha=hyperparams["alpha"], 
                                       _lambda=hyperparams["_lambda"],
                                       device=DEVICE)(
                                                 full_dataloader=dataloader_train, 
                                                 forget_dataloader=dataloader_forget,
                                                 return_dampening=True)
            
            decision_boundary_filename = f"{dataset_number}_decision_boundary_{cfg.unlearn.method}_alpha{hyperparams['alpha']}_lambda{hyperparams['_lambda']}"
            plot_title = f"SSD\n$\\alpha={hyperparams['alpha']:.2f}$, $\\lambda={hyperparams['_lambda']:.2f}$"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
            fig = visualize_parameter_dampening(dampenings)
            plt.savefig(f'{results_dir}/{dataset_number}_{cfg.unlearn.method}_alpha={hyperparams["alpha"]}_lambda={hyperparams["_lambda"]}_dampening.png')
        
        elif cfg.unlearn.method == "ssd_v2":
            from src.unlearners.selective_synaptic_dampening_v2 import SelectiveSynapticDampening
            # hyperparams = {'alpha': 30.,
            #                '_lambda': 5.}
            hyperparams = {'alpha': 1.,
                           '_lambda': 1.}
            dampenings = SelectiveSynapticDampening(unlearned_model,
                                       alpha=hyperparams["alpha"], 
                                       _lambda=hyperparams["_lambda"],
                                       device=DEVICE)(full_dataloader=dataloader_train, 
                                                      forget_dataloader=dataloader_forget,
                                                      return_dampening=True)
            
            decision_boundary_filename = f"{dataset_number}_decision_boundary_{cfg.unlearn.method}_alpha{hyperparams['alpha']}_lambda{hyperparams['_lambda']}"
            plot_title = f"SSD\n$\\alpha={hyperparams['alpha']:.2f}$, $\\lambda={hyperparams['_lambda']:.2f}$"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
            
            fig = visualize_parameter_dampening(dampenings)
            plt.savefig(f'{results_dir}/{dataset_number}_{cfg.unlearn.method}_alpha={hyperparams["alpha"]}_lambda={hyperparams["_lambda"]}_dampening.png')
        
        elif cfg.unlearn.method == "ssd_v3":
            from src.unlearners.selective_synaptic_dampening_v3 import SelectiveSynapticDampening
            hyperparams, dampenings = SelectiveSynapticDampening(unlearned_model, 
                                                     device=DEVICE)(full_dataloader=dataloader_train, 
                                                                    forget_dataloader=dataloader_forget,
                                                                    validation_dataloader=dataloader_val,
                                                                    return_dampening=True)
            
            decision_boundary_filename = f"{dataset_number}_decision_boundary_{cfg.unlearn.method}" # ? Do not include hyperparams in the filename, because these are not static. They are found with BO.
            plot_title = f"BO-SSD\n$\\alpha={hyperparams['alpha']:.2f}$, $\\lambda={hyperparams['_lambda']:.2f}$"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
            
            fig = visualize_parameter_dampening(dampenings)
            plt.savefig(f'{results_dir}/{dataset_number}_{cfg.unlearn.method}_dampening.png')

        elif cfg.unlearn.method == "ssd_v6":
            from src.unlearners.selective_synaptic_dampening_v6 import SelectiveSynapticDampening
            P = 2 # TODO: Hyperparam to config
            hyperparams, dampenings = SelectiveSynapticDampening(unlearned_model,
                                                     P=P,
                                                     k=0.999,
                                                     smooth_dampening=False,
                                                     n_bo_iter=100,
                                                     device=DEVICE)(full_dataloader=dataloader_train, 
                                                                    forget_dataloader=dataloader_forget,
                                                                    validation_dataloader=dataloader_val,
                                                                    return_dampening=True)
            
            decision_boundary_filename = f"{dataset_number}_decision_boundary_{cfg.unlearn.method}_P{P}" # ? Do not include hyperparams in the filename, because these are not static. They are found with BO.
            

            plot_title = f"BO-SSD\n{ssd_bo_title(hyperparams)}"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
            fig = visualize_parameter_dampening(dampenings)
            plt.savefig(f'{results_dir}/{dataset_number}_{cfg.unlearn.method}_dampening.png')

        elif cfg.unlearn.method == "ssd_v6_smooth":
            from src.unlearners.selective_synaptic_dampening_v6 import SelectiveSynapticDampening
            P = 2 # TODO: Hyperparam to config
            hyperparams, dampenings = SelectiveSynapticDampening(unlearned_model,
                                                     P=P,
                                                     k=0.999,
                                                     smooth_dampening=True,
                                                     n_bo_iter=100,
                                                     device=DEVICE)(full_dataloader=dataloader_train, 
                                                                    forget_dataloader=dataloader_forget,
                                                                    validation_dataloader=dataloader_val,
                                                                    return_dampening=True)
            
            decision_boundary_filename = f"{dataset_number}_decision_boundary_{cfg.unlearn.method}_P{P}" # ? Do not include hyperparams in the filename, because these are not static. They are found with BO.
            
            plot_title = f"BO-SSD (smooth)\n{ssd_bo_title(hyperparams)}"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
            
            fig = visualize_parameter_dampening(dampenings)
            plt.savefig(f'{results_dir}/{dataset_number}_{cfg.unlearn.method}_dampening.png')
    
        elif cfg.unlearn.method == "assd":
            from src.unlearners.adaptive_ssd import AdaptiveSSD
            hyperparams, dampenings = AdaptiveSSD(unlearned_model, device=DEVICE)(dataloader_train, dataloader_forget, return_dampening=True)
            decision_boundary_filename = f"{dataset_number}_decision_boundary_{cfg.unlearn.method}" # ? Do not include hyperparams in the filename, because these are not static. They are found with BO.
            plot_title = f"Adaptive SSD"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
            
            fig = visualize_parameter_dampening(dampenings)
            plt.savefig(f'{results_dir}/{dataset_number}_{cfg.unlearn.method}_dampening.png')
        
        else:
            raise NotImplementedError(f"Unlearning method {cfg.unlearn.method} not implemented")

if __name__ == "__main__":
    main()
