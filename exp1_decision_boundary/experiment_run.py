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
from src.utils.get_git_root_path import get_git_root
from src.plotting.decision_boundary_plot import decision_boundary_plot

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


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    DEVICE = 'cpu' #('cuda' if torch.cuda.is_available() else 'cpu')

    # ============= Preparing data and results directory =============
    git_root = get_git_root()
    dataset_path = cfg.data.dataset
    # path relative to current working directory
    dataset_path = os.path.join(git_root, dataset_path)
    validation_path = os.path.join("/",*dataset_path.split("/")[1:-1], "validation_data.npz")

    rogue = dataset_path.split("/")[-2] # rogue_many or rogue_one
    results_dir = os.path.join(os.path.dirname(__file__), "results", rogue)
    os.makedirs(results_dir, exist_ok=True)


    # ============== Load data ============== 
    X, y, forget_idxs = load_dataset(dataset_path)
    # retrain data should be all the data except the index of the rogue point
    X_val, y_val, _ = load_dataset(validation_path)

    if forget_idxs.ndim == 0:
        forget_idxs = torch.tensor([forget_idxs])

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


        plot_title = f"Retrained"
        decision_boundary_filename = f"{dataset_number}_{cfg.unlearn.method}_decision_boundary" 
        decision_boundary_plot(model, original_model, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)

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
                                  n_rounds=4, # min + max rounds
                                  n_repair_rounds=6 # min only rounds 
                                  )
            plot_title = f"SCRUB+R"
            decision_boundary_filename = f"{dataset_number}_{cfg.unlearn.method}_decision_boundary"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
        
        elif cfg.unlearn.method == "ssd":
            from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening

            SelectiveSynapticDampening(unlearned_model, 
                                       alpha=1, 
                                       _lambda=1,
                                       device=DEVICE)(full_dataloader=dataloader_train, 
                                                      forget_dataloader=dataloader_forget)
            
            plot_title = f"SSD"
            decision_boundary_filename = f"{dataset_number}_{cfg.unlearn.method}_decision_boundary"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
        
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

            plot_title = f"SAE"
            decision_boundary_filename = f"{dataset_number}_{cfg.unlearn.method}_decision_boundary"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
      
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

            plot_title = f"Amnesiac"
            decision_boundary_filename = f"{dataset_number}_{cfg.unlearn.method}_decision_boundary"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)

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

            plot_title = f"SISA"
            decision_boundary_filename = f"{dataset_number}_{cfg.unlearn.method}_decision_boundary"
            decision_boundary_plot(sisa, sisa_pre_unlearning, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
                                
            # remove saved SISA weights
            shutil.rmtree(f'{weights_dir}/sisa/{sisa.experiment_id}')

        else:
            raise NotImplementedError(f"Unlearning method {cfg.unlearn.method} not implemented")

if __name__ == "__main__":
    main()