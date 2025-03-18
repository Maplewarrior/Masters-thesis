import hydra
import wandb
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from omegaconf import OmegaConf
from src.datasets.synthetic_dataset import SyntheticDataset
import copy

from src.models.neural_network import NeuralNet
from src.trainers.neural_network_trainer import NeuralNetworkTrainer

from src.evaluation.decision_boundary import DecisionBoundaryCreator

def load_dataset(file):
    npz_file = np.load(file, allow_pickle=True)
    X = npz_file["X"]
    y = npz_file["y"]

    if "rogue_point_idx" in npz_file and npz_file["rogue_point_idx"].item() is not None:
        forget_idx = int(npz_file["rogue_point_idx"])
    else:
        forget_idx = None

    return torch.from_numpy(X), torch.from_numpy(y), forget_idx

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
    dataset_train = SyntheticDataset(X, y, dataset_name="train")
    dataset_val = SyntheticDataset(X_val, y_val, dataset_name="validation")
    dataset_retrain = SyntheticDataset(X_retrain, y_retrain, dataset_name="retrain")
    dataset_forget = SyntheticDataset(X_forget, y_forget, dataset_name="forget")

    batch_size = cfg.data.batch_size
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    dataloader_retrain = DataLoader(dataset_retrain, batch_size=batch_size, shuffle=True)
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



    else:
        raise NotImplementedError(f"Logger {cfg.logging.logger} not implemented")
    
    if cfg.unlearn.method == "retrain":

        model = NeuralNet(M=X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)
        original_model = copy.deepcopy(model)

        # TODO Train a model on the retrain dataset, X_retrain, y_retrain
        print("Epochs: ", cfg.trainer.n_epochs)
        # We train on retrain as it is the Retrain unlearning method
        trainer = NeuralNetworkTrainer(model=model, 
                                       train_dataloader=dataloader_retrain, 
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


        dataset_name = cfg.data.dataset.split("/")[-1].split(".")[0]
        dataset_number = dataset_name.split("_")[1]
        creator = DecisionBoundaryCreator(model, dataloader_retrain)
        plot1 = creator.plot_decision_boundary((-8.5, 8.5), (-8.5, 8.5))
        # plot an 
        plot1.savefig(f"{results_dir}/{dataset_number}_decision_boundary_{cfg.unlearn.method}_unlearned.png")


        creator = DecisionBoundaryCreator(original_model, dataloader_train)
        plot2 = creator.plot_decision_boundary((-8.5, 8.5), (-8.5, 8.5))
        plot2.savefig(f"{results_dir}/{dataset_number}_decision_boundary_{cfg.unlearn.method}_original.png")



        
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
                           gamma=1)(retain_dataloader=dataloader_retrain, 
                                    forget_dataloader=dataloader_forget, 
                                    val_dataloader=dataloader_val, 
                                    n_rounds=10)
            

            creator = DecisionBoundaryCreator(model, dataloader_retrain)
            plot1 = creator.plot_decision_boundary((-8.5, 8.5), (-8.5, 8.5))
            # plot an 
            plot1.savefig(f"{results_dir}/{dataset_number}_decision_boundary_{cfg.unlearn.method}_unlearned.png")

            creator = DecisionBoundaryCreator(original_model, dataloader_train)
            plot2 = creator.plot_decision_boundary((-8.5, 8.5), (-8.5, 8.5))
            plot2.savefig(f"{results_dir}/{dataset_number}_decision_boundary_{cfg.unlearn.method}_original.png")
            

        else:
            raise NotImplementedError(f"Unlearning method {cfg.unlearn.method} not implemented")

if __name__ == "__main__":
    main()
