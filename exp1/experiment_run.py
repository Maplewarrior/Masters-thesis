
import hydra
import wandb
from src.modelling.UnlearningManager import UnlearningManager
import torch
import os


def load_dataset(file):
    import numpy as np

    npz_file = np.load(file)
    X = npz_file["X"]
    y = npz_file["y"]
    forget_idx = npz_file["rogue_point_idx"]

    return torch.from_numpy(X), torch.from_numpy(y), int(forget_idx)

@hydra.main(config_path=".", config_name="config")
def main(cfg):

    # ============= Load training and validation data =============
    dataset_path = cfg.data.dataset
    # path relative to current working directory
    dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)
    validation_path = os.path.join(os.path.dirname(__file__), "data/validation_data.npz")

    # ============= Load data =============
    X, y, forget_idx = load_dataset(dataset_path)
    # retrain data should be all the data except the index of the rogue point
    X_val, y_val, _ = load_dataset(validation_path)

    # retrain data should be all the data except the index of the rogue point
    all_indices = torch.arange(X.shape[0])
    X_retrain, y_retrain = X[all_indices != forget_idx], y[all_indices != forget_idx]
    X_forget, y_forget = X[all_indices == forget_idx], y[all_indices == forget_idx]
    
    # X, y, X_val, y_val, X_retrain, y_retrain, X_forget, y_forget


    model = cfg.model.name
    n_epochs = cfg.model.n_epochs
    lr = cfg.model.lr
    batch_size = cfg.model.batch_size
    seed = cfg.model.seed

    logger = wandb.init(project_name="exp1", config=cfg)

    unlearning_manager = UnlearningManager(config=cfg, device="cuda", wandb=logger, track_performance=True)
    
    

    if model.name == "retrain":
        raise NotImplementedError("Retrain is not implemented")

        data.retain
        # TODO Train a model on the retrain dataset, X_retrain, y_retrain

        # 
        
    else:
        #TODO apply_unlearning(model, X_forget, y_forget, X_val, y_val)


if __name__ == "__main__":
    main()
