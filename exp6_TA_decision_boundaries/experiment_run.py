import hydra
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from src.datasets.synthetic_dataset import SyntheticDataset
import copy
from src.models.neural_network import NeuralNet
from src.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.utils.get_git_root_path import get_git_root
from src.plotting.decision_boundary_plot import decision_boundary_plot

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
    
    model = NeuralNet(M=X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)

    original_model = copy.deepcopy(model)
    retrained_model = copy.deepcopy(model)

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
    trainer = NeuralNetworkTrainer(model=retrained_model, 
                                    train_dataloader=dataloader_retain, 
                                    val_dataloader=dataloader_val, 
                                    logger=logger, 
                                    device=cfg.model.device, 
                                    learning_rate=cfg.trainer.lr, 
                                    n_epochs=cfg.trainer.n_epochs, 
                                    disable_tqdm=cfg.trainer.disable_tqdm, 
                                    do_early_stopping=cfg.trainer.do_early_stopping)()

    plot_title = f"Retrained"
    decision_boundary_filename = f"{dataset_number}_decision_boundary_retrained"
    decision_boundary_plot(retrained_model, original_model, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)

    from src.unlearners.teacher_ascend import TeacherAscender
    epochs = 20
    _lambda = 2 
    ta_model = copy.deepcopy(original_model)
    ta = TeacherAscender(ta_model, n_epochs=epochs, 
                        _lambda=_lambda, device=DEVICE)
    ta_metrics = ta(dataloader_retain, dataloader_forget, dataloader_val, eval=True, version="entropy-retain", is_FIM_ratio=True)
    
    plot_title = f"Teacher Ascend"
    decision_boundary_filename = f"{dataset_number}_decision_boundary_teacher_ascend"
    decision_boundary_plot(ta_model, original_model, dataloader_retain, dataloader_train, X_forget, results_dir, plot_title=plot_title, filename=decision_boundary_filename)

if __name__ == "__main__":
    main()
