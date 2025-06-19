# These functions are used to load the rogue data and create the dataloaders
# The rogue datasets are generated in the exp1_decision_boundary folder
# The functions are used in the experiments folders

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets.synthetic_dataset import SyntheticDataset

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

def get_data(dataset_path, validation_path, batch_size, n_classes):

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
    dataset_train = SyntheticDataset(X, y, dataset_name="train", n_classes=n_classes)
    dataset_val = SyntheticDataset(X_val, y_val, dataset_name="validation", n_classes=n_classes)
    dataset_retain = SyntheticDataset(X_retrain, y_retrain, dataset_name="retrain", n_classes=n_classes)
    dataset_forget = SyntheticDataset(X_forget, y_forget, dataset_name="forget", n_classes=n_classes)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    dataloader_retain = DataLoader(dataset_retain, batch_size=batch_size, shuffle=True)
    dataloader_forget = DataLoader(dataset_forget, batch_size=batch_size, shuffle=True)

    return dataloader_train, dataloader_val, dataloader_retain, dataloader_forget, forget_idxs
