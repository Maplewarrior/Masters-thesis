import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import pdb

def download_dataset(root_dir: str, dataset_name: str):
    if dataset_name == 'MNIST':
        return download_mnist_dataset(root_dir)
    elif dataset_name == 'CIFAR10':
        return download_cifar_dataset(root_dir)
    else:
        raise NotImplementedError(f"The requested dataset {dataset_name} is not supported.")


def download_cifar_dataset(root_dir: str):
    os.makedirs(root_dir, exist_ok=True)
    cifar_train_dataset= CIFAR10(root_dir, train=True, download=True)
    cifar_test_dataset= CIFAR10(root_dir, train=False, download=True)
    return cifar_train_dataset, cifar_test_dataset


def download_mnist_dataset(root_dir: str):
    os.makedirs(root_dir, exist_ok=True)
    # x_transform = lambda x: F.pil_to_tensor(x).view(-1)
    # y_transform = lambda y: torch.tensor([y], dtype=torch.long)
    mnist_train_dataset = MNIST(root=root_dir, train=True, 
                                # transform=x_transform, target_transform=y_transform, 
                                download=True)
    mnist_test_dataset = MNIST(root=root_dir, train=False, 
                                # transform=x_transform, target_transform=y_transform,
                               download=True)
    
    return mnist_train_dataset, mnist_test_dataset


def split_retain_forget_random(X_train, y_train, n_forget_points: int):
    forget_mask = torch.zeros(X_train.size(0), dtype=bool)
    forget_idxs = torch.randperm(X_train.size(0))[:n_forget_points].sort().values
    forget_mask[forget_idxs] = True
    
    X_retain = X_train[~forget_mask, :]
    y_retain = y_train[~forget_mask, :]
    X_forget = X_train[forget_mask, :]
    y_forget = y_train[forget_mask, :]
    return X_retain, y_retain, X_forget, y_forget, forget_idxs


def subsample_train(X_train, y_train, subsample_size: int = None):
    idxs = torch.randperm(X_train.size(0))[:subsample_size]
    return X_train[idxs], y_train[idxs]

def preprocess_cifar_data(train_dataset, test_dataset):
    X_train = torch.tensor(train_dataset.data)
    y_train = torch.tensor(train_dataset.targets)
    X_test = torch.tensor(test_dataset.data)
    y_test = torch.tensor(test_dataset.targets)

    train_channel_means = X_train.to(torch.float32).mean(dim=(0, 1, 2)) / 255
    train_channel_std = X_train.to(torch.float32).std(dim=(0, 1, 2)) / 255

    image_transforms = transforms.Compose([
                                            transforms.Normalize(train_channel_means, train_channel_std)
                                          ])
    patch_size = (4, 4)
    n_channels = 3
    # create patches
    X_train_patched = X_train.view(X_train.size(0), -1, patch_size[0], patch_size[1], n_channels)
    


def preprocess_mnist_data(train_dataset, test_dataset):
    """
    A function that preprocesses the MNIST dataset. The preprocessing steps are:
        - Reshaping the image data
        - Min max normalizing the images
        - one-hot encoding the targets.
    """
    X_train = train_dataset.data
    X_test = test_dataset.data
    # extract tensors and reshape
    X_train = X_train.reshape(X_train.size(0), -1)
    X_test = X_test.reshape(X_test.size(0), -1)

    y_train =  train_dataset.targets
    y_test = test_dataset.targets
    n_classes = len(y_train.unique())

    # min max normalize images using train min and max values
    x_max = X_train.max()
    x_min = X_train.min()
    X_train = (X_train - x_max) / (x_max - x_min)
    X_test = (X_test - x_max) / (x_max - x_min)
    
    # one-hot encode y values
    y_train = torch.zeros((y_train.size(0), n_classes)).scatter_(dim=1, index=y_train.unsqueeze(1), value=1)
    y_test = torch.zeros((y_test.size(0), n_classes)).scatter_(dim=1, index=y_test.unsqueeze(1), value=1)
    
    return X_train, y_train, X_test, y_test


class MNISTDataset(Dataset):
    def __init__(self, X: torch.tensor, y: torch.tensor, dataset_name: str, use_indices: bool = True) -> None:
        super().__init__()
        self.X = X
        self.y = y.to(torch.float32)
        self.name = dataset_name
        self.use_indices = use_indices
        self.indices = self.indices = torch.arange(len(self.X)) if use_indices else None

    def __getitem__(self, index) -> tuple:
        if self.use_indices:
            return self.X[index], self.y[index], self.indices[index]
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.size(0)


def get_mnist_unlearn_data(root_dir: str, dataset_name: str, n_forget_points: int, subsample_size: int, batch_size: int, seed: int):
    torch.manual_seed(seed)
    if not os.path.exists(f'{root_dir}/{dataset_name}/processed'):
        # download train & test datasets
        train_dataset, test_dataset = download_dataset(root_dir, dataset_name)
        # reshape, normalize, one-hot encode
        X_train, y_train, X_test, y_test = preprocess_mnist_data(train_dataset, test_dataset)
        # take a subsample of the train set
        X_train, y_train = subsample_train(X_train, y_train, subsample_size)
        print(f'X train dimensions: {X_train.size(0)} x {X_train.size(1)}')
        # save preprocssed tensors
        os.makedirs(f'{root_dir}/{dataset_name}/processed')
        torch.save(X_train, f=f'{root_dir}/{dataset_name}/processed/X_train.pt')
        torch.save(y_train, f=f'{root_dir}/{dataset_name}/processed/y_train.pt')
        torch.save(X_test, f=f'{root_dir}/{dataset_name}/processed/X_test.pt')
        torch.save(y_test, f=f'{root_dir}/{dataset_name}/processed/y_test.pt')
    else:
        X_train = torch.load(f'{root_dir}/{dataset_name}/processed/X_train.pt')
        y_train = torch.load(f'{root_dir}/{dataset_name}/processed/y_train.pt')
        X_test = torch.load(f'{root_dir}/{dataset_name}/processed/X_test.pt')
        y_test = torch.load(f'{root_dir}/{dataset_name}/processed/y_test.pt')

    # create retain-forget split
    X_retain, y_retain, X_forget, y_forget, forget_idxs = split_retain_forget_random(X_train, y_train, n_forget_points=n_forget_points)
    
    print(f'Forget class distribution:\n{y_forget.argmax(dim=-1).unique(return_counts=True)[1].tolist()}')
    # create datasets
    train_dataset = MNISTDataset(X_train, y_train, dataset_name)
    retain_dataset = MNISTDataset(X_retain, y_retain, dataset_name)
    forget_dataset = MNISTDataset(X_forget, y_forget, dataset_name)
    test_dataset = MNISTDataset(X_test, y_test, dataset_name)
    # create dataloaders
    train_loader, retain_loader, forget_loader, test_loader = create_image_dataloaders(train_dataset, retain_dataset, forget_dataset, test_dataset, batch_size)
    return train_loader, retain_loader, forget_loader, test_loader, forget_idxs


def get_unlearn_data(root_dir: str, dataset_name: str, n_forget_points: int, subsample_size: int, batch_size: int, seed: int):
    if dataset_name == 'MNIST':
        get_mnist_unlearn_data(root_dir, dataset_name, n_forget_points, subsample_size, batch_size, seed)
    elif dataset_name == 'CIFAR10':
        pass
    
    else:
        raise NotImplementedError()


def create_image_dataloaders(train_dataset, retain_dataset, forget_dataset, test_dataset, batch_size: int):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, retain_loader, forget_loader, test_loader



if __name__ == '__main__':
    # dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = get_mnist_unlearn_data(n_forget_points=500,
    #                        batch_size=16, seed=42)
    # _x, _y = next(iter(dataloader_train))
    # pdb.set_trace()
    cifar_train, cifar_test = download_cifar_dataset('exp4/data')
    preprocess_cifar_data(cifar_train, cifar_test)
    pdb.set_trace()