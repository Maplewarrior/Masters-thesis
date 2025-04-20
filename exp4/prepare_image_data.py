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
    
    X_retain = X_train[~forget_mask]
    y_retain = y_train[~forget_mask]
    X_forget = X_train[forget_mask]
    y_forget = y_train[forget_mask]
    return X_retain, y_retain, X_forget, y_forget, forget_idxs


def subsample_train(X_train, y_train, subsample_size: int = None):
    idxs = torch.randperm(X_train.size(0))[:subsample_size]
    return X_train[idxs], y_train[idxs]


def patch_sanity_check(X_train, image_size: int, patch_size: tuple[int], n_channels):
    n_h = image_size[0] // patch_size[0]
    n_w = image_size[1] // patch_size[1]
    n_patches = int(n_h * n_w)
    # create patches
    ### The slow approach
    x_true = X_train[-2]
    x_patched = torch.zeros((n_patches, patch_size[0], patch_size[1], n_channels))
    
    for i in range(n_h):
        h_start = int(i * patch_size[0])
        h_end = int((i+1) * patch_size[0])
        for j in range(n_w):
            w_start = int(j * patch_size[1])
            w_end = int((j+1) * patch_size[1])
            x_patched[i * n_h + j] = x_true[h_start:h_end, w_start:w_end, :]
        
    ### The fast approach
    x_patch = x_true.reshape(n_h, patch_size[0], n_w, patch_size[1], n_channels)
    x_patch = x_patch.permute(0, 2, 1, 3, 4).reshape(n_patches, patch_size[0], patch_size[1], n_channels)

    assert x_patch == x_patched, 'The vectorized reshaping is incorrect!'


def preprocess_cifar_data(train_dataset, test_dataset, patch_size: tuple[int]):
    X_train = torch.tensor(train_dataset.data).to(torch.float32)
    y_train = torch.tensor(train_dataset.targets)
    X_test = torch.tensor(test_dataset.data).to(torch.float32)
    y_test = torch.tensor(test_dataset.targets)

    # compute dataset properties
    n_classes = len(y_train.unique())
    img_size = (X_train.size(1), X_train.size(2))
    n_channels = X_train.size(3)

    n_h = img_size[0] // patch_size[0]
    n_w = img_size[1] // patch_size[1]
    n_patches = n_h * n_w
    
    train_channel_mean = X_train.mean(dim=(0, 1, 2))
    train_channel_std = X_train.std(dim=(0, 1, 2))

    # normalize train and test images
    X_train = (X_train - train_channel_mean) / train_channel_std
    X_test = (X_test - train_channel_mean) / train_channel_std

    # reshape data to patches
    ### Dimension mapping: [n_images x H x W x C] --> [n_images x n_patches x patch_h x patch_w x C]
    X_train = X_train.reshape(X_train.size(0), n_h, patch_size[0], n_w, patch_size[1], n_channels)
    X_train = X_train.permute(0, 1, 3, 2, 4, 5).reshape(X_train.size(0), n_patches, patch_size[0], patch_size[1], n_channels)

    X_test = X_test.reshape(X_test.size(0), n_h, patch_size[0], n_w, patch_size[1], n_channels)
    X_test = X_test.permute(0, 1, 3, 2, 4, 5).reshape(X_test.size(0), n_patches, patch_size[0], patch_size[1], n_channels)

    # patch_sanity_check(X_train, img_size, patch_size, n_channels)
    
    # reshape [n_images x n_patches x patch_h x patch_w x C] --> [n_images x n_paches x patch_h * patch_w * C]
    X_train = X_train.reshape(X_train.size(0), n_patches, -1)
    X_test = X_test.reshape(X_test.size(0), n_patches, -1)

    # one-hot encode y values
    y_train = torch.zeros((y_train.size(0), n_classes)).scatter_(dim=1, index=y_train.unsqueeze(1), value=1)
    y_test = torch.zeros((y_test.size(0), n_classes)).scatter_(dim=1, index=y_test.unsqueeze(1), value=1)
    
    return X_train, y_train, X_test, y_test

    # X_train_patched = X_train.contiguous().view(X_train.size(0), -1, patch_size[0], patch_size[1], n_channels)
    # import matplotlib.pyplot as plt
    
    
    # # plt.imshow(X_train[-2].numpy())
    # # plt.show()
    # x_true = X_train[-2]

    # x_rec = torch.zeros_like(X_train[-2])
    # x_patch = x_true.reshape(n_patches, patch_size[0], patch_size[1], n_channels)


    # pdb.set_trace()
    # for i in range(n_patches):
        

    #     x_rec[i]

    
    # plt.show()

    # pdb.set_trace()
    # plt.close()


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

class CIFARDataset(Dataset):
    def __init__(self, X, y, dataset_name: str, use_indices: bool = True) -> None:
        super().__init__()
        self.X = X
        self.y = y
        self.name = dataset_name
        self.use_indices = use_indices
        self.indices = self.indices = torch.arange(len(self.X)) if use_indices else None

    def __getitem__(self, index):
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

def get_cifar_unlearn_data(root_dir: str, dataset_name: str, n_forget_points: int, subsample_size: int, patch_size: tuple[int], batch_size: int, seed: int):
    torch.manual_seed(seed)
    if not os.path.exists(f'{root_dir}/{dataset_name}/processed'):
        train_dataset, test_dataset = download_dataset(root_dir, dataset_name)
        # normalize data, create patches & one-hot encode y values
        X_train, y_train, X_test, y_test = preprocess_cifar_data(train_dataset, test_dataset, patch_size=patch_size)
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

    
    # randomly sample forget points
    X_retain, y_retain, X_forget, y_forget, forget_idxs = split_retain_forget_random(X_train, y_train, n_forget_points)

    train_dataset = CIFARDataset(X_train, y_train, dataset_name, use_indices=True)
    retain_dataset = CIFARDataset(X_retain, y_retain, dataset_name, use_indices=True)
    forget_dataset = CIFARDataset(X_forget, y_forget, dataset_name, use_indices=True)
    test_dataset = CIFARDataset(X_test, y_test, dataset_name, use_indices=True)

    train_loader, retain_loader, forget_loader, test_loader = create_image_dataloaders(train_dataset, retain_dataset, forget_dataset, test_dataset, batch_size)
    return train_loader, retain_loader, forget_loader, test_loader, forget_idxs
    
def get_unlearn_data(root_dir: str, dataset_name: str, n_forget_points: int, subsample_size: int, patch_size: tuple[int], batch_size: int, seed: int):
    if dataset_name == 'MNIST':
        return get_mnist_unlearn_data(root_dir, dataset_name, n_forget_points, subsample_size, batch_size, seed)
    elif dataset_name == 'CIFAR10':
        return get_cifar_unlearn_data(root_dir, dataset_name, n_forget_points, subsample_size, patch_size, batch_size, seed)
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
    # cifar_train, cifar_test = download_cifar_dataset('exp4/data')
    # preprocess_cifar_data(cifar_train, cifar_test)
    root_dir = os.path.join(os.path.dirname(__file__), 'data')
    train_loader, retain_loader, forget_loader, test_loader, forget_idxs = get_unlearn_data(root_dir, 'CIFAR10', 150, None, (4, 4), 16, 42)
    from src.models.vision_transformer import ViT
    d_patch = 4 * 4 * 3
    model = ViT(d_patch=d_patch, d_hidden=128, d_ff=256, d_k = 36, n_layers=3, n_heads=4, n_classes=10, n_patches=64, tau=6, img_size=(32,32), patch_size=(4,4))
    x, y, _ = next(iter(train_loader))
    out = model(x)

