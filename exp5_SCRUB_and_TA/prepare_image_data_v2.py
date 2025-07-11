import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.patheffects


def download_dataset(root_dir: str, dataset_name: str):
    if dataset_name == 'MNIST':
        return download_mnist_dataset(root_dir)
    else:
        raise NotImplementedError(f"The requested dataset {dataset_name} is not supported.")


def preprocess_mnist_data(train_data, train_target, test_data, test_target, x_min=None, x_max=None, n_classes=None):
    """
    A function that preprocesses the MNIST dataset. The preprocessing steps are:
        - Reshaping the image data
        - Min max normalizing the images
        - one-hot encoding the targets.
    """
    X_train = train_data
    X_test = test_data
    # extract tensors and reshape
    X_train = X_train.reshape(X_train.size(0), -1)
    X_test = X_test.reshape(X_test.size(0), -1)

    y_train = train_target
    y_test = test_target
    if not n_classes:
        n_classes = len(y_train.unique())

    # min max normalize images using train min and max values
    if not x_min:
        x_min = X_train.min()
    if not x_max:
        x_max = X_train.max()

    X_train = (X_train - x_min) / (x_max - x_min)
    X_test = (X_test - x_min) / (x_max - x_min)
    
    # one-hot encode y values
    y_train = torch.zeros((y_train.size(0), n_classes)).scatter_(dim=1, index=y_train.unsqueeze(1), value=1)
    y_test = torch.zeros((y_test.size(0), n_classes)).scatter_(dim=1, index=y_test.unsqueeze(1), value=1)
    
    return X_train, y_train, X_test, y_test

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


def split_data_by_tsne_box(boundary: dict = None,
                           tsne_params: dict = None,
                           return_tsne_results: bool = False,
                           root_dir: str = './data',
                           dataset_name: str = 'MNIST',
                           subsample_size: int = None,
                           seed: int = 42) -> tuple:
    """
    Splits X_train and y_train based on a t-SNE map of the original dataset.

    First performs t-SNE on the raw `train_dataset`, then uses the boundary box
    to find indices. Finally, it uses these indices to split the transformed
    X_train and y_train tensors.
    
    Args:
        boundary (dict): Dictionary containing boundary box coordinates.
        tsne_params (dict, optional): Parameters for t-SNE. Defaults to standard values.
        return_tsne_results (bool): Whether to return the t-SNE plot coordinates.
    
    Returns:
        tuple: A tuple containing:
               (X_retain, y_retain, X_forget, y_forget, forget_indices)
               or if return_tsne_results is True:
               (X_retain, y_retain, X_forget, y_forget, forget_indices, tsne_results)
    """
    # Default t-SNE parameters
    if tsne_params is None:
        tsne_params = {
            'n_components': 2, 'perplexity': 30, 'max_iter': 300, 'random_state': 42
        }

    if boundary is None:
        boundary = {
            'x_min': 12.2, 'x_max': 12.6, 'y_min': -3.8, 'y_max': -1.5
        }
    
    # === STEP 1: Load the raw dataset ===
    train_dataset, test_dataset = download_dataset(root_dir, dataset_name)
    raw_train_data = train_dataset.data
    x_max = raw_train_data.max()
    x_min = raw_train_data.min()
    n_classes = len(train_dataset.targets.unique())

    if subsample_size is not None:
        # random subsample the data
        torch.manual_seed(seed)
        indices = torch.randperm(len(raw_train_data))[:subsample_size]
        raw_train_data = raw_train_data[indices]
        raw_train_targets = train_dataset.targets[indices]

    # === STEP 2: Run t-SNE on the RAW data ===
    # Ensure raw data is a 2D array for t-SNE (n_samples, n_features)
    raw_data_flat = raw_train_data.reshape(len(raw_train_data), -1)
    
    print(f"Running t-SNE on the original train_dataset with {len(raw_train_data)} samples...")
    tsne = TSNE(**tsne_params)
    tsne_results = tsne.fit_transform(raw_data_flat)
    print("t-SNE finished.")

    # === STEP 3: Preprocess the data to get the tensors you want to split ===
    X_train, y_train, X_test, y_test = preprocess_mnist_data(raw_train_data, raw_train_targets, test_dataset.data, test_dataset.targets, x_min=x_min, x_max=x_max, n_classes=n_classes)
    
    # === STEP 4: Find indices from the t-SNE results ===
    # Create a DataFrame for easier filtering. The index aligns with the original data.
    df_tsne = pd.DataFrame(tsne_results, columns=['tsne-1', 'tsne-2'])
    
    print(f"Finding points inside boundary: {boundary['x_min']} <= x <= {boundary['x_max']}, {boundary['y_min']} <= y <= {boundary['y_max']}")

    # Create a boolean mask to identify points inside the boundary box
    forget_mask = (
        (df_tsne['tsne-1'] >= boundary['x_min']) & 
        (df_tsne['tsne-1'] <= boundary['x_max']) &
        (df_tsne['tsne-2'] >= boundary['y_min']) & 
        (df_tsne['tsne-2'] <= boundary['y_max'])
    )
    
    # Get the original indices for the forget and retain sets
    forget_indices = df_tsne[forget_mask].index.to_numpy()
    retain_indices = df_tsne[~forget_mask].index.to_numpy()
    
    # === STEP 5: Split the PROCESSED X_train and y_train using these indices ===
    X_forget = X_train[forget_indices]
    y_forget = y_train[forget_indices]
    X_retain = X_train[retain_indices]
    y_retain = y_train[retain_indices]
    
    print(f"\nRetain set size: {len(X_retain)}")
    print(f"Forget set size: {len(X_forget)}")
    print("\nLabel distribution in forget set:")
    
    # Analyze the labels of the forgotten data
    y_train_labels = torch.argmax(y_train, dim=1)
    forget_labels = y_train_labels[forget_indices].numpy()
    unique_labels, counts = np.unique(forget_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} samples")


    train_dataset = MNISTDataset(X_train, y_train, dataset_name, use_indices=False)
    retain_dataset = MNISTDataset(X_retain, y_retain, dataset_name, use_indices=False)
    forget_dataset = MNISTDataset(X_forget, y_forget, dataset_name, use_indices=False)
    validation_dataset = MNISTDataset(X_test, y_test, dataset_name, use_indices=False)

    if return_tsne_results:
        return train_dataset, retain_dataset, forget_dataset, validation_dataset, forget_indices, tsne_results
    else:
        return train_dataset, retain_dataset, forget_dataset, validation_dataset, forget_indices


def get_image_unlearn_data(root_dir: str, batch_size: int, seed: int, boundary: dict = None, subsample_size: int = None):
    
    if boundary is None:
        boundary = {
            'x_min': 12.2,
            'x_max': 12.6,
            'y_min': -3.8,
            'y_max': -1.5
        }

    # Make a folder if it does not exist
    coords_string = f"{boundary['x_min']}_{boundary['x_max']}_{boundary['y_min']}_{boundary['y_max']}"
    folder_name = os.path.join(root_dir, f"tsne_box_{coords_string}_{subsample_size}")

    # load data if it exists
    if os.path.exists(os.path.join(folder_name, "data.pt")):
        print(f"Loading data from {folder_name}")
        data = torch.load(os.path.join(folder_name, "data.pt"), weights_only=False)
        return data['train_loader'], data['retain_loader'], data['forget_loader'], data['validation_loader'], data['forget_indices']

    train_dataset, retain_dataset, forget_dataset, test_dataset, forget_indices = split_data_by_tsne_box(
        boundary=boundary,
        return_tsne_results=False,
        subsample_size=subsample_size
    )

    from prepare_image_data import create_image_dataloaders

    train_loader, retain_loader, forget_loader, validation_loader = create_image_dataloaders(train_dataset=train_dataset, 
                                                                               retain_dataset=retain_dataset, 
                                                                               forget_dataset=forget_dataset, 
                                                                               test_dataset=test_dataset, 
                                                                               batch_size=batch_size, 
                                                                               seed=seed)


    # There should be more than 0 forget points
    assert len(forget_indices) > 0, "No forget points found. Make another boundary box."

    # save data to folder in one pt file
    print(f"Saving data to {folder_name}")
    os.makedirs(folder_name, exist_ok=True)
    torch.save({
        'train_loader': train_loader,
        'retain_loader': retain_loader,
        'forget_loader': forget_loader,
        'validation_loader': validation_loader,
        'forget_indices': forget_indices
    }, os.path.join(folder_name, "data.pt"))


    return train_loader, retain_loader, forget_loader, validation_loader, forget_indices

if __name__ == '__main__':
    dataset_name = 'MNIST'
    # rootdire should be the path of the current file
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # ================================ Preprocessing ================================

    # 1. Edge of class
    boundary = {
        'x_min': 10.2,
        'x_max': 11.0,
        'y_min': -4.3,
        'y_max': -0.4
    }


    # 2. In between classes
    # boundary = {
    #     'x_min': -6.20,
    #     'x_max': -3.4,
    #     'y_min': -1.6,
    #     'y_max': 1.0
    # }
    
    # # 3. Two overlapping classes
    # boundary = {
    #     'x_min': -4.2,
    #     'x_max': -2.6,
    #     'y_min': -6.25,
    #     'y_max': -4.8
    # }

    # slice X_train and y_train to 10000 samples
    # X_train = X_train[:1000]
    # y_train = y_train[:1000]
    
    subsample_size=10000
    # Create retain and forget datasets
    train_dataset, retain_dataset, forget_dataset, validation_dataset, forget_indices, tsne_results = split_data_by_tsne_box(
        boundary=boundary,
        return_tsne_results=True,
        subsample_size=subsample_size
    )

    y_train = train_dataset.y
    # convert y_test from onehot to class label
    y_train_labels = torch.argmax(y_train, dim=1)
    # ================================ Plotting ================================
    # Set a paper-friendly style
    plt.style.use('seaborn-v0_8-paper')
    plt.figure(figsize=(8, 7))
    

    colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600', '#aa5382', '#eea152']
    # Create the scatter plot with better styling
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                         c=y_train_labels.numpy(),
                         cmap=ListedColormap(colors),  # Now using the imported ListedColormap
                         alpha=0.85,
                         s=50,  # Slightly larger points
                         edgecolor='white',  # White edges around points
                         linewidth=0.5)
    
    # Add a legend only for the forget points
    forget_marker = plt.Line2D([0], [0], marker='o', color='none',
                              markerfacecolor='red',
                              markeredgecolor='white',
                              markersize=12,
                              label=f'Forget Points ({len(forget_indices)} images)',
                              path_effects=[
                                  matplotlib.patheffects.withStroke(linewidth=3,
                                                                  foreground='red',
                                                                  alpha=0.5)
                              ])
    
    plt.legend(handles=[forget_marker], loc='best', fontsize=18)
    
    # Plot boundary box with improved styling
    plt.gca().add_patch(Rectangle(
        (boundary['x_min'], boundary['y_min']),
        boundary['x_max'] - boundary['x_min'],
        boundary['y_max'] - boundary['y_min'],
        edgecolor='red',
        facecolor='red',
        alpha=0.1,
        lw=2,
        linestyle='--'
    ))



    # encircle forget indices points with a more bubbly appearance
    for i in forget_indices:
        # Add a larger outer circle for the bubble effect
        plt.scatter(tsne_results[i, 0], tsne_results[i, 1],
                    c='none', alpha=0.5, s=150,  # Larger size for outer circle
                    edgecolor='red', linewidth=2)  # Red border
        # Add the inner point
        plt.scatter(tsne_results[i, 0], tsne_results[i, 1],
                    c='red', alpha=0.6, s=80,  # Slightly larger than regular points
                    edgecolor='white', linewidth=0.5)
    
    # Add labels and title with better formatting
    plt.xlabel('t-SNE Dimension 1', fontsize=18)
    plt.ylabel('t-SNE Dimension 2', fontsize=18)
    plt.title("t-SNE Visualization of MNIST Dataset", 
              fontsize=22, pad=20)
    
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # Adjust layout to prevent cutting off elements
    plt.tight_layout()
    
    # plt.show()
    # save to pdf
    boundary_name = f'{boundary["x_min"]}_{boundary["x_max"]}_{boundary["y_min"]}_{boundary["y_max"]}'
    plots_folder_name = os.path.join(root_dir, f"plots/data/dataset_forget_retain_{boundary_name}_{subsample_size}")
    os.makedirs(plots_folder_name, exist_ok=True)
    plt.savefig(os.path.join(plots_folder_name, f'dataset_forget_retain_{boundary_name}_{subsample_size}.pdf'), bbox_inches='tight')
    # also save as png
    plt.savefig(os.path.join(plots_folder_name, f'dataset_forget_retain_{boundary_name}_{subsample_size}.png'), bbox_inches='tight')



    # ================================ Bar Plot of Forget Set Distribution ================================
    plt.figure(figsize=(8, 7))
    forget_labels = y_train_labels[forget_indices].numpy()
    n_classes = y_train.shape[1]
    
    # Ensure we have counts for all classes from 0 to n_classes-1
    unique_labels, counts = np.unique(forget_labels, return_counts=True)
    full_counts = np.zeros(n_classes, dtype=int)
    full_counts[unique_labels] = counts
    class_labels = np.arange(n_classes)

    # Create a DataFrame for easier plotting with seaborn
    df_forget_dist = pd.DataFrame({'Class': class_labels, 'Count': full_counts})

    sns.barplot(x='Class', y='Count', data=df_forget_dist, palette=colors, hue='Class', dodge=False, legend=False)

    plt.xlabel('Class Label', fontsize=18)
    plt.ylabel('Number of Samples', fontsize=18)
    plt.title('Distribution of Classes in Forget Set', fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    # Save the bar plot
    barplot_filename_pdf = os.path.join(plots_folder_name, f'forget_set_distribution_{boundary_name}_{subsample_size}.pdf')
    plt.savefig(barplot_filename_pdf, bbox_inches='tight')
    
    barplot_filename_png = os.path.join(plots_folder_name, f'forget_set_distribution_{boundary_name}_{subsample_size}.png')
    plt.savefig(barplot_filename_png, bbox_inches='tight')

    print(f"\nSaved forget set distribution bar plot to {barplot_filename_pdf} and {barplot_filename_png}")



    