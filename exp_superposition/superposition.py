# Some of the code is borrowed from https://github.com/zroe1/toy-models-of-superposition/blob/main/demonstrating_superposition/demonstrating_superposition.ipynb
# %%
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim 
import os
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


def generate_synthetic_data(n_samples, n_features, sparsity, importance_list=None, device='cpu'):
    """
    Generates synthetic data as described in the "Toy Models of Superposition" paper.

    Args:
        n_samples (int): The number of samples to generate.
        n_features (int): The number of features (dimensions) in the input vector.
        sparsity (float): The sparsity parameter (probability of a feature being 0). They all have the same sparsity.
        importance_list (list, optional): A list of importance values for each feature. 
                                         If None, all features have equal importance.
                                         Defaults to None.

    Returns:
        torch.Tensor: A 2D tensor representing the synthetic data matrix.
    """
    prob = 1 - sparsity

    # Generate a sparsity mask for all samples at once
    sparsity_tensor = torch.bernoulli(torch.full((n_samples, n_features), prob, device=device))

    # Generate random values for all samples
    x = torch.rand(n_samples, n_features, device=device)

    # Apply the sparsity mask
    x = x * sparsity_tensor

    # If importance_list is not provided, use all ones
    if importance_list is None:
        importance_list = torch.ones(n_features, device=device)

    return x, importance_list

class ToyModel(nn.Module):
    def __init__(self, m, n, include_ReLU, device):
        '''Create a toy model

        Args:
            m (int): the number of neurons (as described in original paper)
            n (int): the number of features the Toy model can map.
            (The weight matrix is declared to be m * n)

            include_ReLU (bool): if True, a nonlinearity is added to the network
        '''
        super().__init__()
        # Initialize parameters on CPU
        self.W = nn.Parameter(torch.randn(m, n), requires_grad=True)
        self.b = nn.Parameter(torch.randn(n, 1), requires_grad=True)

        self.m = m
        self.n = n

        # Move the entire model to the specified device
        self.to(device)

        self.ReLU = nn.ReLU(inplace=True)
        self.include_ReLU = include_ReLU
        
    def forward(self, x): # x is 5 * 1
        h = self.W @ x
        x_prime = self.W.T @ h + self.b

        if self.include_ReLU:
            return self.ReLU(x_prime)
        else:
            return x_prime

class MSEImportance(nn.Module):
    def __init__(self):
        super(MSEImportance, self).__init__()

    def forward(self, predictions, targets, importance):
        sub_total = ((predictions - targets)**2).sum(0).flatten()
        return sum(sub_total * importance)
    

def train_model(model, dataloader, optimizer, loss_fn, n_epochs, importance_weights=None):
    model.train()
    losses = []
    with tqdm(total=n_epochs, desc="Training Progress") as pbar:
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(dataloader):
                x = batch[0]
                # reshape x from batch_size * n_features to batch_size * n_features * 1
                x = x.reshape(-1, model.n, 1)
                pred = model(x)
                loss = loss_fn(x, pred, importance_weights)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            # Update the progress bar with the average loss
            pbar.set_postfix(loss=avg_loss)
            pbar.update(1)
    return losses

def visualize_sparse_vector_1d(vector, sparsity):
    """Visualizes a sparse vector as a line and stem plot."""

    n_features = len(vector)
    plt.figure(figsize=(10, 4))

    # Line Plot
    plt.subplot(1, 2, 1)
    plt.plot(vector, marker='o', linestyle='-')
    plt.title(f'Line Plot (Sparsity = {sparsity})')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')

    # Stem Plot
    plt.subplot(1, 2, 2)
    plt.stem(range(n_features), vector, markerfmt='ro', basefmt=' ')  # ' ' removes the baseline
    plt.title(f'Stem Plot (Sparsity = {sparsity})')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')

    plt.tight_layout()
    plt.show()

def visualize_sparse_matrix(data_matrix, sparsity):
    """Visualizes a matrix of sparse vectors as a heatmap."""
    data_matrix = data_matrix.cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(data_matrix, cmap='viridis', aspect='auto')  # 'viridis' colormap
    plt.colorbar(label='Feature Value')
    plt.title(f'Sparse Data Matrix (Sparsity = {sparsity})')
    plt.xlabel('Feature Index')
    plt.ylabel('Data Example Index')
    plt.tight_layout()
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_script_dir, f'sparse_matrix_{sparsity}.png'))
    plt.savefig(os.path.join(current_script_dir, f'sparse_matrix_{sparsity}.pdf'))

def graph_weights(weights, bias):
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5)) # 1 row, 2 columns
    
    w = weights.clone().cpu().detach()
    to_graph = w.T @ w
    colors = [(.4, 0, 1), (1, 1, 1), (1, .4, 0)]  # Purple -> White -> Orange
    n_bins = 100 
    cm = LinearSegmentedColormap.from_list("", colors, N=n_bins)
    
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    
    axs[0].imshow(to_graph, cmap=cm, norm=norm)
    # ax = plt.gca()
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    graph_biases(bias, axs[1])
    plt.subplots_adjust(left=0.0, right=1.4)
    plt.tight_layout()
    return plt

def graph_biases(bias, ax_obj):
    b = bias.clone().detach().cpu()
    colors = [(.4, 0, 1), (1, 1, 1), (1, .4, 0)]  # Purple -> White -> Orange
    n_bins = 100 
    cm = LinearSegmentedColormap.from_list("", colors, N=n_bins)
    
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    
    ax_obj.imshow(b, cmap=cm, norm=norm)

    ax_obj.set_xticks([])
    ax_obj.set_yticks([])

# %%

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    n_features = 20
    n_samples = 512000
    n_epochs = 10
    sparsity = 0.0

    importance_list = (0.7 ** torch.arange(0, n_features)).to(device)
    x, importance_list = generate_synthetic_data(n_samples=n_samples, n_features=n_features, sparsity=sparsity, device=device, importance_list=importance_list)  # Assuming you have the generate_synthetic_data function
    # Visualize the data with heatmap
    visualize_sparse_matrix(x, sparsity)

    dataset = TensorDataset(x)  # Create a dataset with only the input data

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    
    model = ToyModel(m=5, n=n_features, include_ReLU=False, device=device)
    loss_fn = MSEImportance()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses = train_model(model, dataloader, optimizer, loss_fn, n_epochs=n_epochs, importance_weights=importance_list)

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    plt = graph_weights(model.W, model.b)
    # Add a title to the plot
    plt.suptitle('Visualization of W^T W and b \n Linear Model with 0.0 sparsity in data')
    plt.tight_layout()
    plt.savefig(os.path.join(current_script_dir, f'linear_model_weights.png'))
    plt.savefig(os.path.join(current_script_dir, f'linear_model_weights.pdf'))
    plt.close()

    # model_relu = ToyModel(m=5, n=n_features, include_ReLU=True, device=device)
    # optimizer_relu = optim.Adam(model_relu.parameters(), lr=1e-3)
    # loss_fn_relu = MSEImportance()
    # relu_losses = train_model(model_relu, dataloader, optimizer_relu, loss_fn_relu, n_epochs=n_epochs, importance_weights=importance_list)
    # plt = graph_weights(model_relu.W, model_relu.b)
    # plt.suptitle('Visualization of W^T W and b \\ ReLU Model with 0.0 sparsity in data')
    # plt.savefig(os.path.join(current_script_dir, f'relu_model_weights.png'))
    # plt.savefig(os.path.join(current_script_dir, f'relu_model_weights.pdf'))
    # plt.close()
    # print()


