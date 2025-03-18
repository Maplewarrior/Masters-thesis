import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data_utils.synthetic_data import SyntheticDataset

def create_outlier_clusters(X, y, class_idx, n_clusters=2, n_outliers_per_cluster=10, 
                           outlier_distance=5.0, outlier_variance=0.5, angle_separation=None, 
                           random_state=None):
    """
    Creates multiple outlier clusters for a specific class by moving subsets of points away 
    from the class center in different directions.
    
    Args:
        X: Feature matrix
        y: Labels
        class_idx: Index of the class to create outliers for
        n_clusters: Number of outlier clusters to create
        n_outliers_per_cluster: Number of outlier points per cluster
        outlier_distance: Distance to move the outlier clusters from the class center
        outlier_variance: Variance of the outlier clusters
        angle_separation: Angle separation between clusters in degrees (only for 2D data)
                         If None, random directions will be used
        random_state: Random state for reproducibility
        
    Returns:
        X_with_outliers: Updated feature matrix with outliers
        y_with_outliers: Updated labels
        outlier_indices: List of indices of the outlier points for each cluster
    """
    # Set random seed for reproducibility
    rng = np.random.RandomState(random_state)
    
    # Make a copy of the data
    X_with_outliers = X.copy()
    y_with_outliers = y.copy()
    
    # Get indices of the specified class
    class_indices = np.where(y == class_idx)[0]
    
    # Calculate total outliers needed
    total_outliers = n_clusters * n_outliers_per_cluster
    
    # Ensure we have enough points in this class
    if len(class_indices) < total_outliers:
        raise ValueError(f"Class {class_idx} has only {len(class_indices)} points, but {total_outliers} outliers were requested")
    
    # Calculate the center of the class
    class_center = np.mean(X[class_indices], axis=0)
    
    # Randomly select points to become outliers (without replacement)
    all_outlier_indices = rng.choice(class_indices, size=total_outliers, replace=False)
    
    # Split into clusters
    outlier_indices_by_cluster = np.array_split(all_outlier_indices, n_clusters)
    all_outlier_indices = []  # Reset to collect all indices
    
    # Generate directions for each cluster
    if angle_separation is not None and X.shape[1] == 2:
        # For 2D data with specified angle separation
        base_angle = rng.uniform(0, 360)
        directions = []
        for i in range(n_clusters):
            angle = base_angle + i * (angle_separation % 360)
            angle_rad = np.radians(angle)
            direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            directions.append(direction)
    else:
        # Random directions for higher dimensions or when angle_separation is None
        directions = []
        for _ in range(n_clusters):
            direction = rng.randn(X.shape[1])
            direction = direction / np.linalg.norm(direction)  # Normalize to unit vector
            directions.append(direction)
    
    # Create each cluster
    for cluster_idx, (cluster_indices, direction) in enumerate(zip(outlier_indices_by_cluster, directions)):
        # Move the selected points in the cluster's direction
        for idx in cluster_indices:
            # Move the point away from class center in the direction
            X_with_outliers[idx] = class_center + direction * outlier_distance
            
            # Add some random noise to create a cluster rather than a single point
            X_with_outliers[idx] += rng.randn(X.shape[1]) * outlier_variance
        
        # Add these indices to our overall list
        all_outlier_indices.extend(cluster_indices)
    
    return X_with_outliers, y_with_outliers, np.array(all_outlier_indices)

def plot_data(X, y, outlier_indices=None, title=None, savefig=None):
    """
    Plot the data with different colors for each class and highlight outliers.
    
    Args:
        X: Feature matrix
        y: Labels
        outlier_indices: Indices of outlier points to highlight
        title: Plot title
        savefig: Path to save the figure (if None, the figure is not saved)
    """
    plt.figure(figsize=(10, 8))
    
    # Get unique classes
    classes = np.unique(y)
    
    # Create colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    
    # Plot each class
    for i, cls in enumerate(classes):
        plt.scatter(X[y == cls, 0], X[y == cls, 1], 
                   color=colors[i], label=f"Class {cls}", 
                   alpha=0.7, edgecolors='k', linewidths=0.5)


    # Highlight outliers if provided
    if outlier_indices is not None:
        outlier_class = np.unique(y[outlier_indices])[0]

        plt.scatter(X[outlier_indices, 0], X[outlier_indices, 1], 
                   color=colors[outlier_class], marker='x', s=100, label=f"Outliers for class {outlier_class}", 
                   alpha=1.0, linewidths=2)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if title:
        plt.title(title)
    else:
        plt.title("Synthetic Classification Data")
        
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if savefig:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(savefig), exist_ok=True)
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {savefig}")
    else:
        plt.show()

def knn_outlier_detection(X, y, k=5, percentile=95, visualize=True):
    """
    Detects outliers using k-Nearest Neighbors (k-NN) distance.

    Parameters:
    - X: np.ndarray, shape (n_samples, n_features)
        The dataset (features only).
    - y: np.ndarray, shape (n_samples,)
        The labels corresponding to X.
    - k: int, optional (default=5)
        The number of nearest neighbors to consider.
    - percentile: float, optional (default=95)
        The percentile threshold for determining outliers.
    - visualize: bool, optional (default=True)
        Whether to visualize the results (works only for 2D datasets).

    Returns:
    - outliers: np.ndarray, shape (n_samples,)
        Boolean mask indicating outliers (True for outliers, False for normal points).
    - k_distances: np.ndarray, shape (n_samples,)
        The k-th nearest neighbor distances for each point.
    """
    # Fit k-NN model
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X)

    # Compute k-NN distances
    distances, _ = nbrs.kneighbors(X)
    k_distances = distances[:, -1]  # k-th nearest neighbor distance

    # Compute outlier threshold
    threshold = np.percentile(k_distances, percentile)

    # Identify outliers
    outliers = k_distances > threshold

    # Visualization (only for 2D data)
    if visualize and X.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        
        # Get unique classes
        classes = np.unique(y)
        
        # Create a colormap
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, len(classes)))
        
        # Plot each class with its own color
        for i, cls in enumerate(classes):
            # Plot normal points for this class
            mask = (y == cls) & (~outliers)
            plt.scatter(X[mask, 0], X[mask, 1], 
                       color=colors[i], label=f"Class {cls}", 
                       alpha=0.7, edgecolors="k", linewidths=0.5)
            
            # Plot outliers for this class with the same color but 'x' marker
            outlier_mask = (y == cls) & outliers
            if np.any(outlier_mask):
                plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], 
                           color=colors[i], marker="x", s=100, 
                           label=f"Outliers (Class {cls})", 
                           alpha=1.0, linewidths=2)
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(f"k-NN Outlier Detection (k={k}, {percentile}th percentile)")
        plt.show()
        
        # Also create a heatmap of the k-distances
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=k_distances, 
                             cmap="coolwarm", s=50, alpha=0.8, 
                             edgecolors="k", linewidths=0.5)
        plt.colorbar(scatter, label="k-NN Distance")
        plt.title(f"k-NN Distance Heatmap (k={k})")
        plt.grid(True, alpha=0.3)
        plt.show()

    return outliers, k_distances

def split_data(X, y, outlier_indices, train_size=0.7, val_size=0.15, test_size=0.15, random_state=None):
    """
    Split data into train, validation, and test sets while keeping track of outlier indices.
    
    Args:
        X: Feature matrix
        y: Labels
        outlier_indices: Indices of outlier points
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
        
    Returns:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
        outlier_indices_train: Outlier indices in training set
        outlier_indices_val: Outlier indices in validation set
        outlier_indices_test: Outlier indices in test set
    """
    # Set random seed for reproducibility
    rng = np.random.RandomState(random_state)
    
    # Create a boolean mask for outliers
    n_samples = len(X)
    is_outlier = np.zeros(n_samples, dtype=bool)
    is_outlier[outlier_indices] = True
    
    # First split: train vs (val+test)
    X_train, X_temp, y_train, y_temp, is_outlier_train, is_outlier_temp = train_test_split(
        X, y, is_outlier, 
        test_size=(val_size + test_size),
        random_state=rng,
        stratify=y  # Maintain class distribution
    )
    
    # Second split: val vs test
    # Adjust the test_size to get the correct proportion
    temp_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test, is_outlier_val, is_outlier_test = train_test_split(
        X_temp, y_temp, is_outlier_temp,
        test_size=temp_test_size,
        random_state=rng,
        stratify=y_temp  # Maintain class distribution
    )
    
    # Get the indices of outliers in each split
    outlier_indices_train = np.where(is_outlier_train)[0]
    outlier_indices_val = np.where(is_outlier_val)[0]
    outlier_indices_test = np.where(is_outlier_test)[0]
    
    # Print split statistics
    print(f"Data split statistics:")
    print(f"  Total samples: {n_samples}")
    print(f"  Total outliers: {np.sum(is_outlier)}")
    print(f"  Train set: {len(X_train)} samples, {len(outlier_indices_train)} outliers")
    print(f"  Validation set: {len(X_val)} samples, {len(outlier_indices_val)} outliers")
    print(f"  Test set: {len(X_test)} samples, {len(outlier_indices_test)} outliers")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, outlier_indices_train, outlier_indices_val, outlier_indices_test

def main():
    # Data generation parameters
    n_samples = 1000
    n_features = 2
    n_informative = 2
    n_redundant = 0
    n_clusters_per_class = 1
    n_classes = 4
    random_state = 40
    
    # Outlier parameters
    outlier_class = 2  # Class to create outliers for (0-indexed)
    n_clusters = 4     # Number of outlier clusters
    n_outliers_per_cluster = 8  # Number of outliers per cluster
    outlier_distance = 4.0  # Distance to move outliers
    outlier_variance = 0.5  # Variance of the outlier cluster
    angle_separation = 120  # Angle between clusters in degrees (for 2D data)
    
    # Generate base data
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_informative, 
        n_redundant=n_redundant, 
        n_clusters_per_class=n_clusters_per_class, 
        random_state=random_state, 
        n_classes=n_classes
    )
    
    # Plot original data
    # plot_data(X, y, title="Original Data")
    
    # Create outlier clusters
    X_with_outliers, y_with_outliers, outlier_indices = create_outlier_clusters(
        X, y, 
        class_idx=outlier_class,
        n_clusters=n_clusters,
        n_outliers_per_cluster=n_outliers_per_cluster,
        outlier_distance=outlier_distance,
        outlier_variance=outlier_variance,
        angle_separation=angle_separation,
        random_state=random_state
    )


    folder = f"data/synthetic_data"
    os.makedirs(folder, exist_ok=True)

    # convert to tensors
    X_with_outliers = torch.from_numpy(X_with_outliers).float()
    y_with_outliers = torch.from_numpy(y_with_outliers).long()
    outlier_indices = torch.from_numpy(outlier_indices).long()

    # Save dataset X_with_outliers and y_with_outliers and outlier_indices with torch
    torch.save({
        "X_with_outliers": X_with_outliers,
        "y_with_outliers": y_with_outliers,
        "outlier_indices": outlier_indices
    }, f"{folder}/synthetic_ood_data_outlier_class_{outlier_class}_seed_{random_state}.pt")


    # # outlier_detection
    # outliers, k_distances = knn_outlier_detection(X_with_outliers, 
    #                                               y_with_outliers, 
    #                                               k=5, 
    #                                               percentile=95, 
    #                                               visualize=False)


    # # Create a boolean mask for all points
    # all_indices = np.arange(len(X_with_outliers))
    # # Get indices of non-outlier points
    # not_outlier_indices = np.setdiff1d(all_indices, outlier_indices)
    
    # # Extract outlier and non-outlier points
    # X_outliers = X_with_outliers[outlier_indices]
    # y_outliers = y_with_outliers[outlier_indices]
    
    # X_not_outliers = X_with_outliers[not_outlier_indices]
    # y_not_outliers = y_with_outliers[not_outlier_indices]
    

    # # plot_data(X_not_outliers, y_not_outliers, title="In Distribution Points")
    # # plot_data(X_outliers, y_outliers, title="Outlier Points")

def create_data_loaders(X, y, outlier_indices, batch_size=32, onehot_labels=False, use_indices=False, device="cpu"):
    # convert to tensors
    X = X.float()
    y = y.long()
    outlier_indices = outlier_indices.long()
    
    # Split data into train, val, test
    X_train, X_val, X_test, \
        y_train, y_val, y_test, \
            outlier_indices_train, \
                outlier_indices_val, \
                    outlier_indices_test = split_data(X, y, outlier_indices, random_state=40)

    # Create DataLoaders
    full_train_dataset = SyntheticDataset(X_train, y_train, onehot_labels=onehot_labels, use_indices=use_indices, device=device)
    full_val_dataset = SyntheticDataset(X_val, y_val, onehot_labels=onehot_labels, use_indices=use_indices, device=device)
    full_test_dataset = SyntheticDataset(X_test, y_test, onehot_labels=onehot_labels, use_indices=use_indices, device=device)

    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
    full_val_loader = DataLoader(full_val_dataset, batch_size=batch_size, shuffle=False)
    full_test_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False)



    train_indices = np.arange(len(X_train))
    forget_indices_train = np.where(outlier_indices_train)[0]
    retain_indices_train = np.setdiff1d(train_indices, outlier_indices_train)

    X_train_retain = X_train[retain_indices_train]
    y_train_retain = y_train[retain_indices_train]
    X_train_forget = X_train[forget_indices_train]
    y_train_forget = y_train[forget_indices_train]

    retain_dataset = SyntheticDataset(X_train_retain, y_train_retain, onehot_labels=onehot_labels, use_indices=use_indices, device=device)
    forget_dataset = SyntheticDataset(X_train_forget, y_train_forget, onehot_labels=onehot_labels, use_indices=use_indices, device=device)

    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)


    return {
        "train_full_loader": full_train_loader,
        "val_loader": full_val_loader,
        "test_loader": full_test_loader,
        "train_retain_loader": retain_loader,
        "train_forget_loader": forget_loader,
        "forget_idx_in_train": forget_indices_train,
        "retain_idx_in_train": retain_indices_train
    }

class DataGenerator:
    def __init__(self, random_state=None):
        self.random_state = random_state

        # just load the data
        data = torch.load("data/synthetic_data/synthetic_ood_data_outlier_class_2_seed_40_forget_id.pt")
        self.X_with_outliers = data["X_with_outliers"]
        self.y_with_outliers = data["y_with_outliers"]
        self.outlier_indices = data["outlier_indices"]
    
    def draw_forget_set(self, *args, **kwargs):
        # Just do nothing. Makes it fit in the existing framework.
        pass

if __name__ == "__main__":
    # main()


    # load data from file 
    data = torch.load("data/synthetic_data/synthetic_ood_data_outlier_class_2_seed_40.pt")
    X_with_outliers = data["X_with_outliers"]
    y_with_outliers = data["y_with_outliers"]
    outlier_indices = data["outlier_indices"]


    # pick some forget indices that are not outliers within the outlier class
    forget_indices = np.where(y_with_outliers == 2)[0]
    forget_indices = np.setdiff1d(forget_indices, outlier_indices)

    # pick len(outlier_indices) random indices from forget_indices
    forget_indices = np.random.choice(forget_indices, size=len(outlier_indices), replace=False)


    outlier_indices = forget_indices

    #Save data to neew file
    torch.save({
        "X_with_outliers": X_with_outliers,
        "y_with_outliers": y_with_outliers,
        "outlier_indices": outlier_indices
    }, f"data/synthetic_data/ood_data/synthetic_ood_data_outlier_class_2_seed_40_forget_id.pt")


    # Split data into train, val, test, still keep track of outlier indices
    X_train, X_val, X_test, \
        y_train, y_val, y_test, \
            outlier_indices_train, \
                outlier_indices_val, \
                    outlier_indices_test = split_data(X_with_outliers, 
                                                    y_with_outliers, 
                                                    outlier_indices, 
                                                    random_state=40)


    # plot train data with outliers
    plot_data(X_train, 
              y_train, 
              outlier_indices=outlier_indices_train, 
              title="Train Data with Outliers", 
              savefig=f"data/synthetic_data/ood_data/train_data_with_outliers_forget_id.png")

    plot_data(X_val, 
              y_val, 
              outlier_indices=outlier_indices_val, 
              title="Validation Data with Outliers",
              savefig=f"data/synthetic_data/ood_data/val_data_with_outliers_forget_id.png")

    plot_data(X_test, 
              y_test, 
              outlier_indices=outlier_indices_test, 
              title="Test Data with Outliers",
              savefig=f"data/synthetic_data/ood_data/test_data_with_outliers_forget_id.png")
