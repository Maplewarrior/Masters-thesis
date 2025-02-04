from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb

from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (numpy.ndarray): Features of shape (num_samples, num_features).
            y (numpy.ndarray): Labels of shape (num_samples,).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloaders(data_generator, batch_size=32):
    """
    Create PyTorch DataLoaders for the full dataset, retain set, and forget set.

    Args:
        data_generator (DataGenerator): Instance of the DataGenerator.
        batch_size (int): Batch size for the DataLoaders.

    Returns:
        dict: A dictionary containing the DataLoaders.
    """
    print(f"Creating dataloaders for synthetic data with batch size {batch_size}")

    # Full dataset
    full_dataset = SyntheticDataset(data_generator.X, data_generator.y)
    print(f"  Full dataset size: {len(full_dataset)}")
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    # Retain set
    if data_generator.retain_X is not None and data_generator.retain_y is not None:
        retain_dataset = SyntheticDataset(data_generator.retain_X, data_generator.retain_y)
        retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
        print(f"  Retain dataset size: {len(retain_dataset)}")
    else:
        retain_loader = None

    # Forget set
    if data_generator.forget_X is not None and data_generator.forget_y is not None:
        forget_dataset = SyntheticDataset(data_generator.forget_X, data_generator.forget_y)
        forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)
        print(f"  Forget dataset size: {len(forget_dataset)}")
    else:
        forget_loader = None

    return {
        "full_loader": full_loader,
        "retain_loader": retain_loader,
        "forget_loader": forget_loader
    }


class DataGenerator:
    def __init__(self, random_state=None):
        self.X = None
        self.y = None
        self.random_state = random_state
        self.forget_X = None
        self.forget_y = None
        self.retain_X = None
        self.retain_y = None
        self.outlier_idx = None

    def generate_data(self, n_samples=5000, n_features=2, n_informative=2, n_redundant=0, 
                      n_clusters_per_class=1, n_classes=4, n_outliers=100, outlier_scale=2.0, outlier_variance=0.2, outlier_class=None):
        self.X, self.y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, 
                           n_redundant=n_redundant, n_clusters_per_class=n_clusters_per_class, random_state=self.random_state, n_classes=n_classes)
        

        self.make_outliers(n_outliers=n_outliers, scale=outlier_scale, scale_variance=outlier_variance, class_idx=outlier_class)

        print("Data generated successfully with the following properties:")
        print(f"   Number of samples: {n_samples}")
        print(f"   Number of features: {n_features}")
        print(f"   Number of informative features: {n_informative}")
        print(f"   Number of redundant features: {n_redundant}")
        print(f"   Number of clusters per class: {n_clusters_per_class}")
        print(f"   Number of classes: {n_classes}")
        print(f"   Number of outliers: {n_outliers}")


    def make_outliers(self, n_outliers=100, scale=2.0, scale_variance=0.2, class_idx=None):
        """Moves n_outliers points away from their class centroids with varied displacement, optionally for a specific class."""
        np.random.seed(self.random_state)

        # Compute centroids for each class
        unique_classes = np.unique(self.y)
        centroids = {c: self.X[self.y == c].mean(axis=0) for c in unique_classes}

        # Select classes to add outliers (either all or a specific class)
        target_classes = [class_idx] if class_idx is not None else unique_classes

        # Store outlier indices
        outlier_indices = []

        # Copy dataset to modify outlier locations
        X_outliers = self.X.copy()

        # Distribute outliers among selected classes
        outliers_per_class = n_outliers if class_idx is not None else n_outliers // len(target_classes)

        for class_label in target_classes:
            if class_label not in unique_classes:
                print(f"Warning: Class {class_label} not found in dataset.")
                continue

            # Get class-specific indices
            class_indices = np.where(self.y == class_label)[0]

            # Randomly select points to become outliers
            selected_indices = np.random.choice(class_indices, size=min(outliers_per_class, len(class_indices)), replace=False)
            outlier_indices.extend(selected_indices)

            # Compute direction vectors
            class_centroid = centroids[class_label]
            vectors_from_centroid = X_outliers[selected_indices] - class_centroid
            normalized_vectors = vectors_from_centroid / np.linalg.norm(vectors_from_centroid, axis=1, keepdims=True)

            # Generate varied movement distances using normal distribution
            random_scales = np.abs(np.random.normal(loc=scale, scale=scale_variance, size=len(selected_indices)))

            # Move points further away with variance in distance
            X_outliers[selected_indices] += normalized_vectors * random_scales[:, np.newaxis]  

        # Update stored dataset and outlier indices
        self.X = X_outliers
        self.outlier_idx = np.array(outlier_indices)

        print(f"Outliers added with scale={scale} ± variance={scale_variance}.")
        print(f"Total outliers: {len(self.outlier_idx)}, Applied to class: {class_idx if class_idx is not None else 'All'}")


    def plot_data(self):
        if self.X is None or self.y is None:
            raise ValueError("Data not generated yet. Call generate_data() first.")

        plt.figure(figsize=(10, 6))

        # Define unique classes
        classes = np.unique(self.y)
        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))


        if self.retain_X is not None and self.retain_y is not None:
            X, y = self.retain_X, self.retain_y
        else: 
            X, y = self.X, self.y

        # Scatter plot for each class with a label
        if self.outlier_idx is not None:
            plt.scatter(self.X[self.outlier_idx, 0], self.X[self.outlier_idx, 1], color="black", s=15, label="Outliers", alpha=1, facecolors='none', edgecolors='black', linewidths=1, marker='o')

        for class_idx, color in zip(classes, colors):
            plt.scatter(X[y == class_idx, 0], X[y == class_idx, 1], color=color, s=10, label=f"Class {class_idx}", alpha=0.5)

        if self.forget_X is not None and self.forget_y is not None:
            plt.scatter(self.forget_X[:, 0], self.forget_X[:, 1], color="red", s=10, label="Forget Set", alpha=0.5)


        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Scatter Plot of the Generated Data")
        plt.legend(title="Classes", markerscale=2)  # Make legend markers bigger
        plt.show()

    def get_data(self):
        if self.X is None or self.y is None:
            raise ValueError("Data not generated yet. Call generate_data() first.")
        return self.X, self.y

    def draw_forget_set(self, n_points, class_idx=None, ood_ratio=0.0):
        """
        Selects a mix of in-distribution (ID) and out-of-distribution (OOD) points for forgetting.

        Parameters:
        - n_points (int): Total number of points to forget.
        - class_idx (int, optional): If provided, selects ID and OOD points from this class only.
        - ood_ratio (float, optional): Fraction (0-1) of forget set that should be OOD.

        Returns:
        - forget_X (ndarray): The forgotten points.
        - forget_y (ndarray): Their labels.
        - retain_X (ndarray): The retained points.
        - retain_y (ndarray): Their labels.

        Raises:
        - ValueError: If data or outliers are missing.
        """

        if self.X is None or self.y is None:
            raise ValueError("Data not generated yet. Call generate_data() first.")
        
        if not (0.0 <= ood_ratio <= 1.0):
            raise ValueError("ood_ratio must be between 0 and 1.")

        if self.outlier_idx is None or len(self.outlier_idx) == 0:
            raise ValueError("No OOD points available. Run make_outliers() first.")

        # Compute the number of ID and OOD points to select
        n_ood = int(n_points * ood_ratio)
        n_id = n_points - n_ood

        # Select in-distribution (ID) points
        if class_idx is not None:
            available_id_indices = np.where(self.y == class_idx)[0]
            available_ood_indices = np.array([idx for idx in self.outlier_idx if self.y[idx] == class_idx])  # OOD from same class
        else:
            available_id_indices = np.arange(len(self.X))
            available_ood_indices = self.outlier_idx  # Any OOD points

        # Ensure enough ID points exist
        if len(available_id_indices) < n_id:
            raise ValueError(f"Not enough in-distribution points available (needed: {n_id}, found: {len(available_id_indices)})")

        # Ensure enough OOD points exist for selection
        n_ood = min(n_ood, len(available_ood_indices))

        # Select random ID points
        forget_id_idx = np.random.choice(available_id_indices, size=n_id, replace=False)

        # Select OOD points (from the same class if applicable)
        forget_ood_idx = np.random.choice(available_ood_indices, size=n_ood, replace=False)

        # Combine ID and OOD forget indices
        forget_idx = np.concatenate([forget_id_idx, forget_ood_idx])

        self.forget_X = self.X[forget_idx]
        self.forget_y = self.y[forget_idx]

        # Create the retain set
        self.retain_X = np.delete(self.X, forget_idx, axis=0)
        self.retain_y = np.delete(self.y, forget_idx, axis=0)

        print(f"Forget set drawn with {n_id} ID points and {n_ood} OOD points from class: {class_idx if class_idx is not None else 'All'}")
        
        return self.forget_X, self.forget_y, self.retain_X, self.retain_y



if __name__ == "__main__":
    data_generator = DataGenerator(random_state=42)
    data_generator.generate_data(n_samples=1000, n_features=2, 
                                 n_informative=2, n_redundant=0, 
                                 n_outliers=50, outlier_scale=4.0, 
                                 outlier_variance=0.4, outlier_class=1)
    data_generator.draw_forget_set(n_points=20, class_idx=1, ood_ratio=0.5)

    dataloaders = create_dataloaders(data_generator, batch_size=32)


    # Iterate over full dataset
    for batch in dataloaders["full_loader"]:
        X_batch, y_batch = batch
        print("Full Dataset - Batch X:", X_batch.shape, "Batch y:", y_batch.shape)
        break 

    for batch in dataloaders["retain_loader"]:
        X_batch, y_batch = batch
        print("Retain Dataset - Batch X:", X_batch.shape, "Batch y:", y_batch.shape)
        break 

    for batch in dataloaders["forget_loader"]:
        X_batch, y_batch = batch
        print("Forget Dataset - Batch X:", X_batch.shape, "Batch y:", y_batch.shape)
        break 



