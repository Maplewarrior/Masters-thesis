from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb

from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, X, y, use_indices: bool = False, onehot_labels: bool = False, n_classes: int = None, device="cpu"):
        """
        Args:
            X (numpy.ndarray): Features of shape (num_samples, num_features).
            y (numpy.ndarray): Labels of shape (num_samples,).
            use_indices (bool): Whether to return indices with each batch.
            onehot_labels (bool): Whether to one-hot encode the labels.
            n_classes (int): Number of classes for one-hot encoding.
            device (str): The device to move the data to.
        """
        self.device = device
        # Convert features to float32
        self.X = torch.tensor(X, dtype=torch.float32).to(self.device)
        # Convert labels to long (integer)
        self.y = torch.tensor(y, dtype=torch.long).to(self.device)
        self.indices = torch.arange(len(self.X), device=self.device) if use_indices else None

        self.onehot_labels = onehot_labels

        self.n_classes = n_classes

        if self.onehot_labels:
            self.y = self.onehot_encode_labels(self.y, self.n_classes)

    def __len__(self):
        return len(self.X)
    
    def onehot_encode_labels(self, y, n_classes):
        if n_classes is None:
            n_classes = y.max() + 1
        return torch.zeros(len(y), n_classes, device=self.device).scatter_(1, y.unsqueeze(1), 1)

    def __getitem__(self, idx):
        if self.indices is not None:
            return self.X[idx], self.y[idx], self.indices[idx]
        else:
            return self.X[idx], self.y[idx]
        
    
def create_dataloaders(data_generator, batch_size=32, onehot_labels=False, use_indices=False, device="cpu", shuffle=True):
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
    full_train_dataset = SyntheticDataset(data_generator.train_X, data_generator.train_y, onehot_labels=onehot_labels, n_classes=data_generator.n_classes, use_indices=use_indices, device=device)
    full_val_dataset = SyntheticDataset(data_generator.val_X, data_generator.val_y, onehot_labels=onehot_labels, n_classes=data_generator.n_classes, use_indices=use_indices, device=device)
    full_test_dataset = SyntheticDataset(data_generator.test_X, data_generator.test_y, onehot_labels=onehot_labels, n_classes=data_generator.n_classes, use_indices=use_indices, device=device)

    print(f"  Full dataset size: {len(full_train_dataset)}")
    print(f"  Full val dataset size: {len(full_val_dataset)}")
    print(f"  Full test dataset size: {len(full_test_dataset)}")

    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=shuffle)
    full_val_loader = DataLoader(full_val_dataset, batch_size=batch_size, shuffle=shuffle)
    full_test_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=shuffle)

    # Retain set
    if data_generator.retain_X is not None and data_generator.retain_y is not None:
        retain_dataset = SyntheticDataset(data_generator.retain_X, data_generator.retain_y, onehot_labels=onehot_labels, n_classes=data_generator.n_classes, use_indices=use_indices, device=device)
        retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
        print(f"  Retain dataset size: {len(retain_dataset)}")
    else:
        retain_loader = None

    # Forget set
    if data_generator.forget_X is not None and data_generator.forget_y is not None:
        forget_dataset = SyntheticDataset(data_generator.forget_X, data_generator.forget_y, onehot_labels=onehot_labels, n_classes=data_generator.n_classes, use_indices=use_indices, device=device)

        forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)
        print(f"  Forget dataset size: {len(forget_dataset)}")
    else:
        forget_loader = None


    return {
        "train_full_loader": full_train_loader,
        "val_loader": full_val_loader,
        "test_loader": full_test_loader,
        "train_retain_loader": retain_loader,
        "train_forget_loader": forget_loader,
        "forget_idx_in_train": data_generator.forget_idx_in_train,
        "retain_idx_in_train": data_generator.retain_idx_in_train
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
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.val_X = None
        self.val_y = None
        self.test_X = None
        self.test_y = None
        self.train_X = None
        self.train_y = None
        self.n_classes = None
        self.forget_idx_in_train = None
        self.retain_idx_in_train = None

        self.indices = None

    def generate_data(self, n_samples=5000, n_features=2, n_informative=2, n_redundant=0, 
                      n_clusters_per_class=1, n_classes=4, n_outliers=100, outlier_scale=2.0, outlier_variance=0.2, outlier_class=None):
        self.X, self.y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, 
                           n_redundant=n_redundant, n_clusters_per_class=n_clusters_per_class, random_state=self.random_state, n_classes=n_classes)
        self.indices = np.arange(len(self.X))

        self.n_classes = n_classes


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
        outliers_per_class = n_outliers // len(target_classes)
        
        # Distribute remaining outliers
        n_outliers_per_class_list = [outliers_per_class] * len(target_classes)
        remaining = n_outliers - sum(n_outliers_per_class_list)
        for i in range(remaining):
            n_outliers_per_class_list[i] += 1

        for class_label in target_classes:
            if class_label not in unique_classes:
                print(f"Warning: Class {class_label} not found in dataset.")
                continue

            # Get class-specific indices
            class_indices = np.where(self.y == class_label)[0]

            # Randomly select points to become outliers
            n_out = n_outliers_per_class_list[class_label] if class_idx is None else outliers_per_class
            selected_indices = np.random.choice(class_indices, size=min(n_out, len(class_indices)), replace=False)
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

        # also plot circles around the test and val set
        if self.test_X is not None and self.test_y is not None:
            plt.scatter(self.test_X[:, 0], self.test_X[:, 1], color="black", s=15, label="Test Set", alpha=1, facecolors='none', edgecolors='gold', linewidths=1, marker='o')
        if self.val_X is not None and self.val_y is not None:
            plt.scatter(self.val_X[:, 0], self.val_X[:, 1], color="black", s=15, label="Validation Set", alpha=1, facecolors='none', edgecolors='purple', linewidths=1, marker='o')


        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Scatter Plot of the Generated Data")
        plt.legend(title="Classes", markerscale=2)  # Make legend markers bigger
        plt.show()

    def get_data(self):
        if self.X is None or self.y is None:
            raise ValueError("Data not generated yet. Call generate_data() first.")
        return self.X, self.y
    
    def split_data(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        if self.X is None or self.y is None:
            raise ValueError("Data not generated yet. Call generate_data() first.")
        
        # Ensure ratios sum to 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            raise ValueError("Ratios must sum to 1")

        # Create array of indices
        indices = np.arange(len(self.X))
        
        # First split: separate test set
        remaining_idx, self.test_idx = np.split(
            np.random.permutation(indices), 
            [int((train_ratio + val_ratio) * len(indices))]
        )
        
        # Second split: separate train and validation from remaining
        self.train_idx, self.val_idx = np.split(
            remaining_idx,
            [int((train_ratio/(train_ratio + val_ratio)) * len(remaining_idx))]
        )

        # Update train, val and test sets
        self.train_X = self.X[self.train_idx]
        self.train_y = self.y[self.train_idx]

        self.train_indices = self.indices[self.train_idx]
    
        self.val_X = self.X[self.val_idx]
        self.val_y = self.y[self.val_idx]

        self.test_X = self.X[self.test_idx]
        self.test_y = self.y[self.test_idx]


        return self.train_idx, self.val_idx, self.test_idx

    def draw_forget_set(self, n_points, class_idx=None, ood_ratio=0.0, random_state=None):
        """
        Selects a mix of in-distribution (ID) and out-of-distribution (OOD) points for forgetting.

        Parameters:
        - n_points (int): Total number of points to forget.
        - class_idx (int, optional): If provided, selects ID and OOD points from this class only.
        - ood_ratio (float, optional): Fraction (0-1) of forget set that should be OOD.
        - random_state (int, optional): Random seed for reproducible forget set selection.

        Returns:
        - forget_X (ndarray): The forgotten points.
        - forget_y (ndarray): Their labels.
        - retain_X (ndarray): The retained points.
        - retain_y (ndarray): Their labels.
        """
        # Set random state for reproducibility
        rng = np.random.RandomState(random_state) if random_state is not None else np.random

        # Reset forget and retain sets
        self.forget_X = None
        self.forget_y = None
        self.retain_X = None
        self.retain_y = None
        self.forget_idx_in_train = None
        self.retain_idx_in_train = None

        if self.X is None or self.y is None:
            raise ValueError("Data not generated yet. Call generate_data() first.")
        
        if not (0.0 <= ood_ratio <= 1.0):
            raise ValueError("ood_ratio must be between 0 and 1.")

        if self.outlier_idx is None or len(self.outlier_idx) == 0:
            raise ValueError("No OOD points available. Run make_outliers() first.")
        

        if self.train_X is None or self.train_y is None:
            raise ValueError("Train data not generated yet. Call split_data() first.")

        # Compute the number of ID and OOD points to select
        n_ood = int(n_points * ood_ratio)
        n_id = n_points - n_ood

        # check if there are enough ID points
        if len(self.train_idx) - len(self.outlier_idx) < n_id:
            raise ValueError(f"Not enough in-distribution points available (needed: {n_id}, found: {len(self.train_idx) - len(self.outlier_idx)})")
        
        # check if there are enough OOD points
        if len(self.outlier_idx) < n_ood:
            raise ValueError(f"Not enough out-of-distribution points available (needed: {n_ood}, found: {len(self.outlier_idx)})")

        # Select in-distribution (ID) points
        if class_idx is not None:
            print(f"Selecting ID and OOD points from class {class_idx}")
            available_id_indices = np.where(self.y == class_idx)[0]
            available_ood_indices = np.array([idx for idx in self.outlier_idx if self.y[idx] == class_idx])  # OOD from same class
            
        else:
            print(f"Selecting ID and OOD points from all classes")
            available_id_indices = np.arange(len(self.X))

            available_ood_indices = self.outlier_idx  # Any OOD points

        # Only get outliers that is within the train set
        available_ood_indices = np.intersect1d(available_ood_indices, self.train_idx)

        # Only get ID points that is within the train set
        available_id_indices = np.intersect1d(available_id_indices, self.train_idx)
        # Only get ID points that is not an outlier
        available_id_indices = np.setdiff1d(available_id_indices, self.outlier_idx)

        # Ensure enough ID points exist
        if len(available_id_indices) < n_id:
            raise ValueError(f"Not enough in-distribution points available (needed: {n_id}, found: {len(available_id_indices)})")

        # Ensure enough OOD points exist for selection
        n_ood = min(n_ood, len(available_ood_indices))

        print(f" Available ID points: {len(available_id_indices)}")
        print(f" Available OOD points: {len(available_ood_indices)}")


        # Select random ID points using the controlled random state
        forget_id_idx = rng.choice(available_id_indices, size=n_id, replace=False)

        # Select OOD points using the controlled random state
        forget_ood_idx = rng.choice(available_ood_indices, size=n_ood, replace=False)

        # Combine ID and OOD forget indices
        forget_idx = np.concatenate([forget_id_idx, forget_ood_idx])

        self.forget_X = self.X[forget_idx]
        self.forget_y = self.y[forget_idx]

        to_delete_idx = np.concatenate([forget_idx, self.val_idx, self.test_idx])
        retain_idx = np.setdiff1d(self.train_idx, to_delete_idx)

        # make retain set
        self.retain_X = np.delete(self.X, to_delete_idx, axis=0)
        self.retain_y = np.delete(self.y, to_delete_idx, axis=0)

        # Get all indices in self.train_indices that are in forget_idx
        self.forget_idx_in_train = np.where(np.isin(self.train_indices, forget_idx))[0]
        self.retain_idx_in_train = np.where(np.isin(self.train_indices, retain_idx))[0]

        print(f"Forget set drawn with {n_id} ID points and {n_ood} OOD points from class: {class_idx if class_idx is not None else 'All'}")
        
        return self.forget_X, self.forget_y, self.retain_X, self.retain_y, self.val_X, self.val_y, self.test_X, self.test_y


if __name__ == "__main__":
    # Initialize data generator and create base dataset
    data_generator = DataGenerator(random_state=42)
    data_generator.generate_data(n_samples=1000, n_features=2, 
                               n_informative=2, n_redundant=0, 
                               n_outliers=50, outlier_scale=4.0, 
                               outlier_variance=0.4, outlier_class=None)
    data_generator.split_data(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    print("\n=== First forget set (random_state=42) ===")
    data_generator.draw_forget_set(n_points=20, class_idx=1, ood_ratio=0.5, random_state=42)
    data_generator.plot_data()  # Visualize first forget set

    # Create and check first set of dataloaders
    dataloaders_1 = create_dataloaders(data_generator, batch_size=32, onehot_labels=True)
    print(f"First forget set size: {len(dataloaders_1['train_forget_loader'].dataset)}")
    print(f"First retain set size: {len(dataloaders_1['train_retain_loader'].dataset)}")

    print("\n=== Second forget set (random_state=43) ===")
    data_generator.draw_forget_set(n_points=20, class_idx=1, ood_ratio=0.5, random_state=43)
    data_generator.plot_data()  # Visualize second forget set

    # Create and check second set of dataloaders
    dataloaders_2 = create_dataloaders(data_generator, batch_size=32, onehot_labels=True)
    print(f"Second forget set size: {len(dataloaders_2['train_forget_loader'].dataset)}")
    print(f"Second retain set size: {len(dataloaders_2['train_retain_loader'].dataset)}")

    print("\n=== Third forget set (random_state=42 again) ===")
    data_generator.draw_forget_set(n_points=20, class_idx=1, ood_ratio=0.5, random_state=42)
    data_generator.plot_data()  # Visualize third forget set (should match first)

    # Create and check third set of dataloaders (should match first)
    dataloaders_3 = create_dataloaders(data_generator, batch_size=32, onehot_labels=True)
    print(f"Third forget set size: {len(dataloaders_3['train_forget_loader'].dataset)}")
    print(f"Third retain set size: {len(dataloaders_3['train_retain_loader'].dataset)}")

    # Verify that first and third forget sets are identical (same random_state)
    forget_indices_1 = dataloaders_1['forget_idx_in_train']
    forget_indices_3 = dataloaders_3['forget_idx_in_train']
    print("\n=== Verification ===")
    print(f"First and third forget sets are identical: {np.array_equal(forget_indices_1, forget_indices_3)}")



