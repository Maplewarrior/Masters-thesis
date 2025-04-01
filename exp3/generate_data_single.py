# %%
import numpy as np
from matplotlib import pyplot as plt

def generate_data(centroids: np.ndarray = None, stds: np.ndarray = None, sizes: np.ndarray = None, rogue_point: tuple = None):
    # Make three cluster centroids
    if centroids is None:
        centroids = np.random.randn(3, 2) * 5  # Spread centroids further apart
    # Make three cluster standard deviations
    if stds is None:
        stds = np.abs(np.random.randn(3, 2)) * 0.5 + 0.5  # Ensure positive std values
    # Make three cluster sizes
    if sizes is None:
        sizes = np.random.randint(100, 1000, 3)

    # Generate points from each cluster
    X = []
    y = []
    
    for i in range(3):
        cluster_points = np.random.randn(sizes[i], 2) * stds[i] + centroids[i]
        X.append(cluster_points)
        y.append(np.ones(sizes[i], dtype=int) * i)  # Assign class label i to all points in cluster i

    # Concatenate points and labels
    X = np.concatenate(X)
    y = np.concatenate(y)

    if rogue_point is not None:
        # Unpack the rogue point tuple containing both X and y

        rogue_X, rogue_y = rogue_point
        rogue_X = np.array([rogue_X])
        X = np.concatenate([X, rogue_X])
        y = np.concatenate([y, rogue_y])

        rogue_point_idx = len(X) - 1

        return X, y, rogue_point_idx
    
    return X, y, None


def plot_data(X, y, rogue_point_idx=None):
    # nicer colors
    color_map = plt.colormaps['viridis']
    colors = [color_map(i/3) for i in range(3)]  # Create 3 evenly spaced colors
    num_classes = np.unique(y).size

    for class_idx in range(num_classes):
        plt.scatter(X[y == class_idx, 0], X[y == class_idx, 1], color=colors[class_idx], label=f'Class {class_idx}')
    
    if rogue_point_idx is not None:
        plt.scatter(X[rogue_point_idx, 0], X[rogue_point_idx, 1], c='red', marker='x', label=f'Rogue point (class {y[rogue_point_idx]})')
    
    plt.legend()
    return plt

def save_data(X, y, rogue_point_idx, filename=None):
    if filename is None:
        filename = "data.npz"
    np.savez(filename, X=X, y=y, rogue_point_idx=rogue_point_idx)

def save_fig(plot, filename, folder):
    """Save both to png and pdf"""
    plot.savefig(os.path.join(folder, filename + '.png'))
    plot.savefig(os.path.join(folder, filename + '.pdf'))

# %% 
if __name__ == "__main__":
    import os
    # hardcode centroids, stds, sizes
    centroids = np.array([[0, -3], [-3, 3], [3, 3]])
    stds = np.array([[0.7, 0.7], [0.7, 0.7], [0.7, 0.7]])
    sizes = np.array([20, 20, 20])
    validation_sizes = np.array([10, 10, 10])

    data_folder = 'exp3/data'
    data_plots_folder = os.path.join(data_folder, 'plots')
    os.makedirs(data_plots_folder, exist_ok=True)

    X_val, y_val, _ = generate_data(centroids, stds, validation_sizes)
    save_data(X_val, y_val, None, os.path.join(data_folder, 'validation_data.npz'))
    save_fig(plot_data(X_val, y_val), 'validation_data', data_plots_folder)
    plt.close()

    # %%
    # 1. Rogue point with same distance to all centroids
    rogue_point = (np.mean(centroids, axis=0), np.array([1]))
    X, y, rogue_point_idx = generate_data(centroids, stds, sizes, rogue_point=rogue_point)
    plot = plot_data(X,y, rogue_point_idx)
    save_data(X, y, rogue_point_idx, os.path.join(data_folder, 'data_1.npz'))
    save_fig(plot, 'data_1', data_plots_folder)
    # clear plot 
    plt.close()

    # %%
    # 2. Rogue point with same centroid as the class with 
    rogue_point = (centroids[1], np.array([1]))
    X, y, rogue_point_idx = generate_data(centroids, stds, sizes, rogue_point=rogue_point)
    plot = plot_data(X,y, rogue_point_idx)
    save_data(X, y, rogue_point_idx, os.path.join(data_folder, 'data_2.npz'))
    save_fig(plot, 'data_2', data_plots_folder)
    plt.close()


    # %% 
    # 3. Rogue point with same centroid as the class with a different label
    rogue_point = (centroids[2], np.array([1]))
    X, y, rogue_point_idx = generate_data(centroids, stds, sizes, rogue_point=rogue_point)
    plot = plot_data(X,y, rogue_point_idx)
    save_data(X, y, rogue_point_idx, os.path.join(data_folder, 'data_3.npz'))
    save_fig(plot, 'data_3', data_plots_folder)
    plt.close()

    # %%
    # 4. Rogue point far away from its centroid, but probably in the same decision boundary
    rogue_point = (np.array([-8, 8]), np.array([1]))
    X, y, rogue_point_idx = generate_data(centroids, stds, sizes, rogue_point=rogue_point)
    plot = plot_data(X,y, rogue_point_idx)
    save_data(X, y, rogue_point_idx, os.path.join(data_folder, 'data_4.npz'))
    save_fig(plot, 'data_4', data_plots_folder)
    plt.close()


    # %%
    # 5. Rogue point far away from its centroid, but probably in the same decision boundary
    rogue_point = (np.array([1, -8]), np.array([1]))
    X, y, rogue_point_idx = generate_data(centroids, stds, sizes, rogue_point=rogue_point)
    plot = plot_data(X,y, rogue_point_idx)
    save_data(X, y, rogue_point_idx, os.path.join(data_folder, 'data_5.npz'))
    save_fig(plot, 'data_5', data_plots_folder)
    plt.close()


    