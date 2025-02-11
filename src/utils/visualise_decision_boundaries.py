import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch
import pdb

def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    shard_id: int,
    h: float = 0.02,  # Step size in the mesh
    affected_shards: list[int] = None
):
    """
    Plot the decision boundary for a Neural Network.
    
    Args:
        model: Trained NeuralNetwork
        X: Input features (must be 2D)
        y: Labels
        shard_id: ID of the shard to visualise
        h: Mesh step size
    """
    if X.shape[1] != 2:
        raise ValueError("This visualization only works with 2D input data")

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Get predictions for all mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_tensor = torch.tensor(mesh_points, dtype=torch.float32)
    Z = model.predict(mesh_tensor).numpy()
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.colorbar()

    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.legend(*scatter.legend_elements(), title="Classes")

    # If we are visualising a shard with an affected point, we state it in the title
    if int(shard_id.split("_")[1]) in affected_shards:
        title = f"Shard {shard_id} - has forgotten point"
    else:
        title = f"Shard {shard_id}"

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    return plt

def fig_to_array(fig):
    """Convert a Matplotlib Figure to a numpy array."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.array(canvas.renderer.buffer_rgba())

def plot_many_decision_boundaries(shard_models: dict, 
                                  X: np.ndarray, 
                                  y: np.ndarray, 
                                  n_rows: int,
                                  n_cols: int = 2,
                                  h: float = 0.02,  
                                  affected_shards: list[int] = None):
    """
    Plot the decision boundaries for all shards.
    """
    # Convert X to numpy array
    X = X.numpy()
    y = y.numpy()

    # if affected_shards is not None:
    #     shard_models = {shard_id: shard_models[shard_id] for shard_id in affected_shards if shard_id in shard_models}

    num_shards = len(shard_models)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]  # Ensure axs is always a 1D array


    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_tensor = torch.tensor(mesh_points, dtype=torch.float32)

    # Plot decision boundaries
    for i, (shard_id, model_dict) in enumerate(shard_models.items()):

        last_slice_key = f"slice_{len(model_dict) - 1}"
        model = model_dict[last_slice_key]

        # Ensure model.predict works with numpy
        Z = model.predict(mesh_tensor).detach().numpy()  # Ensure numpy format
        Z = Z.reshape(xx.shape)


        axs[i].contourf(xx, yy, Z, alpha=0.4)
        scatter = axs[i].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        axs[i].legend(*scatter.legend_elements(), title="Classes")
        if int(shard_id.split("_")[1]) in affected_shards:
            axs[i].set_title(f"Shard {shard_id} - has forgotten point")
        else:
            axs[i].set_title(f"Shard {shard_id}")

    plt.tight_layout()
    plt.show()

def plot_many_decision_boundaries_pre_post_forget(pre_forget_models: dict,
                                                post_forget_models: dict, 
                                                X: np.ndarray, 
                                                y: np.ndarray, 
                                                h: float = 0.02,  
                                                affected_shards: list[int] = None):
    """
    Plot the decision boundaries for all shards side by side.
    Left column: pre-forget models
    Right column: post-forget models
    """
    # Convert X to numpy array
    X = X.numpy()
    y = y.numpy()

    # sort the pre_forget_models and post_forget_models by shard_id
    pre_forget_models = dict(sorted(pre_forget_models.items(), key=lambda x: int(x[0].split("_")[1])))
    post_forget_models = dict(sorted(post_forget_models.items(), key=lambda x: int(x[0].split("_")[1])))

    n_rows = len(pre_forget_models)
    n_cols = 2

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    if n_rows == 1:
        axs = np.array([axs])  # Convert to 2D array for consistent indexing

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_tensor = torch.tensor(mesh_points, dtype=torch.float32)

    # Plot both pre and post forget models row by row
    for i, ((pre_shard_id, pre_model), (post_shard_id, post_model)) in enumerate(zip(pre_forget_models.items(), post_forget_models.items())):
        # Plot pre-forget model (left column)
        Z = pre_model.predict(mesh_tensor).detach().numpy()
        Z = Z.reshape(xx.shape)
        axs[i, 0].contourf(xx, yy, Z, alpha=0.4)
        scatter = axs[i, 0].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        axs[i, 0].legend(*scatter.legend_elements(), title="Classes")
        title = f"Pre-forget: Shard {pre_shard_id}"
        axs[i, 0].set_title(title)

        # Plot post-forget model (right column)
        Z = post_model.predict(mesh_tensor).detach().numpy()
        Z = Z.reshape(xx.shape)
        axs[i, 1].contourf(xx, yy, Z, alpha=0.4)
        scatter = axs[i, 1].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        axs[i, 1].legend(*scatter.legend_elements(), title="Classes")
        title = f"Post-forget: Shard {post_shard_id}"
        axs[i, 1].set_title(title)

    plt.tight_layout()
    plt.show()