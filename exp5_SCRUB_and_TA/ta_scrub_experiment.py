
"""
This experiment constitutes future work.

We assess the following:
    - Can a repair phase improve teacher ascent overforgetting?
    - How sensitive is SCRUB+R to the number of epochs when one includes the repair phase at each epoch? 
"""

import hydra
import wandb
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
from omegaconf import OmegaConf
import copy
import graphviz
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata

from torch.utils.data import RandomSampler, SequentialSampler

from src.models.neural_network import NeuralNetRS
from src.models.vision_transformer_tiny import ViT
from src.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.evaluation.membership_inference_attack import MIA
from src.evaluation.unlearning_evaluator import UnlearningEvaluator
from prepare_image_data import get_image_unlearn_data
from prepare_image_data_tsne import get_image_unlearn_data as get_image_unlearn_data_tsne_box
import time
import json
import numpy as np


def has_shuffle(dataloader) -> bool:
    """
    Returns True if dataloader was instantiated with shuffle=True and False otherwise.
    """
    if isinstance(dataloader.sampler, SequentialSampler):
        return False
    elif isinstance(dataloader.sampler, RandomSampler):
        return True
    else:
        raise NotImplementedError(f"The sampler: {type(dataloader.sampler)} is not supported by this function!")

def re_instantiate_dataloaders(dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, seed: int):
    """
    A function used for re-instantiating dataloaders. This is for reproducibility purposes.
    If this was not used, the order of execution would affect the resulting model when shuffle=True.
    """
    def recreate(dataloader):
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            dataloader.dataset, 
            shuffle=has_shuffle(dataloader), 
            batch_size=dataloader.batch_size, 
            generator=g
        )
    dataloader_train = recreate(dataloader_train)
    dataloader_retain = recreate(dataloader_retain)
    dataloader_forget = recreate(dataloader_forget)
    dataloader_val = recreate(dataloader_val)
    return dataloader_train, dataloader_retain, dataloader_forget, dataloader_val

def build_nn(M: int, n_classes: int, n_layers: int, width_factor: int, seed: int) -> nn.Module:
    model = NeuralNetRS(M=M, 
                        n_classes=n_classes, 
                        n_layers=n_layers, 
                        width_factor=width_factor, 
                        seed=seed)
    return model

def calculate_model_metrics(model, dataloaders, device, model_name=None, save_path=None):
    """
    Calculate accuracy and MIA metrics for a given model across multiple datasets.
    
    Args:
        model: The PyTorch model to evaluate
        dataloaders: Dictionary containing dataloaders for 'retain', 'forget', and 'val' sets
        device: Device to run the model on ('cuda' or 'cpu')
        model_name: Optional name for the model (for logging purposes)
    
    Returns:
        dict: Dictionary containing accuracy and MIA metrics
    """
    model.eval()
    metrics = {
        'retain': {'acc': []},
        'forget': {'acc': []},
        'val': {'acc': []},
        'mia': {'acc': []}
    }
    
    # Calculate accuracy for each dataset
    with torch.no_grad():
        for dataset_name, dataloader in dataloaders.items():
            if dataset_name not in ['retain', 'forget', 'val']:
                continue
                
            correct = 0
            total = 0
            
            for batch in dataloader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                
                # Handle different model output formats
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('probabilities'))
                else:
                    logits = outputs
                
                _, predicted = torch.max(logits, 1)
                _, true_labels = torch.max(labels, 1)  # Assuming one-hot encoded labels
                
                correct += (predicted == true_labels).sum().item()
                total += labels.size(0)
            
            accuracy = correct / total
            metrics[dataset_name]['acc'].append(accuracy)
    
    # Calculate MIA metric
    mia_model = MIA(device=device)
    mia_score = mia_model(model, 
                         dataloaders['retain'], 
                         dataloaders['forget'], 
                         dataloaders['val'])
    metrics['mia']['acc'].append(mia_score)
    
    if model_name:
        print(f"\nMetrics for {model_name}:")
        print(f"Retain Accuracy: {metrics['retain']['acc'][-1]:.4f}")
        print(f"Forget Accuracy: {metrics['forget']['acc'][-1]:.4f}")
        print(f"Val Accuracy: {metrics['val']['acc'][-1]:.4f}")
        print(f"MIA Score: {metrics['mia']['acc'][-1]:.4f}")
    

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(metrics, f)

    return metrics

def get_nearest_neighbor_idxs(X, y, tsne_results, num_neighbors: int = 5):
    """
    A function that finds the nearest neighbors in t-SNE space to each point in the dataset X.
    The neighbors are constrained such that they must belong to the same class as the sample.

    Returns:
        A numpy array of size [N x num_neighbors] containing the indices in X which have
    """
    # calculate distance from every point to every other point in t-SNE space
    distances = np.linalg.norm((tsne_results[:, np.newaxis, :] - tsne_results[np.newaxis, :, :]), ord=None, axis=2)
    np.fill_diagonal(distances, np.inf) # make sure to exclude diagonal entries
    sorted_idxs = np.argsort(distances, axis=1) # sort distances in ascending order and get indices.

    labels = y.argmax(dim=-1) # convert from OHE to labels
    neighbor_labels = labels[sorted_idxs] # N x N (labels of all neighbors)

    same_class_mask = (neighbor_labels == labels[:, None]) # N x N mask where true means the neighbor has the same class
    masked_sorted_idxs = np.where(same_class_mask, sorted_idxs, -1) # fill different classes with index -1
    
    nn_idxs = np.full((X.size(0), num_neighbors), -1)
    for i in range(X.size(0)):
        valid_idxs = sorted_idxs[i][masked_sorted_idxs[i] != -1]
        nn_idxs[i, :min(num_neighbors, len(valid_idxs))] = valid_idxs[:num_neighbors]
        
    return nn_idxs

def interpolate_points_generator(X, nn_idxs, n_interp_points: int = 100):
    """
        X: N x M matrix of points
        nn_idxs: A N x n_neighbors matrix of the same order as X.
    """
    N = X.shape[0]
    n_neighbors = nn_idxs.shape[1]
    interp_range = np.arange(1/n_interp_points, 1, step=1/n_interp_points)
    # X_interp = np.zeros((N * n_interp_points * len(interp_range), X.shape[1]))
    
    idx = 0 
    for i in range(X.shape[0]):
        for j in range(n_neighbors):
            x = X[i]
            x_nbr = X[nn_idxs[i][j]]
            for alpha in interp_range:
                x_interp = x * alpha + (1-alpha) * x_nbr
                yield x_interp.unsqueeze(0)


def interpolate_points_batch(X, nn_idxs, n_interp_points: int = 100, batch_size: int = 512):
    from itertools import islice
    generator = interpolate_points_generator(X, nn_idxs, n_interp_points)
    while True:
        batch = list(islice(generator, batch_size))
        if not batch:
            break
        yield torch.cat(batch)

def create_entropy_tsne_plot(model, y, 
                             image_generator, 
                             tsne_results,
                             tsne_point_generator, 
                             tsne_bounds_x: tuple[float, float], 
                             tsne_bounds_y: tuple[float, float],
                             n_imgs_per_batch: int = 10, 
                             grid_size: int = 500,
                             device='cuda'):
    
    # Setup
    class_colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', 
                    '#f95d6a', '#ff7c43', '#ffa600', '#aa5382', '#eea152']
    plt.style.use('seaborn-v0_8-paper') 
    plt.figure(figsize=(8, 6))
    
    y = y.argmax(dim=-1)  # from one-hot to label

    # Accumulate data
    all_tsne = []
    all_entropy = []
    all_labels = []

    itt = 0
    while True:
        imgs = next(iter(image_generator), None)
        tsne_points = next(iter(tsne_point_generator), None)
        if imgs is None: #or tsne_points is None:
            break

        imgs = imgs.to(device)
        ys = y[itt * n_imgs_per_batch : (itt + 1) * n_imgs_per_batch]
        ys = ys.repeat_interleave(imgs.size(0) // n_imgs_per_batch)

        probs = model.inference(imgs)['probabilities']
        entropies = -(torch.log(probs + 1e-8) * probs).sum(dim=-1)

        all_tsne.append(tsne_points.cpu().numpy())
        all_entropy.append(entropies.cpu().numpy())
        all_labels.append(ys.cpu().numpy())

        itt += 1

    all_tsne = np.concatenate(all_tsne, axis=0)
    all_entropy = np.concatenate(all_entropy, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Interpolate entropy
    x_min, x_max = all_tsne[:, 0].min(), all_tsne[:, 0].max()
    y_min, y_max = all_tsne[:, 1].min(), all_tsne[:, 1].max()

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )

    grid_entropy = griddata(
        all_tsne, all_entropy, 
        (grid_x, grid_y), 
        #method='linear'  # This leaves NaNs in missing regions
    )

    # Set colormap with transparency for missing areas
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color=(1, 1, 1, 0))  # Fully transparent for NaNs

    # Show entropy heatmap with transparency for missing data
    plt.imshow(
        grid_entropy, 
        extent=(x_min, x_max, y_min, y_max),
        origin='lower',
        cmap=cmap,
        alpha=1.0,
        aspect='auto'
    )
    plt.colorbar(label='Entropy')

    # Overlay t-SNE points
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                c=y, cmap=ListedColormap(class_colors),
                s=10, alpha=0.2)

    plt.title("Interpolated Entropy in t-SNE Space")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig('grid_entropies.png', dpi=300)
    import pdb; pdb.set_trace()
    plt.show()



    
    

@hydra.main(config_path=".", config_name="mnist_config")
def main(cfg):
    print("Starting experiment with configuration: %s", cfg.data.dataset_name)
    hpc_root_path = '/work3/s204138/MachineUnlearning'
    absolute_root_path = os.path.dirname(__file__) if cfg.absolute_root_path == 'local' else cfg.absolute_root_path
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: %s", DEVICE)
    
    
    print(f'Running Teacher Ascent Analysis experiments on device "{DEVICE}"!')
    print(f"All results will be saved to: {absolute_root_path}")
    dataset_dir = os.path.join(absolute_root_path, 'data')

    if cfg.unlearn.method == 'teacher_ascend':
        hyperparams_postfix = f"{cfg.unlearn.teacher_ascend.n_epochs}epochs_{cfg.unlearn.teacher_ascend._lambda}lambda"
        results_root_dir = os.path.join(absolute_root_path, 'results', f'teacher_ascend_{hyperparams_postfix}')
        # weights_root_dir = os.path.join(absolute_root_path, 'weights', f'teacher_ascend_{hyperparams_postfix}')
        weights_root_dir = os.path.join(hpc_root_path, 'weights', f'teacher_ascend_{hyperparams_postfix}')

    elif cfg.unlearn.method == 'scrub':
        hyperparams_postfix = f"{cfg.unlearn.scrub.alpha}alpha_{cfg.unlearn.scrub.gamma}gamma_{cfg.unlearn.scrub.n_rounds}rounds_{cfg.unlearn.scrub.n_repair_rounds}repair_rounds"
        results_root_dir = os.path.join(absolute_root_path, 'results', f'scrub_{hyperparams_postfix}')
        weights_root_dir = os.path.join(absolute_root_path, 'weights', f'scrub_{hyperparams_postfix}')
    else:
        raise NotImplementedError(f"The unlearning method {cfg.unlearn.method} is not supported!")


    if cfg.data.split_type == "tsne_box":
        # coords on the format {"x_min": x_min_val, "x_max": x_max_val, "y_min": y_min_val, "y_max": y_max_val}
        bounding_box_coords = {"x_min": cfg.data.tsne_box_coordinates.x_min, "x_max": cfg.data.tsne_box_coordinates.x_max, "y_min": cfg.data.tsne_box_coordinates.y_min, "y_max": cfg.data.tsne_box_coordinates.y_max}
        bounding_box_name = f"{cfg.data.tsne_box_coordinates.x_min}_{cfg.data.tsne_box_coordinates.x_max}_{cfg.data.tsne_box_coordinates.y_min}_{cfg.data.tsne_box_coordinates.y_max}"
        results_folder_name = f"{cfg.data.split_type}_{bounding_box_name}"
    
    elif cfg.data.split_type == "random":
        results_folder_name = f"{cfg.data.split_type}_{cfg.data.n_forget_points}"

    results_dir = os.path.join(results_root_dir, results_folder_name)
    weights_dir = os.path.join(weights_root_dir, results_folder_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # ========================== Load data and prepare data ==========================
    print("Loading and preparing dataset: %s", cfg.data.dataset_name)
    if cfg.data.dataset_name == 'MNIST':
        # --------- random sampled forget set ---------
        if cfg.data.split_type == 'random':
            dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, forget_idxs = get_image_unlearn_data(root_dir=dataset_dir,
                                                                                                                         dataset_name=cfg['data']['dataset_name'],
                                                                                                                         n_forget_points=cfg.data.n_forget_points,
                                                                                                                         subsample_size=cfg.data.subsample_size,
                                                                                                                         patch_size=cfg.data.patch_size,
                                                                                                                         batch_size=cfg.data.batch_size,
                                                                                                                         seed=cfg.data.seed)
        # --------- tsne box forget set ---------
        elif cfg.data.split_type == 'tsne_box':
            dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, forget_idxs, tsne_results = get_image_unlearn_data_tsne_box(root_dir=dataset_dir,
                                                                                                                                  batch_size=cfg.data.batch_size,
                                                                                                                                  subsample_size=cfg.data.subsample_size,
                                                                                                                                  seed=cfg.data.seed,
                                                                                                                                  boundary=bounding_box_coords,
                                                                                                                                  return_tsne_results=True)
                                                                       
        print("MNIST dataloaders created. Train size: %d, Forget size: %d", 
                   len(dataloader_train.dataset), len(dataloader_forget.dataset))
    
    else:
        raise NotImplementedError(f"The dataset {cfg.data.dataset_name} is not supported!")
    
    
    assert (dataloader_train.dataset.y[forget_idxs] == dataloader_forget.dataset.y).all(), 'Forget indices applied to train do not correspond to the forget data!'
    
    # ========================== Initialize logger ==========================    
    logger = None 
    

    for seed_idx, seed in enumerate(cfg.model.seeds):
        print(f"Running experiment with seed: {seed}, {seed_idx+1} of {len(cfg.model.seeds)}")
        if DEVICE == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            print("CUDA seed set for reproducibility")

        # ========================== Train/load original model ==========================
        # --------- instantiate original model ---------
        os.makedirs(f'{weights_dir}/original_model_{seed}_checkpoints', exist_ok=True)
        if cfg.model.model_type == 'neural-network':
            original_model = build_nn(M=dataloader_train.dataset.X.shape[-1], 
                                      n_classes=cfg.data.n_classes, 
                                      n_layers=cfg.model.neural_network_parameters.n_layers, 
                                      width_factor=cfg.model.neural_network_parameters.width_factor, 
                                      seed=seed).to(DEVICE)
            save_checkpoints = True    
        else:
            raise NotImplementedError(f"The model type {cfg.model.model_type} is not supported!")
        
        # --------- load or train original model ---------
        orig_model_path = os.path.join(weights_dir, f"original_model_weights_{seed}.pt")
        if not os.path.exists(orig_model_path):
            print("Training original model from scratch")
            hyperparams = {}
            # re-initialize dataloaders
            dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                                dataloader_forget, dataloader_val,
                                                                                                                seed)
            trainer = NeuralNetworkTrainer(model=original_model, 
                                        train_dataloader=dataloader_train, 
                                        val_dataloader=dataloader_val, 
                                        logger=logger, 
                                        device=DEVICE, 
                                        learning_rate=cfg.trainer.lr,
                                        weight_decay=cfg.trainer.weight_decay,
                                        n_epochs=cfg.trainer.n_epochs,
                                        optimizer_name=cfg.trainer.optimizer_name,
                                        lr_scheduler=cfg.trainer.lr_scheduler,
                                        save_checkpoints=save_checkpoints,
                                        checkpoint_dir = f'{weights_dir}/original_model_{seed}_checkpoints',
                                        disable_tqdm=cfg.trainer.disable_tqdm, 
                                        do_early_stopping=cfg.trainer.do_early_stopping)
            trainer()
            print("Saving original model weights to: %s", orig_model_path)
            torch.save(original_model.state_dict(), orig_model_path)
        else:
            # Load original model weights
            print("Loading pre-trained original model weights")
            original_model.load_state_dict(torch.load(orig_model_path, map_location=DEVICE))

        # Calculate and save metrics for original model
        calculate_model_metrics(original_model, 
                                {"retain": dataloader_retain, "forget": dataloader_forget, "val": dataloader_val}, 
                                DEVICE, 'Original model',
                                save_path=f'{results_dir}/original_model_metrics_{seed}.json')

        
        # ========================== Train/load retrained model ==========================
        # --------- instantiate retrained model ---------
        os.makedirs(f'{weights_dir}/retrained_model_{seed}_checkpoints', exist_ok=True)
        if cfg.model.model_type == 'neural-network':
            print("Initializing retrained model as a neural network")
            retrained_model = build_nn(M=dataloader_train.dataset.X.shape[-1],
                                       n_classes=cfg.data.n_classes,
                                       n_layers=cfg.model.neural_network_parameters.n_layers,
                                       width_factor=cfg.model.neural_network_parameters.width_factor,
                                       seed=seed).to(DEVICE)
            save_checkpoints = True #False


        # --------- train/load retrained model ---------
        if not os.path.exists(f'{weights_dir}/retrained_model_weights_{seed}.pt'):
            print("Training retrained model from scratch")
            hyperparams = {}
            # re-initialize dataloaders
            dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                                dataloader_forget, dataloader_val,
                                                                                                                seed)
            trainer = NeuralNetworkTrainer(model=retrained_model, 
                                            train_dataloader=dataloader_retain, 
                                            val_dataloader=dataloader_val, 
                                            logger=logger, 
                                            device=DEVICE, 
                                            learning_rate=cfg.trainer.lr,
                                            weight_decay=cfg.trainer.weight_decay,
                                            n_epochs=cfg.trainer.n_epochs,
                                            optimizer_name=cfg.trainer.optimizer_name,
                                            lr_scheduler=cfg.trainer.lr_scheduler,
                                            save_checkpoints=save_checkpoints,
                                            checkpoint_dir = f'{weights_dir}/retrained_model_{seed}_checkpoints',
                                            disable_tqdm=cfg.trainer.disable_tqdm, 
                                            do_early_stopping=cfg.trainer.do_early_stopping)
            
            trainer()
            torch.save(retrained_model.state_dict(), f'{weights_dir}/retrained_model_weights_{seed}.pt')
        
        else:
            print("Loading pre-trained retrained model weights")
            retrained_model_sd = torch.load(f'{weights_dir}/retrained_model_weights_{seed}.pt', map_location=DEVICE)
            retrained_model.load_state_dict(retrained_model_sd)
        
        # Calculate and save metrics for retrained model
        calculate_model_metrics(retrained_model, 
                                {"retain": dataloader_retain, "forget": dataloader_forget, "val": dataloader_val}, 
                                DEVICE, 'Retrained model',
                                save_path=f'{results_dir}/retrained_model_metrics_{seed}.json')

        num_neighbors = 3
        n_interp_points = 25
        n_imgs_per_batch = 10
        batch_size = n_interp_points * num_neighbors * n_imgs_per_batch
        
        nn_idxs = get_nearest_neighbor_idxs(dataloader_train.dataset.X, dataloader_train.dataset.y, tsne_results, num_neighbors)
        interp_img_generator = interpolate_points_batch(dataloader_train.dataset.X, nn_idxs, n_interp_points=n_interp_points, batch_size=batch_size)
        interp_tsne_generator = interpolate_points_batch(torch.tensor(tsne_results), nn_idxs, n_interp_points=n_interp_points, batch_size=batch_size)
        
        create_entropy_tsne_plot(original_model, dataloader_train.dataset.y,
                                 interp_img_generator, tsne_results,
                                 interp_tsne_generator,
                                 (tsne_results[:, 0].min(), tsne_results[:, 0].max()),
                                 (tsne_results[:, 1].min(), tsne_results[:, 1].max()),
                                 n_imgs_per_batch, device='cuda')
        
        import pdb; pdb.set_trace()

        # ========================== Unlearn: Teacher Ascender ==========================
        if cfg.unlearn.method == 'teacher_ascend':
            print("Initializing Teacher Ascender")
            from src.unlearners.teacher_ascend import TeacherAscender
            hyperparams = {'n_epochs': cfg.unlearn.teacher_ascend.n_epochs, '_lambda': cfg.unlearn.teacher_ascend._lambda, 'lr': 1e-3}

            mia_model = MIA(device=DEVICE)
            unlearning_evaluator = UnlearningEvaluator(device=DEVICE)
            js_div_func = unlearning_evaluator.JS_divergence

            # ta_versions_lists = cfg.unlearn.teacher_ascend.versions
            # convert to list of tuples 
            ta_versions = [("entropy-retain-repair", True)] # include repair phase & use the FIM ratio. #[tuple(v) for v in ta_versions_lists]
            # ta_versions = [("entropy-retain", True)]
            for (version, is_FIM_ratio) in ta_versions:
                print(f"Running Teacher Ascent with version: {version} and is_FIM_ratio: {is_FIM_ratio}")
                ta_version = TeacherAscender(copy.deepcopy(original_model), 
                                    n_epochs=hyperparams['n_epochs'], 
                                    _lambda=hyperparams['_lambda'], 
                                    device=DEVICE, 
                                    MIA=mia_model, 
                                    js_div_func=js_div_func, 
                                    retrain_model=retrained_model, 
                                    lr=hyperparams['lr'])
                ta_version_metrics = ta_version(dataloader_retain, dataloader_forget, dataloader_val, eval=True, version=version, is_FIM_ratio=is_FIM_ratio)
                fim_ratio_string = "_fimratio" if is_FIM_ratio else ""
                with open(f'{results_dir}/ta_metrics_{version}{fim_ratio_string}_{seed}.json', 'w') as f:
                    json.dump(ta_version_metrics, f)
            
            dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                                dataloader_forget, dataloader_val,
                                                                                                                seed)


        # ========================== Unlearn: SCRUB ==========================
        if cfg.unlearn.method == 'scrub':
            print("Initializing SCRUB")
            mia_model = MIA(device=DEVICE)

            from src.unlearners.scrub import ScrubR
            unlearning_evaluator = UnlearningEvaluator(device=DEVICE)
            js_div_func = unlearning_evaluator.JS_divergence
            hyperparams = {'alpha': cfg.unlearn.scrub.alpha, 'gamma': cfg.unlearn.scrub.gamma, 'n_epochs': cfg.unlearn.scrub.n_rounds, 'n_repair_rounds': cfg.unlearn.scrub.n_repair_rounds}

            print("Initializing SCRUB")
            scrub_model = copy.deepcopy(original_model)
            scrub = ScrubR(scrub_model, original_model, alpha=hyperparams['alpha'], gamma=hyperparams['gamma'], device=DEVICE, MIA=mia_model, js_div_func=js_div_func, retrain_model=retrained_model)


            scrub_metrics = scrub(dataloader_retain, dataloader_forget, dataloader_val, n_rounds=hyperparams['n_epochs'], n_repair_rounds=hyperparams['n_repair_rounds'], verbose=True)

            with open(f'{results_dir}/scrub_metrics_{seed}.json', 'w') as f:
                json.dump(scrub_metrics, f)

    # ========================== Unlearn: Gradient Ascent ==========================

    if cfg.unlearn.do_gradient_ascent:
        print("Initializing Gradient Ascent")
        from src.unlearners.gradient_ascent import GradientAscent
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            seed)
        gradient_ascent = GradientAscent(copy.deepcopy(original_model), hyperparams['n_epochs'], DEVICE, MIA=mia_model)
        ga_metrics = gradient_ascent(dataloader_retain, dataloader_forget, dataloader_val, verbose=True)
        # save metrics to json
        with open(f'{results_dir}/ga_metrics_{seed}.json', 'w') as f:
            json.dump(ga_metrics, f)

if __name__ == "__main__":
    main()