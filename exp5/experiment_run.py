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
from torch.utils.data import RandomSampler, SequentialSampler

from src.models.neural_network import NeuralNetRS
from src.models.vision_transformer_tiny import ViT
from src.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.evaluation.membership_inference_attack import MIA
from src.evaluation.unlearning_evaluator import UnlearningEvaluator
from prepare_image_data import get_image_unlearn_data
from prepare_image_data_v2 import get_image_unlearn_data as get_image_unlearn_data_tsne_box
import time
import json

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

def build_nn(M: int, cfg: dict) -> nn.Module:
    model = NeuralNetRS(M=M, 
                        n_classes=cfg.data.n_classes, 
                        n_layers=cfg.model.neural_network_parameters.n_layers, 
                        width_factor=cfg.model.neural_network_parameters.width_factor, 
                        seed=cfg.model.seed)
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

@hydra.main(config_path=".", config_name="mnist_config")
def main(cfg):
    print("Starting experiment with configuration: %s", cfg.data.dataset_name)

    absolute_root_path = os.path.dirname(__file__) if cfg.absolute_root_path == 'local' else cfg.absolute_root_path
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: %s", DEVICE)
    
    if DEVICE == 'cuda':
        torch.cuda.manual_seed(cfg.model.seed)
        torch.cuda.manual_seed_all(cfg.model.seed)
        torch.backends.cudnn.deterministic = True
        print("CUDA seed set for reproducibility")
    
    print(f'Running Teacher Ascent Analysis experiments on device "{DEVICE}"!')
    print(f"All results will be saved to: {absolute_root_path}")
    dataset_dir = os.path.join(absolute_root_path, 'data')

    if cfg.unlearn.method == 'teacher_ascend':
        hyperparams_postfix = f"{cfg.unlearn.teacher_ascend.n_epochs}epochs_{cfg.unlearn.teacher_ascend._lambda}lambda"
        results_root_dir = os.path.join(absolute_root_path, 'results', f'teacher_ascend_{hyperparams_postfix}')
        weights_root_dir = os.path.join(absolute_root_path, 'weights', f'teacher_ascend_{hyperparams_postfix}')
    elif cfg.unlearn.method == 'scrub':
        hyperparams_postfix = f"{cfg.unlearn.scrub.alpha}alpha_{cfg.unlearn.scrub.gamma}gamma_{cfg.unlearn.scrub.n_rounds}rounds"
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
                                                                                                                        seed=cfg.model.seed)
        # --------- tsne box forget set ---------
        elif cfg.data.split_type == 'tsne_box':
            dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, forget_idxs = get_image_unlearn_data_tsne_box(root_dir=dataset_dir,
                                                                                                                 batch_size=cfg.data.batch_size,
                                                                                                                 subsample_size=cfg.data.subsample_size,
                                                                                                                 seed=cfg.model.seed,
                                                                                                                 boundary=bounding_box_coords)                                                               
        print("MNIST dataloaders created. Train size: %d, Forget size: %d", 
                   len(dataloader_train.dataset), len(dataloader_forget.dataset))
    
    else:
        raise NotImplementedError(f"The dataset {cfg.data.dataset_name} is not supported!")
    
    assert (dataloader_train.dataset.y[forget_idxs] == dataloader_forget.dataset.y).all(), 'Forget indices applied to train do not correspond to the forget data!'
    
    # ========================== Initialize logger ==========================    
    logger = None 
    
    # ========================== Train/load original model ==========================
    # --------- instantiate original model ---------
    os.makedirs(f'{weights_dir}/original_model_{cfg.data.split_type}_{cfg.data.n_forget_points}', exist_ok=True)
    if cfg.model.model_type == 'neural-network':
        original_model = build_nn(M=dataloader_train.dataset.X.shape[-1], cfg=cfg).to(DEVICE)
        save_checkpoints = True    
    else:
        raise NotImplementedError(f"The model type {cfg.model.model_type} is not supported!")
    
    # --------- load or train original model ---------
    orig_model_path = os.path.join(weights_dir, f"original_model_weights.pt")
    if not os.path.exists(orig_model_path):
        print("Training original model from scratch")
        hyperparams = {}
        # re-initialize dataloaders
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)
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
                                       checkpoint_dir = f'{weights_dir}/original_model_{cfg.data.split_type}_{cfg.data.n_forget_points}',
                                       disable_tqdm=cfg.trainer.disable_tqdm, 
                                       do_early_stopping=cfg.trainer.do_early_stopping)
        trainer()
        print("Saving original model weights to: %s", orig_model_path)
        torch.save(original_model.state_dict(), orig_model_path)
    else:
        # Load original model weights
        print("Loading pre-trained original model weights")
        original_model.load_state_dict(torch.load(orig_model_path))

    # Calculate and save metrics for original model
    calculate_model_metrics(original_model, 
                            {"retain": dataloader_retain, "forget": dataloader_forget, "val": dataloader_val}, 
                            DEVICE, 'Original model',
                            save_path=f'{results_dir}/original_model_metrics.json')

    
    # ========================== Train/load retrained model ==========================
    # --------- instantiate retrained model ---------
    os.makedirs(f'{weights_dir}/retrained_model_{cfg.data.split_type}_{cfg.data.n_forget_points}', exist_ok=True)
    if cfg.model.model_type == 'neural-network':
        print("Initializing retrained model as a neural network")
        retrained_model = build_nn(M=dataloader_train.dataset.X.shape[-1], cfg=cfg).to(DEVICE)
        save_checkpoints = True #False

    # --------- train/load retrained model ---------
    if not os.path.exists(f'{weights_dir}/retrained_model_{cfg.data.split_type}_{cfg.data.n_forget_points}/retrained_model_weights.pt'):
        print("Training retrained model from scratch")
        hyperparams = {}
        # re-initialize dataloaders
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)
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
                                        checkpoint_dir = f'{weights_dir}/retrained_model_{cfg.data.split_type}_{cfg.data.n_forget_points}',
                                        disable_tqdm=cfg.trainer.disable_tqdm, 
                                        do_early_stopping=cfg.trainer.do_early_stopping)
        
        trainer()
        torch.save(retrained_model.state_dict(), f'{weights_dir}/retrained_model_{cfg.data.split_type}_{cfg.data.n_forget_points}/retrained_model_weights.pt')
    
    else:
        print("Loading pre-trained retrained model weights")
        retrained_model_sd = torch.load(f'{weights_dir}/retrained_model_{cfg.data.split_type}_{cfg.data.n_forget_points}/retrained_model_weights.pt')
        retrained_model.load_state_dict(retrained_model_sd)
    
    # Calculate and save metrics for retrained model
    calculate_model_metrics(retrained_model, 
                            {"retain": dataloader_retain, "forget": dataloader_forget, "val": dataloader_val}, 
                            DEVICE, 'Retrained model',
                            save_path=f'{results_dir}/retrained_model_metrics.json')


    # ========================== Unlearn: Teacher Ascender ==========================
    if cfg.unlearn.method == 'teacher_ascend':
        print("Initializing Teacher Ascender")
        from src.unlearners.teacher_ascend import TeacherAscender
        hyperparams = {'n_epochs': cfg.unlearn.teacher_ascend.n_epochs, '_lambda': cfg.unlearn.teacher_ascend._lambda}

        mia_model = MIA(device=DEVICE)
        unlearning_evaluator = UnlearningEvaluator(device=DEVICE)
        js_div_func = unlearning_evaluator.JS_divergence

        ta_versions_lists = cfg.unlearn.teacher_ascend.versions
        ta_versions = [tuple(v) for v in ta_versions_lists]
        # convert to list of tuples 
        for (version, is_FIM_ratio) in ta_versions:
            print(f"Running Teacher Ascent with version: {version} and is_FIM_ratio: {is_FIM_ratio}")
            ta_version = TeacherAscender(copy.deepcopy(original_model), n_epochs=hyperparams['n_epochs'], 
                                _lambda=hyperparams['_lambda'], device=DEVICE, MIA=mia_model, js_div_func=js_div_func, retrain_model=retrained_model)
            ta_version_metrics = ta_version(dataloader_retain, dataloader_forget, dataloader_val, eval=True, version=version, is_FIM_ratio=is_FIM_ratio)
            fim_ratio_string = "_fimratio" if is_FIM_ratio else ""
            with open(f'{results_dir}/ta_metrics_{version}{fim_ratio_string}.json', 'w') as f:
                json.dump(ta_version_metrics, f)
        
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)



    # ========================== Unlearn: SCRUB ==========================
    if cfg.unlearn.method == 'scrub':
        print("Initializing SCRUB")
        mia_model = MIA(device=DEVICE)

        from src.unlearners.scrub import ScrubR
        unlearning_evaluator = UnlearningEvaluator(device=DEVICE)
        js_div_func = unlearning_evaluator.JS_divergence
        hyperparams = {'alpha': cfg.unlearn.scrub.alpha, 'gamma': cfg.unlearn.scrub.gamma, 'n_epochs': cfg.unlearn.scrub.n_rounds}

        print("Initializing SCRUB")
        scrub_model = copy.deepcopy(original_model)
        scrub = ScrubR(scrub_model, original_model, alpha=hyperparams['alpha'], gamma=hyperparams['gamma'], device=DEVICE, MIA=mia_model, js_div_func=js_div_func, retrain_model=retrained_model)


        scrub_metrics = scrub(dataloader_retain, dataloader_forget, dataloader_val, n_rounds=hyperparams['n_epochs'], verbose=True)

        with open(f'{results_dir}/scrub_metrics.json', 'w') as f:
            json.dump(scrub_metrics, f)

    # ========================== Unlearn: Gradient Ascent ==========================
    print("Initializing Gradient Ascent")
    from src.unlearners.gradient_ascent import GradientAscent
    dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                        dataloader_forget, dataloader_val,
                                                                                                        cfg.model.seed)
    gradient_ascent = GradientAscent(copy.deepcopy(original_model), hyperparams['n_epochs'], DEVICE, MIA=mia_model)
    ga_metrics = gradient_ascent(dataloader_retain, dataloader_forget, dataloader_val, verbose=True)
    # save metrics to json
    with open(f'{results_dir}/ga_metrics.json', 'w') as f:
        json.dump(ga_metrics, f)

if __name__ == "__main__":
    main()