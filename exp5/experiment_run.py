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
import time
import json

class ExperimentSpecs:
    def __init__(self, model, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                 result_dir, model_name, dataset_name, hyperparameters, seed, device) -> None:
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_retain = dataloader_retain
        self.dataloader_forget = dataloader_forget
        self.dataloader_val = dataloader_val
        self.result_dir = result_dir
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.hyperparameters = hyperparameters
        self.seed = seed
        self.device = device

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

def build_vit(cfg: dict) -> nn.Module:
    vit_params = cfg.model.vision_transformer_parameters
    d_patch = int(cfg.data.patch_size[0] * cfg.data.patch_size[1] * cfg.data.n_channels)
    n_patches = int(cfg.data.image_size[0] // cfg.data.patch_size[0] * cfg.data.image_size[1] // cfg.data.patch_size[1])
    vit_model = ViT(d_patch=d_patch,
                    d_hidden=vit_params.d_hidden,
                    d_ff=vit_params.d_ff,
                    d_k = vit_params.d_k,
                    n_layers=vit_params.n_layers,
                    n_heads=vit_params.n_heads,
                    n_classes=cfg.data.n_classes,
                    n_patches=n_patches,
                    dropout_prob=vit_params.dropout_prob,
                    tau=vit_params.tau,
                    pooling_type=vit_params.pooling_type
                    )
    return vit_model

def build_nn(M: int, cfg: dict) -> nn.Module:
    model = NeuralNetRS(M=M, 
                        n_classes=cfg.data.n_classes, 
                        n_layers=cfg.model.neural_network_parameters.n_layers, 
                        width_factor=cfg.model.neural_network_parameters.width_factor, 
                        seed=cfg.model.seed)
    return model

def eval_single_model(model, 
                      dataloader_retain, 
                      dataloader_forget, 
                      dataloader_val,
                      time: float,
                      result_dir: str,
                      model_name: str,
                      hyperparameters,
                      seed: int,
                      device: str):
    
    if os.path.exists(f'{result_dir}/all_results.json'):
        with open(f'{result_dir}/all_results.json', 'r') as f:
            results = json.load(f)
    else:
        results = {'model-name': [],
                   'MIA-probability': [],
                   'retain-accuracy': [],
                   'forget-accuracy': [],
                   'val-accuracy': [],
                   'time (sec)': [],
                   'hyperparameters': [],
                   'seed': []
                    }
    
    if os.path.exists(f'{result_dir}/wrong_forget_preds.json'):
        with open(f'{result_dir}/wrong_forget_preds.json') as f:
            wrong_preds = json.load(f)
    else:
        wrong_preds = {'model-name': [],
                       'wrong-preds': []}

    mia_model = MIA(device=device)
    UE = UnlearningEvaluator(device=device)
    retain_probs, y_retain = UE.get_model_probs(model, dataloader_retain)
    forget_probs, y_forget = UE.get_model_probs(model, dataloader_forget)
    val_probs, y_val = UE.get_model_probs(model, dataloader_val)
    
    results['model-name'].append(model_name)
    results['retain-accuracy'].append(UE.calculate_accuracy(retain_probs, y_retain))
    results['forget-accuracy'].append(UE.calculate_accuracy(forget_probs, y_forget))
    results['val-accuracy'].append(UE.calculate_accuracy(val_probs, y_val))
    results['MIA-probability'].append(mia_model(model, dataloader_retain, dataloader_forget, dataloader_val))
    results['time (sec)'].append(time)
    results['hyperparameters'].append(hyperparameters)
    results['seed'].append(seed)
    
    df_subset = pd.DataFrame.from_dict(results)[['model-name', 'MIA-probability', 'retain-accuracy', 
                                                'forget-accuracy', 'val-accuracy', 'time (sec)']]
    print(f"Successfully updated the results! The file now looks like:\n{df_subset}")
    
    logits, xs, ys = get_model_predictions(model, dataloader_forget, device)
    disagree_idxs = get_wrong_predictions(logits, ys)
    wrong_preds['model-name'].append(model_name)
    wrong_preds['wrong-preds'].append(disagree_idxs)

    with open(f'{result_dir}/all_results.json', 'w') as f:
        json.dump(results, f)
    
    with open(f'{result_dir}/wrong_forget_preds.json', 'w') as f:
        json.dump(wrong_preds, f)

    return results
    
def create_simple_model_visualization(model, save_path, input_shape):
    """Create a simple flowchart visualization of the model architecture.
    
    Args:
        model: The neural network model to visualize
        save_path: Path to save the visualization
        input_shape: Shape of the input data (for labeling input node)
    """
    dot = graphviz.Digraph(comment='Neural Network Architecture')
    
    # Set graph attributes for better appearance
    dot.attr(rankdir='LR', bgcolor='white', dpi='300', fontname='Helvetica')
    
    # Set node attributes
    dot.attr('node', shape='box', style='filled,rounded', 
             fillcolor='#E8F0FE', color='#4285F4', 
             fontname='Helvetica', fontsize='14', fontcolor='#333333')
    
    # Set edge attributes
    dot.attr('edge', color='#4285F4', penwidth='1.5', arrowsize='0.8')
    # Add input node
    dot.node('input', f'Input\n({input_shape} features)', shape='oval')
    
    # Add nodes for each layer in the Sequential model
    prev_node = 'input'
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.Linear):
            node_name = f'linear_{i}'
            label = f'Linear\n{layer.in_features} → {layer.out_features}'
            dot.node(node_name, label)
            dot.edge(prev_node, node_name)
            prev_node = node_name
        elif isinstance(layer, nn.ReLU):
            node_name = f'relu_{i}'
            dot.node(node_name, 'ReLU')
            dot.edge(prev_node, node_name)
            prev_node = node_name
    
    # Add output node
    dot.node('output', f'Output\n({model.net[-1].out_features} classes)', shape='oval')
    dot.edge(prev_node, 'output')
    
    # Render the visualization
    dot.render(save_path, format='png')
    return dot

def get_model_predictions(model: nn.Module, dataloader, device: str):
    # ensure shuffle is turned off
    dataloader = DataLoader(dataloader.dataset, shuffle=False, batch_size = dataloader.batch_size)
    xs = []
    ys = []
    probs = []
    for batch in dataloader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        model_out = model.inference(x)
        xs.append(x)
        ys.append(y)
        probs.append(model_out['probabilities'])

    xs = torch.cat(xs)
    ys = torch.cat(ys)
    probs = torch.cat(probs)
    return probs, xs, ys

def get_wrong_predictions(probs, ys) -> list:
    return torch.where(probs.argmax(dim=-1) != ys.argmax(dim=-1))[0].cpu().tolist()

def plot_wrong_predictions(model, dataloader, device, model_name: str):
    probs, xs, ys = get_model_predictions(model, dataloader, device)
    disagree_idxs = get_wrong_predictions(probs, ys)
    x_disagree = xs[disagree_idxs].view(-1, 28, 28).cpu()
    save_dir = f'{model_name}_wrong_predictions'
    os.makedirs(save_dir, exist_ok=True)
    for i in range(x_disagree.size(0)):
        y_pred = probs.argmax(dim=-1)[disagree_idxs[i]].item()
        y_true = ys.argmax(dim=-1)[disagree_idxs[i]].item()
        plt.imshow(x_disagree[i], cmap='grey')
        plt.title(f'True label: {y_true}. Predicted label: {y_pred}.')
        plt.savefig(f'{save_dir}/x_disagree_{disagree_idxs[i]}.png')

def run_and_eval(func: callable, experiment_specs: object):
    # call function and time the call
    start = time.time()
    func()
    end = time.time()
    elapsed = end-start
    # re-instantiate all dataloaders
    train_loader = experiment_specs.dataloader_train
    retain_loader = experiment_specs.dataloader_retain
    forget_loader = experiment_specs.dataloader_forget
    val_loader = experiment_specs.dataloader_val
    train_loader, retain_loader, forget_loader, val_loader = re_instantiate_dataloaders(train_loader, retain_loader, 
                                                                                        forget_loader, val_loader, 
                                                                                        experiment_specs.seed)
    experiment_specs.dataloader_train = train_loader
    experiment_specs.dataloader_retain = retain_loader
    experiment_specs.dataloader_forget = forget_loader
    experiment_specs.dataloader_val = val_loader
    
    eval_single_model(experiment_specs.model, experiment_specs.dataloader_retain, experiment_specs.dataloader_forget, 
                      experiment_specs.dataloader_val, elapsed, experiment_specs.result_dir, experiment_specs.model_name, 
                      experiment_specs.hyperparameters, experiment_specs.seed, experiment_specs.device)

def plot_mia_metrics(models, figsize=(12, 8)):
    """
    Plot MIA metrics for multiple models.
    """
    plt.style.use('default')
    colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
    
    # Create subplots - one for each metric
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Model MIA Comparison Across Epochs', fontsize=16, fontweight='bold')
    
    metrics = ['mia']
    metric_titles = ['MIA Probability']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i]
        
        # Plot each model
        for j, (model_name, model_data) in enumerate(models.items()):
            if metric in model_data and 'acc' in model_data[metric]:
                acc_values = model_data[metric]['acc']
                epochs = range(1, len(acc_values) + 1)
                
                ax.plot(epochs, acc_values, 
                       marker='o', linewidth=2, markersize=4,
                       color=colors[j % len(colors)], 
                       label=model_name)
                
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MIA Probability')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set y-axis limits to better show accuracy range
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return plt

def plot_model_accuracies(models, figsize=(12, 8)):
    """
    Plot accuracy curves for multiple models showing retain, forget, and val accuracies.
    
    Args:
        models: Dictionary where keys are model names and values are dictionaries
                with 'retain', 'forget', 'val' keys containing accuracy lists
        figsize: Tuple for figure size (width, height)
    """
    # Set up the plot style
    plt.style.use('default')
    colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
    
    # Create subplots - one for each metric
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Model Accuracy Comparison Across Epochs', fontsize=16, fontweight='bold')
    
    metrics = ['retain', 'forget', 'val']
    metric_titles = ['Retain Accuracy', 'Forget Accuracy', 'Validation Accuracy']

    # Remove MIA metrics
    for model_name, model_data in models.items():
        if 'mia' in model_data:
            del model_data['mia']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i]
        
        # Plot each model
        for j, (model_name, model_data) in enumerate(models.items()):
            if metric in model_data and 'acc' in model_data[metric]:
                acc_values = model_data[metric]['acc']
                epochs = range(1, len(acc_values) + 1)
                
                ax.plot(epochs, acc_values, 
                       marker='o', linewidth=2, markersize=4,
                       color=colors[j % len(colors)], 
                       label=model_name)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set y-axis limits to better show accuracy range
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return plt

def calculate_model_metrics(model, dataloaders, device, model_name=None):
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
    results_dir = os.path.join(absolute_root_path, 'results', f'{cfg.data.dataset_name}', f'seed_{cfg.model.seed}')
    weights_dir = os.path.join(absolute_root_path, 'weights', f'{cfg.data.dataset_name}', f'seed_{cfg.model.seed}')
    os.makedirs(results_dir, exist_ok=True)
    

    # ============= Load data and prepare data =============
    print("Loading and preparing dataset: %s", cfg.data.dataset_name)
    if cfg.data.dataset_name == 'MNIST':
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, forget_idxs = get_image_unlearn_data(root_dir=dataset_dir,
                                                                                                                 dataset_name=cfg['data']['dataset_name'],
                                                                                                                 n_forget_points=cfg.data.n_forget_points,
                                                                                                                 subsample_size=cfg.data.subsample_size,
                                                                                                                 patch_size=cfg.data.patch_size,
                                                                                                                 batch_size=cfg.data.batch_size,
                                                                                                                 seed=cfg.model.seed)
        print("MNIST dataloaders created. Train size: %d, Forget size: %d", 
                   len(dataloader_train.dataset), len(dataloader_forget.dataset))
    
    else:
        dataloader_train, dataloader_retain, dataloader_retain_no_aug, dataloader_forget, dataloader_val, forget_idxs = get_image_unlearn_data(root_dir=dataset_dir,
                                                                                                                 dataset_name=cfg['data']['dataset_name'],
                                                                                                                 n_forget_points=cfg.data.n_forget_points,
                                                                                                                 subsample_size=cfg.data.subsample_size,
                                                                                                                 patch_size=cfg.data.patch_size,
                                                                                                                 batch_size=cfg.data.batch_size,
                                                                                                                 seed=cfg.model.seed)


    assert (dataloader_train.dataset.y[forget_idxs] == dataloader_forget.dataset.y).all(), 'Forget indices applied to train do not correspond to the forget data!'

    # # ============= Initialize logger =============    
    logger = None 
    
    # ============= Instantiate original model =============
    os.makedirs(f'{weights_dir}/original_model', exist_ok=True)
    if cfg.model.model_type == 'neural-network':
        original_model = build_nn(M=dataloader_train.dataset.X.shape[-1], cfg=cfg).to(DEVICE)
        save_checkpoints = True
    
    elif cfg.model.model_type == 'vision-transformer':
        original_model = build_vit(cfg).to(DEVICE)
        save_checkpoints = True
    
    else:
        raise NotImplementedError(f"The model type {cfg.model.model_type} is not supported!")
    

    # ============= Train/load original model =============
    if not os.path.exists(f'{weights_dir}/original_model/original_model_weights.pt'):
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
                                       checkpoint_dir = f'{weights_dir}/original_model',
                                       disable_tqdm=cfg.trainer.disable_tqdm, 
                                       do_early_stopping=cfg.trainer.do_early_stopping)
        original_train_fn = lambda: trainer()
        original_experiment_specs = ExperimentSpecs(original_model, dataloader_train, dataloader_retain, 
                                                    dataloader_forget, dataloader_val, results_dir, 'Original model', 
                                                    cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        run_and_eval(original_train_fn, original_experiment_specs)
        original_model
        print("Saving original model weights to: %s", f'{weights_dir}/original_model/original_model_weights.pt')
        torch.save(original_model.state_dict(), f'{weights_dir}/original_model/original_model_weights.pt')
        
    else:
        print("Loading pre-trained original model weights")
        original_model_sd = torch.load(f'{weights_dir}/original_model/original_model_weights.pt')
        original_model.load_state_dict(original_model_sd)
        
    ## Copy original model
    unlearned_model = copy.deepcopy(original_model)
    
    # ============= Initialize retrained model =============
    os.makedirs(f'{weights_dir}/retrained_model', exist_ok=True)
    if cfg.model.model_type == 'neural-network':
        print("Initializing retrained model as a neural network")
        retrained_model = build_nn(M=dataloader_train.dataset.X.shape[-1], cfg=cfg).to(DEVICE)
        save_checkpoints = True #False
    
    elif cfg.model.model_type == 'vision-transformer':
        print("Initializing retrained model as a vision transformer")
        retrained_model = build_vit(cfg)
        save_checkpoints = True
    
    # ============= Train/load retrained model =============
    if not os.path.exists(f'{weights_dir}/retrained_model/retrained_model_weights.pt'):
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
                                        checkpoint_dir = f'{weights_dir}/retrained_model',
                                        disable_tqdm=cfg.trainer.disable_tqdm, 
                                        do_early_stopping=cfg.trainer.do_early_stopping)
        
        retrained_train_fn = lambda: trainer()
        retrained_experiment_specs = ExperimentSpecs(retrained_model, dataloader_train, dataloader_retain, 
                                                    dataloader_forget, dataloader_val, results_dir, 'Retrained model', 
                                                    cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        run_and_eval(retrained_train_fn, retrained_experiment_specs)
        torch.save(retrained_model.state_dict(), f'{weights_dir}/retrained_model/retrained_model_weights.pt')
    
    else:
        print("Loading pre-trained retrained model weights")
        retrained_model_sd = torch.load(f'{weights_dir}/retrained_model/retrained_model_weights.pt')
        retrained_model.load_state_dict(retrained_model_sd)
    
    # plot_wrong_predictions(retrained_model, dataloader_forget, DEVICE, 'retrained_model')
    # plot_wrong_predictions(original_model, dataloader_forget, DEVICE, 'original_model')
    

    mia_model = MIA(device=DEVICE)
    retrained_model_metrics = calculate_model_metrics(retrained_model, {"retain": dataloader_retain, "forget": dataloader_forget, "val": dataloader_val}, DEVICE, 'Retrained model')
    original_model_metrics = calculate_model_metrics(original_model, {"retain": dataloader_retain, "forget": dataloader_forget, "val": dataloader_val}, DEVICE, 'Original model')

    # save metrics to json
    with open(f'{results_dir}/retrained_model_metrics.json', 'w') as f:
        json.dump(retrained_model_metrics, f)
    with open(f'{results_dir}/original_model_metrics.json', 'w') as f:
        json.dump(original_model_metrics, f)


    # ========================= Teacher Ascender =========================
    print("Initializing Teacher Ascender")
    from src.unlearners.teacher_ascend import TeacherAscender
    hyperparams = {'n_epochs': 100, '_lambda': 64}

    mia_model = MIA(device=DEVICE)
    ta_ce = TeacherAscender(copy.deepcopy(unlearned_model), n_epochs=hyperparams['n_epochs'], 
                            _lambda=hyperparams['_lambda'], device=DEVICE, MIA=mia_model)
    ta_entropy = TeacherAscender(copy.deepcopy(unlearned_model), n_epochs=hyperparams['n_epochs'], 
                            _lambda=hyperparams['_lambda'], device=DEVICE, MIA=mia_model)
    te_ce_retain = TeacherAscender(copy.deepcopy(unlearned_model), n_epochs=hyperparams['n_epochs'], 
                            _lambda=hyperparams['_lambda'], device=DEVICE, MIA=mia_model)
    te_entropy_retain = TeacherAscender(copy.deepcopy(unlearned_model), n_epochs=hyperparams['n_epochs'], 
                            _lambda=hyperparams['_lambda'], device=DEVICE, MIA=mia_model)
    
    dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                        dataloader_forget, dataloader_val,
                                                                                                        cfg.model.seed)
    print("Running Teacher Ascender (original-ce)")
    ta_ce_metrics = ta_ce(dataloader_retain, dataloader_forget, dataloader_val, eval=True, version='original-ce')
    # Save metrics to json
    with open(f'{results_dir}/ta_ce_metrics.json', 'w') as f:
        json.dump(ta_ce_metrics, f)

    print("Running Teacher Ascender (original-entropy)")
    ta_entropy_metrics = ta_entropy(dataloader_retain, dataloader_forget, dataloader_val, eval=True, version='original-entropy')
    # Save metrics to json
    with open(f'{results_dir}/ta_ce_metrics.json', 'w') as f:
        json.dump(ta_ce_metrics, f)
    
    print("Running Teacher Ascender (original-ce-retain)")
    te_ce_retain_metrics = te_ce_retain(dataloader_retain, dataloader_forget, dataloader_val, eval=True, version='original-ce-retain')
    # Save metrics to json
    with open(f'{results_dir}/te_ce_retain_metrics.json', 'w') as f:
        json.dump(te_ce_retain_metrics, f)
    
    
    print("Running Teacher Ascender (original-entropy-retain)")
    te_entropy_retain_metrics = te_entropy_retain(dataloader_retain, dataloader_forget, dataloader_val, eval=True, version='original-entropy-retain')
    # Save metrics to json
    with open(f'{results_dir}/te_ce_retain_metrics.json', 'w') as f:
        json.dump(te_ce_retain_metrics, f)

    with open(f'{results_dir}/te_entropy_retain_metrics.json', 'w') as f:
        json.dump(te_entropy_retain_metrics, f)
    dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                        dataloader_forget, dataloader_val,
                                                                                                        cfg.model.seed)
    
    # ========================= Gradient Ascent =========================
    print("Initializing Gradient Ascent")
    from src.unlearners.gradient_ascent import GradientAscent
    dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                        dataloader_forget, dataloader_val,
                                                                                                        cfg.model.seed)
    gradient_ascent = GradientAscent(copy.deepcopy(unlearned_model), hyperparams['n_epochs'], DEVICE, MIA=mia_model)
    ga_metrics = gradient_ascent(dataloader_retain, dataloader_forget, dataloader_val, verbose=True)
    # save metrics to json
    with open(f'{results_dir}/ga_metrics.json', 'w') as f:
        json.dump(ga_metrics, f)



    # ========================= Merge all metrics =========================
    # Merge all metrics into a single dictionary
    all_metrics = {'Gradient Ascent': ga_metrics,
                   'Teacher Ascent (original-ce)': ta_ce_metrics,
                   'Teacher Ascent (original-entropy)': ta_entropy_metrics,
                   'Teacher Ascent (original-ce-retain)': te_ce_retain_metrics,
                   'Teacher Ascent (original-entropy-retain)': te_entropy_retain_metrics}
    
    # Save all metrics to a json file   
    with open(f'{results_dir}/all_metrics.json', 'w') as f:
        json.dump(all_metrics, f)

    print("Plotting model accuracies")
    plot = plot_model_accuracies(all_metrics)
    plot.savefig(f'{results_dir}/model_accuracies.png')

    plot = plot_mia_metrics(all_metrics)
    plot.savefig(f'{results_dir}/mia_metrics.png')
    
    plot = plot_model_accuracies(all_metrics)
    plot.savefig(f'{results_dir}/model_accuracies.png')
    print("Experiment completed successfully")

if __name__ == "__main__":
    main()