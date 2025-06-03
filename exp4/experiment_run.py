import hydra
import wandb
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from src.datasets.synthetic_dataset import SyntheticDataset
import copy
import graphviz
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from torch.utils.data import RandomSampler, SequentialSampler

from src.models.neural_network import NeuralNetRS
from src.models.vision_transformer_tiny import ViT
from src.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.evaluation.decision_boundary import DecisionBoundaryCreator
from src.evaluation.membership_inference_attack import MIA
from src.evaluation.unlearning_evaluator import UnlearningEvaluator
from prepare_image_data import get_image_unlearn_data
import time
import json

""" AMNESIAC ANALYSE
10000 punkter total
100 unlearn punkter

batch size 16

10000/ 16 = 625 total batches pr. epoch

100/625 = 16% sandsynlighed for at et forget punkt er indeholdt i et batch.

Det her beskriver casen hvor vi specifikt gemmer gradienter for forget punkter:
Worst-case:
Hvert "forget batch" indeholder præcis ét forget punkt.
Gem 100 gradienter.

Best-case:
Hvert "forget batch" indeholder kun forget punkter.
Gem ceil(100/16) = ceil(6.25) = 7 gradienter.

Average-case:
Ved konstruktionen af første batch vælger vi 16 punkter ud af 10000 og sampler uden replacement.

Ved konstruktionen af andet batch har vi samplet alt mellem 0 og 16 forget punkter allerede og vi har 10000 - 16 punkter total at vælge udfra.

Ved konstruktion af det k'te batch har vi samplet alt mellem 0 og min(k * 16, 100) forget punkter og vi har 10000 - 16 * k punker total at vælge udfra.

"""

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

    import pdb; pdb.set_trace()

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

# @hydra.main(config_path=".", config_name="cifar_config")
@hydra.main(config_path=".", config_name="mnist_config")
def main(cfg):
    absolute_root_path = os.path.dirname(__file__) if cfg.absolute_root_path == 'local' else cfg.absolute_root_path
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    if DEVICE == 'cuda':
        torch.cuda.manual_seed(cfg.model.seed)
        torch.cuda.manual_seed_all(cfg.model.seed)
        torch.backends.cudnn.deterministic = True
    
    print(f'Running experiment 4 on device "{DEVICE}"!')
    print(f"All results will be saved to: {absolute_root_path}")
    dataset_dir = os.path.join(absolute_root_path, 'data')
    results_dir = os.path.join(absolute_root_path, 'results', f'{cfg.data.dataset_name}', f'seed_{cfg.model.seed}')
    weights_dir = os.path.join(absolute_root_path, 'weights', f'{cfg.data.dataset_name}', f'seed_{cfg.model.seed}')
    os.makedirs(results_dir, exist_ok=True)
    
    # ============= Load data and prepare data =============
    if cfg.data.dataset_name == 'MNIST':
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, forget_idxs = get_image_unlearn_data(root_dir=dataset_dir,
                                                                                                                 dataset_name=cfg['data']['dataset_name'],
                                                                                                                 n_forget_points=cfg.data.n_forget_points,
                                                                                                                 subsample_size=cfg.data.subsample_size,
                                                                                                                 patch_size=cfg.data.patch_size,
                                                                                                                 batch_size=cfg.data.batch_size,
                                                                                                                 seed=cfg.model.seed)
    else:
        dataloader_train, dataloader_retain, dataloader_retain_no_aug, dataloader_forget, dataloader_val, forget_idxs = get_image_unlearn_data(root_dir=dataset_dir,
                                                                                                                 dataset_name=cfg['data']['dataset_name'],
                                                                                                                 n_forget_points=cfg.data.n_forget_points,
                                                                                                                 subsample_size=cfg.data.subsample_size,
                                                                                                                 patch_size=cfg.data.patch_size,
                                                                                                                 batch_size=cfg.data.batch_size,
                                                                                                                 seed=cfg.model.seed)

    assert (dataloader_train.dataset.y[forget_idxs] == dataloader_forget.dataset.y).all(), 'Forget indices applied to train do not correspond to the forget data!'

    # ============= Initialize logger =============
    if cfg.logging.logger == "wandb":
        cfg_dict = OmegaConf.to_container(cfg, resolve=True) # ? Convert to dict to avoid issues with wandb

        os.environ["WANDB_MODE"] = cfg.logging.wandb.mode
        wandb.setup(settings=wandb.Settings(mode=cfg.logging.wandb.mode))
        logger = wandb.init(project=cfg.logging.project, 
                            config=cfg_dict, 
                            name=cfg.logging.name, 
                            group=cfg.logging.group,
                            mode=cfg.logging.wandb.mode,
                            dir=cfg.logging.dir)

    else:
        raise NotImplementedError(f"Logger {cfg.logging.logger} not implemented")
    
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
        torch.save(original_model.state_dict(), f'{weights_dir}/original_model/original_model_weights.pt')
        
    else:
        # original_model_sd = torch.load(f'{weights_dir}/{cfg["data"]["dataset_name"]}/original_model/original_model_val-acc=0.7525712025316458_epoch=164.pt', map_location=DEVICE)
        original_model_sd = torch.load(f'{weights_dir}/original_model/original_model_weights.pt')
        original_model.load_state_dict(original_model_sd)
        
    ## Copy original model
    unlearned_model = copy.deepcopy(original_model)
    
    # ============= Initialize retrained model =============
    os.makedirs(f'{weights_dir}/retrained_model', exist_ok=True)
    if cfg.model.model_type == 'neural-network':
        retrained_model = build_nn(M=dataloader_train.dataset.X.shape[-1], cfg=cfg).to(DEVICE)
        save_checkpoints = True #False
    
    elif cfg.model.model_type == 'vision-transformer':
        retrained_model = build_vit(cfg)
        save_checkpoints = True
    
    # ============= Train/load retrained model =============
    if not os.path.exists(f'{weights_dir}/retrained_model/retrained_model_weights.pt'):
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
        retrained_model_sd = torch.load(f'{weights_dir}/retrained_model/retrained_model_weights.pt')
        retrained_model.load_state_dict(retrained_model_sd)
    
    plot_wrong_predictions(retrained_model, dataloader_forget, DEVICE, 'retrained_model')
    plot_wrong_predictions(original_model, dataloader_forget, DEVICE, 'original_model')

    # import pandas as pd
    # with open(f'{results_dir}/{cfg["data"]["dataset_name"]}/all_results.json', 'r') as f:
    #     res_dict = json.load(f)
    # df_results = pd.DataFrame.from_dict(res_dict)
    # import pdb; pdb.set_trace()

    if cfg.unlearn.method == "ssd":
        from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening
        hyperparams = {'alpha': 5.,
                        '_lambda': 3.}
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)
        ssd = SelectiveSynapticDampening(unlearned_model, 
                                         alpha=hyperparams["alpha"], 
                                         _lambda=hyperparams["_lambda"],
                                          device=DEVICE)
        ssd_experiment_specs = ExperimentSpecs(unlearned_model, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                                               results_dir, 'SSD', cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        apply_ssd_fn = lambda: ssd(dataloader_train, dataloader_forget)
        run_and_eval(apply_ssd_fn, ssd_experiment_specs)

    elif cfg.unlearn.method == "scrubr":
        from src.unlearners.scrub import ScrubR
        hyperparams = {'alpha': 2,
                       'gamma': 2,
                       'nrounds': 3}
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)
        scrubr = ScrubR(model=unlearned_model, 
               original_model=original_model,
               alpha=hyperparams['alpha'],
               gamma=hyperparams['gamma'],
               device=DEVICE)
        scrubr_experiment_specs = ExperimentSpecs(unlearned_model, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                                               results_dir, 'Scrub+R', cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        apply_scrubr_fn = lambda: scrubr(retain_dataloader=dataloader_retain, forget_dataloader=dataloader_forget, 
                                         val_dataloader=dataloader_val, n_rounds=hyperparams['nrounds'])
        run_and_eval(apply_scrubr_fn, scrubr_experiment_specs)

    elif cfg.unlearn.method == 'assd':
        from src.unlearners.adaptive_ssd import AdaptiveSSD
        hyperparams = {}
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)
        adaptive_ssd = AdaptiveSSD(unlearned_model, device=DEVICE)
        assd_experiment_specs = ExperimentSpecs(unlearned_model, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                                               results_dir, 'Adaptive SSD', cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        adaptive_ssd_fn = lambda: adaptive_ssd(dataloader_train, dataloader_forget)
        run_and_eval(adaptive_ssd_fn, assd_experiment_specs)

    elif cfg.unlearn.method == 'ssd_v6':
        from src.unlearners.selective_synaptic_dampening_v6 import SelectiveSynapticDampening
        hyperparams = {'P': 3, 'k': 0.75, 'smooth_dampening': False, 'n_bo_iter': 20}
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)
        ssd = SelectiveSynapticDampening(unlearned_model, P=hyperparams['P'], k=hyperparams['k'], 
                                         smooth_dampening=hyperparams['smooth_dampening'], n_bo_iter=hyperparams['n_bo_iter'], 
                                         device=DEVICE)
        ssd_v6_fn = lambda: ssd(dataloader_train, dataloader_forget, dataloader_val)
        ssd_v6_experiment_specs = ExperimentSpecs(unlearned_model, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                                               results_dir, 'SSD v6', cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        run_and_eval(ssd_v6_fn, ssd_v6_experiment_specs)
    
    elif cfg.unlearn.method == 'ssd_v6_smooth':
        from src.unlearners.selective_synaptic_dampening_v6 import SelectiveSynapticDampening
        hyperparams = {'P': 3, 'k': 0.75, 'smooth_dampening': True, 'n_bo_iter': 20}
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)
        ssd = SelectiveSynapticDampening(unlearned_model, P=hyperparams['P'], k=hyperparams['k'], 
                                         smooth_dampening=hyperparams['smooth_dampening'], n_bo_iter=hyperparams['n_bo_iter'], 
                                         device=DEVICE)
        ssd_v6_fn = lambda: ssd(dataloader_train, dataloader_forget, dataloader_val)
        ssd_v6_experiment_specs = ExperimentSpecs(unlearned_model, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                                               results_dir, 'SSD v6', cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        run_and_eval(ssd_v6_fn, ssd_v6_experiment_specs)
    
    elif cfg.unlearn.method == 'ssd_v7':
        from src.unlearners.selective_synaptic_dampening_v7 import SelectiveSynapticDampening
        hyperparams = {'P': 1, 'k': 0.75, 'n_trials': 100}
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)
        ssd = SelectiveSynapticDampening(unlearned_model, P=hyperparams['P'],k=hyperparams['k'],n_trials=hyperparams['n_trials'], device=DEVICE)
        ssd_v7_fn = lambda: ssd(dataloader_train, dataloader_forget, dataloader_val)
        ssd_v7_experiment_specs = ExperimentSpecs(unlearned_model, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                                               results_dir, 'SSD v7', cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        run_and_eval(ssd_v7_fn, ssd_v7_experiment_specs)
        
    elif cfg.unlearn.method == 'sae':
        from src.models.SAE import SAE
        from src.models.neural_net_with_sae import NeuralNetWithSAE
        from src.trainers.sae_trainer import SAETrainer
        from src.unlearners.sae_unlearner import SAEUnlearner
        hyperparams = {'layer_num': 6, '_lambda': 0.01, 'alpha': 0.5}
        hyperparams.update({'m': 6 * unlearned_model.net[hyperparams['layer_num']-2].lin_layer.out_features})
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)
        
        # instantiate & train SAE
        sae = SAE(d=unlearned_model.net[hyperparams['layer_num']-2].in_features, m=hyperparams['m'], _lambda=hyperparams['_lambda']).to(DEVICE)
        
        unlearned_model = NeuralNetWithSAE(unlearned_model, sae, layer_num=hyperparams['layer_num'])
        sae_trainer = SAETrainer(unlearned_model, dataloader_train, dataloader_val, device=DEVICE)
        sae_train_fn = lambda: sae_trainer.train()
        sae_train_specs = ExperimentSpecs(unlearned_model, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                                               results_dir, 'SAE original', cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        run_and_eval(sae_train_fn, sae_train_specs)

        # unlearn with SAE
        sae_unlearner = SAEUnlearner(unlearned_model, alpha=hyperparams['alpha'], device=DEVICE)
        sae_unlearn_fn = lambda: sae_unlearner(dataloader_retain, dataloader_forget)
        sae_unlearn_specs = ExperimentSpecs(unlearned_model, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                                               results_dir, 'SAE', cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        run_and_eval(sae_unlearn_fn, sae_unlearn_specs)

    elif cfg.unlearn.method == 'amnesiac':
        from src.models.amnesiac_model import AmnesiacModelRS
        from src.trainers.amnesiac_trainer import AmnesiacTrainer
        from src.unlearners.amnesiac_unlearner import AmnesiacUnlearner
        from src.utils.misc import check_statedict_equivalent
        hyperparams = {'repair': False, 'n_repair_epochs': 0}
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)
        # "simulate original model training with saved gradients"
        unlearned_model = AmnesiacModelRS(NeuralNetRS(M=dataloader_retain.dataset.X.shape[1], 
                                                      n_classes=cfg.data.n_classes, 
                                                      n_layers=cfg.model.neural_network_parameters.n_layers,
                                                      width_factor=cfg.model.neural_network_parameters.width_factor, seed=cfg.model.seed)).to(DEVICE)
        amnesiac_trainer = AmnesiacTrainer(model=unlearned_model,
                        train_dataloader=dataloader_train, 
                        val_dataloader=dataloader_val, 
                        logger=logger,
                        device=DEVICE, 
                        learning_rate=cfg.trainer.lr,
                        weight_decay=cfg.trainer.weight_decay,
                        n_epochs=cfg.trainer.n_epochs, 
                        disable_tqdm=cfg.trainer.disable_tqdm,
                        do_early_stopping=cfg.trainer.do_early_stopping,
                        cache_gradients=False,
                        save_dir=f'{weights_dir}/amnesiac',
                        checkpoint_dir = f'{weights_dir}/original_model',
                        )
        amnesiac_original_train_fn = lambda: amnesiac_trainer(indices_to_forget=forget_idxs)
        amnesiac_original_experiment_specs = ExperimentSpecs(unlearned_model, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                                               results_dir, 'Amnesiac original', cfg.data.dataset_name, {}, cfg.model.seed, DEVICE)
        run_and_eval(amnesiac_original_train_fn, amnesiac_original_experiment_specs)
        
        assert check_statedict_equivalent(unlearned_model.state_dict(), original_model.state_dict()), "Amnesiac model's state dict is not identical to the original model's!\nComparison with the other methods is unfair."
        
        ### perform amnesiac unlearning with gradient rollback
        amnesiac_unlearner = AmnesiacUnlearner(model=unlearned_model, unlearn_parameters=cfg.unlearn)
        amnesiac_unlearn_fn = lambda: amnesiac_unlearner(indices_to_forget=forget_idxs)
        amnesiac_experiment_specs = ExperimentSpecs(unlearned_model, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                                               results_dir, 'Amnesiac', cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        
        run_and_eval(amnesiac_unlearn_fn, amnesiac_experiment_specs)
        
        ## run repair phase to get those results as well
        hyperparams['repair'] = True
        hyperparams['n_repair_epochs'] = 3

        if hyperparams['repair']: # run a repair phase after rolling back gradients
            amnesiac_repair_trainer = AmnesiacTrainer(model=unlearned_model, 
                            train_dataloader=dataloader_retain, 
                            val_dataloader=dataloader_val, 
                            logger=logger, 
                            device=DEVICE,
                            learning_rate=cfg.trainer.lr,
                            weight_decay=cfg.trainer.weight_decay,
                            n_epochs=hyperparams['n_repair_epochs'], 
                            disable_tqdm=cfg.trainer.disable_tqdm, 
                            do_early_stopping=cfg.trainer.do_early_stopping,
                            cache_gradients=True
                            )
            amnesiac_repair_fn = lambda: amnesiac_repair_trainer(repair=True)
            amnesiac_repair_experiment_specs = ExperimentSpecs(unlearned_model, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                                               results_dir, 'Amnesiac', cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
            
            run_and_eval(amnesiac_repair_fn, amnesiac_repair_experiment_specs)

        # free up space by removing gradients
        import shutil
        shutil.rmtree(path=f'{weights_dir}/amnesiac')
    
    elif cfg.unlearn.method == 'sisa':
        from src.unlearners.sisa_unlearner import SISAUnlearner
        from src.models.sisa_class import SISA
        from src.trainers.sisa_trainer import SISATrainer
        hyperparams = {'n_epochs_without_slicing': 10}
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)
        if cfg.sisa.model_type == 'neural-network':
            sisa_model_fn = NeuralNetRS
            sisa_model_parameters = {'M': dataloader_train.dataset.X.shape[1],
                                     'n_classes': cfg.data.n_classes,
                                     'width_factor': cfg.sisa.neural_network_parameters.width_factor,
                                     'n_layers': cfg.sisa.neural_network_parameters.n_layers,
                                     'seed': cfg.model.seed
                                    }
            sisa_optimizer_parameters = {'lr': 0.001}
        else:
            raise NotImplementedError()
        # instantiate sisa class
        sisa = SISA(dataloader_train,
                    dataloader_val,
                    model_fn=sisa_model_fn,
                    model_params=sisa_model_parameters,
                    optimizer_parms=sisa_optimizer_parameters,
                    n_classes=cfg.data.n_classes, 
                    n_epochs=hyperparams['n_epochs_without_slicing'], 
                    n_shards=cfg.sisa.n_shards,
                    n_slices=cfg.sisa.n_slices,
                    save_dir=f'{weights_dir}/sisa',
                    device=DEVICE,
                    seed=cfg.model.seed)
        # train constituent models
        sisa_trainer = SISATrainer(
                model=None, 
                sisa=sisa, 
                train_dataloader=dataloader_train,
                val_dataloader=dataloader_val,
                logger=None, 
                device=DEVICE,
                learning_rate=cfg.trainer.lr, 
                n_epochs=hyperparams['n_epochs_without_slicing'],
                disable_tqdm=cfg.trainer.disable_tqdm, 
                do_early_stopping=cfg.trainer.do_early_stopping)
        sisa_train_fn = lambda: sisa_trainer()
        sisa_train_specs = ExperimentSpecs(sisa, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                                               results_dir, 'SISA original', cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        run_and_eval(sisa_train_fn, sisa_train_specs)

        sisa.constituent_models = None # avoids keeping original constituent models in memory
        # run SISA unlearning
        sisa_unlearner = SISAUnlearner(sisa)
        sisa_unlearn_fn = lambda: sisa_unlearner(forget_indices=forget_idxs)
        sisa_experiment_specs = ExperimentSpecs(sisa, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                                               results_dir, 'SISA', cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        run_and_eval(sisa_unlearn_fn, sisa_experiment_specs)
        # free up space by removing constituent models and data indices
        import shutil
        shutil.rmtree(f'{weights_dir}/sisa')

    elif cfg.unlearn.method =='teacher-ascend':
        from src.unlearners.teacher_ascend import TeacherAscender
        hyperparams = {'n_epochs': 100, '_lambda': 64}
        ta = TeacherAscender(unlearned_model, n_epochs=hyperparams['n_epochs'], 
                             _lambda=hyperparams['_lambda'], device=DEVICE)
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)
        ta_fn = lambda: ta(dataloader_retain, dataloader_forget)
        ta_experiment_specs = ExperimentSpecs(unlearned_model, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
                                               results_dir, 'Teacher Ascend', cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        run_and_eval(ta_fn, ta_experiment_specs)
    
    elif cfg.unlearn.method == "visualize-bo-v6":
        from src.unlearners.ssd_v6_bo_visualizer import SSDVisualizer

        hyperparams = {'P': 1, 'k': 0.75, 'smooth_dampening': False, 'n_bo_iter': 150}
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)
        ssd = SSDVisualizer(unlearned_model, 
                            P=hyperparams['P'], 
                            k=hyperparams['k'], 
                            smooth_dampening=hyperparams['smooth_dampening'], 
                            n_bo_iter=hyperparams['n_bo_iter'], 
                            device=DEVICE)
        ssd_v6_fn = lambda: ssd(dataloader_train, dataloader_forget, dataloader_val)
        bo_result = ssd_v6_fn()

        alphas = np.array([bo_result['result'][i]['params']['alpha_0'] for i in range(len(bo_result['result']))])
        targets = np.array([bo_result['result'][i]['target'] for i in range(len(bo_result['result']))])
        # sorted_idx = np.argsort(alphas)
        
        sampling_order = np.linspace(0, 1, len(alphas))

        plt.figure()
        sc = plt.scatter(alphas, targets, c=sampling_order, cmap='coolwarm', s=50, edgecolor='k')
        # plt.plot(alphas[sorted_idx], targets[sorted_idx], marker='o', linestyle='-')
        # plt.scatter(alphas, targets)
        plt.xlabel('alpha')
        plt.ylabel('neg. difference')
        plt.title("Objective function vs. alpha (λ=1, P=1)")
        cbar = plt.colorbar(sc)
        cbar.set_label('Sampling order (0 = early, 1 = late)')
        plt.savefig(f'bo_ojective_function.png')
        
        # ssd_v6_experiment_specs = ExperimentSpecs(unlearned_model, dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, 
        #                                        results_dir, 'SSD v6', cfg.data.dataset_name, hyperparams, cfg.model.seed, DEVICE)
        # run_and_eval(ssd_v6_fn, ssd_v6_experiment_specs)

    elif cfg.unlearn.method == "visualize-bo-v7":
        from src.unlearners.ssd_v7_bo_visualizer import SSDVisualizer

        hyperparams = {'P': 1, 'k': 0.75, 'n_trials': 150}
        dataloader_train, dataloader_retain, dataloader_forget, dataloader_val = re_instantiate_dataloaders(dataloader_train, dataloader_retain, 
                                                                                                            dataloader_forget, dataloader_val,
                                                                                                            cfg.model.seed)
        ssd = SSDVisualizer(unlearned_model, 
                            P=hyperparams['P'], 
                            k=hyperparams['k'], 
                            n_trials=hyperparams['n_trials'], 
                            device=DEVICE)
        ssd_v7_fn = lambda: ssd(dataloader_train, dataloader_forget, dataloader_val)
        bo_result = ssd_v7_fn()

        alphas = np.array([bo_result['result'][i]['params']['alpha_0'] for i in range(len(bo_result['result']))])
        targets = np.array([bo_result['result'][i]['target'] for i in range(len(bo_result['result']))])
        # sorted_idx = np.argsort(alphas)
        sampling_order = np.linspace(0, 1, len(alphas))
        plt.figure()
        sc = plt.scatter(alphas, targets, c=sampling_order, cmap='coolwarm', s=50, edgecolor='k')
        # plt.plot(alphas[sorted_idx], targets[sorted_idx], marker='o', linestyle='-')
        # plt.scatter(alphas, targets)
        plt.xlabel('alpha')
        plt.ylabel('neg. difference')
        plt.title("Objective function vs. alpha (λ=1, P=1)")
        cbar = plt.colorbar(sc)
        cbar.set_label('Sampling order (0 = early, 1 = late)')
        plt.savefig(f'bo_ojective_function.png')
    else:
        raise NotImplementedError(f"Unlearning method {cfg.unlearn.method} not implemented")

if __name__ == "__main__":
    main()
