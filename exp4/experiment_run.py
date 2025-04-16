import hydra
import wandb
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from omegaconf import OmegaConf
from src.datasets.synthetic_dataset import SyntheticDataset
import copy
import graphviz
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from src.models.neural_network import NeuralNetRS
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



"""
TODO:
    - Check det kører på MPS
    - Check retrained performance på 100 forget punkter
"""



def eval_single_model(model, 
                      dataloader_retain, 
                      dataloader_forget, 
                      dataloader_val,
                      time: float,
                      result_dir: str,
                      dataset_name: str,
                      model_name: str,
                      hyperparameters,
                      seed: int,
                      device: str):
    if os.path.exists(f'{result_dir}/{dataset_name}/all_results.json'):
        with open(f'{result_dir}/{dataset_name}/all_results.json', 'r') as f:
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

    mia_model = MIA(device=device)
    UE = UnlearningEvaluator(device=device)

    retain_logits, y_retain = UE.get_model_logits(model, dataloader_retain)
    forget_logits, y_forget = UE.get_model_logits(model, dataloader_forget)
    val_logits, y_val = UE.get_model_logits(model, dataloader_val)
    
    results['model-name'].append(model_name)
    results['retain-accuracy'].append(UE.calculate_accuracy(retain_logits, y_retain))
    results['forget-accuracy'].append(UE.calculate_accuracy(forget_logits, y_forget))
    results['val-accuracy'].append(UE.calculate_accuracy(val_logits, y_val))
    results['MIA-probability'].append(mia_model(model, dataloader_retain, dataloader_forget, dataloader_val))
    results['time (sec)'].append(time)
    results['hyperparameters'].append(hyperparameters)
    results['seed'].append(seed)
    
    """
    2 cases:
    
    99% retain accuracy for unlearned + retrained
        - JS divergence: unlearned er: [1,0] og retrained er [0.51, 0.49]
    
    50% retain accuracy for unlearned + retrained
        - 
    
    """

    import pdb; pdb.set_trace()
    
    with open(f'{result_dir}/{dataset_name}/all_results.json', 'w') as f:
        json.dump(results, f)
        
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

@hydra.main(config_path=".", config_name="config")
def main(cfg):
    absolute_root_path = os.path.dirname(__file__) if cfg.absolute_root_path == 'local' else cfg.absolute_root_path
    DEVICE = 'mps' #('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_dir = os.path.join(absolute_root_path, 'data')
    results_dir = os.path.join(absolute_root_path, 'results')
    weights_dir = os.path.join(absolute_root_path, 'weights')
    os.makedirs(results_dir, exist_ok=True)

    # ============= Load data and prepare data =============
    dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, forget_idxs = get_image_unlearn_data(root_dir=dataset_dir,
                                                                                                    dataset_name=cfg['data']['dataset_name'],
                                                                                                    n_forget_points=cfg.data.n_forget_points,
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
    
    # ============= Train/load original model =============
    os.makedirs(f'{weights_dir}/{cfg["data"]["dataset_name"]}/original_model', exist_ok=True)
    original_model = NeuralNetRS(M=dataloader_train.dataset.X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed).to(DEVICE)
    if not os.path.exists(f'{weights_dir}/{cfg["data"]["dataset_name"]}/original_model/original_model_weights.pt'):
        start = time.time()
        NeuralNetworkTrainer(model=original_model, 
                             train_dataloader=dataloader_train, 
                             val_dataloader=dataloader_val, 
                             logger=logger, 
                             device=DEVICE, 
                             learning_rate=cfg.trainer.lr, 
                             n_epochs=cfg.trainer.n_epochs, 
                             disable_tqdm=cfg.trainer.disable_tqdm, 
                             do_early_stopping=cfg.trainer.do_early_stopping)()
        end = time.time()
        original_train_time = end-start
        torch.save(original_model.state_dict(), f'{weights_dir}/{cfg["data"]["dataset_name"]}/original_model/original_model_weights.pt')
        original_model_results = eval_single_model(original_model, dataloader_retain, dataloader_forget, dataloader_val, 
                                                   time=original_train_time, result_dir=results_dir, dataset_name=cfg["data"]["dataset_name"], 
                                                   model_name='original_model', hyperparameters={}, seed=cfg.model.seed, device=DEVICE)
    else:
        original_model_sd = torch.load(f'{weights_dir}/{cfg["data"]["dataset_name"]}/original_model/original_model_weights.pt')
        original_model.load_state_dict(original_model_sd)
    
    ## Copy original model
    unlearned_model = copy.deepcopy(original_model)
    
    # ============= Train/load retrained model =============
    os.makedirs(f'{weights_dir}/{cfg["data"]["dataset_name"]}/retrained_model', exist_ok=True)
    retrained_model = NeuralNetRS(M=dataloader_retain.dataset.X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed).to(DEVICE)
    if not os.path.exists(f'{weights_dir}/{cfg["data"]["dataset_name"]}/retrained_model/retrained_model_weights.pt'):
        start = time.time()
        NeuralNetworkTrainer(model=retrained_model, 
                             train_dataloader=dataloader_retain, 
                             val_dataloader=dataloader_val, 
                             logger=logger, 
                             device=DEVICE, 
                             learning_rate=cfg.trainer.lr, 
                             n_epochs=cfg.trainer.n_epochs, 
                             disable_tqdm=cfg.trainer.disable_tqdm, 
                             do_early_stopping=cfg.trainer.do_early_stopping)()
        end=time.time()
        retrain_time = end-start
        torch.save(retrained_model.state_dict(), f'{weights_dir}/{cfg["data"]["dataset_name"]}/retrained_model/retrained_model_weights.pt')
        retrained_model_results = eval_single_model(retrained_model, dataloader_retain, dataloader_forget, dataloader_val, 
                                                    time=retrain_time, result_dir=results_dir, dataset_name=cfg["data"]["dataset_name"], 
                                                    model_name='retrained_model', hyperparameters={}, seed=cfg.model.seed, device=DEVICE)
    else:
        retrained_model_sd = torch.load(f'{weights_dir}/{cfg["data"]["dataset_name"]}/retrained_model/retrained_model_weights.pt')
        retrained_model.load_state_dict(retrained_model_sd)

    import pandas as pd
    with open(f'{results_dir}/{cfg["data"]["dataset_name"]}/all_results.json', 'r') as f:
        res_dict = json.load(f)
    df_results = pd.DataFrame.from_dict(res_dict)
    import pdb; pdb.set_trace()

    if cfg.unlearn.method == "ssd":
        from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening
        hyperparams = {'alpha': 5.,
                        '_lambda': 3.}
        # hyperparams = {'alpha': 0.1,
        #                 '_lambda': 1.}
        
        start = time.time()
        ssd = SelectiveSynapticDampening(unlearned_model, 
                                    alpha=hyperparams["alpha"], 
                                    _lambda=hyperparams["_lambda"],
                                    device=DEVICE)
        ssd(dataloader_train, dataloader_forget)
        end = time.time()
        unlearn_time = end-start
        eval_single_model(unlearned_model, dataloader_retain, dataloader_forget, dataloader_val, unlearn_time, results_dir,
                          cfg["data"]["dataset_name"], 'SSD', hyperparams, seed=cfg.model.seed, device=DEVICE)

    elif cfg.unlearn.method == "scrubr":
        from src.unlearners.scrub import ScrubR
        hyperparams = {'alpha': 2,
                       'gamma': 2,
                       'nrounds': 3}
        start = time.time()
        ScrubR(model=unlearned_model, 
               original_model=original_model,
               alpha=hyperparams['alpha'],
               gamma=hyperparams['gamma'],
               device=DEVICE)(
                                            retain_dataloader=dataloader_retain, 
                                            forget_dataloader=dataloader_forget, 
                                            val_dataloader=dataloader_val, 
                                            n_rounds=hyperparams['nrounds']
                                          )
        
        end = time.time()
        unlearn_time = end-start
        eval_single_model(unlearned_model, dataloader_retain, dataloader_forget, dataloader_val, 
                          unlearn_time, results_dir, cfg["data"]["dataset_name"], 'Scrub+R', hyperparams, seed=cfg.model.seed, device=DEVICE)
        
    elif cfg.unlearn.method == 'assd':
        from src.unlearners.adaptive_ssd import AdaptiveSSD
        hyperparams = {}
        adaptive_ssd = AdaptiveSSD(unlearned_model, device=DEVICE)
        start = time.time()
        adaptive_ssd(dataloader_train, dataloader_forget)
        end = time.time()
        unlearn_time = end-start
        eval_single_model(unlearned_model, dataloader_retain, dataloader_forget, dataloader_val, 
                          unlearn_time, results_dir, cfg["data"]["dataset_name"], 'Adaptive SSD', hyperparams, seed=cfg.model.seed, device=DEVICE)

        
    elif cfg.unlearn.method == 'ssd_v5':
        from src.unlearners.selective_synaptic_dampening_v5 import SelectiveSynapticDampening
        hyperparams = {'P': 3}
        start = time.time()
        ssd = SelectiveSynapticDampening(unlearned_model, P=hyperparams['P'], device=DEVICE)
        learned_hyperparams = ssd(full_dataloader=dataloader_train, 
                                  forget_dataloader=dataloader_forget,
                                  validation_dataloader=dataloader_val)
        end = time.time()
        unlearn_time = end-start
        eval_single_model(unlearned_model, dataloader_retain, dataloader_forget, dataloader_val, 
                          unlearn_time, results_dir, cfg["data"]["dataset_name"], 'SSD v5', hyperparams, seed=cfg.model.seed, device=DEVICE)

    elif cfg.unlearn.method == 'sae':
        from src.models.SAE import SAE
        from src.models.neural_net_with_sae import NeuralNetWithSAE
        from src.trainers.sae_trainer import SAETrainer
        from src.unlearners.sae_unlearner import SAEUnlearner
        hyperparams = {'layer_num': 6, '_lambda': 1, 'alpha': 0.9}
        hyperparams.update({'m': 4 * unlearned_model.net[hyperparams['layer_num']-2].lin_layer.out_features})

        start = time.time()
        # instantiate & train SAE
        sae = SAE(d=unlearned_model.net[hyperparams['layer_num']-2].in_features, m=hyperparams['m'], _lambda=hyperparams['_lambda']).to(DEVICE)
        unlearned_model = NeuralNetWithSAE(unlearned_model, sae, layer_num=hyperparams['layer_num'])
        sae_trainer = SAETrainer(unlearned_model, dataloader_train, dataloader_val, device=DEVICE)
        sae_trainer.train()

        # unlearn with feature dampening
        sae_unlearner = SAEUnlearner(unlearned_model, alpha=hyperparams['alpha'])
        sae_unlearner(dataloader_retain, dataloader_forget)
        end = time.time()
        unlearn_time = end-start

        eval_single_model(unlearned_model, dataloader_retain, dataloader_forget, dataloader_val, 
                          unlearn_time, results_dir, cfg["data"]["dataset_name"], 'SAE', hyperparams, seed=cfg.model.seed, device=DEVICE)

    elif cfg.unlearn.method == 'amnesiac':
        from src.models.amnesiac_model import AmnesiacModelRS
        from src.trainers.amnesiac_trainer import AmnesiacTrainer
        from src.unlearners.amnesiac_unlearner import AmnesiacUnlearner
        from src.utils.misc import check_statedict_equivalent

        hyperparams = {'repair': True}
        # "simulate original model training with saved gradients"
        unlearned_model = AmnesiacModelRS(NeuralNetRS(M=dataloader_train.dataset.X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)).to(DEVICE)
        AmnesiacTrainer(model=unlearned_model,
                        train_dataloader=dataloader_train, 
                        val_dataloader=dataloader_val, 
                        logger=logger, 
                        device=DEVICE, 
                        learning_rate=cfg.trainer.lr, 
                        n_epochs=cfg.trainer.n_epochs, 
                        disable_tqdm=cfg.trainer.disable_tqdm, 
                        do_early_stopping=cfg.trainer.do_early_stopping,
                        cache_gradients=False,
                        save_dir=os.path.join(cfg['amnesiac']['gradient_checkpoint_dir'], f"{cfg['data']['dataset_name']}/amnesiac/")
                        )(indices_to_forget=forget_idxs
                          )
        
        assert check_statedict_equivalent(unlearned_model.state_dict(), original_model.state_dict()), "Amnesiac model's state dict is not identical to the original model's!\nComparison with the other methods is unfair."
        
        start = time.time()
        AmnesiacUnlearner(model=unlearned_model, 
                          unlearn_parameters=cfg.unlearn)(indices_to_forget=forget_idxs)
        end=time.time()
        unlearn_time = end-start
        eval_single_model(unlearned_model, dataloader_retain, dataloader_forget, dataloader_val, unlearn_time, results_dir, 
                          cfg["data"]["dataset_name"], 'Amnesiac', hyperparameters={'repair': False}, seed=cfg.model.seed, device=DEVICE)
        
        if hyperparams['repair']: # run a repair phase after rolling back gradients
            start = time.time()
            AmnesiacTrainer(model=unlearned_model, 
                        train_dataloader=dataloader_retain, 
                        val_dataloader=dataloader_val, 
                        logger=logger, 
                        device=cfg.model.device, 
                        learning_rate=cfg.trainer.lr, 
                        n_epochs=cfg.trainer.n_epochs, 
                        disable_tqdm=cfg.trainer.disable_tqdm, 
                        do_early_stopping=cfg.trainer.do_early_stopping,
                        cache_gradients=True
                        )(repair=True
                          )

            end = time.time()
            unlearn_time = unlearn_time + end-start # accounts for the fact that MIA takes some time
            eval_single_model(unlearned_model, dataloader_retain, dataloader_forget, dataloader_val, unlearn_time, results_dir, 
                              cfg["data"]["dataset_name"], 'Amnesiac', hyperparams, seed=cfg.model.seed, device=DEVICE)
    
    elif cfg.unlearn.method =='teacher-ascend':
        from src.unlearners.teacher_ascend import TeacherAscender
        ta = TeacherAscender(unlearned_model, n_epochs=20, device=DEVICE)
        
        start = time.time()
        ta(dataloader_retain, dataloader_forget)
        end = time.time()
        unlearn_time = end-start
        
        eval_single_model(unlearned_model, dataloader_retain, dataloader_forget, dataloader_val, unlearn_time, results_dir, 
                              cfg["data"]["dataset_name"], 'Teacher ascend', hyperparams, seed=cfg.model.seed, device=DEVICE)
        
        
    else:
        raise NotImplementedError(f"Unlearning method {cfg.unlearn.method} not implemented")

if __name__ == "__main__":
    main()
