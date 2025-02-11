import argparse
import pdb
import pandas as pd
import torch.nn as nn
import copy

from src.modelling.trainer import Trainer
from src.modelling.neural_network import NeuralNet
from src.data_utils.synthetic_data import DataGenerator, create_dataloaders
from src.modelling.selective_synaptic_dampening import SelectiveSynapticDampening
from src.modelling.scrub import ScrubR
from src.modelling.sae_unlearner import SAEUnlearner
from src.modelling.SAE import SAE

from src.modelling.unlearning_evaluator import UnlearningEvaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='experiment', choices=['experiment', 'visualize'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--unlearn-type', type=str, default='SAE', choices=['SSD', 'Scrub+R', 'SAE'], help='What type of unlearning algorithm to apply.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')    
    
    args = parser.parse_args()
    ### set constants
    ## for the experiment:
    n_repeats = 10
    n_epochs = 20
    ## for the dataset:
    n_features = 25
    n_classes = 4
    ## SSD hyperparameters:
    # alpha = 2.5
    # _lambda = 0.1 #0.1

    ### Create synthetic dataset
    data_generator = DataGenerator(random_state=42)
    data_generator.generate_data(n_samples=1000, n_features=n_features,
                                 n_classes=n_classes, n_informative=2, 
                                 n_redundant=0, n_outliers=50, outlier_scale=4.0, 
                                 outlier_variance=0.4, outlier_class=1)
    data_generator.split_data(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    # data_generator.draw_forget_set(n_points=20, class_idx=1, ood_ratio=0.5)
    data_generator.draw_forget_set(n_points=50, class_idx=None, ood_ratio=0.5)
    dataloaders = create_dataloaders(data_generator, batch_size=32, onehot_labels=True)
    x, y = next(iter(dataloaders['train_full_loader']))

    results = {'unlearned_model': {'retain_acc': [], 'forget_acc': [], 'val_acc': []},
               'retrained_model': {'retain_acc': [], 'forget_acc': [], 'val_acc': []},
               'functional_equivalence': {'retain': [], 'forget': [], 'val': [], 'forget w retrained': []}}
    
    for _ in range(n_repeats):
        # define and train unlearned model on the full dataset
        unlearned_model = NeuralNet(n_features, n_classes)
        unlearned_trainer = Trainer(unlearned_model, train_dataloader=dataloaders['train_full_loader'], val_dataloader=dataloaders['val_loader'], n_epochs=n_epochs)
        unlearned_trainer.train()
        unlearned_trainer.eval()
        # define and train retrained model on the retain dataset
        retrained_model = NeuralNet(n_features, n_classes)
        retrained_trainer = Trainer(retrained_model, train_dataloader=dataloaders['train_retain_loader'], val_dataloader=dataloaders['val_loader'], n_epochs=n_epochs)
        retrained_trainer.train()
        retrained_trainer.eval()
        
        original_model = copy.deepcopy(unlearned_model)
        # print(f'Before scrub:\n{unlearned_model.net[0].weight}')


        if args.unlearn_type == 'Scrub+R':
            scrub = ScrubR(unlearned_model, original_model, alpha=1., gamma=1.)
            scrub(dataloaders['train_retain_loader'], dataloaders['train_forget_loader'], dataloaders['val_loader'], n_rounds=6)
        
        elif args.unlearn_type == 'SSD':
            criterion = nn.CrossEntropyLoss()
            SSD = SelectiveSynapticDampening(unlearned_model, criterion, alpha=2.5, _lambda=0.1)
            SSD(full_dataloader=dataloaders['train_full_loader'], forget_dataloader=dataloaders['train_forget_loader'])
        
        elif args.unlearn_type == 'SAE':
            _lambda = 1.0
            layer_num = 4 # which layer to apply the SAE
            d = unlearned_model.net[layer_num-2].out_features
            sae = SAE(d=d, 
                      m=d*4,
                      _lambda=_lambda)
            
            sae_unlearner = SAEUnlearner(unlearned_model, sae, layer_num)
            sd_before = sae_unlearner.model.state_dict()
            sae_trainer = Trainer(sae_unlearner, train_dataloader=dataloaders['train_full_loader'], val_dataloader=dataloaders['val_loader'], n_epochs=n_epochs)
            sae_trainer.train_sae()
            sd_after = sae_unlearner.model.state_dict()

        
        # evaluate unlearning performance 
        unlearning_evaluator = UnlearningEvaluator()
        
        unlearned_retain_acc = unlearning_evaluator.evaluate_performance(unlearned_model, dataloaders['train_retain_loader'])['accuracy']
        unlearned_forget_acc = unlearning_evaluator.evaluate_performance(unlearned_model, dataloaders['train_forget_loader'])['accuracy']
        unlearned_val_acc = unlearning_evaluator.evaluate_performance(unlearned_model, dataloaders['val_loader'])['accuracy']
        
        retrained_retain_acc = unlearning_evaluator.evaluate_performance(retrained_model, dataloaders['train_retain_loader'])['accuracy']
        retrained_forget_acc = unlearning_evaluator.evaluate_performance(retrained_model, dataloaders['train_forget_loader'])['accuracy']
        retrained_val_acc = unlearning_evaluator.evaluate_performance(retrained_model, dataloaders['val_loader'])['accuracy']
        
        results['unlearned_model']['retain_acc'].append(unlearned_retain_acc)
        results['unlearned_model']['forget_acc'].append(unlearned_forget_acc)
        results['unlearned_model']['val_acc'].append(unlearned_val_acc)

        results['retrained_model']['retain_acc'].append(retrained_retain_acc)
        results['retrained_model']['forget_acc'].append(retrained_forget_acc)
        results['retrained_model']['val_acc'].append(retrained_val_acc)

        retain_func_equiv = unlearning_evaluator.functional_equivalence(original_model, unlearned_model, metric='JS-divergence', dataloader=dataloaders['train_retain_loader'])
        forget_func_equiv = unlearning_evaluator.functional_equivalence(original_model, unlearned_model, metric='JS-divergence', dataloader=dataloaders['train_forget_loader'])
        val_func_equiv = unlearning_evaluator.functional_equivalence(original_model, unlearned_model, metric='JS-divergence', dataloader=dataloaders['val_loader'])

        forget_func_equiv_retrained = unlearning_evaluator.functional_equivalence(retrained_model, unlearned_model, metric='JS-divergence', dataloader=dataloaders['train_forget_loader'])
        
        results['functional_equivalence']['retain'].append(retain_func_equiv)
        results['functional_equivalence']['forget'].append(forget_func_equiv)
        results['functional_equivalence']['val'].append(val_func_equiv)
        results['functional_equivalence']['forget w retrained'].append(forget_func_equiv_retrained)

    
    df_results_u = pd.DataFrame.from_dict(results['unlearned_model'])
    df_results_r = pd.DataFrame.from_dict(results['retrained_model'])
    df_results_fe = pd.DataFrame.from_dict(results['functional_equivalence'])

    print(f'Unlearning results:\n{df_results_u}')
    print(f'mean retain_acc: {df_results_u['retain_acc'].mean():.3f}, mean forget_acc: {df_results_u['forget_acc'].mean():.3f} mean val_acc: {df_results_u['val_acc'].mean():.3f}')
    
    print(f'Retraining results:\n{df_results_r}')
    print(f'mean retain_acc: {df_results_r['retain_acc'].mean():.3f}, mean forget_acc: {df_results_r['forget_acc'].mean():.3f} mean val_acc: {df_results_r['val_acc'].mean():.3f}')
    
    print(f'Functional equivalence results:\n{df_results_fe}')
    pdb.set_trace()

