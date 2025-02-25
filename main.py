import argparse
import pdb
import pandas as pd
import torch.nn as nn
import copy
import json
import os

from src.modelling.trainer import Trainer
from src.modelling.neural_network import NeuralNet
from src.data_utils.synthetic_data import DataGenerator, create_dataloaders
from src.modelling.selective_synaptic_dampening import SelectiveSynapticDampening
from src.modelling.scrub import ScrubR
from src.modelling.sisa import SISA
from src.modelling.sae_unlearner import SAEUnlearner
from src.modelling.SAE import SAE
from src.evaluation.unlearning_evaluator import UnlearningEvaluator
from src.evaluation.results_table import create_latex_table
from src.modelling.amnesiac import AmnesiacTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='experiment', choices=['experiment', 'visualize', 'get_latex_results'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--unlearn-type', type=str, default='SSD', choices=['SSD', 'Scrub+R', 'SISA', 'amnesiac','SAE'], help='What type of unlearning algorithm to apply.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')    
    
    args = parser.parse_args()
    if args.mode == 'experiment':
        ### set constants
        ## for the experiment:
        n_repeats = 2
        n_epochs = 20
        ## for the dataset:
        n_features = 25
        n_classes = 4
        ## SSD hyperparameters:
        # alpha = 2.5
        # _lambda = 0.1 #0.1
        ## set evaluation measures
        efficacy_metrics = ['accuracy', 'Hamming PD', 'min max normalized HPD']
        func_equivalence_metrics = ['avg norm prediction difference', 'JS divergence']
        evaluation_metrics = efficacy_metrics + func_equivalence_metrics

        ### Create synthetic dataset
        data_generator = DataGenerator(random_state=42)
        data_generator.generate_data(n_samples=1000, n_features=n_features,
                                    n_classes=n_classes, n_informative=2, 
                                    n_redundant=0, n_outliers=50, outlier_scale=4.0, 
                                    outlier_variance=0.4, outlier_class=1)
        data_generator.split_data(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        # data_generator.draw_forget_set(n_points=20, class_idx=1, ood_ratio=0.5)
        data_generator.draw_forget_set(n_points=50, class_idx=None, ood_ratio=0.5)

        if args.unlearn_type == 'amnesiac':
            dataloaders = create_dataloaders(data_generator, batch_size=1, use_indices=True, device=args.device)
            x, y, indices = next(iter(dataloaders['train_full_loader']))
        else:
            dataloaders = create_dataloaders(data_generator, batch_size=32, onehot_labels=True, device=args.device)
            x, y = next(iter(dataloaders['train_full_loader']))

        results = {'unlearned vs. original': {'retain': [], 'forget': [], 'validation': []},
                'unlearned vs. retrained': {'retain': [], 'forget': [], 'validation': []},
                }

        for i in range(n_repeats):
            # TODO Shouldn't we draw a new dataset or at least a new forget set for each repeat??
            if args.unlearn_type in ['Scrub+R', 'SSD']:
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
            
            elif args.unlearn_type == 'SISA':
                unlearned_model = SISA(dataloader=dataloaders['train_full_loader'],
                                       n_shards=10, n_slices=10, n_features=n_features,
                                       n_classes=n_classes, n_epochs=n_epochs,
                                       save_dir='./experiments/checkpoints/SISA/original_model')
                unlearned_shards_dict = unlearned_model.process_data()
                original_shards_dict = unlearned_shards_dict
                unlearned_model.train_all_models()

                retrained_model = SISA(dataloader=dataloaders['train_retain_loader'], 
                                       n_shards=10, n_slices=10, n_features=n_features,
                                       n_classes=n_classes, n_epochs=n_epochs,
                                       save_dir='./experiments/checkpoints/SISA/retrained_model')
                retrained_model.process_data()
                retrained_model.train_all_models()
            
            elif args.unlearn_type == 'amnesiac':

                unlearned_model = NeuralNet(n_features, n_classes)
                trainer = AmnesiacTrainer(unlearned_model, lr=0.1, device=args.device, cache_gradients=True)

                train_full_loader = dataloaders['train_full_loader']
                val_loader = dataloaders['val_loader']
                test_loader = dataloaders['test_loader']
                retain_loader = dataloaders['train_retain_loader']
                forget_loader = dataloaders['train_forget_loader']
                indices_to_forget = dataloaders['forget_idx_in_train']

                # training on full dataset while storing gradients for sensitive batches/points to forget later
                trainer.train(train_full_loader, 
                              epochs=n_epochs, 
                              indices_to_forget=indices_to_forget, 
                              save_accuracy_to_file=False)
                
                # define and train retrained model on the retain dataset
                retrained_model = NeuralNet(n_features, n_classes)
                retrained_trainer = AmnesiacTrainer(retrained_model, lr=0.1, device=args.device, cache_gradients=True)
                retrained_trainer.train(retain_loader, repair=True)            
            
            original_model = copy.deepcopy(unlearned_model)
            original_model.model_type = 'pre_forget'
            unlearned_model.model_type = 'post_forget'
            
            if args.unlearn_type == 'Scrub+R':
                alpha = 1.
                gamma = 1.
                scrub = ScrubR(unlearned_model, original_model, alpha=alpha, gamma=gamma)
                scrub(dataloaders['train_retain_loader'], dataloaders['train_forget_loader'], dataloaders['val_loader'], n_rounds=6)
            
            elif args.unlearn_type == 'SSD':
                alpha=7.5
                _lambda = 0.5
                criterion = nn.CrossEntropyLoss()
                SSD = SelectiveSynapticDampening(unlearned_model, criterion, alpha=alpha, _lambda=_lambda)
                SSD(full_dataloader=dataloaders['train_full_loader'], forget_dataloader=dataloaders['train_forget_loader'])
            
            elif args.unlearn_type == 'SISA':
                unlearned_model.forget_datapoints(datapoint_idxs=dataloaders['forget_idx_in_train'])#.forget_datapoint(datapoint_idx=0)

            elif args.unlearn_type == 'SAE':
                _lambda = 1.0 # regularization strength
                layer_num = 4 # which layer to apply the SAE
                m_multiple = 4 # how many times larger m should be than d
                d = unlearned_model.net[layer_num-2].out_features # dimensionality of the SAE input activation (assuming neural network has a bias term)
                sae = SAE(d=d, 
                        m=d * m_multiple,
                        _lambda=_lambda)
                
                sae_unlearner = SAEUnlearner(unlearned_model, sae, layer_num)
                sd_before = sae_unlearner.model.state_dict()
                sae_trainer = Trainer(sae_unlearner, train_dataloader=dataloaders['train_full_loader'], val_dataloader=dataloaders['val_loader'], n_epochs=n_epochs)
                sae_trainer.train_sae()
                sd_after = sae_unlearner.model.state_dict()

            elif args.unlearn_type == 'amnesiac':
                trainer.forget(indices_to_forget=None) # Set to None to forget all sensitive batches, used when storing only gradients for sensitive batches

                # repair phase, hardcoded to 10 epochs
                trainer.train(train_full_loader, epochs=10, 
                            save_accuracy_to_file=False, 
                            repair=True)
                
            # Model evaluation
            unlearning_evaluator = UnlearningEvaluator()

            ### evaluate unlearning efficacy and functional equivalence
            unlearned_original_retain = unlearning_evaluator.evaluate(unlearned_model, original_model, evaluation_metrics, dataloaders['train_retain_loader'])
            unlearned_original_forget = unlearning_evaluator.evaluate(unlearned_model, original_model, evaluation_metrics, dataloaders['train_forget_loader'])
            unlearned_original_val = unlearning_evaluator.evaluate(unlearned_model, original_model, evaluation_metrics, dataloaders['val_loader'])
            
            unlearned_retrained_retain = unlearning_evaluator.evaluate(unlearned_model, retrained_model, evaluation_metrics, dataloaders['train_retain_loader'])
            unlearned_retrained_forget = unlearning_evaluator.evaluate(unlearned_model, retrained_model, evaluation_metrics, dataloaders['train_forget_loader'])
            unlearned_retrained_val = unlearning_evaluator.evaluate(unlearned_model, retrained_model, evaluation_metrics, dataloaders['val_loader'])
            
            # ### evaluate functional equivalence
            # func_equiv_unlearned_original_retain = unlearning_evaluator.evaluate(unlearned_model, original_model, func_equivalence_metrics, dataloaders['train_retain_loader'])
            # func_equiv_unlearned_original_forget = unlearning_evaluator.evaluate(unlearned_model, original_model, func_equivalence_metrics, dataloaders['train_forget_loader'])
            # func_equiv_unlearned_original_val = unlearning_evaluator.evaluate(unlearned_model, original_model, func_equivalence_metrics, dataloaders['val_loader'])
            
            # func_equiv_unlearned_retrained_retain = unlearning_evaluator.evaluate(unlearned_model, retrained_model, func_equivalence_metrics, dataloaders['train_retain_loader'])
            # func_equiv_unlearned_retrained_forget = unlearning_evaluator.evaluate(unlearned_model, retrained_model, func_equivalence_metrics, dataloaders['train_forget_loader'])
            # func_equiv_unlearned_retrained_val = unlearning_evaluator.evaluate(unlearned_model, retrained_model, func_equivalence_metrics, dataloaders['val_loader'])
            
            ### store results
            results['unlearned vs. original']['retain'].append(unlearned_original_retain)
            results['unlearned vs. original']['forget'].append(unlearned_original_forget)
            results['unlearned vs. original']['validation'].append(unlearned_original_val)

            results['unlearned vs. retrained']['retain'].append(unlearned_retrained_retain)
            results['unlearned vs. retrained']['forget'].append(unlearned_retrained_forget)
            results['unlearned vs. retrained']['validation'].append(unlearned_retrained_val)

        results = {args.unlearn_type: results}

        os.makedirs('experiments/results', exist_ok=True)
        
        with open(f'experiments/results/{args.unlearn_type}_results.json', 'w') as f:
            json.dump(results, f)

        results_table = create_latex_table(results)
        pdb.set_trace()


    elif args.mode == 'get_latex_results':
        result_files = os.listdir('experiments/results/')
        all_results = {}
        for file in result_files:
            with open(f'experiments/results/{file}', 'r') as f:
                results = json.load(f)
            all_results.update(results)
        
        print(f"\n\n\n{create_latex_table(all_results)}")


