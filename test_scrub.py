import pdb
import pandas as pd
import torch.nn as nn

from src.modelling.trainer import Trainer
from src.modelling.neural_network import NeuralNet
from src.data_utils.synthetic_data import DataGenerator, create_dataloaders
from src.modelling.scrub import ScrubR
from src.modelling.unlearning_evaluator import UnlearningEvaluator

if __name__ == '__main__':
    ### set constants
    ## for the experiment:
    n_repeats = 5
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
    data_generator.draw_forget_set(n_points=20, class_idx=1, ood_ratio=0.5)
    dataloaders = create_dataloaders(data_generator, batch_size=32, onehot_labels=True)
    x, y = next(iter(dataloaders['train_full_loader']))

    results = {'unlearned_model': {'retain_acc': [], 'forget_acc': [], 'val_acc': []},
               'retrained_model': {'retain_acc': [], 'forget_acc': [], 'val_acc': []}}
    
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

        # apply SSD on unlearned model
        criterion = nn.CrossEntropyLoss()
        mu = 0
        for param in unlearned_model.parameters():
            mu += param.data.mean()
        print(f'Mu before: {mu}')
        scrub = ScrubR(unlearned_model, alpha=1, gamma=1.)
        scrub(dataloaders['train_retain_loader'], dataloaders['train_forget_loader'], n_rounds=1)
        mu = 0
        for param in unlearned_model.parameters():
            mu += param.data.mean()
        print(f'Mu after: {mu}')
        pdb.set_trace()
        for name, param in unlearned_model.named_parameters():
            print(f'name: {name}')
            print(f'Param: \n', param)
        
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
    
    df_results_u = pd.DataFrame.from_dict(results['unlearned_model'])
    df_results_r = pd.DataFrame.from_dict(results['retrained_model'])
    print(f'Unlearning results:\n{df_results_u}')
    print(f'Retraining results:\n{df_results_r}')
    pdb.set_trace()

