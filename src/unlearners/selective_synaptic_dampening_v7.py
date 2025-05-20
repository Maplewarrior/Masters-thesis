import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.unlearners.base_unlearner import BaseUnlearner
from src.datasets.synthetic_dataset import SyntheticDataset
from src.utils.misc import check_statedict_equivalent
from functools import partial
from math import floor
import copy
import pdb
import optuna


"""
This version of SSD automatically determines the optimal values of the hyperparameters lambda and alpha.
The method learns two values of alpha and two values of lambda.
(alpha1, lambda1) are applied to the first half of layers in the model and (alpha2, lambda2) are applied to the last half.

Additionally, in case that a class is not present in the forget set, we include it in the "generalization dataset". This might help mitigate catastrophic forgetting.
"""

"""
Bayesian Optimization, Hyperparameter Selection, Efficient, Streisand, Repair

Has to include Selective Synaptic Dampening (SSD)

Bayesian optimization for efficient hyperparameter Tuning of SSD: BET-SSD


Bayesian Repair Optimization: BRO-SSD


"""

from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening as SSD

class SelectiveSynapticDampening(SSD):
    def __init__(self, 
                 model,
                 P: int,
                 k: int,
                 n_trials: int = 100,
                 device: str = 'cpu') -> None:
        super().__init__(model, alpha=None, _lambda=None, device=device)
        self.P = P
        self.k = k
        self.n_trials = n_trials
        self.n_param_groups = len(list(self.model.parameters()))
        # assuming all layers have both weight & bias terms
        assert self.n_param_groups // 2 >= self.P, 'Cannot have more pairs of (alpha, lambda) values than the number of layers!'

    def layer_idx_to_parameter_idx(self, idx):
        return floor((idx * self.P) / (self.n_param_groups + 1))
    
    def construct_validation_set(self, forget_dataloader, val_dataloader):
        """
        A function for constructing a validation set that is "of the same distribution" as the forget dataset. This is the +R step in the method.
        "of the same distribution" is interpreted as being in terms of the label distribution.
        
        In the case that all classes are not represented in the forget set, then we add some samples from those.
        This, along with adding a retain set, should mitigate catastrophic forgetting.
        """
        
        forget_dataset = forget_dataloader.dataset
        val_dataset = val_dataloader.dataset
        # X_val = val_dataset.X.to(self.device)
        # y_val_ohe = val_dataset.y.to(self.device)
        # y_val = val_dataset.y.argmax(dim=1).to(self.device)
        X_val, y_val_ohe, entropies = self.entropy_sampling(val_dataloader)
        y_val = y_val_ohe.argmax(dim=-1)
        # draw random samples from the top k% highest entropy test samples
        k = int(len(X_val) * self.k)
        idx_shuffle = torch.randperm(k)
        X_val = X_val[:k][idx_shuffle]
        y_val = y_val[:k][idx_shuffle]
        y_val_ohe = y_val_ohe[:k][idx_shuffle]

        n_classes = int(y_val.max()+1) # assumes all classes are in the validation set...
        forget_labels, forget_counts = torch.unique(y_val, return_counts=True)
        forget_labels = forget_labels.to(self.device)
        
        sample_sizes = []
        X_values = []
        y_values = []
        # [5, 11, 8, 4, 8, 12, 11, 14, 15, 12]
        # max_x (15 * x) < forget_counts[argmax(15)]
        for i, label in enumerate(forget_labels):
            # find val set indexes that correspond to the label
            val_label_idx = torch.where(y_val == label)[0]
            # find sample size
            # sample_size = min(forget_counts[i].item(), len(val_label_idx))
            sample_size = max(forget_counts[i].item(), len(val_label_idx))
            sample_sizes.append(sample_size)
            # draw random samples
            idxs = list(range(sample_size)) #torch.randperm(len(val_label_idx))[:sample_size]
            # draw subset of data based on index
            X = X_val[val_label_idx[idxs]]
            y = y_val_ohe[val_label_idx[idxs]]
            X_values.append(X)
            y_values.append(y)

        # check if exact distribution could be constructed..
        # if not (torch.tensor(sample_sizes) == forget_counts).all():
        #     print("Exact distribution could not be constructed..")
        
        val_labels, val_counts = torch.unique(y_val, return_counts=True)
        # not all classes are represented in the forget set --> this may lead to catastrophic forgetting when choosing the best param update
        X_forget_add = []
        y_forget_add = []
        generator = torch.Generator()
        generator.manual_seed(self.model.seed)
        if not (val_labels == forget_labels).all():
            X_forget_add.append(forget_dataloader.dataset.X)
            y_forget_add.append(forget_dataloader.dataset.y)
            for i, label in enumerate(val_labels):
                if label in forget_labels: # continue if samples from class already exist
                    continue
                # add samples from remaining class(es) to "generalization" set
                sample_size = min(np.min(sample_sizes) // (len(val_labels) - len(forget_labels)), val_counts[i].item() // 2)
                val_label_idx = torch.where(y_val == label)[0]
                idxs = torch.randperm(len(val_label_idx))[:int(2 * sample_size)]

                X = X_val[val_label_idx[idxs[:len(idxs) // 2]]]
                y = y_val_ohe[val_label_idx[idxs[:len(idxs) // 2]]]
                X_values.append(X)
                y_values.append(y)

                # add to forget set for missing classes
                X = X_val[val_label_idx[idxs[len(idxs) // 2:]]]
                y = y_val_ohe[val_label_idx[idxs[len(idxs) // 2:]]]
                X_forget_add.append(X)
                y_forget_add.append(y)

            X_forget = torch.cat(X_forget_add)
            y_forget = torch.cat(y_forget_add)
            forget_dataset = SyntheticDataset(X_forget, y_forget, n_classes=n_classes)

            updated_forget_dataloader = DataLoader(forget_dataset, batch_size=4, generator=generator)

        X_values = torch.cat(X_values)
        y_values = torch.cat(y_values)
        
        dataset = SyntheticDataset(X_values.clone().detach(), y_values.clone().detach(), n_classes=n_classes)
        generalization_dataloader = DataLoader(dataset, batch_size=32, generator=generator)
        
        if len(X_forget_add): # if additions to forget dataset were made, return it
            return generalization_dataloader, updated_forget_dataloader
        
        return generalization_dataloader, None
    
    def calculate_loss(self, dataloader, n_classes = int):
        class_sums = torch.zeros(n_classes, device=self.device)
        class_counts = torch.zeros(n_classes, device=self.device)

        for batch in dataloader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)  # assumed one-hot encoded
            out = self.model.inference(x)

            # Compute loss and entropy
            loss = self.model.loss(out, y, reduction='none')  # shape: (batch_size,)
            entropy = -(out['probabilities'] * torch.log(out['probabilities'] + 1e-8)).sum(dim=-1)  # shape: (batch_size,)
            total_loss = loss + entropy  # shape: (batch_size,)

            # Get class indices from one-hot labels
            class_indices = y.argmax(dim=-1)  # shape: (batch_size,)

            # Accumulate total_loss per class
            class_sums += torch.bincount(class_indices, weights=total_loss, minlength=n_classes)
            class_counts += torch.bincount(class_indices, minlength=n_classes)

        # Avoid division by zero
        class_means = torch.where(class_counts > 0, class_sums / class_counts, torch.zeros_like(class_sums))
        return class_means
    
    def update_parameters(self, FIM_full, FIM_forget, **kwargs):
        # go through the parameters
        with torch.no_grad():
            for i, (name, param) in enumerate(self.model.named_parameters()):
                param_idx = self.layer_idx_to_parameter_idx(i)
                alpha = kwargs[f'alpha_{param_idx}']
                _lambda = kwargs[f'_lambda_{param_idx}']
                
                updated_parameter = param.data.clone()
                dampen_mask = FIM_forget[name] > alpha * FIM_full[name] # find which paramters to dampen
                # print(f'Original parameter: {param}')
                # print(f'Dampen mask: {dampen_mask}')
                # calculate dampening factors and apply them
                beta = torch.min((_lambda * FIM_full[name][dampen_mask] / FIM_forget[name][dampen_mask]), torch.tensor(1))
                updated_parameter[dampen_mask] = beta * updated_parameter[dampen_mask]
                # update parameter in the model
                param.copy_(updated_parameter)
                # print(f'Updated parameter: {param}')
    
    def entropy_sampling(self, test_dataloader):
        xs = []
        ys = []
        entropies = []
        for batch in test_dataloader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            probs = self.model.inference(x)['probabilities']
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            xs.append(x)
            ys.append(y)
            entropies.append(entropy)
        xs = torch.cat(xs)
        ys = torch.cat(ys)
        entropies = torch.cat(entropies)
        entropies, idxs = entropies.sort(descending=True)
        xs = xs[idxs]
        ys = ys[idxs]
        return xs, ys, entropies

    def objective_function(self,
                           FIM_full,
                           FIM_forget,
                           forget_loader: DataLoader,
                           gen_losses: dict,
                           sd_original,
                           n_classes: int,
                           **kwargs):
        # load original weights
        self.model.load_state_dict(sd_original)
        # make sure parameters were reset
        assert check_statedict_equivalent(self.model.state_dict(), sd_original), 'Model parameters were not reset to original values!'
        # update parameters
        self.update_parameters(FIM_full, FIM_forget, **kwargs)
        # calculate loss on forget set
        forget_losses = self.calculate_loss(forget_loader, n_classes)
        # pdb.set_trace()
        # calculate squared error between forget an generalization loss
        diff = diff = ((forget_losses - gen_losses)**2).mean()
        return -diff

    def search_hyperparameters_TPE(self,
                                  FIM_full,
                                  FIM_forget,
                                  forget_loader: DataLoader,
                                  gen_losses: np.array, 
                                  sd_original,
                                  n_classes: int,
                                  ):
        # remove logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        objective_func = partial(self.objective_function,
                                FIM_full=FIM_full,
                                FIM_forget=FIM_forget,
                                forget_loader=forget_loader,
                                gen_losses=gen_losses,
                                sd_original=sd_original,
                                n_classes=n_classes)
        
        def objective(trial):
            params = {f'alpha_{i}': trial.suggest_float(f"alpha_{i}", 0.01, 100.0, log=False) for i in range(self.P)}
            params.update({f'_lambda_{i}': trial.suggest_float(f"_lambda_{i}", 0.01, 5.0, log=False) for i in range(self.P)})
            
            # Run your dampening + evaluation logic on the GPU
            loss = objective_func(**params)  # Must return a scalar loss (torch or float)
            return float(loss)  # Optuna needs a float, not a tensor

        study = optuna.create_study(direction="maximize", 
                                    sampler=optuna.samplers.TPESampler(seed=self.model.seed),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
        
        study.optimize(objective, n_trials=self.n_trials)
        
        ### unpack result
        values = [trial.value for trial in study.trials]
        params = [trial.params for trial in study.trials]
        result = [{'target': values[i], 'params': params[i]} for i in range(self.n_trials)]
        opt_trial = {'target': study.best_trial.value,
                     'params': study.best_trial.params}        
        
        return {'result': result,
                'max': opt_trial}
        # return {'result': bayesian_optimizer.res, 'max': bayesian_optimizer.max}
    
    def select_optimal_parameters(self, bo_result: dict):
        """
        If there are multiple optimal parameters: Choose the one which changes the original model the least.
        """
        scores = np.array([e['target'] for e in bo_result['result']])
        best_scores_idxs = np.where(scores == max(scores))[0]
        best_params = [bo_result['result'][i]['params'] for i in best_scores_idxs]
        param_sums = [sum(e.values()) for e in best_params]
        # return best_params[np.argmax(param_sums)]
        return best_params[np.argmin(param_sums)]

    def __call__(self, 
                 full_dataloader,
                 forget_dataloader,
                 validation_dataloader,
                 FIM_full: dict = None, 
                 FIM_forget: dict = None
                 ):
                         
        # calculate FIM matrices if necessary
        if FIM_forget is None:
            FIM_forget = self.calculate_FIM(forget_dataloader)
        
        if FIM_full is None:
            FIM_full = self.calculate_FIM(full_dataloader)
        
        sd_original = copy.deepcopy(self.model.state_dict())
        n_classes = torch.unique(validation_dataloader.dataset.y, dim=0).size(0)

        generalization_dataloader, updated_forget_loader = self.construct_validation_set(forget_dataloader, validation_dataloader)
        generalization_losses = self.calculate_loss(generalization_dataloader, n_classes)
        
        n_forget_classes = torch.unique(forget_dataloader.dataset.y, dim=0).size(0)
        if n_classes != n_forget_classes:
            # forget_dataloader_old = copy.deepcopy(forget_dataloader)
            forget_dataloader = updated_forget_loader
        
        # find optimal alpha and lambda values via bayesian optimization
        opt_result = self.search_hyperparameters_TPE(FIM_full, FIM_forget, forget_dataloader, generalization_losses, sd_original, n_classes)
        best_params = opt_result['max']['params']
        # best_params = self.select_optimal_parameters(opt_result)
        
        # go through the parameters
        self.model.load_state_dict(sd_original)
        # print(f'SD identical? {self.check_statedict_equivalent(sd_original, self.model.state_dict())}')
        self.update_parameters(FIM_full, FIM_forget, **best_params)
        # print(f'SD identical? {self.check_statedict_equivalent(sd_original, self.model.state_dict())}')
        return best_params