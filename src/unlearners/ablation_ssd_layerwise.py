import torch
import torch.optim as optim
from src.unlearners.base_unlearner import BaseUnlearner
import pdb
import copy
from functools import partial
import numpy as np
from bayes_opt import BayesianOptimization
from torch.utils.data import DataLoader
from src.datasets.synthetic_dataset import SyntheticDataset
from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening as SSD

"""
This version of SSD only applies dampening in the final layer of the model
"""


class SelectiveSynapticDampening_layerwise(SSD):
    def __init__(self, 
                 model,
                 alpha: float,
                 _lambda: float,
                 layer_target: list[int] = None,
                 device: str = 'cpu') -> None:
        super().__init__(model, alpha, _lambda, device)
        self.layer_target = layer_target

    def __call__(self, 
                 full_dataloader,
                 forget_dataloader,
                 FIM_full: dict = None, 
                 FIM_forget: dict = None,
                 return_dampening: bool = False
                 ):
        all_dampenings = {}
        # calculate FIM matrices if necessary
        if FIM_forget is None:
            FIM_forget = self.calculate_FIM(forget_dataloader)
        
        if FIM_full is None:
            FIM_full = self.calculate_FIM(full_dataloader)
        
        # go through the parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # skip the first layers :D
                if int(name.split('.')[1]) not in self.layer_target:
                    print(f'Layer {int(name.split('.')[1])} is not in the target layer')
                    all_dampenings[name] = torch.ones_like(param)
                    continue
                updated_parameter = param.data.clone()
                dampen_mask = FIM_forget[name] > self.alpha * FIM_full[name] # find which paramters to dampen
                # print(f'Original parameter: {param}')
                # print(f'Dampen mask: {dampen_mask}')
                # calculate dampening factors and apply them
                beta = torch.min((self._lambda * FIM_full[name][dampen_mask] / FIM_forget[name][dampen_mask]), torch.tensor(1))
                updated_parameter[dampen_mask] = beta * updated_parameter[dampen_mask]
               
                # update parameter in the model
                param.copy_(updated_parameter)

                if return_dampening:
                    dampen_values = torch.ones_like(dampen_mask, dtype = beta.dtype)
                    dampen_values[dampen_mask] = beta
                    all_dampenings[name] = dampen_values
        
        if return_dampening:
            return all_dampenings


class SelectiveSynapticDampening_bo(SSD):
    def __init__(self, 
                 model,
                 layer_target: list[int] = None,
                 device: str = 'cpu') -> None:
        super().__init__(model, alpha=None, _lambda=None, device=device)
        self.layer_target = layer_target

    def construct_validation_set(self, forget_dataloader, val_dataloader):
        """
        A function for constructing a validation set that is "of the same distribution" as the forget dataset. This is the +R step in the method.
        "of the same distribution" is interpreted as being in terms of the label distribution.
        """
        forget_dataset = forget_dataloader.dataset
        val_dataset = val_dataloader.dataset
        X_val = val_dataset.X
        y_val_ohe = val_dataset.y
        y_val = val_dataset.y.argmax(dim=1)
        n_classes = int(y_val.max()+1) # assumes all classes are in the validation set...
        labels, counts = torch.unique(torch.argmax(forget_dataset.y, dim=1), return_counts=True)

        sample_sizes = []

        X_values = []
        y_values = []
        for i, label in enumerate(labels):
            # find val set indexes that correspond to the label
            val_label_idx = torch.where(y_val == label)[0]
            # find sample size
            sample_size = min(counts[i].item(), len(val_label_idx))
            sample_sizes.append(sample_size)
            # draw random samples
            idxs = torch.randperm(len(val_label_idx))[:sample_size]
            # draw subset of data based on index
            X = X_val[val_label_idx[idxs]]
            y = y_val_ohe[val_label_idx[idxs]]
            X_values.append(X)
            y_values.append(y)

        # check if exact distribution could be constructed..
        if not (torch.tensor(sample_sizes) == counts).all():
            print("Exact distribution could not be constructed..")

        X_values = torch.cat(X_values)
        y_values = torch.cat(y_values)

        dataset = SyntheticDataset(X_values, y_values, n_classes=n_classes)
        dataloader = DataLoader(dataset, batch_size=val_dataloader.batch_size)
        return dataloader
    
    def update_parameters(self, FIM_full, FIM_forget, alpha, _lambda, return_dampening: bool = False):
        all_dampenings = {}
        # go through the parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if int(name.split('.')[1]) not in self.layer_target:
                    print(f'Layer {int(name.split('.')[1])} is not in the target layer')
                    all_dampenings[name] = torch.ones_like(param)
                    continue
                updated_parameter = param.data.clone()
                dampen_mask = FIM_forget[name] > alpha * FIM_full[name] # find which paramters to dampen
                # calculate dampening factors and apply them
                beta = torch.min((_lambda * FIM_full[name][dampen_mask] / FIM_forget[name][dampen_mask]), torch.tensor(1))
                updated_parameter[dampen_mask] = beta * updated_parameter[dampen_mask]
                # update parameter in the model
                param.copy_(updated_parameter)
                if return_dampening:
                    dampen_values = torch.ones_like(dampen_mask, dtype = beta.dtype)
                    dampen_values[dampen_mask] = beta
                    all_dampenings[name] = dampen_values
                
        if return_dampening:
            return all_dampenings
        
    def search_hyperparams_exhaustive(self, 
                           alphas: list[float], 
                           lambdas: list[float], 
                           FIM_full, FIM_forget, 
                           forget_loader,
                           gen_losses: np.array, 
                           sd_original,
                           n_classes):
        
        best_diff = np.inf
        best_params = {'alpha': None,
                       'lambda': None}
        for alpha in alphas:
            for _lambda in lambdas:
                self.model.load_state_dict(sd_original)
                self.update_parameters(FIM_full, FIM_forget, alpha, _lambda)
                forget_losses = np.array(list(self.calculate_loss(forget_loader, n_classes).values()))
                diff = np.mean((forget_losses - gen_losses)**2)
                if diff < best_diff:
                    best_diff = diff
                    best_params.update({'alpha': alpha, 'lambda': _lambda})
        
        return best_params, best_diff
    
    def calculate_loss(self, dataloader, n_classes = int):
        losses = {i: [] for i in range(n_classes)}

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0]
                y = batch[1]
                out = self.model(x)
                loss = self.model.loss(out, y)
                for i in range(y.size(0)):
                    losses[y[i].argmax(dim=-1).item()].append(loss.item())
        # losses = {k: np.mean(v) if len(v) else 0 for k, v in losses.items()}
        
        losses = np.array([np.mean(e) if len(e) else 0 for e in losses.values()])
        return losses
    
    def bo_objective_function(self, 
                              alpha, 
                              _lambda,
                              FIM_full,
                              FIM_forget,
                              forget_loader: DataLoader,
                              gen_losses: dict,
                              sd_original,
                              n_classes: int,):
        # load original weights
        self.model.load_state_dict(sd_original)
        # make sure parameters were reset
        assert not self.check_statedict_diff(self.model.state_dict(), sd_original), 'Model parameters were not reset to original values!'
        # update parameters
        self.update_parameters(FIM_full, FIM_forget, alpha, _lambda)
        # calculate loss on forget set
        forget_losses = self.calculate_loss(forget_loader, n_classes)
        # calculate squared error between forget an generalization loss
        diff = np.mean((forget_losses - gen_losses)**2)
        return -diff

    def search_hyperparameters_bo(self,
                                  FIM_full,
                                  FIM_forget,
                                  forget_loader: DataLoader,
                                  gen_losses: np.array, 
                                  sd_original,
                                  n_classes: int,
                                  ):
        
        pbounds = {'alpha': (0.01, 10), '_lambda': (0.1, 50)}
        objective_func = partial(self.bo_objective_function,
                            FIM_full=FIM_full,
                            FIM_forget=FIM_forget,
                            forget_loader=forget_loader,
                            gen_losses=gen_losses,
                            sd_original=sd_original,
                            n_classes=n_classes)
        bayesian_optimizer = BayesianOptimization(
            f=objective_func,
            pbounds=pbounds,
            random_state=1,
            verbose=0
        )
        bayesian_optimizer.maximize(n_iter=100)
        
        return {'result': bayesian_optimizer.res, 'max': bayesian_optimizer.max}

    def check_statedict_diff(self, sd1, sd2):
        """
        returns true if state dicts are different and false otherwise
        """
        for key in sd1.keys():
            if not torch.isclose(sd1[key], sd2[key]).all():
                return True
        return False
    
    def select_optimal_parameters(self, bo_result: dict):
        """
        If there are multiple optimal parameters: Choose the one which changes the original model the least.
        """
        scores = [e['target'] for e in bo_result['result']]
        best_scores_idxs = np.where(scores == max(scores))[0]
        # pdb.set_trace()
        best_params = [bo_result['result'][i]['params'] for i in best_scores_idxs]
        param_sums = [sum((e['_lambda'], e['alpha'])) for e in best_params]
        # pdb.set_trace()
        return best_params[np.argmax(param_sums)]

    def __call__(self, 
                 full_dataloader,
                 forget_dataloader,
                 validation_dataloader,
                 FIM_full: dict = None, 
                 FIM_forget: dict = None,
                 return_dampening: bool = False
                 ):
        
        # calculate FIM matrices if necessary
        if FIM_forget is None:
            FIM_forget = self.calculate_FIM(forget_dataloader)
        
        if FIM_full is None:
            FIM_full = self.calculate_FIM(full_dataloader)
        
        sd_original = copy.deepcopy(self.model.state_dict())
        n_classes = validation_dataloader.dataset.y.argmax(dim=-1).max() + 1
        generalization_dataloader = self.construct_validation_set(forget_dataloader, validation_dataloader)
        generalization_losses = self.calculate_loss(generalization_dataloader, n_classes)
        
        # forget_losses = self.calculate_loss(forget_dataloader, n_classes)
        # alphas = np.arange(start=0.01, stop=1.0, step=0.01)
        # lambdas = np.arange(start=10, stop=50., step=0.1)
        # best_params, best_diff = self.search_hyperparams_exhaustive(alphas, lambdas, FIM_full, FIM_forget, forget_dataloader, generalization_losses, sd_original, n_classes)
        
        # find optimal alpha and lambda values via bayesian optimization
        bo_result = self.search_hyperparameters_bo(FIM_full, FIM_forget, forget_dataloader, generalization_losses, sd_original, n_classes)
        best_params = self.select_optimal_parameters(bo_result)

        alpha_opt = best_params['alpha'] # bo_result['max']['params']['alpha']
        lambda_opt = best_params['_lambda'] # bo_result['max']['params']['_lambda']
        
        # reset state dict
        self.model.load_state_dict(sd_original)
        # update parameters
        all_dampenings = self.update_parameters(FIM_full, FIM_forget, alpha_opt, lambda_opt, return_dampening)
        
        return best_params, all_dampenings
    
        # alpha_opt = bo_result['max']['params']['alpha']
        # lambda_opt = bo_result['max']['params']['_lambda']
        # return bo_result['max']['params']
        