import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.unlearners.base_unlearner import BaseUnlearner
from src.datasets.synthetic_dataset import SyntheticDataset
from bayes_opt import BayesianOptimization
from functools import partial
import copy
import pdb

"""
This version of SSD automatically determiens the optimal values of the hyperparameters lambda and alpha.
The method learns two values of alpha and two values of lambda.
(alpha1, lambda1) are applied to the first half of layers in the model and (alpha2, lambda2) are applied to the last half.
"""

class SelectiveSynapticDampening(BaseUnlearner):
    def __init__(self, 
                 model, 
                 criterion,
                 alpha: float,
                 _lambda: float) -> None:
        super().__init__(model, {'alpha': alpha, '_lambda': _lambda})
        self.criterion = criterion
        self.alpha = alpha
        self._lambda = _lambda
        
    def calculate_FIM(self, dataloader) -> dict:
        """
        A function that calculates the diagonal of the Fisher Information of a model for a specific dataset.

        @param dataloader: An iterable dataloader for which the FIM diagonal should be calculated.
        returns: A dictionary where keys are the names of parameters and values are the FIM of that parameter
        """
        FIM = {k: 0 for k in self.model.state_dict().keys()}
        # define optimizer to allow gradient computation
        optimizer = optim.SGD(self.model.parameters())
        # turn of dropout if applicable
        self.model.eval()
        
        for i, batch in enumerate(dataloader):
            x, y = batch[0], batch[1]
            optimizer.zero_grad()
            # forward pass
            logits = self.model(x)['logits']
            # calculate loss
            loss = self.criterion(logits, y)
            # calculate gradients
            loss.backward()
            for i, (name, param) in enumerate(self.model.named_parameters()):
                FIM[name] += param.grad.data.clone().pow(2) #optimizer.param_groups[0]['params'][i].grad.pow(2)  
        # account for batched inference
        for k in FIM.keys():
            FIM[k] = FIM[k] / len(dataloader)
        
        return FIM
    
    def construct_validation_set(self, forget_dataloader, val_dataloader):
        """
        A function for constructing a validation set that is "of the same distribution" as the forget dataset. This is the +R step in the method.
        "of the same distribution" is interpreted as being in terms of the label distribution.
        """
        
        forget_dataset = forget_dataloader.dataset
        val_dataset = val_dataloader.dataset
        X_val = val_dataset.X
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
            sample_size = min(counts[i].item(), len(val_label_idx), 5)
            sample_sizes.append(sample_size)
            # draw random samples
            idxs = torch.randperm(sample_size)
            # draw subset of data based on index
            X = X_val[val_label_idx[idxs]]
            y = y_val[val_label_idx[idxs]]
            X_values.append(X)
            y_values.append(y)

        # check if exact distribution could be constructed..
        if not (torch.tensor(sample_sizes) == counts).all():
            print("Exact distribution could not be constructed..")

        X_values = torch.cat(X_values)
        y_values = torch.cat(y_values)
        
        dataset = SyntheticDataset(X_values, y_values, n_classes=n_classes)
        dataloader = DataLoader(dataset, batch_size=1)
        return dataloader

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
                    losses[y[i].argmax(dim=-1).item()].append(loss)
        # losses = {k: np.mean(v) if len(v) else 0 for k, v in losses.items()}
        losses = np.array([np.mean(e) if len(e) else 0 for e in losses.values()])
        return losses
    
    def update_parameters(self, FIM_full, FIM_forget, alpha1, lambda1, alpha2, lambda2):
        n_param_types = len(FIM_full.keys())
        # go through the parameters
        with torch.no_grad():
            for i, (name, param) in enumerate(self.model.named_parameters()):
                if i <= n_param_types / 2: # apply alpha1 and lambda1 to first half
                    alpha=alpha1
                    _lambda = lambda1
                else: # apply alpha2, lambda2
                    alpha=alpha2
                    _lambda = lambda2
                
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
    
    def bo_objective_function(self, 
                              alpha1, 
                              lambda1,
                              alpha2,
                              lambda2,
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
        self.update_parameters(FIM_full, FIM_forget, alpha1, lambda1, alpha2, lambda2)
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
        
        pbounds = {'alpha1': (0.01, 10), 'lambda1': (0.1, 50),
                   'alpha2': (0.01, 10), 'lambda2': (0.1, 50)}
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
                 FIM_forget: dict = None
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
        
        # best_params = self.select_optimal_parameters(bo_result)
        # alpha_opt = best_params['alpha']#bo_result['max']['params']['alpha']
        # lambda_opt = best_params['_lambda'] # bo_result['max']['params']['_lambda']

        alpha1_opt = bo_result['max']['params']['alpha1']
        lambda1_opt = bo_result['max']['params']['lambda1']

        alpha2_opt = bo_result['max']['params']['alpha2']
        lambda2_opt = bo_result['max']['params']['lambda2']


        # go through the parameters
        self.model.load_state_dict(sd_original)
        self.update_parameters(FIM_full, FIM_forget, alpha1_opt, lambda1_opt, alpha2_opt, lambda2_opt)
                
        return bo_result['max']['params']