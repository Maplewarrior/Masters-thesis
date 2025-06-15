
"""
This version of SSD automatically determiens the optimal values of the hyperparameters lambda and alpha.
Dampening is applied to all model layers.
"""


import numpy as np
from math import floor
import warnings
import copy
import time
import pdb
import torch
from torch.utils.data import DataLoader

import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize

from src.datasets.synthetic_dataset import SyntheticDataset
from src.utils.misc import check_statedict_equivalent

from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening as SSD

class SelectiveSynapticDampening(SSD):
    def __init__(self,
                 model,
                 k: int = 1.0, # top-k % entropy test data to sample from when constructing repair set
                 n_bo_iter: int = 100, # how many iterations to run bayesian optimization
                 device: str = 'cpu') -> None:
        super().__init__(model, alpha=None, _lambda=None, device=device)
        assert 0 < k <= 1, 'k must range between [0, 1]'
        self.k = k
        self.n_bo_iter = n_bo_iter
        self.n_param_groups = len(list(self.model.parameters()))
        # assuming all layers have both weight & bias terms

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

    def construct_validation_set(self, forget_dataloader, val_dataloader):
        """
        A function for constructing a validation set that is "of the same distribution" as the forget dataset. This is the +R step in the method.
        "of the same distribution" is interpreted as being in terms of the label distribution.
        
        In the case that all classes are not represented in the forget set, then we add some samples from those.
        This, along with adding a retain set, should mitigate catastrophic forgetting.
        """
        generator = torch.Generator()
        generator.manual_seed(self.model.seed)
        
        forget_dataset = forget_dataloader.dataset
        y_forget_ohe = forget_dataset.y
        y_forget = y_forget_ohe.argmax(dim=-1)
        
        val_dataset = val_dataloader.dataset
        X_val = val_dataset.X.to(self.device)
        y_val_ohe = val_dataset.y.to(self.device)
        y_val = val_dataset.y.argmax(dim=1).to(self.device)

        idx_shuffle = torch.randperm(len(X_val))
        X_val = X_val[idx_shuffle]
        y_val = y_val[idx_shuffle]
        y_val_ohe = y_val_ohe[idx_shuffle]
        

        # # draw random samples from the top k% highest entropy test samples
        # X_val, y_val_ohe, entropies = self.entropy_sampling(val_dataloader)
        # y_val = y_val_ohe.argmax(dim=-1)

        # k = int(len(X_val) * self.k)
        # idx_shuffle = torch.randperm(len(X_val))
        # X_val = X_val[idx_shuffle]
        # y_val = y_val[idx_shuffle]
        # y_val_ohe = y_val_ohe[idx_shuffle]

        n_classes = int(y_val.max()+1) # assumes all classes are in the validation set...

        ### Sample according to predicted labels (works better in adversarial settings)
        # forget_preds = self.model(forget_dataset.X)['predictions']
        # forget_labels, forget_counts = torch.unique(forget_preds, return_counts=True)
        # forget_preds_ohe = torch.zeros(len(forget_preds), n_classes).scatter_(1, forget_preds.unsqueeze(1), 1)

        ### Sample according to true forget labels (breaks in adversarial settings)
        forget_labels, forget_counts = torch.unique(y_forget, return_counts=True)
        forget_preds_ohe = None

        if forget_labels != y_forget.unique():
            updated_forget_dataloader = copy.deepcopy(forget_dataloader)
            updated_forget_dataloader.dataset.y = forget_preds_ohe

        else:
            updated_forget_dataloader = forget_dataloader
            
        forget_labels = forget_labels.to(self.device)
        
        sample_sizes = []
        X_values = []
        y_values = []
        
        for i, label in enumerate(forget_labels):
            # find val set indexes that correspond to the label
            val_label_idx = torch.where(y_val == label)[0]
            # find sample size
            sample_size = min(forget_counts[i].item(), len(val_label_idx))
            # sample_size = max(forget_counts[i].item(), len(val_label_idx))
            sample_sizes.append(sample_size)
            # draw random samples
            idxs = list(range(sample_size)) #torch.randperm(len(val_label_idx))[:sample_size]
            # draw subset of data based on index
            X = X_val[val_label_idx[idxs]]
            y = y_val_ohe[val_label_idx[idxs]]
            X_values.append(X)
            y_values.append(y)

        # check if exact distribution could be constructed..
        if not (torch.tensor(sample_sizes, device=self.device) == forget_counts).all():
            print("Exact distribution could not be constructed..")
        
        X_values = torch.cat(X_values)
        y_values = torch.cat(y_values)
        
        dataset = SyntheticDataset(X_values.clone().detach(), y_values.clone().detach(), n_classes=n_classes)
        generalization_dataloader = DataLoader(dataset, batch_size=32, generator=generator)
        
        # if len(X_forget_add): # if additions to forget dataset were made, return it
        #     return generalization_dataloader, updated_forget_dataloader
        
        return generalization_dataloader, updated_forget_dataloader

    def calculate_loss(self, dataloader, n_classes = int):

        class_sums = torch.zeros(n_classes, device=self.device)
        class_counts = torch.zeros(n_classes, device=self.device)

        for batch in dataloader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)  # assumed one-hot encoded
            out = self.model.inference(x)

            # Compute loss and entropy
            loss = self.model.loss(out, y, reduction='none')  # shape: (batch_size,)
            # entropy = -(out['probabilities'] * torch.log(out['probabilities'] + 1e-8)).sum(dim=-1)  # shape: (batch_size,)
            total_loss = loss #entropy  # shape: (batch_size,)

            # Get class indices from one-hot labels
            class_indices = y.argmax(dim=-1)  # shape: (batch_size,)

            # Accumulate total_loss per class
            class_sums += torch.bincount(class_indices, weights=total_loss, minlength=n_classes)
            class_counts += torch.bincount(class_indices, minlength=n_classes)

        # Avoid division by zero
        class_means = torch.where(class_counts > 0, class_sums / class_counts, torch.zeros_like(class_sums))
        return class_means

    def bo_objective_function(self,
                          parameterization,
                          FIM_full,
                          FIM_forget,
                          forget_loader: DataLoader,
                          gen_losses: dict,
                          sd_original,
                          n_classes: int):
        """Objective function for BoTorch optimization"""
        # Extract parameters from the parameterization dict - directly on device
        alphas = torch.tensor([parameterization[f'alpha']], device=self.device)
        lambdas = torch.tensor([parameterization[f'_lambda']], device=self.device)

        # Load original weights
        self.model.load_state_dict(sd_original)
        # Make sure parameters were reset
        assert check_statedict_equivalent(self.model.state_dict(), sd_original), 'Model parameters were not reset to original values!'
        # prepare inputs
        kwargs = {
            'alphas': alphas,
            'lambdas': lambdas
        }
        # Update model parameteres
        self.update_gpu_parameters(FIM_full, FIM_forget, **kwargs)

        # Calculate loss on forget set
        forget_losses = self.calculate_loss(forget_loader, n_classes)

        # Calculate squared error between forget and generalization loss
        if not isinstance(gen_losses, torch.Tensor):
            gen_losses = torch.tensor(gen_losses, device=self.device)

        diff = ((forget_losses - gen_losses)**2).mean()
        return -diff

    def update_gpu_parameters(self, FIM_full, FIM_forget, alphas, lambdas, return_dampening: bool = False):
        all_dampenings = {}
        """GPU-optimized version of update_parameters"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                
                # Use tensor values directly from GPU
                alpha = alphas[0]
                _lambda = lambdas[0]

                updated_parameter = param.data.clone()
                dampen_mask = FIM_forget[name] > alpha * FIM_full[name]  # find which parameters to dampen

                # Stay on GPU for all operations
                if dampen_mask.any():
                    beta = torch.minimum((_lambda * FIM_full[name][dampen_mask] / FIM_forget[name][dampen_mask]),
                                    torch.ones(1, device=self.device))
                    updated_parameter[dampen_mask] = beta * updated_parameter[dampen_mask]

                # update parameter in the model
                param.copy_(updated_parameter)
                if return_dampening:
                    dampen_values = torch.ones_like(dampen_mask, dtype = torch.float)
                    if dampen_mask.any():
                        dampen_values[dampen_mask] = beta
                    all_dampenings[name] = dampen_values
        
        if return_dampening:
            return all_dampenings
    
    def search_hyperparameters_bo(self,
                                FIM_full,
                                FIM_forget,
                                forget_loader: DataLoader,
                                gen_losses: np.array,
                                sd_original,
                                n_classes: int):
        """Bayesian optimization using BoTorch with full GPU optimizations"""

        # Start timer for performance measurement
        start_time = time.time()

        # Convert gen_losses to tensor if not already
        if not isinstance(gen_losses, torch.Tensor):
            gen_losses = torch.tensor(gen_losses, device=self.device)

        # Define raw bounds for the problem
        bounds = torch.tensor([
            [0.1, 0.1], # Lower bounds (alpha, lambda)
            [1000.0, 100.0] # Upper bounds (alpha, lambda)
        ], dtype=torch.double, device=self.device)

        # Prepare objective function wrapper with GPU optimization
        def wrapped_objective(x_dict):

            return self.bo_objective_function(
                parameterization=x_dict,
                FIM_full=FIM_full,
                FIM_forget=FIM_forget,
                forget_loader=forget_loader,
                gen_losses=gen_losses,
                sd_original=sd_original,
                n_classes=n_classes
            )

        # Use Sobol sequence for better initial coverage of the search space
        n_initial = 10
        sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        train_x = sobol_engine.draw(n_initial).to(dtype=torch.double, device=self.device)

        # Evaluate initial points in parallel if possible
        train_obj_list = []

        # Process initial points in batches for memory efficiency
        batch_size = min(5, n_initial)  # Adjust based on GPU memory
        for b in range(0, n_initial, batch_size):
            batch_end = min(b + batch_size, n_initial)
            batch_x = train_x[b:batch_end]
            batch_obj = torch.zeros(len(batch_x), dtype=torch.double, device=self.device)

            # Evaluate each point in the batch
            for j, x in enumerate(batch_x):
                params = {'alpha': x[0].item(),
                          '_lambda': x[1].item()}
                batch_obj[j] = wrapped_objective(params)

            train_obj_list.append(batch_obj)

        train_obj = torch.cat(train_obj_list)

        # Standardize for better GP performance
        # train_obj = standardize(train_obj)

        # Tracking best results
        best_value = train_obj.max().item()
        best_idx = train_obj.argmax().item()
        best_params = train_x[best_idx].clone()

        # Configure optimization parameters
        n_iterations = self.n_bo_iter  # You can adjust this based on your needs

        # For better performance on GPU
        num_restarts = 20  # Increased for better exploration
        raw_samples = 256  # Increased for better initial points in acquisition optimization
        
        max_refits = 4 # max number of times to re-fit the GP hyperparameters during optimization
        num_refits = 0

        # Keep track of all evaluated points
        all_x = train_x.clone()
        all_obj = train_obj.clone()

        # Set up GP model parameters
        gp = SingleTaskGP(
            train_x,
            train_obj.unsqueeze(-1),
            input_transform=Normalize(d=train_x.shape[-1]),
            outcome_transform=Standardize(m=1)
            )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Main optimization loop
        for iteration in range(n_iterations):
            iter_start = time.time()

            # Define acquisition function - LogEI often works better than regular EI
            # EI = LogExpectedImprovement(model=gp, best_f=train_obj.max(), maximize=True)
            UCB = UpperConfidenceBound(model=gp, beta=2.5)
            
            #### NEW CODE ####
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                candidate, acq_value = optimize_acqf(
                    acq_function=UCB,#EI,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"batch_limit": 10, "maxiter": 100},
                )
                bad_candidates_warning = any(
                                issubclass(warning.category, BadInitialCandidatesWarning) 
                                for warning in w
                            )
        
            # Extract parameters for the new candidate point
            new_params = {'alpha': candidate[0, 0].item(),
                          '_lambda': candidate[0, 1].item()}
            
            new_obj = wrapped_objective(new_params)

            # Update training points
            train_x = torch.cat([train_x, candidate])
            train_obj = torch.cat([train_obj, torch.tensor([new_obj], dtype=torch.double, device=self.device)])

            # Update all evaluated points
            all_x = torch.cat([all_x, candidate])
            all_obj = torch.cat([all_obj, torch.tensor([new_obj], dtype=torch.double, device=self.device)])

            # update GP with new observations
            gp.set_train_data(train_x, train_obj, strict=False)

            if bad_candidates_warning and num_refits <= max_refits:
                print("RE-FITTING GP!!")
                gp = SingleTaskGP(
                train_x,
                train_obj.unsqueeze(-1),
                input_transform=Normalize(d=train_x.shape[-1]),
                outcome_transform=Standardize(m=1)
                )
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                num_refits += 1
            
            # Update best params
            current_best_idx = train_obj.argmax().item()
            current_best = train_obj[current_best_idx].item()
            if current_best > best_value:
                best_value = current_best
                best_params = train_x[current_best_idx].clone()

            # Report progress
            iter_time = time.time() - iter_start
            print(f"Iteration {iteration+1}/{n_iterations} completed in {iter_time:.2f}s, best value: {best_value:.6f}")

        # Unpack the best parameters back to the original space
        best_final_params = {'alpha': best_params[0].item(),
                             '_lambda': best_params[1].item(),
                             'target': -best_value # convert back from negative
                             }

        # Format results
        all_results = []
        for i in range(len(all_x)):
            params_dict = {'alpha': all_x[i, 0].item(),
                           '_lambda': all_x[i, 1].item()}
            
            all_results.append({
                'params': params_dict,
                'target': all_obj[i].item()
            })

        total_time = time.time() - start_time
        print(f"Optimization completed in {total_time:.2f}s with {n_initial + n_iterations} function evaluations")
        return {'result': all_results, 'max': best_final_params}

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
                 FIM_forget: dict = None,
                 return_dampening: bool = False
                 ):

        start = time.time()
        # calculate FIM matrices if necessary
        if FIM_forget is None:
            FIM_forget = self.calculate_FIM(forget_dataloader)

        if FIM_full is None:
            FIM_full = self.calculate_FIM(full_dataloader)
        end=time.time()
        print(f'FIM computation time: {end-start:.4f} sec.')
        sd_original = copy.deepcopy(self.model.state_dict())
        n_classes = torch.unique(validation_dataloader.dataset.y, dim=0).size(0)

        start = time.time()
        generalization_dataloader, forget_dataloader = self.construct_validation_set(forget_dataloader, validation_dataloader)
        generalization_losses = self.calculate_loss(generalization_dataloader, n_classes)

        # ### Update forget dataloader to include all classes (avoids catastrophic forgetting)
        # n_forget_classes = torch.unique(forget_dataloader.dataset.y, dim=0).size(0)
        # if n_classes != n_forget_classes:
        #     forget_dataloader_old = copy.deepcopy(forget_dataloader)
        #     forget_dataloader = updated_forget_loader

        end=time.time()
        print(f'repair set time: {end-start:.4f} sec.')

        start = time.time()
        # find optimal alpha and lambda values via bayesian optimization
        bo_result = self.search_hyperparameters_bo(FIM_full, FIM_forget, forget_dataloader, generalization_losses, sd_original, n_classes)
        end=time.time()
        print(f'bayesian opt time: {end-start:.4f} sec.')
        best_params = self.select_optimal_parameters(bo_result)

        # apply original parameters
        self.model.load_state_dict(sd_original)
        # print(f'SD identical? {self.check_statedict_equivalent(sd_original, self.model.state_dict())}')
        alphas = torch.tensor([best_params[f'alpha']], device=self.device)
        lambdas = torch.tensor([best_params['_lambda']], device=self.device)
        
        all_dampenings = self.update_gpu_parameters(FIM_full, FIM_forget, **{'alphas': alphas, 'lambdas': lambdas}, return_dampening=return_dampening)
        # print(f'SD identical? {self.check_statedict_equivalent(sd_original, self.model.state_dict())}')
        # return best_params
        return bo_result['max'], all_dampenings


"""
OLD CODE BELOW:
"""

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from src.unlearners.base_unlearner import BaseUnlearner
# from src.datasets.synthetic_dataset import SyntheticDataset
# from bayes_opt import BayesianOptimization
# from functools import partial
# import copy
# import pdb
# from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening as SSD

# class SelectiveSynapticDampening(SSD):
#     def __init__(self, 
#                  model,
#                  device: str = 'cpu') -> None:
#         super().__init__(model, alpha=None, _lambda=None, device=device)
    
#     def construct_validation_set(self, forget_dataloader, val_dataloader):
#         """
#         A function for constructing a validation set that is "of the same distribution" as the forget dataset. This is the +R step in the method.
#         "of the same distribution" is interpreted as being in terms of the label distribution.
#         """
#         forget_dataset = forget_dataloader.dataset
#         val_dataset = val_dataloader.dataset
#         X_val = val_dataset.X
#         y_val_ohe = val_dataset.y
#         y_val = val_dataset.y.argmax(dim=1)
#         n_classes = int(y_val.max()+1) # assumes all classes are in the validation set...
#         labels, counts = torch.unique(torch.argmax(forget_dataset.y, dim=1), return_counts=True)

#         sample_sizes = []

#         X_values = []
#         y_values = []
#         for i, label in enumerate(labels):
#             # find val set indexes that correspond to the label
#             val_label_idx = torch.where(y_val == label)[0]
#             # find sample size
#             sample_size = min(counts[i].item(), len(val_label_idx))
#             sample_sizes.append(sample_size)
#             # draw random samples
#             idxs = torch.randperm(len(val_label_idx))[:sample_size]
#             # draw subset of data based on index
#             X = X_val[val_label_idx[idxs]]
#             y = y_val_ohe[val_label_idx[idxs]]
#             X_values.append(X)
#             y_values.append(y)

#         # check if exact distribution could be constructed..
#         if not (torch.tensor(sample_sizes) == counts).all():
#             print("Exact distribution could not be constructed..")

#         X_values = torch.cat(X_values)
#         y_values = torch.cat(y_values)

#         dataset = SyntheticDataset(X_values, y_values, n_classes=n_classes)
#         dataloader = DataLoader(dataset, batch_size=val_dataloader.batch_size)
#         return dataloader

#     def calculate_loss(self, dataloader, n_classes = int):
#         losses = {i: [] for i in range(n_classes)}

#         self.model.eval()
#         with torch.no_grad():
#             for batch in dataloader:
#                 x = batch[0]
#                 y = batch[1]
#                 out = self.model(x)
#                 loss = self.model.loss(out, y)
#                 for i in range(y.size(0)):
#                     losses[y[i].argmax(dim=-1).item()].append(loss.item())
#         # losses = {k: np.mean(v) if len(v) else 0 for k, v in losses.items()}
        
#         losses = np.array([np.mean(e) if len(e) else 0 for e in losses.values()])
#         return losses
    
#     def update_parameters(self, FIM_full, FIM_forget, alpha, _lambda, return_dampening: bool = False):
#         all_dampenings = {}
#         # go through the parameters
#         with torch.no_grad():
#             for name, param in self.model.named_parameters():
#                 updated_parameter = param.data.clone()
#                 dampen_mask = FIM_forget[name] > alpha * FIM_full[name] # find which paramters to dampen
#                 # print(f'Original parameter: {param}')
#                 # print(f'Dampen mask: {dampen_mask}')
#                 # calculate dampening factors and apply them
#                 beta = torch.min((_lambda * FIM_full[name][dampen_mask] / FIM_forget[name][dampen_mask]), torch.tensor(1))
#                 updated_parameter[dampen_mask] = beta * updated_parameter[dampen_mask]
#                 # update parameter in the model
#                 param.copy_(updated_parameter)
#                 if return_dampening:
#                     dampen_values = torch.ones_like(dampen_mask, dtype = beta.dtype)
#                     dampen_values[dampen_mask] = beta
#                     all_dampenings[name] = dampen_values
        
#         if return_dampening:
#             return all_dampenings

            
        
#     def search_hyperparams_exhaustive(self, 
#                            alphas: list[float], 
#                            lambdas: list[float], 
#                            FIM_full, FIM_forget, 
#                            forget_loader,
#                            gen_losses: np.array, 
#                            sd_original,
#                            n_classes):
        
#         best_diff = np.inf
#         best_params = {'alpha': None,
#                        'lambda': None}
#         for alpha in alphas:
#             for _lambda in lambdas:
#                 self.model.load_state_dict(sd_original)
#                 self.update_parameters(FIM_full, FIM_forget, alpha, _lambda)
#                 forget_losses = np.array(list(self.calculate_loss(forget_loader, n_classes).values()))
#                 diff = np.mean((forget_losses - gen_losses)**2)
#                 if diff < best_diff:
#                     best_diff = diff
#                     best_params.update({'alpha': alpha, 'lambda': _lambda})
        
#         return best_params, best_diff
    
#     def bo_objective_function(self, 
#                               alpha, 
#                               _lambda,
#                               FIM_full,
#                               FIM_forget,
#                               forget_loader: DataLoader,
#                               gen_losses: dict,
#                               sd_original,
#                               n_classes: int,):
#         # load original weights
#         self.model.load_state_dict(sd_original)
#         # make sure parameters were reset
#         assert not self.check_statedict_diff(self.model.state_dict(), sd_original), 'Model parameters were not reset to original values!'
#         # update parameters
#         self.update_parameters(FIM_full, FIM_forget, alpha, _lambda)
#         # calculate loss on forget set
#         forget_losses = self.calculate_loss(forget_loader, n_classes)
#         # calculate squared error between forget an generalization loss
#         diff = np.mean((forget_losses - gen_losses)**2)
#         return -diff

#     def search_hyperparameters_bo(self,
#                                   FIM_full,
#                                   FIM_forget,
#                                   forget_loader: DataLoader,
#                                   gen_losses: np.array, 
#                                   sd_original,
#                                   n_classes: int,
#                                   ):
        
#         pbounds = {'alpha': (0.01, 10), '_lambda': (0.1, 50)}
#         objective_func = partial(self.bo_objective_function,
#                             FIM_full=FIM_full,
#                             FIM_forget=FIM_forget,
#                             forget_loader=forget_loader,
#                             gen_losses=gen_losses,
#                             sd_original=sd_original,
#                             n_classes=n_classes)
#         bayesian_optimizer = BayesianOptimization(
#             f=objective_func,
#             pbounds=pbounds,
#             random_state=1,
#             verbose=0
#         )
#         bayesian_optimizer.maximize(n_iter=100)
        
#         return {'result': bayesian_optimizer.res, 'max': bayesian_optimizer.max}

#     def check_statedict_diff(self, sd1, sd2):
#         """
#         returns true if state dicts are different and false otherwise
#         """
#         for key in sd1.keys():
#             if not torch.isclose(sd1[key], sd2[key]).all():
#                 return True
#         return False
    
#     def select_optimal_parameters(self, bo_result: dict):
#         """
#         If there are multiple optimal parameters: Choose the one which changes the original model the least.
#         """
#         scores = [e['target'] for e in bo_result['result']]
#         best_scores_idxs = np.where(scores == max(scores))[0]
#         best_params = [bo_result['result'][i]['params'] for i in best_scores_idxs]
#         param_sums = [sum((e['_lambda'], e['alpha'])) for e in best_params]
#         return best_params[np.argmax(param_sums)]

#     def __call__(self, 
#                  full_dataloader,
#                  forget_dataloader,
#                  validation_dataloader,
#                  FIM_full: dict = None, 
#                  FIM_forget: dict = None,
#                  return_dampening: bool = False
#                  ):
        
#         # calculate FIM matrices if necessary
#         if FIM_forget is None:
#             FIM_forget = self.calculate_FIM(forget_dataloader)
        
#         if FIM_full is None:
#             FIM_full = self.calculate_FIM(full_dataloader)
        
#         sd_original = copy.deepcopy(self.model.state_dict())
#         n_classes = validation_dataloader.dataset.y.argmax(dim=-1).max() + 1
#         generalization_dataloader = self.construct_validation_set(forget_dataloader, validation_dataloader)
#         generalization_losses = self.calculate_loss(generalization_dataloader, n_classes)
        
#         # forget_losses = self.calculate_loss(forget_dataloader, n_classes)
#         # alphas = np.arange(start=0.01, stop=1.0, step=0.01)
#         # lambdas = np.arange(start=10, stop=50., step=0.1)
#         # best_params, best_diff = self.search_hyperparams_exhaustive(alphas, lambdas, FIM_full, FIM_forget, forget_dataloader, generalization_losses, sd_original, n_classes)
        
#         # find optimal alpha and lambda values via bayesian optimization
#         bo_result = self.search_hyperparameters_bo(FIM_full, FIM_forget, forget_dataloader, generalization_losses, sd_original, n_classes)
#         best_params = self.select_optimal_parameters(bo_result)

#         alpha_opt = best_params['alpha'] # bo_result['max']['params']['alpha']
#         lambda_opt = best_params['_lambda'] # bo_result['max']['params']['_lambda']
        
#         # reset state dict
#         self.model.load_state_dict(sd_original)
#         # update parameters
#         all_dampenings = self.update_parameters(FIM_full, FIM_forget, alpha_opt, lambda_opt, return_dampening=return_dampening)
        
#         return best_params, all_dampenings
#         # alpha_opt = bo_result['max']['params']['alpha']
#         # lambda_opt = bo_result['max']['params']['_lambda']
#         # return bo_result['max']['params']