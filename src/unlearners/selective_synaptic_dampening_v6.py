import numpy as np
from functools import partial
from math import floor
import copy
import time
import pdb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# bayesian optimization modules
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from botorch.models.transforms import Normalize, Standardize
from src.unlearners.base_unlearner import BaseUnlearner
from src.datasets.synthetic_dataset import SyntheticDataset
from src.utils.misc import check_statedict_equivalent

"""
A torch compatible implementation of SSD v5 with the following additions:

- Sampling from the top k% highest entropy values in forget dataset.
- A more smooth update parameter rule compared to SSD. 
"""

from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening as SSD

class SelectiveSynapticDampening(SSD):
    def __init__(self,
                 model,
                 P: int, # how many pairs of (alpha, lambda) values to use for dampening
                 k: int, # top-k % entropy test data to sample from when constructing repair set
                 smooth_dampening: bool, # which dampening rule to apply
                 n_bo_iter: int, # how many iterations to run bayesian optimization
                 device: str = 'cpu') -> None:
        super().__init__(model, alpha=None, _lambda=None, device=device)
        assert 0 < k <= 1, 'k must range between [0, 1]'
        self.P = P
        self.k = k
        self.smooth_dampening = smooth_dampening
        self.n_bo_iter = n_bo_iter 

        self.n_param_groups = len(list(self.model.parameters()))
        # assuming all layers have both weight & bias terms
        assert self.n_param_groups // 2 >= self.P, 'Cannot have more pairs of (alpha, lambda) values than the number of layers!'

    def layer_idx_to_parameter_idx(self, idx):
        return floor((idx * self.P) / (self.n_param_groups + 1))

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
        
        val_labels, val_counts = torch.unique(y_val, return_counts=True)
        # not all classes are represented in the forget set --> this may lead to catastrophic forgetting when choosing the best param update
        X_forget_add = []
        y_forget_add = []
        generator = torch.Generator()
        generator.manual_seed(self.model.seed)
        # if not (val_labels == forget_labels).all():
        #     X_forget_add.append(forget_dataloader.dataset.X)
        #     y_forget_add.append(forget_dataloader.dataset.y)
        #     for i, label in enumerate(val_labels):
        #         if label in forget_labels: # continue if samples from class already exist
        #             continue
        #         # add samples from remaining class(es) to "generalization" set
        #         sample_size = min(np.min(sample_sizes) // (len(val_labels) - len(forget_labels)), val_counts[i].item() // 2)
        #         val_label_idx = torch.where(y_val == label)[0]
        #         idxs = torch.randperm(len(val_label_idx))[:int(2 * sample_size)]

        #         X = X_val[val_label_idx[idxs[:len(idxs) // 2]]]
        #         y = y_val_ohe[val_label_idx[idxs[:len(idxs) // 2]]]
        #         X_values.append(X)
        #         y_values.append(y)

        #         # add to forget set for missing classes
        #         X = X_val[val_label_idx[idxs[len(idxs) // 2:]]]
        #         y = y_val_ohe[val_label_idx[idxs[len(idxs) // 2:]]]
        #         X_forget_add.append(X)
        #         y_forget_add.append(y)

        #     X_forget = torch.cat(X_forget_add)
        #     y_forget = torch.cat(y_forget_add)
        #     forget_dataset = SyntheticDataset(X_forget, y_forget, n_classes=n_classes)

        #     updated_forget_dataloader = DataLoader(forget_dataset, batch_size=4, generator=generator)

        X_values = torch.cat(X_values)
        y_values = torch.cat(y_values)
        
        dataset = SyntheticDataset(X_values.clone().detach(), y_values.clone().detach(), n_classes=n_classes)
        generalization_dataloader = DataLoader(dataset, batch_size=32, generator=generator)
        
        # if len(X_forget_add): # if additions to forget dataset were made, return it
        #     return generalization_dataloader, updated_forget_dataloader
        
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
            # entropy = -(out['probabilities'] * torch.log(out['probabilities'] + 1e-8)).sum(dim=-1)  # shape: (batch_size,)
            total_loss = loss #+ entropy  # shape: (batch_size,)

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
        alphas = torch.tensor([parameterization[f'alpha_{i}'] for i in range(self.P)], device=self.device)
        lambdas = torch.tensor([parameterization[f'_lambda_{i}'] for i in range(self.P)], device=self.device)

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
        if self.smooth_dampening:
            self.update_gpu_parameters_smooth(FIM_full, FIM_forget, **kwargs)
        else:
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
        """GPU-optimized version of update_parameters that avoids .item() calls"""
        with torch.no_grad():
            for i, (name, param) in enumerate(self.model.named_parameters()):
                param_idx = self.layer_idx_to_parameter_idx(i)

                # Use tensor values directly from GPU
                alpha = alphas[param_idx]
                _lambda = lambdas[param_idx]

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
    
    def update_gpu_parameters_smooth(self, FIM_full, FIM_forget, alphas, lambdas, return_dampening: bool = False):
        
        """
        A more smooth variant of the SSD update. Trades off the selectiveness to get a more smooth objective function surface.
        The update changes to:

        d = sigmoid(lambda * (FIM(D_forget) / FIM(D_train) - alpha))
        theta_i = (1-d) * theta_i

       Notice:
        - If lambda is very high, the update becomes more selective.
        - If the ratio FIM(D_forget) / FIM(D_train) is very high then we remove those parameters entriely.
        """
        all_dampenings = {}
        with torch.no_grad():
            for i, (name, param) in enumerate(self.model.named_parameters()):
                param_idx = self.layer_idx_to_parameter_idx(i)

                # Use tensor values directly from GPU
                alpha = alphas[param_idx]
                _lambda = lambdas[param_idx]

                updated_parameter = param.data.clone()
                dampen_factor = 1 - torch.nn.functional.sigmoid(_lambda * (FIM_forget[name] / (FIM_full[name]+1e-20) - alpha))
                updated_parameter = dampen_factor * updated_parameter

                
                # Stay on GPU for all operations
                # if dampen_mask.any():
                #     beta = torch.minimum((_lambda * FIM_full[name][dampen_mask] / FIM_forget[name][dampen_mask]),
                #                     torch.ones(1, device=self.device))
                #     updated_parameter[dampen_mask] = beta * updated_parameter[dampen_mask]

                # update parameter in the model
                param.copy_(updated_parameter)
                if return_dampening:
                    all_dampenings[name] = dampen_factor
        
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
            [0.1] * self.P + [0.1] * self.P,  # Lower bounds
            [1000.0] * self.P + [100.0] * self.P  # Upper bounds
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
        n_initial = 30
        sobol_engine = torch.quasirandom.SobolEngine(dimension=2*self.P, scramble=True)
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
                params = {}
                for i in range(self.P):
                    params[f'alpha_{i}'] = x[i].item()  # Still need .item() for dict keys
                    params[f'_lambda_{i}'] = x[i + self.P].item()
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

            # Use higher num_restarts for better convergence and better GPU utilization
            candidate, acq_value = optimize_acqf(
                acq_function=UCB,#EI,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": 10, "maxiter": 100},
            )

            # Extract parameters for the new candidate point
            new_params = {}
            for i in range(self.P):
                # Still need to use .item() here for dictionary keys
                new_params[f'alpha_{i}'] = candidate[0, i].item()
                new_params[f'_lambda_{i}'] = candidate[0, i + self.P].item()
            # Evaluate new point
            new_obj = wrapped_objective(new_params)

            # Update training points
            train_x = torch.cat([train_x, candidate])
            train_obj = torch.cat([train_obj, torch.tensor([new_obj], dtype=torch.double, device=self.device)])

            # Update all evaluated points
            all_x = torch.cat([all_x, candidate])
            all_obj = torch.cat([all_obj, torch.tensor([new_obj], dtype=torch.double, device=self.device)])

            # Re-standardize the objectives
            # train_obj = standardize(train_obj)

            # update GP with new observations
            gp.set_train_data(train_x, train_obj, strict=False)
            
            # # re-learn optimal hyperparameters of GP
            # gp = SingleTaskGP(
            # train_x,
            # train_obj.unsqueeze(-1),
            # input_transform=Normalize(d=train_x.shape[-1]),
            # outcome_transform=Standardize(m=1)
            # )
            # mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            # fit_gpytorch_mll(mll)

            # Update best params
            current_best_idx = train_obj.argmax().item()
            current_best = train_obj[current_best_idx].item()
            if current_best > best_value:
                best_value = current_best
                best_params = train_x[current_best_idx].clone()

            # Report progress
            iter_time = time.time() - iter_start
            print(f"Iteration {iteration+1}/{n_iterations} completed in {iter_time:.2f}s, best value: {best_value:.6f}")

        # Unnormalize the best parameters back to the original space

        best_raw_params = {}
        for i in range(self.P):
            best_raw_params[f'alpha_{i}'] = best_params[i].item()
            best_raw_params[f'_lambda_{i}'] = best_params[i + self.P].item()

        best_raw_params['target'] = -best_value  # Convert back from negative

        # Format results similar to the original BayesianOptimization package
        all_results = []
        for i in range(len(all_x)):
            params_dict = {}
            for j in range(self.P):
                params_dict[f'alpha_{j}'] = all_x[i, j].item()
                params_dict[f'_lambda_{j}'] = all_x[i, j+self.P].item()

            # We need to work with the original (non-standardized) objective values
            all_results.append({
                'params': params_dict,
                'target': all_obj[i].item()  # Convert back from negative
            })

        total_time = time.time() - start_time
        print(f"Optimization completed in {total_time:.2f}s with {n_initial + n_iterations} function evaluations")
        return {'result': all_results, 'max': best_raw_params}

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
        generalization_dataloader, updated_forget_loader = self.construct_validation_set(forget_dataloader, validation_dataloader)
        generalization_losses = self.calculate_loss(generalization_dataloader, n_classes)

        n_forget_classes = torch.unique(forget_dataloader.dataset.y, dim=0).size(0)
        if n_classes != n_forget_classes:
            forget_dataloader_old = copy.deepcopy(forget_dataloader)
            forget_dataloader = updated_forget_loader

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
        alphas = torch.tensor([best_params[f'alpha_{i}'] for i in range(self.P)], device=self.device)
        lambdas = torch.tensor([best_params[f'_lambda_{i}'] for i in range(self.P)], device=self.device)
        
        # update with optimal parameters
        if self.smooth_dampening:
            all_dampenings = self.update_gpu_parameters_smooth(FIM_full, FIM_forget, **{'alphas': alphas, 'lambdas': lambdas}, return_dampening=return_dampening)
        else:
            all_dampenings = self.update_gpu_parameters(FIM_full, FIM_forget, **{'alphas': alphas, 'lambdas': lambdas}, return_dampening=return_dampening)
        # print(f'SD identical? {self.check_statedict_equivalent(sd_original, self.model.state_dict())}')
        # return best_params
        return bo_result, all_dampenings