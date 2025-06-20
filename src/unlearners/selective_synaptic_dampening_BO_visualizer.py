from torch.utils.data import DataLoader
import numpy as np
import torch
import copy
import pdb
import time
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
from unlearners.selective_synaptic_dampening_BO_pairwise import SelectiveSynapticDampeningBOPairwise
from src.utils.misc import check_statedict_equivalent

class SSDVisualizer(SelectiveSynapticDampeningBOPairwise):
    def __init__(self, model, 
                 P: int = 1, 
                 k: int = 0.999, 
                 smooth_dampening: bool = False, 
                 n_bo_iter: int = 100, 
                 device: str = 'cpu') -> None:
        super().__init__(model, P, k, smooth_dampening, n_bo_iter, device)

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
        lambdas = torch.ones_like(alphas) * 1

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
            [0.1] * self.P,  # Lower bounds
            [200.0] * self.P  # Upper bounds
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
        sobol_engine = torch.quasirandom.SobolEngine(dimension=self.P, scramble=True)
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
                    params[f'alpha_{i}'] = x[i].item()
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
                acq_function=UCB,
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
            

        best_raw_params['target'] = -best_value  # Convert back from negative

        # Format results similar to the original BayesianOptimization package
        all_results = []
        for i in range(len(all_x)):
            params_dict = {}
            for j in range(self.P):
                params_dict[f'alpha_{j}'] = all_x[i, j].item()

            # We need to work with the original (non-standardized) objective values
            all_results.append({
                'params': params_dict,
                'target': all_obj[i].item()  # Convert back from negative
            })

        total_time = time.time() - start_time
        print(f"Optimization completed in {total_time:.2f}s with {n_initial + n_iterations} function evaluations")
        return {'result': all_results, 'max': best_raw_params}
    
    def search_hyperparameters_exhaustive(self, full_loader, forget_loader, val_loader, alphas):
        
        targets = []
        sd_original = copy.deepcopy(self.model.state_dict())
        FIM_full = self.calculate_FIM(full_loader)
        FIM_forget = self.calculate_FIM(forget_loader)
        n_classes = full_loader.dataset.y.size(1)

        generalization_dataloader, updated_forget_loader = self.construct_validation_set(forget_loader, val_loader)
        generalization_losses = self.calculate_loss(generalization_dataloader, n_classes)
        
        def wrapped_objective(x_dict):

            return self.bo_objective_function(
                parameterization=x_dict,
                FIM_full=FIM_full,
                FIM_forget=FIM_forget,
                forget_loader=forget_loader,
                gen_losses=generalization_losses,
                sd_original=sd_original,
                n_classes=n_classes
            )
        
        for i, alpha in enumerate(alphas):
            new_params = {'alpha_0': alpha}
            L_bo = wrapped_objective(new_params)
            targets.append(L_bo.item())
        
        return targets

    def __call__(self,
                 full_dataloader,
                 forget_dataloader,
                 validation_dataloader,
                 FIM_full: dict = None,
                 FIM_forget: dict = None
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

        # best_params = self.select_optimal_parameters(bo_result)

        # apply original parameters
        self.model.load_state_dict(sd_original)
        # print(f'SD identical? {self.check_statedict_equivalent(sd_original, self.model.state_dict())}')
        alphas = torch.tensor([bo_result['max'][f'alpha_{i}'] for i in range(self.P)], device=self.device)
        lambdas = torch.ones_like(alphas)
        
        # update with optimal parameters
        if self.smooth_dampening:
            self.update_gpu_parameters_smooth(FIM_full, FIM_forget, **{'alphas': alphas, 'lambdas': lambdas})
        else:
            self.update_gpu_parameters(FIM_full, FIM_forget, **{'alphas': alphas, 'lambdas': lambdas})
        # print(f'SD identical? {self.check_statedict_equivalent(sd_original, self.model.state_dict())}')
        # return best_params
        return bo_result