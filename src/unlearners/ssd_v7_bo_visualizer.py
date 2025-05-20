from src.unlearners.selective_synaptic_dampening_v7 import SelectiveSynapticDampening
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial
import copy
import pdb
import optuna

class SSDVisualizer(SelectiveSynapticDampening):
    def __init__(self, model, P: int, k: int, n_trials: int = 100, device: str = 'cpu') -> None:
        super().__init__(model, P, k, n_trials, device)
    
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
            params.update({f'_lambda_{i}': 1 for i in range(self.P)})
            loss = objective_func(**params)
            return float(loss)

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
        bo_result = self.search_hyperparameters_TPE(FIM_full, FIM_forget, forget_dataloader, generalization_losses, sd_original, n_classes)
        best_params = bo_result['max']['params']
        best_params.update({'_lambda_0': 1})
        # best_params = self.select_optimal_parameters(opt_result)
        
        # go through the parameters
        self.model.load_state_dict(sd_original)
        # print(f'SD identical? {self.check_statedict_equivalent(sd_original, self.model.state_dict())}')
        self.update_parameters(FIM_full, FIM_forget, **best_params)
        # print(f'SD identical? {self.check_statedict_equivalent(sd_original, self.model.state_dict())}')
        return bo_result