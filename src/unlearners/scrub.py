import torch
import torch.nn as nn
import torch.optim as optim
import copy
import pdb
from src.data_utils.synthetic_data import SyntheticDataset
from torch.utils.data import DataLoader
import numpy as np
import os
from src.unlearners.base_unlearner import BaseUnlearner

class ScrubR(BaseUnlearner):
    def __init__(self, model, original_model, alpha, gamma, device: str, MIA: callable = None, js_div_func: callable = None, retrain_model: callable = None):
        super().__init__(model, {'alpha': alpha, 'gamma': gamma})
        self.original_model = original_model
        # self.__freeze_original_model()
        # self.CE = nn.CrossEntropyLoss(reduction='sum')
        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.KL = nn.KLDivLoss(reduction='batchmean')
        self.alpha = alpha # hyperparam for distance between student & teacher on retain data
        self.gamma = gamma # hyperparam for cross entropy
        self.device = device
        self.MIA = MIA

        # if js_div_func is provided, then we need the retrain_model to be provided as well
        if js_div_func is not None:
            assert retrain_model is not None, "retrain_model must be provided if js_div_func is provided"
        self.retrain_model = retrain_model
        self.js_div_func = js_div_func
        
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-3)
        self.log_softmax = nn.LogSoftmax(dim=-1)

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

        # set generator for reproducibility
        generator = torch.Generator()
        generator.manual_seed(self.model.seed)
        # Ensure we're using Python native types, not torch.int64
        dataset = SyntheticDataset(X_values.numpy(), y_values.numpy(), n_classes=int(n_classes))
        dataloader = DataLoader(dataset, batch_size=val_dataloader.batch_size)
        return dataloader


    def calculate_accuracy(self, model, dataloader):
        model.eval()
        correct = 0
        total = 0
        for batch in dataloader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            logits = model.inference(x)['logits']
            preds = logits.argmax(dim=1)
            # Convert one-hot encoded y to class indices
            y_indices = y.argmax(dim=1)
            correct += (preds == y_indices).sum().item()
            total += len(y)

        model.train()
        return correct / total

    def calculate_error(self, model, dataloader, return_scalar=True):
        model.eval()
        losses = []
        
        for batch in dataloader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            logits = model.inference(x)['logits']
            loss = self.CE(logits, y)
            losses.append(loss)

        if return_scalar:
            return torch.mean(torch.cat(losses))
        
        return torch.cat(losses)

    def max_step(self, x, y):
        """
        Maximization step of the SCRUB loss: Encourages the model to have high error on forget data.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        student_log_probs = self.log_softmax(self.model(x)['logits'])
        teacher_probs = self.original_model.inference(x)['probabilities']
        loss = -self.KL(input=student_log_probs, target=teacher_probs) # maximize KL divergence
        loss.backward()
        self.optimizer.step()

    def min_step(self, x, y):
        """
        Miminimzation step of the SCRUB loss: Encourages performance to have low error and have similar outputs to original model on retain data.
        """
        self.optimizer.zero_grad()

        student_out = self.model(x)
        student_log_probs = self.log_softmax(student_out['logits'])
        teacher_probs = self.original_model.inference(x)['probabilities']

        reg_term = self.KL(input=student_log_probs, target=teacher_probs)
        fit_term = self.CE(student_out['logits'], y).mean()
        loss = self.alpha * reg_term + self.gamma * fit_term
        # print(f'Min loss: {loss}')
        loss.backward()
        self.optimizer.step()

    def __call__(self, retain_dataloader, forget_dataloader, val_dataloader, n_rounds: int, remove_weights: bool = True, verbose: bool = False):
        validate_err_dataloader = self.construct_validation_set(forget_dataloader, val_dataloader)
        forget_errors = []
        
        metrics = {'retain': {'acc': [], 'loss': []},
                   'forget': {'acc': [], 'loss': []},
                   'val': {'acc': [], 'loss': []}
                  }
        if self.MIA is not None:
            metrics['mia'] = []

        if self.js_div_func is not None:
            metrics['js_div'] = {"retain": [], "forget": [], "val": []}
        
        # Create epoch iterator with tqdm if verbose
        epoch_iterator = range(n_rounds)
        if verbose:
            from tqdm import tqdm
            epoch_iterator = tqdm(epoch_iterator, desc='Training epochs', leave=True)
        
        for i in epoch_iterator:
            # calculate MIA score
            if self.MIA is not None:
                mia_score = self.MIA(self.model, retain_dataloader, forget_dataloader, val_dataloader)
                metrics['mia'].append(mia_score)
            
            if self.js_div_func is not None:
                # JS_divergence(self, probs_u, probs_c, log_base: float = 2.0)
                probs_u = self.model.inference(retain_dataloader.dataset.X.to(self.device))['probabilities']
                probs_c = self.retrain_model.inference(retain_dataloader.dataset.X.to(self.device))['probabilities']
                js_div = self.js_div_func(probs_u, probs_c)
                metrics['js_div']['retain'].append(js_div)
                probs_u = self.model.inference(forget_dataloader.dataset.X.to(self.device))['probabilities']
                probs_c = self.retrain_model.inference(forget_dataloader.dataset.X.to(self.device))['probabilities']
                js_div = self.js_div_func(probs_u, probs_c)
                metrics['js_div']['forget'].append(js_div)
                probs_u = self.model.inference(val_dataloader.dataset.X.to(self.device))['probabilities']
                probs_c = self.retrain_model.inference(val_dataloader.dataset.X.to(self.device))['probabilities']
                js_div = self.js_div_func(probs_u, probs_c)
                metrics['js_div']['val'].append(js_div)

            # calculate retain and forget errors
            retain_err = self.calculate_error(self.model, retain_dataloader)
            forget_err = self.calculate_error(self.model, forget_dataloader)
            metrics['retain']['loss'].append(retain_err.item())
            metrics['forget']['loss'].append(forget_err.item())

            # calculate val error
            val_err = self.calculate_error(self.model, val_dataloader)
            metrics['val']['loss'].append(val_err.item())

            # calculate retain and forget accuracies
            retain_acc = self.calculate_accuracy(self.model, retain_dataloader)
            forget_acc = self.calculate_accuracy(self.model, forget_dataloader)
            metrics['retain']['acc'].append(retain_acc)
            metrics['forget']['acc'].append(forget_acc)

            # calculate val accuracy
            val_acc = self.calculate_accuracy(self.model, val_dataloader)
            metrics['val']['acc'].append(val_acc)    

            # Create batch iterators with tqdm if verbose
            forget_iterator = forget_dataloader
            retain_iterator = retain_dataloader
            if verbose:
                forget_iterator = tqdm(forget_iterator, desc='Max step (forget)', leave=False)
                retain_iterator = tqdm(retain_iterator, desc='Min step (retain)', leave=False)

            # max step
            for batch in forget_iterator:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                self.max_step(x, y)
            
            # min step    
            for batch in retain_iterator:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                self.min_step(x, y)

            err = self.calculate_error(self.model, forget_dataloader)
            forget_errors.append(err.item())
            os.makedirs('weights/tmp', exist_ok=True)
            torch.save(self.model.state_dict(), f'weights/tmp/scrub+r_epoch{i+1}.pth')

            # Update progress bar with current metrics if verbose
            if verbose:
                epoch_iterator.set_postfix({
                    'retain_acc': f"{retain_acc:.4f}",
                    'forget_acc': f"{forget_acc:.4f}",
                    'val_acc': f"{val_acc:.4f}"
                })

        err_threshold = self.calculate_error(self.model, validate_err_dataloader)
        
        # choose best epoch as the latest one where the error was below threshold
        best_epoch = torch.where(torch.tensor(forget_errors, device=self.device) < err_threshold)[0]
        if len(best_epoch):
            best_epoch = (best_epoch[-1] + 1).item()
        else: # err_threshold lower than all elements in forget_errors
            best_epoch = 1

        os.makedirs('weights/scrub+r', exist_ok=True)
        best_ckpt = f'weights/tmp/scrub+r_epoch{best_epoch}.pth'
        sd = torch.load(best_ckpt, weights_only=False)

        self.model.load_state_dict(sd)
        self.model.eval()

        if remove_weights:
            files = os.listdir('weights/tmp')
            for file in files:
                os.remove(f"weights/tmp/{file}")

        return metrics 