import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data_utils.synthetic_data import SyntheticDataset
from src.evaluation.membership_inference_attack import MIA

"""
scrub_r models = []
retrained_models = []

for model in scrub_r models:
    best = 0
    for retrained_model in retrained_models:
        scrub_r preds == retrained preds
        update best
        

     
"""

class UnlearningEvaluator:
    def __init__(self, device: str = 'cpu') -> None:
        
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.metricname2function = {'accuracy': self.calculate_accuracy,
                                    'Hamming PD': self.hamming_PD,
                                    'Error corrected HPD': self.hamming_PD_error_corrected,
                                    'min max normalized HPD': self.min_max_norm_disagreement,
                                    'avg norm prediction difference': self.avg_norm_pred_diff,
                                    'KL divergence': self.KL_divergence,
                                    'JS divergence': self.JS_divergence,}
        self.device = device
        
    def get_model_logits(self, 
                         model,
                         dataloader: DataLoader[SyntheticDataset]):
        logits = []
        ys = []
        for batch in dataloader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            logits.append(model.inference(x)['logits'])
    
            # Make sure y is one-hot encoded. Some models do not use one-hot encoding.
            if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'onehot_labels') and not dataloader.dataset.onehot_labels:
                y = dataloader.dataset.onehot_encode_labels(y, dataloader.dataset.n_classes).to(self.device)
            ys.append(y)

        logits = torch.cat(logits)
        ys = torch.cat(ys)
        return logits, ys
    
    def get_model_probs(self, 
                         model,
                         dataloader: DataLoader[SyntheticDataset]):
        probs = []
        ys = []
        for batch in dataloader:
            # import time
            # start = time.time()
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            probs.append(model.inference(x)['probabilities'])
            # end = time.time()
            # elapsed = end - start
            # Make sure y is one-hot encoded. Some models do not use one-hot encoding.
            if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'onehot_labels') and not dataloader.dataset.onehot_labels:
                y = dataloader.dataset.onehot_encode_labels(y, dataloader.dataset.n_classes).to(self.device)
            ys.append(y)

        probs = torch.cat(probs)
        ys = torch.cat(ys)
        return probs, ys

    def evaluate(self,
                 unlearned_model, 
                 comparison_model, 
                 metrics: list[str], 
                 dataloader: DataLoader[SyntheticDataset]):
        result = {}
        probs_u, y_values = self.get_model_probs(unlearned_model, dataloader)
        probs_c, _ = self.get_model_probs(comparison_model)

        for metric in metrics:
            if metric == 'accuracy':
                result[f'{metric}_unlearned_model'] = self.calculate_accuracy(probs_u, y_values)
                result[f'{metric}_comparison_model'] = self.calculate_accuracy(probs_c, y_values)

            elif metric in ['min max normalized HPD', 'Error corrected HPD']:
                result[metric] = self.metricname2function[metric](probs_u, probs_c, y_values)
            else:
                result[metric] = self.metricname2function[metric](probs_u, probs_c)
        
        return result
    
    def evaluate_MIA(self, model, retain_dataloader, forget_dataloader, val_dataloader):
        """
        Returns the membership inference attack probability of the forget data.
        The MIA model is a logistic regression trained for binary classification. It's train dataset is the following:
            - X: The entropy of the target model's outputs.
            - y: 1 if the datapoint belongs to retain 0 if it belongs to test.
        """
        mia_model = MIA()
        return mia_model(model, retain_dataloader, forget_dataloader, val_dataloader)

    def calculate_accuracy(self, probs, y):
        return ((probs.argmax(dim=1) == y.argmax(dim=1)).sum() / y.size(0)).item()
    
    def hamming_PD(self, probs_u, probs_c):
        estimate = (probs_u.argmax(dim=1) != probs_c.argmax(dim=1)).sum() / probs_u.size(0)
        return estimate.item()
    
    def hamming_PD_error_corrected(self, probs_u, probs_c, y):
        error_u = (probs_u.argmax(dim=1) != y.argmax(dim=1)).sum() / y.size(0)
        return self.hamming_PD(probs_u, probs_c) / error_u
    
    def min_max_norm_disagreement(self, probs_u, probs_c, y):
        error_u = (probs_u.argmax(dim=1) != y.argmax(dim=1)).sum() / y.size(0)
        error_c = (probs_c.argmax(dim=1) != y.argmax(dim=1)).sum() / y.size(0)
        min_disagreement = torch.abs(error_u - error_c)
        max_disagreement = torch.min(error_u + error_c, torch.tensor(1))
        estimate = (self.hamming_PD(probs_u, probs_c) - min_disagreement) / ((max_disagreement - min_disagreement) + 1e-6)
        return estimate.item()
    
    def avg_norm_pred_diff(self, probs_u, probs_c):
        return (torch.linalg.norm(probs_u - probs_c, dim=1, ord=1).sum() / (2 * probs_u.size(0))).item()

    def KL_divergence(self, probs_u, probs_c):
        """
        @param preds_u: The predictions of the unlearned model.
        @param preds_c: The predictions of the comparison model. This is either the original or retrained model.
        
        If we denote P = preds_c and Q = preds_u the formula then becomes: KL[P || Q] = Σ P(x) * log(P(x) / Q(x))
        """
        # NOTE The below 3 implementations have been tested and appear to be equivalent...
        # kl_ls = F.kl_div(self.log_softmax(preds_u), self.softmax(preds_c), reduction='batchmean').item()
        # kl = F.kl_div(torch.log(self.softmax(preds_u)), self.softmax(preds_c), reduction='batchmean').item()
        # kl_manual = (self.softmax(preds_c) * torch.log(self.softmax(preds_c) / self.softmax(preds_u))).sum(dim=-1).mean()
        
        log_probs_u = torch.clamp(probs_u, min=1e-8).log()
        probs_c = torch.clamp(probs_c, min=1e-8) # avoid numeric issues
        return F.kl_div(log_probs_u, probs_c, reduction='batchmean').item()

    def JS_divergence(self, probs_u, probs_c, log_base: float = 2.0):
        tolerance = 1e-6
        log_probs_m = (torch.clamp((probs_u + probs_c) / 2, min=1e-8)).log()

        estimate_u = F.kl_div(log_probs_m, probs_u, reduction='batchmean') # self.KL_divergence(preds_m, preds_u)
        estimate_c = F.kl_div(log_probs_m, probs_c, reduction='batchmean')
        estimate = (estimate_u + estimate_c) / 2

        if log_base == 2.0:
            estimate = (estimate/torch.log(torch.tensor(2.)))
        
        if estimate.item() < 0 - tolerance or estimate.item() > 1 + tolerance:
            print("JS Divergence out of bounds!")
        
        return estimate.item()