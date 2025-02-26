import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data_utils.synthetic_data import SyntheticDataset

class UnlearningEvaluator:
    def __init__(self) -> None:
        
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.metricname2function = {'accuracy': self.calculate_accuracy,
                                    'Hamming PD': self.hamming_PD,
                                    'Error corrected HPD': self.hamming_PD_error_corrected,
                                    'min max normalized HPD': self.min_max_norm_disagreement,
                                    'avg norm prediction difference': self.avg_norm_pred_diff,
                                    'KL divergence': self.KL_divergence,
                                    'JS divergence': self.JS_divergence,}

    def get_model_predictions(self, unlearned_model, 
                                    comparison_model,
                                    dataloader: DataLoader[SyntheticDataset]):
        preds_u = []
        preds_c = []
        ys = []
        if unlearned_model != None and comparison_model != None:
            for (x, y, *extra) in dataloader:                
                preds_u.append(unlearned_model.inference(x)['logits'])
                preds_c.append(comparison_model.inference(x)['logits'])

                # Make sure y is one-hot encoded. Some models do not use one-hot encoding.
                if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'onehot_labels') and not dataloader.dataset.onehot_labels:
                    y = dataloader.dataset.onehot_encode_labels(y, dataloader.dataset.n_classes)
                ys.append(y)

            preds_u = torch.cat(preds_u)
            preds_c = torch.cat(preds_c)
            ys = torch.cat(ys)
        return preds_u, preds_c, ys

    def evaluate(self, 
                 unlearned_model, 
                 comparison_model, 
                 metrics: list[str], 
                 dataloader: DataLoader[SyntheticDataset]):
        result = {}
        preds_u, preds_c, y_values = self.get_model_predictions(unlearned_model, comparison_model, dataloader)

        for metric in metrics:
            if metric == 'accuracy':
                result[f'{metric}_unlearned_model'] = self.calculate_accuracy(preds_u, y_values)
                result[f'{metric}_comparison_model'] = self.calculate_accuracy(preds_c, y_values)

            elif metric in ['min max normalized HPD', 'Error corrected HPD']:
                result[metric] = self.metricname2function[metric](preds_u, preds_c, y_values)
            else:
                result[metric] = self.metricname2function[metric](preds_u, preds_c)
        

        return result
    
    
    def calculate_accuracy(self, preds, y):
        return ((preds.argmax(dim=1) == y.argmax(dim=1)).sum() / y.size(0)).item()
    
    def hamming_PD(self, preds_u, preds_c):
        estimate = (preds_u.argmax(dim=1) != preds_c.argmax(dim=1)).sum() / preds_u.size(0)
        return estimate.item()
    
    def hamming_PD_error_corrected(self, preds_u, preds_c, y):
        error_u = (preds_u.argmax(dim=1) != y.argmax(dim=1)).sum() / y.size(0)
        return self.hamming_PD(preds_u, preds_c) / error_u
    
    def min_max_norm_disagreement(self, preds_u, preds_c, y):
        error_u = (preds_u.argmax(dim=1) != y.argmax(dim=1)).sum() / y.size(0)
        error_c = (preds_c.argmax(dim=1) != y.argmax(dim=1)).sum() / y.size(0)
        min_disagreement = torch.abs(error_u - error_c)
        max_disagreement = torch.min(error_u + error_c, torch.tensor(1))
        estimate = (self.hamming_PD(preds_u, preds_c) - min_disagreement) / ((max_disagreement - min_disagreement) + 1e-6)
        return estimate.item()
    
    def avg_norm_pred_diff(self, preds_u, preds_c):
        p_u = self.softmax(preds_u)
        p_c = self.softmax(preds_c)
        return (torch.linalg.norm(p_u - p_c, dim=1, ord=1).sum() / (2 * preds_u.size(0))).item()

    """
    This introduces many errors (6 falied)
    """
    def KL_divergence(self, preds_u, preds_c):
        """
        @param preds_u: The predictions of the unlearned model.
        @param preds_c: The predictions of the comparison model. This is either the original or retrained model.
        
        If we denote P = preds_c and Q = preds_u the formula then becomes: KL[P || Q] = Σ P(x) * log(P(x) / Q(x))
        """
        # NOTE The below 3 implementations have been tested and appear to be equivalent...
        # kl_ls = F.kl_div(self.log_softmax(preds_u), self.softmax(preds_c), reduction='batchmean').item()
        # kl = F.kl_div(torch.log(self.softmax(preds_u)), self.softmax(preds_c), reduction='batchmean').item()
        # kl_manual = (self.softmax(preds_c) * torch.log(self.softmax(preds_c) / self.softmax(preds_u))).sum(dim=-1).mean()
        
        log_probs_u = torch.clamp(self.softmax(preds_u), min=1e-8).log()
        probs_c = torch.clamp(self.softmax(preds_c), min=1e-8) # avoid numeric issues
        return F.kl_div(log_probs_u, probs_c, reduction='batchmean').item()

    
    def JS_divergence(self, preds_u, preds_c, log_base: float = 2.0):
        tolerance = 1e-6
        probs_u = self.softmax(preds_u)
        probs_c = self.softmax(preds_c)
        log_probs_m = (torch.clamp((probs_u + probs_c) / 2, min=1e-8)).log()

        estimate_u = F.kl_div(log_probs_m, probs_u, reduction='batchmean') # self.KL_divergence(preds_m, preds_u)
        estimate_c = F.kl_div(log_probs_m, probs_c, reduction='batchmean')
        estimate = (estimate_u + estimate_c) / 2

        if log_base == 2.0:
            estimate = (estimate/torch.log(torch.tensor(2.)))
        
        if estimate.item() < 0 - tolerance or estimate.item() > 1 + tolerance:
            print("JS Divergence out of bounds!")
            # pdb.set_trace()
        
        return estimate.item()