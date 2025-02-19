import torch
import torch.nn.functional as F
import pdb
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
                                    dataloader):
        preds_u = []
        preds_c = []
        ys = []
        if unlearned_model != None and comparison_model != None:
            unlearned_model.eval()
            comparison_model.eval()
            with torch.no_grad():
                for (x, y) in dataloader:
                    preds_u.append(unlearned_model.predict(x)['logits'])
                    preds_c.append(comparison_model.predict(x)['logits'])
                    ys.append(y)
                preds_u = torch.cat(preds_u)
                preds_c = torch.cat(preds_c)
                ys = torch.cat(ys)
                        
        return preds_u, preds_c , ys

    def evaluate(self, 
                 unlearned_model, 
                 comparison_model, 
                 metrics: list[str], 
                 dataloader):
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
        return (torch.linalg.norm(self.softmax(preds_u) - self.softmax(preds_c), dim=1, ord=2).sum() / (preds_u.size(0) * preds_u.size(1))).item()

    def KL_divergence(self, preds_u, preds_c):
        return F.kl_div(self.log_softmax(preds_u), self.softmax(preds_c), reduction='batchmean').item()
    
    def JS_divergence(self, preds_u, preds_c):
        preds_m = (preds_u + preds_c) / 2
        estimate_u = F.kl_div(self.log_softmax(preds_u), self.softmax(preds_m), reduction='batchmean')
        estimate_c = F.kl_div(self.log_softmax(preds_c), self.softmax(preds_m), reduction='batchmean')
        estimate = (estimate_u + estimate_c) / 2
        return estimate.item()

    
