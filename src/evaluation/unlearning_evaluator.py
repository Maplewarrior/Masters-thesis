import torch
import torch.nn.functional as F
import pdb
class UnlearningEvaluator:
    def __init__(self) -> None:
        
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def evaluate_performance(self, model, dataloader):
        model.eval()
        acc = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                out = model(x)
                acc += ((out['probabilities'].argmax(dim=1)) == y.argmax(dim=1)).sum().item()
        
        acc = acc / len(dataloader.dataset)

        return {'accuracy': acc}
    
    def functional_equivalence(self, 
                               original_model,
                               unlearned_model,
                               metric: str,
                               dataloader
                               ):
        

        original_model.eval()
        unlearned_model.eval()
        preds_o = []
        preds_u = []
        # make predictions
        with torch.no_grad():
            for (x, y) in dataloader:
                preds_o.append(original_model(x)['logits'])
                preds_u.append(unlearned_model(x)['logits'])
            preds_o = torch.cat(preds_o)
            preds_u = torch.cat(preds_u)

            # calculate equivalence        
            if metric == 'KL': # use KL divergence
                estimate = F.kl_div(self.log_softmax(preds_u), self.softmax(preds_o), reduction='batchmean')
            
            elif metric == 'JS-divergence': # use Jensen-Shannon Divergence
                preds_m = (preds_o + preds_u) / 2
                estimate_o = F.kl_div(self.log_softmax(preds_o), self.softmax(preds_m), reduction='batchmean')
                estimate_u = F.kl_div(self.log_softmax(preds_u), self.softmax(preds_m), reduction='batchmean')
                estimate = (estimate_o + estimate_u) / 2

                ### sanity check:
                # estimate_o2 = F.kl_div(self.log_softmax(preds_o), self.softmax(preds_m), reduction='none')
                # estimate_u2 = F.kl_div(self.log_softmax(preds_u), self.softmax(preds_m), reduction='none')
                # est = (estimate_o2 + estimate_u2).sum() / (2 * len(dataloader.dataset))

            else:
                raise NotImplementedError(f"The specified metric {metric} is not supported.")


        return estimate.item()
    