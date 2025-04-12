from typing import Any
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

class MIA:
    def __init__(self) -> None:
        pass

    def entropy(self, probs: torch.tensor):
        """
        Computes the entropy of a 2d tensor containing samples from a discrete probability distribution
        args:
            probs: torch tensor of dimension N x C.
        
        returns: A torch tensor of dimension N.
        """
        return -(probs * probs.log()).sum(dim=-1)
    
    def collect_probs(self, model, dataloader):
        probs = []
        for batch in dataloader:
            x = batch[0]
            out = model.inference(x)
            probs.append(out['probabilities'])
        return torch.cat(probs)

    def construct_MIA_dataset(self, model, retain_loader, forget_loader, val_loader):
        retain_probs = self.collect_probs(model, retain_loader)
        val_probs = self.collect_probs(model, val_loader)
        forget_probs = self.collect_probs(model, forget_loader)
        

        X_r = torch.cat([self.entropy(retain_probs), self.entropy(val_probs)]).view(-1, 1).cpu().numpy()
        y_r = torch.cat([torch.ones((retain_probs.size(0))), torch.zeros(val_probs.size(0))]).cpu().numpy()
        X_f = self.entropy(forget_probs).view(-1, 1).cpu().numpy()
        y_f = torch.ones(forget_probs.size(0)).cpu().numpy()

        return X_r, y_r, X_f, y_f

    def __call__(self, model, retain_loader, forget_loader, val_loader):
        X_r, y_r, X_f, y_f = self.construct_MIA_dataset(model, retain_loader, forget_loader, val_loader)
        mia_model = LogisticRegression(class_weight='balanced', solver='lbfgs')
        mia_model.fit(X_r, y_r)
        mia_probs = mia_model.predict(X_f)
        return mia_probs.mean()        
