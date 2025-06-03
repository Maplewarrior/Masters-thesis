import copy
import torch
import numpy as np
import torch.optim as optim
import pdb

"""
"Bad teacher" loss for the maximization step? 
    - Turn it into minimizing KL divergence between bad teacher and unlearned model.
"""

class TeacherAscender:
    def __init__(self, model, n_epochs: int, _lambda: float, device: str = 'cpu') -> None:
        self.model = model
        self.n_epochs = n_epochs
        self._lambda = _lambda
        self.device = device

    def calculate_FIM(self, dataloader) -> dict:
        """
        A function that calculates the diagonal of the Fisher Information of a model for a specific dataset.

        @param dataloader: An iterable dataloader for which the FIM diagonal should be calculated.
        returns: A dictionary where keys are the names of parameters and values are the FIM of that parameter
        """
        FIM = {k: 0 for k in self.model.state_dict().keys()}
        # define optimizer to allow gradient computation
        optimizer = optim.SGD(self.model.parameters())
        # turn of dropout if applicable
        self.model.eval()
        
        for i, batch in enumerate(dataloader):
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            optimizer.zero_grad()
            # forward pass
            model_out = self.model(x)
            # calculate loss
            loss = self.model.loss(model_out, y)
            # calculate gradients
            loss.backward()
            for i, (name, param) in enumerate(self.model.named_parameters()):
                FIM[name] += param.grad.data.clone().pow(2) #optimizer.param_groups[0]['params'][i].grad.pow(2)  
        # account for batched inference
        for k in FIM.keys():
            FIM[k] = FIM[k] / len(dataloader)
        
        return FIM

    def calculate_reg_term(self, FIM_original, original_sd):
        reg_term = 0
        for name, param in self.model.named_parameters():
            reg_term += (FIM_original[name] * (param - original_sd[name])**2).sum()
        return reg_term

    def calculate_fit_term(self, model_out, y):
        model_loss = self.model.loss(model_out, y)
        return model_loss
        # log_probs = (model_out['probabilities'] + 1e-8).log()
        # entropy = -(model_out['probabilities'] * log_probs).sum(dim=-1).mean()
        # return entropy
    
    def calculate_entropy(self, model_out):
        log_probs = (model_out['probabilities'] + 1e-8).log()
        entropy = -(model_out['probabilities'] * log_probs).sum(dim=-1).mean()
        return entropy

    def calculate_FIM_ratio(self, FIM_retain, FIM_forget):

        def nanmax(tensor):
            min_value = torch.finfo(tensor.dtype).min
            return tensor.nan_to_num(min_value).max()
        
        FIM_ratio = {k: 0 for k in FIM_retain.keys()}
        for k in FIM_retain.keys():
            ratio = FIM_retain[k] / FIM_forget[k]
            FIM_ratio[k] = torch.nan_to_num(ratio, nan=0.0, posinf=nanmax(ratio)).clamp(0., 1000)
        return FIM_ratio

    def eval(self, dataloader):
        self.model.eval()
        losses = []
        acc = 0

        with torch.no_grad():

            for batch in dataloader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)

                out = self.model(x)
                loss = self.model.loss(out, y)

                # update metrics
                losses.append(loss.item())
                acc += (out['probabilities'].argmax(dim=1) == y.argmax(dim=1)).sum().item()

        acc = acc / len(dataloader.dataset)
        self.model.train()
        return np.mean(losses), acc
    
    def calculate_entanglement_score(self, retain_loader, forget_loader):
        retain_embeddings = []
        forget_embeddings = []
        N_r, N_f = len(retain_loader), len(forget_loader)

        for batch in retain_loader:
            x = batch[0].to(self.device)
            emb = self.model.inference(x, stop_idx=-1)['logits']
            retain_embeddings.append(emb)
        
        for batch in forget_loader:
            x = batch[0].to(self.device)
            emb = self.model.inference(x, stop_idx=-1)['logits']
            forget_embeddings.append(emb)
        
        retain_embeddings = torch.cat(retain_embeddings)
        forget_embeddings = torch.cat(forget_embeddings)
        
        

        mu_retain = retain_embeddings.mean(dim=0)
        mu_forget = forget_embeddings.mean(dim=0)
        mu_total = mu_retain * (N_r / (N_r + N_f)) + mu_forget * (N_f / (N_r + N_f))

        
        
        ES_enumerator = ((retain_embeddings - mu_retain).norm(dim=-1, p=2).pow(2).mean() + (forget_embeddings - mu_forget).norm(dim=-1, p=2).pow(2).mean())
        ES_denominator = 0.5 * ((mu_retain - mu_total).norm(p=2).pow(2) + (mu_forget - mu_total).norm(p=2).pow(2))
        ES = ES_enumerator / ES_denominator

        self.model.train()

        return ES

    def MDL(self, retain_loader):
        self.model.eval()
        mdl =  0
        with torch.no_grad():
            for batch in retain_loader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)

                model_out = self.model(x)
                log_likelihood = model_out['probabilities'].gather(dim=1, index=y.argmax(dim=-1).unsqueeze(1)).squeeze(1).log().mean()
                entropy = self.calculate_entropy(model_out)

                mdl += -log_likelihood + entropy
        
        mdl = mdl / len(retain_loader.dataset)
        return mdl

    def __call__(self, retain_loader, forget_loader, val_loader=None, eval: bool = False):
        """
        Performs gradient ascend on forget set labels while regularizing with ∑ F (p_u - p_o)^2

        Options:
            - Maximize entropy of label distribution on forget data --> probably better for OOD points
            - Maximize cross entropy between prediction and y on forget data --> works better for data poisoning.
            - Both?
        """
        metrics = {'retain': {'acc': [], 'loss': []},
                   'forget': {'acc': [], 'loss': []},
                   'val': {'acc': [], 'loss': []}
                  }
        
        # calculate FIM for original model on retain set
        original_sd = copy.deepcopy(self.model.state_dict())
        optimizer = optim.Adam(self.model.parameters(), lr=1e-2)

        FIM_forget = self.calculate_FIM(forget_loader) # used for descend
        FIM_original = self.calculate_FIM(retain_loader) # used for ascend
        FIM_ratio = self.calculate_FIM_ratio(FIM_original, FIM_forget)

        # mdls = []
        # entanglement_scores = []
        # retain_terms = []
        # forget_terms = []

        for _ in range(self.n_epochs):
            
            # rt = []
            # ft = []
            # entanglement_scores.append(self.calculate_entanglement_score(retain_loader, forget_loader))
            
            #### Gradient ascent
            for batch in forget_loader:
                optimizer.zero_grad()

                x_f = batch[0].to(self.device)
                y_f = batch[1].to(self.device)

                batch_retain = next(iter(retain_loader))
                x_r = batch_retain[0].to(self.device)
                y_r = batch_retain[1].to(self.device)

                out_f = self.model(x_f)
                out_r = self.model(x_r)

                retain_term = self.calculate_fit_term(out_r, y_r)
                forget_term = self.calculate_entropy(out_f)
                reg_term = self.calculate_reg_term(FIM_ratio, original_sd)
                # loss = -forget_term + self._lambda / 2 * reg_term # minimize CE, maximize reg term
                loss = -forget_term + retain_term + self._lambda / 2 * reg_term # minimize CE, maximize reg term

                loss.backward()
                optimizer.step()

            #     rt.append(retain_term)
            #     ft.append(forget_term)

            # retain_terms.append(torch.tensor(rt).mean())
            # forget_terms.append(torch.tensor(ft).mean())

            # mdl = self.MDL(retain_loader)
            # mdls.append(mdl)
            if eval:
                forget_loss, forget_acc = self.eval(forget_loader)
                metrics['forget']['acc'].append(forget_acc)
                metrics['forget']['loss'].append(forget_loss)

                retain_loss, retain_acc = self.eval(retain_loader)
                metrics['retain']['acc'].append(retain_acc)
                metrics['retain']['loss'].append(retain_loss)

                val_loss, val_acc = self.eval(val_loader)
                metrics['val']['acc'].append(val_acc)
                metrics['val']['loss'].append(val_loss)
            
        if eval:
            return metrics
            
