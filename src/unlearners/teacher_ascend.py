import copy
import torch
import torch.optim as optim
import pdb

"""
"Bad teacher" loss for the maximization step? 
    - Turn it into minimizing KL divergence between bad teacher and unlearned model.
"""

class TeacherAscender:
    def __init__(self, model, n_epochs: int, device: str = 'cpu') -> None:
        self.model = model
        self.n_epochs = n_epochs
        self._lambda = 4
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
            x, y = batch[0], batch[1]
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
            reg_term += ((FIM_original[name] * (param - original_sd[name]))**2).sum()
        return reg_term
    
    def schedule_reg_term_calculation(self):
        pass

    def calculate_ascend_term(self, model_out, y):
        log_probs = (model_out['probabilities'] + 1e-8).log()
        entropy = -(model_out['probabilities'] * log_probs).sum(dim=-1).mean()
        model_loss = self.model.loss(model_out, y)
        return -(0.5 * entropy + 0.5 * model_loss)
        # return -model_loss
        # return -entropy


    def __call__(self, retain_loader, forget_loader):
        """
        Performs gradient ascend on forget set labels while regularizing with ∑ F (p_u - p_o)^2

        Options:
            - Maximize entropy of label distribution on forget data --> probably better for OOD points
            - Maximize cross entropy between prediction and y on forget data --> works better for data poisoning.
            - Both?
        """
        # calculate FIM for original model on retain set
        original_sd = copy.deepcopy(self.model.state_dict())
        FIM_original = self.calculate_FIM(retain_loader)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        for _ in range(self.n_epochs):
            for batch in forget_loader:
                optimizer.zero_grad()
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                model_out = self.model(x)
        
                reg_term = self.calculate_reg_term(FIM_original, original_sd)
                ascend_term = self.calculate_ascend_term(model_out, y)
                loss = ascend_term + self._lambda / 2 * reg_term
                loss.backward()
                optimizer.step()
                



    