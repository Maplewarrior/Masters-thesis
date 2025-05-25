import copy
import torch
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
            reg_term += ((FIM_original[name] * (param - original_sd[name]))**2).sum()
        return reg_term

    def calculate_fit_term(self, model_out, y):
        # model_loss = self.model.loss(model_out, y)
        # return model_loss
        log_probs = (model_out['probabilities'] + 1e-8).log()
        entropy = -(model_out['probabilities'] * log_probs).sum(dim=-1).mean()
        return entropy
        

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
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        FIM_forget = self.calculate_FIM(forget_loader) # used for descend
        FIM_original = self.calculate_FIM(retain_loader) # used for ascend
        for _ in range(self.n_epochs):

            #### Gradient descent  
            # for batch in retain_loader:
            #     optimizer.zero_grad()
            #     x = batch[0].to(self.device)
            #     y = batch[1].to(self.device)
            #     model_out = self.model(x)
            #     reg_term = self.calculate_reg_term(FIM_forget, original_sd)
            #     fit_term = self.calculate_fit_term(model_out, y)
            #     loss = fit_term - self._lambda / 2 * reg_term # minimize CE, maximize reg term
            #     loss.backward()
            #     optimizer.step()
            
            #### Gradient ascent
            for batch in forget_loader:
                optimizer.zero_grad()
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                model_out = self.model(x)
        
                reg_term = self.calculate_reg_term(FIM_original, original_sd)
                fit_term = self.calculate_fit_term(model_out, y)
                loss = -fit_term + self._lambda / 2 * reg_term # minimize CE, maximize reg term
                # loss = -fit_term + self._lambda / 2 * reg_term # # maximize CE, minimize reg term
                loss.backward()
                optimizer.step()
                



    