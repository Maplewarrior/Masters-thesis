import copy
import torch
import numpy as np
import torch.optim as optim
import pdb

class OrthogonalGradients:
    def __init__(self, model, n_epochs: int, _lambda: float, device: str = 'cpu') -> None:
        self.model = model
        self.n_epochs = n_epochs
        self._lambda = _lambda
        self.device = device

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
        
    def parameters_to_vector(self, parameters: dict):
        vec = [param.view(-1) for _, param in parameters.items()]
        vec = torch.cat(vec)
        return vec
    
    def vector_to_parameters(self, vec: torch.tensor, parameters: dict):
        offset = 0
        for name, param in parameters.items():
            param_size = param.shape
            try:
                if param.ndim == 2:
                    parameters[name] = vec[offset : offset + param_size[0] * param_size[1]].view(param_size[0], param_size[1])
                    offset += param_size[0] * param_size[1]
                
                elif param.ndim == 1:
                    parameters[name] = vec[offset : offset + param_size[0]]
                    offset += param_size[0]
                else:
                    print(f'Parameters of dimension: {param.ndim} are not supported.')
            except Exception as e:
                pdb.set_trace()
                print(e)

        return parameters

    def project(self, u, v):
        v_sq_norm = v.norm(p=2)**2
        if v_sq_norm < 1e-8:
            return torch.zeros_like(v)
        return (u @ v) / (v_sq_norm) * v

    def collect_retain_grads(self, retain_loader):
        """
        strategy: one of ['single', 'subset', 'all']
        """

        class_gradients = {k: 0 for k in self.model.state_dict().keys()}
        optimizer = optim.SGD(self.model.parameters())
        self.model.eval()
        for batch in retain_loader:
            optimizer.zero_grad()
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
        
            model_out = self.model(x)
            neg_loss = -self.model.loss(model_out, y)
            neg_loss.backward()

            # collect gradients
            for name, param in self.model.named_parameters():
                class_gradients[name] += param.grad.data.clone()
        # account for batched inference
        for k in class_gradients.keys():
            class_gradients[k] = class_gradients[k] / len(retain_loader)
        
        # convert gradient as parameters to a single vector
        class_gradients = self.parameters_to_vector(class_gradients)
            
        del optimizer

        return class_gradients

    def construct_retain_basis(self, retain_grads):
        classes = list(retain_grads.keys())
        n_classes = len(classes)
        grad_dim = len(retain_grads[classes[0]])

        retain_basis = torch.zeros((n_classes, grad_dim), 
                                   device=self.device, 
                                   dtype=retain_grads[classes[0]].dtype)

        subtract_term = torch.zeros_like(retain_grads[classes[0]])
        for i, label in enumerate(classes):
            retain_basis[i, :] = retain_grads[label] - subtract_term
            if i == 0:
                subtract_term -= retain_grads[label]
            else:
                subtract_term -= self.project(retain_grads[label], retain_basis[i-1, :])
        
        return retain_basis

    def project_forget_gradient(self, forget_grad: torch.tensor, retain_grad: torch.tensor):
        forget_grad = forget_grad - self.project(forget_grad, retain_grad)
        return forget_grad
    
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
    
    def __call__(self, retain_loader, forget_loader, val_loader = None):
        
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        retain_grad = self.collect_retain_grads(retain_loader)
        
        forget_grad_norms = []
        metrics = {'retain': {'acc': [], 'loss': []},
                   'forget': {'acc': [], 'loss': []},
                   'val': {'acc': [], 'loss': []}
                  }
        
        for _ in range(self.n_epochs):
            norms = []
            #### Gradient ascent
            for batch in forget_loader:
                optimizer.zero_grad()
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)

                model_out = self.model(x)
                loss = -self.calculate_fit_term(model_out, y)
                loss.backward()
                
                grad = {k: 0 for k in self.model.state_dict().keys()}
                for name, param in self.model.named_parameters():
                    grad[name] += param.grad.data.clone()
                
                forget_grad = self.parameters_to_vector(grad)
                forget_grad = self.project_forget_gradient(forget_grad, retain_grad)
                norms.append(forget_grad.norm(p=2).item())
                updated_forget_grad = self.vector_to_parameters(forget_grad, grad)
                
                for name, param in self.model.named_parameters():
                    param.grad = updated_forget_grad[name]

                optimizer.step()
            
            forget_grad_norms.append(np.mean(norms))
            forget_loss, forget_acc = self.eval(forget_loader)
            metrics['forget']['acc'].append(forget_acc)
            metrics['forget']['loss'].append(forget_loss)

            retain_loss, retain_acc = self.eval(retain_loader)
            metrics['retain']['acc'].append(retain_acc)
            metrics['retain']['loss'].append(retain_loss)

            val_loss, val_acc = self.eval(val_loader)
            metrics['val']['acc'].append(val_acc)
            metrics['val']['loss'].append(val_loss)
        
        pdb.set_trace()

        return metrics