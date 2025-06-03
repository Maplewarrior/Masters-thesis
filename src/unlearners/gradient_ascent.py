import numpy as np
import torch
import torch.optim as optim

class GradientAscent:
    def __init__(self, model, n_epochs, device: str) -> None:
        self.model = model
        self.n_epochs = n_epochs
        self.device = device

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
    
    def __call__(self, retain_loader, forget_loader, val_loader):
        
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        metrics = {'retain': {'acc': [], 'loss': []},
                   'forget': {'acc': [], 'loss': []},
                   'val': {'acc': [], 'loss': []}
                  }
        for _ in range(self.n_epochs):

            #### Gradient ascent
            for batch in forget_loader:
                optimizer.zero_grad()

                x = batch[0].to(self.device)
                y = batch[1].to(self.device)

                model_out = self.model(x)

                loss = -self.model.loss(model_out, y) # negate loss to ascend
                loss.backward()
                optimizer.step()
            
            forget_loss, forget_acc = self.eval(forget_loader)
            metrics['forget']['acc'].append(forget_acc)
            metrics['forget']['loss'].append(forget_loss)

            retain_loss, retain_acc = self.eval(retain_loader)
            metrics['retain']['acc'].append(retain_acc)
            metrics['retain']['loss'].append(retain_loss)

            val_loss, val_acc = self.eval(val_loader)
            metrics['val']['acc'].append(val_acc)
            metrics['val']['loss'].append(val_loss)
        
        return metrics