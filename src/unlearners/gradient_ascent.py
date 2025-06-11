import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

class GradientAscent:
    def __init__(self, model, n_epochs, device: str, MIA: callable = None) -> None:
        self.model = model
        self.n_epochs = n_epochs
        self.device = device
        self.MIA = MIA

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
    
    def __call__(self, retain_loader, forget_loader, val_loader, verbose: bool = True):
        
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        metrics = {'retain': {'acc': [], 'loss': []},
                   'forget': {'acc': [], 'loss': []},
                   'val': {'acc': [], 'loss': []}
                  }
        if self.MIA is not None:
            metrics['mia'] = []

        # Create epoch iterator with tqdm if verbose
        epoch_iterator = tqdm(range(self.n_epochs), desc='Epochs', disable=not verbose)

        for _ in epoch_iterator:
            if self.MIA is not None:
                mia_prob = self.MIA(self.model, retain_loader, forget_loader, val_loader)
                metrics['mia'].append(mia_prob)
 
            forget_loss, forget_acc = self.eval(forget_loader)
            metrics['forget']['acc'].append(forget_acc)
            metrics['forget']['loss'].append(forget_loss)

            retain_loss, retain_acc = self.eval(retain_loader)
            metrics['retain']['acc'].append(retain_acc)
            metrics['retain']['loss'].append(retain_loss)

            val_loss, val_acc = self.eval(val_loader)
            metrics['val']['acc'].append(val_acc)
            metrics['val']['loss'].append(val_loss)

            if verbose:
                epoch_iterator.set_postfix({
                    'forget_acc': f'{forget_acc:.4f}',
                    'retain_acc': f'{retain_acc:.4f}',
                    'val_acc': f'{val_acc:.4f}'
                })

            #### Gradient ascent
            # Create batch iterator with tqdm if verbose
            batch_iterator = tqdm(forget_loader, desc='Batches', leave=False, disable=not verbose)
            
            for batch in batch_iterator:
                optimizer.zero_grad()

                x = batch[0].to(self.device)
                y = batch[1].to(self.device)

                model_out = self.model(x)

                loss = -self.model.loss(model_out, y) # negate loss to ascend
                loss.backward()
                optimizer.step()

                if verbose:
                    batch_iterator.set_postfix({'loss': f'{-loss.item():.4f}'})
            
        return metrics