import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pdb

class Trainer:
    def __init__(self, model, 
                 train_dataloader, 
                 val_dataloader, 
                 n_epochs=20, 
                 lr=3e-3, 
                 device='cpu',
                 loss: nn.Module = nn.CrossEntropyLoss(),
                 disable_tqdm=False
                 ) -> None:
        ### initialization
        self.model = model
        self.device = device
        self.n_epochs = n_epochs
        self.model.to(self.device)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = loss
        self.disable_tqdm = disable_tqdm
    def train_sae(self):
        losses = []
        self.model.train()
        num_steps = len(self.train_dataloader) * self.n_epochs
        acc = 0
        with tqdm(range(num_steps), disable=self.disable_tqdm) as pbar:
            for step in pbar:
                ipt, label = next(iter(self.train_dataloader))
                ipt = ipt.to(self.device)
                label = label.to(self.device)

                out = self.model(ipt, return_reconstruction=True)
                pred = self.model.predict_from_reconstruction(out['xhat'])
                
                acc += ((pred['probabilities'].argmax(dim=1)) == label.argmax(dim=1)).sum().item()
            
                self.optimizer.zero_grad()
                loss = self.model.sae.loss(out['xact'], out)
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())

                if step % 5 ==0 :
                    epoch = int(step * self.n_epochs / num_steps) + 1
                    pbar.set_description(f"epoch={epoch}, step={step}, loss={torch.mean(torch.tensor(losses)):.1f}, acc={acc/((step+1)*self.train_dataloader.batch_size):.4f}")

    def train(self, n_epochs=None, disable_tqdm=False):
        """
        Train the model for a specified number of epochs.

        Args:
            n_epochs (int, optional): The number of epochs to train for. If None, the number of epochs specified in the constructor is used.
        """
        losses = []
        self.model.train()
        n_epochs = self.n_epochs if n_epochs is None else n_epochs
        
        with tqdm(range(n_epochs), disable=disable_tqdm) as epoch_pbar:
            for epoch in epoch_pbar:
                epoch_loss = 0.0
                epoch_acc = 0
                n_samples = 0
                
                # Properly iterate over all batches in the epoch
                for batch_idx, (ipt, label) in enumerate(self.train_dataloader):
                    ipt = ipt.to(self.device)
                    label = label.to(self.device)
                    
                    self.optimizer.zero_grad()
                    out = self.model(ipt)
                    
                    # Update accuracy
                    batch_correct = (out['probabilities'].argmax(dim=1) == label.argmax(dim=1)).sum().item()
                    epoch_acc += batch_correct
                    n_samples += len(label)
                    
                    # Compute loss and update
                    loss = self.criterion(out['logits'], label)
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    losses.append(loss.item())
                
                # Compute epoch metrics
                avg_epoch_loss = epoch_loss / len(self.train_dataloader)
                epoch_accuracy = epoch_acc / n_samples
                
                # Update progress bar
                epoch_pbar.set_description(
                    f"epoch={epoch+1}/{n_epochs}, "
                    f"loss={avg_epoch_loss:.4f}, "
                    f"acc={epoch_accuracy:.4f}"
                )

    def eval(self, disable_tqdm=False):
        """
        Evaluate the model on the validation set.
        
        Returns:
            tuple: (mean loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for ipt, label in self.val_dataloader:
                ipt = ipt.to(self.device)
                label = label.to(self.device)
                
                out = self.model(ipt)
                loss = self.criterion(out['logits'], label)
                
                total_loss += loss.item()
                total_correct += ((out['probabilities'].argmax(dim=1)) == label.argmax(dim=1)).sum().item()
                total_samples += len(label)
        
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = total_correct / total_samples
        
        print(f'Eval loss: {avg_loss:.3f}')
        print(f'Eval accuracy: {accuracy:.3f}')
        
        return avg_loss, accuracy