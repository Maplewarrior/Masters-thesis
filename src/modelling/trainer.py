import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
        
        train_l0_norms = []
        val_l0_norms = []
        train_accs = []
        val_accs = []
        epochs = []
        with tqdm(range(self.n_epochs), disable=self.disable_tqdm) as epoch_pbar:
            for epoch in epoch_pbar:
                acc = 0
                for ipt, label in self.train_dataloader:
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
                    # pdb.set_trace()
                val_loss, val_acc, val_l0 = self.eval_sae()
                epoch_pbar.set_description(
                                           f"epoch={epoch}, loss={torch.mean(torch.tensor(losses)):.2f}, "
                                           f"acc={acc/len(self.train_dataloader.dataset):.2f}, "
                                           f"avg. l0-norm: {out['z'].norm(p=0, dim=-1).mean():.2f}, "
                                           f"val_acc={val_acc:.2f}, "
                                           f"val l0-norm {val_l0:.2f}"
                                           )
                epochs.append(epoch)
                train_accs.append(acc/len(self.train_dataloader.dataset))
                train_l0_norms.append(out['z'].norm(p=0, dim=-1).mean().detach().item())
                val_accs.append(val_acc)
                val_l0_norms.append(val_l0.item())
            pdb.set_trace()
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mtick

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))         
            ax1.plot(epochs, train_accs, 'o-', color='#1f77b4', label='Train Accuracy', linewidth=2, markersize=4)
            ax1.plot(epochs, val_accs, 's-', color='#ff7f0e', label='Validation Accuracy', linewidth=2, markersize=4)
            ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            ax1.set_title('Model Accuracy over Training', fontsize=14, fontweight='bold')
            ax1.legend(frameon=True, fontsize=10)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_xlim(0, max(epochs) + 1)
            ax1.set_ylim(0, 1.05)
            ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            # Plot 2: L0 Norm vs Epochs
            ax2.plot(epochs, train_l0_norms, 'o-', color='#2ca02c', label='Train L0 Norm', linewidth=2, markersize=4)
            ax2.plot(epochs, val_l0_norms, 's-', color='#d62728', label='Validation L0 Norm', linewidth=2, markersize=4)
            ax2.set_xlabel('Epochs', fontsize=12, fontweight='bold')
            ax2.set_ylabel('L0 Norm', fontsize=12, fontweight='bold')
            ax2.set_title('L0 Norm over Training', fontsize=14, fontweight='bold')
            ax2.legend(frameon=True, fontsize=10)
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_xlim(0, max(epochs) + 1)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

            plt.savefig('SAE_dynamics_original_loss.png')

    def train(self, n_epochs=None, disable_tqdm=False):
        """
        Train the model for a specified number of epochs.

        Args:
            n_epochs (int, optional): The number of epochs to train for. If None, the number of epochs specified in the constructor is used.
        """
        losses = []
        self.model.train()
        n_epochs = self.n_epochs if n_epochs is None else n_epochs
        val_losses = []
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
                
                val_loss, val_acc = self.eval()
                val_losses.append(val_loss)
                # Update progress bar
                epoch_pbar.set_description(
                    f"epoch={epoch+1}/{n_epochs}, "
                    f"loss={avg_epoch_loss:.4f}, "
                    f"acc={epoch_accuracy:.4f}, "
                    f"val_acc: {val_acc}"
                )

                if epoch > 4 and not val_losses[-1] < np.mean(val_losses[:-4:-1]):
                    print("\Ending due to early stopping")
                    return
                
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
                
                out = self.model.inference(ipt)
                
                if 'z' in out.keys():
                    loss = self.criterion(out['logits'], label, z=out['z'])
                else:
                    loss = self.criterion(out['logits'], label)
                
                total_loss += loss.item()
                total_correct += ((out['probabilities'].argmax(dim=1)) == label.argmax(dim=1)).sum().item()
                total_samples += len(label)
        
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = total_correct / total_samples
        
        # print(f'Eval loss: {avg_loss:.3f}')
        # print(f'Eval accuracy: {accuracy:.3f}')
        
        return avg_loss, accuracy
    
    def eval_sae(self, disable_tqdm=False):
        """
        Evaluate the model on the validation set.
        
        Returns:
            tuple: (mean loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        avg_l0_norm = 0
        with torch.no_grad():
            for ipt, label in self.val_dataloader:
                ipt = ipt.to(self.device)
                label = label.to(self.device)
                
                out = self.model(ipt)
                pred = self.model.predict_from_reconstruction(out['xhat'])
                
                loss = self.model.sae.loss(out['xact'], out)

                total_loss += loss.item()
                total_correct += ((pred['probabilities'].argmax(dim=1)) == label.argmax(dim=1)).sum().item()
                total_samples += len(label)
                avg_l0_norm += out['z'].norm(p=0, dim=-1).mean()
                
        
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = total_correct / total_samples
        avg_l0_norm = avg_l0_norm / len(self.val_dataloader)
        
        # print(f'Eval loss: {avg_loss:.3f}')
        # print(f'Eval accuracy: {accuracy:.3f}')
        
        return avg_loss, accuracy, avg_l0_norm
    
    def fine_tune_w_sae(self, n_epochs=None, disable_tqdm=False):
        """
        Train the model for a specified number of epochs.

        Args:
            n_epochs (int, optional): The number of epochs to train for. If None, the number of epochs specified in the constructor is used.
        """
        losses = []
        self.model.train()
        n_epochs = self.n_epochs if n_epochs is None else n_epochs
        val_losses = []
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
                    out = self.model(ipt, train_sae=False)
                    # Update accuracy
                    batch_correct = (out['probabilities'].argmax(dim=1) == label.argmax(dim=1)).sum().item()
                    epoch_acc += batch_correct
                    n_samples += len(label)
                    
                    # Compute loss and update
                    loss = self.criterion(out['logits'], label, out['z'])
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    losses.append(loss.item())
                
                # Compute epoch metrics
                avg_epoch_loss = epoch_loss / len(self.train_dataloader)
                epoch_accuracy = epoch_acc / n_samples
                
                val_loss, val_acc = self.eval()
                val_losses.append(val_loss)
                # Update progress bar
                epoch_pbar.set_description(
                    f"epoch={epoch+1}/{n_epochs}, "
                    f"loss={avg_epoch_loss:.4f}, "
                    f"acc={epoch_accuracy:.4f}, "
                    f"val_acc: {val_acc}"
                )

                # if epoch > 4 and not val_losses[-1] < np.mean(val_losses[:-4:-1]):
                #     print("\Ending due to early stopping")
                #     return
