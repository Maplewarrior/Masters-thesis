import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader) -> None:
        ### initialization
        self.model = model
        self.device = 'cpu'
        self.n_epochs = 70
        self.model.to(self.device)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-3)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        losses = []
        self.model.train()
        num_steps = len(self.train_dataloader) * self.n_epochs
        acc = 0
        with tqdm(range(num_steps)) as pbar:
            for step in pbar:
                ipt, label = next(iter(self.train_dataloader))
                ipt = ipt.to(self.device)
                label = label.to(self.device)

                out = self.model(ipt)
                acc += ((out['probabilities'].argmax(dim=1)) == label).sum().item()

                self.optimizer.zero_grad()
                loss = self.criterion(out['logits'], label)
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())

                if step % 5 ==0 :
                    epoch = int(step * self.n_epochs / num_steps) + 1
                    pbar.set_description(f"epoch={epoch}, step={step}, loss={torch.mean(torch.tensor(losses)):.1f}, acc={acc/((step+1)*self.train_dataloader.batch_size):.4f}")
                
    
    def eval(self):
        losses = []
        self.model.eval()
        acc = 0 
        with tqdm(range(len(self.val_dataloader))) as pbar:
            for step in pbar:
                ipt, label = next(iter(self.val_dataloader))
                ipt = ipt.to(self.device)
                label = label.to(self.device)

                out = self.model(ipt)
                loss = self.criterion(out['logits'], label)
                
                acc += ((out['probabilities'].argmax(dim=1)) == label).sum().item()

                losses.append(loss.item())
                # if step % 5 ==0 :            
                #     pbar.set_description(f"iter={step}/{len(self.val_dataloader)} loss={torch.mean(torch.tensor(losses)):.1f}")
        
        acc = acc / (len(self.val_dataloader) * self.val_dataloader.batch_size)
        print(f'Eval loss: {torch.mean(torch.tensor(losses)):.3f}')
        print(f'Eval accuracy: {acc:.3f}')