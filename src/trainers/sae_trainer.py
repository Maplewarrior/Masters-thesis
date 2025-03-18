import torch
from src.trainers.base_trainer import BaseTrainer

class SAETrainer(BaseTrainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def step(self, x: torch.tensor, y: torch.tensor):
        self.optimizer.zero_grad()
        out = self.model(x, return_reconstruction=True)
        loss = self.model.sae.loss(out['xact'], out)
        loss.backward()
        self.optimizer.step()
        return out, loss
    
    def eval(self):
        raise NotImplementedError()
    
    def train_sae(self):
        losses = []
        self.model.train()
        
        
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
                
                out = self.model(ipt, return_reconstruction=True)
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