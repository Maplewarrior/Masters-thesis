from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import pdb
class ScrubR:
    def __init__(self, model, alpha, gamma) -> None:
        self.model = model
        self.original_model = copy.deepcopy(self.model)
        self.__freeze_original_model()
        self.CE = nn.CrossEntropyLoss()
        self.KL = nn.KLDivLoss(reduction='batchmean')
        self.alpha = alpha # hyperparam for distance between student & teacher on retain data
        self.gamma = gamma # hyperparam for cross entropy
        # self.min_optimizer = optim.Adam(self.model.parameters(), lr = 1e-4)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-10)
        
    def __freeze_original_model(self):
        for param in self.original_model.parameters():
            param.requires_grad=False

    def max_step(self, x, y):
        """
        Maximization step of the SCRUB loss: Encourages the model to have high error on forget data.
        """
        self.optimizer.zero_grad()
        probabilities = self.model(x)['probabilities']
        loss = -self.KL(input=torch.log(probabilities) + 1e-6, target=y) # maximize KL divergence
        loss.backward()
        self.optimizer.step()

    def min_step(self, x, y):
        """
        Miminimzation step of the SCRUB loss: Encourages performance to have low error and have similar outputs to original model on retain data.
        """
        self.optimizer.zero_grad()
        with torch.no_grad():
            teacher_probs = self.original_model(x)['probabilities']
        
        student_out = self.model(x)
        reg_term = self.KL(input=torch.log(student_out['probabilities']), target=teacher_probs)
        fit_term = self.CE(student_out['logits'], y)
        loss = self.alpha * reg_term + self.gamma * fit_term
        loss.backward()
        self.optimizer.step()

    def __call__(self, retain_dataloader, forget_dataloader, n_rounds: int) -> Any:
        
        for _ in range(n_rounds):
            for x, y in forget_dataloader:
                self.max_step(x, y)
            
            for x, y in retain_dataloader:
                self.min_step(x, y)
    
