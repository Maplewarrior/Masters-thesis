import torch.optim as optim
import pdb

class SelectiveSynapticDampening:
    def __init__(self, model, criterion) -> None:
        self.model = model
        self.criterion = criterion
    
    
    def calculate_FIM(self, dataloader, savename: str):
        """
        
        """
        FIM_dictionary = {k: 0 for k in self.model.state_dict().keys()}
        # define optimizer to allow gradient computation
        optimizer = optim.SGD(self.model.parameters())
        # turn of dropout if applicable
        self.model.eval()
        
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            # forward pass
            logits = self.model(x)['logits']
            # calculate loss
            loss = self.criterion(logits, y)
            # calculate gradients
            loss.backward()
            for i, (name, param) in enumerate(self.model.named_parameters()):
                FIM_dictionary[name] += param.grad.data.clone().pow(2) #optimizer.param_groups[0]['params'][i].grad.pow(2)  
        # account for batched inference
        for k in FIM_dictionary.keys():
            FIM_dictionary[k] = FIM_dictionary[k] / len(dataloader)
        
        pdb.set_trace()



