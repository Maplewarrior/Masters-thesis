import torch

class UnlearningEvaluator:
    def __init__(self) -> None:
        pass

    def evaluate_performance(self, model, dataloader):
        model.eval()
        acc = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                out = model(x)
                acc += ((out['probabilities'].argmax(dim=1)) == y.argmax(dim=1)).sum().item()
        
        acc = acc / len(dataloader.dataset)

        return {'accuracy': acc}
            
        





