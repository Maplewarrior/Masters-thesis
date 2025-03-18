from src.trainers.base_trainer import BaseTrainer

class SAETrainer(BaseTrainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def _initialize_criterion(self):
        raise NotImplementedError()
    
    def step(self):
        raise NotImplementedError()
    
    def eval(self):
        raise NotImplementedError()
    