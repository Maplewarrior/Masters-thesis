from src.trainers.base_trainer import BaseTrainer
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.sisa_class import SISA

class SISATrainer(BaseTrainer):
    def __init__(self, 
                 model: nn.Module,
                 sisa: SISA,
                 train_dataloader, 
                 val_dataloader,
                 logger=None,
                 device="cpu",
                 learning_rate=0.001,
                 disable_tqdm=False,
                 do_early_stopping=True,
                 **kwargs) -> None:
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.sisa = sisa

        super().__init__(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            logger=logger,
            disable_tqdm=disable_tqdm,
            do_early_stopping=do_early_stopping
        )
    
    def train_client_models(self):
        # TODO: Call self.train() as many times as there are n_clients 
        self.sisa.train_client_models()
    
    def __call__(self):
        return self.train_client_models()
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.data_utils.synthetic_data import SyntheticDataset
    n_features = 600
    n_classes = 2
    n_epochs = 30
    n_shards = 8
    n_slices = 8
    # data_generator = DataGenerator(random_state=42)
    # data_generator.generate_data(n_samples=500, n_features=n_features, 
    #                          n_informative=2, n_redundant=0, n_classes=n_classes, n_outliers=10)
    # data_generator.split_data()
    # data_generator.draw_forget_set(n_points=10, class_idx=1, ood_ratio=0.5)
    # dataloaders = create_dataloaders(data_generator, batch_size=32)
    
    # train_loader = dataloaders["train_full_loader"]
    # val_loader = dataloaders["val_loader"]
    
    from src.sisa_implementation.sisa_dataloader import load
    X_train, y_train = load(indices=range(1000), category='train')
    X_test, y_test = load(indices=range(1000), category='test')

    # convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(SyntheticDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(SyntheticDataset(X_test, y_test), batch_size=32, shuffle=False)

    sisa = SISA(train_loader, n_classes=n_classes, 
                n_features=n_features, n_epochs=n_epochs, 
                n_shards=n_shards, n_slices=n_slices)
    shards_dict = sisa.process_data()

    SISATrainer(
        model=sisa.model, 
        sisa=sisa, 
        train_dataloader=train_loader, 
        val_dataloader=test_loader, 
        logger=None, 
        device="cpu", 
        learning_rate=0.001, 
        n_epochs=n_epochs, 
        disable_tqdm=False, 
        do_early_stopping=False)()