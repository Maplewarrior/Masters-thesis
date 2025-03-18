
from src.modelling.neural_network import SimpleNeuralNet
from src.modelling.maso import maso_dataset, MASO, MasoDataset
from src.modelling.trainer import Trainer

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pdb
def main():
    n_features = 2
    n_classes = 4
    
    model1 = SimpleNeuralNet(M=n_features, n_classes=n_classes)
    model1.load_state_dict(torch.load("src/modelling/maso_model/simple_model.pth", weights_only=True))
    model2 = SimpleNeuralNet(M=n_features, n_classes=n_classes)
    model2.load_state_dict(torch.load("src/modelling/maso_model/simple_model2.pth", weights_only=True))

    x = torch.tensor([[0., 0.]])
    pdb.set_trace()


def setup():
    # ==============================
    # Define dataset structure
    # ==============================
    n_samples = 1000
    n_features = 2
    n_classes = 4
    np.random.seed(42)
    X, y = maso_dataset(n_samples=n_samples, 
                       n_features=n_features, 
                       n_classes=n_classes)
    
    # Train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # Since y is already one-hot encoded from maso_dataset
    train_dataset = MasoDataset(X_train, y_train, onehot_labels=True, n_classes=n_classes)
    val_dataset = MasoDataset(X_val, y_val, onehot_labels=True, n_classes=n_classes)
    test_dataset = MasoDataset(X_test, y_test, onehot_labels=True, n_classes=n_classes)

    
    # Convert to DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
    # ==============================
    # Create model
    # ==============================
    model = SimpleNeuralNet(M=n_features, n_classes=n_classes)

    # Load model
    model.load_state_dict(torch.load("src/modelling/maso_model/simple_model.pth"))

    # maso = MASO(model, train_loader, _lambda=0.3)

    # # ==============================
    # # Train model
    # # ==============================
    # trainer = Trainer(model, 
    #                  train_dataloader=train_loader, 
    #                  val_dataloader=val_loader, 
    #                  lr=0.001, 
    #                  device='cpu',
    #                  loss=maso.loss)
    
    # trainer.train(n_epochs=40)
    # trainer.eval()

    # # Save model
    # torch.save(model.state_dict(), "src/modelling/maso_model/simple_model2.pth")
    return model

if __name__ == '__main__':
    main()
    

    # Load model
    # model.load_state_dict(torch.load("src/modelling/maso_model/simple_model.pth"))

    