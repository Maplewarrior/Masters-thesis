
import pdb

import torch.nn as nn
from src.modelling.trainer import Trainer
from src.modelling.neural_network import NeuralNetwork
from src.data_utils.synthetic_data import DataGenerator, create_dataloaders
from src.modelling.selective_synaptic_dampening import SelectiveSynapticDampening

if __name__ == '__main__':
    n_features = 2
    n_classes = 4
    data_generator = DataGenerator(random_state=42)
    data_generator.generate_data(n_samples=1000, n_features=n_features,
                                 n_classes=n_classes, n_informative=2, 
                                 n_redundant=0, n_outliers=50, outlier_scale=4.0, 
                                 outlier_variance=0.4, outlier_class=1)
    data_generator.draw_forget_set(n_points=20, class_idx=1, ood_ratio=0.5)
    dataloaders = create_dataloaders(data_generator, batch_size=32)

    model = NeuralNetwork(n_features, n_classes)
    trainer = Trainer(model, train_dataloader=dataloaders['full_loader'], val_dataloader=None)
    trainer.train()

    criterion = nn.CrossEntropyLoss()
    SSD = SelectiveSynapticDampening(model, criterion)
    SSD.calculate_FIM(dataloaders['full_loader'], savename='full_synthetic_dataset')
    

