import pytest
import torch
import numpy as np
from src.modelling.amnesiac import AmnesiacTrainer
from src.modelling.trainer import Trainer
from src.data_utils.synthetic_data import DataGenerator, create_dataloaders
from src.modelling.neural_network import NeuralNet
import copy

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def setup_synthetic_data(seed):
    """Setup synthetic datasets with indices for both trainers."""
    n_classes = 4
    n_samples = 1000
    n_features = 2
    
    # Generate synthetic data
    data_generator = DataGenerator(random_state=seed)
    data_generator.generate_data(
        n_samples=n_samples, 
        n_features=n_features,
        n_informative=n_features, 
        n_redundant=0,
        n_outliers=50, 
        n_classes=n_classes, 
        outlier_scale=4.0,
        outlier_variance=0.4, 
        outlier_class=1
    )
    
    data_generator.split_data(
        train_ratio=0.8, 
        val_ratio=0.1, 
        test_ratio=0.1
    )
    
    data_generator.draw_forget_set(
        n_points=10, 
        class_idx=1, 
        ood_ratio=1.0
    )

    return create_dataloaders(
        data_generator, 
        batch_size=32, 
        use_indices=True,
        onehot_labels=False,
        device=device,
        shuffle=False
    ), n_features, n_classes

def train_and_evaluate_models(seed, epochs=5, lr=3e-3):
    """Train both models and return their validation accuracies."""
    torch.manual_seed(seed)
    
    # Setup data and models
    dataloaders, n_features, n_classes = setup_synthetic_data(seed)
    
    # Setup Amnesiac model and data
    model_amnesiac = NeuralNet(M=n_features, n_classes=n_classes)
    full_loader_amnesiac = dataloaders['train_full_loader']
    val_loader_amnesiac = dataloaders['val_loader']

    # Setup Normal model and data
    model_normal = copy.deepcopy(model_amnesiac)
    full_loader_normal = copy.deepcopy(full_loader_amnesiac)
    full_loader_normal.dataset.indices = None
    full_loader_normal.dataset.onehot_labels = True
    full_loader_normal.dataset.y = full_loader_normal.dataset.onehot_encode_labels(
        full_loader_normal.dataset.y, 
        full_loader_normal.dataset.n_classes
    )

    val_loader_normal = copy.deepcopy(val_loader_amnesiac)
    val_loader_normal.dataset.indices = None
    val_loader_normal.dataset.onehot_labels = True
    val_loader_normal.dataset.y = val_loader_normal.dataset.onehot_encode_labels(
        val_loader_normal.dataset.y, 
        val_loader_normal.dataset.n_classes
    )

    # Train normal model
    trainer = Trainer(model_normal, full_loader_normal, val_loader_normal, device=device, lr=lr)
    trainer.train(n_epochs=epochs)
    normal_acc = trainer.eval()[1]

    # Train amnesiac model
    amnesiac_trainer = AmnesiacTrainer(model_amnesiac, device=device, lr=lr)
    amnesiac_trainer.train(
        train_loader=full_loader_amnesiac, 
        val_loader=val_loader_amnesiac, 
        epochs=epochs, 
        repair=True  # Don't store gradients for comparison
    )
    amnesiac_acc = amnesiac_trainer.evaluate(val_loader_amnesiac)[1]

    return normal_acc, amnesiac_acc

@pytest.mark.amnesiac
def test_trainer_performance_similarity():
    """
    Test that both trainers achieve similar performance within a threshold.
    
    We train multiple models with different seeds and compare their average performance.
    The difference in mean accuracy should be within the specified threshold.
    """
    n_runs = 5
    accuracy_threshold = 0.05  # 5% difference threshold
    epochs = 5
    lr = 3e-3
    
    normal_accs = []
    amnesiac_accs = []
    
    # Train multiple models with different seeds
    for seed in range(n_runs):
        normal_acc, amnesiac_acc = train_and_evaluate_models(seed, epochs, lr)
        normal_accs.append(normal_acc)
        amnesiac_accs.append(amnesiac_acc)
    
    # Calculate statistics
    normal_mean = np.mean(normal_accs)
    amnesiac_mean = np.mean(amnesiac_accs)
    normal_std = np.std(normal_accs)
    amnesiac_std = np.std(amnesiac_accs)
    
    # Print detailed results
    print("\nTest Results:")
    print(f"Normal Trainer - Mean: {normal_mean:.4f}, Std: {normal_std:.4f}")
    print(f"Amnesiac Trainer - Mean: {amnesiac_mean:.4f}, Std: {amnesiac_std:.4f}")
    print(f"Absolute difference in means: {abs(normal_mean - amnesiac_mean):.4f}")
    
    # Assert that the difference in mean accuracy is within threshold
    assert abs(normal_mean - amnesiac_mean) < accuracy_threshold, \
        f"Performance difference ({abs(normal_mean - amnesiac_mean):.4f}) exceeds threshold ({accuracy_threshold})"

# Global device setup
device = get_device()