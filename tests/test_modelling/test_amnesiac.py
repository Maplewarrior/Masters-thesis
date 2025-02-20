import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.modelling.amnesiac import AmnesiacModel, AmnesiacTrainer
import os

class SimpleDataset(Dataset):
    def __init__(self):
        # Create a simple 2D dataset with 4 points, 2 classes
        self.x = torch.tensor([
            [0.0, 0.0],  # class 0
            [0.0, 1.0],  # class 1
            [1.0, 0.0],  # class 0
            [1.0, 1.0],  # class 1
        ])
        self.y = torch.tensor([0, 1, 0, 1])
        self.indices = torch.tensor([0, 1, 2, 3])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.indices[idx]

@pytest.fixture
def setup_trainer():
    # Create model, trainer and dataloaders
    model = AmnesiacModel(M=2, n_classes=2)
    trainer = AmnesiacTrainer(model, lr=0.1, device="cpu")
    
    # Create datasets
    dataset = SimpleDataset()
    
    # Create indices for forget and retain sets
    forget_indices = [1]  # forget the point [0,1]
    retain_indices = [0, 2, 3]
    
    # Create dataloaders
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    forget_loader = DataLoader(
        dataset, 
        batch_size=1, 
        sampler=torch.utils.data.SubsetRandomSampler(forget_indices)
    )
    retain_loader = DataLoader(
        dataset, 
        batch_size=1, 
        sampler=torch.utils.data.SubsetRandomSampler(retain_indices)
    )
    
    return {
        'trainer': trainer,
        'train_loader': train_loader,
        'forget_loader': forget_loader,
        'retain_loader': retain_loader,
        'forget_indices': forget_indices
    }

def test_training_stores_gradients(setup_trainer):
    trainer = setup_trainer['trainer']
    train_loader = setup_trainer['train_loader']
    
    # Train for 1 epoch
    trainer.train(train_loader, epochs=1)
    
    # Check that gradients were stored
    assert 0 in trainer.batch_mapping
    assert len(trainer.batch_mapping[0]) == 4  # Should have 4 samples
    assert len(trainer.batch_params[0]) == 4   # Should have 4 batches of parameters

def test_selective_gradient_storage(setup_trainer):
    trainer = setup_trainer['trainer']
    train_loader = setup_trainer['train_loader']
    forget_indices = setup_trainer['forget_indices']
    
    # Train storing only gradients for forget indices
    trainer.train(train_loader, epochs=1, indices_to_forget=forget_indices)
    
    # Check that only gradients for forget indices were stored
    assert 0 in trainer.batch_mapping
    assert len(trainer.batch_mapping[0]) == 1  # Should only store the forget sample
    assert len(trainer.batch_params[0]) == 1   # Should only have 1 batch of parameters

def test_forgetting(setup_trainer):
    trainer = setup_trainer['trainer']
    train_loader = setup_trainer['train_loader']
    forget_loader = setup_trainer['forget_loader']
    forget_indices = setup_trainer['forget_indices']
    
    # Train and get accuracy on forget set
    trainer.train(train_loader, epochs=1)
    _, initial_forget_acc = trainer.evaluate(forget_loader)
    
    # Perform forgetting
    trainer.forget(indices_to_forget=forget_indices)
    
    # Check accuracy on forget set after forgetting
    _, final_forget_acc = trainer.evaluate(forget_loader)
    
    # Accuracy on forget set should decrease
    assert final_forget_acc <= initial_forget_acc

def test_repair(setup_trainer):
    trainer = setup_trainer['trainer']
    train_loader = setup_trainer['train_loader']
    retain_loader = setup_trainer['retain_loader']
    forget_indices = setup_trainer['forget_indices']
    
    # Train, forget, and get accuracy on retain set
    trainer.train(train_loader, epochs=1)
    trainer.forget(indices_to_forget=forget_indices)
    _, initial_retain_acc = trainer.evaluate(retain_loader)
    
    # Perform repair
    trainer.train(retain_loader, epochs=1, repair=True)
    
    # Check accuracy on retain set after repair
    _, final_retain_acc = trainer.evaluate(retain_loader)
    
    # Accuracy on retain set should improve or stay the same
    assert final_retain_acc >= initial_retain_acc

def test_checkpoint_saving(setup_trainer, tmp_path):
    trainer = setup_trainer['trainer']
    train_loader = setup_trainer['train_loader']
    
    # Train with checkpoint saving
    trainer.train(train_loader, epochs=1, ckpt=True)
    
    # Check that checkpoint file exists
    checkpoint_path = "results/amnesiac/checkpoints/checkpoint_0.pth"
    # create the directory if it doesn't exist
    checkpoint = torch.load(checkpoint_path)
    
    assert 'model_state_dict' in checkpoint
    assert 'optimizer_state_dict' in checkpoint
    assert 'batch_mapping' in checkpoint
    assert 'batch_params' in checkpoint