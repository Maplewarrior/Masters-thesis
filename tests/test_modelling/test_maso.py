import pytest
import torch
import numpy as np
from src.modelling.maso import MASO, MasoDataset, maso_dataset
from src.modelling.neural_network import NeuralNet
import torch.nn as nn

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    y = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    return X, y

@pytest.fixture
def maso_dataset_fixture(sample_data):
    """Create a MasoDataset instance for testing"""
    X, y = sample_data
    return MasoDataset(X, y, onehot_labels=True, n_classes=2)

@pytest.fixture
def model_fixture():
    """Create a simple neural network model for testing"""
    return NeuralNet(M=2, n_classes=2)

@pytest.fixture
def maso_fixture(maso_dataset_fixture, model_fixture):
    """Create a MASO instance for testing"""
    dataloader = torch.utils.data.DataLoader(maso_dataset_fixture, batch_size=2)
    return MASO(model_fixture, dataloader, _lambda=0.1)

@pytest.fixture
def maso_dataset_fixture(sample_data):
    """Create a MasoDataset instance for testing"""
    X, y = sample_data
    return MasoDataset(X, y, onehot_labels=True, n_classes=2)

def test_maso_dataset_creation(sample_data):
    """Test MasoDataset initialization and basic functionality"""
    X, y = sample_data
    dataset = MasoDataset(X, y, onehot_labels=True, n_classes=2)
    
    assert len(dataset) == 3
    assert dataset.X.shape == (3, 2)
    assert dataset.y.shape == (3, 2)
    
    # Test __getitem__
    x, y = dataset[0]
    assert x.shape == (2,)
    assert y.shape == (2,)

def test_maso_dataset_with_indices(sample_data):
    """Test MasoDataset with indices enabled"""
    X, y = sample_data
    dataset = MasoDataset(X, y, use_indices=True, onehot_labels=True, n_classes=2)
    
    x, y, idx = dataset[0]
    assert isinstance(idx, torch.Tensor)
    assert idx.item() == 0

def test_maso_loss_computation(maso_fixture):
    """Test MASO loss computation"""
    batch_size = 2
    n_classes = 2
    logits = torch.randn(batch_size, n_classes)
    y = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    
    loss = maso_fixture.loss(logits, y)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar value

def test_maso_params_extraction(maso_fixture):
    """Test extraction of MASO parameters"""
    maso_params = maso_fixture.maso_params
    
    assert isinstance(maso_params, list)
    assert len(maso_params) > 0
    
    # Check structure of parameters
    for W, b in maso_params:
        assert isinstance(W, np.ndarray)
        assert isinstance(b, np.ndarray)
        assert W.ndim == 2  # Weight matrix should be 2D
        assert b.ndim == 1  # Bias vector should be 1D

def test_partition_indices_computation(maso_fixture, sample_data):
    """Test computation of partition indices"""
    X, _ = sample_data
    partition_map = maso_fixture.compute_partition_indices(X)
    
    assert isinstance(partition_map, dict)
    assert len(partition_map) > 0
    
    # Check structure of partition indices
    for layer_idx, indices in partition_map.items():
        assert isinstance(layer_idx, int)
        assert isinstance(indices, np.ndarray)
        assert indices.shape[0] == len(X)

def test_maso_dataset_generator():
    """Test synthetic dataset generation"""
    n_samples = 100
    n_features = 2
    n_classes = 3
    
    X, y = maso_dataset(n_samples, n_features, n_classes)
    
    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples, n_classes)

def test_maso_distance_computation(maso_fixture):
    """Test distance computation between partition indices"""
    x1_idx, x2_idx = 0, 1
    distance = maso_fixture.distance(x1_idx, x2_idx)
    
    assert isinstance(distance, float)
    assert 0 <= distance <= 1  # Distance should be normalized

def test_onehot_encode_labels():
    """Test one-hot encoding functionality"""
    n_samples = 100
    n_features = 2
    n_classes = 3
    
    X, y = maso_dataset(n_samples, n_features, n_classes)

    dataset = MasoDataset(X, y, onehot_labels=True, n_classes=n_classes)

    assert dataset.y.shape == (n_samples, n_classes)
    assert torch.all(torch.sum(dataset.y, dim=1) == 1)

# TODO: Add tests for plotting each layer's splines
