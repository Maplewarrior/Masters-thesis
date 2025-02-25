import pytest
import numpy as np
import torch
from src.data_utils.synthetic_data import DataGenerator, SyntheticDataset, create_dataloaders

@pytest.fixture
def sample_data():
    """Fixture for sample data used in tests"""
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 4, size=100)
    n_classes = 4
    return X, y, n_classes

@pytest.mark.synthetic_dataset
class TestSyntheticDataset:
    def test_dataset_initialization(self, sample_data):
        X, y, _ = sample_data
        dataset = SyntheticDataset(X, y)
        assert len(dataset) == 100
        assert torch.is_tensor(dataset.X)
        assert torch.is_tensor(dataset.y)

    def test_onehot_encoding(self, sample_data):
        X, y, n_classes = sample_data
        dataset = SyntheticDataset(X, y, onehot_labels=True, n_classes=n_classes)
        # Check shape of one-hot encoded labels
        assert dataset.y.shape == (100, n_classes)
        # Check if one-hot encoding is valid
        assert torch.all(dataset.y.sum(dim=1) == 1)

    def test_getitem(self, sample_data):
        X, y, _ = sample_data
        dataset = SyntheticDataset(X, y)
        X_item, y_item = dataset[0]
        assert X_item.shape == (2,)
        assert isinstance(y_item.item(), int)

    def test_dataset_with_indices(self, sample_data):
        X, y, _ = sample_data
        dataset = SyntheticDataset(X, y, use_indices=True)
        X_item, y_item, idx = dataset[0]
        assert X_item.shape == (2,)
        assert isinstance(y_item.item(), int)
        assert isinstance(idx.item(), int)
        assert idx.item() == 0

@pytest.fixture
def data_generator():
    """Fixture for DataGenerator"""
    return DataGenerator(random_state=42)

@pytest.mark.data_generator
class TestDataGenerator:
    def test_generate_data(self, data_generator):
        n_samples = 1000
        n_features = 2
        n_classes = 4
        data_generator.generate_data(
            n_samples=n_samples, 
            n_features=n_features,
            n_classes=n_classes
        )
        assert data_generator.X.shape == (n_samples, n_features)
        assert len(data_generator.y) == n_samples
        assert len(np.unique(data_generator.y)) == n_classes

    @pytest.mark.outliers
    def test_make_outliers_all_classes(self, data_generator):
        data_generator.generate_data(n_samples=1000, n_outliers=100, 
                                   outlier_scale=2.0, outlier_variance=0.2, 
                                   outlier_class=None)
        
        outlier_idx = data_generator.outlier_idx
        assert len(outlier_idx) == 100

        # Check if outliers are split across all classes
        for class_idx in range(data_generator.n_classes):
            assert len(np.intersect1d(outlier_idx, 
                np.where(data_generator.y == class_idx)[0])) > 0

    @pytest.mark.outliers
    def test_make_outliers_single_class(self, data_generator):
        data_generator.generate_data(n_samples=1000, n_outliers=100, 
                                   outlier_scale=2.0, outlier_variance=0.2, 
                                   outlier_class=1)

        outlier_idx = data_generator.outlier_idx
        assert len(outlier_idx) == 100
        # Check if outliers are only from class 1
        assert all(data_generator.y[outlier_idx] == 1)

    def test_split_data(self, data_generator):
        data_generator.generate_data(n_samples=1000)
        train_idx, val_idx, test_idx = data_generator.split_data(0.8, 0.1, 0.1)
        
        # Check sizes
        assert abs(len(train_idx) / 1000 - 0.8) < 0.01
        assert abs(len(val_idx) / 1000 - 0.1) < 0.01
        assert abs(len(test_idx) / 1000 - 0.1) < 0.01
        
        # Check no overlap between sets
        assert len(np.intersect1d(train_idx, val_idx)) == 0
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert len(np.intersect1d(val_idx, test_idx)) == 0

    @pytest.mark.forget_set
    def test_draw_forget_set_single_class(self, data_generator):
        n_samples = 1000
        data_generator.generate_data(n_samples=n_samples)
        frac_train, frac_val, frac_test = 0.8, 0.1, 0.1
        data_generator.split_data(frac_train, frac_val, frac_test)
        n_forget = 20

        forget_X, forget_y, retain_X, retain_y, val_X, val_y, test_X, test_y = \
            data_generator.draw_forget_set(n_points=n_forget, class_idx=1, ood_ratio=0.5)
        
        n_train = int(n_samples * frac_train)
        n_val = int(n_samples * frac_val)
        n_test = int(n_samples * frac_test)

        # Check sizes
        assert len(forget_X) == n_forget
        assert len(forget_y) == n_forget
        assert len(retain_X) == n_train - n_forget
        assert len(retain_y) == n_train - n_forget
        assert len(val_X) == n_val
        assert len(val_y) == n_val
        assert len(test_X) == n_test
        assert len(test_y) == n_test

        # Check that forget set is from specified class
        assert all(y == 1 for y in forget_y)

    @pytest.mark.forget_set
    def test_multiple_forget_sets(self, data_generator):
        """Test that we can draw multiple different forget sets from the same data."""
        n_samples = 1000
        data_generator.generate_data(n_samples=n_samples)
        data_generator.split_data(0.8, 0.1, 0.1)
        n_forget = 20

        # Draw forget sets with different random states
        forget_X1, forget_y1, retain_X1, retain_y1, _, _, _, _ = \
            data_generator.draw_forget_set(n_points=n_forget, class_idx=1, 
                                         ood_ratio=0.5, random_state=42)
        forget_idx1 = data_generator.forget_idx_in_train.copy()
        retain_idx1 = data_generator.retain_idx_in_train.copy()

        forget_X2, forget_y2, retain_X2, retain_y2, _, _, _, _ = \
            data_generator.draw_forget_set(n_points=n_forget, class_idx=1, 
                                         ood_ratio=0.5, random_state=43)
        forget_idx2 = data_generator.forget_idx_in_train.copy()
        retain_idx2 = data_generator.retain_idx_in_train.copy()

        # Verify different random states produce different forget sets
        assert not np.array_equal(forget_idx1, forget_idx2)
        assert not np.array_equal(retain_idx1, retain_idx2)
        assert len(forget_X1) == len(forget_X2)
        assert len(retain_X1) == len(retain_X2)

    @pytest.mark.forget_set
    def test_forget_set_random_state(self, data_generator):
        """Test that the same random state produces identical forget sets."""
        n_samples = 1000
        data_generator.generate_data(n_samples=n_samples)
        data_generator.split_data(0.8, 0.1, 0.1)
        n_forget = 20

        # Draw two forget sets with same random state
        forget_X1, forget_y1, retain_X1, retain_y1, _, _, _, _ = \
            data_generator.draw_forget_set(n_points=n_forget, class_idx=1, 
                                         ood_ratio=0.5, random_state=42)
        forget_idx1 = data_generator.forget_idx_in_train.copy()
        retain_idx1 = data_generator.retain_idx_in_train.copy()

        forget_X2, forget_y2, retain_X2, retain_y2, _, _, _, _ = \
            data_generator.draw_forget_set(n_points=n_forget, class_idx=1, 
                                         ood_ratio=0.5, random_state=42)
        forget_idx2 = data_generator.forget_idx_in_train.copy()
        retain_idx2 = data_generator.retain_idx_in_train.copy()

        # Verify same random state produces identical forget sets
        assert np.array_equal(forget_idx1, forget_idx2)
        assert np.array_equal(retain_idx1, retain_idx2)
        assert np.array_equal(forget_X1, forget_X2)
        assert np.array_equal(forget_y1, forget_y2)
        assert np.array_equal(retain_X1, retain_X2)
        assert np.array_equal(retain_y1, retain_y2)

@pytest.fixture
def setup_dataloaders():
    """Fixture for dataloader tests"""
    data_generator = DataGenerator(random_state=42)
    data_generator.generate_data(n_samples=1000)
    data_generator.split_data(0.8, 0.1, 0.1)
    data_generator.draw_forget_set(n_points=20, class_idx=1, ood_ratio=0.5)
    return data_generator

@pytest.mark.dataloaders
class TestDataLoaders:
    def test_dataloader_creation(self, setup_dataloaders):
        dataloaders = create_dataloaders(setup_dataloaders, batch_size=32)
        
        # Check if all required loaders are present
        assert 'train_full_loader' in dataloaders
        assert 'val_loader' in dataloaders
        assert 'test_loader' in dataloaders
        assert 'train_retain_loader' in dataloaders
        assert 'train_forget_loader' in dataloaders

    def test_batch_shapes(self, setup_dataloaders):
        dataloaders = create_dataloaders(setup_dataloaders, batch_size=32)
        
        X_batch, y_batch = next(iter(dataloaders['train_full_loader']))
        assert X_batch.shape[1] == 2  # number of features
        assert X_batch.shape[0] <= 32  # batch size

    def test_dataloader_with_indices(self, setup_dataloaders):
        dataloaders = create_dataloaders(setup_dataloaders, batch_size=32, use_indices=True)
        X_batch, y_batch, indices = next(iter(dataloaders['train_full_loader']))
        assert len(indices) == X_batch.shape[0]
        assert torch.all(indices < len(setup_dataloaders.train_X)) 