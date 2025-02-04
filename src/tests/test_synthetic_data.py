import unittest
import numpy as np
import torch
from src.data_utils.synthetic_data import DataGenerator, SyntheticDataset, create_dataloaders

class TestSyntheticDataset(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.X = np.random.randn(100, 2)
        self.y = np.random.randint(0, 4, size=100)
        self.n_classes = 4

    def test_dataset_initialization(self):
        dataset = SyntheticDataset(self.X, self.y)
        self.assertEqual(len(dataset), 100)
        self.assertTrue(torch.is_tensor(dataset.X))
        self.assertTrue(torch.is_tensor(dataset.y))

    def test_onehot_encoding(self):
        dataset = SyntheticDataset(self.X, self.y, onehot_labels=True, n_classes=self.n_classes)
        # Check shape of one-hot encoded labels
        self.assertEqual(dataset.y.shape, (100, self.n_classes))
        # Check if one-hot encoding is valid (sum along class dimension should be 1)
        self.assertTrue(torch.all(dataset.y.sum(dim=1) == 1))

    def test_getitem(self):
        dataset = SyntheticDataset(self.X, self.y)
        X_item, y_item = dataset[0]
        self.assertEqual(X_item.shape, (2,))
        self.assertTrue(isinstance(y_item.item(), int))

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        self.data_generator = DataGenerator(random_state=42)
        
    def test_generate_data(self):
        n_samples = 1000
        n_features = 2
        n_classes = 4
        self.data_generator.generate_data(
            n_samples=n_samples, 
            n_features=n_features,
            n_classes=n_classes
        )
        self.assertEqual(self.data_generator.X.shape, (n_samples, n_features))
        self.assertEqual(len(self.data_generator.y), n_samples)
        self.assertEqual(len(np.unique(self.data_generator.y)), n_classes)

    def test_make_outliers(self):

        # Test with outliers accross all classes
        self.data_generator.generate_data(n_samples=1000, n_outliers=100, outlier_scale=2.0, outlier_variance=0.2, outlier_class=None)
        
        outlier_idx = self.data_generator.outlier_idx
        self.assertEqual(len(outlier_idx), 100)

        # check if outliers are split accross all aclasses
        for class_idx in range(self.data_generator.n_classes):
            self.assertGreater(len(np.intersect1d(outlier_idx, np.where(self.data_generator.y == class_idx)[0])), 0)


        # Single class test
        self.data_generator.generate_data(n_samples=1000, n_outliers=100, outlier_scale=2.0, outlier_variance=0.2, outlier_class=1)

        outlier_idx = self.data_generator.outlier_idx
        self.assertEqual(len(outlier_idx), 100)

        # check if outliers are only from class 1
        self.assertTrue(all(self.data_generator.y[outlier_idx] == 1))


    def test_split_data(self):
        self.data_generator.generate_data(n_samples=1000)
        train_idx, val_idx, test_idx = self.data_generator.split_data(0.8, 0.1, 0.1)
        
        # Check sizes
        self.assertAlmostEqual(len(train_idx) / 1000, 0.8, places=1)
        self.assertAlmostEqual(len(val_idx) / 1000, 0.1, places=1)
        self.assertAlmostEqual(len(test_idx) / 1000, 0.1, places=1)
        
        # Check no overlap between sets
        self.assertEqual(len(np.intersect1d(train_idx, val_idx)), 0)
        self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)
        self.assertEqual(len(np.intersect1d(val_idx, test_idx)), 0)

    def test_draw_forget_set_single_class(self):
        n_samples = 1000
        self.data_generator.generate_data(n_samples=n_samples)
        frac_train, frac_val, frac_test = 0.8, 0.1, 0.1
        self.data_generator.split_data(frac_train, frac_val, frac_test)
        n_train, n_test, n_val = int(n_samples * frac_train), int(n_samples * frac_test), int(n_samples * frac_val)
        n_forget = 20

        forget_X, forget_y, retain_X, retain_y, val_X, val_y, test_X, test_y = \
            self.data_generator.draw_forget_set(n_points=n_forget, class_idx=1, ood_ratio=0.5)
        
        # Check sizes
        self.assertEqual(len(forget_X), n_forget)
        self.assertEqual(len(forget_y), n_forget)
        self.assertEqual(len(retain_X), n_train - n_forget)
        self.assertEqual(len(retain_y), n_train - n_forget)
        self.assertEqual(len(val_X), n_val)
        self.assertEqual(len(val_y), n_val)
        self.assertEqual(len(test_X), n_test)
        self.assertEqual(len(test_y), n_test)

        # Check that forget set is from specified class
        self.assertTrue(all(y == 1 for y in forget_y))

    def test_draw_forget_set_multiple_classes(self):
        n_samples = 1000
        self.data_generator.generate_data(n_samples=n_samples)
        frac_train, frac_val, frac_test = 0.8, 0.1, 0.1
        self.data_generator.split_data(frac_train, frac_val, frac_test)
        n_forget = 20
        n_train, n_test, n_val = int(n_samples * frac_train), int(n_samples * frac_test), int(n_samples * frac_val)



        forget_X, forget_y, retain_X, retain_y, val_X, val_y, test_X, test_y = \
            self.data_generator.draw_forget_set(n_points=n_forget, class_idx=1, ood_ratio=0.5)
        
        # Check sizes
        self.assertEqual(len(forget_X), n_forget)
        self.assertEqual(len(forget_y), n_forget)
        self.assertEqual(len(retain_X), n_train - n_forget)
        self.assertEqual(len(retain_y), n_train - n_forget)
        self.assertEqual(len(val_X), n_val)
        self.assertEqual(len(val_y), n_val)
        self.assertEqual(len(test_X), n_test)
        self.assertEqual(len(test_y), n_test)



class TestDataLoaders(unittest.TestCase):
    def setUp(self):
        self.data_generator = DataGenerator(random_state=42)
        self.data_generator.generate_data(n_samples=1000)
        self.data_generator.split_data(0.8, 0.1, 0.1)
        self.data_generator.draw_forget_set(n_points=20, class_idx=1, ood_ratio=0.5)

    def test_dataloader_creation(self):
        dataloaders = create_dataloaders(self.data_generator, batch_size=32)
        
        # Check if all required loaders are present
        self.assertIn('train_full_loader', dataloaders)
        self.assertIn('val_loader', dataloaders)
        self.assertIn('test_loader', dataloaders)
        self.assertIn('train_retain_loader', dataloaders)
        self.assertIn('train_forget_loader', dataloaders)

    def test_batch_shapes(self):
        dataloaders = create_dataloaders(self.data_generator, batch_size=32)
        
        # Check batch shapes
        X_batch, y_batch = next(iter(dataloaders['train_full_loader']))
        self.assertEqual(X_batch.shape[1], 2)  # number of features
        self.assertTrue(X_batch.shape[0] <= 32)  # batch size

if __name__ == '__main__':
    unittest.main() 