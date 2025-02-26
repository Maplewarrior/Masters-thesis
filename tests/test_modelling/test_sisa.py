import pytest
import numpy as np
import torch
from src.modelling.sisa import SISA, Shard, ShardsDict
from src.data_utils.synthetic_data import DataGenerator, create_dataloaders

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data_generator = DataGenerator(random_state=42)
    data_generator.generate_data(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2
    )
    data_generator.split_data()
    dataloaders = create_dataloaders(data_generator, batch_size=32)
    return dataloaders["train_full_loader"]

@pytest.fixture
def sisa_instance(sample_data):
    """Create a SISA instance with small configuration for testing"""
    return SISA(
        dataloader=sample_data,
        n_shards=2,
        n_slices=2,
        n_features=2,
        n_classes=2,
        n_epochs=2
    )

@pytest.mark.sisa
def test_sisa_initialization(sisa_instance):
    """Test if SISA is initialized correctly"""
    assert sisa_instance.n_shards == 2
    assert sisa_instance.n_slices == 2
    assert isinstance(sisa_instance.shards_dict, ShardsDict)
    assert len(sisa_instance.shards_dict.shards) == 0

@pytest.mark.sisa
def test_shard_data(sisa_instance):
    """Test if data is sharded correctly"""
    sisa_instance.shard_data()
    assert len(sisa_instance.shards_dict.shards) == 2
    
    # Check if all shards combined contain all data points
    all_indices = []
    for shard in sisa_instance.shards_dict.shards.values():
        all_indices.extend(shard.shard_indices)
    assert len(set(all_indices)) == len(sisa_instance.dataset.X)

@pytest.mark.sisa
def test_slice_shards(sisa_instance):
    """Test if shards are sliced correctly"""
    sisa_instance.shard_data()
    sisa_instance.slice_shards()
    
    for shard in sisa_instance.shards_dict.shards.values():
        assert len(shard.slices) == 2  # n_slices
        
        # Check if all slices combined contain all shard indices
        slice_indices = []
        for slice_data in shard.slices:
            slice_indices.extend(slice_data)
        assert set(slice_indices) == set(shard.shard_indices)

@pytest.mark.sisa
def test_process_data(sisa_instance):
    """Test if process_data combines sharding and slicing correctly"""
    shards_dict = sisa_instance.process_data()
    assert isinstance(shards_dict, ShardsDict)
    assert len(shards_dict.shards) == 2

# @pytest.mark.sisa
# def test_train_model_on_shard(sisa_instance):
#     """Test if model training works for a single shard"""
#     sisa_instance.process_data()
#     shard_models = sisa_instance.train_model_on_shard(0)
    
#     assert f"shard_0" in shard_models
#     assert len(shard_models["shard_0"]) == 2  # n_slices
    
#     # Check if models are different for different slices
#     model_0 = shard_models["shard_0"]["slice_0"]
#     model_1 = shard_models["shard_0"]["slice_1"]
#     assert id(model_0) == id(model_1)  # Same model instance, updated incrementally

# @pytest.mark.sisa
# def test_train_all_models(sisa_instance):
#     """Test if all models are trained correctly"""
#     sisa_instance.process_data()
#     shard_models = sisa_instance.train_all_models()
    
#     assert len(shard_models) == 2  # n_shards
#     for shard_id in range(2):
#         assert f"shard_{shard_id}" in shard_models
#         assert len(shard_models[f"shard_{shard_id}"]) == 2  # n_slices

@pytest.mark.sisa
def test_predict(sisa_instance):
    """Test if prediction works correctly"""
    sisa_instance.process_data()
    sisa_instance.train_all_models()
    
    # Create sample data for prediction
    X_test = np.random.rand(5, 2)
    predictions = sisa_instance.predict(X_test)
    
    assert len(predictions) == 5
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)
    assert all(0 <= pred <= 1 for pred in predictions)  # n_classes - 1

@pytest.mark.sisa
def test_find_slice_for_datapoint(sisa_instance):
    """Test if datapoint location is found correctly"""
    sisa_instance.process_data()
    
    # Get a known datapoint index
    test_idx = sisa_instance.shards_dict.shards["shard_0"].slices[0][0]
    
    shard_id, slice_idx = sisa_instance.find_slice_for_datapoint(test_idx)
    assert isinstance(shard_id, int)
    assert isinstance(slice_idx, int)
    assert shard_id == 0
    assert slice_idx == 0

# @pytest.mark.sisa
# def test_forget_datapoint(sisa_instance):
#     """Test if forgetting works correctly"""
#     sisa_instance.process_data()
#     sisa_instance.train_all_models()
    
#     # Get a known datapoint index to forget
#     test_idx = sisa_instance.shards_dict.shards["shard_0"].slices[0][0]
    
#     # Store the original slice length
#     original_slice_len = len(sisa_instance.shards_dict.shards["shard_0"].slices[0])
    
#     # Forget the datapoint
#     sisa_instance.forget_datapoint(test_idx)

#     # Check if the models are removed
#     assert len(sisa_instance.shard_models["shard_0"]) == 1
    
#     # Check if the datapoint was removed
#     new_slice_len = len(sisa_instance.shards_dict.shards["shard_0"].slices[0])
#     assert new_slice_len == original_slice_len - 1
    
#     # Verify the datapoint is not found anymore
#     shard_id, slice_idx = sisa_instance.find_slice_for_datapoint(test_idx)
#     assert shard_id is None
#     assert slice_idx is None

# @pytest.mark.sisa
# def test_forget_invalid_datapoint(sisa_instance):
#     """Test if forgetting invalid datapoint raises error"""
#     sisa_instance.process_data()
#     with pytest.raises(ValueError):
#         sisa_instance.forget_datapoint(-1)