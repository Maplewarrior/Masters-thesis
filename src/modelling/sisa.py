from src.data_utils.synthetic_data import SyntheticDataset, DataGenerator
from pydantic import BaseModel
from neural_network import NeuralNetwork
from pprint import pprint
import numpy as np
import pdb

class Shard(BaseModel):
    shard_indices: list[int]
    slices: list[list[int]]

class Slice(BaseModel):
    slice_id: str
    slice_indices: list[int]

class ShardsDict(BaseModel):
    shards: dict[str, Shard]

class SISA:
    def __init__(self, dataset: SyntheticDataset, 
                 n_shards: int = 10, n_slices: int = 10):
        self.dataset = dataset

        self.shards_dict = ShardsDict(shards={})

        self.n_shards = n_shards
        self.n_slices = n_slices
        self.shard_size = len(self.dataset.X) // self.n_shards
        self.slice_size = self.shard_size // self.n_slices

        """
        We could save all the shards in files, but we want to save them in memory for now.
        Thus, ww create a dictionary on the form: 
        shards_dict = {
            shard_id: {
                "shard": indecies of the original data that are in the shard,
                "slices": [indecies of the original data that are in each slice]
            }
        }
        """
    
    def shard_data(self):
        """
        We shard the data into n_shards shards.
        We create a Shard object for each shard and add it to the shards_dict.
        """
        for i in range(self.n_shards):
            shard_indices = np.random.choice(len(self.dataset.X), self.shard_size, replace=False)
            self.shards_dict.shards[f"shard_{i}"] = Shard(shard_indices=shard_indices, slices=[])

    def slice_shards(self):
        """
        We slice the shards into n_slices slices.
        We create a Slice object for each slice and add it to the Shard object.
        """
        for shard in self.shards_dict.shards.values():
            shard_indices = shard.shard_indices
            for i in range(self.n_slices):
                slice_indices = np.random.choice(shard_indices, self.slice_size, replace=False)
                shard.slices.append(slice_indices)

    def process_data(self):
        """
        We shard the data into n_shards shards and slice each shard into n_slices slices.
        
        Args:
            save_to_file (bool): If True, we save the shards_dict to a file. Defaults to False which means we save the shards_dict in memory.

        Returns:
            shards_dict (ShardsDict): The shards_dict with the shards and slices.
        """

        self.shard_data()
        self.slice_shards()

        return self.shards_dict
    
    def train_model_on_shard(self, model: NeuralNetwork, shard_id: int):
        shard = self.shards_dict.shards[f"shard_{shard_id}"] # This might be unclear to some
        shard_indices = shard.shard_indices
        shard_data = self.dataset.X[shard_indices]
        shard_labels = self.dataset.y[shard_indices]

        model.fit(shard_data, shard_labels)

if __name__ == "__main__":
    data_generator = DataGenerator(random_state=42)
    data_generator.generate_data(n_samples=1000, n_features=2, 
                             n_informative=2, n_redundant=0)
    
    sisa = SISA(data_generator)
    shards_dict = sisa.process_data()

    model = NeuralNetwork(M=2, n_classes=2)
    sisa.train_model_on_shard(model, 0)