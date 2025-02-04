from src.data_utils.synthetic_data import DataGenerator, create_dataloaders
from torch.utils.data import DataLoader
from pydantic import BaseModel
from neural_network import NeuralNetwork
from pprint import pprint
import numpy as np
import pdb
import torch

class Shard(BaseModel):
    shard_id: int
    shard_indices: list[int]
    slices: list[list[int]] = []

class ShardsDict(BaseModel):
    shards: dict[str, Shard]

class SISA:
    def __init__(self, dataloader: DataLoader, 
                 forget_loader: DataLoader = None,
                 n_shards: int = 10, n_slices: int = 10,
                 n_features: int = 2, n_classes: int = 2,
                 n_epochs: int = 10):
        self.dataloader = dataloader
        self.forget_loader = forget_loader

        self.dataset = self.dataloader.dataset if self.dataloader is not None else None
        self.forget_dataset = self.forget_loader.dataset if self.forget_loader is not None else None

        self.shard_models = {} # {shard_id: {slice_id: model}}

        self.n_classes = n_classes
        self.n_features = n_features

        self.shards_dict = ShardsDict(shards={})

        self.n_shards = n_shards
        self.n_slices = n_slices
        self.shard_size = len(self.dataset.X) // self.n_shards
        self.slice_size = self.shard_size // self.n_slices

        self.n_epochs = n_epochs

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
        splits = np.array_split(np.arange(len(self.dataset.X)), self.n_shards)
        for shard_id, shard_indices in enumerate(splits):
            shard = Shard(shard_id=shard_id, shard_indices=shard_indices.tolist())
            self.shards_dict.shards[f"shard_{shard_id}"] = shard

    def slice_shards(self):
        """
        We slice the shards into n_slices slices.
        We create a Slice object for each slice and add it to the Shard object.
        """
        for shard in self.shards_dict.shards.values():
            shard_indices = shard.shard_indices
            splits = np.array_split(shard_indices, self.n_slices)
            for slice_id, slice_indices in enumerate(splits):
                shard.slices.append(slice_indices.tolist())

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
    
    def train_model_on_shard(self, shard_id: int, start_slice: int = 0):
        """
        We train the model on a distinct shard.
        This function loops over all slices for a shard and trains a model on each slice.

        We also implement a dynamic epoch size based on SISA (Eq. 1):
        # e'D = sum(e_i * (iD/R), i=1 to R)
        #     = e 
        #     = (2R/(R+1))e'

        Where R is the number of slices we train on and e' is the number of epochs to train without slicing.
        
        Args:
            shard_id (int): The id of the shard to train on.

        Returns:
            model (NeuralNetwork): The trained model.
        """

        model = NeuralNetwork(self.n_features, self.n_classes)

        shard = self.shards_dict.shards[f"shard_{shard_id}"] # This might be unclear to some
        shard_slices = shard.slices[start_slice:]
        shard_models = {}

        # If we start from a slice, we check whether we have a trained model from the previous slice ready.
        # Otherwise we send an error.
        if start_slice > 0:
            if f"slice_{start_slice-1}" in self.shard_models[f"shard_{shard_id}"]:
                model = self.shard_models[f"shard_{shard_id}"][f"slice_{start_slice-1}"]
                print(f"Model found for slice {start_slice-1} of shard {shard_id}, rewinding model and re-training")
            else:
                raise ValueError(f"No trained model found for slice {start_slice-1}")
        
        # Here we incrementally increase the amount of slices we train on.
        # M_k,1 uses 1 slice, M_k,2 uses 1:2 slices, ..., M_k,k uses 1:k slices.
        for slice_id, _ in enumerate(shard_slices):
            slice_indices = np.array(shard_slices[:slice_id+1]).flatten()
            amount_of_slices = slice_id + 1
            n_epochs = int((2 * amount_of_slices / (amount_of_slices + 1)) * self.n_epochs)
            
            print(f"\nTraining model on slices 1:{amount_of_slices} of shard {shard_id}")
            print(f"Dynamic epochs: {n_epochs}")
            
            slice_data = self.dataset.X[slice_indices]
            slice_labels = self.dataset.y[slice_indices]

            # Convert labels to one-hot encoding
            slice_labels = torch.tensor(slice_labels, dtype=torch.long)

            # Convert shard_data and shard_labels to tensors
            slice_data = torch.tensor(slice_data, dtype=torch.float32)

            model.fit(slice_data, slice_labels, n_epochs=n_epochs)

            shard_models[f"slice_{slice_id}"] = model

        self.shard_models[f"shard_{shard_id}"] = shard_models

        print(f"\nFinished training models on shard {shard_id}")
        return self.shard_models

    def train_all_models(self):
        for shard_id in range(self.n_shards):
            self.train_model_on_shard(shard_id)

        return self.shard_models

    def predict(self, X: np.ndarray):
        """
        We predict the class probabilities using weighted voting across shards.
        Instead of using hard predictions, we use the probability outputs
        from each model's forward pass.

        Args:
            X (np.ndarray): The data to predict.

        Returns:
            predictions (np.ndarray): The predicted classes.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        # Store probability outputs from each shard
        # Shape: (n_shards, n_samples, n_classes)
        shard_outputs = np.zeros((self.n_shards, len(X), self.n_classes))

        # Get probability outputs from each shard's final model
        for idx, shard_models in enumerate(self.shard_models.values()):
            last_model = list(shard_models.values())[-1]
            # Modify your NeuralNetwork class to return probabilities instead of argmax
            probs = last_model.predict_proba(X)  # This should return softmax outputs
            shard_outputs[idx] = probs

        # Use uniform weights for now
        weights = np.ones(self.n_shards) / self.n_shards

        # Weighted voting using probability outputs
        # This multiplies each shard's probabilities by its weight and sums them
        # https://github.com/cleverhans-lab/machine-unlearning/blob/master/aggregation.py#L65
        weighted_probs = np.tensordot(
            weights.reshape(1, weights.shape[0]), 
            shard_outputs, 
            axes=1
        )
        
        # Get the class with highest probability
        predictions = np.argmax(weighted_probs, axis=2).reshape((len(X),))

        return predictions

    def find_slice_for_datapoint(self, datapoint_idx: int):
        """
        We find the slice that contains the datapoint.
        """
        for shard in self.shards_dict.shards.values():
            for slice_idx, slice_indices in enumerate(shard.slices):
                if datapoint_idx in slice_indices:
                    print(f"Datapoint {datapoint_idx} found in shard {shard.shard_id} slice {slice_idx}")
                    return shard.shard_id, slice_idx
        return None, None

    def forget_datapoint(self, datapoint_idx: int):
        """
        We forget a datapoint by finding the slice that contains the datapoint,
        and then 'rewinding' the model to the previous slice.
        """
        shard_id, slice_idx = self.find_slice_for_datapoint(datapoint_idx)
        if shard_id is None:
            raise ValueError(f"Datapoint {datapoint_idx} not found in any shard")
        
        self.train_model_on_shard(shard_id, start_slice=slice_idx-1)

        return self.shard_models

if __name__ == "__main__":
    n_features = 10
    n_classes = 4
    data_generator = DataGenerator(random_state=42)
    data_generator.generate_data(n_samples=1000, n_features=n_features, 
                             n_informative=2, n_redundant=0, n_classes=n_classes)
    data_generator.split_data()
    data_generator.draw_forget_set(n_points=20, class_idx=1, ood_ratio=0.5)
    dataloaders = create_dataloaders(data_generator, batch_size=32)
    
    sisa = SISA(dataloaders["train_full_loader"], n_classes=n_classes, n_features=n_features)
    shards_dict = sisa.process_data()

    sisa.train_all_models()

    pre_forget_predictions = None
    post_forget_predictions = None

    # Predict on 10 samples
    predictions = sisa.predict(data_generator.X[:10].reshape(10, -1))
    pre_forget_predictions = predictions

    # Forget a datapoint
    sisa.forget_datapoint(100)

    # Predict on the same 10 samples again
    predictions = sisa.predict(data_generator.X[:10].reshape(10, -1))
    post_forget_predictions = predictions

    print(pre_forget_predictions)
    print(post_forget_predictions)
