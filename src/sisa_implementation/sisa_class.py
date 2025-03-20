from src.data_utils.synthetic_data import SyntheticDataset
from torch.utils.data import DataLoader
from pydantic import BaseModel
from src.models.neural_network import NeuralNet
from src.trainers.base_trainer import BaseTrainer
import numpy as np
import os
from uuid import uuid4
from tqdm import tqdm
import torch
import torch.optim as optim
from src.experiment_logger import create_logger

class Shard(BaseModel):
    shard_id: int
    shard_indices: list[int]
    slices: list[list[int]] = []

class ShardsDict(BaseModel):
    shards: dict[str, Shard]

class SISA:
    def __init__(self, dataloader: DataLoader,
                 n_shards: int = 10, n_slices: int = 10,
                 n_features: int = 2, n_classes: int = 2,
                 n_epochs: int = 10,
                 save_dir: str = None,
                 disable_tqdm: bool = True):
        
        self.disable_tqdm = disable_tqdm
        self.experiment_id = str(uuid4())
        if save_dir is None:
            self.save_dir = f'./src/sisa_implementation/checkpoints/SISA'
        else:
            self.save_dir = save_dir
        self.root_save_dir = self.save_dir
        # We add uuid to the save_dir
        self.save_dir = f"{self.save_dir}/{self.experiment_id}"
        
        self.dataloader = dataloader

        self.dataset = self.dataloader.dataset if self.dataloader is not None else None

        self.shard_models_path = self.save_dir
        os.makedirs(self.shard_models_path, exist_ok=True)
        self.n_classes = n_classes
        self.n_features = n_features

        self.shards_dict = ShardsDict(shards={})

        self.n_shards = n_shards
        self.n_slices = n_slices
        self.shard_size = len(self.dataset.X) // self.n_shards
        self.slice_size = self.shard_size // self.n_slices

        self.n_epochs = n_epochs

        self.model = NeuralNet(M=self.n_features, n_classes=self.n_classes)

        # We need this to be able to call the trainer.train() method.
        self.trainer = BaseTrainer(
            model=self.model, 
            optimizer=None,
            train_dataloader=None,
            val_dataloader=None,
            logger=None,
            device="cpu",
            n_epochs=None,
            disable_tqdm=self.disable_tqdm,
            do_early_stopping=False)

        """
        We could save all the shards in files, but we want to save them in memory for now.
        Thus, we create a dictionary on the form: 
        shards_dict = {
            shard_id: {
                "shard": indecies of the original data that are in the shard,
                "slices": [indecies of the original data that are in each slice]
            }
        }
        """
    
    def copy(self):
        """
        We copy the models from the src_dir to the dst_dir.
        """
        import copy, shutil
        copy_dir = self.root_save_dir + '/' + str(uuid4())
        new_model = copy.deepcopy(self)
        new_model.save_dir = copy_dir
        shutil.copytree(self.save_dir, copy_dir)
        return new_model
       
    def shard_data(self):
        """
        We shard the data into n_shards shards.
        We create a Shard object for each shard and add it to the shards_dict.
        """
        # Shuffle the data
        indices = np.arange(len(self.dataset.X))
        np.random.shuffle(indices)
        splits = np.array_split(indices, self.n_shards)
        for shard_id, shard_indices in enumerate(splits):
            os.makedirs(f"{self.shard_models_path}/shard_{shard_id}", exist_ok=True)
            self.shards_dict.shards[f"shard_{shard_id}"] = Shard(shard_id=shard_id, shard_indices=shard_indices.tolist())

    def slice_shards(self):
        """
        We slice the shards into n_slices slices.
        We create a Slice object for each slice and add it to the Shard object.
        """
        for shard in self.shards_dict.shards.values():
            shard_indices = shard.shard_indices
            splits = np.array_split(shard_indices, self.n_slices)
            for slice_id, slice_indices in enumerate(splits):
                os.makedirs(f"{self.shard_models_path}/shard_{shard.shard_id}/slice_{slice_id}", exist_ok=True)
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

        assert len(self.shards_dict.shards) == self.n_shards
        assert len(self.shards_dict.shards['shard_0'].slices) == self.n_slices
        # from pprint import pprint; pprint(self.shards_dict.model_dump())
        return self.shards_dict
    
    def save_model(self, model: NeuralNet, shard_id: int, slice_id: int):
        """
        We save the models for a shard and slice.
        """
        with open(f"{self.shard_models_path}/shard_{shard_id}/slice_{slice_id}/{slice_id}.pth", "wb") as f:
            torch.save(model.state_dict(), f)

    def load_model(self, shard_id: int, slice_id: int):
        model = NeuralNet(M=self.n_features, n_classes=self.n_classes)
        with open(f"{self.shard_models_path}/shard_{shard_id}/slice_{slice_id}/{slice_id}.pth", "rb") as f:
            model.load_state_dict(torch.load(f))
        return model
    
    def remove_model(self, shard_id: int, slice_id: int):
        os.remove(f"{self.shard_models_path}/shard_{shard_id}/slice_{slice_id}/{slice_id}.pth")

    def dynamic_epochs(self, slice_id):
        return int((2*slice_id) / (self.n_slices + 1) * self.n_epochs)

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
            model (NeuralNet): The trained model.
        """

        model = NeuralNet(M=self.n_features, n_classes=self.n_classes) # new model

        shard = self.shards_dict.shards[f"shard_{shard_id}"] # This might be unclear to some
        shard_slices = shard.slices[start_slice:] # We only train on the slices that are left

        # If we start from a slice, we check whether we have a trained model from the previous slice ready.
        # Otherwise we send an error.
        if start_slice > 0:
            model = self.load_model(shard_id, start_slice-1)
            print(f"Model found for slice {start_slice-1} of shard {shard_id}, rewinding model and re-training")
        
        # Here we incrementally increase the amount of slices we train on.
        # M_k,1 uses 1 slice, M_k,2 uses 1:2 slices, ..., M_k,k uses 1:k slices.
        for slice_id, _ in tqdm(enumerate(shard_slices),disable=self.disable_tqdm):
            # Below looks wierd because we need to handle different slice sizes.
            # We flatten the slices and concatenate them.
            slice_indices = np.concatenate([np.asarray(slice_arr).flatten() for slice_arr in shard_slices[:slice_id+1]])
            n_epochs = self.dynamic_epochs(slice_id)
            
            slice_id = start_slice + slice_id
            
            slice_data = self.dataset.X[slice_indices]
            slice_labels = self.dataset.y[slice_indices]
            # print(f"training on slices {slice_indices}")
            # # onehot encode the labels
            # slice_labels = torch.nn.functional.one_hot(slice_labels, num_classes=self.n_classes)
            # slice_labels = slice_labels.to(torch.float32)
            
            slice_dataset = torch.utils.data.TensorDataset(slice_data, slice_labels)
            assert len(slice_dataset) > 0, f"Slice dataset is empty for shard {shard_id} slice {slice_id}"
            # Set dataset name
            slice_dataset.name = f'shard_{shard_id}_slice_{slice_id}'
            self.trainer.train_dataloader = DataLoader(slice_dataset, batch_size=32, shuffle=True)
            self.trainer.val_dataloader = DataLoader(slice_dataset, batch_size=32, shuffle=False) # Just for making it work with logging
            self.trainer.model = model
            self.trainer.optimizer = optim.Adam(model.parameters(), lr=0.001)
            # self.trainer.logger = create_logger(project_name='sisa', experiment_name=f'shard_{shard_id}_slice_{start_slice}')
            
            self.trainer.n_epochs = n_epochs
            self.trainer.train()

            self.save_model(model, shard_id, slice_id)

        # print(f"\nFinished training models on shard {shard_id}")
        return model

    def train_client_models(self):    
        for shard_id in range(self.n_shards):
            self.train_model_on_shard(shard_id)

        return

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
        for idx, shard_id in tqdm(enumerate(range(self.n_shards)), disable=self.disable_tqdm):
            last_model = self.load_model(shard_id, self.n_slices - 1) # Assumes that there are models for all slices
            # Modify your NeuralNetwork class to return probabilities instead of argmax
            probs = last_model.inference(X)  # This should return softmax outputs
            shard_outputs[idx] = probs['probabilities'].detach().numpy()

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

    def inference(self, x: torch.tensor):
        """
        @param x: A torch tensor of shape (batch_size x n_features)
        returns: A tensor of size (batch_size x n_classes) with the averaged logits of all client models for x.
        """
        
        all_model_logits = []
        for idx, shard_id in tqdm(enumerate(range(self.n_shards)),disable=self.disable_tqdm):
            last_model = self.load_model(shard_id, self.n_slices - 1)

            # Modify your NeuralNetwork class to return probabilities instead of argmax
            with torch.no_grad():
                logits = last_model(x)['logits']  # This should return softmax outputs
            all_model_logits.append(logits)
        
        all_model_logits = torch.stack(all_model_logits)

        return {'logits': all_model_logits.mean(dim=0)}
        
    def find_slice_for_datapoint(self, datapoint_idx: int):
        """
        We find the slice that contains the datapoint.
        """
        for shard in self.shards_dict.shards.values():
            for slice_idx, slice_indices in enumerate(shard.slices):
                if datapoint_idx in slice_indices:
                    # print(f"Datapoint {datapoint_idx} found in shard {shard.shard_id} slice {slice_idx}")
                    return shard.shard_id, slice_idx
        return None, None

    def forget_datapoint(self, shard_id: int, slice_idx: int, datapoint_idx: int):
        """
        We forget a datapoint by finding the slice that contains the datapoint,
        and then 'rewinding' the model to the previous slice.
        """
        if shard_id is None or slice_idx is None:
            print(f"shard_id: {shard_id}, slice_idx: {slice_idx} datapoint_idx: {datapoint_idx} not found")
            return

        # We remove the datapoint from the slice
        slice_forget_point_index = self.shards_dict.shards[f"shard_{shard_id}"].slices[slice_idx].index(datapoint_idx)
        self.shards_dict.shards[f"shard_{shard_id}"].slices[slice_idx].pop(slice_forget_point_index)
            
        # We make sure the datapoints is has been removed
        assert datapoint_idx not in self.shards_dict.shards[f"shard_{shard_id}"].slices[slice_idx], f"Datapoint {datapoint_idx} is still in slice {slice_idx} of shard {shard_id}"

        return

    def forget_datapoints(self, datapoint_idxs: list[int]):
        """
        We forget a list of datapoints.
        We make it batches such that we dont do unecessary unlearning. For example, if we get 100 datapoints to forget,
        and 10 of them are in shard 0, we only unlearn shard 0 once.
        
        Args:
            datapoint_idxs (list[int]): The indices of the datapoints to forget.
        """

        deserved_forget_batches = {} # Containing the earliest slice to forget for each shard.

        for datapoint_idx in datapoint_idxs:
            shard_id, slice_idx = self.find_slice_for_datapoint(datapoint_idx)
            if shard_id not in deserved_forget_batches or slice_idx < deserved_forget_batches[shard_id]:
                deserved_forget_batches[shard_id] = slice_idx
            
            self.forget_datapoint(shard_id, slice_idx, datapoint_idx)

        print(f"Forgetting {len(datapoint_idxs)} datapoints in {deserved_forget_batches}")
        # We remove the models for the slice we just forgot and all slices after it.
        for shard_id, slice_idx in deserved_forget_batches.items():
            for i in range(slice_idx, len(self.shards_dict.shards[f"shard_{shard_id}"].slices)):
                # print(f"Removing model for slice {i} from shard {shard_id}")
                self.remove_model(shard_id, i)

        for shard_id, slice_idx in deserved_forget_batches.items():
            print(f"Retraining shard {shard_id} from slice {slice_idx}")
            self.train_model_on_shard(shard_id, start_slice=slice_idx)

        return

if __name__ == "__main__":
    n_features = 600
    n_classes = 2
    n_epochs = 30
    n_shards = 8
    n_slices = 8
    # data_generator = DataGenerator(random_state=42)
    # data_generator.generate_data(n_samples=500, n_features=n_features, 
    #                          n_informative=2, n_redundant=0, n_classes=n_classes, n_outliers=10)
    # data_generator.split_data()
    # data_generator.draw_forget_set(n_points=10, class_idx=1, ood_ratio=0.5)
    # dataloaders = create_dataloaders(data_generator, batch_size=32)
    
    # train_loader = dataloaders["train_full_loader"]
    # val_loader = dataloaders["val_loader"]
    
    from src.sisa_implementation.sisa_dataloader import load
    X_train, y_train = load(indices=range(1000), category='train')
    X_test, y_test = load(indices=range(1000), category='test')

    # convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(SyntheticDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(SyntheticDataset(X_test, y_test), batch_size=32, shuffle=False)

    sisa = SISA(train_loader, n_classes=n_classes, 
                n_features=n_features, n_epochs=n_epochs, 
                n_shards=n_shards, n_slices=n_slices)
    shards_dict = sisa.process_data()

    sisa.train_client_models()

    # print("Accuracy pre-forget: ", sisa.evaluate(val_loader.dataset.X, val_loader.dataset.y))

    # Forget 10 random datapoints
    datapoint_idxs = np.random.choice(range(len(X_test)), size=10, replace=False)
    sisa.forget_datapoints(datapoint_idxs)



