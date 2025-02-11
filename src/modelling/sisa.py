from src.data_utils.synthetic_data import DataGenerator, create_dataloaders, SyntheticDataset
from torch.utils.data import DataLoader
from pydantic import BaseModel
from src.modelling.neural_network import NeuralNetwork
from src.modelling.sisa_nn import Model
from src.utils.visualise_decision_boundaries import plot_many_decision_boundaries, plot_many_decision_boundaries_pre_post_forget
from pprint import pprint
import numpy as np
import pdb
import os
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
from collections import defaultdict

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

        self.shard_models_path = "./src/sisa/models"
        os.makedirs(self.shard_models_path, exist_ok=True)
        self.n_classes = n_classes
        self.n_features = n_features

        self.shards_dict = ShardsDict(shards={})

        self.shards_with_forgotten_points = []

        self.n_shards = n_shards
        self.n_slices = n_slices
        self.shard_size = len(self.dataset.X) // self.n_shards
        self.slice_size = self.shard_size // self.n_slices

        self.n_epochs = n_epochs
        self.avg_epochs_per_slice = (2 * self.n_slices / (self.n_slices + 1)) * (self.n_epochs / self.n_slices)

        self.pre_forget_plots = []
        self.post_forget_plots = []

        # Add a counter for retraining per shard
        self.retrain_counts = defaultdict(int)  # {shard_id: count}

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
        # Shuffle the data
        indices = np.arange(len(self.dataset.X))
        np.random.shuffle(indices)
        splits = np.array_split(indices, self.n_shards)
        for shard_id, shard_indices in enumerate(splits):
            os.makedirs(f"{self.shard_models_path}/shard_{shard_id}", exist_ok=True)
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

        print("Shape of shards_dict:")
        print("Amount of shards: ", len(self.shards_dict.shards))
        for shard in self.shards_dict.shards.values():
            print(f"Shard {shard.shard_id}: Amount of slices: {len(shard.slices)}")

        return self.shards_dict
    
    def save_model(self, model: NeuralNetwork, shard_id: int, slice_id: int):
        """
        We save the models for a shard and slice.
        """
        with open(f"{self.shard_models_path}/shard_{shard_id}/slice_{slice_id}/{slice_id}.pth", "wb") as f:
            torch.save(model.state_dict(), f)

    def load_model(self, shard_id: int, slice_id: int):
        model = Model(input_shape=(self.n_features,), nb_classes=self.n_classes)
        with open(f"{self.shard_models_path}/shard_{shard_id}/slice_{slice_id}/{slice_id}.pth", "rb") as f:
            model.load_state_dict(torch.load(f))
        return model
    
    def remove_model(self, shard_id: int, slice_id: int):
        os.remove(f"{self.shard_models_path}/shard_{shard_id}/slice_{slice_id}/{slice_id}.pth")

    def save_pre_forget_model(self, model: NeuralNetwork, shard_id: int):
        torch.save(model.state_dict(), f"{self.shard_models_path}/shard_{shard_id}/pre_forget_model.pth")

    def load_pre_forget_model(self, shard_id: int):
        model = Model(input_shape=(self.n_features,), nb_classes=self.n_classes)
        model.load_state_dict(torch.load(f"{self.shard_models_path}/shard_{shard_id}/pre_forget_model.pth"))
        return model

    def save_post_forget_model(self, model: NeuralNetwork, shard_id: int):
        torch.save(model.state_dict(), f"{self.shard_models_path}/shard_{shard_id}/post_forget_model.pth")

    def load_post_forget_model(self, shard_id: int):
        model = Model(input_shape=(self.n_features,), nb_classes=self.n_classes)
        model.load_state_dict(torch.load(f"{self.shard_models_path}/shard_{shard_id}/post_forget_model.pth"))
        return model

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
            model (NeuralNetwork): The trained model.
        """

        # model = NeuralNetwork(self.n_features, self.n_classes) # old model
        model = Model(input_shape=(self.n_features,), nb_classes=self.n_classes) # new model

        shard = self.shards_dict.shards[f"shard_{shard_id}"] # This might be unclear to some
        shard_slices = shard.slices[start_slice:] # We only train on the slices that are left

        # If we start from a slice, we check whether we have a trained model from the previous slice ready.
        # Otherwise we send an error.
        if start_slice > 0:
            model = self.load_model(shard_id, start_slice-1)
            print(f"Model found for slice {start_slice-1} of shard {shard_id}, rewinding model and re-training")
        
        # Here we incrementally increase the amount of slices we train on.
        # M_k,1 uses 1 slice, M_k,2 uses 1:2 slices, ..., M_k,k uses 1:k slices.
        for slice_id, _ in tqdm(enumerate(shard_slices)):
            # Below looks wierd because we need to handle different slice sizes.
            # We flatten the slices and concatenate them.
            slice_indices = np.concatenate([np.asarray(slice_arr).flatten() for slice_arr in shard_slices[:slice_id+1]])
            n_epochs = self.dynamic_epochs(slice_id)
            
            slice_id = start_slice + slice_id

            print(f"\nTraining model on slices {start_slice}:{slice_id} of shard {shard_id}")
            print(f"Dynamic epochs: {n_epochs}")
            
            slice_data = self.dataset.X[slice_indices]
            slice_labels = self.dataset.y[slice_indices]

            model.fit(slice_data, slice_labels, n_epochs=n_epochs)

            self.save_model(model, shard_id, slice_id)

        print(f"\nFinished training models on shard {shard_id}")
        return model

    def train_all_models(self):
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
        for idx, shard_id in tqdm(enumerate(range(self.n_shards))):
            last_model = self.load_model(shard_id, self.n_slices - 1) # Assumes that there are models for all slices
            # Modify your NeuralNetwork class to return probabilities instead of argmax
            probs = last_model.predict_proba(X)  # This should return softmax outputs
            shard_outputs[idx] = probs.detach().numpy()

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
            print(f"Datapoint {datapoint_idx} not found in any shard")
            return
        
        # Increment the retrain counter for this shard
        self.retrain_counts[shard_id] += 1

        # We remove the datapoint from the slice
        slice_forget_point_index = self.shards_dict.shards[f"shard_{shard_id}"].slices[slice_idx].index(datapoint_idx)
        self.shards_dict.shards[f"shard_{shard_id}"].slices[slice_idx].pop(slice_forget_point_index)

        self.shards_with_forgotten_points += [shard_id]

        model = self.load_model(shard_id, self.n_slices - 1)
        self.save_pre_forget_model(model, shard_id)

        # We remove the models for the slice we just forgot and all slices after it.
        for i in range(slice_idx, len(self.shards_dict.shards[f"shard_{shard_id}"].slices)):
            print(f"Removing model for slice {i} from shard {shard_id}")
            # Copy model to pre-forget model
            self.remove_model(shard_id, i)
        
        print(f"Datapoint {datapoint_idx} removed from shard {shard_id} slice {slice_idx}")

        model = self.train_model_on_shard(shard_id, start_slice=slice_idx)

        self.save_post_forget_model(model, shard_id)

        return

    def visualise_decision_boundaries(self, pre_forget: bool = True):
        """
        We visualise the decision boundaries for each shard.
        """
        plot_many_decision_boundaries(self.shard_models, 
                                     self.dataset.X, 
                                     self.dataset.y, 
                                     n_rows=self.n_shards // 2,
                                     affected_shards=self.shards_with_forgotten_points)
        
    def visualise_pre_forget_models_and_post_forget_models(self):
        """
        We visualise the pre-forget models and the post-forget models side by side.
        """
        plot_many_decision_boundaries_pre_post_forget(self.pre_forget_models, 
                                                     self.post_forget_models, 
                                                     self.dataset.X, 
                                                     self.dataset.y,
                                                     affected_shards=self.shards_with_forgotten_points)

    def evaluate_pre_post_forget(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluates and compares model performance before and after forgetting.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Dictionary containing pre and post forget accuracies for each shard
        """
        results = {
            'pre_forget': {},
            'post_forget': {}
        }
        
        # Convert y_test to tensor if it isn't already
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.tensor(y_test)
        
        # We load the pre-forget models
        for shard_id in sorted(self.shards_with_forgotten_points):
            model = self.load_pre_forget_model(shard_id)
            if not isinstance(X_test, torch.Tensor):
                X_test = torch.tensor(X_test, dtype=torch.float32)
            predictions = model.predict(X_test)
            # Convert predictions to same device as y_test if needed
            predictions = torch.tensor(predictions, device=y_test.device)
            accuracy = (predictions == y_test).float().mean().item()
            results['pre_forget'][shard_id] = accuracy

        # We load the post-forget models
        for shard_id in sorted(self.shards_with_forgotten_points):
            model = self.load_post_forget_model(shard_id)

            if not isinstance(X_test, torch.Tensor):
                X_test = torch.tensor(X_test, dtype=torch.float32)
            predictions = model.predict(X_test)
            # Convert predictions to same device as y_test if needed
            predictions = torch.tensor(predictions, device=y_test.device)
            accuracy = (predictions == y_test).float().mean().item()
            results['post_forget'][shard_id] = accuracy
            
        # Calculate aggregate results
        results['pre_forget']['aggregate'] = np.mean(list(results['pre_forget'].values()))
        results['post_forget']['aggregate'] = np.mean(list(results['post_forget'].values()))
        
        return results

    def get_retrain_statistics(self):
        """
        Returns statistics about how many times each shard has been retrained.
        
        Returns:
            dict: Contains statistics about retraining counts
        """
        stats = {
            'per_shard': dict(self.retrain_counts),
            'total_retrains': sum(self.retrain_counts.values()),
            'max_retrains': max(self.retrain_counts.values()) if self.retrain_counts else 0,
            'min_retrains': min(self.retrain_counts.values()) if self.retrain_counts else 0,
            'avg_retrains': sum(self.retrain_counts.values()) / len(self.retrain_counts) if self.retrain_counts else 0
        }
        return stats

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
    
    from src.modelling.sisa_dataloader import load
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

    sisa.train_all_models()

    # Predict on 10 samples
    predictions = sisa.predict(X_test[:100].reshape(100, -1))
    pre_forget_predictions = predictions

    # print("Accuracy pre-forget: ", sisa.evaluate(val_loader.dataset.X, val_loader.dataset.y))

    # Forget 10 random datapoints
    for i in range(100):
        datapoint_idx = np.random.randint(0, len(X_test))
        sisa.forget_datapoint(datapoint_idx)

    # Predict on the same 10 samples again.
    # This is somewhat fyfy, since we are predicting on samples in the same dataset.
    predictions = sisa.predict(X_test[:100].reshape(100, -1))
    post_forget_predictions = predictions

    print("Pre-forget predictions:")
    print(pre_forget_predictions)
    print("Post-forget predictions:")
    print(post_forget_predictions)

    # sisa.visualise_pre_forget_models_and_post_forget_models()

    # print("Accuracy post-forget: ", sisa.evaluate(val_loader.dataset.X, val_loader.dataset.y))

    # Evaluate pre and post forget performance
    evaluation_results = sisa.evaluate_pre_post_forget(X_test, y_test)
    retrain_stats = sisa.get_retrain_statistics()
    
    print("\nAccuracy comparison:")
    print("-" * 70)
    print(f"{'Shard':10} {'Pre-forget':>12} {'Post-forget':>12} {'Diff':>8} {'Retrains':>10}")
    print("-" * 70)
    
    for shard_id in evaluation_results['pre_forget'].keys():
        pre_acc = evaluation_results['pre_forget'][shard_id]
        post_acc = evaluation_results['post_forget'][shard_id]
        diff = post_acc - pre_acc
        diff_symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
        retrains = retrain_stats['per_shard'].get(shard_id, 0)
        print(f"{shard_id:10} {pre_acc:>11.3f}% {post_acc:>11.3f}% {diff:>+7.3f}{diff_symbol} {retrains:>10}")
    
    print("-" * 70)

