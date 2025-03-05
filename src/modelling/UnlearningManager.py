from src.modelling.trainer import Trainer
from src.modelling.neural_network import NeuralNet
from src.modelling.selective_synaptic_dampening import SelectiveSynapticDampening
from src.modelling.scrub import ScrubR
from src.modelling.sisa import SISA
from src.modelling.amnesiac import AmnesiacTrainer
from src.utils.config_loader import Config

import torch
import torch.nn as nn
from typing import Dict, Tuple
import copy


class UnlearningManager:
    def __init__(self, config: Config, device: str = "cpu", wandb=None):
        self.config = config
        self.device = device
        self.unlearn_type = None
        self.wandb = wandb

    def apply_unlearning(self, unlearn_type: str, dataloaders: Dict, config: Config) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """
        Applies the specified unlearning method and returns the unlearned, retrained, and original models.
        
        Args:
            unlearn_type: Type of unlearning to apply
            dataloaders: Dictionary containing the data loaders
            config: Configuration object containing all parameters
        """
        self.unlearn_type = unlearn_type

        if self.unlearn_type == "sisa":
            return self._apply_unlearning_sisa(dataloaders, config.model.n_features, config.model.n_classes, config.model.n_epochs, self.device)
        elif self.unlearn_type == "amnesiac":
            return self._apply_unlearning_amnesiac(dataloaders, config.model.n_features, config.model.n_classes, config.model.n_epochs, self.device)
        elif self.unlearn_type == "sae":
            return self._apply_unlearning_sae(dataloaders, config.model.n_features, config.model.n_classes, config.model.n_epochs, self.device)
        elif self.unlearn_type == "scrub+r" or self.unlearn_type == "ssd":
            return self._apply_unlearning(dataloaders, config.model.n_features, config.model.n_classes, config.model.n_epochs, self.device)
        else:
            raise ValueError(f"Unknown unlearning type: {self.unlearn_type}")

    def _train_standard_model(self, 
                              dataloader, 
                              val_dataloader, 
                              n_features, 
                              n_classes, 
                              n_epochs, 
                              dataset_name,
                              device="cpu"):
        """
        Trains and returns a NeuralNet model using the provided dataloader.
        """
        model = NeuralNet(n_features, n_classes)
        trainer = Trainer(model, 
                        train_dataloader=dataloader, 
                        val_dataloader=val_dataloader, 
                        n_epochs=n_epochs, 
                        device=device, 
                        disable_tqdm=True,
                        wandb=self.wandb,
                        dataset_name=dataset_name)
        trainer.train()
        trainer.eval()
        return model

    def _apply_unlearning(self, dataloaders, n_features, n_classes, n_epochs, device="cpu"):
        """
        Applies either Scrub+R or SSD to the given data. Returns the unlearned model, retrained model, and original model.
        """
        # 1) Train model on full dataset
        unlearned_model = self._train_standard_model(dataloader=dataloaders["train_full_loader"], 
                                    val_dataloader=dataloaders["val_loader"], 
                                    n_features=n_features, 
                                    n_classes=n_classes, 
                                    n_epochs=n_epochs, 
                                    dataset_name="original",
                                    device=device)

        # 2) Train retrained model on retain dataset
        retrained_model = self._train_standard_model(dataloader=dataloaders["train_retain_loader"], 
                                    val_dataloader=dataloaders["val_loader"], 
                                    n_features=n_features, 
                                    n_classes=n_classes, 
                                    n_epochs=n_epochs, 
                                    dataset_name="retrained",
                                    device=device)

        # 3) Copy the original model (before unlearning)
        original_model = copy.deepcopy(unlearned_model)
        original_model.model_type = "pre_forget"
        unlearned_model.model_type = "post_forget"

        # 4) Perform unlearning
        if self.unlearn_type == "scrub+r":
            alpha = 1.0
            gamma = 1.0
            scrub = ScrubR(unlearned_model, original_model, alpha=alpha, gamma=gamma)
            scrub(
                dataloaders["train_retain_loader"],
                dataloaders["train_forget_loader"],
                dataloaders["val_loader"],
                n_rounds=6,
            )
        elif self.unlearn_type == "ssd":
            alpha = 7.5
            _lambda = 0.5
            criterion = nn.CrossEntropyLoss()
            ssd = SelectiveSynapticDampening(unlearned_model, criterion, alpha=alpha, _lambda=_lambda)
            ssd(
                full_dataloader=dataloaders["train_full_loader"],
                forget_dataloader=dataloaders["train_forget_loader"],
            )

        return unlearned_model, retrained_model, original_model

    def _apply_unlearning_sisa(self, dataloaders, n_features, n_classes, n_epochs, device="cpu"):
        """
        Applies SISA unlearning. Returns the unlearned model, retrained model, and original model.
        """
        # Unlearned model
        unlearned_model = SISA(
            dataloader=dataloaders["train_full_loader"],
            n_shards=10,
            n_slices=10,
            n_features=n_features,
            n_classes=n_classes,
            n_epochs=n_epochs,
            save_dir="./experiments/checkpoints/SISA/original_model",
            disable_tqdm=True
        )
        unlearned_shards_dict = unlearned_model.process_data()
        original_shards_dict = unlearned_shards_dict  # in case you need it
        unlearned_model.train_all_models()

        # Retrained model
        retrained_model = SISA(
            dataloader=dataloaders["train_retain_loader"],
            n_shards=10,
            n_slices=10,
            n_features=n_features,
            n_classes=n_classes,
            n_epochs=n_epochs,
            save_dir="./experiments/checkpoints/SISA/retrained_model",
            disable_tqdm=True
        )
        retrained_model.process_data()
        retrained_model.train_all_models()

        # Original model (copy before unlearning)
        original_model = copy.deepcopy(unlearned_model)
        original_model.model_type = "pre_forget"
        unlearned_model.model_type = "post_forget"

        # Forget the datapoints
        unlearned_model.forget_datapoints(datapoint_idxs=dataloaders["forget_idx_in_train"])

        return unlearned_model, retrained_model, original_model

    def _apply_unlearning_amnesiac(self, dataloaders, n_features, n_classes, n_epochs, device="cpu"):
        """
        Applies the amnesiac unlearning method.
        Returns the unlearned model, retrained model, and original model.
        """
        unlearned_model = NeuralNet(n_features, n_classes)
        trainer = AmnesiacTrainer(unlearned_model, 
                                  lr=3e-3, 
                                  device=device, 
                                  cache_gradients=True, 
                                  disable_tqdm=True, 
                                  wandb=self.wandb,
                                  dataset_name="original")

        # Train on full data, storing gradients for sensitive batches
        indices_to_forget = dataloaders["forget_idx_in_train"]
        trainer.train(
            dataloaders["train_full_loader"], 
            epochs=n_epochs,
            indices_to_forget=indices_to_forget,
            save_accuracy_to_file=False
        )

        # Retrained model
        retrained_model = NeuralNet(n_features, n_classes)
        retrained_trainer = AmnesiacTrainer(retrained_model, 
                                            lr=3e-3, 
                                            device=device, 
                                            cache_gradients=True, 
                                            disable_tqdm=True,
                                            dataset_name="retrained")
        retrained_trainer.train(dataloaders["train_retain_loader"], repair=True)

        # Original model
        original_model = copy.deepcopy(unlearned_model)
        original_model.model_type = "pre_forget"
        unlearned_model.model_type = "post_forget"

        # Forget all sensitive batches
        trainer.forget(indices_to_forget=None)

        repair_epochs = int(n_epochs * 0.2)

        # Repair phase
        trainer.train(dataloaders["train_full_loader"], epochs=repair_epochs, save_accuracy_to_file=False, repair=True)

        return unlearned_model, retrained_model, original_model
