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
import time
import psutil
import gc


class UnlearningManager:
    def __init__(self, config: Config, device: str = "cpu", wandb=None, track_performance: bool = False):
        self.config = config
        self.device = device
        self.unlearn_type = None
        self.wandb = wandb
        self.track_performance = track_performance

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
                              device="cpu",
                              track_performance=False):
        """
        Trains and returns a NeuralNet model using the provided dataloader.
        Also logs memory and time usage if track_performance is True.
        
        Args:
            dataloader: Training data loader
            val_dataloader: Validation data loader
            n_features: Number of input features
            n_classes: Number of output classes
            n_epochs: Number of training epochs
            dataset_name: Name of the dataset for logging
            device: Device to train on ("cpu" or "cuda")
            track_performance: Whether to track and log memory and time usage
        """
        track_performance = self.track_performance
        # Start time measurement if tracking performance
        start_time = None
        initial_cpu_mem = None
        initial_gpu_mem = None
        
        if track_performance:
            start_time = time.time()
            
            # Force garbage collection before measuring
            gc.collect()
            
            # Get initial memory usage
            process = psutil.Process()
            initial_cpu_mem = process.memory_info().rss / (1024 ** 2)  # MB
            
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                initial_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        # Create and train model
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
        
        # Log performance metrics if tracking is enabled
        if track_performance:
            # Force garbage collection before measuring again
            gc.collect()
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # End time measurement
            end_time = time.time()
            training_time = end_time - start_time
            
            # Get final memory usage
            final_cpu_mem = process.memory_info().rss / (1024 ** 2)  # MB
            cpu_mem_diff = final_cpu_mem - initial_cpu_mem
            cpu_peak = max(initial_cpu_mem, final_cpu_mem)
            
            # Log metrics
            log_dict = {
                f"performance_tracking/{dataset_name}/training_time_seconds": training_time,
                f"performance_tracking/{dataset_name}/cpu_memory_usage_mb": cpu_mem_diff,
                f"performance_tracking/{dataset_name}/cpu_peak_memory_mb": cpu_peak,
            }
            
            if device == "cuda" and torch.cuda.is_available():
                final_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                gpu_mem_diff = final_gpu_mem - initial_gpu_mem
                
                log_dict.update({
                    f"performance_tracking/{dataset_name}/gpu_memory_usage_mb": gpu_mem_diff,
                    f"performance_tracking/{dataset_name}/peak_gpu_memory_mb": peak_gpu_mem
                })
            
            if self.wandb:
                self.wandb.log(log_dict)
                
            
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
                                    device=device,
                                    track_performance=self.track_performance)

        # 2) Train retrained model on retain dataset
        retrained_model = self._train_standard_model(dataloader=dataloaders["train_retain_loader"], 
                                    val_dataloader=dataloaders["val_loader"], 
                                    n_features=n_features, 
                                    n_classes=n_classes, 
                                    n_epochs=n_epochs, 
                                    dataset_name="retrained",
                                    device=device,
                                    track_performance=self.track_performance)

        # 3) Copy the original model (before unlearning)
        original_model = copy.deepcopy(unlearned_model)
        original_model.model_type = "pre_forget"
        unlearned_model.model_type = "post_forget"

        # 4) Perform unlearning - track performance for this step
        if self.track_performance:
            start_time = time.time()
            gc.collect()
            process = psutil.Process()
            initial_cpu_mem = process.memory_info().rss / (1024 ** 2)  # MB
            
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                initial_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB

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

        # Log performance metrics for unlearning
        if self.track_performance:
            gc.collect()
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            end_time = time.time()
            unlearn_time = end_time - start_time
            
            final_cpu_mem = process.memory_info().rss / (1024 ** 2)  # MB
            cpu_mem_diff = final_cpu_mem - initial_cpu_mem
            cpu_peak = max(initial_cpu_mem, final_cpu_mem)
            
            unlearn_log_dict = {
                "performance_tracking/unlearning/time_seconds": unlearn_time,
                "performance_tracking/unlearning/cpu_memory_usage_mb": cpu_mem_diff,
                "performance_tracking/unlearning/cpu_peak_memory_mb": cpu_peak,
            }
            
            if device == "cuda" and torch.cuda.is_available():
                final_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                gpu_mem_diff = final_gpu_mem - initial_gpu_mem
                
                unlearn_log_dict.update({
                    "performance_tracking/unlearning/gpu_memory_usage_mb": gpu_mem_diff,
                    "performance_tracking/unlearning/peak_gpu_memory_mb": peak_gpu_mem
                })
            
            if self.wandb:
                self.wandb.log(unlearn_log_dict)
                
            # Print metrics
            print(f"Unlearning time: {unlearn_time:.2f} seconds")
            print(f"Unlearning CPU memory usage: {cpu_mem_diff:.2f} MB")
            print(f"Unlearning CPU peak memory: {cpu_peak:.2f} MB")
            if device == "cuda" and torch.cuda.is_available():
                print(f"Unlearning GPU memory usage: {gpu_mem_diff:.2f} MB")
                print(f"Unlearning peak GPU memory: {peak_gpu_mem:.2f} MB")

        return unlearned_model, retrained_model, original_model

    def _apply_unlearning_sisa(self, dataloaders, n_features, n_classes, n_epochs, device="cpu"):
        """
        Applies SISA unlearning. Returns the unlearned model, retrained model, and original model.
        """
        # Start time and memory tracking for original model
        start_time = None
        initial_cpu_mem = None
        initial_gpu_mem = None
        
        if self.track_performance:
            start_time = time.time()
            gc.collect()
            process = psutil.Process()
            initial_cpu_mem = process.memory_info().rss / (1024 ** 2)  # MB
            
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                initial_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
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
        
        # Log performance metrics for original model
        if self.track_performance:
            gc.collect()
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            end_time = time.time()
            training_time = end_time - start_time
            
            final_cpu_mem = process.memory_info().rss / (1024 ** 2)  # MB
            cpu_mem_diff = final_cpu_mem - initial_cpu_mem
            cpu_peak = max(initial_cpu_mem, final_cpu_mem)
            
            log_dict = {
                "performance_tracking/original/training_time_seconds": training_time,
                "performance_tracking/original/cpu_memory_usage_mb": cpu_mem_diff,
                "performance_tracking/original/cpu_peak_memory_mb": cpu_peak,
            }
            
            if device == "cuda" and torch.cuda.is_available():
                final_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                gpu_mem_diff = final_gpu_mem - initial_gpu_mem
                
                log_dict.update({
                    "performance_tracking/original/gpu_memory_usage_mb": gpu_mem_diff,
                    "performance_tracking/original/peak_gpu_memory_mb": peak_gpu_mem
                })
            
            if self.wandb:
                self.wandb.log(log_dict)

        # Start tracking for retrained model
        if self.track_performance:
            retrain_start_time = time.time()
            gc.collect()
            retrain_initial_cpu_mem = process.memory_info().rss / (1024 ** 2)
            
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                retrain_initial_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)

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
        
        # Log performance metrics for retrained model
        if self.track_performance:
            gc.collect()
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            retrain_end_time = time.time()
            retrain_time = retrain_end_time - retrain_start_time
            
            retrain_final_cpu_mem = process.memory_info().rss / (1024 ** 2)
            retrain_cpu_diff = retrain_final_cpu_mem - retrain_initial_cpu_mem
            retrain_cpu_peak = max(retrain_initial_cpu_mem, retrain_final_cpu_mem)
            
            retrain_log_dict = {
                "performance_tracking/retrained/training_time_seconds": retrain_time,
                "performance_tracking/retrained/cpu_memory_usage_mb": retrain_cpu_diff,
                "performance_tracking/retrained/cpu_peak_memory_mb": retrain_cpu_peak,
            }
            
            if device == "cuda" and torch.cuda.is_available():
                retrain_final_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
                retrain_peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                retrain_gpu_diff = retrain_final_gpu_mem - retrain_initial_gpu_mem
                
                retrain_log_dict.update({
                    "performance_tracking/retrained/gpu_memory_usage_mb": retrain_gpu_diff,
                    "performance_tracking/retrained/peak_gpu_memory_mb": retrain_peak_gpu_mem
                })
            
            if self.wandb:
                self.wandb.log(retrain_log_dict)

        # Original model (copy before unlearning)
        original_model = copy.deepcopy(unlearned_model)
        original_model.model_type = "pre_forget"
        unlearned_model.model_type = "post_forget"

        # Start tracking for unlearning
        if self.track_performance:
            unlearn_start_time = time.time()
            gc.collect()
            unlearn_initial_cpu_mem = process.memory_info().rss / (1024 ** 2)
            
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                unlearn_initial_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)

        # Forget the datapoints
        unlearned_model.forget_datapoints(datapoint_idxs=dataloaders["forget_idx_in_train"])
        
        # Log performance metrics for unlearning
        if self.track_performance:
            gc.collect()
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            unlearn_end_time = time.time()
            unlearn_time = unlearn_end_time - unlearn_start_time
            
            unlearn_final_cpu_mem = process.memory_info().rss / (1024 ** 2)
            unlearn_cpu_diff = unlearn_final_cpu_mem - unlearn_initial_cpu_mem
            unlearn_cpu_peak = max(unlearn_initial_cpu_mem, unlearn_final_cpu_mem)
            
            unlearn_log_dict = {
                "performance_tracking/unlearning/time_seconds": unlearn_time,
                "performance_tracking/unlearning/cpu_memory_usage_mb": unlearn_cpu_diff,
                "performance_tracking/unlearning/cpu_peak_memory_mb": unlearn_cpu_peak,
            }
            
            if device == "cuda" and torch.cuda.is_available():
                unlearn_final_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
                unlearn_peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                unlearn_gpu_diff = unlearn_final_gpu_mem - unlearn_initial_gpu_mem
                
                unlearn_log_dict.update({
                    "performance_tracking/unlearning/gpu_memory_usage_mb": unlearn_gpu_diff,
                    "performance_tracking/unlearning/peak_gpu_memory_mb": unlearn_peak_gpu_mem
                })
            
            if self.wandb:
                self.wandb.log(unlearn_log_dict)
                

        return unlearned_model, retrained_model, original_model

    def _apply_unlearning_amnesiac(self, dataloaders, n_features, n_classes, n_epochs, device="cpu"):
        """
        Applies the amnesiac unlearning method.
        Returns the unlearned model, retrained model, and original model.
        """
        # Start time and memory tracking for original model
        start_time = None
        initial_cpu_mem = None
        initial_gpu_mem = None
        
        if self.track_performance:
            start_time = time.time()
            gc.collect()
            process = psutil.Process()
            initial_cpu_mem = process.memory_info().rss / (1024 ** 2)  # MB
            
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                initial_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        # Train original model
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
        
        # Log performance metrics for original model
        if self.track_performance:
            gc.collect()
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            end_time = time.time()
            training_time = end_time - start_time
            
            final_cpu_mem = process.memory_info().rss / (1024 ** 2)  # MB
            cpu_mem_diff = final_cpu_mem - initial_cpu_mem
            cpu_peak = max(initial_cpu_mem, final_cpu_mem)
            
            log_dict = {
                "performance_tracking/original/training_time_seconds": training_time,
                "performance_tracking/original/cpu_memory_usage_mb": cpu_mem_diff,
                "performance_tracking/original/cpu_peak_memory_mb": cpu_peak,
            }
            
            if device == "cuda" and torch.cuda.is_available():
                final_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                gpu_mem_diff = final_gpu_mem - initial_gpu_mem
                
                log_dict.update({
                    "performance_tracking/original/gpu_memory_usage_mb": gpu_mem_diff,
                    "performance_tracking/original/peak_gpu_memory_mb": peak_gpu_mem
                })
            
            if self.wandb:
                self.wandb.log(log_dict)

        # Start tracking for retrained model
        if self.track_performance:
            retrain_start_time = time.time()
            gc.collect()
            retrain_initial_cpu_mem = process.memory_info().rss / (1024 ** 2)
            
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                retrain_initial_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)

        # Retrained model
        retrained_model = NeuralNet(n_features, n_classes)
        retrained_trainer = AmnesiacTrainer(retrained_model, 
                                            lr=3e-3, 
                                            device=device, 
                                            cache_gradients=True, 
                                            disable_tqdm=True,
                                            dataset_name="retrained")
        retrained_trainer.train(dataloaders["train_retain_loader"], repair=True)
        
        # Log performance metrics for retrained model
        if self.track_performance:
            gc.collect()
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            retrain_end_time = time.time()
            retrain_time = retrain_end_time - retrain_start_time
            
            retrain_final_cpu_mem = process.memory_info().rss / (1024 ** 2)
            retrain_cpu_diff = retrain_final_cpu_mem - retrain_initial_cpu_mem
            retrain_cpu_peak = max(retrain_initial_cpu_mem, retrain_final_cpu_mem)
            
            retrain_log_dict = {
                "performance_tracking/retrained/training_time_seconds": retrain_time,
                "performance_tracking/retrained/cpu_memory_usage_mb": retrain_cpu_diff,
                "performance_tracking/retrained/cpu_peak_memory_mb": retrain_cpu_peak,
            }
            
            if device == "cuda" and torch.cuda.is_available():
                retrain_final_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
                retrain_peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                retrain_gpu_diff = retrain_final_gpu_mem - retrain_initial_gpu_mem
                
                retrain_log_dict.update({
                    "performance_tracking/retrained/gpu_memory_usage_mb": retrain_gpu_diff,
                    "performance_tracking/retrained/peak_gpu_memory_mb": retrain_peak_gpu_mem
                })
            
            if self.wandb:
                self.wandb.log(retrain_log_dict)

        # Original model
        original_model = copy.deepcopy(unlearned_model)
        original_model.model_type = "pre_forget"
        unlearned_model.model_type = "post_forget"

        # Start tracking for unlearning (forget + repair)
        if self.track_performance:
            unlearn_start_time = time.time()
            gc.collect()
            unlearn_initial_cpu_mem = process.memory_info().rss / (1024 ** 2)
            
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                unlearn_initial_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)

        # Forget all sensitive batches
        trainer.forget(indices_to_forget=None)

        repair_epochs = int(n_epochs * 0.2)

        # Repair phase
        trainer.train(dataloaders["train_retain_loader"], epochs=repair_epochs, save_accuracy_to_file=False, repair=True)
        
        # Log performance metrics for unlearning
        if self.track_performance:
            gc.collect()
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            unlearn_end_time = time.time()
            unlearn_time = unlearn_end_time - unlearn_start_time
            
            unlearn_final_cpu_mem = process.memory_info().rss / (1024 ** 2)
            unlearn_cpu_diff = unlearn_final_cpu_mem - unlearn_initial_cpu_mem
            unlearn_cpu_peak = max(unlearn_initial_cpu_mem, unlearn_final_cpu_mem)
            
            unlearn_log_dict = {
                "performance_tracking/unlearning/time_seconds": unlearn_time,
                "performance_tracking/unlearning/cpu_memory_usage_mb": unlearn_cpu_diff,
                "performance_tracking/unlearning/cpu_peak_memory_mb": unlearn_cpu_peak,
            }
            
            if device == "cuda" and torch.cuda.is_available():
                unlearn_final_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
                unlearn_peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                unlearn_gpu_diff = unlearn_final_gpu_mem - unlearn_initial_gpu_mem
                
                unlearn_log_dict.update({
                    "performance_tracking/unlearning/gpu_memory_usage_mb": unlearn_gpu_diff,
                    "performance_tracking/unlearning/peak_gpu_memory_mb": unlearn_peak_gpu_mem
                })
            
            if self.wandb:
                self.wandb.log(unlearn_log_dict)
                

        return unlearned_model, retrained_model, original_model
