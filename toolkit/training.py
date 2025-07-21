"""
Advanced training utilities for SNN models
Includes early stopping, learning rate scheduling, and model selection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
import copy
from abc import ABC, abstractmethod

from .logging_utils import MetricsLogger, TrainingMetrics
from .config import SNNConfig


class EarlyStopping:
    """Early stopping implementation to prevent overfitting"""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.best_model_state = None
        self.early_stop = False
    
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """
        Check if training should stop early
        
        Args:
            val_score: Current validation score (higher is better)
            model: PyTorch model
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self.restore_checkpoint(model)
        else:
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, model: nn.Module):
        """Save model state"""
        if self.restore_best_weights:
            self.best_model_state = copy.deepcopy(model.state_dict())
    
    def restore_checkpoint(self, model: nn.Module):
        """Restore best model state"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print('Restored best model weights')


class LRScheduler(ABC):
    """Abstract base class for learning rate schedulers"""
    
    @abstractmethod
    def get_lr(self, epoch: int, current_lr: float) -> float:
        pass


class StepLRScheduler(LRScheduler):
    """Step learning rate scheduler"""
    
    def __init__(self, step_size: int, gamma: float = 0.1):
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self, epoch: int, current_lr: float) -> float:
        if epoch > 0 and epoch % self.step_size == 0:
            return current_lr * self.gamma
        return current_lr


class CosineAnnealingLRScheduler(LRScheduler):
    """Cosine annealing learning rate scheduler"""
    
    def __init__(self, T_max: int, eta_min: float = 0):
        self.T_max = T_max
        self.eta_min = eta_min
        self.initial_lr = None
    
    def get_lr(self, epoch: int, current_lr: float) -> float:
        if self.initial_lr is None:
            self.initial_lr = current_lr
        
        return self.eta_min + (self.initial_lr - self.eta_min) * \
               (1 + np.cos(np.pi * epoch / self.T_max)) / 2


class SNNTrainer:
    """Advanced trainer for SNN models with comprehensive features"""
    
    def __init__(
        self,
        model: nn.Module,
        config: SNNConfig,
        device: Optional[torch.device] = None,
        logger: Optional[MetricsLogger] = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger or MetricsLogger()
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # Training components
        self.early_stopping = None
        self.lr_scheduler = None
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Setup early stopping if enabled
        if config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.patience,
                restore_best_weights=True
            )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config"""
        if self.config.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def set_lr_scheduler(self, scheduler: LRScheduler):
        """Set learning rate scheduler"""
        self.lr_scheduler = scheduler
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Reset model state for SNN
            if hasattr(self.model, 'reset_state'):
                self.model.reset_state()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # Handle temporal outputs (take final time step for classification)
            if outputs.dim() > 2:
                outputs = outputs[:, -1, :]
            
            # Compute loss and backpropagate
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Log batch progress
            if batch_idx % 100 == 0:
                self.logger.logger.debug(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: '
                    f'Loss: {loss.item():.4f}'
                )
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float, float]:
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        total_spikes = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Reset model state
                if hasattr(self.model, 'reset_state'):
                    self.model.reset_state()
                
                # Forward pass
                outputs = self.model(data, record_spikes=True)
                
                # Handle temporal outputs
                if outputs.dim() > 2:
                    outputs = outputs[:, -1, :]
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                # Compute accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Count spikes for analysis
                if hasattr(self.model, 'get_spike_counts'):
                    try:
                        spike_counts = self.model.get_spike_counts()
                        total_spikes += sum(counts.sum().item() for counts in spike_counts)
                    except:
                        pass
        
        avg_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        spike_rate = total_spikes / (total * self.model.layers[-1].output_size) if hasattr(self.model, 'layers') else 0.0
        
        return avg_loss, accuracy, spike_rate
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop with all features
        
        Returns:
            Training history dictionary
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        # Log training start
        dataset_info = {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'batch_size': train_loader.batch_size
        }
        self.logger.log_training_start(num_epochs, dataset_info)
        self.logger.log_model_info(self.model, self.config.to_dict())
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Update learning rate
            if self.lr_scheduler:
                new_lr = self.lr_scheduler.get_lr(epoch, self.optimizer.param_groups[0]['lr'])
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch + 1)
            
            # Validation phase
            val_loss, val_acc, spike_rate = self.validate(val_loader)
            
            # Log metrics
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                spike_rate=spike_rate,
                learning_rate=self.optimizer.param_groups[0]['lr']
            )
            self.logger.log_epoch_metrics(metrics)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if self.config.save_checkpoints:
                    self._save_checkpoint(epoch + 1, val_acc)
            
            # Early stopping check
            if self.early_stopping:
                if self.early_stopping(val_acc, self.model):
                    self.logger.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Progress logging
            epoch_time = time.time() - epoch_start_time
            self.logger.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s"
            )
        
        # Log training completion
        self.logger.log_training_end()
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_accuracy': self.val_accuracies
        }
    
    def _save_checkpoint(self, epoch: int, val_acc: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_acc,
            'config': self.config.to_dict()
        }
        
        checkpoint_path = f"snn_checkpoint_epoch_{epoch}_acc_{val_acc:.4f}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.logger.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.logger.info(
            f"Checkpoint loaded: epoch {checkpoint['epoch']}, "
            f"val_acc {checkpoint['val_accuracy']:.4f}"
        )
        
        return checkpoint


class ModelSelection:
    """Model selection utilities for comparing different SNN architectures"""
    
    def __init__(self, metric: str = 'val_accuracy'):
        self.metric = metric
        self.results = []
    
    def evaluate_model(
        self,
        model_name: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: SNNConfig,
        num_epochs: int = 5
    ) -> Dict[str, Any]:
        """Evaluate a single model configuration"""
        
        # Create trainer
        trainer = SNNTrainer(model, config)
        
        # Train model
        history = trainer.train(train_loader, val_loader, num_epochs)
        
        # Extract final metrics
        final_val_acc = history['val_accuracy'][-1]
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        
        # Calculate additional metrics
        max_val_acc = max(history['val_accuracy'])
        convergence_epoch = history['val_accuracy'].index(max_val_acc) + 1
        
        result = {
            'model_name': model_name,
            'final_val_accuracy': final_val_acc,
            'max_val_accuracy': max_val_acc,
            'convergence_epoch': convergence_epoch,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'config': config.to_dict(),
            'history': history
        }
        
        self.results.append(result)
        return result
    
    def get_best_model(self) -> Dict[str, Any]:
        """Get the best performing model based on the metric"""
        if not self.results:
            raise ValueError("No models have been evaluated yet")
        
        best_result = max(self.results, key=lambda x: x[self.metric])
        return best_result
    
    def compare_models(self) -> List[Dict[str, Any]]:
        """Get sorted comparison of all models"""
        return sorted(self.results, key=lambda x: x[self.metric], reverse=True)
    
    def save_results(self, filepath: str):
        """Save comparison results to file"""
        import json
        
        results_data = {
            'metric': self.metric,
            'results': self.results,
            'best_model': self.get_best_model()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Model selection results saved to {filepath}")


# Import time for timing
import time