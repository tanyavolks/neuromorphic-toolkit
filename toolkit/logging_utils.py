"""
Logging and metrics utilities for SNN Toolkit
"""

import logging
import time
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

try:
    import torch
except ImportError:
    torch = None


@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    spike_rate: float
    learning_rate: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'train_accuracy': self.train_accuracy,
            'val_loss': self.val_loss,
            'val_accuracy': self.val_accuracy,
            'spike_rate': self.spike_rate,
            'learning_rate': self.learning_rate,
            'timestamp': self.timestamp
        }


class MetricsLogger:
    """Centralized metrics logging and tracking"""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"snn_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history: List[TrainingMetrics] = []
        self.best_metrics: Optional[TrainingMetrics] = None
        self.start_time = time.time()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Metrics file path
        self.metrics_file = os.path.join(log_dir, f"{self.experiment_name}_metrics.json")
        
        self.logger.info(f"Metrics logging initialized for experiment: {self.experiment_name}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger(f"snn_toolkit_{self.experiment_name}")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # File handler
        log_file = os.path.join(self.log_dir, f"{self.experiment_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_epoch_metrics(self, metrics: TrainingMetrics):
        """Log metrics for an epoch"""
        self.metrics_history.append(metrics)
        
        # Update best metrics
        if self.best_metrics is None or metrics.val_accuracy > self.best_metrics.val_accuracy:
            self.best_metrics = metrics
            self.logger.info(f"New best validation accuracy: {metrics.val_accuracy:.4f}")
        
        # Log to console and file
        self.logger.info(
            f"Epoch {metrics.epoch}: "
            f"Train Loss: {metrics.train_loss:.4f}, "
            f"Train Acc: {metrics.train_accuracy:.4f}, "
            f"Val Loss: {metrics.val_loss:.4f}, "
            f"Val Acc: {metrics.val_accuracy:.4f}, "
            f"Spike Rate: {metrics.spike_rate:.4f}"
        )
        
        # Save metrics to file
        self._save_metrics()
    
    def log_model_info(self, model: Any, config: Dict[str, Any]):
        """Log model architecture and configuration"""
        if torch is not None and hasattr(model, 'parameters'):
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.logger.info(f"Model parameters: {num_params:,} total, {trainable_params:,} trainable")
        
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    def log_training_start(self, num_epochs: int, dataset_info: Dict[str, Any]):
        """Log training start information"""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Dataset info: {dataset_info}")
        self.start_time = time.time()
    
    def log_training_end(self):
        """Log training completion information"""
        total_time = time.time() - self.start_time
        
        if self.best_metrics:
            self.logger.info(
                f"Training completed in {total_time:.2f}s. "
                f"Best validation accuracy: {self.best_metrics.val_accuracy:.4f} "
                f"at epoch {self.best_metrics.epoch}"
            )
        else:
            self.logger.info(f"Training completed in {total_time:.2f}s")
    
    def log_spike_analysis(self, spike_metrics: Dict[str, float]):
        """Log spike analysis metrics"""
        self.logger.info(f"Spike analysis: {spike_metrics}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of training"""
        if not self.metrics_history:
            return {}
        
        val_accuracies = [m.val_accuracy for m in self.metrics_history]
        train_losses = [m.train_loss for m in self.metrics_history]
        spike_rates = [m.spike_rate for m in self.metrics_history]
        
        return {
            'total_epochs': len(self.metrics_history),
            'best_val_accuracy': max(val_accuracies),
            'final_val_accuracy': val_accuracies[-1],
            'avg_spike_rate': np.mean(spike_rates),
            'final_train_loss': train_losses[-1],
            'training_time': time.time() - self.start_time
        }
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        metrics_data = {
            'experiment_name': self.experiment_name,
            'metrics_history': [m.to_dict() for m in self.metrics_history],
            'best_metrics': self.best_metrics.to_dict() if self.best_metrics else None,
            'summary_stats': self.get_summary_stats()
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def load_metrics(self, filepath: str):
        """Load previously saved metrics"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metrics_history = [
            TrainingMetrics(**m) for m in data['metrics_history']
        ]
        
        if data['best_metrics']:
            self.best_metrics = TrainingMetrics(**data['best_metrics'])
        
        self.logger.info(f"Loaded {len(self.metrics_history)} metric records from {filepath}")


class PerformanceProfiler:
    """Performance profiling utilities for SNN operations"""
    
    def __init__(self):
        self.timers = {}
        self.counters = {}
        
    def start_timer(self, name: str):
        """Start a named timer"""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a named timer and return elapsed time"""
        if name not in self.timers:
            raise ValueError(f"Timer '{name}' was never started")
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a named counter"""
        self.counters[name] = self.counters.get(name, 0) + value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all performance statistics"""
        return {
            'active_timers': list(self.timers.keys()),
            'counters': self.counters.copy()
        }
    
    def reset(self):
        """Reset all timers and counters"""
        self.timers.clear()
        self.counters.clear()


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup basic logging for the toolkit"""
    logger = logging.getLogger("snn_toolkit")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger