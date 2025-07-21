"""
Configuration management for SNN Toolkit
Provides default configurations and validation
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import os
import json
import yaml


@dataclass
class SNNConfig:
    """Configuration class for SNN models and training"""
    
    # Model architecture
    input_size: int = 784
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    output_size: int = 10
    neuron_type: str = "LIF"
    threshold: float = 1.0
    decay: float = 0.9
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 10
    time_steps: int = 100
    
    # Backend and device
    backend: str = "spiking_jelly"
    device: str = "auto"
    
    # Data processing
    encoding: str = "rate"
    normalize: bool = True
    
    # Optimization
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    early_stopping: bool = True
    patience: int = 5
    
    # Visualization
    enable_visualization: bool = False
    plot_interval: int = 10
    
    # Logging
    log_level: str = "INFO"
    save_checkpoints: bool = True
    checkpoint_interval: int = 5
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        if self.input_size <= 0:
            errors.append(f"input_size must be positive, got {self.input_size}")
        if self.output_size <= 0:
            errors.append(f"output_size must be positive, got {self.output_size}")
        if any(h <= 0 for h in self.hidden_sizes):
            errors.append(f"All hidden sizes must be positive, got {self.hidden_sizes}")
        if self.threshold <= 0:
            errors.append(f"threshold must be positive, got {self.threshold}")
        if not 0 < self.decay < 1:
            errors.append(f"decay must be between 0 and 1, got {self.decay}")
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        if self.time_steps <= 0:
            errors.append(f"time_steps must be positive, got {self.time_steps}")
        if self.neuron_type not in ["LIF"]:
            errors.append(f"Unsupported neuron_type: {self.neuron_type}")
        if self.backend not in ["spiking_jelly", "snntorch", "brian2"]:
            errors.append(f"Unsupported backend: {self.backend}")
        if self.encoding not in ["rate", "temporal", "latency"]:
            errors.append(f"Unsupported encoding: {self.encoding}")
        if self.optimizer not in ["adam", "sgd", "rmsprop"]:
            errors.append(f"Unsupported optimizer: {self.optimizer}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SNNConfig':
        """Create from dictionary"""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save configuration to file (supports .json and .yaml)"""
        config_dict = self.to_dict()
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format. Use .json or .yaml, got {filepath}")
        
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SNNConfig':
        """Load configuration from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format. Use .json or .yaml, got {filepath}")
        
        config = cls.from_dict(config_dict)
        config.validate()
        return config


def get_default_mnist_config() -> SNNConfig:
    """Get default configuration for MNIST classification"""
    return SNNConfig(
        input_size=784,
        hidden_sizes=[128, 64],
        output_size=10,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=10,
        time_steps=100,
        encoding="rate"
    )


def get_default_cifar10_config() -> SNNConfig:
    """Get default configuration for CIFAR-10 classification"""
    return SNNConfig(
        input_size=3072,  # 32*32*3
        hidden_sizes=[256, 128],
        output_size=10,
        learning_rate=0.0005,
        batch_size=64,
        num_epochs=20,
        time_steps=150,
        encoding="rate"
    )


def create_config_template(filepath: str = "snn_config_template.yaml"):
    """Create a configuration template file"""
    config = SNNConfig()
    config.save(filepath)
    print(f"Configuration template created at {filepath}")
    print("Edit this file and use SNNConfig.load() to load your custom configuration")