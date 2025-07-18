"""
Utility functions for SNN data processing and common operations
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
import os
import pickle

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    F = None
    DataLoader = None
    TensorDataset = None

try:
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    torchvision = None
    transforms = None


def load_dataset(
    dataset_name: str = "mnist",
    data_dir: str = "./data",
    batch_size: int = 64,
    time_steps: int = 100,
    encoding: str = "rate",
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Load and preprocess datasets for SNN training
    
    Args:
        dataset_name: Name of dataset ("mnist", "cifar10", "fashion_mnist")
        data_dir: Directory to store/load data
        batch_size: Batch size for DataLoaders
        time_steps: Number of time steps for spike encoding
        encoding: Spike encoding method ("rate", "temporal", "latency")
        normalize: Whether to normalize input data
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if torch is None or torchvision is None:
        raise ImportError("PyTorch and torchvision are required for dataset loading")
    
    # Setup data transforms
    transform_list = []
    if normalize:
        if dataset_name.lower() == "mnist":
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        elif dataset_name.lower() == "cifar10":
            transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    
    transform = transforms.Compose([transforms.ToTensor()] + transform_list)
    
    # Load dataset
    if dataset_name.lower() == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
    elif dataset_name.lower() == "fashion_mnist":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
    elif dataset_name.lower() == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Convert to spike trains
    train_spike_data = []
    train_labels = []
    
    print(f"Converting {dataset_name} training data to spike trains...")
    for i, (data, label) in enumerate(train_dataset):
        if i >= 1000:  # Limit for demo purposes
            break
        spike_train = encode_spikes(data, time_steps, encoding)
        train_spike_data.append(spike_train)
        train_labels.append(label)
    
    test_spike_data = []
    test_labels = []
    
    print(f"Converting {dataset_name} test data to spike trains...")
    for i, (data, label) in enumerate(test_dataset):
        if i >= 200:  # Limit for demo purposes
            break
        spike_train = encode_spikes(data, time_steps, encoding)
        test_spike_data.append(spike_train)
        test_labels.append(label)
    
    # Create DataLoaders
    train_spike_tensor = torch.stack(train_spike_data)
    train_label_tensor = torch.tensor(train_labels)
    train_spike_dataset = TensorDataset(train_spike_tensor, train_label_tensor)
    train_loader = DataLoader(train_spike_dataset, batch_size=batch_size, shuffle=True)
    
    test_spike_tensor = torch.stack(test_spike_data)
    test_label_tensor = torch.tensor(test_labels)
    test_spike_dataset = TensorDataset(test_spike_tensor, test_label_tensor)
    test_loader = DataLoader(test_spike_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    print(f"Spike trains: {train_spike_tensor.shape}, {test_spike_tensor.shape}")
    
    return train_loader, test_loader


def encode_spikes(
    data: torch.Tensor,
    time_steps: int,
    encoding: str = "rate"
) -> torch.Tensor:
    """
    Convert input data to spike trains
    
    Args:
        data: Input tensor (image or feature vector)
        time_steps: Number of time steps
        encoding: Encoding method ("rate", "temporal", "latency")
    
    Returns:
        Spike train tensor of shape (time_steps, *data.shape)
    """
    if torch is None:
        raise ImportError("PyTorch is required for spike encoding")
    
    # Flatten spatial dimensions but keep channels
    original_shape = data.shape
    if data.dim() > 1:
        data = data.view(data.shape[0], -1) if data.dim() == 3 else data.flatten()
    
    if encoding == "rate":
        # Rate coding: spike probability proportional to input intensity
        spike_train = torch.rand(time_steps, *data.shape) < data.unsqueeze(0)
        return spike_train.float()
    
    elif encoding == "temporal":
        # Temporal coding: spike timing based on input intensity
        spike_train = torch.zeros(time_steps, *data.shape)
        spike_times = (data * (time_steps - 1)).long()
        
        for t in range(time_steps):
            spike_train[t] = (spike_times == t).float()
        
        return spike_train
    
    elif encoding == "latency":
        # Latency coding: first spike time inversely related to intensity
        spike_train = torch.zeros(time_steps, *data.shape)
        
        # Avoid division by zero
        data_safe = torch.clamp(data, min=1e-7)
        spike_times = ((1.0 - data_safe) * (time_steps - 1)).long()
        
        for t in range(time_steps):
            mask = spike_times == t
            spike_train[t] = mask.float()
            # Ensure each neuron spikes only once
            spike_times[mask] = time_steps  # Set to invalid time
        
        return spike_train
    
    else:
        raise ValueError(f"Unknown encoding method: {encoding}")


def preprocess_spikes(
    spike_data: torch.Tensor,
    method: str = "normalize",
    **kwargs
) -> torch.Tensor:
    """
    Preprocess spike train data
    
    Args:
        spike_data: Spike train tensor
        method: Preprocessing method ("normalize", "smooth", "subsample")
        **kwargs: Additional parameters for preprocessing methods
    
    Returns:
        Preprocessed spike data
    """
    if torch is None:
        raise ImportError("PyTorch is required for spike preprocessing")
    
    if method == "normalize":
        # Normalize spike rates
        spike_rates = spike_data.mean(dim=1, keepdim=True)  # Average over time
        normalized_data = spike_data / (spike_rates + 1e-7)
        return torch.clamp(normalized_data, 0, 1)
    
    elif method == "smooth":
        # Temporal smoothing with sliding window
        window_size = kwargs.get("window_size", 5)
        kernel = torch.ones(1, 1, window_size) / window_size
        
        # Apply smoothing
        smoothed_data = torch.zeros_like(spike_data)
        for i in range(spike_data.shape[-1]):  # For each feature
            feature_data = spike_data[:, :, i:i+1].transpose(0, 1)  # (batch, time, 1)
            smoothed = F.conv1d(feature_data, kernel, padding=window_size//2)
            smoothed_data[:, :, i] = smoothed.transpose(0, 1).squeeze()
        
        return smoothed_data
    
    elif method == "subsample":
        # Temporal subsampling
        factor = kwargs.get("factor", 2)
        return spike_data[::factor]
    
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


def calculate_spike_metrics(spike_data: torch.Tensor) -> Dict[str, float]:
    """
    Calculate various spike train metrics
    
    Args:
        spike_data: Spike train tensor of shape (time_steps, batch_size, num_neurons)
    
    Returns:
        Dictionary of spike metrics
    """
    if torch is None:
        raise ImportError("PyTorch is required for spike metrics")
    
    # Convert to numpy for easier computation
    spikes = spike_data.cpu().numpy() if isinstance(spike_data, torch.Tensor) else spike_data
    
    # Basic metrics
    total_spikes = np.sum(spikes)
    num_neurons = spikes.shape[-1]
    num_timesteps = spikes.shape[0]
    batch_size = spikes.shape[1] if spikes.ndim > 2 else 1
    
    metrics = {
        'total_spikes': float(total_spikes),
        'spike_rate': float(total_spikes / (num_timesteps * num_neurons * batch_size)),
        'active_neurons': float(np.sum(np.any(spikes, axis=0))),
        'sparsity': float(1.0 - (total_spikes / spikes.size)),
    }
    
    # Temporal metrics
    if num_timesteps > 1:
        # Inter-spike intervals
        spike_times = []
        for neuron in range(num_neurons):
            neuron_spikes = spikes[:, 0, neuron] if spikes.ndim > 2 else spikes[:, neuron]
            spike_indices = np.where(neuron_spikes > 0)[0]
            if len(spike_indices) > 1:
                intervals = np.diff(spike_indices)
                spike_times.extend(intervals)
        
        if spike_times:
            metrics['mean_isi'] = float(np.mean(spike_times))
            metrics['std_isi'] = float(np.std(spike_times))
        else:
            metrics['mean_isi'] = 0.0
            metrics['std_isi'] = 0.0
    
    return metrics


def save_model_checkpoint(
    model: Any,
    optimizer: Any,
    epoch: int,
    loss: float,
    accuracy: float,
    filepath: str
):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        filepath: Path to save checkpoint
    """
    if torch is None:
        raise ImportError("PyTorch is required for saving checkpoints")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_model_checkpoint(filepath: str) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
    
    Returns:
        Checkpoint dictionary
    """
    if torch is None:
        raise ImportError("PyTorch is required for loading checkpoints")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath)
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}, Accuracy: {checkpoint['accuracy']:.4f}")
    
    return checkpoint


def create_synthetic_dataset(
    num_samples: int = 1000,
    num_features: int = 784,
    num_classes: int = 10,
    time_steps: int = 100,
    spike_prob: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic spike train dataset for testing
    
    Args:
        num_samples: Number of samples
        num_features: Number of input features
        num_classes: Number of output classes
        time_steps: Number of time steps
        spike_prob: Base spike probability
        seed: Random seed
    
    Returns:
        Tuple of (spike_data, labels)
    """
    if torch is None:
        raise ImportError("PyTorch is required for synthetic dataset creation")
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate class-dependent spike patterns
    spike_data = []
    labels = []
    
    for i in range(num_samples):
        label = np.random.randint(0, num_classes)
        
        # Create class-specific patterns
        class_prob = spike_prob * (1.0 + 0.5 * label / num_classes)
        
        # Generate spike train with temporal structure
        sample_spikes = torch.zeros(time_steps, num_features)
        
        for t in range(time_steps):
            # Add temporal dynamics
            temporal_prob = class_prob * (1.0 + 0.3 * np.sin(2 * np.pi * t / time_steps))
            spikes = torch.rand(num_features) < temporal_prob
            sample_spikes[t] = spikes.float()
        
        spike_data.append(sample_spikes)
        labels.append(label)
    
    spike_tensor = torch.stack(spike_data)
    label_tensor = torch.tensor(labels)
    
    print(f"Created synthetic dataset: {spike_tensor.shape}, {num_classes} classes")
    
    return spike_tensor, label_tensor


def analyze_network_connectivity(model: Any) -> Dict[str, Any]:
    """
    Analyze connectivity patterns in the network
    
    Args:
        model: SNN model
    
    Returns:
        Dictionary of connectivity analysis results
    """
    connectivity_info = {
        'layers': [],
        'total_parameters': 0,
        'total_connections': 0
    }
    
    if torch is None:
        return connectivity_info
    
    total_params = 0
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'linear'):
            weight_matrix = layer.linear.weight.data.cpu().numpy()
            
            layer_info = {
                'layer_index': i,
                'input_size': weight_matrix.shape[1],
                'output_size': weight_matrix.shape[0],
                'num_connections': weight_matrix.size,
                'weight_mean': float(np.mean(weight_matrix)),
                'weight_std': float(np.std(weight_matrix)),
                'sparsity': float(np.sum(np.abs(weight_matrix) < 1e-6) / weight_matrix.size)
            }
            
            connectivity_info['layers'].append(layer_info)
            total_params += weight_matrix.size
    
    connectivity_info['total_parameters'] = total_params
    connectivity_info['total_connections'] = total_params
    
    return connectivity_info


def convert_to_neuromorphic_format(
    spike_data: torch.Tensor,
    format_type: str = "aedat"
) -> bytes:
    """
    Convert spike data to neuromorphic data formats
    
    Args:
        spike_data: Spike train tensor
        format_type: Output format ("aedat", "n_mnist", "dvs")
    
    Returns:
        Serialized data in specified format
    """
    # Simplified implementation - real neuromorphic formats would need proper libraries
    if format_type == "aedat":
        # Address Event Representation
        events = []
        time_steps, batch_size, num_neurons = spike_data.shape
        
        for t in range(time_steps):
            for b in range(batch_size):
                for n in range(num_neurons):
                    if spike_data[t, b, n] > 0:
                        events.append({
                            'timestamp': t,
                            'x': n % 28,  # Assuming 28x28 spatial layout
                            'y': n // 28,
                            'polarity': 1
                        })
        
        return pickle.dumps(events)
    
    else:
        raise ValueError(f"Unsupported format: {format_type}")