"""
Advanced data processing and augmentation for neuromorphic data
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Callable
import random

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
except ImportError:
    torch = None
    F = None
    Dataset = None
    DataLoader = None
    transforms = None


class SpikeAugmentation:
    """Data augmentation techniques for spike trains"""
    
    @staticmethod
    def add_noise(spike_train: torch.Tensor, noise_prob: float = 0.01) -> torch.Tensor:
        """Add random noise spikes to the spike train"""
        if torch is None:
            raise ImportError("PyTorch is required for spike augmentation")
        
        noise_mask = torch.rand_like(spike_train) < noise_prob
        return torch.clamp(spike_train + noise_mask.float(), 0, 1)
    
    @staticmethod
    def temporal_jitter(spike_train: torch.Tensor, max_jitter: int = 2) -> torch.Tensor:
        """Apply temporal jittering to spike times"""
        if torch is None:
            raise ImportError("PyTorch is required for spike augmentation")
        
        time_steps, *other_dims = spike_train.shape
        jittered_train = torch.zeros_like(spike_train)
        
        # Apply jitter to each spike
        spike_indices = torch.nonzero(spike_train, as_tuple=True)
        for i in range(len(spike_indices[0])):
            t_orig = spike_indices[0][i].item()
            
            # Random jitter
            jitter = random.randint(-max_jitter, max_jitter)
            t_new = max(0, min(time_steps - 1, t_orig + jitter))
            
            # Copy spike to new time
            coord = tuple(spike_indices[j][i] for j in range(len(spike_indices)))
            new_coord = (t_new,) + coord[1:]
            jittered_train[new_coord] = 1.0
        
        return jittered_train
    
    @staticmethod
    def dropout_spikes(spike_train: torch.Tensor, dropout_prob: float = 0.1) -> torch.Tensor:
        """Randomly drop some spikes"""
        if torch is None:
            raise ImportError("PyTorch is required for spike augmentation")
        
        keep_mask = torch.rand_like(spike_train) > dropout_prob
        return spike_train * keep_mask.float()
    
    @staticmethod
    def time_warp(spike_train: torch.Tensor, warp_factor: float = 0.1) -> torch.Tensor:
        """Apply time warping to spike trains"""
        if torch is None:
            raise ImportError("PyTorch is required for spike augmentation")
        
        time_steps = spike_train.shape[0]
        
        # Create warped time indices
        base_indices = torch.linspace(0, time_steps - 1, time_steps)
        warp_offset = warp_factor * torch.randn(time_steps)
        warped_indices = torch.clamp(base_indices + warp_offset, 0, time_steps - 1)
        
        # Interpolate spike train
        warped_train = torch.zeros_like(spike_train)
        for t in range(time_steps):
            orig_t = int(warped_indices[t].round())
            warped_train[t] = spike_train[orig_t]
        
        return warped_train


class NeuromorphicDataset(Dataset):
    """Dataset class for neuromorphic/spike data"""
    
    def __init__(
        self,
        spike_data: torch.Tensor,
        labels: torch.Tensor,
        transform: Optional[Callable] = None,
        augmentation: Optional[SpikeAugmentation] = None,
        augment_prob: float = 0.5
    ):
        if Dataset is None:
            raise ImportError("PyTorch is required for NeuromorphicDataset")
        
        self.spike_data = spike_data
        self.labels = labels
        self.transform = transform
        self.augmentation = augmentation
        self.augment_prob = augment_prob
        
        assert len(spike_data) == len(labels), "Data and labels must have same length"
    
    def __len__(self) -> int:
        return len(self.spike_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spike_train = self.spike_data[idx]
        label = self.labels[idx]
        
        # Apply augmentation during training
        if self.augmentation and random.random() < self.augment_prob:
            # Randomly choose augmentation
            aug_type = random.choice(['noise', 'jitter', 'dropout'])
            
            if aug_type == 'noise':
                spike_train = self.augmentation.add_noise(spike_train, 0.01)
            elif aug_type == 'jitter':
                spike_train = self.augmentation.temporal_jitter(spike_train, 2)
            elif aug_type == 'dropout':
                spike_train = self.augmentation.dropout_spikes(spike_train, 0.1)
        
        # Apply transform if provided
        if self.transform:
            spike_train = self.transform(spike_train)
        
        return spike_train, label


class AdaptiveEncoding:
    """Advanced spike encoding methods with adaptive parameters"""
    
    @staticmethod
    def adaptive_rate_encoding(
        data: torch.Tensor,
        time_steps: int,
        min_rate: float = 0.01,
        max_rate: float = 0.9
    ) -> torch.Tensor:
        """Rate encoding with adaptive spike rates based on input distribution"""
        if torch is None:
            raise ImportError("PyTorch is required for adaptive encoding")
        
        # Normalize to [0, 1] with adaptive scaling
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = torch.zeros_like(data)
        
        # Scale to desired rate range
        spike_probs = min_rate + normalized_data * (max_rate - min_rate)
        
        # Generate spikes
        spike_train = torch.rand(time_steps, *data.shape) < spike_probs.unsqueeze(0)
        return spike_train.float()
    
    @staticmethod
    def population_vector_encoding(
        data: torch.Tensor,
        time_steps: int,
        num_neurons_per_feature: int = 10
    ) -> torch.Tensor:
        """Population vector encoding for continuous values"""
        if torch is None:
            raise ImportError("PyTorch is required for population encoding")
        
        # Create population of neurons with overlapping tuning curves
        original_shape = data.shape
        data_flat = data.flatten()
        
        # Define neuron preferences (uniformly distributed)
        preferences = torch.linspace(0, 1, num_neurons_per_feature)
        
        encoded_spikes = []
        for value in data_flat:
            # Calculate activation for each neuron (Gaussian tuning curves)
            activations = torch.exp(-((preferences - value) ** 2) / 0.1)
            
            # Convert to spike probabilities
            spike_probs = activations / activations.sum()
            
            # Generate spikes over time
            neuron_spikes = torch.rand(time_steps, num_neurons_per_feature) < spike_probs.unsqueeze(0)
            encoded_spikes.append(neuron_spikes)
        
        # Reshape to match original data structure
        encoded_tensor = torch.stack(encoded_spikes, dim=-1)  # (time, neurons_per_feature, flattened_data)
        
        # Reshape back to original spatial structure
        new_shape = (time_steps, num_neurons_per_feature * data.numel())
        return encoded_tensor.view(new_shape)
    
    @staticmethod
    def rank_order_encoding(
        data: torch.Tensor,
        time_steps: int
    ) -> torch.Tensor:
        """Rank order encoding based on pixel intensity ranking"""
        if torch is None:
            raise ImportError("PyTorch is required for rank order encoding")
        
        # Flatten spatial dimensions
        original_shape = data.shape
        data_flat = data.flatten()
        
        # Get ranking of pixel intensities
        _, sorted_indices = torch.sort(data_flat, descending=True)
        ranks = torch.zeros_like(data_flat)
        ranks[sorted_indices] = torch.arange(len(data_flat), dtype=torch.float)
        
        # Convert ranks to spike times (higher intensity = earlier spike)
        max_rank = len(data_flat) - 1
        spike_times = ((max_rank - ranks) / max_rank * (time_steps - 1)).long()
        
        # Create spike train
        spike_train = torch.zeros(time_steps, len(data_flat))
        for neuron_idx, spike_time in enumerate(spike_times):
            if spike_time < time_steps:
                spike_train[spike_time, neuron_idx] = 1.0
        
        return spike_train


class EventBasedProcessor:
    """Processor for event-based neuromorphic data (DVS, AEDAT, etc.)"""
    
    @staticmethod
    def dvs_to_spike_tensor(
        events: List[Dict[str, Any]],
        width: int,
        height: int,
        time_steps: int,
        time_window: float
    ) -> torch.Tensor:
        """Convert DVS events to spike tensor"""
        if torch is None:
            raise ImportError("PyTorch is required for DVS processing")
        
        # Initialize spike tensor: (time, height, width, 2) for on/off events
        spike_tensor = torch.zeros(time_steps, height, width, 2)
        
        # Time binning
        for event in events:
            x, y = int(event['x']), int(event['y'])
            polarity = int(event['polarity'])
            timestamp = float(event['timestamp'])
            
            # Convert timestamp to time bin
            time_bin = int((timestamp / time_window) * time_steps)
            
            if 0 <= x < width and 0 <= y < height and 0 <= time_bin < time_steps:
                spike_tensor[time_bin, y, x, polarity] = 1.0
        
        return spike_tensor
    
    @staticmethod
    def accumulate_events(
        events: List[Dict[str, Any]],
        width: int,
        height: int,
        time_window: float = None
    ) -> torch.Tensor:
        """Accumulate events into a frame representation"""
        if torch is None:
            raise ImportError("PyTorch is required for event accumulation")
        
        frame = torch.zeros(height, width, 2)  # Separate channels for on/off
        
        if time_window:
            # Filter events by time window
            max_time = max(event['timestamp'] for event in events)
            min_time = max_time - time_window
            events = [e for e in events if e['timestamp'] >= min_time]
        
        for event in events:
            x, y = int(event['x']), int(event['y'])
            polarity = int(event['polarity'])
            
            if 0 <= x < width and 0 <= y < height:
                frame[y, x, polarity] += 1.0
        
        return frame


def create_balanced_dataset(
    spike_data: torch.Tensor,
    labels: torch.Tensor,
    samples_per_class: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create balanced dataset by sampling equal numbers from each class"""
    if torch is None:
        raise ImportError("PyTorch is required for dataset balancing")
    
    unique_labels = torch.unique(labels)
    
    if samples_per_class is None:
        # Use minimum class count
        min_count = min((labels == label).sum().item() for label in unique_labels)
        samples_per_class = min_count
    
    balanced_data = []
    balanced_labels = []
    
    for label in unique_labels:
        # Get indices for this class
        class_indices = torch.where(labels == label)[0]
        
        # Sample randomly if we have more than needed
        if len(class_indices) > samples_per_class:
            selected_indices = torch.randperm(len(class_indices))[:samples_per_class]
            class_indices = class_indices[selected_indices]
        
        balanced_data.append(spike_data[class_indices])
        balanced_labels.append(labels[class_indices])
    
    return torch.cat(balanced_data), torch.cat(balanced_labels)


def split_dataset(
    data: torch.Tensor,
    labels: torch.Tensor,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: Optional[int] = None
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Split dataset into train/validation/test sets with stratification"""
    if torch is None:
        raise ImportError("PyTorch is required for dataset splitting")
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    # Random permutation
    indices = torch.randperm(total_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = (data[train_indices], labels[train_indices])
    val_data = (data[val_indices], labels[val_indices])
    test_data = (data[test_indices], labels[test_indices])
    
    return train_data, val_data, test_data