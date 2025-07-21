"""
Tests for data processing utilities
"""

import pytest
import torch
import numpy as np
from toolkit.data_processing import (
    SpikeAugmentation, NeuromorphicDataset, AdaptiveEncoding,
    create_balanced_dataset, split_dataset
)


class TestSpikeAugmentation:
    """Test spike augmentation methods"""
    
    def setup_method(self):
        """Setup test data"""
        self.spike_train = torch.rand(100, 50) > 0.9  # Sparse spike train
        self.spike_train = self.spike_train.float()
    
    def test_add_noise(self):
        """Test noise addition"""
        noisy_spikes = SpikeAugmentation.add_noise(self.spike_train, noise_prob=0.05)
        
        assert noisy_spikes.shape == self.spike_train.shape
        assert torch.all(noisy_spikes >= 0)
        assert torch.all(noisy_spikes <= 1)
        assert torch.sum(noisy_spikes) >= torch.sum(self.spike_train)  # Should have more spikes
    
    def test_dropout_spikes(self):
        """Test spike dropout"""
        dropped_spikes = SpikeAugmentation.dropout_spikes(self.spike_train, dropout_prob=0.2)
        
        assert dropped_spikes.shape == self.spike_train.shape
        assert torch.all(dropped_spikes >= 0)
        assert torch.all(dropped_spikes <= 1)
        assert torch.sum(dropped_spikes) <= torch.sum(self.spike_train)  # Should have fewer spikes
    
    def test_temporal_jitter(self):
        """Test temporal jittering"""
        jittered_spikes = SpikeAugmentation.temporal_jitter(self.spike_train, max_jitter=2)
        
        assert jittered_spikes.shape == self.spike_train.shape
        assert torch.all(jittered_spikes >= 0)
        assert torch.all(jittered_spikes <= 1)
        # Spike count may change slightly due to boundary conditions in jittering
        # but should be approximately preserved
        original_count = torch.sum(self.spike_train)
        jittered_count = torch.sum(jittered_spikes)
        assert abs(jittered_count - original_count) <= original_count * 0.1  # Allow 10% variation


class TestNeuromorphicDataset:
    """Test neuromorphic dataset class"""
    
    def setup_method(self):
        """Setup test data"""
        self.spike_data = torch.rand(100, 50, 10)  # 100 samples, 50 time steps, 10 neurons
        self.labels = torch.randint(0, 5, (100,))  # 5 classes
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        dataset = NeuromorphicDataset(self.spike_data, self.labels)
        
        assert len(dataset) == 100
        assert dataset[0][0].shape == (50, 10)
        assert dataset[0][1].shape == ()
    
    def test_dataset_with_augmentation(self):
        """Test dataset with augmentation"""
        augmentation = SpikeAugmentation()
        dataset = NeuromorphicDataset(
            self.spike_data, 
            self.labels, 
            augmentation=augmentation,
            augment_prob=1.0  # Always augment for testing
        )
        
        # Get multiple samples of the same index to check augmentation
        sample1 = dataset[0][0]
        sample2 = dataset[0][0]
        
        # Should be different due to augmentation
        assert not torch.equal(sample1, sample2)


class TestAdaptiveEncoding:
    """Test adaptive encoding methods"""
    
    def setup_method(self):
        """Setup test data"""
        self.data = torch.rand(28, 28)  # MNIST-like image
        self.time_steps = 100
    
    def test_adaptive_rate_encoding(self):
        """Test adaptive rate encoding"""
        spike_train = AdaptiveEncoding.adaptive_rate_encoding(
            self.data, self.time_steps, min_rate=0.01, max_rate=0.9
        )
        
        assert spike_train.shape == (self.time_steps, 28, 28)
        assert torch.all(spike_train >= 0)
        assert torch.all(spike_train <= 1)
    
    def test_population_vector_encoding(self):
        """Test population vector encoding"""
        spike_train = AdaptiveEncoding.population_vector_encoding(
            self.data, self.time_steps, num_neurons_per_feature=5
        )
        
        expected_neurons = 5 * self.data.numel()
        assert spike_train.shape == (self.time_steps, expected_neurons)
        assert torch.all(spike_train >= 0)
        assert torch.all(spike_train <= 1)
    
    def test_rank_order_encoding(self):
        """Test rank order encoding"""
        spike_train = AdaptiveEncoding.rank_order_encoding(self.data, self.time_steps)
        
        assert spike_train.shape == (self.time_steps, self.data.numel())
        assert torch.all(spike_train >= 0)
        assert torch.all(spike_train <= 1)
        
        # Each neuron should spike at most once
        spike_counts = torch.sum(spike_train, dim=0)
        assert torch.all(spike_counts <= 1)


class TestDatasetUtilities:
    """Test dataset utility functions"""
    
    def setup_method(self):
        """Setup test data"""
        # Create imbalanced dataset
        self.data = torch.rand(100, 50)
        self.labels = torch.cat([
            torch.zeros(50),  # 50 samples of class 0
            torch.ones(30),   # 30 samples of class 1
            torch.full((20,), 2)  # 20 samples of class 2
        ]).long()
    
    def test_create_balanced_dataset(self):
        """Test dataset balancing"""
        balanced_data, balanced_labels = create_balanced_dataset(
            self.data, self.labels, samples_per_class=15
        )
        
        # Should have 15 samples per class
        assert len(balanced_data) == 45  # 3 classes * 15 samples
        
        # Check class distribution
        unique_labels, counts = torch.unique(balanced_labels, return_counts=True)
        assert len(unique_labels) == 3
        assert torch.all(counts == 15)
    
    def test_split_dataset(self):
        """Test dataset splitting"""
        train_data, val_data, test_data = split_dataset(
            self.data, self.labels,
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
            random_seed=42
        )
        
        train_x, train_y = train_data
        val_x, val_y = val_data
        test_x, test_y = test_data
        
        # Check sizes
        assert len(train_x) == 60  # 60% of 100
        assert len(val_x) == 20   # 20% of 100
        assert len(test_x) == 20  # 20% of 100
        
        # Check that total equals original
        assert len(train_x) + len(val_x) + len(test_x) == 100
        
        # Check shapes
        assert train_x.shape[1:] == self.data.shape[1:]
        assert val_x.shape[1:] == self.data.shape[1:]
        assert test_x.shape[1:] == self.data.shape[1:]
    
    def test_split_dataset_invalid_ratios(self):
        """Test invalid ratio handling"""
        with pytest.raises(AssertionError):
            split_dataset(
                self.data, self.labels,
                train_ratio=0.5, val_ratio=0.3, test_ratio=0.3  # Sum > 1
            )