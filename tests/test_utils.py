#!/usr/bin/env python3
"""
Tests for utility functions
"""

import unittest
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import numpy as np
    from toolkit.utils import (
        encode_spikes, preprocess_spikes, calculate_spike_metrics,
        create_synthetic_dataset
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSpikeEncoding(unittest.TestCase):
    """Test spike encoding functions"""
    
    def setUp(self):
        self.time_steps = 100
        self.data_shape = (28, 28)
        self.test_data = torch.rand(self.data_shape)
    
    def test_rate_encoding(self):
        """Test rate encoding"""
        spike_train = encode_spikes(
            self.test_data, 
            self.time_steps, 
            encoding="rate"
        )
        
        # Check shape
        expected_shape = (self.time_steps, *self.data_shape)
        self.assertEqual(spike_train.shape, expected_shape)
        
        # Check data type and range
        self.assertTrue(torch.all(spike_train >= 0))
        self.assertTrue(torch.all(spike_train <= 1))
    
    def test_temporal_encoding(self):
        """Test temporal encoding"""
        spike_train = encode_spikes(
            self.test_data,
            self.time_steps,
            encoding="temporal"
        )
        
        # Check shape
        expected_shape = (self.time_steps, *self.data_shape)
        self.assertEqual(spike_train.shape, expected_shape)
        
        # Check that spikes are binary
        unique_values = torch.unique(spike_train)
        self.assertTrue(torch.all((unique_values == 0) | (unique_values == 1)))
    
    def test_latency_encoding(self):
        """Test latency encoding"""
        spike_train = encode_spikes(
            self.test_data,
            self.time_steps,
            encoding="latency"
        )
        
        # Check shape
        expected_shape = (self.time_steps, *self.data_shape)
        self.assertEqual(spike_train.shape, expected_shape)
        
        # Check that spikes are binary
        unique_values = torch.unique(spike_train)
        self.assertTrue(torch.all((unique_values == 0) | (unique_values == 1)))
    
    def test_invalid_encoding(self):
        """Test invalid encoding method"""
        with self.assertRaises(ValueError):
            encode_spikes(self.test_data, self.time_steps, encoding="invalid")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSpikePreprocessing(unittest.TestCase):
    """Test spike preprocessing functions"""
    
    def setUp(self):
        self.batch_size = 4
        self.time_steps = 50
        self.num_features = 100
        self.spike_data = torch.rand(
            self.time_steps, self.batch_size, self.num_features
        ) > 0.8  # Create sparse spikes
        self.spike_data = self.spike_data.float()
    
    def test_normalize_preprocessing(self):
        """Test spike normalization"""
        processed = preprocess_spikes(self.spike_data, method="normalize")
        
        # Check shape preservation
        self.assertEqual(processed.shape, self.spike_data.shape)
        
        # Check value range
        self.assertTrue(torch.all(processed >= 0))
        self.assertTrue(torch.all(processed <= 1))
    
    def test_subsample_preprocessing(self):
        """Test spike subsampling"""
        factor = 2
        processed = preprocess_spikes(
            self.spike_data, 
            method="subsample", 
            factor=factor
        )
        
        # Check subsampled shape
        expected_time_steps = self.time_steps // factor
        expected_shape = (expected_time_steps, self.batch_size, self.num_features)
        self.assertEqual(processed.shape, expected_shape)
    
    def test_invalid_preprocessing(self):
        """Test invalid preprocessing method"""
        with self.assertRaises(ValueError):
            preprocess_spikes(self.spike_data, method="invalid")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSpikeMetrics(unittest.TestCase):
    """Test spike metrics calculation"""
    
    def setUp(self):
        self.time_steps = 100
        self.batch_size = 4
        self.num_neurons = 50
        
        # Create test spike data
        self.spike_data = torch.rand(
            self.time_steps, self.batch_size, self.num_neurons
        ) > 0.9  # Sparse spikes
        self.spike_data = self.spike_data.float()
    
    def test_basic_metrics(self):
        """Test basic spike metrics calculation"""
        metrics = calculate_spike_metrics(self.spike_data)
        
        # Check required metrics are present
        required_metrics = ['total_spikes', 'spike_rate', 'active_neurons', 'sparsity']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
        
        # Check metric validity
        self.assertGreaterEqual(metrics['spike_rate'], 0)
        self.assertLessEqual(metrics['spike_rate'], 1)
        self.assertGreaterEqual(metrics['sparsity'], 0)
        self.assertLessEqual(metrics['sparsity'], 1)
        self.assertGreaterEqual(metrics['active_neurons'], 0)
        self.assertLessEqual(metrics['active_neurons'], self.num_neurons)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSyntheticDataset(unittest.TestCase):
    """Test synthetic dataset creation"""
    
    def test_dataset_creation(self):
        """Test synthetic dataset creation"""
        num_samples = 100
        num_features = 784
        num_classes = 10
        time_steps = 50
        
        spike_data, labels = create_synthetic_dataset(
            num_samples=num_samples,
            num_features=num_features,
            num_classes=num_classes,
            time_steps=time_steps,
            seed=42
        )
        
        # Check shapes
        expected_spike_shape = (num_samples, time_steps, num_features)
        expected_label_shape = (num_samples,)
        
        self.assertEqual(spike_data.shape, expected_spike_shape)
        self.assertEqual(labels.shape, expected_label_shape)
        
        # Check label range
        self.assertTrue(torch.all(labels >= 0))
        self.assertTrue(torch.all(labels < num_classes))
        
        # Check spike data type
        self.assertTrue(torch.all(spike_data >= 0))
        self.assertTrue(torch.all(spike_data <= 1))
    
    def test_reproducibility(self):
        """Test that synthetic dataset is reproducible with seed"""
        seed = 123
        
        data1, labels1 = create_synthetic_dataset(
            num_samples=50, seed=seed
        )
        data2, labels2 = create_synthetic_dataset(
            num_samples=50, seed=seed
        )
        
        # Check reproducibility
        self.assertTrue(torch.allclose(data1, data2))
        self.assertTrue(torch.equal(labels1, labels2))


class TestUtilsWithoutTorch(unittest.TestCase):
    """Test utility functions that don't require PyTorch"""
    
    def test_imports(self):
        """Test that module can be imported even without dependencies"""
        try:
            from toolkit import utils
            # Basic import should work even without torch
            self.assertTrue(hasattr(utils, 'encode_spikes'))
        except ImportError as e:
            # Should only fail on torch-specific functions
            self.assertIn('torch', str(e).lower())


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)