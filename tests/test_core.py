#!/usr/bin/env python3
"""
Tests for core SNN functionality
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import numpy as np
    from toolkit.core import SNNModel, SNNLayer, SNNBackend
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSNNLayer(unittest.TestCase):
    """Test SNN layer functionality"""
    
    def setUp(self):
        self.input_size = 10
        self.output_size = 5
        self.batch_size = 2
        self.time_steps = 20
        
    def test_layer_creation(self):
        """Test basic layer creation"""
        try:
            layer = SNNLayer(
                input_size=self.input_size,
                output_size=self.output_size,
                backend=SNNBackend.SPIKING_JELLY,
                threshold=1.0
            )
            self.assertEqual(layer.input_size, self.input_size)
            self.assertEqual(layer.output_size, self.output_size)
        except ImportError:
            self.skipTest("SpikingJelly not available")
    
    def test_layer_forward(self):
        """Test layer forward pass"""
        try:
            layer = SNNLayer(
                input_size=self.input_size,
                output_size=self.output_size,
                backend=SNNBackend.SPIKING_JELLY
            )
            
            # Test input
            x = torch.randn(self.batch_size, self.input_size)
            
            # Forward pass
            output, state = layer(x)
            
            # Check output shape
            self.assertEqual(output.shape, (self.batch_size, self.output_size))
            
        except ImportError:
            self.skipTest("SpikingJelly not available")
    
    def test_different_backends(self):
        """Test different SNN backends"""
        backends_to_test = []
        
        try:
            import spikingjelly
            backends_to_test.append(SNNBackend.SPIKING_JELLY)
        except ImportError:
            pass
            
        try:
            import snntorch
            backends_to_test.append(SNNBackend.SNNTORCH)
        except ImportError:
            pass
        
        if not backends_to_test:
            self.skipTest("No SNN backends available")
        
        for backend in backends_to_test:
            with self.subTest(backend=backend):
                layer = SNNLayer(
                    input_size=self.input_size,
                    output_size=self.output_size,
                    backend=backend
                )
                self.assertIsNotNone(layer)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSNNModel(unittest.TestCase):
    """Test SNN model functionality"""
    
    def setUp(self):
        self.architecture = [
            {
                'input_size': 784,
                'output_size': 128,
                'neuron_type': 'LIF',
                'threshold': 1.0,
                'decay': 0.9
            },
            {
                'input_size': 128,
                'output_size': 10,
                'neuron_type': 'LIF',
                'threshold': 1.0,
                'decay': 0.9
            }
        ]
        self.batch_size = 4
        self.time_steps = 50
        self.input_size = 784
    
    def test_model_creation(self):
        """Test model creation from architecture"""
        try:
            model = SNNModel(
                architecture=self.architecture,
                backend=SNNBackend.SPIKING_JELLY
            )
            self.assertEqual(len(model.layers), len(self.architecture))
        except ImportError:
            self.skipTest("SpikingJelly not available")
    
    def test_model_forward(self):
        """Test model forward pass"""
        try:
            model = SNNModel(
                architecture=self.architecture,
                backend=SNNBackend.SPIKING_JELLY
            )
            
            # Create test input
            x = torch.randn(self.batch_size, self.time_steps, self.input_size)
            
            # Forward pass
            output = model(x, record_spikes=True)
            
            # Check output shape
            expected_shape = (self.batch_size, self.time_steps, self.architecture[-1]['output_size'])
            self.assertEqual(output.shape, expected_shape)
            
            # Check spike recordings
            self.assertTrue(hasattr(model, 'spike_recordings'))
            self.assertEqual(len(model.spike_recordings), len(self.architecture))
            
        except ImportError:
            self.skipTest("SpikingJelly not available")
    
    def test_spike_counting(self):
        """Test spike count functionality"""
        try:
            model = SNNModel(
                architecture=self.architecture,
                backend=SNNBackend.SPIKING_JELLY
            )
            
            # Create test input
            x = torch.randn(self.batch_size, self.time_steps, self.input_size)
            
            # Forward pass with spike recording
            output = model(x, record_spikes=True)
            
            # Get spike counts
            spike_counts = model.get_spike_counts()
            
            # Check that we have spike counts for each layer
            self.assertEqual(len(spike_counts), len(self.architecture))
            
        except ImportError:
            self.skipTest("SpikingJelly not available")


class TestSNNUtils(unittest.TestCase):
    """Test utility functions that don't require SNN backends"""
    
    def test_backend_enum(self):
        """Test backend enumeration"""
        self.assertEqual(SNNBackend.SPIKING_JELLY, "spiking_jelly")
        self.assertEqual(SNNBackend.SNNTORCH, "snntorch")
        self.assertEqual(SNNBackend.BRIAN2, "brian2")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)