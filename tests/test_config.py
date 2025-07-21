"""
Tests for configuration management
"""

import pytest
import tempfile
import os
import json
from toolkit.config import SNNConfig, get_default_mnist_config, get_default_cifar10_config


class TestSNNConfig:
    """Test SNNConfig class"""
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config = SNNConfig()
        assert config.input_size == 784
        assert config.output_size == 10
        assert config.learning_rate == 0.001
        assert config.backend == "spiking_jelly"
    
    def test_config_validation_success(self):
        """Test successful validation"""
        config = SNNConfig(
            input_size=100,
            hidden_sizes=[64, 32],
            output_size=5,
            threshold=1.5,
            decay=0.95
        )
        assert config.validate() is True
    
    def test_config_validation_failures(self):
        """Test validation failures"""
        
        # Test negative input size
        with pytest.raises(ValueError, match="input_size must be positive"):
            config = SNNConfig(input_size=-1)
            config.validate()
        
        # Test invalid decay
        with pytest.raises(ValueError, match="decay must be between 0 and 1"):
            config = SNNConfig(decay=1.5)
            config.validate()
        
        # Test invalid threshold
        with pytest.raises(ValueError, match="threshold must be positive"):
            config = SNNConfig(threshold=-0.5)
            config.validate()
        
        # Test invalid backend
        with pytest.raises(ValueError, match="Unsupported backend"):
            config = SNNConfig(backend="invalid_backend")
            config.validate()
    
    def test_config_serialization(self):
        """Test config to/from dict conversion"""
        config = SNNConfig(input_size=100, output_size=5)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['input_size'] == 100
        assert config_dict['output_size'] == 5
        
        # Test from_dict
        new_config = SNNConfig.from_dict(config_dict)
        assert new_config.input_size == 100
        assert new_config.output_size == 5
    
    def test_config_save_load_json(self):
        """Test saving and loading JSON configuration"""
        config = SNNConfig(input_size=200, learning_rate=0.01)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save(temp_path)
            assert os.path.exists(temp_path)
            
            loaded_config = SNNConfig.load(temp_path)
            assert loaded_config.input_size == 200
            assert loaded_config.learning_rate == 0.01
        finally:
            os.unlink(temp_path)
    
    def test_invalid_file_format(self):
        """Test invalid file format handling"""
        config = SNNConfig()
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            config.save("test.txt")
        
        # Create a dummy file to test load with invalid format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
            f.write("dummy content")
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                SNNConfig.load(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_missing_file(self):
        """Test loading missing file"""
        with pytest.raises(FileNotFoundError):
            SNNConfig.load("nonexistent_file.json")


class TestDefaultConfigs:
    """Test default configuration functions"""
    
    def test_mnist_config(self):
        """Test MNIST default configuration"""
        config = get_default_mnist_config()
        assert config.input_size == 784
        assert config.output_size == 10
        assert config.encoding == "rate"
        assert config.validate() is True
    
    def test_cifar10_config(self):
        """Test CIFAR-10 default configuration"""
        config = get_default_cifar10_config()
        assert config.input_size == 3072  # 32*32*3
        assert config.output_size == 10
        assert config.batch_size == 64
        assert config.validate() is True