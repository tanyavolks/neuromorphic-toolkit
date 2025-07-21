"""
Neuromorphic SNN Toolkit
A hardware-agnostic Spiking Neural Network development toolkit

Unified API for building, training, and tuning SNNs on CPU/GPU using Python
"""

__version__ = "0.1.0"
__author__ = "Neuromorphic AI Team"

# Core components
try:
    from .core import SNNModel, SNNLayer, SNNBackend
except ImportError as e:
    print(f"Warning: Could not import core components: {e}")
    SNNModel = SNNLayer = SNNBackend = None

# Configuration and logging
try:
    from .config import SNNConfig, get_default_mnist_config, get_default_cifar10_config
except ImportError as e:
    print(f"Warning: Could not import config: {e}")
    SNNConfig = get_default_mnist_config = get_default_cifar10_config = None

try:
    from .logging_utils import MetricsLogger, TrainingMetrics, setup_logging
except ImportError as e:
    print(f"Warning: Could not import logging_utils: {e}")
    MetricsLogger = TrainingMetrics = setup_logging = None

# Utilities
try:
    from .utils import load_dataset, preprocess_spikes, encode_spikes, calculate_spike_metrics
except ImportError as e:
    print(f"Warning: Could not import utils: {e}")
    load_dataset = preprocess_spikes = encode_spikes = calculate_spike_metrics = None

# Advanced data processing
try:
    from .data_processing import (
        SpikeAugmentation,
        NeuromorphicDataset,
        AdaptiveEncoding,
        EventBasedProcessor,
        create_balanced_dataset,
        split_dataset
    )
except ImportError as e:
    print(f"Warning: Could not import data_processing: {e}")
    SpikeAugmentation = NeuromorphicDataset = AdaptiveEncoding = None
    EventBasedProcessor = create_balanced_dataset = split_dataset = None

# Tuning (optional - requires working core)
try:
    from .tuning import AutoTuner, TuningConfig, SurrogateGradientTuner
except ImportError as e:
    print(f"Warning: Could not import tuning: {e}")
    AutoTuner = TuningConfig = SurrogateGradientTuner = None

# Visualization (optional - may have additional dependencies)
try:
    from .visualization import SpikeRasterPlot, NetworkActivityMonitor
except ImportError as e:
    print(f"Warning: Could not import visualization: {e}")
    SpikeRasterPlot = NetworkActivityMonitor = None

# Build __all__ list dynamically based on what actually imported
__all__ = []

# Add items only if they were successfully imported
_exports = {
    # Core
    "SNNModel": SNNModel,
    "SNNLayer": SNNLayer,
    "SNNBackend": SNNBackend,
    
    # Visualization
    "SpikeRasterPlot": SpikeRasterPlot,
    "NetworkActivityMonitor": NetworkActivityMonitor,
    
    # Tuning
    "AutoTuner": AutoTuner,
    "TuningConfig": TuningConfig,
    "SurrogateGradientTuner": SurrogateGradientTuner,
    
    # Utils
    "load_dataset": load_dataset,
    "preprocess_spikes": preprocess_spikes,
    "encode_spikes": encode_spikes,
    "calculate_spike_metrics": calculate_spike_metrics,
    
    # Configuration
    "SNNConfig": SNNConfig,
    "get_default_mnist_config": get_default_mnist_config,
    "get_default_cifar10_config": get_default_cifar10_config,
    
    # Logging
    "MetricsLogger": MetricsLogger,
    "TrainingMetrics": TrainingMetrics,
    "setup_logging": setup_logging,
    
    # Data processing
    "SpikeAugmentation": SpikeAugmentation,
    "NeuromorphicDataset": NeuromorphicDataset,
    "AdaptiveEncoding": AdaptiveEncoding,
    "EventBasedProcessor": EventBasedProcessor,
    "create_balanced_dataset": create_balanced_dataset,
    "split_dataset": split_dataset
}

# Only export items that were successfully imported (not None)
for name, obj in _exports.items():
    if obj is not None:
        __all__.append(name)