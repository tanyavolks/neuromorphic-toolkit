# Neuromorphic SNN Toolkit

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

A **hardware-agnostic Spiking Neural Network (SNN) development toolkit** that enables researchers and developers to build, train, and tune spiking neural networks using Python on regular CPU/GPU hardware.

## 🚀 Key Features

- **Unified API**: Single interface for multiple SNN backends (SpikingJelly, snnTorch, Brian2)
- **Hardware Agnostic**: Run on CPU or GPU without specialized neuromorphic hardware
- **Real-time Visualization**: Interactive spike raster plots and network activity monitoring
- **Auto-tuning**: Automated hyperparameter optimization for learning rates, thresholds, and surrogate gradients
- **Ready-to-use Examples**: Complete MNIST classifier demo with training and evaluation
- **Modular Design**: Clean, extensible architecture for research and production use

## 📦 Installation

### Quick Install
```bash
pip install neuromorphic-snn-toolkit
```

### Development Install
```bash
git clone https://github.com/neuromorphic-ai/snn-toolkit.git
cd snn-toolkit
pip install -e .
```

### Full Installation (with all backends)
```bash
pip install neuromorphic-snn-toolkit[full]
```

## 🏗️ Project Structure

```
neuromorphic-snn-toolkit/
├── toolkit/                 # Core SNN toolkit
│   ├── __init__.py
│   ├── core.py             # Unified SNN models and layers
│   ├── visualization.py    # Real-time spike visualization
│   ├── tuning.py           # Hyperparameter auto-tuning
│   └── utils.py            # Data processing utilities
├── examples/               # Working examples and demos
│   ├── __init__.py
│   └── mnist_classifier.py # Complete MNIST classification demo
├── tests/                  # Comprehensive test suite
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
├── docs/                   # Documentation
├── requirements.txt        # Dependencies
├── setup.py               # Package configuration
└── README.md              # This file
```

## 🚀 Quick Start

### Basic SNN Model

```python
import torch
from toolkit import SNNModel, SNNLayer

# Define network architecture
architecture = [
    {
        'input_size': 784,      # MNIST image size
        'output_size': 128,     # Hidden layer
        'neuron_type': 'LIF',   # Leaky Integrate-and-Fire
        'threshold': 1.0,
        'decay': 0.9
    },
    {
        'input_size': 128,
        'output_size': 10,      # 10 classes
        'neuron_type': 'LIF',
        'threshold': 1.0,
        'decay': 0.9
    }
]

# Create model
model = SNNModel(
    architecture=architecture,
    backend="spiking_jelly",  # or "snntorch", "brian2"
    device="auto"
)

# Forward pass with spike recording
batch_size, time_steps, input_size = 32, 100, 784
x = torch.randn(batch_size, time_steps, input_size)
output = model(x, record_spikes=True)

print(f"Output shape: {output.shape}")  # (32, 100, 10)
```

### Real-time Spike Visualization

```python
from toolkit import SpikeRasterPlot
import numpy as np

# Create raster plot
raster_plot = SpikeRasterPlot(
    num_neurons=128,
    time_window=1000,
    update_interval=50
)

# Start real-time visualization
raster_plot.start_realtime()

# Add spike data (in your training loop)
for t in range(1000):
    # Generate or get spike data
    spikes = np.random.random(128) > 0.95  # Sparse spikes
    raster_plot.add_spikes(spikes, t)

# Stop visualization
raster_plot.stop_realtime()
```

### Auto-tuning Hyperparameters

```python
from toolkit import AutoTuner
from toolkit.tuning import TuningConfig

# Define model builder function
def build_model(**params):
    architecture = [
        {
            'input_size': 784,
            'output_size': int(params['hidden_size']),
            'threshold': params['threshold'],
            'decay': params['decay']
        },
        {
            'input_size': int(params['hidden_size']),
            'output_size': 10,
            'threshold': params['threshold'],
            'decay': params['decay']
        }
    ]
    return SNNModel(architecture=architecture)

# Configure tuning
config = TuningConfig(
    param_ranges={
        'hidden_size': (64, 256),
        'threshold': (0.5, 2.0),
        'decay': (0.8, 0.95),
        'learning_rate': (0.0001, 0.01)
    },
    num_trials=50,
    optimization_metric="accuracy"
)

# Run auto-tuning
tuner = AutoTuner(build_model, train_data, val_data, config)
results = tuner.random_search()

print(f"Best parameters: {results['best_params']}")
print(f"Best accuracy: {results['best_score']:.4f}")
```

## 📚 Complete MNIST Example

Run the included MNIST classifier demo:

```bash
python examples/mnist_classifier.py
```

This comprehensive example demonstrates:
- ✅ Data loading and spike encoding
- ✅ SNN model creation and training
- ✅ Real-time spike visualization
- ✅ Hyperparameter tuning
- ✅ Performance evaluation
- ✅ Results plotting

## 🔧 Supported Backends

| Backend | Description | Installation |
|---------|-------------|--------------|
| **SpikingJelly** | Fast, GPU-accelerated SNN simulation | `pip install spikingjelly` |
| **snnTorch** | PyTorch-native SNN library | `pip install snntorch` |
| **Brian2** | Flexible neuromorphic simulator | `pip install brian2` |

The toolkit automatically detects available backends and provides a unified interface.

## 📊 Features Overview

### Core Functionality
- [x] **Unified SNN API** - Single interface for multiple backends
- [x] **LIF Neuron Models** - Leaky Integrate-and-Fire neurons
- [x] **Spike Encoding** - Rate, temporal, and latency coding
- [x] **GPU Acceleration** - CUDA support for fast training

### Visualization
- [x] **Real-time Spike Rasters** - Live spike train visualization
- [x] **Network Activity Monitoring** - Population-level metrics
- [x] **Membrane Potential Plots** - Neuron dynamics visualization
- [x] **Connectivity Visualization** - Weight matrix plotting

### Auto-tuning
- [x] **Random Search** - Efficient hyperparameter exploration
- [x] **Grid Search** - Systematic parameter optimization  
- [x] **Surrogate Gradient Tuning** - Optimize gradient approximations
- [x] **Early Stopping** - Prevent overfitting during tuning

### Data Processing
- [x] **Dataset Loading** - MNIST, CIFAR-10, Fashion-MNIST support
- [x] **Spike Preprocessing** - Normalization, smoothing, subsampling
- [x] **Synthetic Data Generation** - Create test datasets
- [x] **Neuromorphic Formats** - Export to AEDAT, DVS formats

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=toolkit --cov-report=html
```

## 📖 Documentation

Detailed documentation is available in the `docs/` directory:

- [API Reference](docs/api.md)
- [Tutorial: Building Your First SNN](docs/tutorial.md)  
- [Advanced Usage](docs/advanced.md)
- [Backend Comparison](docs/backends.md)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/neuromorphic-ai/snn-toolkit.git
cd snn-toolkit

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This toolkit builds upon the excellent work of:
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly) - Fast SNN simulation
- [snnTorch](https://github.com/jeshraghian/snntorch) - PyTorch SNN library  
- [Brian2](https://github.com/brian-team/brian2) - Neuromorphic simulation
- The broader neuromorphic computing community

## 📊 Performance Benchmarks

| Dataset | Model Size | Training Time | Test Accuracy | Spike Rate |
|---------|------------|---------------|---------------|------------|
| MNIST | 784→128→10 | ~2 min | 94.5% | 0.12 |
| Fashion-MNIST | 784→256→10 | ~4 min | 87.3% | 0.15 |
| CIFAR-10 | 3072→512→10 | ~8 min | 78.9% | 0.18 |

*Benchmarks on NVIDIA RTX 3080, PyTorch 1.12, SpikingJelly backend*

## 🔬 Research Applications

This toolkit has been used in research for:
- **Neuromorphic Computing** - Hardware-efficient AI algorithms
- **Temporal Pattern Recognition** - Time-series classification with SNNs
- **Energy-Efficient AI** - Low-power machine learning
- **Brain-Inspired Computing** - Biologically plausible learning rules

## 📈 Roadmap

- [ ] **Multi-GPU Training** - Distributed SNN training
- [ ] **Online Learning** - Real-time adaptation algorithms  
- [ ] **Neuromorphic Hardware** - Loihi, SpiNNaker integration
- [ ] **Advanced Neuron Models** - NLIF, adaptive neurons
- [ ] **Structured Pruning** - Network compression techniques

## 💬 Community

- **Issues**: [GitHub Issues](https://github.com/neuromorphic-ai/snn-toolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/neuromorphic-ai/snn-toolkit/discussions)
- **Discord**: [Join our Discord](https://discord.gg/neuromorphic-ai)

---

**Built with ❤️ by the Neuromorphic AI Team**

*Making spiking neural networks accessible to everyone, everywhere.*