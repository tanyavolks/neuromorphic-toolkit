"""
Neuromorphic SNN Toolkit
A hardware-agnostic Spiking Neural Network development toolkit

Unified API for building, training, and tuning SNNs on CPU/GPU using Python
"""

__version__ = "0.1.0"
__author__ = "Neuromorphic AI Team"

from .core import SNNModel, SNNLayer
from .visualization import SpikeRasterPlot
from .tuning import AutoTuner
from .utils import load_dataset, preprocess_spikes

__all__ = [
    "SNNModel", 
    "SNNLayer", 
    "SpikeRasterPlot", 
    "AutoTuner",
    "load_dataset",
    "preprocess_spikes"
]