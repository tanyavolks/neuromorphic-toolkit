"""
Core SNN model and layer implementations
Unified API for different SNN backends (SpikingJelly, snnTorch, Brian2)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

try:
    import spikingjelly.activation_based.neuron as sj_neuron
    sj = True
except ImportError:
    sj_neuron = None
    sj = None

try:
    import snntorch as snn
except ImportError:
    snn = None

try:
    import brian2 as b2
except ImportError:
    b2 = None


class SNNBackend:
    """Enum for available SNN backends"""
    SPIKING_JELLY = "spiking_jelly"
    SNNTORCH = "snntorch"
    BRIAN2 = "brian2"


class SNNLayer(nn.Module):
    """
    Unified SNN layer that wraps different backend implementations
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        backend: str = SNNBackend.SPIKING_JELLY,
        neuron_type: str = "LIF",
        threshold: float = 1.0,
        decay: float = 0.9,
        **kwargs
    ):
        super().__init__()
        
        # Input validation
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if not 0 < decay < 1:
            raise ValueError(f"decay must be between 0 and 1, got {decay}")
        if neuron_type not in ["LIF"]:
            raise ValueError(f"Unsupported neuron type: {neuron_type}")
        
        self.input_size = input_size
        self.output_size = output_size
        self.backend = backend
        self.neuron_type = neuron_type
        self.threshold = threshold
        self.decay = decay
        
        # Initialize backend-specific layer
        self._init_backend_layer(**kwargs)
        
    def _init_backend_layer(self, **kwargs):
        """Initialize the backend-specific layer"""
        if self.backend == SNNBackend.SPIKING_JELLY and sj is not None:
            self.linear = nn.Linear(self.input_size, self.output_size)
            if self.neuron_type == "LIF":
                self.neuron = sj_neuron.LIFNode(
                    tau=2.0, 
                    v_threshold=self.threshold,
                    v_reset=0.0
                )
            else:
                raise ValueError(f"Neuron type {self.neuron_type} not supported for SpikingJelly")
                
        elif self.backend == SNNBackend.SNNTORCH and snn is not None:
            self.linear = nn.Linear(self.input_size, self.output_size)
            if self.neuron_type == "LIF":
                self.neuron = snn.Leaky(beta=self.decay, threshold=self.threshold)
            else:
                raise ValueError(f"Neuron type {self.neuron_type} not supported for snnTorch")
                
        else:
            raise ValueError(f"Backend {self.backend} not available or not installed")
    
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """Forward pass through the SNN layer"""
        x = self.linear(x)
        
        if self.backend == SNNBackend.SPIKING_JELLY:
            return self.neuron(x), None
            
        elif self.backend == SNNBackend.SNNTORCH:
            if state is None:
                state = self.neuron.init_leaky()
            spk, mem = self.neuron(x, state)
            return spk, mem
            
        return x, state


class SNNModel(nn.Module):
    """
    High-level SNN model class that manages multiple layers and training
    """
    
    def __init__(
        self,
        architecture: List[Dict[str, Any]],
        backend: str = SNNBackend.SPIKING_JELLY,
        device: str = "auto"
    ):
        super().__init__()
        self.architecture = architecture
        self.backend = backend
        self.device = self._get_device(device)
        
        self.layers = nn.ModuleList()
        self.spike_recordings = []
        self.membrane_recordings = []
        
        self._build_network()
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _build_network(self):
        """Build the network from architecture specification"""
        for i, layer_config in enumerate(self.architecture):
            layer = SNNLayer(
                backend=self.backend,
                **layer_config
            )
            self.layers.append(layer)
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, record_spikes: bool = False):
        """
        Forward pass through the entire network
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_size)
            record_spikes: Whether to record spike trains for visualization
        """
        batch_size, time_steps, _ = x.shape
        
        if record_spikes:
            self.spike_recordings = [[] for _ in self.layers]
            self.membrane_recordings = [[] for _ in self.layers]
        
        states = [None] * len(self.layers)
        
        outputs = []
        for t in range(time_steps):
            layer_input = x[:, t, :]
            
            for i, layer in enumerate(self.layers):
                spikes, states[i] = layer(layer_input, states[i])
                
                if record_spikes:
                    self.spike_recordings[i].append(spikes.detach().cpu())
                    if states[i] is not None:
                        self.membrane_recordings[i].append(states[i].detach().cpu())
                
                layer_input = spikes
            
            outputs.append(spikes)
        
        return torch.stack(outputs, dim=1)  # (batch_size, time_steps, output_size)
    
    def get_spike_counts(self) -> torch.Tensor:
        """Get total spike counts for each neuron over the simulation"""
        if not self.spike_recordings:
            raise ValueError("No spike recordings available. Run forward() with record_spikes=True")
        
        spike_counts = []
        for layer_spikes in self.spike_recordings:
            if layer_spikes:
                total_spikes = torch.stack(layer_spikes).sum(dim=0)
                spike_counts.append(total_spikes)
        
        return spike_counts
    
    def reset_state(self):
        """Reset the internal state of all neurons"""
        for layer in self.layers:
            if hasattr(layer.neuron, 'reset'):
                layer.neuron.reset()