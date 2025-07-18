"""
Real-time spike visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Optional, Tuple, Dict, Any
import time
from collections import deque

try:
    import torch
except ImportError:
    torch = None


class SpikeRasterPlot:
    """
    Real-time spike raster plot visualization
    """
    
    def __init__(
        self,
        num_neurons: int,
        time_window: int = 1000,
        figsize: Tuple[int, int] = (12, 8),
        update_interval: int = 50
    ):
        self.num_neurons = num_neurons
        self.time_window = time_window
        self.update_interval = update_interval
        
        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(0, time_window)
        self.ax.set_ylim(0, num_neurons)
        self.ax.set_xlabel('Time Steps')
        self.ax.set_ylabel('Neuron Index')
        self.ax.set_title('Real-time Spike Raster Plot')
        
        # Data storage
        self.spike_times = deque(maxlen=time_window * num_neurons)
        self.neuron_ids = deque(maxlen=time_window * num_neurons)
        self.current_time = 0
        
        # Plot elements
        self.scatter = self.ax.scatter([], [], s=2, c='black', alpha=0.7)
        
        # Animation
        self.animation = None
        self.is_running = False
    
    def add_spikes(self, spikes: np.ndarray, time_step: Optional[int] = None):
        """
        Add spike data for the current time step
        
        Args:
            spikes: Boolean array of shape (num_neurons,) indicating which neurons spiked
            time_step: Optional explicit time step, otherwise uses internal counter
        """
        if time_step is None:
            time_step = self.current_time
            self.current_time += 1
        
        # Convert to numpy if torch tensor
        if torch is not None and isinstance(spikes, torch.Tensor):
            spikes = spikes.cpu().numpy()
        
        # Find spiking neurons
        spiking_neurons = np.where(spikes > 0)[0]
        
        # Add spike data
        for neuron_id in spiking_neurons:
            self.spike_times.append(time_step)
            self.neuron_ids.append(neuron_id)
    
    def update_plot(self, frame):
        """Update the plot for animation"""
        if len(self.spike_times) > 0:
            # Filter data within time window
            current_max_time = max(self.spike_times) if self.spike_times else 0
            min_time = max(0, current_max_time - self.time_window)
            
            # Get data within window
            times = np.array(self.spike_times)
            neurons = np.array(self.neuron_ids)
            
            mask = times >= min_time
            filtered_times = times[mask] - min_time
            filtered_neurons = neurons[mask]
            
            # Update scatter plot
            self.scatter.set_offsets(np.column_stack([filtered_times, filtered_neurons]))
            
            # Update x-axis
            self.ax.set_xlim(0, self.time_window)
        
        return self.scatter,
    
    def start_realtime(self):
        """Start real-time animation"""
        if not self.is_running:
            self.animation = animation.FuncAnimation(
                self.fig, 
                self.update_plot, 
                interval=self.update_interval,
                blit=True,
                cache_frame_data=False
            )
            self.is_running = True
            plt.show(block=False)
    
    def stop_realtime(self):
        """Stop real-time animation"""
        if self.animation:
            self.animation.event_source.stop()
            self.is_running = False
    
    def save_plot(self, filename: str):
        """Save current plot to file"""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
    
    def clear(self):
        """Clear all spike data"""
        self.spike_times.clear()
        self.neuron_ids.clear()
        self.current_time = 0
        self.scatter.set_offsets(np.empty((0, 2)))


class NetworkActivityMonitor:
    """
    Monitor and visualize network-wide activity metrics
    """
    
    def __init__(self, time_window: int = 1000):
        self.time_window = time_window
        self.activity_history = deque(maxlen=time_window)
        self.time_steps = deque(maxlen=time_window)
        self.current_step = 0
        
        # Initialize plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Activity rate plot
        self.ax1.set_title('Network Activity Rate')
        self.ax1.set_ylabel('Spikes/Time Step')
        self.line1, = self.ax1.plot([], [], 'b-', alpha=0.7)
        
        # Population spike histogram
        self.ax2.set_title('Population Spike Distribution')
        self.ax2.set_xlabel('Time Steps')
        self.ax2.set_ylabel('Neuron Index')
    
    def update_activity(self, spike_count: float):
        """Update activity metrics"""
        self.activity_history.append(spike_count)
        self.time_steps.append(self.current_step)
        self.current_step += 1
        
        # Update activity plot
        if len(self.time_steps) > 1:
            self.line1.set_data(list(self.time_steps), list(self.activity_history))
            self.ax1.set_xlim(min(self.time_steps), max(self.time_steps))
            self.ax1.set_ylim(0, max(self.activity_history) * 1.1 if self.activity_history else 1)
    
    def plot_spike_histogram(self, spikes_by_neuron: np.ndarray, bins: int = 50):
        """Plot histogram of spike counts by neuron"""
        self.ax2.clear()
        self.ax2.hist(spikes_by_neuron, bins=bins, alpha=0.7, edgecolor='black')
        self.ax2.set_title('Spike Count Distribution Across Neurons')
        self.ax2.set_xlabel('Spike Count')
        self.ax2.set_ylabel('Number of Neurons')
    
    def show(self):
        """Display the plots"""
        plt.tight_layout()
        plt.show()


def plot_membrane_potential(
    membrane_traces: List[np.ndarray], 
    neuron_indices: Optional[List[int]] = None,
    time_steps: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot membrane potential traces for selected neurons
    
    Args:
        membrane_traces: List of membrane potential arrays for each time step
        neuron_indices: Indices of neurons to plot (default: first 5)
        time_steps: Time step array (default: 0 to len(membrane_traces))
        figsize: Figure size
    """
    if not membrane_traces:
        print("No membrane traces to plot")
        return
    
    # Convert to numpy array
    traces = np.stack(membrane_traces)  # Shape: (time_steps, batch_size, num_neurons)
    
    if neuron_indices is None:
        neuron_indices = list(range(min(5, traces.shape[-1])))
    
    if time_steps is None:
        time_steps = np.arange(traces.shape[0])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, neuron_idx in enumerate(neuron_indices):
        if neuron_idx < traces.shape[-1]:
            # Plot first batch item for simplicity
            membrane_trace = traces[:, 0, neuron_idx]
            ax.plot(time_steps, membrane_trace, label=f'Neuron {neuron_idx}', alpha=0.8)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Membrane Potential')
    ax.set_title('Membrane Potential Traces')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_connectivity_plot(
    weights: np.ndarray,
    source_layer: str = "Input",
    target_layer: str = "Output",
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Visualize connectivity weights between layers
    
    Args:
        weights: Weight matrix of shape (input_size, output_size)
        source_layer: Name of source layer
        target_layer: Name of target layer
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(weights.T, cmap='RdBu', aspect='auto')
    ax.set_xlabel(f'{source_layer} Neurons')
    ax.set_ylabel(f'{target_layer} Neurons')
    ax.set_title(f'Connectivity: {source_layer} → {target_layer}')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight Strength')
    
    plt.tight_layout()
    plt.show()